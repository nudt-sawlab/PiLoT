#!/usr/bin/env python3
"""Render DOM/DSM views from ContextCapture BlocksExchange XML cameras."""

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rasterio
import yaml
from pyproj import CRS, Transformer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_yawfix_refinement_update import (
    _checkerboard,
    _edge_overlay,
    _make_overlay,
    _safe_jsonable,
    _write_rgb,
)
from tools.run_dom_dsm_single_full import _depth_stats


DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_XML = "data_caiwangcun/CaiWangCun.xml"
DEFAULT_QUERY_DIR = "data_caiwangcun/query/images/exif_test"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_yawfix.txt"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/contextcapture_xml_initial_domdsm_exif_test"
XML_PROJECTION_ROTATION = "R_world_to_camera = R_xml"
RENDER_RAY_ROTATION = "R_camera_to_world = R_xml.T"
WRONG_RAY_ROTATION = "R_world_to_camera = R_xml"
RAY_ROTATION_CONVENTIONS = ("cam_to_world_correct", "world_to_cam_wrong_for_ray")
CONVENTIONS = RAY_ROTATION_CONVENTIONS
LEGACY_ROTATION_CONVENTIONS = {
    "R_xml_transpose": "cam_to_world_correct",
    "R_xml": "world_to_cam_wrong_for_ray",
}
LEGACY_ROTATION_BY_RAY_CONVENTION = {v: k for k, v in LEGACY_ROTATION_CONVENTIONS.items()}
AXIS_TRANSFORMS = {
    "ppp": [1.0, 1.0, 1.0],
    "ppm": [1.0, 1.0, -1.0],
    "pmp": [1.0, -1.0, 1.0],
    "pmm": [1.0, -1.0, -1.0],
    "mpp": [-1.0, 1.0, 1.0],
    "mpm": [-1.0, 1.0, -1.0],
    "mmp": [-1.0, -1.0, 1.0],
    "mmm": [-1.0, -1.0, -1.0],
}


@dataclass
class Intrinsics:
    photogroup_index: int
    width: int
    height: int
    focal: float
    cx: float
    cy: float
    k1: float
    k2: float
    k3: float
    p1: float
    p2: float
    aspect: float
    skew: float

    @property
    def fx(self) -> float:
        return self.focal

    @property
    def fy(self) -> float:
        return self.focal / self.aspect if abs(self.aspect) > 1e-12 else self.focal

    def as_dict(self) -> Dict[str, Any]:
        return {
            "photogroup_index": self.photogroup_index,
            "width": self.width,
            "height": self.height,
            "focal": self.focal,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "distortion": {
                "k1": self.k1,
                "k2": self.k2,
                "k3": self.k3,
                "p1": self.p1,
                "p2": self.p2,
            },
            "aspect": self.aspect,
            "skew": self.skew,
            "fy_rule": "fy = focal / aspect",
        }


@dataclass
class XmlPhoto:
    photo_id: str
    image_path: str
    center_xml: List[float]
    rotation: np.ndarray
    intrinsics: Intrinsics

    def as_dict(self) -> Dict[str, Any]:
        return {
            "photo_id": self.photo_id,
            "image_path": self.image_path,
            "center_xml": self.center_xml,
            "rotation": self.rotation.tolist(),
            "intrinsics": self.intrinsics.as_dict(),
        }


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalize_ray_convention(convention: str) -> str:
    if convention in RAY_ROTATION_CONVENTIONS:
        return convention
    if convention in LEGACY_ROTATION_CONVENTIONS:
        return LEGACY_ROTATION_CONVENTIONS[convention]
    raise ValueError(convention)


def _legacy_rotation_convention(ray_convention: str) -> str:
    return LEGACY_ROTATION_BY_RAY_CONVENTION[_normalize_ray_convention(ray_convention)]


def _ray_rotation_description(ray_convention: str) -> str:
    ray_convention = _normalize_ray_convention(ray_convention)
    if ray_convention == "cam_to_world_correct":
        return RENDER_RAY_ROTATION
    return WRONG_RAY_ROTATION


def _float_text(parent: ET.Element, path: str, default: Optional[float] = None) -> float:
    text = parent.findtext(path)
    if text is None:
        if default is None:
            raise KeyError(path)
        return float(default)
    return float(text)


def _parse_xml(xml_path: Path) -> Tuple[str, List[XmlPhoto]]:
    root = ET.parse(xml_path).getroot()
    xml_srs = root.findtext("SpatialReferenceSystems/SRS/Definition")
    if not xml_srs:
        raise ValueError(f"No SRS definition in {xml_path}")
    photos: List[XmlPhoto] = []
    for pg_idx, pg in enumerate(root.iter("Photogroup"), start=1):
        intr = Intrinsics(
            photogroup_index=pg_idx,
            width=int(_float_text(pg, "ImageDimensions/Width")),
            height=int(_float_text(pg, "ImageDimensions/Height")),
            focal=_float_text(pg, "FocalLengthPixels"),
            cx=_float_text(pg, "PrincipalPoint/x"),
            cy=_float_text(pg, "PrincipalPoint/y"),
            k1=_float_text(pg, "Distortion/K1", 0.0),
            k2=_float_text(pg, "Distortion/K2", 0.0),
            k3=_float_text(pg, "Distortion/K3", 0.0),
            p1=_float_text(pg, "Distortion/P1", 0.0),
            p2=_float_text(pg, "Distortion/P2", 0.0),
            aspect=_float_text(pg, "AspectRatio", 1.0),
            skew=_float_text(pg, "Skew", 0.0),
        )
        for ph in pg.findall("Photo"):
            center = ph.find("Pose/Center")
            rotation = ph.find("Pose/Rotation")
            if center is None or rotation is None:
                continue
            mat = np.array(
                [
                    [_float_text(rotation, "M_00"), _float_text(rotation, "M_01"), _float_text(rotation, "M_02")],
                    [_float_text(rotation, "M_10"), _float_text(rotation, "M_11"), _float_text(rotation, "M_12")],
                    [_float_text(rotation, "M_20"), _float_text(rotation, "M_21"), _float_text(rotation, "M_22")],
                ],
                dtype=np.float64,
            )
            photos.append(
                XmlPhoto(
                    photo_id=ph.findtext("Id") or "",
                    image_path=ph.findtext("ImagePath") or "",
                    center_xml=[_float_text(center, "x"), _float_text(center, "y"), _float_text(center, "z")],
                    rotation=mat,
                    intrinsics=intr,
                )
            )
    return xml_srs, photos


def _load_pose_file_projected(pose_file: Path, xml_srs: str) -> Dict[str, Dict[str, Any]]:
    transformer = Transformer.from_crs("EPSG:4326", xml_srs, always_xy=True)
    out: Dict[str, Dict[str, Any]] = {}
    for line in pose_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        name = parts[0]
        lon, lat, alt = map(float, parts[1:4])
        x, y = transformer.transform(lon, lat)
        out[name] = {
            "image": name,
            "lon_lat_alt": [lon, lat, alt],
            "center_xml_from_exif": [float(x), float(y), float(alt)],
        }
        out[name.lower()] = out[name]
    return out


def _match_photos(
    image_paths: Sequence[Path],
    pose_records: Dict[str, Dict[str, Any]],
    photos: Sequence[XmlPhoto],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    matches: Dict[str, Dict[str, Any]] = {}
    report: List[Dict[str, Any]] = []
    for image_path in image_paths:
        pose = pose_records.get(image_path.name) or pose_records.get(image_path.name.lower())
        if pose is None:
            report.append({"image": image_path.name, "status": "missing_exif_pose"})
            continue
        qx, qy, qz = pose["center_xml_from_exif"]
        ranked = sorted(
            (
                (
                    math.hypot(qx - photo.center_xml[0], qy - photo.center_xml[1]),
                    abs(qz - photo.center_xml[2]),
                    photo,
                )
                for photo in photos
            ),
            key=lambda item: item[0],
        )
        dxy, dz, photo = ranked[0]
        record = {
            "image": image_path.name,
            "status": "ok",
            "matched_photo_id": photo.photo_id,
            "matched_xml_image_path": photo.image_path,
            "match_distance_xy_m": float(dxy),
            "match_distance_z_m": float(dz),
            "match_warning": bool(dxy > 1.0),
            "exif_projected_center_xml_srs": pose["center_xml_from_exif"],
            "xml_photo": photo.as_dict(),
        }
        matches[image_path.name] = {"photo": photo, "match": record}
        report.append(record)
    return matches, report


class ContextCaptureDOMDSMRenderer:
    def __init__(self, render_config: Dict[str, Any], xml_srs: str, chunk_rows: int = 192) -> None:
        dom_dsm_cfg = render_config["dom_dsm"]
        self.dom = rasterio.open(dom_dsm_cfg["dom_path"])
        self.dsm = rasterio.open(dom_dsm_cfg["dsm_path"])
        self.nodata = dom_dsm_cfg.get("nodata")
        self.near_m = float(dom_dsm_cfg.get("near_m", 330.0))
        self.far_m = float(dom_dsm_cfg.get("far_m", 430.0))
        self.step_m = float(dom_dsm_cfg.get("ray_step_m", 10.0))
        self.ray_refine_iters = int(dom_dsm_cfg.get("ray_refine_iters", 10))
        self.dsm_sampling_mode = str(dom_dsm_cfg.get("dsm_sampling_mode", "bilinear"))
        self.dom_sampling_mode = str(dom_dsm_cfg.get("dom_sampling_mode", "bilinear"))
        self.chunk_rows = int(chunk_rows)
        self.dom_array = np.moveaxis(self.dom.read(), 0, -1)
        self.dsm_array = self.dsm.read(1)
        self.dom_transform = self.dom.transform
        self.dsm_transform = self.dsm.transform
        self.xml_to_raster = Transformer.from_crs(CRS.from_user_input(xml_srs), self.dsm.crs, always_xy=True)
        self.depth_values = np.arange(self.near_m, self.far_m + self.step_m, self.step_m, dtype=np.float32)

    @staticmethod
    def _xy_to_rowcol(transform: Any, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cols = np.floor((xs - transform.c) / transform.a).astype(np.int64)
        rows = np.floor((ys - transform.f) / transform.e).astype(np.int64)
        return rows, cols

    @staticmethod
    def _bilinear_2d(array: np.ndarray, rows_f: np.ndarray, cols_f: np.ndarray) -> np.ndarray:
        out = np.full(rows_f.shape, np.nan, dtype=np.float32)
        r0 = np.floor(rows_f).astype(np.int64)
        c0 = np.floor(cols_f).astype(np.int64)
        r1 = r0 + 1
        c1 = c0 + 1
        valid = (r0 >= 0) & (r1 < array.shape[0]) & (c0 >= 0) & (c1 < array.shape[1])
        if not np.any(valid):
            return out
        wr = rows_f[valid] - r0[valid]
        wc = cols_f[valid] - c0[valid]
        v00 = array[r0[valid], c0[valid]].astype(np.float32)
        v01 = array[r0[valid], c1[valid]].astype(np.float32)
        v10 = array[r1[valid], c0[valid]].astype(np.float32)
        v11 = array[r1[valid], c1[valid]].astype(np.float32)
        out[valid] = (
            (1.0 - wr) * (1.0 - wc) * v00
            + (1.0 - wr) * wc * v01
            + wr * (1.0 - wc) * v10
            + wr * wc * v11
        )
        return out

    @staticmethod
    def _bilinear_3d(array: np.ndarray, rows_f: np.ndarray, cols_f: np.ndarray) -> np.ndarray:
        out = np.zeros((len(rows_f), 3), dtype=np.uint8)
        r0 = np.floor(rows_f).astype(np.int64)
        c0 = np.floor(cols_f).astype(np.int64)
        r1 = r0 + 1
        c1 = c0 + 1
        valid = (r0 >= 0) & (r1 < array.shape[0]) & (c0 >= 0) & (c1 < array.shape[1])
        if not np.any(valid):
            return out
        wr = rows_f[valid] - r0[valid]
        wc = cols_f[valid] - c0[valid]
        v00 = array[r0[valid], c0[valid], :3].astype(np.float32)
        v01 = array[r0[valid], c1[valid], :3].astype(np.float32)
        v10 = array[r1[valid], c0[valid], :3].astype(np.float32)
        v11 = array[r1[valid], c1[valid], :3].astype(np.float32)
        vals = (
            ((1.0 - wr) * (1.0 - wc))[:, None] * v00
            + ((1.0 - wr) * wc)[:, None] * v01
            + (wr * (1.0 - wc))[:, None] * v10
            + (wr * wc)[:, None] * v11
        )
        out[valid] = np.clip(vals, 0, 255).astype(np.uint8)
        return out

    def _sample_dsm(self, xs: np.ndarray, ys: np.ndarray, sampling_mode: str = "nearest") -> np.ndarray:
        if sampling_mode == "bilinear":
            cols = (xs - self.dsm_transform.c) / self.dsm_transform.a
            rows = (ys - self.dsm_transform.f) / self.dsm_transform.e
            return self._bilinear_2d(self.dsm_array, rows, cols)
        rows, cols = self._xy_to_rowcol(self.dsm_transform, xs, ys)
        out = np.full(xs.shape, np.nan, dtype=np.float32)
        valid = (rows >= 0) & (rows < self.dsm_array.shape[0]) & (cols >= 0) & (cols < self.dsm_array.shape[1])
        if np.any(valid):
            out[valid] = self.dsm_array[rows[valid], cols[valid]].astype(np.float32)
        return out

    def sample_dsm_nearest(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        return self._sample_dsm(xs, ys, "nearest")

    def sample_dsm_bilinear(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        return self._sample_dsm(xs, ys, "bilinear")

    def _sample_dom(self, xs: np.ndarray, ys: np.ndarray, sampling_mode: str = "nearest") -> np.ndarray:
        if sampling_mode == "bilinear":
            cols = (xs - self.dom_transform.c) / self.dom_transform.a
            rows = (ys - self.dom_transform.f) / self.dom_transform.e
            return self._bilinear_3d(self.dom_array, rows, cols)
        rows, cols = self._xy_to_rowcol(self.dom_transform, xs, ys)
        out = np.zeros((len(xs), 3), dtype=np.uint8)
        valid = (rows >= 0) & (rows < self.dom_array.shape[0]) & (cols >= 0) & (cols < self.dom_array.shape[1])
        if np.any(valid):
            out[valid] = np.clip(self.dom_array[rows[valid], cols[valid], :3], 0, 255).astype(np.uint8)
        return out

    def sample_dom_nearest(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        return self._sample_dom(xs, ys, "nearest")

    def sample_dom_bilinear(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        return self._sample_dom(xs, ys, "bilinear")

    def _intersect_rays_with_dsm(
        self,
        origin: np.ndarray,
        dirs_world: np.ndarray,
        coarse_depth_values: np.ndarray,
        dsm_sampling_mode: str = "bilinear",
        refine_iters: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_pix = dirs_world.shape[0]
        hit_x = np.full(n_pix, np.nan, dtype=np.float64)
        hit_y = np.full(n_pix, np.nan, dtype=np.float64)
        depth = np.zeros(n_pix, dtype=np.float32)
        active = np.ones(n_pix, dtype=bool)
        prev_t = np.zeros(n_pix, dtype=np.float64)
        prev_f = np.full(n_pix, np.nan, dtype=np.float64)
        prev_valid = np.zeros(n_pix, dtype=bool)

        for z_depth in coarse_depth_values.astype(np.float64):
            active_ids = np.flatnonzero(active)
            if active_ids.size == 0:
                break
            dirs = dirs_world[active_ids]
            xs = origin[0] + dirs[:, 0] * z_depth
            ys = origin[1] + dirs[:, 1] * z_depth
            zs = origin[2] + dirs[:, 2] * z_depth
            terrain = self._sample_dsm(xs, ys, dsm_sampling_mode)
            valid = np.isfinite(terrain)
            if self.nodata is not None:
                valid &= terrain != float(self.nodata)
            f_curr = zs - terrain.astype(np.float64)
            hits = valid & (f_curr <= 0.0)
            if np.any(hits):
                hit_ids = active_ids[hits]
                t_hit = np.full(hit_ids.shape, float(z_depth), dtype=np.float64)
                refine_mask = prev_valid[hit_ids] & (prev_f[hit_ids] > 0.0) & (refine_iters > 0)
                if np.any(refine_mask):
                    refine_ids = hit_ids[refine_mask]
                    t_hit[refine_mask] = self._refine_ray_dsm_intersections(
                        origin,
                        dirs_world[refine_ids],
                        prev_t[refine_ids].astype(np.float64),
                        np.full(refine_ids.shape, float(z_depth), dtype=np.float64),
                        dsm_sampling_mode,
                        refine_iters,
                    )
                p_hit = origin[None, :] + dirs_world[hit_ids] * t_hit[:, None]
                hit_x[hit_ids] = p_hit[:, 0]
                hit_y[hit_ids] = p_hit[:, 1]
                depth[hit_ids] = t_hit.astype(np.float32)
                active[hit_ids] = False
            still_active = active_ids[valid & ~hits]
            prev_t[still_active] = z_depth
            prev_f[still_active] = f_curr[valid & ~hits]
            prev_valid[still_active] = True
        return hit_x, hit_y, depth

    def _refine_ray_dsm_intersections(
        self,
        origin: np.ndarray,
        ray_dirs: np.ndarray,
        t0: np.ndarray,
        t1: np.ndarray,
        dsm_sampling_mode: str,
        refine_iters: int,
    ) -> np.ndarray:
        t0 = t0.astype(np.float64).copy()
        t1 = t1.astype(np.float64).copy()
        p0 = origin[None, :] + ray_dirs * t0[:, None]
        terrain0 = self._sample_dsm(p0[:, 0], p0[:, 1], dsm_sampling_mode).astype(np.float64)
        f0 = p0[:, 2] - terrain0
        valid = np.isfinite(f0)
        for _ in range(refine_iters):
            tm = 0.5 * (t0 + t1)
            pm = origin[None, :] + ray_dirs * tm[:, None]
            terrain_m = self._sample_dsm(pm[:, 0], pm[:, 1], dsm_sampling_mode).astype(np.float64)
            fm = pm[:, 2] - terrain_m
            finite = valid & np.isfinite(fm)
            left_hit = finite & (f0 * fm <= 0.0)
            right_hit = finite & ~left_hit
            t1[left_hit] = tm[left_hit]
            t0[right_hit] = tm[right_hit]
            f0[right_hit] = fm[right_hit]
        return 0.5 * (t0 + t1)

    def _refine_ray_dsm_intersection(
        self,
        origin: np.ndarray,
        ray_dir: np.ndarray,
        t0: float,
        t1: float,
        dsm_sampling_mode: str,
        refine_iters: int,
    ) -> float:
        def f_at(t: float) -> float:
            p = origin + ray_dir * t
            terrain = self._sample_dsm(np.asarray([p[0]]), np.asarray([p[1]]), dsm_sampling_mode)[0]
            if not np.isfinite(terrain):
                return np.nan
            return float(p[2] - terrain)

        f0 = f_at(t0)
        if not np.isfinite(f0):
            return t1
        for _ in range(refine_iters):
            tm = 0.5 * (t0 + t1)
            fm = f_at(tm)
            if not np.isfinite(fm):
                break
            if f0 * fm <= 0.0:
                t1 = tm
            else:
                t0 = tm
                f0 = fm
        return 0.5 * (t0 + t1)

    @staticmethod
    def _undistort_normalized(xd: np.ndarray, yd: np.ndarray, intr: Intrinsics, enabled: bool) -> Tuple[np.ndarray, np.ndarray]:
        if not enabled:
            return xd.astype(np.float32), yd.astype(np.float32)
        x = xd.astype(np.float32).copy()
        y = yd.astype(np.float32).copy()
        k1, k2, k3, p1, p2 = map(np.float32, [intr.k1, intr.k2, intr.k3, intr.p1, intr.p2])
        for _ in range(5):
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            x = (xd - dx) / np.maximum(radial, 1e-8)
            y = (yd - dy) / np.maximum(radial, 1e-8)
        return x.astype(np.float32), y.astype(np.float32)

    def render(
        self,
        photo: XmlPhoto,
        convention: str,
        distortion_enabled: bool = True,
        axis_transform: Sequence[float] = (1.0, 1.0, 1.0),
        principal_point_mode: str = "xml",
        sampling_mode: str = "nearest",
        ray_step_m: Optional[float] = None,
        render_scale: float = 1.0,
        dsm_sampling_mode: Optional[str] = None,
        dom_sampling_mode: Optional[str] = None,
        ray_refine_iters: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        intr = photo.intrinsics
        width = int(round(intr.width * float(render_scale)))
        height = int(round(intr.height * float(render_scale)))
        fx = intr.fx * float(render_scale)
        fy = intr.fy * float(render_scale)
        cx = intr.cx * float(render_scale)
        cy_value = intr.cy if principal_point_mode == "xml" else (intr.height - intr.cy)
        cy = cy_value * float(render_scale)
        color_img = np.zeros((height, width, 3), dtype=np.uint8)
        depth_img = np.zeros((height, width), dtype=np.float32)
        cx_xml, cy_xml, alt = photo.center_xml
        cam_x, cam_y = self.xml_to_raster.transform(cx_xml, cy_xml)
        ray_convention = _normalize_ray_convention(convention)
        R_world_to_camera = photo.rotation
        R_camera_to_world = photo.rotation.T
        ray_rotation = R_camera_to_world if ray_convention == "cam_to_world_correct" else R_world_to_camera
        axis = np.asarray(axis_transform, dtype=np.float32).reshape(1, 3)
        dsm_mode = dsm_sampling_mode or sampling_mode or self.dsm_sampling_mode
        dom_mode = dom_sampling_mode or sampling_mode or self.dom_sampling_mode
        refine_iters = self.ray_refine_iters if ray_refine_iters is None else int(ray_refine_iters)
        depth_values = (
            np.arange(self.near_m, self.far_m + float(ray_step_m), float(ray_step_m), dtype=np.float32)
            if ray_step_m is not None
            else self.depth_values
        )
        t0 = time.perf_counter()
        for y0 in range(0, height, self.chunk_rows):
            y1 = min(y0 + self.chunk_rows, height)
            rows = np.arange(y0, y1, dtype=np.float32)[:, None]
            cols = np.arange(width, dtype=np.float32)[None, :]
            xd = (cols - np.float32(cx)) / np.float32(fx)
            yd = (rows - np.float32(cy)) / np.float32(fy)
            xd, yd = np.broadcast_arrays(xd, yd)
            x, y = self._undistort_normalized(xd, yd, intr, distortion_enabled)
            dirs_cam = np.stack([x, y, np.ones_like(x, dtype=np.float32)], axis=-1).reshape(-1, 3)
            dirs_cam = dirs_cam * axis
            dirs_world = (ray_rotation @ dirs_cam.T).T
            n_pix = dirs_world.shape[0]
            origin = np.asarray([float(cam_x), float(cam_y), float(alt)], dtype=np.float64)
            hit_x, hit_y, depth = self._intersect_rays_with_dsm(
                origin,
                dirs_world,
                depth_values,
                dsm_mode,
                refine_iters,
            )
            color = np.zeros((n_pix, 3), dtype=np.uint8)
            hit_mask = depth > 0
            if np.any(hit_mask):
                color[hit_mask] = self._sample_dom(hit_x[hit_mask], hit_y[hit_mask], dom_mode)
            color_img[y0:y1] = color.reshape(y1 - y0, width, 3)
            depth_img[y0:y1] = depth.reshape(y1 - y0, width)
        debug = {
            "camera_center_xml_srs": photo.center_xml,
            "camera_center_raster_crs": [float(cam_x), float(cam_y), float(alt)],
            "xml_projection_rotation": XML_PROJECTION_ROTATION,
            "render_ray_convention": ray_convention,
            "render_ray_rotation": _ray_rotation_description(ray_convention),
            "ray_rotation_matrix_source": "photo.rotation.T" if ray_convention == "cam_to_world_correct" else "photo.rotation",
            "legacy_rotation_convention": _legacy_rotation_convention(ray_convention),
            "rotation_convention": _legacy_rotation_convention(ray_convention),
            "axis_transform": [float(v) for v in axis.reshape(-1).tolist()],
            "principal_point_mode": principal_point_mode,
            "distortion_enabled": bool(distortion_enabled),
            "sampling_mode": sampling_mode,
            "dsm_sampling_mode": dsm_mode,
            "dom_sampling_mode": dom_mode,
            "ray_step_m": float(ray_step_m) if ray_step_m is not None else self.step_m,
            "ray_refine_iters": refine_iters,
            "render_scale": float(render_scale),
            "render_time_sec": float(time.perf_counter() - t0),
        }
        return color_img, depth_img, debug


def _read_query(path: Path, width: int, height: int) -> Tuple[np.ndarray, bool, Tuple[int, int]]:
    bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    source_shape = (int(bgr.shape[1]), int(bgr.shape[0]))
    resized = False
    if source_shape != (width, height):
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        resized = True
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), resized, source_shape


def _render_visuals(
    image_path: Path,
    photo: XmlPhoto,
    match: Dict[str, Any],
    renderer: ContextCaptureDOMDSMRenderer,
    output_dir: Path,
    convention: str,
    distortion_enabled: bool,
    checker_tile: int,
    axis_transform: Sequence[float] = (1.0, 1.0, 1.0),
    principal_point_mode: str = "xml",
    sampling_mode: str = "nearest",
    ray_step_m: Optional[float] = None,
    render_scale: float = 1.0,
    dsm_sampling_mode: Optional[str] = None,
    dom_sampling_mode: Optional[str] = None,
    ray_refine_iters: Optional[int] = None,
    subdir: str = "xml_initial",
) -> Dict[str, Any]:
    out_dir = output_dir / image_path.stem / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    intr = photo.intrinsics
    width = int(round(intr.width * float(render_scale)))
    height = int(round(intr.height * float(render_scale)))
    query_rgb, query_resized, source_shape = _read_query(image_path, width, height)
    render_rgb, depth, debug = renderer.render(
        photo,
        convention,
        distortion_enabled,
        axis_transform,
        principal_point_mode,
        sampling_mode,
        ray_step_m,
        render_scale,
        dsm_sampling_mode,
        dom_sampling_mode,
        ray_refine_iters,
    )
    overlay = _make_overlay(query_rgb, render_rgb)
    edge_overlay, edge_metrics = _edge_overlay(query_rgb, render_rgb)
    checkerboard = _checkerboard(query_rgb, render_rgb, checker_tile)
    _write_rgb(out_dir / "query_same_camera.png", query_rgb)
    _write_rgb(out_dir / "rendered_rgb.png", render_rgb)
    _write_rgb(out_dir / "overlay.png", overlay)
    _write_rgb(out_dir / "edge_overlay.png", edge_overlay)
    _write_rgb(out_dir / "checkerboard.png", checkerboard)
    metrics = {
        "image": image_path.name,
        "candidate": subdir,
        "query_image_path": str(image_path),
        "output_dir": str(out_dir),
        "render_width": width,
        "render_height": height,
        "source_query_width": source_shape[0],
        "source_query_height": source_shape[1],
        "query_resized_for_overlay": query_resized,
        "xml_photo_id": photo.photo_id,
        "xml_image_path": photo.image_path,
        "match_distance_xy_m": match.get("match_distance_xy_m"),
        "match_distance_z_m": match.get("match_distance_z_m"),
        "match_warning": match.get("match_warning"),
        "intrinsics": intr.as_dict(),
        **debug,
        **_depth_stats(depth),
        **edge_metrics,
    }
    _write_json(out_dir / "metrics.json", metrics)
    return metrics


def _select_convention(smoke_rows: Sequence[Dict[str, Any]]) -> str:
    valid = [r for r in smoke_rows if float(r.get("valid_depth_ratio", 0.0)) > 0.01 and np.isfinite(float(r.get("edge_chamfer", float("inf"))))]
    if not valid:
        return "cam_to_world_correct"
    return sorted(valid, key=lambda r: (float(r["edge_chamfer"]), -float(r["edge_overlap_ratio"]), -float(r["valid_depth_ratio"])))[0]["render_ray_convention"]


def _load_selected_ray_convention(convention_cfg: Dict[str, Any], fallback: str) -> str:
    raw = convention_cfg.get("render_ray_convention") or convention_cfg.get("rotation_convention") or fallback
    if raw == "auto":
        return raw
    return _normalize_ray_convention(raw)


def _load_convention_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = REPO_ROOT / path if not Path(path).is_absolute() else Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=DEFAULT_XML)
    parser.add_argument("--query-dir", default=DEFAULT_QUERY_DIR)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--images", nargs="+", default=None)
    parser.add_argument("--try-rotation-conventions", action="store_true")
    parser.add_argument(
        "--rotation-convention",
        choices=["auto", *RAY_ROTATION_CONVENTIONS, *LEGACY_ROTATION_CONVENTIONS.keys()],
        default="auto",
    )
    parser.add_argument("--convention-file", default=None)
    parser.add_argument("--disable-distortion", action="store_true")
    parser.add_argument("--axis-transform", choices=list(AXIS_TRANSFORMS), default="ppp")
    parser.add_argument("--principal-point-mode", choices=["xml", "flip_y"], default="xml")
    parser.add_argument("--sampling-mode", choices=["nearest", "bilinear"], default="nearest")
    parser.add_argument("--dsm-sampling-mode", choices=["nearest", "bilinear"], default=None)
    parser.add_argument("--dom-sampling-mode", choices=["nearest", "bilinear"], default=None)
    parser.add_argument("--ray-step-m", type=float, default=None)
    parser.add_argument("--ray-refine-iters", type=int, default=None)
    parser.add_argument("--render-scale", type=float, default=1.0)
    parser.add_argument("--checker-tile", type=int, default=128)
    parser.add_argument("--chunk-rows", type=int, default=192)
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    xml_srs, photos = _parse_xml((REPO_ROOT / args.xml).resolve())
    query_dir = (REPO_ROOT / args.query_dir).resolve()
    if args.images:
        image_paths = [query_dir / name for name in args.images]
    else:
        image_paths = sorted([p for p in query_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}])
    pose_records = _load_pose_file_projected((REPO_ROOT / args.pose_file).resolve(), xml_srs)
    matches, match_report = _match_photos(image_paths, pose_records, photos)
    _write_json(output_dir / "camera_match_report.json", {"xml_srs": xml_srs, "matches": match_report})
    renderer = ContextCaptureDOMDSMRenderer(config["render_config"], xml_srs, args.chunk_rows)
    convention_cfg = _load_convention_file(args.convention_file)
    distortion_enabled = bool(convention_cfg.get("distortion_enabled", not args.disable_distortion))
    selected_convention = _load_selected_ray_convention(convention_cfg, args.rotation_convention)
    axis_key = convention_cfg.get("axis_transform_key", args.axis_transform)
    axis_transform = convention_cfg.get("axis_transform", AXIS_TRANSFORMS[axis_key])
    principal_point_mode = convention_cfg.get("principal_point_mode", args.principal_point_mode)
    sampling_mode = convention_cfg.get("sampling_mode", args.sampling_mode)
    dsm_sampling_mode = convention_cfg.get("dsm_sampling_mode", args.dsm_sampling_mode or renderer.dsm_sampling_mode)
    dom_sampling_mode = convention_cfg.get("dom_sampling_mode", args.dom_sampling_mode or renderer.dom_sampling_mode)
    ray_step_m = convention_cfg.get("ray_step_m", args.ray_step_m)
    ray_refine_iters = convention_cfg.get("ray_refine_iters", args.ray_refine_iters)
    render_scale = float(convention_cfg.get("render_scale", args.render_scale))
    smoke_rows: List[Dict[str, Any]] = []
    if args.try_rotation_conventions or selected_convention == "auto":
        first = next((p for p in image_paths if p.name in matches), None)
        if first is not None:
            for conv in RAY_ROTATION_CONVENTIONS:
                item = matches[first.name]
                smoke_rows.append(
                    _render_visuals(
                        first,
                        item["photo"],
                        item["match"],
                        renderer,
                        output_dir,
                        conv,
                        distortion_enabled,
                        args.checker_tile,
                        axis_transform,
                        principal_point_mode,
                        sampling_mode,
                        ray_step_m,
                        render_scale,
                        dsm_sampling_mode,
                        dom_sampling_mode,
                        ray_refine_iters,
                        subdir=f"xml_initial_{conv}",
                    )
                )
            selected_convention = _select_convention(smoke_rows)
    rows: List[Dict[str, Any]] = []
    if args.try_rotation_conventions:
        rows = smoke_rows
    else:
        assert selected_convention in RAY_ROTATION_CONVENTIONS
        for image_path in image_paths:
            item = matches.get(image_path.name)
            if item is None:
                continue
            rows.append(
                _render_visuals(
                    image_path,
                    item["photo"],
                    item["match"],
                    renderer,
                    output_dir,
                    selected_convention,
                    distortion_enabled,
                    args.checker_tile,
                    axis_transform,
                    principal_point_mode,
                    sampling_mode,
                    ray_step_m,
                    render_scale,
                    dsm_sampling_mode,
                    dom_sampling_mode,
                    ray_refine_iters,
                    subdir=convention_cfg.get("subdir", "xml_initial"),
                )
            )
    csv_path = output_dir / "contextcapture_xml_render_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "image",
            "xml_photo_id",
            "rotation_convention",
            "legacy_rotation_convention",
            "render_ray_convention",
            "render_ray_rotation",
            "xml_projection_rotation",
            "distortion_enabled",
            "dsm_sampling_mode",
            "dom_sampling_mode",
            "ray_refine_iters",
            "render_width",
            "render_height",
            "source_query_width",
            "source_query_height",
            "query_resized_for_overlay",
            "match_distance_xy_m",
            "match_distance_z_m",
            "edge_chamfer",
            "edge_overlap_ratio",
            "query_edge_count",
            "render_edge_count",
            "edge_overlap_count",
            "valid_depth_ratio",
            "depth_min",
            "depth_max",
            "render_time_sec",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    exif_summary_path = REPO_ROOT / "docs/experiments/dom_dsm_prepare/initial_domdsm_exif_intrinsics_exif_test/summary_metrics.json"
    exif_comparison = None
    if exif_summary_path.exists():
        exif = json.loads(exif_summary_path.read_text(encoding="utf-8"))
        exif_comparison = {
            "source": str(exif_summary_path.relative_to(REPO_ROOT)),
            "mean_edge_chamfer": exif.get("metrics_mean", {}).get("edge_chamfer"),
            "mean_edge_overlap_ratio": exif.get("metrics_mean", {}).get("edge_overlap_ratio"),
        }
    summary = {
        "experiment": "ContextCapture XML initial DOM/DSM render",
        "xml": args.xml,
        "xml_srs": xml_srs,
        "query_dir": args.query_dir,
        "config": args.config,
        "output_dir": args.output_dir,
        "try_rotation_conventions": args.try_rotation_conventions,
        "selected_render_ray_convention": selected_convention,
        "selected_rotation_convention": _legacy_rotation_convention(selected_convention),
        "xml_projection_rotation": XML_PROJECTION_ROTATION,
        "render_ray_rotation": _ray_rotation_description(selected_convention),
        "legacy_rotation_convention": _legacy_rotation_convention(selected_convention),
        "axis_transform_key": axis_key,
        "axis_transform": axis_transform,
        "principal_point_mode": principal_point_mode,
        "sampling_mode": sampling_mode,
        "dsm_sampling_mode": dsm_sampling_mode,
        "dom_sampling_mode": dom_sampling_mode,
        "ray_step_m": ray_step_m if ray_step_m is not None else renderer.step_m,
        "ray_refine_iters": ray_refine_iters if ray_refine_iters is not None else renderer.ray_refine_iters,
        "render_scale": render_scale,
        "distortion_enabled": distortion_enabled,
        "num_images_requested": len(image_paths),
        "num_images_rendered": len(rows),
        "camera_match_report": match_report,
        "metrics_mean": {
            "edge_chamfer": float(np.mean([r["edge_chamfer"] for r in rows])) if rows else None,
            "edge_overlap_ratio": float(np.mean([r["edge_overlap_ratio"] for r in rows])) if rows else None,
            "valid_depth_ratio": float(np.mean([r["valid_depth_ratio"] for r in rows])) if rows else None,
        },
        "exif_yawfix_initial_comparison": exif_comparison,
        "images": rows,
    }
    _write_json(output_dir / "summary_metrics.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
