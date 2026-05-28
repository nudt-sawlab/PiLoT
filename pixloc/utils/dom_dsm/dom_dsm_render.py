"""Prototype DOM + DSM renderer for PiLoT.

This backend renders a perspective RGB reference and a camera-view depth map
from georeferenced DOM/DSM rasters. It intentionally favors correctness and
debuggability over speed for the first integration pass.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


class DOMDSMRenderer:
    """Render RGB and camera-depth from DOM/DSM GeoTIFF files."""

    def __init__(self, config: Dict) -> None:
        dom_dsm_cfg = config["dom_dsm"]
        self.dom_path = Path(dom_dsm_cfg["dom_path"])
        self.dsm_path = Path(dom_dsm_cfg["dsm_path"])
        self.nodata = dom_dsm_cfg.get("nodata")
        self.near_m = float(dom_dsm_cfg.get("near_m", 1.0))
        self.far_m = float(dom_dsm_cfg.get("far_m", 1500.0))
        self.step_m = float(dom_dsm_cfg.get("ray_step_m", 2.0))
        self.ray_refine_iters = int(dom_dsm_cfg.get("ray_refine_iters", 10))
        self.dsm_sampling_mode = str(dom_dsm_cfg.get("dsm_sampling_mode", "bilinear"))
        self.dom_sampling_mode = str(dom_dsm_cfg.get("dom_sampling_mode", "bilinear"))
        self.debug_dir = dom_dsm_cfg.get("debug_dir")
        self.debug_every = int(dom_dsm_cfg.get("debug_every", 1))
        self.render_backend = str(dom_dsm_cfg.get("render_backend", "prototype"))
        self.gpu_renderer = str(dom_dsm_cfg.get("gpu_renderer", "nvdiffrast"))
        self.fallback_backend = str(dom_dsm_cfg.get("fallback_backend", "prototype"))
        self.min_valid_depth_ratio = float(dom_dsm_cfg.get("min_valid_depth_ratio", 0.05))
        self.mesh_gsd = float(dom_dsm_cfg.get("mesh_gsd", 0.5))
        self.tile_size_m = float(dom_dsm_cfg.get("tile_size_m", 512.0))
        self.tile_margin_m = float(dom_dsm_cfg.get("tile_margin_m", 80.0))
        self.texture_v_flip = bool(dom_dsm_cfg.get("texture_v_flip", True))
        self.output_y_flip = bool(dom_dsm_cfg.get("output_y_flip", False))
        self.debug_texture_mode = str(dom_dsm_cfg.get("debug_texture_mode", "none"))
        self._render_count = 0
        self._gpu_renderer = None
        self._gpu_init_error: Optional[str] = None
        self.last_render_metadata: Dict[str, Any] = {}

        if not self.dom_path.is_file():
            raise FileNotFoundError(f"DOM GeoTIFF not found: {self.dom_path}")
        if not self.dsm_path.is_file():
            raise FileNotFoundError(f"DSM GeoTIFF not found: {self.dsm_path}")

        self.dom = rasterio.open(self.dom_path)
        self.dsm = rasterio.open(self.dsm_path)
        self._validate_rasters()
        self.dom_array = np.moveaxis(self.dom.read(), 0, -1)
        self.dsm_array = self.dsm.read(1)
        self.dom_transform = self.dom.transform
        self.dsm_transform = self.dsm.transform

        render_camera = np.asarray(config["render_camera"], dtype=np.float64)
        self.width = int(render_camera[0])
        self.height = int(render_camera[1])
        self.cx = float(render_camera[2])
        self.cy = float(render_camera[3])
        self.fx = float(render_camera[4])
        self.fy = float(render_camera[5])

        self.to_raster = Transformer.from_crs(
            "EPSG:4326", self.dsm.crs, always_xy=True
        )
        self.from_raster = Transformer.from_crs(
            self.dsm.crs, "EPSG:4326", always_xy=True
        )

        grid_x, grid_y = np.meshgrid(
            np.arange(self.width, dtype=np.float32),
            np.arange(self.height, dtype=np.float32),
        )
        self.camera_dirs = np.stack(
            [
                (grid_x - self.cx) / self.fx,
                (grid_y - self.cy) / self.fy,
                np.ones_like(grid_x),
            ],
            axis=-1,
        )
        self.depth_values = np.arange(
            self.near_m, self.far_m + self.step_m, self.step_m, dtype=np.float32
        )

        logger.info(
            "DOMDSMRenderer ready: %dx%d backend=%s, DOM=%s, DSM=%s",
            self.width,
            self.height,
            self.render_backend,
            self.dom_path,
            self.dsm_path,
        )

    @staticmethod
    def _xy_to_rowcol(transform, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        channels = min(array.shape[2], 3)
        v00 = array[r0[valid], c0[valid], :channels].astype(np.float32)
        v01 = array[r0[valid], c1[valid], :channels].astype(np.float32)
        v10 = array[r1[valid], c0[valid], :channels].astype(np.float32)
        v11 = array[r1[valid], c1[valid], :channels].astype(np.float32)
        vals = (
            ((1.0 - wr) * (1.0 - wc))[:, None] * v00
            + ((1.0 - wr) * wc)[:, None] * v01
            + (wr * (1.0 - wc))[:, None] * v10
            + (wr * wc)[:, None] * v11
        )
        if channels == 1:
            vals = np.repeat(vals, 3, axis=1)
        out[valid] = np.clip(vals[:, :3], 0, 255).astype(np.uint8)
        return out

    def _validate_rasters(self) -> None:
        if self.dom.crs is None:
            raise ValueError(f"DOM has no CRS: {self.dom_path}")
        if self.dsm.crs is None:
            raise ValueError(f"DSM has no CRS: {self.dsm_path}")
        if self.dom.crs != self.dsm.crs:
            raise ValueError(f"DOM/DSM CRS mismatch: {self.dom.crs} vs {self.dsm.crs}")
        if self.dom.transform.a == 0 or self.dsm.transform.a == 0:
            raise ValueError("Invalid raster transform")
        logger.info(
            "DOM/DSM CRS=%s DOM bounds=%s DSM bounds=%s DOM res=%s DSM res=%s",
            self.dsm.crs,
            self.dom.bounds,
            self.dsm.bounds,
            self.dom.res,
            self.dsm.res,
        )

    def render(
        self,
        trans: List[float],
        euler: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render color and camera-view depth.

        Args:
            trans: [lon, lat, alt] in WGS-84.
            euler: [pitch, roll, yaw] in degrees.

        Returns:
            color: HxWx3 uint8 RGB perspective render.
            depth: HxW float32 camera z-depth in meters.
        """
        R_camera_to_world = R.from_euler("xyz", euler, degrees=True).as_matrix()
        return self.render_matrix(trans, R_camera_to_world)

    def render_matrix(
        self,
        trans: List[float],
        R_camera_to_world: np.ndarray,
        K: Optional[np.ndarray] = None,
        distortion: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render color and camera-view depth from an explicit camera-to-world matrix.

        Args:
            trans: [lon, lat, alt] in WGS-84.
            R_camera_to_world: 3x3 rotation from camera ray coordinates to raster/world axes.
            K: Optional 3x3 intrinsics override. Distortion is accepted for API
                compatibility but not applied by this renderer yet.
            distortion: Reserved for future distortion handling.
        """
        start = time.perf_counter()
        metadata: Dict[str, Any] = {
            "backend_requested": self.render_backend,
            "backend_used": self.render_backend,
            "fallback_reason": None,
            "depth_convention": "camera_z_m",
        }

        if self.render_backend == "prototype":
            color, depth = self._render_prototype_matrix(trans, R_camera_to_world, K, distortion)
            metadata.update(self._metadata_for_depth(depth, start, "prototype"))
            self.last_render_metadata = metadata
            self._write_debug(color, depth, metadata)
            return color, depth

        if self.render_backend == "gpu_mesh":
            try:
                if K is not None:
                    raise RuntimeError("gpu_mesh render_matrix does not support K overrides yet")
                if distortion is not None:
                    logger.debug("gpu_mesh ignores distortion for DOM/DSM rendering")
                color, depth, gpu_metadata = self._render_gpu_matrix(trans, R_camera_to_world)
                metadata.update(gpu_metadata)
                metadata.update(self._metadata_for_depth(depth, start, "gpu_mesh"))
                self.last_render_metadata = metadata
                self._write_debug(color, depth, metadata)
                return color, depth
            except Exception as exc:  # fallback must be explicit and recorded
                fallback_reason = f"{type(exc).__name__}: {exc}"
                logger.warning("DOM/DSM gpu_mesh render failed; falling back to prototype: %s", fallback_reason)
                color, depth = self._render_prototype_matrix(trans, R_camera_to_world, K, distortion)
                metadata.update(self._metadata_for_depth(depth, start, "prototype"))
                metadata["backend_used"] = "prototype"
                metadata["fallback_reason"] = fallback_reason
                self.last_render_metadata = metadata
                self._write_debug(color, depth, metadata)
                return color, depth

        raise ValueError(f"Unsupported DOM/DSM render_backend: {self.render_backend}")

    def _render_gpu_matrix(
        self,
        trans: List[float],
        R_camera_to_world: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        renderer = self._get_gpu_renderer()
        result = renderer.render(trans, [0.0, 0.0, 0.0], R_camera_to_world=R_camera_to_world)
        return result.color, result.depth, result.metadata

    def _get_gpu_renderer(self):
        if self._gpu_renderer is not None:
            return self._gpu_renderer
        if self._gpu_init_error is not None:
            raise RuntimeError(self._gpu_init_error)
        try:
            if self.gpu_renderer != "nvdiffrast":
                raise ValueError(f"Unsupported gpu_renderer: {self.gpu_renderer}")
            from pixloc.utils.dom_dsm.terrain_mesh_renderer import TerrainMeshRenderer

            self._gpu_renderer = TerrainMeshRenderer(
                {
                    "dom_dsm": {
                        "nodata": self.nodata,
                        "mesh_gsd": self.mesh_gsd,
                        "tile_size_m": self.tile_size_m,
                        "tile_margin_m": self.tile_margin_m,
                        "near_m": self.near_m,
                        "far_m": self.far_m,
                        "min_valid_depth_ratio": self.min_valid_depth_ratio,
                        "texture_v_flip": self.texture_v_flip,
                        "output_y_flip": self.output_y_flip,
                        "debug_texture_mode": self.debug_texture_mode,
                    },
                    "render_camera": [
                        self.width,
                        self.height,
                        self.cx,
                        self.cy,
                        self.fx,
                        self.fy,
                    ],
                },
                self.dom,
                self.dsm,
                self.to_raster,
            )
            return self._gpu_renderer
        except Exception as exc:
            self._gpu_init_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(self._gpu_init_error)

    def _metadata_for_depth(
        self,
        depth: np.ndarray,
        start_time: float,
        backend_used: str,
    ) -> Dict[str, Any]:
        valid = np.isfinite(depth) & (depth > 0)
        valid_ratio = float(np.count_nonzero(valid) / depth.size)
        out: Dict[str, Any] = {
            "backend_used": backend_used,
            "valid_depth_ratio": valid_ratio,
            "render_time_ms": (time.perf_counter() - start_time) * 1000.0,
        }
        if valid.any():
            out.update(
                {
                    "depth_min_m": float(depth[valid].min()),
                    "depth_max_m": float(depth[valid].max()),
                    "depth_median_m": float(np.median(depth[valid])),
                }
            )
        return out

    def _render_prototype_matrix(
        self,
        trans: List[float],
        R_camera_to_world: np.ndarray,
        K: Optional[np.ndarray] = None,
        distortion: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        del distortion
        lon, lat, alt = map(float, trans)
        cam_x, cam_y = self.to_raster.transform(lon, lat)

        if K is None:
            camera_dirs = self.camera_dirs.reshape(-1, 3)
        else:
            K = np.asarray(K, dtype=np.float64)
            fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
            grid_x, grid_y = np.meshgrid(
                np.arange(self.width, dtype=np.float32),
                np.arange(self.height, dtype=np.float32),
            )
            camera_dirs = np.stack(
                [
                    (grid_x - cx) / fx,
                    (grid_y - cy) / fy,
                    np.ones_like(grid_x),
                ],
                axis=-1,
            ).reshape(-1, 3)

        R_camera_to_world = np.asarray(R_camera_to_world, dtype=np.float64)
        dirs_world = (R_camera_to_world @ camera_dirs.T).T
        n_pix = dirs_world.shape[0]
        origin = np.asarray([float(cam_x), float(cam_y), float(alt)], dtype=np.float64)
        hit_x, hit_y, depth = self._intersect_rays_with_dsm(
            origin,
            dirs_world,
            self.depth_values,
            self.dsm_sampling_mode,
            self.ray_refine_iters,
        )

        color = np.zeros((n_pix, 3), dtype=np.uint8)
        hit_mask = depth > 0
        if np.any(hit_mask):
            color[hit_mask] = self._sample_dom(hit_x[hit_mask], hit_y[hit_mask], self.dom_sampling_mode)

        color_img = color.reshape(self.height, self.width, 3)
        depth_img = depth.reshape(self.height, self.width)
        return color_img, depth_img

    def _intersect_rays_with_dsm(
        self,
        origin: np.ndarray,
        dirs_world: np.ndarray,
        coarse_depth_values: np.ndarray,
        dsm_sampling: str = "bilinear",
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
            terrain = self._sample_dsm(xs, ys, dsm_sampling)
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
                        dsm_sampling,
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
        dsm_sampling: str,
        refine_iters: int,
    ) -> np.ndarray:
        t0 = t0.astype(np.float64).copy()
        t1 = t1.astype(np.float64).copy()
        p0 = origin[None, :] + ray_dirs * t0[:, None]
        terrain0 = self._sample_dsm(p0[:, 0], p0[:, 1], dsm_sampling).astype(np.float64)
        f0 = p0[:, 2] - terrain0
        valid = np.isfinite(f0)
        for _ in range(refine_iters):
            tm = 0.5 * (t0 + t1)
            pm = origin[None, :] + ray_dirs * tm[:, None]
            terrain_m = self._sample_dsm(pm[:, 0], pm[:, 1], dsm_sampling).astype(np.float64)
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
        dsm_sampling: str,
        refine_iters: int,
    ) -> float:
        def f_at(t: float) -> float:
            p = origin + ray_dir * t
            terrain = self._sample_dsm(np.asarray([p[0]]), np.asarray([p[1]]), dsm_sampling)[0]
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

    def sample_dsm_nearest(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        rows, cols = self._xy_to_rowcol(self.dsm_transform, xs, ys)
        out = np.full(xs.shape, np.nan, dtype=np.float32)
        valid = (rows >= 0) & (rows < self.dsm_array.shape[0]) & (cols >= 0) & (cols < self.dsm_array.shape[1])
        if np.any(valid):
            out[valid] = self.dsm_array[rows[valid], cols[valid]].astype(np.float32)
        return out

    def sample_dsm_bilinear(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        cols = (xs - self.dsm_transform.c) / self.dsm_transform.a
        rows = (ys - self.dsm_transform.f) / self.dsm_transform.e
        return self._bilinear_2d(self.dsm_array, rows, cols)

    def sample_dom_nearest(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        rows, cols = self._xy_to_rowcol(self.dom_transform, xs, ys)
        out = np.zeros((len(xs), 3), dtype=np.uint8)
        valid = (rows >= 0) & (rows < self.dom_array.shape[0]) & (cols >= 0) & (cols < self.dom_array.shape[1])
        if np.any(valid):
            rgb = self.dom_array[rows[valid], cols[valid], :3]
            if rgb.shape[1] == 1:
                rgb = np.repeat(rgb, 3, axis=1)
            out[valid] = np.clip(rgb[:, :3], 0, 255).astype(np.uint8)
        return out

    def sample_dom_bilinear(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        cols = (xs - self.dom_transform.c) / self.dom_transform.a
        rows = (ys - self.dom_transform.f) / self.dom_transform.e
        return self._bilinear_3d(self.dom_array, rows, cols)

    def _sample_dsm(self, xs: np.ndarray, ys: np.ndarray, sampling_mode: Optional[str] = None) -> np.ndarray:
        mode = sampling_mode or self.dsm_sampling_mode
        if mode == "bilinear":
            return self.sample_dsm_bilinear(xs, ys)
        if mode == "nearest":
            return self.sample_dsm_nearest(xs, ys)
        raise ValueError(f"Unsupported DSM sampling mode: {mode}")

    def _sample_dom(self, xs: np.ndarray, ys: np.ndarray, sampling_mode: Optional[str] = None) -> np.ndarray:
        mode = sampling_mode or self.dom_sampling_mode
        if mode == "bilinear":
            return self.sample_dom_bilinear(xs, ys)
        if mode == "nearest":
            return self.sample_dom_nearest(xs, ys)
        raise ValueError(f"Unsupported DOM sampling mode: {mode}")

    def _write_debug(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._render_count += 1
        if not self.debug_dir or self.debug_every <= 0:
            return
        if (self._render_count - 1) % self.debug_every != 0:
            return

        debug_dir = Path(self.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            os.fspath(debug_dir / f"rendered_rgb_{self._render_count:04d}.png"),
            cv2.cvtColor(color, cv2.COLOR_RGB2BGR),
        )

        depth_valid = depth[depth > 0]
        if depth_valid.size:
            d_min, d_max = float(depth_valid.min()), float(depth_valid.max())
            scaled = (depth - d_min) / max(d_max - d_min, 1e-6)
            scaled[depth <= 0] = 0
            depth_vis = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
        else:
            depth_vis = np.zeros_like(depth, dtype=np.uint8)
        cv2.imwrite(
            os.fspath(debug_dir / f"rendered_depth_{self._render_count:04d}.png"),
            depth_vis,
        )
        if metadata is not None:
            with open(debug_dir / f"render_metadata_{self._render_count:04d}.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
