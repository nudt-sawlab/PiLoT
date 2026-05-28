"""GPU terrain-mesh renderer for DOM/DSM reference views.

The public output contract matches :class:`DOMDSMRenderer`: RGB is returned as
``HxWx3`` uint8 in RGB order and depth is returned as ``HxW`` float32 camera-z
depth in meters. The rasterizer's own depth output is intentionally not used as
PiLoT depth.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


@dataclass
class TerrainMeshRenderResult:
    color: np.ndarray
    depth: np.ndarray
    metadata: Dict[str, Any]


def build_nvdiffrast_camera_projection(
    vertices_local_m: "torch.Tensor",
    camera_local_m: "torch.Tensor",
    R_camera_to_world: "torch.Tensor",
    K: "torch.Tensor",
    width: int,
    height: int,
    near_m: float,
    far_m: float,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Convert PiLoT/OpenCV camera geometry into nvdiffrast clip vertices.

    This is the only place where the OpenCV/PiLoT image convention is converted
    for nvdiffrast. PiLoT uses x-right, y-down, z-forward camera coordinates.
    nvdiffrast receives clip-space coordinates with y-up NDC, so the image-space
    y conversion is centralized here instead of being scattered through the
    renderer.

    Returns:
        ``(pos_clip, camera_z_m)``. ``camera_z_m`` is the per-vertex camera-z
        attribute that must be interpolated to produce PiLoT depth.
    """
    import torch

    cam_xyz = (vertices_local_m - camera_local_m[None, :]) @ R_camera_to_world
    camera_z_m = cam_xyz[:, 2]

    z_safe = torch.clamp(camera_z_m, min=max(float(near_m), 1.0e-6))
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * (cam_xyz[:, 0] / z_safe) + cx
    v = fy * (cam_xyz[:, 1] / z_safe) + cy

    x_ndc = 2.0 * u / float(width) - 1.0
    y_ndc = 1.0 - 2.0 * v / float(height)
    z_ndc = 2.0 * (z_safe - float(near_m)) / max(float(far_m - near_m), 1.0e-6) - 1.0

    pos_clip = torch.stack(
        [x_ndc * z_safe, y_ndc * z_safe, z_ndc * z_safe, z_safe],
        dim=1,
    )
    return pos_clip.contiguous(), camera_z_m.contiguous()


class TerrainMeshRenderer:
    """Render DSM terrain mesh textured by DOM with a GPU offscreen rasterizer."""

    def __init__(
        self,
        config: Dict[str, Any],
        dom: "rasterio.io.DatasetReader",
        dsm: "rasterio.io.DatasetReader",
        to_raster: Any,
    ) -> None:
        dom_dsm_cfg = config["dom_dsm"]
        self.dom = dom
        self.dsm = dsm
        self.to_raster = to_raster
        self.nodata = dom_dsm_cfg.get("nodata")
        self.mesh_gsd = float(dom_dsm_cfg.get("mesh_gsd", 0.5))
        self.tile_size_m = float(dom_dsm_cfg.get("tile_size_m", 512.0))
        self.tile_margin_m = float(dom_dsm_cfg.get("tile_margin_m", 80.0))
        self.near_m = float(dom_dsm_cfg.get("near_m", 1.0))
        self.far_m = float(dom_dsm_cfg.get("far_m", 1500.0))
        self.min_valid_depth_ratio = float(dom_dsm_cfg.get("min_valid_depth_ratio", 0.05))
        self.texture_v_flip = bool(dom_dsm_cfg.get("texture_v_flip", True))
        self.output_y_flip = bool(dom_dsm_cfg.get("output_y_flip", False))
        self.debug_texture_mode = str(dom_dsm_cfg.get("debug_texture_mode", "none"))

        render_camera = np.asarray(config["render_camera"], dtype=np.float64)
        self.width = int(render_camera[0])
        self.height = int(render_camera[1])
        self.K_np = np.asarray(
            [
                [float(render_camera[4]), 0.0, float(render_camera[2])],
                [0.0, float(render_camera[5]), float(render_camera[3])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        if self.mesh_gsd <= 0:
            raise ValueError("mesh_gsd must be positive")
        if self.tile_size_m <= 0:
            raise ValueError("tile_size_m must be positive")

        import torch
        import nvdiffrast.torch as dr

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for gpu_mesh renderer")

        self.torch = torch
        self.dr = dr
        self.device = torch.device("cuda")
        self.ctx = dr.RasterizeCudaContext()
        self.K = torch.as_tensor(self.K_np, dtype=torch.float32, device=self.device)
        self._texture = None
        logger.info(
            "TerrainMeshRenderer ready: %dx%d mesh_gsd=%.3fm tile=%.1fm margin=%.1fm",
            self.width,
            self.height,
            self.mesh_gsd,
            self.tile_size_m,
            self.tile_margin_m,
        )

    def render(
        self,
        trans: List[float],
        euler: List[float],
        R_camera_to_world: Optional[np.ndarray] = None,
    ) -> TerrainMeshRenderResult:
        start = time.perf_counter()
        lon, lat, alt = map(float, trans)
        cam_x, cam_y = self.to_raster.transform(lon, lat)
        camera_world = np.asarray([cam_x, cam_y, alt], dtype=np.float32)
        if R_camera_to_world is None:
            R_camera_to_world = R.from_euler("xyz", euler, degrees=True).as_matrix()

        mesh = self._build_mesh(camera_world)
        color, depth, stats = self._render_mesh(mesh, camera_world, R_camera_to_world)
        render_time_ms = (time.perf_counter() - start) * 1000.0

        valid = np.isfinite(depth) & (depth > 0)
        valid_ratio = float(np.count_nonzero(valid) / depth.size)
        metadata: Dict[str, Any] = {
            "backend_requested": "gpu_mesh",
            "backend_used": "gpu_mesh",
            "fallback_reason": None,
            "valid_depth_ratio": valid_ratio,
            "render_time_ms": render_time_ms,
            "depth_convention": "camera_z_m",
            "texture_v_flip": self.texture_v_flip,
            "output_y_flip": self.output_y_flip,
            "debug_texture_mode": self.debug_texture_mode,
            **stats,
        }
        if valid_ratio < self.min_valid_depth_ratio:
            raise RuntimeError(
                "gpu_mesh valid depth ratio "
                f"{valid_ratio:.6f} below threshold {self.min_valid_depth_ratio:.6f}"
            )
        if not np.all(np.isfinite(depth[valid])):
            raise RuntimeError("gpu_mesh produced non-finite valid depth values")
        return TerrainMeshRenderResult(color=color, depth=depth, metadata=metadata)

    def _build_mesh(self, camera_world: np.ndarray) -> Dict[str, np.ndarray]:
        half = 0.5 * self.tile_size_m + self.tile_margin_m
        minx, maxx = float(camera_world[0] - half), float(camera_world[0] + half)
        miny, maxy = float(camera_world[1] - half), float(camera_world[1] + half)

        dsm_window = from_bounds(minx, miny, maxx, maxy, transform=self.dsm.transform)
        dsm_window = dsm_window.round_offsets().round_lengths()
        full = Window(0, 0, self.dsm.width, self.dsm.height)
        dsm_window = dsm_window.intersection(full)
        if dsm_window.width < 2 or dsm_window.height < 2:
            raise RuntimeError("DSM tile window is too small for mesh rendering")

        tile_transform = self.dsm.window_transform(dsm_window)
        dsm_tile = self.dsm.read(
            1,
            window=dsm_window,
            out_dtype="float32",
            boundless=False,
        )

        row_step_px = max(1, int(round(self.mesh_gsd / max(abs(tile_transform.e), 1.0e-9))))
        col_step_px = max(1, int(round(self.mesh_gsd / max(abs(tile_transform.a), 1.0e-9))))
        rows = np.arange(0, dsm_tile.shape[0], row_step_px, dtype=np.int64)
        cols = np.arange(0, dsm_tile.shape[1], col_step_px, dtype=np.int64)
        if rows[-1] != dsm_tile.shape[0] - 1:
            rows = np.append(rows, dsm_tile.shape[0] - 1)
        if cols[-1] != dsm_tile.shape[1] - 1:
            cols = np.append(cols, dsm_tile.shape[1] - 1)
        grid_cols, grid_rows = np.meshgrid(cols, rows)

        xs, ys = rasterio.transform.xy(tile_transform, grid_rows, grid_cols, offset="center")
        xs = np.asarray(xs, dtype=np.float32).reshape(grid_rows.shape)
        ys = np.asarray(ys, dtype=np.float32).reshape(grid_rows.shape)
        zs = dsm_tile[grid_rows, grid_cols].astype(np.float32)

        vertices_world = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)
        local_origin = np.asarray([camera_world[0], camera_world[1], 0.0], dtype=np.float32)
        vertices_local = vertices_world - local_origin[None, :]

        inv_dom_transform = ~self.dom.transform
        dom_cols, dom_rows = inv_dom_transform * (xs.reshape(-1), ys.reshape(-1))
        u = (np.asarray(dom_cols, dtype=np.float32) + 0.5) / float(self.dom.width)
        v_raster = (np.asarray(dom_rows, dtype=np.float32) + 0.5) / float(self.dom.height)
        v = 1.0 - v_raster if self.texture_v_flip else v_raster
        uv = np.stack([u, v], axis=1).astype(np.float32)

        valid_v = np.isfinite(zs)
        if self.nodata is not None:
            valid_v &= zs != float(self.nodata)
        uv_grid = uv.reshape(valid_v.shape[0], valid_v.shape[1], 2)
        valid_v &= (
            np.isfinite(uv_grid[:, :, 0])
            & np.isfinite(uv_grid[:, :, 1])
            & (uv_grid[:, :, 0] >= 0.0)
            & (uv_grid[:, :, 0] <= 1.0)
            & (uv_grid[:, :, 1] >= 0.0)
            & (uv_grid[:, :, 1] <= 1.0)
        )

        tri = self._build_triangles(valid_v)
        if tri.size == 0:
            raise RuntimeError("DSM tile produced no valid triangles")

        return {
            "vertices_local": vertices_local.astype(np.float32),
            "uv": uv,
            "triangles": tri.astype(np.int32),
            "local_origin": local_origin,
            "vertex_count": int(vertices_local.shape[0]),
            "triangle_count": int(tri.shape[0]),
            "dsm_window": [
                int(dsm_window.col_off),
                int(dsm_window.row_off),
                int(dsm_window.width),
                int(dsm_window.height),
            ],
        }

    @staticmethod
    def _build_triangles(valid_v: np.ndarray) -> np.ndarray:
        h, w = valid_v.shape
        ids = np.arange(h * w, dtype=np.int32).reshape(h, w)
        valid00 = valid_v[:-1, :-1]
        valid01 = valid_v[:-1, 1:]
        valid10 = valid_v[1:, :-1]
        valid11 = valid_v[1:, 1:]

        tri_a_mask = valid00 & valid10 & valid11
        tri_b_mask = valid00 & valid11 & valid01
        tri_a = np.stack(
            [ids[:-1, :-1][tri_a_mask], ids[1:, :-1][tri_a_mask], ids[1:, 1:][tri_a_mask]],
            axis=1,
        )
        tri_b = np.stack(
            [ids[:-1, :-1][tri_b_mask], ids[1:, 1:][tri_b_mask], ids[:-1, 1:][tri_b_mask]],
            axis=1,
        )
        if tri_a.size == 0:
            return tri_b
        if tri_b.size == 0:
            return tri_a
        return np.concatenate([tri_a, tri_b], axis=0)

    def _load_dom_texture(self) -> "torch.Tensor":
        if self._texture is not None:
            return self._texture
        if self.debug_texture_mode == "quadrant":
            dom_rgb = self._make_quadrant_texture(512, 512)
        else:
            dom_array = np.moveaxis(self.dom.read(), 0, -1)
            if dom_array.shape[2] == 1:
                dom_array = np.repeat(dom_array, 3, axis=2)
            dom_rgb = np.clip(dom_array[:, :, :3], 0, 255).astype(np.float32) / 255.0
        self._texture = self.torch.as_tensor(
            dom_rgb,
            dtype=self.torch.float32,
            device=self.device,
        )[None, ...]
        return self._texture

    @staticmethod
    def _make_quadrant_texture(height: int, width: int) -> np.ndarray:
        texture = np.zeros((height, width, 3), dtype=np.float32)
        mid_y = height // 2
        mid_x = width // 2
        texture[:mid_y, :mid_x] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        texture[:mid_y, mid_x:] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        texture[mid_y:, :mid_x] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        texture[mid_y:, mid_x:] = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
        return texture

    def _render_mesh(
        self,
        mesh: Dict[str, np.ndarray],
        camera_world: np.ndarray,
        R_camera_to_world_np: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        torch = self.torch
        dr = self.dr

        vertices = torch.as_tensor(mesh["vertices_local"], dtype=torch.float32, device=self.device)
        triangles = torch.as_tensor(mesh["triangles"], dtype=torch.int32, device=self.device)
        uv = torch.as_tensor(mesh["uv"], dtype=torch.float32, device=self.device)
        local_origin = torch.as_tensor(mesh["local_origin"], dtype=torch.float32, device=self.device)
        camera_local = torch.as_tensor(camera_world, dtype=torch.float32, device=self.device) - local_origin
        R_camera_to_world = torch.as_tensor(
            np.asarray(R_camera_to_world_np, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )

        pos_clip, camera_z_m = build_nvdiffrast_camera_projection(
            vertices,
            camera_local,
            R_camera_to_world,
            self.K,
            self.width,
            self.height,
            self.near_m,
            self.far_m,
        )

        rast, _ = dr.rasterize(
            self.ctx,
            pos_clip[None, ...],
            triangles,
            resolution=[self.height, self.width],
        )
        hit_mask = rast[..., 3:4] > 0
        uv_img, _ = dr.interpolate(uv[None, ...], rast, triangles)
        uv_img = uv_img.contiguous()
        camera_z_img, _ = dr.interpolate(camera_z_m[:, None][None, ...], rast, triangles)

        texture = self._load_dom_texture().contiguous()
        color_img = dr.texture(texture, uv_img, filter_mode="linear")
        color_img = torch.where(hit_mask, color_img, torch.zeros_like(color_img))
        depth_img = torch.where(hit_mask, camera_z_img, torch.zeros_like(camera_z_img))
        depth_img = torch.where(depth_img > 0, depth_img, torch.zeros_like(depth_img))

        color_np = (
            torch.clamp(color_img[0], 0.0, 1.0)
            .detach()
            .cpu()
            .numpy()
            * 255.0
        ).astype(np.uint8)
        depth_np = depth_img[0, :, :, 0].detach().cpu().numpy().astype(np.float32)
        if self.output_y_flip:
            color_np = np.flipud(color_np).copy()
            depth_np = np.flipud(depth_np).copy()

        stats = {
            "vertex_count": mesh["vertex_count"],
            "triangle_count": mesh["triangle_count"],
            "dsm_window": mesh["dsm_window"],
        }
        return color_np, depth_np, stats
