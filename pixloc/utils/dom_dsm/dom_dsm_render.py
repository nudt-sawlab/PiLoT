"""Prototype DOM + DSM renderer for PiLoT.

This backend renders a perspective RGB reference and a camera-view depth map
from georeferenced DOM/DSM rasters. It intentionally favors correctness and
debuggability over speed for the first integration pass.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

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
        self.debug_dir = dom_dsm_cfg.get("debug_dir")
        self.debug_every = int(dom_dsm_cfg.get("debug_every", 1))
        self._render_count = 0

        if not self.dom_path.is_file():
            raise FileNotFoundError(f"DOM GeoTIFF not found: {self.dom_path}")
        if not self.dsm_path.is_file():
            raise FileNotFoundError(f"DSM GeoTIFF not found: {self.dsm_path}")

        self.dom = rasterio.open(self.dom_path)
        self.dsm = rasterio.open(self.dsm_path)
        self._validate_rasters()

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
            "DOMDSMRenderer ready: %dx%d, DOM=%s, DSM=%s",
            self.width,
            self.height,
            self.dom_path,
            self.dsm_path,
        )

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
        lon, lat, alt = map(float, trans)
        cam_x, cam_y = self.to_raster.transform(lon, lat)

        rot_cam_to_enu = R.from_euler("xyz", euler, degrees=True).as_matrix()
        dirs_enu_per_z = self.camera_dirs.reshape(-1, 3) @ rot_cam_to_enu.T

        n_pix = dirs_enu_per_z.shape[0]
        hit_x = np.full(n_pix, np.nan, dtype=np.float64)
        hit_y = np.full(n_pix, np.nan, dtype=np.float64)
        depth = np.zeros(n_pix, dtype=np.float32)
        active = np.ones(n_pix, dtype=bool)

        for z_depth in self.depth_values:
            active_ids = np.flatnonzero(active)
            if active_ids.size == 0:
                break

            dirs = dirs_enu_per_z[active_ids]
            xs = cam_x + dirs[:, 0] * z_depth
            ys = cam_y + dirs[:, 1] * z_depth
            zs = alt + dirs[:, 2] * z_depth

            terrain = self._sample_dsm(xs, ys)
            valid = np.isfinite(terrain)
            if self.nodata is not None:
                valid &= terrain != float(self.nodata)

            hits = valid & (zs <= terrain)
            if np.any(hits):
                hit_ids = active_ids[hits]
                hit_x[hit_ids] = xs[hits]
                hit_y[hit_ids] = ys[hits]
                depth[hit_ids] = z_depth
                active[hit_ids] = False

        color = np.zeros((n_pix, 3), dtype=np.uint8)
        hit_mask = depth > 0
        if np.any(hit_mask):
            color[hit_mask] = self._sample_dom(hit_x[hit_mask], hit_y[hit_mask])

        color_img = color.reshape(self.height, self.width, 3)
        depth_img = depth.reshape(self.height, self.width)
        self._write_debug(color_img, depth_img)
        return color_img, depth_img

    def _sample_dsm(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        coords = list(zip(xs.tolist(), ys.tolist()))
        return np.asarray([v[0] for v in self.dsm.sample(coords)], dtype=np.float32)

    def _sample_dom(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        coords = list(zip(xs.tolist(), ys.tolist()))
        samples = np.asarray(list(self.dom.sample(coords)))
        if samples.ndim != 2 or samples.shape[0] == 0:
            return np.zeros((len(xs), 3), dtype=np.uint8)

        if samples.shape[1] == 1:
            rgb = np.repeat(samples[:, :1], 3, axis=1)
        else:
            rgb = samples[:, :3]
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def _write_debug(self, color: np.ndarray, depth: np.ndarray) -> None:
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
