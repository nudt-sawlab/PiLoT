"""3D Gaussian Splatting renderer wrapper for PiLoT.

Provides the same colour + depth output contract as the OSG renderer so that
``rendering_worker`` in ``main.py`` can switch between backends transparently.
"""

import logging
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from .gaussian_model import GaussianModel
from .gaussian_renderer import render as gs_render
from .graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2
from .pose_convert import dof6_to_matrix

logger = logging.getLogger(__name__)


class GS3DRenderer:
    """Load a trained 3DGS model and render colour + depth from any 6-DOF pose."""

    def __init__(self, config: Dict) -> None:
        gs3d_cfg = config["gs3d"]
        ply_path: str = gs3d_cfg["ply_path"]
        sh_degree: int = gs3d_cfg.get("sh_degree", 3)
        white_bg: bool = gs3d_cfg.get("white_background", False)
        self.cgcs_offset = np.array(
            gs3d_cfg.get("cgcs_offset", [401448, 3131258, 0]),
            dtype=np.float64,
        )

        self.pipe = type(
            "Pipe", (), {"debug": False},
        )()

        bg = [1.0, 1.0, 1.0] if white_bg else [0.0, 0.0, 0.0]
        self.background = torch.tensor(bg, dtype=torch.float32, device="cuda")

        # ---- load gaussian model -----------------------------------------
        self.gaussians = GaussianModel(sh_degree)
        self._load_ply(ply_path)

        # ---- camera intrinsics from render_camera [w, h, cx, cy, fx, fy] -
        render_camera = config["render_camera"]
        w, h = int(render_camera[0]), int(render_camera[1])
        fx, fy = render_camera[4], render_camera[5]

        self.image_width = w
        self.image_height = h
        self.FoVx = focal2fov(fx, w)
        self.FoVy = focal2fov(fy, h)

        znear, zfar = 0.01, 1000.0
        self.projection_matrix = (
            getProjectionMatrix(
                znear=znear, zfar=zfar, fovX=self.FoVx, fovY=self.FoVy,
            )
            .transpose(0, 1)
            .cuda()
        )

        logger.info(
            "GS3DRenderer ready: %dx%d, FoV=(%.2f°, %.2f°)",
            w, h, math.degrees(self.FoVx), math.degrees(self.FoVy),
        )

    # -- model loading ----------------------------------------------------

    def _load_ply(self, ply_path: str) -> None:
        """Load a 3DGS model directly from a PLY file."""
        if not os.path.isfile(ply_path):
            raise FileNotFoundError(f"PLY file not found: {ply_path}")
        self.gaussians.load_ply(ply_path)
        self.scene_trans = np.zeros(3)
        self.scene_scale = 1.0
        logger.info("Loaded 3DGS model from %s", ply_path)

    # -- rendering --------------------------------------------------------

    def render(
        self,
        trans: List[float],
        euler: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render colour and depth from a 6-DOF pose.

        Args:
            trans: ``[lon, lat, alt]`` in WGS-84.
            euler: ``[pitch, roll, yaw]`` in degrees.

        Returns:
            ``(color, depth)`` –
            *color* is ``(H, W, 3)`` uint8 RGB numpy array,
            *depth* is ``(H, W)`` float32 numpy array.
        """
        lat, lon, alt = trans[1], trans[0], trans[2]
        roll_in, pitch_in, yaw_in = euler[1], euler[0], euler[2]

        c2w = dof6_to_matrix(
            lat, lon, alt, roll_in, pitch_in, yaw_in,
            cgcs_offset=self.cgcs_offset,
        )

        R_cw = c2w[:3, :3]
        T_wc = -(R_cw.T @ c2w[:3, 3])

        w2v = getWorld2View2(R_cw, T_wc, self.scene_trans, self.scene_scale)
        world_view_transform = (
            torch.tensor(w2v, dtype=torch.float32).transpose(0, 1).cuda()
        )
        full_proj_transform = (
            world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        camera_center = world_view_transform.inverse()[3, :3]

        cam = type("_Cam", (), {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "FoVx": self.FoVx,
            "FoVy": self.FoVy,
            "world_view_transform": world_view_transform,
            "full_proj_transform": full_proj_transform,
            "camera_center": camera_center,
        })()

        with torch.no_grad():
            pkg = gs_render(cam, self.gaussians, self.pipe, self.background)

        color = torch.clamp(pkg["render"], 0.0, 1.0)
        color_np = (color.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        depth_np = pkg["depth"].squeeze(0).cpu().numpy()

        return color_np, depth_np
