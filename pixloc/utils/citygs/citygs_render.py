"""CityGaussian renderer wrapper for PiLoT.

Loads a CityGaussian checkpoint (as used by CityGaussian render tools) and
renders colour + depth from PiLoT 6-DOF poses.
"""

import logging
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

from .colmap_io import (
    compute_similarity_transform,
    detect_sparse_dir,
    load_reference_c2w_model,
)
from .pose_convert import (
    c2w_to_w2c_rt,
    euler_trans_to_colmap_c2w,
    pilot_pose_to_model_c2w,
    resolve_ecef_to_model,
)

logger = logging.getLogger(__name__)


def resolve_citygaussian_root(path: str) -> str:
    """Resolve CityGaussian install dir; auto-detect sibling clone if missing."""
    root = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(os.path.join(root, "internal")):
        return root
    # pixloc/utils/citygs -> repo root is 3 levels up
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    for candidate in (
        os.path.join(repo_root, "third_party", "CityGaussian"),
        os.path.join(repo_root, "..", "CityGaussian"),
    ):
        candidate = os.path.abspath(candidate)
        if os.path.isdir(os.path.join(candidate, "internal")):
            logger.info("CityGaussian root: %s (auto-detected)", candidate)
            return candidate
    raise FileNotFoundError(
        f"CityGaussian not found at {root!r}. Clone it to "
        "third_party/CityGaussian or set CITYGAUSSIAN_ROOT."
    )


class CityGaussianRenderer:
    """Render colour + depth from PiLoT poses using a CityGaussian checkpoint."""

    def __init__(self, config: Dict) -> None:
        cg_cfg = config["citygs"]
        self.citygs_root = resolve_citygaussian_root(cg_cfg["citygs_root"])
        checkpoint = os.path.abspath(cg_cfg["checkpoint"])
        data_path = os.path.abspath(cg_cfg["data_path"])
        sparse_subdir = cg_cfg.get("sparse_subdir", "")

        if self.citygs_root not in sys.path:
            sys.path.insert(0, self.citygs_root)

        from internal.cameras.cameras import Cameras
        from internal.utils.gaussian_model_loader import GaussianModelLoader

        self._Cameras = Cameras

        self.coordinate_system = cg_cfg.get(
            "coordinate_system",
            config.get("coordinate_system", "ecef"),
        )

        sparse_dir = detect_sparse_dir(data_path, sparse_subdir)
        self.similarity = compute_similarity_transform(sparse_dir)
        logger.info("Loaded COLMAP sparse model from %s", sparse_dir)

        self.ecef_to_model = None
        if self.coordinate_system == "normalized":
            logger.info(
                "CityGaussianRenderer using normalized model coordinates "
                "(no ECEF transform)"
            )
        else:
            reference_image = cg_cfg.get("reference_image")
            reference_c2w_model = None
            if reference_image:
                reference_c2w_model = load_reference_c2w_model(
                    sparse_dir, self.similarity, reference_image,
                )

            calib_cfg = dict(cg_cfg)
            if (
                cg_cfg.get("reference_use_init")
                and reference_c2w_model is not None
            ):
                calib_cfg["reference_calibration"] = {
                    "euler": config["init_rot"],
                    "trans": config["init_trans"],
                }

            self.ecef_to_model = resolve_ecef_to_model(
                calib_cfg, reference_c2w_model,
            )
            logger.info(
                "Resolved ECEF-to-model transform for CityGaussian rendering"
            )

        render_camera = config["render_camera"]
        self.width = int(render_camera[0])
        self.height = int(render_camera[1])
        self.cx = float(render_camera[2])
        self.cy = float(render_camera[3])
        self.fx = float(render_camera[4])
        self.fy = float(render_camera[5])

        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f"CityGaussian checkpoint not found: {checkpoint}"
            )

        device = torch.device("cuda")
        self.model, self.renderer, ckpt = (
            GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
                checkpoint,
                device=device,
                eval_mode=True,
                pre_activate=False,
            )
        )
        self.model.freeze()
        self.model.pre_activate_all_properties()
        self.bkgd_color = torch.tensor(
            ckpt["hyper_parameters"]["background_color"], device=device,
        )
        self.device = device

        logger.info(
            "CityGaussianRenderer ready: %dx%d, checkpoint=%s, gaussians=%d",
            self.width,
            self.height,
            checkpoint,
            self.model.get_xyz.shape[0],
        )

    def _build_camera(self, c2w: np.ndarray):
        R, T = c2w_to_w2c_rt(c2w)
        return self._Cameras(
            R=torch.tensor(R[None], dtype=torch.float32),
            T=torch.tensor(T[None], dtype=torch.float32),
            fx=torch.tensor([self.fx], dtype=torch.float32),
            fy=torch.tensor([self.fy], dtype=torch.float32),
            cx=torch.tensor([self.cx], dtype=torch.float32),
            cy=torch.tensor([self.cy], dtype=torch.float32),
            width=torch.tensor([self.width], dtype=torch.int16),
            height=torch.tensor([self.height], dtype=torch.int16),
            appearance_id=torch.zeros(1, dtype=torch.int),
            normalized_appearance_id=torch.zeros(1, dtype=torch.float32),
            distortion_params=None,
            camera_type=torch.zeros(1, dtype=torch.int8),
        )

    def render_c2w(self, c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Render colour and depth from a COLMAP camera-to-world matrix."""
        cameras = self._build_camera(np.asarray(c2w, dtype=np.float64))
        cam = cameras[0].to_device(self.device)

        with torch.no_grad():
            outputs = self.renderer(cam, self.model, self.bkgd_color)

        color = (
            outputs["render"]
            .clamp(0.0, 1.0)
            .mul(255.0)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        depth = (
            outputs["surf_depth"][0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        return color, depth

    def render(
        self,
        trans: List[float],
        euler: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render colour and depth from a PiLoT 6-DOF pose."""
        if self.coordinate_system == "normalized":
            c2w = euler_trans_to_colmap_c2w(trans, euler)
        else:
            c2w = pilot_pose_to_model_c2w(trans, euler, self.ecef_to_model)
        return self.render_c2w(c2w)
