#!/usr/bin/env python3
"""PiLoT: Neural Pixel-to-3D Registration for UAV-based Ego and Target Geo-localization

Entry script for running localization via shell scripts.
"""
import argparse
import glob
import logging
import os
import queue
import shutil
import time
from multiprocessing import Event, Queue
from pprint import pformat
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import setproctitle
import torch
import torch.multiprocessing as mp
import re
import yaml

from pixloc.localization import target_indicator
from pixloc.pixlib.datasets.view import read_image_list
from pixloc.pixlib.geometry import Camera
from pixloc.utils.eval import evaluate_pose, evaluate_target
from pixloc.utils.get_depth import (
    generate_render_camera,
    pad_to_multiple,
    sample_3d_points,
)
from pixloc.utils.transform import (
    euler_angles_to_matrix_ECEF,
    get_sorted_image_paths_uavscenes,
)
from pixloc.utils.citygs.pose_convert import euler_trans_to_colmap_c2w
from src.utils.pose_utils import (
    load_initial_pose,
    load_initial_pose_normalized,
    load_pose_dict,
    load_pose_dict_normalized,
    load_target_points,
)

mp.set_start_method("spawn", force=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class DualProcessTask:
    """Manages rendering and localization in two parallel processes."""

    def __init__(
        self,
        config: Dict[str, Any],
        name: Optional[str] = None,
    ) -> None:
        self.task_q: Queue = Queue(maxsize=2)
        self.pose_q: Queue = Queue(maxsize=3)
        self.stop_evt = Event()

        self.render_config = config["render_config"]
        self.render_type = self.render_config.get("type", "osg")
        citygs_cfg = self.render_config.get("citygs", {})
        self.coordinate_system = (
            citygs_cfg.get("coordinate_system")
            or self.render_config.get("coordinate_system", "ecef")
        )
        default_confs = config["default_confs"]
        # CLI --name fills in sequence fields when the YAML leaves them blank
        # (feicuiwan demo: --name 3dgs_test -> images/3dgs_test, poses/3dgs_test.txt).
        if name:
            default_confs.setdefault("sequence_name", name)
            default_confs.setdefault("dataset_name", name)
            default_confs.setdefault("output_name", name)

        self.conf = default_confs["from_render_test"]
        self.enable_target = default_confs.get("enable_target_indicator", True)
        self.enable_viz = default_confs.get("enable_visualization", False)

        folder_path = default_confs["dataset_path"]
        self.sequence_name = (
            default_confs.get("sequence_name")
            or default_confs.get("dataset_name")
            or ""
        )
        self.experiment_name = (
            name
            or default_confs.get("output_name")
            or self.sequence_name
        )

        self.refine_conf = default_confs["refine"]
        self.mul = self.refine_conf["mul"]
        # Mean-brightness (0-255) below which a pixel is treated as background
        # and excluded from 3D point sampling. 0 disables the filter.
        self.black_bg_thresh = float(
            self.refine_conf.get("black_bg_thresh", 0)
        )
        # Fraction trimmed from each image side before sampling 3D points, to
        # avoid distorted query edges (render is pinhole). 0 disables.
        self.edge_mask_ratio = float(
            self.refine_conf.get("edge_mask_ratio", 0)
        )
        # Number of 3D points sampled per frame. More points reduce the
        # per-frame sampling variance that drives jitter in the closed loop.
        self.num_samples = int(self.refine_conf.get("num_samples", 500))

        # Output directories (experiment name only affects outputs)
        output_folder = "outputs"
        self.outputs = os.path.join(output_folder, self.experiment_name)
        if os.path.exists(self.outputs):
            shutil.rmtree(self.outputs)
        os.makedirs(self.outputs)

        # Ground-truth / reference pose file (always tied to sequence_name)
        self.gt_pose_path = os.path.join(
            folder_path, "poses", self.sequence_name + ".txt"
        )

        # Estimation output paths
        self.estimated_pose_path = os.path.join(
            output_folder, self.experiment_name + ".txt"
        )

        if self.enable_target:
            self.gt_target_xy_path = os.path.join(
                folder_path, "bbox", self.sequence_name, self.sequence_name + "_xy.txt"
            )
            self.gt_rtk_path = os.path.join(
                folder_path, "target_RTK", self.sequence_name + "_RTK.txt"
            )
            self.estimated_target_path = os.path.join(
                output_folder, self.experiment_name + "_xyz.txt"
            )

        self.last_frame_info: Dict[str, Any] = {
            "observations": [],
            "refine_conf": self.refine_conf,
        }

        logger.info("Configuration:\n%s", pformat(self.conf))

        self._setup_camera(default_confs)
        self._setup_images(folder_path, self.sequence_name, default_confs)
        self._setup_poses(default_confs)

        self.device = "cuda"
        self.origin = torch.tensor(
            self.origin, device=self.device, dtype=torch.float32
        )
        self.query_camera = self.query_camera.to(self.device)
        self.render_camera = self.render_camera.to(self.device)

        # Seed the render queue with the initial pose (one item only)
        init_tag = "0_init.png"
        self.pose_q.put_nowait(
            (self.euler_angles, self.translation, init_tag, None)
        )
        if self.enable_target:
            self.localizer = target_indicator.QueryLocalizer()

    # -- Setup helpers --------------------------------------------------------

    def _setup_camera(self, default_confs: Dict[str, Any]) -> None:
        """Initialize query and render cameras."""
        cam_cfg = default_confs["cam_query"]
        self.query_resize_ratio = cam_cfg["width"] / cam_cfg["max_size"]

        fx, fy, cx, cy = cam_cfg["params"]
        w, h = cam_cfg["width"], cam_cfg["height"]

        self.raw_query_camera = np.array([w, h, cx, cy, fx, fy])
        use_fixed_render = self.render_type in ("3dgs", "citygs")
        explicit_rc = self.render_config.get("render_camera")
        if use_fixed_render and explicit_rc and len(explicit_rc) == 6:
            # render_camera in YAML is at native query resolution; scale it by
            # max_size so the renderer matches the resized query (e.g. 960→512).
            self.render_camera_gs = np.array(explicit_rc, dtype=float)
            if self.query_resize_ratio != 1.0:
                self.render_camera_gs = (
                    self.render_camera_gs / self.query_resize_ratio
                )
        else:
            self.render_camera_gs = np.array([w, h, w / 2, h / 2, fx, fx])
            self.render_camera_gs = self.render_camera_gs / self.query_resize_ratio

        cam_cfg["params"] = (
            np.array(cam_cfg["params"]) / self.query_resize_ratio
        )
        cam_cfg["width"] /= self.query_resize_ratio
        cam_cfg["height"] /= self.query_resize_ratio

        self.query_camera = Camera.from_colmap(cam_cfg)
        self.render_camera = generate_render_camera(
            self.render_camera_gs
        ).float()
        self.render_config["render_camera"] = self.render_camera_gs

    def _setup_images(
        self,
        folder_path: str,
        dataset_name: str,
        default_confs: Dict[str, Any],
    ) -> None:
        """Build the ordered image list and pre-load query tensors."""
        img_path = default_confs.get("query_path")
        if not img_path:
            img_path = os.path.join(folder_path, "images", dataset_name)
        if not os.path.isdir(img_path):
            raise FileNotFoundError(f"Query image directory not found: {img_path}")

        if "interval" in dataset_name:
            self.img_list = get_sorted_image_paths_uavscenes(img_path)
        else:
            self.img_list = sorted(
                glob.glob(os.path.join(img_path, "*.png"))
                + glob.glob(os.path.join(img_path, "*.jpg"))
                + glob.glob(os.path.join(img_path, "*.JPG")),
                key=lambda p: int(os.path.basename(p).split(".")[0]),
            )
        cam_cfg = default_confs["cam_query"]
        self.query_list = read_image_list(
            self.img_list,
            scale=self.query_resize_ratio,
            distortion=cam_cfg["distortion"],
            query_camera=self.raw_query_camera,
        )

    def _setup_poses(self, default_confs: Dict[str, Any]) -> None:
        """Load initial pose, ground-truth poses and target points."""
        self.num_init_pose = default_confs["num_init_pose"]
        self.trust_prior_sequential = default_confs.get(
            "trust_prior_sequential", False
        )
        self.padding = default_confs["padding"]

        if self.coordinate_system == "normalized":
            citygs_cfg = self.render_config.get("citygs", {})
            init_pose_txt = citygs_cfg.get("init_pose_txt")
            self.euler_angles, self.translation, self.origin = (
                load_initial_pose_normalized(
                    self.gt_pose_path,
                    init_pose_txt=init_pose_txt,
                    init_pose_frame=int(citygs_cfg.get("init_pose_frame", 0)),
                    init_c2w_npy=citygs_cfg.get("init_c2w_npy"),
                    init_frame_idx=int(citygs_cfg.get("init_frame_idx", 0)),
                )
            )
            # Anchor origin at init pose (same role as ECEF first-pose origin)
            self.origin = np.asarray(self.translation, dtype=np.float64)
            default_confs["refine"]["origin"] = self.translation
            # Same mul as ECEF: optimizer is trained on coords scaled by 0.001
            default_confs["refine"]["mul"] = 0.001
            default_confs["refine"]["coordinate_system"] = "normalized"
            self.mul = 0.001

            pose_dict_path = self.gt_pose_path
            if not os.path.isfile(pose_dict_path) and init_pose_txt:
                pose_dict_path = init_pose_txt
                logger.warning(
                    "GT pose file missing (%s), using init pose file: %s",
                    self.gt_pose_path, init_pose_txt,
                )
            self.gt_pose_dict = (
                load_pose_dict_normalized(pose_dict_path, origin=self.origin)
                if os.path.isfile(pose_dict_path)
                else {}
            )
            self.trust_init_pose = bool(init_pose_txt)
        else:
            self.euler_angles, self.translation, self.origin = load_initial_pose(
                self.gt_pose_path,
            )
            default_confs["refine"]["origin"] = self.origin
            self.gt_pose_dict = load_pose_dict(
                self.gt_pose_path, origin=self.origin
            )
            self.trust_init_pose = False
        self.render_config["init_rot"] = self.euler_angles
        self.render_config["init_trans"] = self.translation
        self.render_config["coordinate_system"] = self.coordinate_system
        if self.enable_target:
            self.target_xy_dict = load_target_points(self.gt_target_xy_path)

    # -- Workers --------------------------------------------------------------

    def rendering_worker(self) -> None:
        """Process that renders images and computes target locations.

        Supports three backends selected via ``render_config["type"]``:
        * ``"osg"``     – OpenSceneGraph (default, legacy)
        * ``"3dgs"``    – vanilla 3D Gaussian Splatting (PLY)
        * ``"citygs"``  – CityGaussian checkpoint renderer
        """
        setproctitle.setproctitle("PiLoT_Render")
        use_3dgs = self.render_type == "3dgs"
        use_citygs = self.render_type == "citygs"

        if use_citygs:
            from pixloc.utils.citygs.citygs_render import CityGaussianRenderer
            renderer = CityGaussianRenderer(self.render_config)
        elif use_3dgs:
            from pixloc.utils.gs3d.gs3d_render import GS3DRenderer
            renderer = GS3DRenderer(self.render_config)
        else:
            from pixloc.utils.osg import osg_render
            renderer = osg_render.RenderImageProcessor(self.render_config)
            render_stream = torch.cuda.Stream()

        torch.cuda.synchronize()
        target_results: List[str] = []

        while True:
            try:
                item = self.pose_q.get(timeout=1)
            except queue.Empty:
                if self.stop_evt.is_set():
                    break
                continue

            if item is None:
                break

            euler, trans, qname, _fps = item

            if use_3dgs or use_citygs:
                color, depth = renderer.render(trans, euler)
            else:
                with torch.cuda.stream(render_stream):
                    for _ in range(20):
                        renderer.update_pose(trans, euler)
                    color = renderer.get_color_image()
                    depth = renderer.get_depth_image()
                torch.cuda.current_stream().synchronize()

            saved_render = False
            if "init" not in qname and self.enable_target:
                target_pt = (
                    np.array(self.target_xy_dict[qname])
                    / self.query_resize_ratio
                )
                ret = self.localizer.get_target_location(
                    target_pt, [trans, euler], depth, self.render_camera,
                )
                pt3d = ret[1][0]
                pt2d = (int(target_pt[0][0]), int(target_pt[0][1]))

                color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                cv2.circle(
                    color_bgr, pt2d, radius=6,
                    color=(0, 0, 255), thickness=-1,
                )
                label = f"[{pt3d[0]:.6f}, {pt3d[1]:.6f}, {pt3d[2]:.1f}]"
                cv2.putText(
                    color_bgr, label,
                    (pt2d[0] + 10, pt2d[1] - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6, color=(0, 255, 0),
                    thickness=2, lineType=cv2.LINE_AA,
                )
                cv2.imwrite(os.path.join(self.outputs, qname), color_bgr)
                saved_render = True
                target_results.append(
                    f"{qname} {' '.join(map(str, pt3d))}"
                )
            elif self.enable_viz:
                color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(self.outputs, qname), color_bgr)
                saved_render = True

            if saved_render:
                depth_np = (
                    depth if isinstance(depth, np.ndarray) else depth.cpu().numpy()
                )
                np.save(
                    os.path.join(self.outputs, qname.rsplit(".", 1)[0] + ".npy"),
                    depth_np,
                )

            if self.padding:
                color = pad_to_multiple(color, 16)

            try:
                self.task_q.put((color, depth, euler, trans), timeout=1)
            except queue.Full:
                break

        if self.enable_target:
            with open(self.estimated_target_path, "w") as f:
                f.write("\n".join(target_results))

        self.stop_evt.set()
        self.task_q.put(None)
        self.task_q.close()
        self.task_q.join_thread()
        self.pose_q.close()
        self.pose_q.join_thread()
        logger.info("Rendering worker finished.")

    def localization_worker(self) -> None:
        """Process that runs pose refinement on each query frame."""
        from pixloc.localization.localizer import RenderLocalizer

        setproctitle.setproctitle("PiLoT_Localize")
        localizer = RenderLocalizer(self.conf)
        results: List[str] = []
        last_euler: Optional[List[float]] = None
        last_trans: Optional[List[float]] = None

        for idx, (img_path, img_tensor) in enumerate(
            zip(self.img_list, self.query_list)
        ):
            item = self.task_q.get()
            if item is None:
                break

            color, depth, render_euler, render_trans = item

            # Refine relative to the render pose that produced this depth/color
            query_euler, query_trans = render_euler, render_trans
            trust_prior = self.trust_prior_sequential or (
                (idx == 0) and self.trust_init_pose
            )
            is_init = (
                (idx == 0)
                and not self.trust_init_pose
                and not self.trust_prior_sequential
            )
            p3d, T_w2c, T_init, dd = self.back_project(
                depth, render_euler, render_trans,
                query_euler, query_trans, is_init,
                trust_prior=trust_prior,
                color_frame=color,
                num_samples=self.num_samples,
            )

            t0 = time.time()
            ret = localizer.run_query(
                img_path,
                self.query_camera,
                self.render_camera,
                color,
                query_T=T_init,
                render_T=T_w2c,
                Points_3D_ECEF=p3d,
                query_resize_ratio=self.query_resize_ratio,
                dd=dd,
                gt_pose_dict=self.gt_pose_dict,
                last_frame_info=self.last_frame_info,
                image_query=img_tensor,
            )

            if ret.get("success", True) and ret.get("euler_angles") is not None:
                last_euler = ret["euler_angles"]
                last_trans = ret["translation"]
                if hasattr(last_euler, "tolist"):
                    last_euler = last_euler.tolist()
                if hasattr(last_trans, "tolist"):
                    last_trans = last_trans.tolist()
            elif not ret.get("success", True):
                logger.warning(
                    "Frame %d optimization failed, keep render pose", idx
                )
                last_euler, last_trans = query_euler, query_trans
            else:
                last_euler, last_trans = query_euler, query_trans

            qname = os.path.basename(img_path)

            elapsed_ms = (time.time() - t0) * 1000
            if idx % 30 == 0:
                logger.info("Frame %d | %.1f ms", idx, elapsed_ms)

            if idx < len(self.img_list) - 1:
                self.pose_q.put(
                    (last_euler, last_trans, qname, None)
                )

            ea = last_euler
            logger.info(
                "Frame %d | euler=%s | trans=%s",
                idx, ea, last_trans,
            )
            if self.coordinate_system == "normalized":
                results.append(
                    f"{qname} "
                    f"{' '.join(map(str, last_trans))} "
                    f"{' '.join(map(str, ea))}"
                )
            else:
                results.append(
                    f"{qname} "
                    f"{' '.join(map(str, last_trans))} "
                    f"{ea[1]} {ea[0]} {ea[2]}"
                )

            if self.stop_evt.is_set():
                break

        with open(self.estimated_pose_path, "w") as f:
            f.write("\n".join(results))

        if self.enable_viz:
            self._build_viz_video()

        self.pose_q.put(None)
        self.stop_evt.set()
        self.task_q.close()
        self.task_q.join_thread()
        self.pose_q.close()
        self.pose_q.join_thread()
        logger.info("Localization worker finished.")

    # -- Visualization --------------------------------------------------------

    def _compose_viz_frame(
        self,
        ref_bgr: np.ndarray,
        query_bgr: np.ndarray,
        frame_idx: int,
    ) -> np.ndarray:
        """Overlay query on ref at ref resolution (center-window blend).

        ``ref_bgr`` and ``query_bgr`` must be the same shape ``(H, W, 3)``.
        """
        H, W = ref_bgr.shape[:2]
        if query_bgr.shape[:2] != (H, W):
            query_bgr = cv2.resize(query_bgr, (W, H), interpolation=cv2.INTER_AREA)

        margin_y = max(1, H // 4)
        margin_x = max(1, W // 4)
        y0, y1 = margin_y, H - margin_y
        x0, x1 = margin_x, W - margin_x

        out = query_bgr.astype(np.float32)
        ref_block = ref_bgr[y0:y1, x0:x1].astype(np.float32)
        query_block = query_bgr[y0:y1, x0:x1].astype(np.float32)
        alpha = 0
        blended = (1.0 - alpha) * ref_block + alpha * query_block
        out[y0:y1, x0:x1] = blended
        canvas = np.clip(out, 0, 255).astype(np.uint8)

        cv2.putText(
            canvas,
            f"ref|query blend  frame {frame_idx}",
            (max(4, x0), max(20, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        return canvas

    def _build_viz_video(self) -> None:
        """Build ``visualization.mp4`` after the run, pairing by filename.

        Renders are saved during ``rendering_worker`` as ``outputs/<qname>``;
        the query image is ``data_demo/.../images/<seq>/<qname>``. This avoids
        queue/timing mismatch between ``color`` and ``img_path`` in the
        localization loop.
        """
        W = int(self.render_camera_gs[0])
        H = int(self.render_camera_gs[1])
        if W <= 0 or H <= 0:
            return

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_path = os.path.join(self.outputs, "visualization.mp4")
        writer: Optional[cv2.VideoWriter] = None
        n_frames = 0

        for idx, img_path in enumerate(self.img_list):
            qname = os.path.basename(img_path)
            render_path = os.path.join(self.outputs, qname)
            if not os.path.isfile(render_path):
                logger.warning(
                    "Viz skip missing render for %s (expected %s)",
                    qname, render_path,
                )
                continue

            query_bgr = cv2.imread(img_path)
            ref_bgr = cv2.imread(render_path)
            if query_bgr is None or ref_bgr is None:
                logger.warning("Viz skip failed read: %s / %s", img_path, render_path)
                continue

            ref_bgr = cv2.resize(ref_bgr, (W, H), interpolation=cv2.INTER_AREA)
            query_bgr = cv2.resize(query_bgr, (W, H), interpolation=cv2.INTER_AREA)

            canvas = self._compose_viz_frame(ref_bgr, query_bgr, idx)
            if writer is None:
                writer = cv2.VideoWriter(video_path, fourcc, 10, (W, H))
            writer.write(canvas)
            n_frames += 1

        if writer is not None:
            writer.release()
            logger.info(
                "Visualization video saved (%d frames): %s", n_frames, video_path,
            )
        else:
            logger.warning(
                "No visualization video written (no frames; check %s)", self.outputs,
            )

    # -- Helpers --------------------------------------------------------------

    def back_project(
        self,
        depth_frame: np.ndarray,
        euler_angles: List[float],
        translation: List[float],
        query_euler: List[float],
        query_trans: List[float],
        is_init: bool = True,
        trust_prior: bool = False,
        num_samples: int = 500,
        depth_min: float = 1.0,
        depth_max: float = 5000.0,
        color_frame: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Back-project depth to 3D points and build initial pose candidates."""
        device = self.device
        depth = (
            depth_frame.to(device)
            if torch.is_tensor(depth_frame)
            else torch.as_tensor(depth_frame, device=device)
        )

        if self.coordinate_system == "normalized":
            T_c2w_np = euler_trans_to_colmap_c2w(translation, euler_angles)
        else:
            T_c2w_np = euler_angles_to_matrix_ECEF(euler_angles, translation)
        T_c2w = torch.as_tensor(T_c2w_np, device=device, dtype=torch.float32)

        H = int(self.render_camera_gs[1])
        W = int(self.render_camera_gs[0])

        if self.coordinate_system == "normalized":
            depth_min_use, depth_max_use = 0.1, 200.0
        else:
            depth_min_use, depth_max_use = depth_min, depth_max

        valid_mask = (
            (depth >= depth_min_use)
            & (depth <= depth_max_use)
            & torch.isfinite(depth)
        )

        # Exclude near-black background: sky / empty Gaussians still emit a
        # depth value that overlaps the foreground range, so they cannot be
        # filtered by depth alone. Sampling them yields garbage 3D points that
        # bias the learned optimizer (worse at low resolution / fewer pixels).
        if color_frame is not None and self.black_bg_thresh > 0:
            color_t = (
                color_frame
                if torch.is_tensor(color_frame)
                else torch.as_tensor(color_frame, device=device)
            ).to(device)
            # color may be padded to a multiple of 16; crop to depth size
            color_t = color_t[: depth.shape[0], : depth.shape[1]]
            brightness = color_t.float().mean(dim=-1)
            fg_mask = brightness > self.black_bg_thresh
            if fg_mask.shape == valid_mask.shape and fg_mask.any():
                valid_mask = valid_mask & fg_mask

        # Edge mask: drop a border band of the image. The query is a raw drone
        # frame with lens distortion while the render is pure pinhole; only the
        # center aligns, so sampling 3D points near the edges feeds the learned
        # optimizer systematically mismatched correspondences. Restricting the
        # sampled points to the center projects them into the well-aligned query
        # region. Ratio is the fraction trimmed from each side.
        if self.edge_mask_ratio > 0:
            Hd, Wd = depth.shape[0], depth.shape[1]
            my = int(round(Hd * self.edge_mask_ratio))
            mx = int(round(Wd * self.edge_mask_ratio))
            if my > 0 or mx > 0:
                edge_mask = torch.zeros_like(valid_mask)
                edge_mask[my : Hd - my, mx : Wd - mx] = True
                if edge_mask.any():
                    valid_mask = valid_mask & edge_mask

        valid_idx = valid_mask.nonzero(as_tuple=False)
        if valid_idx.numel() == 0:
            valid_mask = (depth > 0) & torch.isfinite(depth)
            valid_idx = valid_mask.nonzero(as_tuple=False)
        if valid_idx.numel() == 0:
            d = depth.detach().float().cpu()
            finite = torch.isfinite(d)
            logger.error(
                "No valid depth pixels (pose euler=%s trans=%s); "
                "depth finite=%d/%d min=%s max=%s",
                euler_angles,
                translation,
                int(finite.sum()),
                d.numel(),
                d[finite].min().item() if finite.any() else "n/a",
                d[finite].max().item() if finite.any() else "n/a",
            )
            raise RuntimeError(
                "No valid depth pixels in render; check pose / renderer output."
            )
        n_pick = min(num_samples, valid_idx.shape[0])
        perm = torch.randperm(valid_idx.shape[0], device=device)[:n_pick]
        sel = valid_idx[perm]
        points2d = torch.stack(
            (sel[:, 1].float(), sel[:, 0].float()), dim=1
        )

        return sample_3d_points(
            points2d, depth, T_c2w, self.render_camera,
            query_euler, query_trans,
            origin=self.origin, mul=self.mul, is_init_frame=is_init,
            trust_prior=trust_prior,
            coordinate_system=self.coordinate_system,
        )

    def evaluate(self) -> None:
        """Evaluate localization results against ground truth."""
        from pixloc.utils.eval import evaluate_pose, evaluate_pose_normalized
        if not os.path.isfile(self.estimated_pose_path):
            logger.warning(
                "Skip evaluation: results not found: %s", self.estimated_pose_path
            )
            return
        if not os.path.isfile(self.gt_pose_path):
            logger.warning("Skip evaluation: GT pose file not found: %s", self.gt_pose_path)
            return
        if self.coordinate_system == "normalized":
            evaluate_pose_normalized(self.estimated_pose_path, self.gt_pose_path)
        else:
            evaluate_pose(self.estimated_pose_path, self.gt_pose_path)
        if self.enable_target:
            evaluate_target(self.estimated_target_path, self.gt_rtk_path)

    def run(self) -> None:
        """Launch rendering and localization as parallel processes."""
        ctx = mp.get_context("spawn")
        p_render = ctx.Process(target=self.rendering_worker, daemon=True)
        p_loc = ctx.Process(target=self.localization_worker, daemon=True)

        p_render.start()
        p_loc.start()

        p_loc.join()
        p_render.join(5)

        if p_render.is_alive():
            p_render.terminate()
            p_render.join()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def expand_env_vars(obj):
    """Expand ``${VAR}`` and ``${VAR:-default}`` in YAML string values."""
    if isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env_vars(v) for v in obj]
    if isinstance(obj, str):
        def _repl(match: re.Match) -> str:
            var = match.group(1)
            default = match.group(2)
            if default is not None:
                return os.environ.get(var, default)
            return os.environ.get(var, match.group(0))
        return re.sub(r"\$\{([^}:]+)(?::-([^}]*))?\}", _repl, obj)
    return obj


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PiLoT: Pixel-Level Localization and Tracking",
    )
    parser.add_argument(
        "-c", "--config", type=str,
        default="configs/demos/smbu_seq2.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Dataset / sequence name (overrides the value in config)",
    )
    # Accepted for shell-script compatibility; the initial pose is read
    # from the first line of the pose file specified in the config.
    parser.add_argument("--init_euler", type=str, default="[0,0,0]")
    parser.add_argument("--init_trans", type=str, default="[0,0,0]")
    parser.add_argument(
        "--viz", action="store_true", default=None,
        help="Enable visualization (save rendered images + PIP video)",
    )
    parser.add_argument(
        "--no-viz", dest="viz", action="store_false",
        help="Disable visualization",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = expand_env_vars(yaml.safe_load(f))

    if config.get("render_config", {}).get("type") == "citygs":
        from pixloc.utils.citygs.citygs_render import resolve_citygaussian_root
        cg = config["render_config"].setdefault("citygs", {})
        cg["citygs_root"] = resolve_citygaussian_root(
            cg.get("citygs_root", "third_party/CityGaussian")
        )

    if args.viz is not None:
        config["default_confs"]["enable_visualization"] = args.viz

    task = DualProcessTask(config, name=args.name)
    task.run()
    task.evaluate()
    logger.info("Done.")


if __name__ == "__main__":
    main()
