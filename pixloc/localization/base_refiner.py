import logging
import time
import copy
import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union
from omegaconf import DictConfig, OmegaConf as oc

from .feature_extractor import FeatureExtractor
from .tracker import BaseTracker
from ..pixlib.geometry import Pose, Camera
from ..pixlib.datasets.view import read_image
from ..utils.transform import kf_predictor, pixloc_to_osg, move_inputs_to_cuda
from ..utils.citygs.pose_convert import c2w_colmap_to_euler_trans
from ..utils.get_depth import zero_pad

logger = logging.getLogger(__name__)

def orthogonalize_rotation_batch(R: torch.Tensor) -> torch.Tensor:
    """Project a batch of rotation matrices to SO(3) using SVD."""
    U, S, Vh = torch.linalg.svd(R)
    return U @ Vh

def build_world_c2w_batch(T_batch: Pose, lever_arm: Optional[torch.Tensor], 
                         scale_factor: float, origin: torch.Tensor,
                         flip_yz: bool = True) -> torch.Tensor:
    """
    Convert a batch of w2c Poses to c2w tensors in the world/ECEF coordinate system.

    `flip_yz` must mirror the convention used when building the LM-space poses in
    sample_3d_points: True for the OSG/ECEF path, False for normalized/COLMAP.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

    # Transfer to target device and type
    r_in = T_batch.R.to(device=device, dtype=dtype)
    t_in = T_batch.t.to(device=device, dtype=dtype)
    origin = torch.as_tensor(origin, device=device, dtype=dtype)
    scale_factor = torch.as_tensor(scale_factor, device=device, dtype=dtype)

    batch_size = r_in.shape[0]
    
    # Orthogonalize and adjust for lever-arm (dd)
    r_ortho = orthogonalize_rotation_batch(r_in)
    t = t_in
    if lever_arm is not None:
        lever_arm = lever_arm.to(device=device, dtype=dtype)
        t = t - (r_ortho @ lever_arm)

    r_c2w = r_ortho.transpose(-1, -2)
    t_c2w = (-r_c2w @ t.unsqueeze(-1)).squeeze(-1) / scale_factor

    T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    T[:, :3, :3] = r_c2w
    if flip_yz:
        T[:, :3, 1:3] *= -1  # OSG/ECEF coordinate convention adjustment
    T[:, :3, 3] = t_c2w + origin

    return T

class BaseRefiner:
    """Base class for pose refinement, optimized for UAV geo-localization."""
    
    base_default_config = dict(
        layer_indices=None,
        min_matches_db=10,
        num_dbs=1,
        min_track_length=3,
        min_points_opt=10,
        point_selection='all',
        average_observations=False,
        normalize_descriptors=True,
        compute_uncertainty=True,
    )

    default_config = dict()
    tracker: BaseTracker = None

    def __init__(self,
                 device: torch.device,
                 optimizer: Union[torch.nn.Module, List],
                 feature_extractor: FeatureExtractor,
                 conf: Union[DictConfig, Dict]):
        self.device = device
        self.optimizer = optimizer
        self.feature_extractor = feature_extractor
        self.prior = False
        self.conf = oc.merge(
            oc.create(self.base_default_config),
            oc.create(self.default_config),
            oc.create(conf))
        # Temporal cost across consecutive query frames.
        self.use_temporal = bool(self.conf.get('use_temporal', False))
        self.last_feature_query = None
        self.last_last_feature_query = None

    def refine_query_pose(self, qname: str, qcamera: Camera, ref_camera: Camera, render_frame: torch.Tensor, 
                          T_query_initial: Pose, T_render: Pose, points_3d_ecef: torch.Tensor, dd=None,
                          last_frame_info: Dict = {}, query_resize_ratio: float = 1.0, 
                          image_query: Optional[torch.Tensor] = None) -> Dict:
        """Refines the query camera pose using dense feature alignment."""
        
        # Prepare inputs
        inputs = move_inputs_to_cuda(qcamera, ref_camera, render_frame, T_query_initial, 
                                     T_render, points_3d_ecef, dd, last_frame_info)
        qcamera, ref_camera, render_frame, T_query_initial, T_render, points_3d_ecef, dd, last_frame_info = inputs
        
        refine_conf = last_frame_info['refine_conf']
        origin = np.array(refine_conf.get('origin', [0, 0, 0]))
        scale_mul = refine_conf.get('mul', 1.0)
        
        # Image padding and processing
        q_w, _ = qcamera.size
        if image_query is None:
            image_query = read_image(qname, scale=query_resize_ratio)

        image_query = zero_pad(int(q_w.item()), image_query)
        render_frame = zero_pad(int(q_w.item()), render_frame)

        # Feature extraction
        features_ref, scales_ref = self.dense_feature_extraction(render_frame)
        features_q, scales_q = self.dense_feature_extraction(image_query)
        
        # Pose refinement
        ret = self.refine_pose_using_features(features_q, scales_q, qcamera, T_query_initial, 
                                              ref_camera, T_render, features_ref, scales_ref, 
                                              p3d=points_3d_ecef)

        # Roll the temporal feature buffer (uses the frame two steps back).
        if self.use_temporal:
            if self.last_feature_query is not None:
                self.last_last_feature_query = self.last_feature_query
            self.last_feature_query = [f.detach().clone() for f in features_q]

        if not ret['success']:
            logger.info(f"Optimization failed for query {qname}")
            return ret

        # Best pose selection from candidates
        candidates_w2c = ret['T_opt']
        overall_loss = ret['overall_loss']
        fail_list = ret['fail_list']
        
        # Candidate validation based on thresholds
        flip_yz = refine_conf.get('coordinate_system', 'ecef') != 'normalized'
        candidates_c2w = build_world_c2w_batch(candidates_w2c, dd, scale_mul, origin, flip_yz=flip_yz)
        
        # Find best index among non-failed candidates
        valid = ~fail_list
        if not any(valid):
            logger.warning("No valid poses found after optimization.")
            return {**ret, 'success': False}

        best_idx_in_valid = torch.argmin(overall_loss[valid])
        final_idx = torch.nonzero(valid)[best_idx_in_valid].item()
        
        T_refined = candidates_w2c[final_idx]
        rot_err, trans_err = (T_query_initial[final_idx].inv() @ T_refined).magnitude()
        
        # Result conversion to world frame
        t_c2w_final = candidates_c2w[final_idx].cpu().numpy()
        if refine_conf.get("coordinate_system") == "normalized":
            # build_world already recovers COLMAP c2w (preprocess flip is cancelled)
            euler, trans = c2w_colmap_to_euler_trans(t_c2w_final)
        else:
            euler, trans, _, _ = pixloc_to_osg(t_c2w_final)

        return {
            **ret,
            'T_refined': T_refined,
            'diff_R': rot_err.item(),
            'diff_t': trans_err.item(),
            'euler_angles': euler,
            'translation': trans,
        }

    def refine_pose_using_features(self, features_q, scales_q, qcamera, T_init, 
                                   rcamera, T_render, features_ref, scales_ref, p3d=None) -> Dict:
        """Iterative optimization across multiple feature scales."""
        features_q = [f.to(self.device) for f in features_q]
        features_ref = [f.to(self.device) for f in features_ref]

        # Previous query frame (two steps back) as the temporal reference.
        features_last_query = None
        weights_last_query = None
        if (self.use_temporal
                and self.last_last_feature_query is not None
                and self.conf.compute_uncertainty
                and self.conf.normalize_descriptors):
            last_fq = [f.to(self.device) for f in self.last_last_feature_query]
            weights_last_query = [f[-1:] for f in last_fq]
            features_last_query = [
                torch.nn.functional.normalize(f[:-1], dim=0) for f in last_fq
            ]

        weights_q, weights_ref = None, None
        if self.conf.compute_uncertainty:
            weights_q = [f[-1:] for f in features_q]
            features_q = [f[:-1] for f in features_q]
            weights_ref = [f[-1:] for f in features_ref]
            features_ref = [f[:-1] for f in features_ref]

        if self.conf.normalize_descriptors:
            features_q = [torch.nn.functional.normalize(f, dim=0) for f in features_q]

        T_curr = T_init
        T_kf = T_init[0]
        
        for idx, level in enumerate(reversed(range(len(features_q)))):
            f_q, f_ref = features_q[level], features_ref[level]
            qcam_lvl = qcamera.scale(scales_q[level])
            rcam_lvl = rcamera.scale(scales_ref[level])
            w_ref_q = (weights_ref[level], weights_q[level]) if self.conf.compute_uncertainty else None

            # Handle multi-level optimizer list
            opt = self.optimizer
            if isinstance(opt, (list, tuple)):
                opt = opt[self.conf.layer_indices[level]] if self.conf.layer_indices else opt[level]

            # Apply the temporal term only at the coarsest level, where the
            # full candidate set is still alive and gets filtered.
            last_F, last_c = None, None
            if features_last_query is not None and idx == 0:
                last_F = features_last_query[level]
                last_c = weights_last_query[level]

            n_iters = {0: 4, 1: 3, 2: 2}.get(level, 2)
            
            T_opt, fail, loss = opt.run(p3d, f_ref, f_q, T_curr.to(f_q), 
                                        qcam_lvl.to_tensor().to(f_q),
                                        T_render.to(f_q),
                                        rcam_lvl.to_tensor().to(f_q),
                                        W_ref_query=w_ref_q,
                                        last_F_query=last_F,
                                        last_c_query=last_c,
                                        prior=self.prior,
                                        T_kf=T_kf.to(f_q),
                                        num_iters=n_iters)            
            if fail.all():
                return {'success': False, 'T_init': T_init}
            T_curr = T_opt  

        return {'success': True, 'T_opt': T_curr, 'overall_loss': loss, 'fail_list': fail}

    def dense_feature_extraction(self, image: np.ndarray, scale: int = 1) -> Tuple[List[torch.Tensor], List[int]]:
        """Extract multi-scale dense features."""
        feats, scales, weights = self.feature_extractor(image, scale) 
        if self.conf.compute_uncertainty:
            feats = [torch.cat([f, w], 0) for f, w in zip(feats, weights)]
        if self.conf.layer_indices is not None:
            feats = [feats[i] for i in self.conf.layer_indices]
            scales = [scales[i] for i in self.conf.layer_indices]
        return feats, scales

    def interp_sparse_observations(self, feature_maps, feature_scales, image_id, 
                                   p3dids=None, T_render=None, p3d=None, 
                                   camera_render=None, points2d=None) -> Union[Dict, Tuple]:
        """Interpolates sparse descriptors from dense feature maps."""
        if p3dids is not None:
            # Logic for database-based lookup
            image = self.model3d.dbs[image_id]
            camera = Camera.from_colmap(self.model3d.cameras[image.camera_id])
            T_w2cam = Pose.from_colmap(image)
            p3d = np.array([self.model3d.points3D[p].xyz for p in p3dids])
            p3d_cam = T_w2cam * p3d
            
            feature_obs, masks = [], []
            for i, (feats, sc) in enumerate(zip(feature_maps, feature_scales)):
                p2d_feat, valid = camera.scale(sc).world2image(p3d_cam)
                opt = self.optimizer
                opt = opt[len(opt)-i-1] if isinstance(opt, (list, tuple)) else opt
                obs, mask, _ = opt.interpolator(feats, p2d_feat.to(feats))
                feature_obs.append(obs)
                masks.append(mask & valid.to(mask))

            combined_mask = torch.all(torch.stack(masks, 0), 0)
            feature_obs = [[feature_obs[i][j] for i in range(len(feature_maps))] for j in range(len(p3dids))]
            return {p: feature_obs[i] for i, p in enumerate(p3dids) if combined_mask[i]}
        else:
            # Logic for direct 3D point rendering
            p3dids = list(range(len(p3d)))
            p3d_cam = T_render * p3d
            feature_obs, masks = [], []
            for i, (feats, sc) in enumerate(zip(feature_maps, feature_scales)):
                p2d_feat, valid = camera_render.scale(sc).world2image(p3d_cam)
                opt = self.optimizer
                opt = opt[len(opt)-i-1] if isinstance(opt, (list, tuple)) else opt
                obs, mask, _ = opt.interpolator(feats, p2d_feat.to(feats))
                feature_obs.append(obs)
                masks.append(mask & valid.to(mask))

            combined_mask = torch.all(torch.stack(masks, 0), 0)
            feature_obs = [[feature_obs[i][j] for i in range(len(feature_maps))] for j in range(len(p3dids))]
            return {p: feature_obs[i] for i, p in enumerate(p3dids) if combined_mask[i]}, p3dids

    def aggregate_features(self, p3did_to_dbids, dbid_p3did_to_feats) -> Dict:
        """Average features from multiple views for a given 3D point."""
        aggregated = defaultdict(list)
        for p3id, dbids in p3did_to_dbids.items():
            feats = [dbid_p3did_to_feats[dbid][p3id] for dbid in dbids if p3id in dbid_p3did_to_feats[dbid]]
            if feats:
                for lvl in range(len(feats[0])):
                    obs = [f[lvl] for f in feats]
                    if self.conf.average_observations:
                        obs_tensor = torch.stack(obs, 0)
                        if self.conf.compute_uncertainty:
                            f_val, w = obs_tensor[:, :-1], obs_tensor[:, -1:]
                            avg_f = (f_val * w).sum(0) / w.sum(0)
                            obs = torch.cat([avg_f, w.mean(0)], -1)
                        else:
                            obs = obs_tensor.mean(0)
                    aggregated[p3id].append(obs)
        return dict(aggregated)