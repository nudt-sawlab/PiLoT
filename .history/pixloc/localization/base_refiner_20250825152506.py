import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union
from omegaconf import DictConfig, OmegaConf as oc
from scipy.spatial.transform import Rotation as R
import os 
import numpy as np
import torch
import time
import copy
import cv2
from .feature_extractor import FeatureExtractor
from .tracker import BaseTracker
from ..pixlib.geometry import Pose, Camera
from ..pixlib.datasets.view import read_image
from ..utils.data import Paths
from ..utils.osg import osg_render
from ..utils.transform import kf_predictor, pixloc_to_osg, orthogonalize_rotation_matrix, move_inputs_to_cuda, visualize_points_on_images
from ..utils.get_depth import pad_to_multiple, zero_pad
logger = logging.getLogger(__name__)
def orthogonalize_batch(R):
    # R = U @ S @ Vᵀ, 正交化只要 U @ Vᵀ 即可
    U, S, Vh = torch.linalg.svd(R)        # all on GPU
    return U @ Vh
def build_c2w_batch(T_batch, dd, mul, origin):
    """
    把一批 Pose（从世界到相机的 w2c 旋转和平移）转换成相机到世界的 c2w 坐标系下的 [B,4,4] 张量。
    - T_batch.R: Tensor[B,3,3]
    - T_batch.t: Tensor[B,3]
    - dd:       Tensor[3] 或 None
    - mul:      float 或 Tensor 标量
    - origin:   Tensor[3] 偏移量
    返回: Tensor[B,4,4]
    """
    # 1) Device & dtype
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.float64

    # 2) 准备输入
    R_in = T_batch.R.to(device=device, dtype=dtype)  # [B,3,3]
    t_in = T_batch.t.to(device=device, dtype=dtype)  # [B,3]
    if dd is not None:
        dd = dd.to(device=device, dtype=dtype)       # [3]

    mul    = torch.as_tensor(mul,    device=device, dtype=dtype)
    origin    = torch.as_tensor(origin,    device=device, dtype=dtype)
    # origin = origin.to(device=device, dtype=dtype)  # [3]

    B = R_in.shape[0]

    # 3) 批量正交化：SVD → U @ Vh
    U, S, Vh = torch.linalg.svd(R_in)
    R = U @ Vh                                        # [B,3,3] 仍是 w2c

    # 4) 调整平移（扣掉 dd）
    t = t_in
    if dd is not None:
        # dd 视作 [3]，自动广播到 [B,3]
        t = t - (R @ dd)

    # 5) 转置成 c2w
    R_c2w = R.transpose(-1, -2)                       # [B,3,3]

    # 6) 构建 [B,4,4] 单位矩阵
    T = torch.eye(4, device=device, dtype=dtype) \
             .unsqueeze(0).repeat(B,1,1)              # [B,4,4]
    T[:, :3, :3] = R_c2w

    # 7) 填平移、缩放、翻轴、加 origin
    #   t_unsq: [B,3,1]  => R_c2w @ t_unsq => [B,3,1]
    tr = (-R_c2w @ t.unsqueeze(-1)).squeeze(-1) / mul  # [B,3]
    T[:, :3, 3] = tr

    # 翻转 Y/Z 轴
    T[:, :3, 1:3] *= -1

    # 加上 origin 偏移
    T[:, :3, 3] += origin

    return T


def build_prior_batch(T_render, dd, mul, origin):
    """
    把单帧渲染 Pose 转为 ECEF c2w，然后复制 B 份：
    - T_render.R: Tensor[3,3]
    - T_render.t: Tensor[3]
    - dd, mul, origin 同上
    - B: 需要复制的 batch 大小
    返回: Tensor[B,4,4]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.float64

    # 上面函数里同样的输入强转
    R = T_render.R.to(device=device, dtype=dtype)
    t = T_render.t.to(device=device, dtype=dtype)
    if dd is not None:
        dd = dd.to(device=device, dtype=dtype)
        t = t - R @ dd

    mul    = torch.as_tensor(mul,    device=device, dtype=dtype)
    origin    = torch.as_tensor(origin,    device=device, dtype=dtype)

    # 转置成 c2w
    R_c2w = R.transpose(-1, -2)  # [3,3]

    # 计算平移
    tr = (-R_c2w @ t.unsqueeze(-1)).squeeze(-1) / mul  # [3]
    tr = tr + origin

    # 组装单个 4×4
    T_single = torch.eye(4, device=device, dtype=dtype)
    T_single[:3, :3] = R_c2w
    T_single[:3, 1:3] *= -1
    T_single[:3,  3] = tr

    # 复制 B 份
    return T_single        # [B,4,4]
class BaseRefiner:
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
                 optimizer: torch.nn.Module,
                 feature_extractor: FeatureExtractor,
                 conf: Union[DictConfig, Dict],
                 ):
        self.device = device
        self.optimizer = optimizer
        self.feature_extractor = feature_extractor
        self.last_feature_query = None
        self.last_last_feature_query = None
        self.prior = False
        self.conf = oc.merge(
            oc.create(self.base_default_config),
            oc.create(self.default_config),
            oc.create(conf))

    def log_dense(self, **kwargs):
        if self.tracker is not None:
            self.tracker.log_dense(**kwargs)

    def log_optim(self, **kwargs):
        if self.tracker is not None:
            self.tracker.log_optim_done(**kwargs)

    def refine(self, **kwargs):
        ''' Implement this in the child class'''
        raise NotImplementedError
    
    def refine_query_pose(self, qname: str, qcamera: Camera, ref_camera: Camera, render_frame, T_query_initial: Pose, T_render: Pose, Points_3D_ECEF, dd = None,
                          multiscales: Optional[List[int]] = None, gt_pose_dict=None, last_frame_info = {}, query_resize_ratio = 1, image_query=None) -> Dict:
        (qcamera, ref_camera, render_frame, T_query_initial, T_render, Points_3D_ECEF, dd, last_frame_info) = \
        move_inputs_to_cuda(qcamera, ref_camera, render_frame, T_query_initial, T_render, Points_3D_ECEF, dd, last_frame_info)
        refine_conf = last_frame_info['refine_conf']
        if 'use_kf' in refine_conf.keys():
            use_kf = refine_conf['use_kf']
        else:
            use_kf = False
        if 'origin' in refine_conf.keys():
            self.origin = np.array(refine_conf['origin'])
        if 'mul' in refine_conf.keys():
            mul = refine_conf['mul']
        R_thes = refine_conf['R_thes']
        dis_thes = refine_conf['dis_thes']
        # Intrinsics 
        query_weight_px, query_height_px = qcamera.size
        ref_weight_px, ref_height_px = ref_camera.size
        qcamera_modified = copy.deepcopy(qcamera)
        # qcamera_modified.c[1] = query_height_px - qcamera_modified.c[1]
        ref_camera_modified = copy.deepcopy(ref_camera)
        # ref_camera_modified.c[1] = ref_height_px - ref_camera_modified.c[1]
        # start_time = time.time()
        if image_query is None:
            image_query = read_image(qname, scale = query_resize_ratio)

        image_query = zero_pad(int(query_weight_px.item()), image_query)
        render_frame = zero_pad(int(query_weight_px.item()), render_frame)

        T_query_initial_poses = copy.deepcopy(T_query_initial) #w2c

        # end_time4 = time.time()
        # print('预处理 ：', end_time4 - start_time)
        # start_time = time.time()
        # Feature extraction
        features_ref_dense, scales_ref = self.dense_feature_extraction(render_frame)
        features_query, scales_query = self.dense_feature_extraction(image_query)
        
        # Refine pose
        # T_gt = gt_pose_dict[qname.split('/')[-1]]['T_w2c']
        # end_time3 = time.time()
        # print('提特征 ：', end_time3 - start_time)
        
        start_time = time.time()
        ret = self.refine_pose_using_features(features_query, scales_query,
                                            qcamera_modified, T_query_initial_poses, ref_camera_modified, T_render,
                                            features_ref_dense, scales_ref, p3d = Points_3D_ECEF)
        # if self.last_feature_query is not None:
        #     self.last_last_feature_query = copy.deepcopy(self.last_feature_query)
        # self.last_feature_query = copy.deepcopy(features_query)
        end_time1 = time.time()
        # print('refine pose costs: ', end_time1 - start_time)
        # start_time = time.time()
        # render estimate pose
        if not ret['success']:
            logger.info(f"Optimization failed for query {qname}")
        else:
            T_query_initial_poses = ret['T_opt']
        
        # print("one query costs: ", end_time - start_time_total)
        if ret['success']:
            overall_loss = ret['overall_loss']
            fail_list = ret['fail_list']
            T_candidtas = ret['T_opt']
            # ------
            T_opt_c2w = build_c2w_batch(T_candidtas, dd, mul, self.origin)
            B = T_candidtas.shape[0]
            T_render_in_ECEF_c2w = build_prior_batch(T_render, dd, mul, self.origin)

            T_prior_ECEF = T_render_in_ECEF_c2w.unsqueeze(0).repeat(B, 1, 1)  # [B,4,4]
            T_prior_ECEF_Pose = Pose.from_Rt(T_prior_ECEF[:, :3, :3], T_prior_ECEF[:, :3, 3])
            T_opt_c2w_Pose = Pose.from_Rt(T_opt_c2w[:, :3, :3], T_opt_c2w[:, :3, 3])
            dR, dt = (T_prior_ECEF_Pose.inv() @ T_opt_c2w_Pose).magnitude()
            
            t_indices = dt <= dis_thes #!
            R_indices = dR <= R_thes

            # 剔除旋转变化量和平移变化量过大的候选
            valid = (~fail_list) & t_indices & R_indices
            valid_loss = overall_loss[valid]

            if not any(valid):
                print('attention')
                import ipdb; ipdb.set_trace()
            min_index_in_valid = torch.argmin(valid_loss)

            # ------
            pose_index = torch.nonzero(valid)[min_index_in_valid].item()
            T_refined = ret['T_opt'][pose_index]
            dR, dt = (T_query_initial_poses[pose_index].inv() @ T_refined).magnitude()
            ret = {
                **ret,
                'T_refined': T_refined,
                'diff_R': dR.item(),
                'diff_t': dt.item(),
            }
            # choose the best estimate
            if T_opt_c2w.is_cuda:
                T_opt_c2w = T_opt_c2w.cpu()
            T_opt_c2w = T_opt_c2w.numpy()
            
            #===============test
            # num = torch.nonzero(valid)
            # # euler_gt = gt_pose_dict[qname.split('/')[-1]]['euler']
            # # trans_gt = gt_pose_dict[qname.split('/')[-1]]['trans']
            # # for ii in num:
            # #     T_opt_c2w_ = T_opt_c2w[ii]
            # #     euler_angles_refined, translation_refined, T_ECEF_estimated, kf_current_frame_es_pose = pixloc_to_osg(T_opt_c2w_)
            # #     print('i: ',ii, euler_angles_refined - euler_gt, overall_loss[ii])
                
            # p3d_r = T_render.transform(Points_3D_ECEF)  # [B, num_init_pose, N, 3]
            # p2d_r, visible_r = ref_camera_modified.world2image(p3d_r)
            
            # B = T_candidtas.shape[0]  # 应该是 4
            # N = Points_3D_ECEF.shape[0]  # 500
            # Points_3D_ECEF = Points_3D_ECEF.unsqueeze(0).expand(B, -1, -1)  # [4, 500, 3]
            # p3d_q = T_candidtas.transform(Points_3D_ECEF)  # [4, 500, 3]
            # p2d_q, visible_q = qcamera_modified.world2image(p3d_q)  # [B, num_init_pose, N, 2], [B, num_init_pose, N]
            # for ii in num:    
            # visualize_points_on_images(render_frame, image_query, p2d_r, p2d_q[pose_index])   
            
            # if 'euler_angles' in last_frame_info:
            #     print(euler_angles_refined - last_frame_info['euler_angles'])
            #===============test
            T_opt_c2w_ = T_opt_c2w[pose_index]
            # print('pose_index:',pose_index)
            euler_angles_refined, translation_refined, T_ECEF_estimated, kf_current_frame_es_pose = pixloc_to_osg(T_opt_c2w_)

            ret['euler_angles'] = euler_angles_refined
            ret['translation'] = translation_refined

            # print("After refined: ", euler_angles_refined, translation_refined)
            # end_time2 = time.time()
            # print('后处理： ', end_time2 - start_time)
            # start_time = time.time()
            # Kalman Filter
            use_kf = False
            
            if use_kf:
                
                if 'observations' in last_frame_info.keys():
                    observations = last_frame_info['observations']
                else:
                    observations = []
                frame_nums = 5
                if len(observations) < frame_nums:
                    observations.append(kf_current_frame_es_pose)
                    ret["observations"] = observations
                else:
                    start_time = time.time()
                    observations = np.vstack([observations, kf_current_frame_es_pose])
                    observations = observations[1:, :]
                    observations, kf_euler, kf_trans = kf_predictor(observations)
                    self.prior = False
                    ret["observations"] = observations  # [lon, lat, alt, pitch, roll, yaw]
                    # ret['euler_angles'] = kf_euler
                    # ret['translation'] = kf_trans
                    last_frame_info['euler_angles'] = kf_euler
                    last_frame_info['translation'] = kf_trans
                    # ret['origin'] = new_origin
                    print("After kf: ", kf_euler, kf_trans)
                    end_time = time.time()
                    print("kf time: ", end_time - start_time)
                    print('hh')
            
            # img1 = color_image
            # img2 = image_query
            # img3 = color_image_refined
            # save_path = os.path.join(self.outputs, 'ays_'+rname)
            # visualize_image_alignment(img1, img2, img3, title1="Reference Image", title2="Query Image", save_path=save_path)
        return ret
    def refine_pose_using_features(self,
                                   features_query: List[torch.tensor],
                                   scales_query: List[float],
                                   qcamera: Camera,
                                   T_initial_pose_candidates,
                                   rcamera,
                                   T_render,
                                   features_ref: List[List[torch.Tensor]],
                                   scales_ref,
                                   p3d = None,
                                   T_gt = None
                                   ) -> Dict:
        """Perform the pose refinement using given dense query feature-map.
        """
        # query dense features decomposition and normalization
        # start_time = time.time()
        features_last_query = None
        features_query = [feat.to(self.device) for feat in features_query]
        features_ref = [feat.to(self.device) for feat in features_ref]
        if self.last_last_feature_query is not None:
            last_features_query = [feat.to(self.device) for feat in self.last_last_feature_query]
            weights_last_query = [feat[-1:] for feat in last_features_query]
            features_last_query = [feat[:-1] for feat in last_features_query]
            # features_last_query = [torch.nn.functional.normalize(feat, dim=0)
            #                   for feat in features_last_query]
        if self.conf.compute_uncertainty:
            weights_query = [feat[-1:] for feat in features_query]
            features_query = [feat[:-1] for feat in features_query]

            weights_ref = [feat[-1:] for feat in features_ref]
            features_ref = [feat[:-1] for feat in features_ref]
            
            
        if self.conf.normalize_descriptors:
            features_query = [torch.nn.functional.normalize(feat, dim=0)
                              for feat in features_query]
            # features_ref = [torch.nn.functional.normalize(feat, dim=0)
            #                   for feat in features_ref]
        # p3d = p3d[list(p3dids), :]

        T_i = T_initial_pose_candidates
        ret = {'T_init': T_initial_pose_candidates}
        # We will start with the low res feature map first
        # end_time1 = time.time()
        # print('LM-预处理： ', end_time1 - start_time)
        T_kf = T_initial_pose_candidates[0]
        for idx, level in enumerate(reversed(range(len(features_query)))):
        # for idx, level in enumerate(range(len(features_query))):
            start_time = time.time()
            F_q, F_ref = features_query[level], features_ref[level]
            qcamera_feat = qcamera.scale(scales_query[level])
            rcamera_feat = rcamera.scale(scales_ref[level])

            if self.conf.compute_uncertainty:
                W_ref_query = (weights_ref[level], weights_query[level])
            else:
                W_ref_query = None

            logger.debug(f'Optimizing at level {level}.')
            opt = self.optimizer
            if isinstance(opt, (tuple, list)):
                if self.conf.layer_indices:
                    opt = opt[self.conf.layer_indices[level]]
                else:
                    opt = opt[level]
            if features_last_query is not None:
                last_F_query = features_last_query[level]
                last_c_query = weights_last_query[level]
            else:
                last_F_query = None
                last_c_query = None
            T_opt, fail, overall_loss = opt.run(p3d, F_ref, F_q, T_i.to(F_q), 
                                qcamera_feat.to_tensor().to(F_q),
                                T_render.to(F_q),
                                rcamera_feat.to_tensor().to(F_q),
                                W_ref_query=W_ref_query,
                                last_F_query = last_F_query,
                                last_c_query = last_c_query,
                                prior = self.prior,
                                T_kf = T_kf.to(F_q)
                                )            
            # vis===================
            _, topk_indices = torch.topk(-overall_loss, 1, dim=-1, largest=True, sorted=True)
            T_query = T_opt[topk_indices]
            p3d_cam = T_query * p3d
            p2d_query_refined, valid = qcamera_feat.world2image(p3d_cam)
            
            T_query = T_i[topk_indices]
            p3d_cam = T_query * p3d
            p2d_query, valid = qcamera_feat.world2image(p3d_cam)
            
            
            p3d_cam = T_render * p3d
            p2d_render, valid = rcamera_feat.world2image(p3d_cam)
            
            inputs = {
                "p2d_render": p2d_render,
                "f_r": F_ref,
                "w_r":weights_ref[level],
                "p2d_query": p2d_query,
                "p2d_query_refined": p2d_query_refined,
                "f_q": F_q,
                "w_q":weights_query[level],
                "p3D": p3d,
            }
            name = str(idx)+'_feature.pt'
            torch.save(inputs, "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/feature_vis_outputs/DJI_20250612174308_0001_V"+name)
            
            
            
            

            # vis =====================
            
            
            if fail.all().item():
                return {**ret, 'success': False}

            T_i = T_opt  #!?
            # end_time2 = time.time()
            # print('LM-计算： ', end_time2 - start_time, idx)
        return {
            'success': True,
            'T_opt': T_opt,
            'overall_loss': overall_loss,
            'fail_list': fail
        }

  
    def dense_feature_extraction(self, image: np.array, image_scale: int = 1) -> Tuple[List[torch.Tensor], List[int]]:
        features, scales, weight = self.feature_extractor(
                image, image_scale) 
        if self.conf.compute_uncertainty:
            assert weight is not None
            # stack them into a single tensor (makes the bookkeeping easier)
            features = [torch.cat([f, w], 0) for f, w in zip(features, weight)]

        # Filter out some layers or keep them all
        if self.conf.layer_indices is not None:
            features = [features[i] for i in self.conf.layer_indices]
            scales = [scales[i] for i in self.conf.layer_indices]

        return features, scales

    def interp_sparse_observations(self,
                                   feature_maps: List[torch.Tensor],
                                   feature_scales: List[float],
                                   image_id: float,
                                   p3dids: List[int] = None,
                                   T_render = None,
                                   p3d = None,
                                   camera_render = None,
                                   points2d = None
                                   ) -> Dict[int, torch.Tensor]:
        if p3dids is not None:
            image = self.model3d.dbs[image_id]
            camera = Camera.from_colmap(self.model3d.cameras[image.camera_id])
            T_w2cam = Pose.from_colmap(image)
            p3d = np.array([self.model3d.points3D[p3did].xyz for p3did in p3dids])
            p3d_cam = T_w2cam * p3d
    
            # interpolate sparse descriptors and store
            feature_obs = []
            masks = []
            for i, (feats, sc) in enumerate(zip(feature_maps, feature_scales)):
                p2d_feat, valid = camera.scale(sc).world2image(p3d_cam)
                
                opt = self.optimizer
                opt = opt[len(opt)-i-1] if isinstance(opt, (tuple, list)) else opt
                obs, mask, _ = opt.interpolator(feats, p2d_feat.to(feats))
                assert not obs.requires_grad
                feature_obs.append(obs)
                masks.append(mask & valid.to(mask))

            mask = torch.all(torch.stack(masks, dim=0), dim=0)

            # We can't stack features because they have different # of channels
            feature_obs = [[feature_obs[i][j] for i in range(len(feature_maps))]
                        for j in range(len(p3dids))]  # N x K x D

            feature_dict = {p3id: feature_obs[i]
                            for i, p3id in enumerate(p3dids) if mask[i]}
            return feature_dict
        else:
            p3dids = list(range(len(p3d)))
            # image = self.model3d.dbs[image_id]
            camera = camera_render
            T_w2cam = T_render

            p3d_cam = T_w2cam * p3d

            # interpolate sparse descriptors and store
            feature_obs = []
            masks = []
            for i, (feats, sc) in enumerate(zip(feature_maps, feature_scales)):
                p2d_feat, valid = camera.scale(sc).world2image(p3d_cam)

                if points2d is not None:
                    false_indices = torch.nonzero(~valid)
                    print(p2d_feat[false_indices], points2d[false_indices])
                opt = self.optimizer
                opt = opt[len(opt)-i-1] if isinstance(opt, (tuple, list)) else opt
                
                obs, mask, _ = opt.interpolator(feats, p2d_feat.to(feats))
                assert not obs.requires_grad
                feature_obs.append(obs)
                masks.append(mask & valid.to(mask))

            mask = torch.all(torch.stack(masks, dim=0), dim=0)

            # We can't stack features because they have different # of channels
            feature_obs = [[feature_obs[i][j] for i in range(len(feature_maps))]
                        for j in range(len(p3dids))]  # N x K x D

            feature_dict = {p3id: feature_obs[i]
                            for i, p3id in enumerate(p3dids) if mask[i]}
            return feature_dict, p3dids
    def interp_sparse_observations_dev(self,
                                   feature_maps: List[torch.Tensor],
                                   feature_scales: List[float],
                                   image_id: float,
                                   p3dids: List[int],
                                   ) -> Dict[int, torch.Tensor]:
        image = self.model3d.dbs[image_id]
        camera = Camera.from_colmap(self.model3d.cameras[image.camera_id])
        T_w2cam = Pose.from_colmap(image)
        p3d = np.array([self.model3d.points3D[p3did].xyz for p3did in p3dids])
        p3d_cam = T_w2cam * p3d

        # interpolate sparse descriptors and store
        feature_obs = []
        masks = []
        for i, (feats, sc) in enumerate(zip(feature_maps, feature_scales)):
            p2d_feat, valid = camera.scale(sc).world2image(p3d_cam)
            opt = self.optimizer
            opt = opt[len(opt)-i-1] if isinstance(opt, (tuple, list)) else opt
            obs, mask, _ = opt.interpolator(feats, p2d_feat.to(feats))
            assert not obs.requires_grad
            feature_obs.append(obs)
            masks.append(mask & valid.to(mask))

        mask = torch.all(torch.stack(masks, dim=0), dim=0)

        # We can't stack features because they have different # of channels
        feature_obs = [[feature_obs[i][j] for i in range(len(feature_maps))]
                       for j in range(len(p3dids))]  # N x K x D

        feature_dict = {p3id: feature_obs[i]
                        for i, p3id in enumerate(p3dids) if mask[i]}

        return feature_dict

    def aggregate_features(self,
                           p3did_to_dbids: Dict,
                           dbid_p3did_to_feats: Dict,
                           ) -> Dict[int, List[torch.Tensor]]:
        """Aggregate descriptors from covisible images through averaging.
        """
        p3did_to_feat = defaultdict(list)
        for p3id, obs_dbids in p3did_to_dbids.items():
            features = []
            for obs_imgid in obs_dbids:
                if p3id not in dbid_p3did_to_feats[obs_imgid]:
                    continue
                features.append(dbid_p3did_to_feats[obs_imgid][p3id])
            if len(features) > 0:
                # list with one entry per layer, grouping all 3D observations
                for level in range(len(features[0])):
                    observation = [f[level] for f in features]
                    if self.conf.average_observations:
                        observation = torch.stack(observation, 0)
                        if self.conf.compute_uncertainty:
                            feat, w = observation[:, :-1], observation[:, -1:]
                            feat = (feat * w).sum(0) / w.sum(0)
                            observation = torch.cat([feat, w.mean(0)], -1)
                        else:
                            observation = observation.mean(0)
                    p3did_to_feat[p3id].append(observation)
        return dict(p3did_to_feat)
