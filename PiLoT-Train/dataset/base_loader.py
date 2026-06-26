import json
import os
from pathlib import Path
from .base_dataset import BaseDataset, set_seed
import torch
import numpy as np
from ..utils.geometry.wrappers import Pose, Camera
from ..utils.utils import generate_random_aa_and_t

from .utils import read_image, resize,numpy_image_to_torch, crop, zero_pad, get_imgaug_seq
import logging
import imgaug as ia

from .Aero_seq import _Dataset_Aero

logger = logging.getLogger(__name__)

class PASTA:
    """
    PASTA: Proportional Amplitude Spectrum Augmentation for Synthetic-to-Real Domain Generalization

    ...

    Attributes
    ----------
    alpha : float
        coefficient of linear term to ensure perturbation strength increases 
        with increasing spatial frequency
    beta : float
        constant perturbation across all frequencies
    k : int
        exponent ensuring non-linear dependence of perturbation on spatial
        frequency

    """
    def __init__(self, alpha: float, beta: float, k: int):
        self.alpha = alpha
        self.beta = beta
        self.k = k
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        fft_src = torch.fft.fftn(img, dim=[-2, -1])
        amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)

        X, Y = amp_src.shape[1:]
        X_range, Y_range = None, None

        if X % 2 == 1:
            X_range = np.arange(-1 * (X // 2), (X // 2) + 1)
        else:
            X_range = np.concatenate(
                [np.arange(-1 * (X // 2) + 1, 1), np.arange(0, X // 2)]
            )

        if Y % 2 == 1: 
            Y_range = np.arange(-1 * (Y // 2), (Y // 2) + 1)
        else:
            Y_range = np.concatenate(
                [np.arange(-1 * (Y // 2) + 1, 1), np.arange(0, Y // 2)]
            )

        XX, YY = np.meshgrid(Y_range, X_range)

        exp = self.k
        lin = self.alpha
        offset = self.beta

        inv = np.sqrt(np.square(XX) + np.square(YY))
        inv *= (1 / inv.max()) * lin
        inv = np.power(inv, exp)
        inv = np.tile(inv, (3, 1, 1))
        inv += offset
        prop = np.fft.fftshift(inv, axes=[-2, -1])
        amp_src = amp_src * np.random.normal(np.ones(prop.shape), prop)

        aug_img = amp_src * torch.exp(1j * pha_src)
        aug_img = torch.fft.ifftn(aug_img, dim=[-2, -1])
        aug_img = torch.real(aug_img)
        aug_img = torch.clip(aug_img, 0, 1)
        return aug_img
def cv2_to_tensor(img_rgb):
    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0  # [0,1]
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)  # (C, H, W)
    return img_tensor

def tensor_to_cv2(img_tensor):
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    img_rgb = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    # img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) 
    return img_rgb
class BOP(BaseDataset):
    default_conf = {
        'dataset_dir': '',
        'background_image_dir': '',
        
        'prerender_dir_name': 'pre_render_with_vertex',
        'train_num_per_slice': 100,
        'val_num_per_slice': 50,
        'random_sample': True,

        'n_divide': 2,
        'with_mask': False,
        'with_depth': False,
        'with_ref': False,
        'with_ref_depth': False,
        'get_top_k_template_views': 1,
        'skip_template_view': 1,
        'geometry_unit_in_meter': 0.001,  # must equal to the geometry_unit_in_meter of preprocess
        'offset_angle_step': 5.0,
        'min_offset_angle': 5.0,  # 5.0,
        'max_offset_angle': 15.0,  # 25.0,
        'offset_translation_step': 0.01,
        'min_offset_translation': 0.005,  # 0.01,# 0.01,  # meter
        'max_offset_translation': 0.015,  # 0.025,# 0.03,  # meter
        'val_offset': True,
        'train_offset': True,
        'skip_frame': 1,

        'grayscale': False,
        'resize': None,
        'resize_by': 'max',
        'crop': False,
        'crop_border': None,
        'pad': None,
        'change_background': False,
        'change_background_thres': 0.5,
        'img_aug': False,
        'seed': 0,
        'sample_vertex_num': 500,

        'opt': False,
        'debug_check_display': False
    }

    strict_conf = False

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        if split == 'train' or split == 'val':
            return _Dataset(self.conf, split)
        elif split == 'test':
            # TODO: implement later
            return _Dataset(self.conf, split)
        else:
            raise NotImplementedError        


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        self.root = Path(conf.dataset_dir)
            
        self.conf, self.split = conf, split

        self.geometry_unit_in_meter = float(conf.geometry_unit_in_meter)
        self.min_offset_angle = float(conf.min_offset_angle)
        self.max_offset_angle = float(conf.max_offset_angle)
        self.min_offset_translation = float(conf.min_offset_translation)
        self.max_offset_translation = float(conf.max_offset_translation)
        self.dataset_list = []

        self.num_samples = conf.num_samples
        

        self.pasta = PASTA(alpha=3.0, beta=0.25, k=2)

        for sub_dataset_name in conf.sub_dataset_dir:
            if sub_dataset_name == 'Mapscape':
                self.dataset_list.append(_Dataset_Aero(self.conf, split))
            # else:
            #     raise NotImplementedError 
        # if split != 'test':
        self.sample_new_items(conf.seed)

    def sample_new_items(self, seed):
        from ..utils.utils import generate_spheric_cameras
        logger.info(f'Sampling new images with seed {seed}')
        set_seed(seed)

        self.items = []
        self.meshes = {}
        self.ref_items = {}
        # in aero class
        for dataset in self.dataset_list:
            dataset.sample_new_items(seed)  # item
            self.items.extend(dataset.items)   #__getitem__!
            if self.conf.with_ref:
                self.ref_items = {**self.ref_items, **dataset.ref_items}
        if self.conf.img_aug:
            ia.seed(seed)

    def image_aug(self, img):
        seq = get_imgaug_seq()
        img_aug = seq(image=img)
        return img_aug

    def read_image(self, image_path, conf, camera: Camera, depth_image_path = None, bbox2d=None, image=None, img_aug=False):
        # read image
        if image is None:
            img = read_image(image_path, conf.grayscale)
        else:
            img = image.copy()
        
        if conf.resize:
            scales = (1, 1)
            if isinstance(conf.resize, int):
                if conf.resize_by == 'max':
                    img, scales = resize(img, conf.resize, fn=max)
                elif (conf.resize_by == 'min' or (conf.resize_by == 'min_if' and min(*img.shape[:2]) < conf.resize)):
                    img, scales = resize(img, conf.resize, fn=min)
            elif len(conf.resize) == 2:
                img, scales = resize(img, list(conf.resize))

            if scales != (1, 1):
                camera = camera.scale(scales)
        if conf.crop:
            centroid = None
            img, camera, bbox = crop(
                img, conf.crop, random=(self.split == 'train'),
                camera=camera, return_bbox=True, centroid=centroid)
        if conf.pad:
            img = zero_pad(conf.pad, img)
            # import ipdb; ipdb.set_trace()
            img = img[0]
        if img_aug:
            img_aug = self.image_aug(img)
        else:
            img_aug = img
        img_aug = img_aug.astype(np.float32)
        # img = img.astype(np.float32)
        # import ipdb; ipdb.set_trace()
        return img, numpy_image_to_torch(img_aug), camera


    def depth2xyzmap(self, depth, camera, uvs=None):
        invalid_mask = (depth<0.1)
        H,W = depth.shape[:2]
        if uvs is None:
            vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
            vs = vs.reshape(-1)
            us = us.reshape(-1)
        else:
            uvs = uvs.round().astype(int)
            us = uvs[:,0]
            vs = uvs[:,1]
        zs = depth[vs,us]
        xs = (us-camera.c[0].item())*zs/camera.f[0].item()
        ys = (vs-camera.c[1].item())*zs/camera.f[1].item()
        pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
        xyz_map = np.zeros((H,W,3), dtype=np.float32)
        xyz_map[vs,us] = pts
        xyz_map[invalid_mask] = 0
        # draw_vertices_to_obj(xyz_map[~invalid_mask], 'src_open/pcd.obj')
        # import ipdb
        # ipdb.set_trace()
        return xyz_map

    def __getitem__(self, idx):
        # idx = 100
        frame_item = self.items[idx]

        depths = []
        xyz_maps = []

        image_path = frame_item['image_path']
        output_name = (frame_item['slice'] + '_' + frame_item['image_name']).replace('/', '_')
        ori_image = read_image(image_path) #, self.conf.grayscale
        
        img_tensor = cv2_to_tensor(ori_image)
        img_aug_tensor = self.pasta(img_tensor)
        ori_image = tensor_to_cv2(img_aug_tensor)
        # obj_id = frame_item['obj_id']
        # load pose
        view_R = torch.tensor(frame_item['image_pose_gt'][:3, :3])
        view_t = torch.tensor(frame_item['image_pose_gt'][:3, 3]) #gt
        gt_body2view_pose = Pose.from_Rt(view_R, view_t)  # w2c

        # initial_body2view_poses = frame_item['image_pose_initial']
        
        # initial_view_R = torch.tensor(frame_item['image_pose_initial'][:, :3, :3])
        # initial_view_t = torch.tensor(frame_item['image_pose_initial'][:, :3, 3])
        # initial_body2view_poses = Pose.from_Rt(initial_view_R, initial_view_t)

        # load intrinsic
        K = torch.tensor(frame_item['image_intrinscis'])
        intrinsic_param = torch.tensor([ori_image.shape[1], ori_image.shape[0],
                                    K[2], K[3], K[4], K[5]], dtype=torch.float32)
        ori_camera = Camera(intrinsic_param)

        diameter = None
        image, aug_image, camera = self.read_image(None, self.conf, ori_camera, image = ori_image,
                                                    img_aug=self.conf.img_aug if self.split == 'train' else False)
        
        Points3D_path = frame_item['Points3D_path']
        origin = frame_item['origin']
        mul = frame_item['mul']
        Points_3D_ECEF = np.load(Points3D_path)
        Points_3D_ECEF_origin_total = Points_3D_ECEF - np.tile(origin, (len(Points_3D_ECEF), 1))   
        indices = np.random.randint(0, len(Points_3D_ECEF_origin_total), size=self.num_samples)
        Points_3D_ECEF_origin_total = Points_3D_ECEF_origin_total[indices]
        points3D_total = torch.from_numpy(Points_3D_ECEF_origin_total * mul).float()
        # points3D_total = torch.from_numpy(frame_item['Points_3D_ECEF_origin_total']).float()
        
        points_max = points3D_total.max(dim=0)[0]
        points_min = points3D_total.min(dim=0)[0]
        points_size = points_max - points_min
        dd = points_min + points_size / 2
        p3ds = points3D_total - dd
        # scale = 1 / points_size.max()
        # p3ds = p3ds * scale
        tt = gt_body2view_pose.t + gt_body2view_pose.R @ dd
        # gt_body2view_pose = Pose.from_Rt(gt_body2view_pose.R, tt*scale)
        gt_body2view_pose = Pose.from_Rt(gt_body2view_pose.R, tt)
        points3D_total = p3ds

        # p2ds, valid = camera.view2image(gt_body2view_pose.transform(p3ds))
        # p2ds = p2ds[::50]
        # valid = valid[::50]
        # tmp_img = draw_centers_in_image(aug_image.permute(1, 2, 0).mul(255).byte().numpy().copy(), p2ds.numpy(), valid.numpy())
        # cv2.imwrite('test1.png', tmp_img[..., ::-1])
        # draw_vertices_to_obj(p3ds.numpy(), 'test.obj')


        ref_num = 1

        aug_image_ref_list = []
        camera_ref_list = []
        points3D_list = []
        body2view_pose_ref_list = []
        initial_body2view_poses_list = []
        for _ in range(ref_num):
            img_path = frame_item['ref_rgb']
            # load image
            ori_image_ref = read_image(img_path) #, self.conf.
            # img_tensor = cv2_to_tensor(ori_image)
            # img_aug_tensor = self.pasta(img_tensor)
            # ori_image = tensor_to_cv2(img_aug_tensor)

            # load intrinsics
            K_ref = frame_item['ref_intrinsics']
            intrinsic_param = torch.tensor([ori_image_ref.shape[1], ori_image_ref.shape[0],
                                            K_ref[2], K_ref[3], K_ref[4], K_ref[5]], dtype=torch.float32)
            ori_camera_ref = Camera(intrinsic_param)
            #load poses
            view_R = torch.tensor(frame_item['ref_poses'][:3, :3]).reshape(3, 3)
            view_t = torch.tensor(frame_item['ref_poses'][:3, 3])
            body2view_pose_ref = Pose.from_Rt(view_R, view_t)  # c2w
            body2view_pose_ref = Pose.from_Rt(body2view_pose_ref.R, body2view_pose_ref.t + body2view_pose_ref.R @ dd)

            random_aa, random_t = generate_random_aa_and_t(self.min_offset_angle, self.max_offset_angle,
                                                           self.min_offset_translation, self.max_offset_translation)
            random_pose = Pose.from_aa(random_aa, random_t)
            initial_body2view_poses = Pose.from_Rt(random_pose.R@gt_body2view_pose[None].R, gt_body2view_pose[None].t+random_pose.t[0])
            # random_R = random_pose.R @ gt_body2view_pose[None].R.inverse()
            # initial_body2view_poses = Pose.from_Rt(random_R.inverse(), gt_body2view_pose[None].t+random_pose.t[0])
            # initial_body2view_poses = gt_body2view_pose @ random_pose

            # padding and resize
            image_ref, aug_image_ref, camera_ref = \
                self.read_image(None, self.conf, ori_camera_ref,  image = ori_image_ref,
                                img_aug=self.conf.img_aug if self.split == 'train' else False)

            # p2ds, valid = camera.view2image(initial_body2view_poses[0].transform(points3D_total))
            # p2ds = p2ds[::50]
            # valid = valid[::50]
            # tmp_img = draw_centers_in_image(aug_image.permute(1, 2, 0).mul(255).byte().numpy().copy(), p2ds.numpy(), valid.numpy())
            # p2ds, valid = camera.view2image(gt_body2view_pose.transform(points3D_total))
            # p2ds = p2ds[::50]
            # valid = valid[::50]
            # tmp_img = draw_centers_in_image(tmp_img, p2ds.numpy(), valid.numpy(), center_color=(0, 255, 0))
            # cv2.imwrite('test.png', tmp_img[..., ::-1])

            # p2ds_ref, valid_ref = camera_ref.view2image(body2view_pose_ref.transform(points3D_total))
            # p2ds_ref = p2ds_ref[::50]
            # valid_ref = valid_ref[::50]
            # tmp_img = draw_centers_in_image(aug_image_ref.permute(1, 2, 0).mul(255).byte().numpy().copy(), p2ds_ref.numpy(), valid_ref.numpy())
            # cv2.imwrite('test_ref.png', tmp_img[..., ::-1])
            # import ipdb
            # ipdb.set_trace()

            # indices = np.random.randint(0, len(points3D_total), size=self.conf.num_samples)
            # points3D = points3D_total[indices].float()
            points3D = points3D_total.float()
            points3D_list.append(points3D)
            aug_image_ref_list.append(aug_image_ref)
            camera_ref_list.append(camera_ref.unsqueeze(0))
            body2view_pose_ref_list.append(body2view_pose_ref.unsqueeze(0))
            initial_body2view_poses_list.append(initial_body2view_poses)
        points3D_list = torch.stack(points3D_list)
        aug_image_ref_list = torch.stack(aug_image_ref_list)
        camera_ref_list = torch.stack(camera_ref_list)
        body2view_pose_ref_list = torch.stack(body2view_pose_ref_list)
        initial_body2view_poses_list = torch.stack(initial_body2view_poses_list)
        # initial_body2view_poses = initial_body2view_poses.repeat(points3D_list.shape[0], 1)#! TODO
        #---------vis
        # p3d_query = gt_body2view_pose.transform(points3D)
        # p2d, visible_2d = camera.view2image(p3d_query)
        
        # p3d_ref = body2view_pose_ref.transform(points3D)
        # p2d_ref, visible_2d_ref = camera_ref.view2image(p3d_ref)
        
        # for i in range(30):
        #     visualize_points_on_images(torch_image_to_numpy(aug_image), torch_image_to_numpy(aug_image_ref), 
        #                     [(p2d[i][0], p2d[i][1])], 
        #                     [(p2d_ref[i][0], p2d_ref[i][1])])

        #---------vis
        #     vertex.append(points3D)

        #     images.append(aug_image)
        #     cameras.append(camera)
        #     gt_body2view_poses.append(gt_body2view_pose)
        #     init_body2view_poses.append(initial_body2view_poses)
        #     output_names.append(output_name)

        #     images_ref.append(aug_image_ref)
        #     cameras_ref.append(camera_ref)
        #     gt_body2view_poses_ref.append(body2view_pose_ref)

        # vertex = torch.stack(vertex)

        # images = torch.stack(images)
        # cameras = torch.stack(cameras)
        # gt_body2view_poses = torch.stack(gt_body2view_poses)
        # init_body2view_poses = torch.stack(init_body2view_poses)
        # # output_names = torch.stack(output_names)

        # images_ref = torch.stack(images_ref)
        # cameras_ref = torch.stack(cameras_ref)
        # gt_body2view_poses_ref = torch.stack(gt_body2view_poses_ref)
        # points3D

        data = {
            'aligned_vertex': points3D[::5],
            # 'closest_template_vertices_ref': points3D_list, 
            # 'init_obj_pose_detection': init_obj_pose_detection,
            'images': aug_image,
            'cameras': camera,
            'gt_body2view_poses': gt_body2view_pose,
            'init_body2view_poses': initial_body2view_poses_list[0, 0],
            'output_name': output_name,
            'ref_path': frame_item['ref_rgb'],
            'query_path': image_path,
            'images_ref': aug_image_ref_list[0],
            'cameras_ref': camera_ref_list[0, 0],
            'gt_body2view_poses_ref': body2view_pose_ref_list[0, 0],
            # "vis_image": image,
            # "vis_image_ref": image_ref,
            # "origin": torch.tensor(frame_item['origin']),
            'OPT': False,
            'sysmetric': False
        }
        #         data = {
        #     'init_body2view_poses': init_body2view_poses, #torch.Size([2])
        #     'init_obj_pose_detection': init_obj_pose_detection, #torch.Size([12, 8])
        #     'images': images, #torch.Size([2, 3, 256, 256])
        #     'cameras': cameras, # torch.Size([2])
        #     'gt_body2view_poses': gt_body2view_poses, #torch.Size([2])
        #     'images_ref': images_ref, # torch.Size([12, 1, 3, 256, 256])
        #     'cameras_ref': cameras_ref, #torch.Size([12, 1])
        #     'gt_body2view_poses_ref': gt_body2view_poses_ref, # torch.Size([12, 1])
        #     'closest_template_vertices_ref': closest_template_vertices_list_ref, # torch.Size([12, 200, 3])
        #     'closest_template_views_ref': closest_template_views_list_ref, #torch.Size([12, 200, 8])
        #     'aligned_vertex': vertex, # torch.Size([500, 3])
        #     'diameter': diameter, # tensor(0.1711)
        #     'obj_name': obj_id, #'ycbv_19'
        #     'output_names': output_names, # ['test_000048_ycbv_19_0_001097', 'test_000048_ycbv_19_0_001095']
        #     'OPT': False,
        #     'sysmetric': False
        # }

        if self.conf.with_depth:
            data['depths'] = depths
            data['xyz_maps'] = xyz_maps


        return data

    def __len__(self):
        return len(self.items)
