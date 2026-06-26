import json
import os
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import copy
import glob
import cv2
import random
from .base_dataset import BaseDataset, set_seed
import torch
import numpy as np
import time
from ..utils.geometry.wrappers import Pose, Camera
from ..utils.utils import project_correspondences_line, get_closest_template_view_index, \
    get_closest_k_template_view_index, generate_random_aa_and_t, get_bbox_from_p2d, generate_spheric_cameras
from .utils import read_template_data, read_image, resize, numpy_image_to_torch, crop, zero_pad, get_imgaug_seq
import logging
from tqdm import tqdm
import imgaug as ia
from pytorch3d.io import load_objs_as_meshes, load_obj, load_ply
from pytorch3d.structures import Meshes
from itertools import combinations
from ..utils.transform import ECEF_to_WGS84, WGS84_to_ECEF, get_rotation_enu_in_ecef
from ..pixlib.geometry import Pose, Camera
from ..utils.get_depth import sample_points_with_valid_depth, get_3D_samples, get_points2D_ECEF, transform_ecef_origin, get_points2D_ECEF_projection
logger = logging.getLogger(__name__)


class _Dataset_Aero(torch.utils.data.Dataset):
    def __init__(self, conf, split):

        self.root = Path(conf.dataset_dir)
        train_split = getattr(conf, 'train_split_dir', 'Train')
        val_split = getattr(conf, 'val_split_dir', 'Validation')
        self.train_dir = os.path.join(self.root, train_split)
        self.validation_dir = os.path.join(self.root, val_split)
        # self.query_path = Path(conf.dataset_dir, 'Query')
        # self.seq_names = ['Jan_seq2']

        self.conf, self.split = conf, split

        self.geometry_unit_in_meter = float(conf.geometry_unit_in_meter)
        self.min_offset_angle = float(conf.min_offset_angle)
        self.max_offset_angle = float(conf.max_offset_angle)
        self.min_offset_translation = float(conf.min_offset_translation)
        self.max_offset_translation = float(conf.max_offset_translation)

        self.pbr_slices = split
        
        self.mul = conf.mul
     
    def add_noise_to_pose(self, euler_angles, translation, noise_std_angle=5.0, noise_std_translation=0.5, num_candidates=7):
        """
        Generate candidate poses by adding noise to Euler angles and translations.

        :param euler_angles: List or array of 3 Euler angles (roll, pitch, yaw) in degrees
        :param t_c2w: List or array of 3 translations (x, y, z)
        :param noise_std_angle: Standard deviation for angle noise in degrees
        :param noise_std_translation: Standard deviation for translation noise
        :param num_candidates: Number of candidate poses to generate
        :return: List of candidate poses, each pose is a dictionary with 'euler_angles' and 't_c2w'
        """
        candidates = []
        lon, lat, _ = translation
        rot_pose_in_enu = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()
        rot_enu_to_ecef = get_rotation_enu_in_ecef(lon, lat)
        R_c2w = np.matmul(rot_enu_to_ecef, rot_pose_in_enu)
        t_c2w = WGS84_to_ECEF(translation)
        # Initialize a 4x4 identity matrix
        render_T = np.eye(4)
        render_T[:3, :3] = R_c2w
        render_T[:3, 3] = t_c2w
        candidates.append(render_T.tolist())

        for _ in range(num_candidates):
            noisy_euler_angles = euler_angles + np.random.normal(0, noise_std_angle, size=3)
            noisy_t_c2w = t_c2w + np.random.normal(0, noise_std_translation, size=3)

            noise_trans = ECEF_to_WGS84(noisy_t_c2w)
            lon, lat, _ =noise_trans
            rot_pose_in_enu = R.from_euler('xyz', noisy_euler_angles, degrees=True).as_matrix()
            rot_enu_to_ecef = get_rotation_enu_in_ecef(lon, lat)
            noisy_R_c2w = np.matmul(rot_enu_to_ecef, rot_pose_in_enu)
            
            # Initialize a 4x4 identity matrix
            noisy_render_T = np.eye(4)
            noisy_render_T[:3, :3] = noisy_R_c2w
            noisy_render_T[:3, 3] = noisy_t_c2w

            candidates.append(noisy_render_T)

        return np.array(candidates)
    def read_image_(self, image_path, camera: Camera, depth_image_path = None, bbox2d=None, image=None, img_aug=False):
        
        img = read_image(image_path)
        # if conf.crop:
        #     if conf.crop_border:
        #         bbox2d[2:] += conf.crop_border * 2
        #     img, camera, bbox = crop(img, bbox2d, camera=camera, return_bbox=True)
        scales = (1, 1)
        img, scales = resize(img, 256, fn=max)
        if scales != (1, 1):
            camera = camera.scale(scales)

        img= zero_pad(256, img)
            # import ipdb; ipdb.set_trace()

        # if img_aug:
        #     img_aug = self.image_aug(img)
        # else:
        #     img_aug = img
        # img_aug = img_aug.astype(np.float32)
        img = img[0].astype(np.float32)
        if depth_image_path is not None:
            depth = cv2.imread(str(depth_image_path), cv2.IMREAD_UNCHANGED)
            depth = cv2.flip(depth, 0)
            depth, scales = resize(depth, 256, fn=max)
            depth= zero_pad(256, depth)[0]
            return img, camera, depth, scales#numpy_image_to_torch(img),  camera, numpy_image_to_torch(depth), scales

        return numpy_image_to_torch(img), numpy_image_to_torch(img_aug), camera, scales
    def sample_new_items(self, seed):
        logger.info(f'Sampling new images with seed {seed}')
        set_seed(seed)
        pbr_slices = []

        if self.pbr_slices == 'none':
            pbr_slices = []
        else:
            # seq_list: seq1/000000, seq1/000001 ,...,seq2/000000  
            if self.pbr_slices == 'all':
                pbr_slices = os.listdir(self.train_dir) + os.listdir(self.validation_dir)
                
            elif self.pbr_slices == 'train':
                # pbr_slices = seq_list[:int(0.7*len(seq_list))]
                pbr_slices = os.listdir(self.train_dir)
                data_folder = self.train_dir
            elif self.pbr_slices == 'val' or self.pbr_slices == 'test':
                pbr_slices = os.listdir(self.validation_dir)
                data_folder = self.validation_dir
            else:
                raise NotImplementedError

        self.items = []
        num_total = 0
        mul = self.mul
        for pbr_slice in tqdm(pbr_slices):
            pbr_list = []
            ref_info_dict = {}

            # reference information
            ref_info_path = os.path.join(data_folder, pbr_slice, 'refer_info.json')
            with open(ref_info_path, 'r', encoding='utf8') as fp:
                ref_info = json.load(fp)
            ref_info_dict[pbr_slice] = ref_info

            # name_list = ref_info.keys()
            
            Point3D_list = os.listdir(os.path.join(data_folder, pbr_slice, 'Points3D'))
            pbr_list.extend(list(map(lambda name: os.path.join(pbr_slice, name.split('.')[0]+'.png'), Point3D_list)))
            # Shuffle
            # random.shuffle(pbr_list)
            
            for query in pbr_list:
                total_start = time.perf_counter()

                stage_t0 = time.perf_counter()
                pbr_slice, img_name = os.path.split(query)
                
                origin = np.array(ref_info_dict[pbr_slice][img_name]['origin'])
                stage_t1 = time.perf_counter()

                RGB_path = os.path.join(data_folder, ref_info_dict[pbr_slice][img_name]['img_path'])
                Points3D_path = os.path.join(data_folder, pbr_slice, 'Points3D', img_name.split('.')[0] +'.npy')
                depth_relative_path = ref_info_dict[pbr_slice][img_name]['img_depth']
                query_depth_path = os.path.join(data_folder, depth_relative_path)
                pose_query = np.array(ref_info_dict[pbr_slice][img_name]['img_pose'])
                K = ref_info_dict[pbr_slice][img_name]['img_intrisic']
                width, height = K[0], K[1]
                cam_query = {
                    'model': 'PINHOLE',
                    'width': width,
                    'height': height,
                    'params': [K[2], K[3], K[4], K[5]]
                }

                # Points_3D_ECEF = np.load(Points3D_path)
                # Points_3D_ECEF_origin_total = Points_3D_ECEF - np.tile(origin, (len(Points_3D_ECEF), 1))   
                # indices = np.random.randint(0, len(Points_3D_ECEF_origin_total), size=500)
                # Points_3D_ECEF_origin_total = Points_3D_ECEF_origin_total[indices]
                # Points_3D_ECEF_origin_total = Points_3D_ECEF_origin_total * mul

                ref_info = ref_info_dict[pbr_slice][img_name]["ref_info"]
                img_ref_path = ref_info["ref_rgb"]
                depth_ref_path = ref_info["ref_depth"]
                pose_ref = np.array(ref_info["ref_poses"])
                K = ref_info["ref_intrinsics"]

                
                pose_query[:3, 3] = pose_query[:3, 3] * mul
                pose_ref[:3, 3] = pose_ref[:3, 3] * mul
                origin_mul = origin * mul

                pose_query_origin, pose_query_origin_w2c = transform_ecef_origin(pose_query, origin_mul)
                pose_ref_origin, pose_ref_origin_w2c = transform_ecef_origin(pose_ref, origin_mul)
                stage_t5 = time.perf_counter()

                ref_dict = copy.deepcopy(ref_info)
                ref_dict["ref_poses"] = pose_ref_origin_w2c
                ref_dict["ref_rgb"] = os.path.join(data_folder, img_ref_path)
                ref_dict["ref_depth"] = os.path.join(data_folder, depth_ref_path)

                item = {
                    'slice': pbr_slice, 
                    'origin': origin,
                    'mul': mul,
                    'image_name': img_name, 
                    'image_path': RGB_path,
                    'image_pose_gt': pose_query_origin_w2c,
                    'image_intrinscis': K, 
                    'Points3D_path': Points3D_path,
                }
                item.update(ref_dict)
                self.items.append(item)
           
            num_total+= len(pbr_list)

        print("Load Ref-Query pairs: ", num_total)
        if self.conf.img_aug:
            ia.seed(seed)
  
    def update_offset_angle_and_translation(self):
        logger.info(f'Offset angle: {self.min_offset_angle}, {self.max_offset_angle}')
        logger.info(f'Offset translation: {self.min_offset_translation}, {self.max_offset_translation}')

    def image_aug(self, img):
        seq = get_imgaug_seq()
        img_aug = seq(image=img)
        return img_aug

    def read_image(self, image_path, conf, camera: Camera, bbox2d, image=None, img_aug=False):

        # read image
        if image is None:
            img = read_image(image_path, conf.grayscale)
        else:
            img = image

        if conf.crop:
            if conf.crop_border:
                bbox2d[2:] += conf.crop_border * 2
            img, camera, bbox = crop(img, bbox2d, camera=camera, return_bbox=True)

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

        # if conf.change_background:
        #     raise NotImplementedError

        if conf.pad:
            img, = zero_pad(conf.pad, img)

        if img_aug:
            img_aug = self.image_aug(img)
        else:
            img_aug = img
        img_aug = img_aug.astype(np.float32)

        return numpy_image_to_torch(img_aug), camera

    def transform_img(self, img, bbox2d, conf):
        if conf.crop:
            if conf.crop_border:
                bbox2d[2:] += conf.crop_border * 2
            img, bbox = crop(img, bbox2d, camera=None, return_bbox=True)

        if conf.resize:
            if isinstance(conf.resize, int):
                if conf.resize_by == 'max':
                    # print('img shape', img.shape)
                    # print('img path', image_path)
                    img, _ = resize(img, conf.resize, fn=max)
                elif (conf.resize_by == 'min' or (conf.resize_by == 'min_if' and min(*img.shape[:2]) < conf.resize)):
                    img, _ = resize(img, conf.resize, fn=min)
            elif len(conf.resize) == 2:
                img, _ = resize(img, list(conf.resize))

        if conf.pad:
            img, = zero_pad(conf.pad, img)

        return numpy_image_to_torch(img)

    def read_mask(self, mask_path, mask_visib_path, bbox2d, conf):
        mask = read_image(mask_path, True)
        mask_visib = read_image(mask_visib_path, True)

        mask_edge = cv2.Canny(mask, 100, 200)
        mask_visib_edge = cv2.Canny(mask_visib, 100, 200)

        # edge_visib = mask_visib
        edge_visib = mask_edge & mask_visib_edge

        return self.transform_img(edge_visib, bbox2d.copy(), conf), self.transform_img(mask_edge, bbox2d.copy(), conf), \
               self.transform_img(mask_visib, bbox2d.copy(), conf)

    def draw_mask(self, template_views, gt_body2view_pose, orientations_in_body, n_sample, camera, image):
        gt_index = get_closest_template_view_index(gt_body2view_pose, orientations_in_body)
        gt_template_view = template_views[gt_index * n_sample:(gt_index + 1) * n_sample, :]
        data_lines = project_correspondences_line(gt_template_view, gt_body2view_pose, camera)
        gt_centers_in_image = data_lines['centers_in_image'].unsqueeze(1).numpy().astype(np.int)
        mask = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)
        mask = cv2.drawContours(mask, [gt_centers_in_image], -1, 1, -1)

        return mask

    def change_background(self, idx, image, mask):

        if np.random.rand() < 0.5:
            return image

        background_path = Path(self.background_image_dir, self.selected_background_image_path[idx])
        background_image = read_image(background_path, self.conf.grayscale)
        background_image, _ = resize(background_image, image.shape[:2])
        mask = np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
        img = np.where(mask == 0, background_image, image)
        # img = torch.where(mask.expand(3, -1, -1) == 0, background_image, image)
        # img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        # cv2.imwrite('./test.png', img)

        return img

    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = item['image_path']
        img_id = item['img_id']
        output_name = item['output_name']
        ori_image = read_image(image_path, self.conf.grayscale)
        obj_id = item['obj_id']
        body2view_R = item['body2view_R'].reshape(3, 3)
        body2view_t = item['body2view_t']
        gt_body2view_pose = Pose.from_Rt(body2view_R, body2view_t)
        K = item['K']
        intrinsic_param = torch.tensor([ori_image.shape[1], ori_image.shape[0],
                                        K[0], K[4], K[2], K[5]], dtype=torch.float32)
        ori_camera = Camera(intrinsic_param)
        orientations_in_body = item['orientations_in_body']
        template_views = item['template_views']
        n_sample = item['n_sample']
        diameter = item['diameter']
        # generate offset to ground truth pose
        if (img_id == 0) or (self.split == 'train' and self.conf.train_offset) or \
                (self.split == 'val' and self.conf.val_offset):  # self.split == 'train' or self.conf.val_offset:
            random_aa, random_t = generate_random_aa_and_t(self.min_offset_angle, self.max_offset_angle,
                                                           self.min_offset_translation, self.max_offset_translation)
            random_pose = Pose.from_aa(random_aa, random_t)
            body2view_pose = gt_body2view_pose @ random_pose[0]
        else:
            # last_body2view_R = item['last_body2view_R'].reshape(3, 3)
            # last_body2view_t = item['last_body2view_t']
            # body2view_pose = Pose.from_Rt(last_body2view_R, last_body2view_t)
            raise NotImplementedError

        # get closest template view
        indices = get_closest_k_template_view_index(body2view_pose,
                                                    orientations_in_body,
                                                    self.conf.get_top_k_template_views * self.conf.skip_template_view)
        closest_template_views = torch.stack([template_views[ind * n_sample:(ind + 1) * n_sample, :]
                                              for ind in indices[::self.conf.skip_template_view]])
        closest_orientations_in_body = orientations_in_body[indices[::self.conf.skip_template_view]]

        # calc bbox
        data_lines = project_correspondences_line(closest_template_views[0], body2view_pose, ori_camera)
        bbox2d = get_bbox_from_p2d(data_lines['centers_in_image'])

        # read image
        image, camera = self.read_image(image_path, self.conf, ori_camera, bbox2d.numpy().copy(), ori_image,
                                        img_aug=self.conf.img_aug if self.split == 'train' else False)

        if self.conf.change_background and self.split == 'train':
            ori_mask = self.draw_mask(template_views, gt_body2view_pose, orientations_in_body,
                                      n_sample, ori_camera, ori_image)
            ori_image_with_background = self.change_background(idx, ori_image, ori_mask)
            image = self.transform_img(ori_image_with_background, bbox2d.numpy().copy(), self.conf)

        # new_image = (new_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        # cv2.imwrite('./test.png', new_image)

        # read mask
        # mask_path = item['mask_path']
        # mask_visib_path = item['mask_visib_path']
        # edge_visib, edge, mask_visib = self.read_mask(mask_path, mask_visib_path, bbox2d.numpy().copy(), self.conf)

        # if self.conf.change_background:
        #     image = self.change_background(idx, image, mask_visib)

        try:
            vertex = item['vertex']
            num_vertex = vertex.shape[0]
            if num_vertex < self.conf.sample_vertex_num:
                expand_num = self.conf.sample_vertex_num // num_vertex + 1
                vertex = vertex.unsqueeze(0).expand(expand_num, -1, -1).reshape(-1, 3)
                vertex = vertex[:self.conf.sample_vertex_num]
            else:
                step = num_vertex // self.conf.sample_vertex_num
                vertex = vertex[::step, :]
                vertex = vertex[:self.conf.sample_vertex_num, :]
        except ValueError:
            import ipdb;
            ipdb.set_trace();
        data = {
            'image': image,
            # 'mask_visib': mask_visib,
            # 'edge_visib': edge_visib,
            # 'edge': edge,
            'camera': camera,
            'body2view_pose': body2view_pose,
            'aligned_vertex': vertex,
            'gt_body2view_pose': gt_body2view_pose,
            'closest_template_views': closest_template_views,
            'closest_orientations_in_body': closest_orientations_in_body,
            'diameter': diameter,
            'image_path': image_path,
            'obj_name': obj_id,
            'output_name': output_name,
            'OPT': item['OPT'],
            'sysmetric': False
        }

        return data

    def __len__(self):
        return len(self.items)