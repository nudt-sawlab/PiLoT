import os
import random
from itertools import combinations
from tqdm import tqdm
import numpy as np
import json
import pyproj
from scipy.spatial.transform import Rotation as R
from wrapper import Pose, Camera
from get_depth import get_3D_samples, get_points2D_ECEF, transform_ecef_origin, get_points2D_ECEF_projection
from transform import ECEF_to_WGS84
import torch
import cv2


def numpy_image_to_torch(image):
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f'Not an image: {image.shape}')
    return torch.from_numpy(image / 255.).float()
def resize(image, size, fn=None, interp='linear'):
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h*scale)), int(round(w*scale))
        # TODO: we should probably recompute the scale like in the second case
        scale = (scale, scale)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f'Incorrect new size: {size}')
    mode = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'nearest': cv2.INTER_NEAREST}[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale

def crop(image, bbox2d, camera=None, return_bbox=False):
    """Random or deterministic crop of an image, adjust depth and intrinsics.
    """
    h, w = image.shape[:2]
    half_w_new, half_h_new = bbox2d[2:].astype(np.int32) // 2
    x, y = bbox2d[:2].astype(np.int32)
    left = np.clip(x - half_w_new, 0, w - 1)
    right = np.clip(x + half_w_new, 0, w - 1)
    top = np.clip(y - half_h_new, 0, h - 1)
    bottom = np.clip(y + half_h_new, 0, h - 1)

    image = image[top:bottom, left:right]
    ret = [image]
    if camera is not None:
        w_new = right-left
        h_new = bottom-top
        ret += [camera.crop((left, top), (w_new, h_new))]
        # ret += [camera.crop((left, top), (half_w_new*2, half_h_new*2))]
    if return_bbox:
        ret += [(top, bottom, left, right)]
    return ret

def zero_pad(size, *images):
    ret = []
    for image in images:
        h, w = image.shape[:2]
        # import ipdb; ipdb.set_trace();
        padded = np.zeros((size, size)+image.shape[2:], dtype=image.dtype)
        padded[:h, :w] = image
        ret.append(padded)
        
    return ret

def WGS84_to_ECEF(pos):
    lon, lat, height = pos
    transprojr = pyproj.Transformer.from_crs(
        "EPSG:4326",
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        always_xy=True,
    )
    xpjr, ypjr, zpjr = transprojr.transform(lon, lat, height, radians=False)
    return [xpjr, ypjr, zpjr]

def get_rotation_enu_in_ecef(lon, lat):
    """
    @param: lon, lat Longitude and latitude in degree
    @return: 3x3 rotation matrix of heading-pith-roll ENU in ECEF coordinate system
    Reference: https://apps.dtic.mil/dtic/tr/fulltext/u2/a484864.pdf, Section 4.3, 4.1
    Reference: https://www.fossen.biz/wiley/ed2/Ch2.pdf, p29
    """
    # 将角度转换为弧度
    latitude_rad = np.radians(lat)
    longitude_rad = np.radians(lon)
    
    # 计算向上的向量（Up Vector）
    up = np.array([
        np.cos(longitude_rad) * np.cos(latitude_rad),
        np.sin(longitude_rad) * np.cos(latitude_rad),
        np.sin(latitude_rad)
    ])
    
    # 计算向东的向量（East Vector）
    east = np.array([
        -np.sin(longitude_rad),
        np.cos(longitude_rad),
        0
    ])
    
    # 计算向北的向量（North Vector），即up向量和east向量的外积（叉积）
    north = np.cross(up, east)
    
    # 构建局部到世界坐标系的转换矩阵
    local_to_world = np.zeros((3, 3))
    local_to_world[:, 0] = east  # 东向分量
    local_to_world[:, 1] = north  # 北向分量
    local_to_world[:, 2] = up  # 上向分量
    return local_to_world

def get_direct_subfolders(base_path):
    """
    Collect folder paths only up to one level deep in the given structure.

    :param base_path: The root directory to start traversing.
    :return: A list of subfolder paths relative to the base_path.
    """
    folder_list = []
    for root, dirs, _ in os.walk(base_path):
        # Check if the current directory is directly below the first-level folder (e.g., seq1/000000)
        if os.path.basename(root).isdigit():  # Second-level directories like 000000
            # Append relative path starting from base_path
            relative_path = os.path.relpath(root, base_path)
            folder_list.append(relative_path)
    return folder_list


def visualize_points_on_images(img1, img2, points1, points2, point_color=(0, 255, 0), radius=5, thickness=-1):
    """
    Visualize points on two images side by side.

    :param img1: First image (numpy array, BGR format).
    :param img2: Second image (numpy array, BGR format).
    :param points1: List of points (x, y) for the first image.
    :param points2: List of points (x, y) for the second image.
    :param point_color: Color of the points (default green).
    :param radius: Radius of the points to draw.
    :param thickness: Thickness of the point circles (-1 means filled circle).
    :return: Combined image with visualized points.
    """
    # Make copies to avoid modifying original images
    img1_vis = img1.copy()
    img2_vis = img2.copy()

    # Draw points on the first image
    for (x, y) in points1:
        cv2.circle(img1_vis, (int(x), int(y)), radius, point_color, thickness)

    # Draw points on the second image
    for (x, y) in points2:
        cv2.circle(img2_vis, (int(x), int(y)), radius, point_color, thickness)

    # Combine the two images side by side
    combined_img = cv2.hconcat([img1_vis, img2_vis])
    save_path = "/home/ubuntu/Documents/code/FPVLoc/datasets/Aero/Raw/vis/verify_reprojection"
    save_name = os.path.join(save_path, str(len(os.listdir(save_path))+1)+'.png')
    cv2.imwrite(save_name, combined_img)

    return combined_img
def read_image(path, grayscale=False):
    mode = cv2.IMREAD_GRAYSCALE if grayscale else None
    if grayscale is True:
        image = cv2.imread(str(path), mode)
    else:
        image = cv2.imread(path)

    if image is None:
        raise IOError(f'Could not read image at {path}.')
    if not grayscale:
        image = image[..., ::-1]
    return image
def read_image_(image_path, camera: Camera, depth_image_path = None, bbox2d=None, image=None, img_aug=False):
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
        depth, scales = resize(depth, 256, fn=max)
        depth= zero_pad(256, depth)[0]
        return img, camera, scales, depth


    return img, camera, scales
def add_noise_to_pose(euler_angles, translation, noise_std_angle=5.0, noise_std_translation=0.5, num_candidates=7):
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
    rot_pose_in_enu = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
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
        rot_pose_in_enu = R.from_euler('xyz', noisy_euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
        rot_enu_to_ecef = get_rotation_enu_in_ecef(lon, lat)
        noisy_R_c2w = np.matmul(rot_enu_to_ecef, rot_pose_in_enu)
        
        # Initialize a 4x4 identity matrix
        noisy_render_T = np.eye(4)
        noisy_render_T[:3, :3] = noisy_R_c2w
        noisy_render_T[:3, 3] = noisy_t_c2w

        candidates.append(noisy_render_T)

    return np.array(candidates)
if __name__ == "__main__":
    base_directory = "/home/ubuntu/Documents/code/FPVLoc/datasets/Aero/" # Replace with the base directory of your sequences
    # all_folders = get_direct_subfolders(base_directory)
    
    # # Print the collected folder paths
    # print("Collected folders:")
    # for folder in all_folders:
    #     print(folder)

    pose_file = "/home/ubuntu/Documents/code/FPVLoc/datasets/Aero/Query/Jan_seq2/000000/pose.txt"
    query_path = "/home/ubuntu/Documents/code/FPVLoc/datasets/Aero/Query"
    reference_path = "/home/ubuntu/Documents/code/FPVLoc/datasets/Aero/Reference"
    pbr_list = []
    seq_list = []
    for root, dirs, _ in os.walk(query_path):
        # Check if the current directory is directly below the first-level folder (e.g., seq1/000000)
        if os.path.basename(root).isdigit():  # Second-level directories like 000000
            # Append relative path starting from base_path
            relative_path = os.path.relpath(root, query_path)
            seq_list.append(relative_path)
    seq_list.sort()
    pbr_slices = seq_list
      
    obj_items = {}
    num_frames = 2
    # pbr_slices
    # pbr_slices: seq1/000000, seq1/000001 ,...,seq2/000000  
    pbr_list = []
    poses_dict = {}
    ref_info_dict = {}
    for pbr_slice in tqdm(pbr_slices):
            data_dir = os.path.join(query_path, pbr_slice)  
            # pose.R, pose.t
            
            # reference information
            ref_info_path = os.path.join(data_dir, 'refer_info.json')
            with open(ref_info_path, 'r', encoding='utf8') as fp:
                ref_info = json.load(fp)
            ref_info_dict[pbr_slice] = ref_info
            
            name_list = list(ref_info.keys())
            pbr_list.extend(list(map(lambda name: os.path.join(pbr_slice, name), name_list)))

    # ------- shuffle
    # Shuffle the image list to ensure randomness
    random.shuffle(pbr_list)

    # Group images into sets of size n
    items_list = []
    i = 0
    while i < len(pbr_list):
        if i + num_frames <= len(pbr_list):
            # Create a group of n images
            items_list.append(tuple(pbr_list[i:i + num_frames]))
            i += num_frames
        else:
            # If not enough images are left, randomly select additional images to complete the group
            remaining_images = pbr_list[i:]
            while len(remaining_images) < num_frames:
                remaining_images.append(random.choice(pbr_list))
            items_list.append(tuple(remaining_images))
            break

    # query RGB
    items = []
    for query_items in items_list:
        frame_items = []
        for query in query_items:
            # query = 'Jan_seq2/000001/3_4320.png'
            pbr_slice, img_name = os.path.split(query)
            base_dir = os.path.join(query_path, pbr_slice)
            # query RGB
            RGB_path = os.path.join(base_dir, 'RGB', img_name)
            # query pose---6-freedom to [R, t]
            T_c2w = np.array(ref_info_dict[pbr_slice][img_name]['img_pose'])
            et = ref_info_dict[pbr_slice][img_name]['img_pose_6']
            T_c2w_list = add_noise_to_pose(et[:3], et[3:])

            # query_pose_euler = [135.4952986366768, 34.61647769927806, 349.72181821408884, 44.58677849397614, -0.04386775097711815, 63.959086793208705]
            # euler_angles = query_pose_euler[3:]
            # translation = query_pose_euler[0:3]
            # lon, lat, height = translation
            # rot_pose_in_ned = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
            # rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
            # R_c2w = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
            # t_c2w = WGS84_to_ECEF(translation)
            # print("drone wgs84: ", translation)
            # # (34.61566232598386,135.49318247040046)
            # # 34.61766354441435,135.4923709425798
            # T_c2w = np.eye(4)
            # T_c2w[:3, :3] = R_c2w
            # T_c2w[:3, 3] = t_c2w
            # T_c2w = T_c2w.tolist()
            # K
            K = ref_info_dict[pbr_slice][img_name]['img_intrisic']
            width, height = K[0], K[1]

            cam_query = {
            'model': 'PINHOLE',
            'width': width,
            'height': height,
            'params': [1200.0, 1200.0, 320.0, 240.0] #np.array(K[2:]
            }   
            
            qcamera = Camera.from_colmap(cam_query)

            # depth
            depth_relative_path = ref_info_dict[pbr_slice][img_name]['img_depth']
            query_depth_path = os.path.join(base_directory, depth_relative_path)
            query_rgb_path = ref_info_dict[pbr_slice][img_name]['img_path']
            img_query = cv2.imread(query_rgb_path)
            depth_image = cv2.imread(query_depth_path, cv2.IMREAD_UNCHANGED)
            query_image = cv2.imread(query)
            # 2D
            points2d_total = sample_points_with_valid_depth(depth_image, num_points=10000, max_depth=2000)

            # points2d_i = 480 - points2d[:, 1]
            point3D, Points_3D_ECEF_origin_total, origin = get_3D_samples(points2d_total, depth_image, T_c2w, qcamera)
            T_c2w_origin = transform_ecef_origin(T_c2w_list, origin)

            # points2d0 = get_points2D_ECEF_projection(T_c2w_origin[0], qcamera, Points_3D_ECEF_origin_total, points2d_total)
            # 验证地理位置准确性
            point3D_wgs84 = ECEF_to_WGS84(point3D[0])
            print("wgs84: ", point3D_wgs84)

            # 验证参考位姿准确性
            ref_info = ref_info_dict[pbr_slice][img_name]["ref_info"]
            pose_ref = np.array(ref_info["ref_poses"])
            img_ref_path = ref_info["ref_rgb"]
            depth_ref_path = ref_info["ref_depth"]
            img_ref = cv2.imread(os.path.join(base_directory, img_ref_path))
            depth_ref = cv2.imread(os.path.join(base_directory, depth_ref_path), cv2.IMREAD_UNCHANGED)

            pose_ref_origin = transform_ecef_origin(pose_ref, origin)
            point2d_ref, point2d_query, Points_3D_ECEF_origin = get_points2D_ECEF_projection(pose_ref_origin, qcamera, Points_3D_ECEF_origin_total, points2d_total, num_samples=500)

            visualize_points_on_images(img_query, img_ref, 
            [(point2d_query[0][0], point2d_query[0][1])], 
            [(point2d_ref[0][0], point2d_ref[0][1])])

            
            intrinsic_param = torch.tensor([width, height,
                                        K[2], K[3], K[4], K[5]], dtype=torch.float32)
            ori_camera = Camera(intrinsic_param)
            ima_query_pad, query_camera, scales, depth_query_pad = read_image_(query_rgb_path, ori_camera, depth_image_path = query_depth_path)

            img_ref_path = os.path.join(base_directory, img_ref_path)
            depth_ref_path = os.path.join(base_directory, depth_ref_path)
            img_ref_pad, ref_camera, scales, depth_ref_pad = read_image_(img_ref_path, ori_camera, depth_image_path = depth_ref_path)
            points2d_pad = point2d_query * scales
            point2d_pad_ref = point2d_ref * scales

            

            cam_query = {
            'model': 'PINHOLE',
            'width': query_camera.size[0],
            'height': query_camera.size[1],
            'params': query_camera._data[2:] #np.array(K[2:]
            }   
            
            qcamera = Camera.from_colmap(cam_query)
            point3D, Points_3D_ECEF_origin, origin_pad = get_3D_samples(points2d_pad, depth_query_pad, T_c2w, qcamera, origin = origin)
            # point2d_ref_pad, point2d_query_pad, Points_3D_ECEF_origin = get_points2D_ECEF_projection(pose_ref_origin, qcamera, Points_3D_ECEF_origin, points2d_pad)
            
            # point2d_ref_pad = point2d_ref[:, :2] *scales
            # visualize_points_on_images(ima_query_pad, img_ref_pad, 
            # [(points2d_pad[0][0], points2d_pad[0][1])], 
            # [(point2d_ref_pad[0][0], point2d_ref_pad[0][1])])

            # visualize_points_on_images(ima_query_pad, img_ref_pad, 
            # [(points2d_pad[0][0], points2d_pad[0][1])], 
            # [(point2d_ref[0][0] *scales[0], point2d_ref[0][1]*scales[1])])


            item = {'slice': pbr_slice, 
                    'image_name': img_name, 
                    'image_path': RGB_path,
                    }

            item.update(ref_info_dict[pbr_slice][img_name])
            frame_items.append(item)
        items.append(frame_items)