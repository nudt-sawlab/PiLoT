import logging
from scipy.spatial.transform import Rotation as R

from pixloc.utils.transform import  WGS84_to_ECEF, get_rotation_enu_in_ecef, rotmat2qvec
from pathlib import Path
from typing import Union, Dict, Tuple, Optional
import numpy as np
from .io import parse_image_list
from .colmap import qvec2rotmat, read_images_binary, read_images_text
logger = logging.getLogger(__name__)

def euler_angles_to_matrix_ECEF_w2c(euler_angles, trans):
    lon, lat, _ = trans
    rot_pose_in_enu = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
    rot_enu_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = np.matmul(rot_enu_in_ecef, rot_pose_in_enu)
    t_c2w = WGS84_to_ECEF(trans)
    # R_w2c_in_ecef = R_c2w.transpose() # 和enu的差异是第二行和第三行取负号
    # t_w2c = -R_w2c_in_ecef.dot(t_c2w)

    # T_render_in_ECEF_w2c = np.eye(4)
    # T_render_in_ECEF_w2c[:3, :3] = R_w2c_in_ecef
    # T_render_in_ECEF_w2c[:3, 3] = t_w2c
    return R_c2w, np.array(t_c2w)
def evaluate(results, gt, only_localized=False):
    """
    Evaluate the accuracy of pose predictions by comparing them with ground truth poses.
    
    Args:
        results (str): Path to the file containing the predicted poses.
        gt (str): Path to the file containing the ground truth poses.
        only_localized (bool): Flag to skip unlocalized images. Defaults to False.
    
    Returns:
        str: A formatted string with the evaluation results.
    """
    predictions = {}
    test_names = []
    with open(results, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            tokens = data.split()
            name = tokens[0].split('/')[-1]
            t, e = np.split(np.array(tokens[1:], dtype=float), [3])
            e = [e[1], e[0], e[2]]
            R_c2w, t_c2w = euler_angles_to_matrix_ECEF_w2c(e, t)
            # trans = WGS84_to_ECEF(t)
            # Convert quaternion to rotation matrix and store with translation
            predictions[name] = (R_c2w, t_c2w, e)
            test_names.append(name)
            

    gts = {}
    with open(gt, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            tokens = data.split()
            name = tokens[0].split('/')[-1]  #'0_0.png, 0.jpg
            name = name[0]+'_0.png'
            # if name not in test_names: continue
            t, e = np.split(np.array(tokens[1:], dtype=float), [3])
            e = [e[1], e[0], e[2]]
            R_c2w, t_c2w = euler_angles_to_matrix_ECEF_w2c(e, t)
            # trans = WGS84_to_ECEF(t)
            # Convert quaternion to rotation matrix and store with translation
            gts[name] = (R_c2w, t_c2w, e)
    errors_t = []
    errors_R = []
    errors_yaw = []
    e_trans_list = []
    for name in test_names:
        if name not in predictions:
            if only_localized:
                continue
            e_t = np.inf
            e_R = 180.
        else:
            # index = name.split('.')[0]
            # gt_names = list(gts.keys())
            # for s in gt_names:
            #     if index in s:
            #         gt_name = s
            # euler_gt, trans_gt = gts[name]
            # euler_es, trans_es = predictions[name]
            
            # e_R = euler_gt - euler_es
            # e_t = trans_gt - trans_es
            R_gt, t_gt, euler_gt = gts[name.split('_')[0]+'_0.png']
            R, t, euler = predictions[name]
            # Calculate translation and rotation errors
            # e_trans = -R_gt.T @ t_gt + R.T @ t
            e_t = np.linalg.norm(-t_gt + t, axis=0)
            cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1., 1.)
            e_R = np.rad2deg(np.abs(np.arccos(cos)))
            delta = (euler[-1] - euler_gt[-1] + 180) % 360 - 180
            e_yaw = abs(delta)
        
        # e_trans_list.append(e_trans)
        errors_t.append(e_t)
        errors_R.append(e_R)
        errors_yaw.append(e_yaw)
    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)
    errors_yaw = np.array(errors_yaw)
    # a = errors_R[500:]
    # b = errors_t[500:]
    med_t = np.median(errors_t)
    std_t = np.std(errors_t)
    med_R = np.median(errors_R)
    std_R = np.std(errors_R)
    med_yaw = np.median(errors_yaw)
    std_yaw = np.std(errors_yaw)
    
    out = '\nMedian errors: {:.3f}m, {:.3f}deg, {:.3f}deg\n'.format(med_t, med_R, med_yaw)
    out += 'Std errors: {:.3f}m, {:.3f}deg, {:.3f}deg\n'.format(std_t, std_R, std_yaw)
    
    out += 'Percentage of test images localized within:'
    threshs_t = [0.2, 0.5, 1, 3, 5]
    threshs_R = [0.2, 0.5, 1.0, 3.0, 5.0]
    threshs_yaw = [0.01, 0.02, 0.05, 0.1, 0.2]
    for th_t, th_R in zip(threshs_t, threshs_R):
        ratio = np.mean((errors_t < th_t) & (errors_R < th_R))
        out += '\n\t{:.0f}cm, {:.1f}deg : {:.2f}%'.format(th_t * 100, th_R, ratio * 100)
    for th_yaw in threshs_yaw:
        ratio_yaw = np.mean((errors_yaw < th_yaw)) 
        out += '\n\t{:.2f}deg : {:.2f}%'.format(th_yaw, ratio_yaw * 100)

    
    print(out)
    return out


def cumulative_recall(errors: np.ndarray) -> Tuple[np.ndarray]:
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    return errors, recall*100
