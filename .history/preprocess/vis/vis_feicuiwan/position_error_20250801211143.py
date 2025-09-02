import os

import numpy as np
import os
import argparse
import csv
import cv2
from transform import WGS84_to_ECEF, euler_angles_to_matrix_ECEF_w2c, get_matrix
from get_depth import get_points2D_ECEF_projection_v2

def normalize_error_dict(error_dict, clip_percentile=(1, 99), eps=1e-6):
    """
    传入 key->error 的字典，做 percentile 截断后归一化到 [0,1]。
    clip_percentile: (low_pct, high_pct) 用于缓解极端值影响。
    """
    errors = np.array(list(error_dict.values()), dtype=np.float32)
    if len(errors) == 0:
        return {}

    # percentile 截断
    low, high = np.percentile(errors, clip_percentile)
    clipped = np.clip(errors, low, high)
    minv = clipped.min()
    maxv = clipped.max()
    if maxv - minv < eps:
        return {k: 0.5 for k in error_dict}  # 全部相同
    norm_vals = (clipped - minv) / (maxv - minv)
    return {k: float(norm_vals[i]) for i, k in enumerate(error_dict.keys())}

def draw_error_points_on_image(image, xy_dict, error_t,
                               point_radius=6, thickness=-1,
                               colormap=cv2.COLORMAP_JET):
    """
    把每个 key 的 error 画在 image 上，使用 colormap 映射误差大小，右侧附带 colorbar。
    xy_dict: {key: (x, y)} 像素位置
    error_t: {key: scalar_error}
    """
    img = image.copy()
    norm_err = normalize_error_dict(error_t)  # key -> [0,1]

    # 画点
    for key, coord in xy_dict.items():
        if key not in error_t:
            # optional: 你可以记录一下缺失的 key
            continue
        x, y = coord[0]
        err_norm = norm_err.get(key, 0.0)
        cmap_val = int(np.round(err_norm * 255))
        color = cv2.applyColorMap(np.array([[cmap_val]], dtype=np.uint8), colormap)[0, 0].tolist()
        cv2.circle(img, (int(x), int(y)), point_radius, color, thickness, lineType=cv2.LINE_AA)

    # 画 colorbar（右侧）
    h, w = img.shape[:2]
    bar_h = int(h * 0.6)
    bar_w = 20
    bar_x = w - bar_w - 10
    bar_y = int((h - bar_h) / 2)

    # 构造 gradient，从 high error (1.0 -> 上) 到 low error (0.0 -> 下)
    gradient = np.linspace(255, 0, bar_h, dtype=np.uint8)[:, None]
    gradient = np.repeat(gradient, bar_w, axis=1)  # shape (bar_h, bar_w)
    colored_bar = cv2.applyColorMap(gradient, colormap)
    img[bar_y:bar_y+bar_h, bar_x:bar_x+bar_w] = colored_bar

    # 标注实际 error 值范围（用原始 error_t，不是归一后）
    font = cv2.FONT_HERSHEY_SIMPLEX
    all_errors = np.array(list(error_t.values()), dtype=np.float32)
    if all_errors.size > 0:
        # 用 clip_percentile 里的值匹配上面 normalization 截断
        low_pct_val = np.percentile(all_errors, 1)
        high_pct_val = np.percentile(all_errors, 99)
        # 上 label: high_pct_val
        cv2.putText(img, f"{high_pct_val:.3f}", (bar_x - 5, bar_y + 12), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
        # 下 label: low_pct_val
        cv2.putText(img, f"{low_pct_val:.3f}", (bar_x - 5, bar_y + bar_h), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
        # 中间 “error”
        cv2.putText(img, "error", (bar_x - 5, bar_y + bar_h//2), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "no error", (bar_x - 5, bar_y + bar_h//2), font, 0.5, (0,0,255), 1, cv2.LINE_AA)

    return img

def evaluate_xyz(results, gt, only_localized=False):
    """
    Compare predicted positions (in WGS84) with GT positions (in WGS84),
    convert to ECEF, compute 2D XY distance error.
    Return error statistics in a dict.
    """
    predictions = {}
    test_names = []
    total_num = 0
    test_num = 0

    if not os.path.exists(results):
        print(f"[Warning] 预测文件不存在: {results}")
        return None
    if not os.path.exists(gt):
        print(f"[Warning] GT文件不存在: {gt}")
        return None

    with open(results, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 4:
                continue
            name = tokens[0].split('/')[-1]
            try:
                t = np.array(tokens[1:4], dtype=float)
                t_ecef = WGS84_to_ECEF(t)
                predictions[name] = t_ecef
                test_names.append(name)
                test_num += 1
            except Exception as e:
                print(f"[Error] 预测读取失败 {name}: {e}")
                continue

    gts = {}
    with open(gt, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 4:
                continue
            name = tokens[0].split('/')[-1]
            try:
                t = np.array(tokens[1:4], dtype=float)
                t_ecef = WGS84_to_ECEF(t)
                gts[name] = t_ecef
                total_num += 1
            except Exception as e:
                print(f"[Error] GT读取失败 {name}: {e}")
                continue

    errors_t = {}
    for name in test_names:
        gt_name = name.split('_')[0] + '_0.png'
        if name not in predictions or gt_name not in gts:
            if only_localized:
                continue
            errors_t[name] = np.inf
        else:
            t_gt = np.array(gts[gt_name])
            t_pred = np.array(predictions[name])
            e_t = np.linalg.norm(t_pred - t_gt)
            errors_t[name] = e_t

    return errors_t

def get_es_xyz(save_xyz_path):
    xyz_dict = {}
    with open(save_xyz_path, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line
            parts = line.strip().split()
            if parts:  # Ensure the line is not empty
                x, y, z = map(float, parts[1: ])
                xyz_dict[parts[0]] = [x, y, z] # Add the first element to the name list
    return xyz_dict 
def get_xy(save_xy_path):
    xy_dict = {}
    with open(save_xy_path, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line
            parts = line.strip().split()
            if parts:  # Ensure the line is not empty
                x, y = map(float, parts[1: ])
                xy_dict[parts[0]] = [[x, y]] # Add the first element to the name list
    return xy_dict 

def get_poses(pose_file, img_name):
    pose_dict = {}
    with open(pose_file, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line
            parts = line.strip().split()
            if parts:  # Ensure the line is not empty
                pose_dict[parts[0]] = {} # Add the first element to the name list
                lon, lat, alt, roll, pitch, yaw = map(float, parts[1: ])
                # pitch, roll, yaw, lon, lat, alt,  = map(float, parts[1: ])
                
                euler_angles = [pitch, roll, yaw]
                translation = [lon, lat, alt]
                pose_dict[parts[0]]['euler_angles'] = euler_angles
                pose_dict[parts[0]]['translation'] = translation
                
                R_c2w = get_matrix(translation, euler_angles)
                if img_name == parts[0]:
                    return R_c2w    
    
def main():
    base_path = "/mnt/sda/MapScape/query/estimation/position_result"
    image_path = "/mnt/sda/MapScape/query/images"
    methods = {
        'ORB@per30': 'Render2ORB',
        'FPVLoc': 'Geopixel',
        'Render2loc': 'Render2Loc',
        'Render2loc@raft': 'Render2RAFT',
        'Pixloc': 'PixLoc',
    }
    rcamera = [3840, 2160, 2700.0, 2700.0, 1915.7, 1075.1]

    results_list = []
    gt_path = os.path.join(base_path, "GT")
    render2orb_path = os.path.join(base_path, "ORB@per30")
    txt_files = [f for f in os.listdir(render2orb_path) if f.endswith('.txt') and 'DJI_' in f]
    pose_path = "/mnt/sda/MapScape/query/estimation/result_images"
    # 主流程（补全部分）
    # 主流程补全（示例）
    for txt_file in txt_files:
        seq = txt_file.split('.')[0]
        gt_file = os.path.join(gt_path, f"{seq}_RTK.txt")
        # save_xy_path = os.path.join("/mnt/sda/MapScape/query/bbox", seq, seq + '_xy.txt')
        
        # xy_dict = get_xy(save_xy_path)  # 需返回 {key: (x, y)}
        print(f"\n-------- Sequence: {seq} --------")
        img_name = '100_0.png'
        image = cv2.imread(os.path.join(image_path, seq, ))
        if image is None:
            print(f"[WARN] 读图失败: {seq}")
            continue

        for method_name, subfolder in methods.items():
            pose_file = os.path.join(pose_path, method_name, f"{seq}.txt")
            result_file = os.path.join(base_path, method_name, f"{seq}.txt")
            
            T_c2w = get_poses(pose_file, img_name)
            xyz_list = get_es_xyz(result_file)
            xy_dict = {}
            for name, xyz in xyz_list.items() :
                xyz = WGS84_to_ECEF(xyz)
                xyz = np.expand_dims(xyz, axis = 0)
                xy = get_points2D_ECEF_projection_v2(T_c2w, rcamera, xyz)
                xy_dict[name] = xy
            print(f"== Method: {method_name} ==")
            error_t = evaluate_xyz(result_file, gt_file)  # 必须返回 {key: error_value}

            if not isinstance(error_t, dict):
                print(f"[ERROR] evaluate_xyz 返回非字典：{type(error_t)}，跳过 {method_name}")
                continue

            vis_img = draw_error_points_on_image(image, xy_dict, error_t)
            out_dir = os.path.join("/mnt/sda/MapScape/query/estimation/position_result/error_vis", method_name, seq)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{seq}_error_vis.png")
            resized_img = cv2.resize(vis_img, (int(vis_img.shape[1] * 0.25), int(vis_img.shape[0] * 0.25)))
            cv2.imwrite(out_path, resized_img)
            print(f"[Saved] {out_path}")

                
if __name__ == "__main__":
    main()
            


