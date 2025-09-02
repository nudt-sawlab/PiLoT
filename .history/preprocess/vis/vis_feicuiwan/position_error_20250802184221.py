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
def to_grayscale_bg(image, descale=True):
    """把原始图转为黑白底图，保留对比，返回3通道 BGR 用于叠加。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 可选做一点局部对比增强：自适应直方图均衡
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    if descale:
        # 轻微降低亮度/对比避免抢占注意力
        enhanced = cv2.normalize(enhanced, None, alpha=30, beta=220, norm_type=cv2.NORM_MINMAX)
    bw = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return bw
def normalize_error_dict_fixed_range(error_dict, min_clip=0.0, max_clip=5.0, eps=1e-6):
    """
    固定把 error 截断在 [min_clip, max_clip] 后归一化到 [0,1]。
    超出范围的都剪切，避免极端值拉扯配色。
    """
    norm_dict = {}
    if not error_dict:
        return norm_dict
    for k, v in error_dict.items():
        if not np.isfinite(v):
            continue
        clipped = np.clip(v, min_clip, max_clip)
        norm = (clipped - min_clip) / (max_clip - min_clip) if (max_clip - min_clip) > eps else 0.5
        norm_dict[k] = float(norm)
    # 对于非 finite 的保底 0（或你想用别的）
    for k, v in error_dict.items():
        if k not in norm_dict:
            norm_dict[k] = 0.0
    return norm_dict
def draw_error_points_on_image(image, xy_dict, error_t,
                               point_radius=10, thickness=-1,
                               colormap=cv2.COLORMAP_TURBO):
    """
    黑白底 + 误差点（大在上）+ 美化 colorbar（固定 clip 0~5m）+ inf 用空心 + 缺失 xy 提示
    全部 OpenCV 调用用位置参数，避免 new style getargs format 错误。
    """
    img = to_grayscale_bg(image)
    font = cv2.FONT_HERSHEY_DUPLEX

    if not error_t or not isinstance(error_t, dict) or len(error_t) == 0:
        cv2.putText(img, "No valid predictions", (10, 30), font, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        return img

    # 归一化（固定 0~5m）
    norm_err = normalize_error_dict_fixed_range(error_t, min_clip=0.0, max_clip=5.0)

    # 先画小误差、后画大误差（大误差在顶层），inf 视为最大
    sorted_keys = sorted(error_t.keys(), key=lambda k: error_t[k] if np.isfinite(error_t[k]) else float('inf'))

    missing_in_xy = []
    for key in sorted_keys:
        if key not in xy_dict:
            missing_in_xy.append(key)
            continue
        coord = xy_dict[key]
        if len(coord) == 0:
            continue
        x, y = coord[0]
        if not np.isfinite(error_t[key]):
            cv2.circle(img, (int(x), int(y)), point_radius, (180, 180, 180), 2, cv2.LINE_AA)
            cv2.putText(img, "?", (int(x) + point_radius, int(y) - point_radius), font, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
            continue
        err_norm = norm_err.get(key, 0.0)
        cmap_val = int(np.round(err_norm * 255))
        color = cv2.applyColorMap(np.array([[cmap_val]], dtype=np.uint8), colormap)[0, 0].tolist()
        # 外发光
        glow_radius = point_radius + 5
        overlay = img.copy()
        cv2.circle(overlay, (int(x), int(y)), glow_radius, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        cv2.circle(img, (int(x), int(y)), point_radius, color, thickness, cv2.LINE_AA)

    # 缺失 xy 提示
    if missing_in_xy:
        sample = missing_in_xy[:5]
        txt = "Missing XY: " + ",".join(sample)
        cv2.rectangle(img, (5, img.shape[0] - 60), (img.shape[1] // 2, img.shape[0] - 5), (0, 0, 0), -1)
        cv2.putText(img, txt, (10, img.shape[0] - 30), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        if len(missing_in_xy) > 5:
            more = f"+{len(missing_in_xy) - 5} more"
            cv2.putText(img, more, (10, img.shape[0] - 12), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    # ---- colorbar（始终画） ----
    h, w = img.shape[:2]
    bar_h = int(h * 0.6)
    bar_w = 40  # 更粗一些
    
    bar_x = w - bar_w - 500
    bar_y = int((h - bar_h) / 2)

    # 梯度从 5m (上) 到 0m (下)
    gradient = np.linspace(255, 0, bar_h, dtype=np.uint8)[:, None]
    gradient = np.repeat(gradient, bar_w, axis=1)
    colored_bar = cv2.applyColorMap(gradient, colormap)

    # 背景板（增强对比）
    pad = 10

    # 贴 colorbar 本体
    img[bar_y:bar_y + bar_h, bar_x:bar_x + bar_w] = colored_bar
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (220, 220, 220), 1)

    # 标题和说明
    cv2.putText(img, "Error", (bar_x, bar_y - 25), font, 0.6, (240, 240, 240), 1)
    cv2.putText(img, "clipped to [0,5]m", (bar_x, bar_y - 8), font, 0.35, (200, 200, 200), 1)

    # 映射函数：value 到 y 位置（5m 在上，0m 在下）
    def val_to_y(v):
        t = (5.0 - v) / 5.0  # 5 -> 0, 0 -> 1
        return int(bar_y + np.clip(t, 0, 1) * bar_h)

    # 主刻度：5.0, 2.5, 0.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 5  # 放大字体
    thickness = 10  # 文字粗一点，去掉描边不要再叠加别的颜色
    major_ticks = [5.0, 2.5, 0.0]

    for v in major_ticks:
        y_pos = val_to_y(v)
        # 主刻度线（长）
        cv2.line(img, (bar_x + bar_w + 2, y_pos), (bar_x + bar_w + 18, y_pos), (220, 220, 220), 2)
        label = f"{v:.1f}m"
        (txt_w, txt_h), baseline = cv2.getTextSize(label, font, scale, thickness)
        label_x = bar_x + bar_w + 20
        label_y = y_pos + txt_h // 2

        # 背景块（留一点 padding）
        pad_x = 4
        pad_y = 2
        top_left = (label_x - pad_x, label_y - txt_h - pad_y)
        bottom_right = (label_x + txt_w + pad_x, label_y + pad_y)
        # cv2.rectangle(img, top_left, bottom_right, (30, 30, 30), -1)

        # 只画一次白色文字（无黑边），开抗锯齿
        cv2.putText(img, label, (label_x, label_y), font, scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    # 副刻度：4,3,1 米（短线）
    for v in [4.0, 3.0, 1.0]:
        y_pos = val_to_y(v)
        cv2.line(img, (bar_x + bar_w + 2, y_pos), (bar_x + bar_w + 10, y_pos), (180, 180, 180), 1)
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
                if parts[0] == img_name:
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
        img_name = '447_0.png'
        image = cv2.imread(os.path.join(image_path, seq, img_name))
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
            out_dir = "/mnt/sda/MapScape/query/estimation/position_result/error_vis"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{method_name}_{seq}_error_vis.png")
            resized_img = cv2.resize(vis_img, (int(vis_img.shape[1] * 0.25), int(vis_img.shape[0] * 0.25)))
            cv2.imwrite(out_path, resized_img)
            print(f"[Saved] {out_path}")
            print('-')

                
if __name__ == "__main__":
    main()
            


