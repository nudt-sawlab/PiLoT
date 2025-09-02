import os

import numpy as np
import os
import argparse
import csv
import cv2
from transform import WGS84_to_ECEF
import os
import cv2
import numpy as np

def normalize_errors(err_vals, eps=1e-6):
    """将 error list 归一化到 [0, 1] 方便映射 colormap。"""
    arr = np.array(list(err_vals), dtype=np.float32)
    minv = np.nanmin(arr)
    maxv = np.nanmax(arr)
    if maxv - minv < eps:
        return {k: 0.5 for k in err_vals}  # 全部相同时用中间色
    norm = (arr - minv) / (maxv - minv)
    return {k: float(norm[i]) for i, k in enumerate(err_vals)}

def draw_error_points_on_image(image, xy_dict, error_t, 
                               point_radius=6, thickness=-1, 
                               colormap=cv2.COLORMAP_JET):
    """
    在 image 上根据 error_t 用颜色画点，返回带图例（colorbar）的图。
    - xy_dict: {key: (x, y)} 像素坐标
    - error_t: {key: error_value} 标量误差
    """
    img = image.copy()
    # 归一化 error 到 0-1
    norm_err = normalize_errors(error_t.values())
    # 映射 key->normalized
    key_to_norm = {k: norm_err[k] for k in error_t.keys()}

    # 先画点（用 colormap 映射）
    for key, (x, y) in xy_dict.items():
        if key not in error_t:
            continue
        err_norm = key_to_norm[key]
        # 0-255 映射
        cmap_val = int(np.round(err_norm * 255))
        color = cv2.applyColorMap(np.array([[cmap_val]], dtype=np.uint8), colormap)[0, 0].tolist()
        # OpenCV 用 BGR
        cv2.circle(img, (int(x), int(y)), point_radius, color, thickness, lineType=cv2.LINE_AA)

    # 画 colorbar：在右侧附加一条条带
    h, w = img.shape[:2]
    bar_h = int(h * 0.6)
    bar_w = 20
    bar_x = w - bar_w - 10
    bar_y = int((h - bar_h) / 2)
    # 创建一个 gradient
    gradient = np.linspace(255, 0, bar_h, dtype=np.uint8)[:, None]  # 从高 error（红）到低（蓝）
    gradient = np.repeat(gradient, bar_w, axis=1)  # shape (bar_h, bar_w)
    colored_bar = cv2.applyColorMap(gradient, colormap)  # (bar_h, bar_w, 3)
    # 把 colorbar 贴上去
    img[bar_y:bar_y+bar_h, bar_x:bar_x+bar_w] = colored_bar

    # 添加刻度和文字（最小/最大）
    font = cv2.FONT_HERSHEY_SIMPLEX
    min_err = min(error_t.values())
    max_err = max(error_t.values())
    # 上 label: max
    cv2.putText(img, f"{max_err:.3f}", (bar_x - 5, bar_y + 10), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
    # 下 label: min
    cv2.putText(img, f"{min_err:.3f}", (bar_x - 5, bar_y + bar_h), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
    # 中间写 “error”
    cv2.putText(img, "error", (bar_x - 5, bar_y + bar_h//2), font, 0.5, (255,255,255), 1, cv2.LINE_AA)

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

    results_list = []
    gt_path = os.path.join(base_path, "GT")
    render2orb_path = os.path.join(base_path, "ORB@per30")
    txt_files = [f for f in os.listdir(render2orb_path) if f.endswith('.txt') and 'DJI_' in f]

    # 主流程（补全部分）
    for txt_file in txt_files:
        # if 'DJI_20250612194622_0018_V' not in txt_file: continue
        seq = txt_file.split('.')[0]
        gt_file = os.path.join(gt_path, f"{seq}_RTK.txt")
        save_xy_path = os.path.join("/mnt/sda/MapScape/query/bbox", seq, seq + '_xy.txt')
        xy_dict = get_xy(save_xy_path)  # 期望返回 {key: (x, y)}
        print(f"\n-------- Sequence: {seq} --------")
        image = cv2.imread(os.path.join(image_path, seq, '0_0.png'))
        if image is None:
            print(f"[WARN] 读图失败: {seq}")
            continue

        for method_name, subfolder in methods.items():
            result_file = os.path.join(base_path, method_name, f"{seq}.txt")
            print(f"== Method: {method_name} ==")
            # 这里假设 evaluate_xyz 可以改成返回 per-key error 字典（如果当前版本返回的是字符串，需要你修改它）
            error_t = evaluate_xyz(result_file, gt_file)  # 应返回 {key: error_value}

            if not isinstance(error_t, dict):
                raise ValueError(f"evaluate_xyz 应返回字典格式的 per-point error，但得到 {type(error_t)}")

            # 画图
            vis_img = draw_error_points_on_image(image, xy_dict, error_t)
            # 保存
            out_dir = os.path.join("/mnt/sda/MapScape/query/estimation/position_result/error_vis", method_name, seq)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{seq}_error_vis.png")
            cv2.imwrite(out_path, vis_img)
            print(f"[Saved] {out_path}")

                
if __name__ == "__main__":
    main()
            


