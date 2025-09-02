import cv2
import os
import numpy as np
from transform import WGS84_to_ECEF
from scipy.spatial.transform import Rotation as R
from get_depth import get_3D_samples, get_points2D_ECEF_projection

from transform import WGS84_to_ECEF ,get_rotation_enu_in_ecef
from matplotlib import rcParams
# 配置
ref_dir = "/mnt/sda/MapScape/query/estimation/mp4_compare/DJI_20241113180128_0042_D/GT"
query_dir = "/mnt/sda/MapScape/query/estimation/mp4_compare/DJI_20241113180128_0042_D/Render2RAFT"
output_path = "/mnt/sda/MapScape/query/estimation/mp4_compare/DJI_20241113180128_0042_D/vis_video_with_error/Render2RAFT"
query_pose_file = os.path.join(query_dir, "DJI_20241113180128_0042_D_test.txt")
ref_pose_file = os.path.join(ref_dir, "DJI_20241113180128_0042_D_test.txt")
start_idx, end_idx = 740, 740
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
    return R_c2w
def generate_full_checkerboard(ref_img, query_img, rows, cols):
    h, w = ref_img.shape[:2]
    cell_h, cell_w = h // rows, w // cols
    output = np.zeros_like(ref_img)
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            if (r + c) % 2 == 0:
                output[y0:y1, x0:x1] = query_img[y0:y1, x0:x1]
            else:
                output[y0:y1, x0:x1] = ref_img[y0:y1, x0:x1]
    return output
def generate_checkerboard_overlay_transparent_v2(ref_crop, query_crop, rows, cols, alpha=0.7):
    """
    生成带透明度叠加的 checkerboard 图像。
    
    Parameters:
        ref_crop:   底图（numpy array）
        query_crop: 顶图（numpy array）
        rows:       网格行数
        cols:       网格列数
        alpha:      顶图透明度（0~1），越大越偏向 query_crop
    
    Returns:
        output:     合成后的 checkerboard 图像
    """
    h, w = ref_crop.shape[:2]
    cell_h, cell_w = h // rows, w // cols
    output = np.zeros_like(ref_crop, dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            ref_block = ref_crop[y0:y1, x0:x1].astype(np.float32)
            query_block = query_crop[y0:y1, x0:x1].astype(np.float32)
            if (r + c) % 2 == 0:
                mix = alpha * query_block + (1 - alpha) * ref_block
            else:
                mix = (1 - alpha) * query_block + alpha * ref_block
            output[y0:y1, x0:x1] = mix.astype(np.uint8)
    return output
def generate_checkerboard_overlay_transparent(ref_crop, query_crop, rows, cols, alpha=0.93):
    """
    生成带透明度叠加的 checkerboard 图像。
    
    Parameters:
        ref_crop:   底图（numpy array）
        query_crop: 顶图（numpy array）
        rows:       网格行数
        cols:       网格列数
        alpha:      顶图透明度（0~1），越大越偏向 query_crop
    
    Returns:
        output:     合成后的 checkerboard 图像
    """
    h, w = ref_crop.shape[:2]
    cell_h, cell_w = h // rows, w // cols
    output = np.zeros_like(ref_crop, dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            ref_block = ref_crop[y0:y1, x0:x1].astype(np.float32)
            query_block = query_crop[y0:y1, x0:x1].astype(np.float32)
            if (r + c) % 2 == 0:
                mix = (1 - alpha) * query_block + alpha * ref_block
            else:
                mix = query_block 
            output[y0:y1, x0:x1] = mix.astype(np.uint8)
    return output
# def generate_checkerboard_overlay(ref_crop, query_crop, rows, cols):
#     h, w = ref_crop.shape[:2]
#     cell_h, cell_w = h // rows, w // cols
#     output = np.zeros_like(ref_crop)
#     for r in range(rows):
#         for c in range(cols):
#             y0, y1 = r * cell_h, (r + 1) * cell_h
#             x0, x1 = c * cell_w, (c + 1) * cell_w
#             if (r + c) % 2 == 0:
#                 output[y0:y1, x0:x1] = query_crop[y0:y1, x0:x1]
#             else:
#                 output[y0:y1, x0:x1] = ref_crop[y0:y1, x0:x1]
#     return output

def draw_shadow_text(img, text, org, font_scale=1.0):
    x, y = org
    cv2.putText(img, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 1, cv2.LINE_AA)
def load_pose(file_path):
    xyz = []
    angles = []
    timestamps = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                name = parts[0]
                if "_" in name:
                    frame_idx = int(name.split("_")[0])
                lon, lat, alt = map(float, parts[1:4])
                roll, pitch, yaw = map(float, parts[4:7])
                xyz.append(WGS84_to_ECEF([lon, lat, alt]))
                e = [pitch, roll, yaw]
                t = [lon, lat, alt]
                R_c2w = euler_angles_to_matrix_ECEF_w2c(e, t)
                angles.append(R_c2w)
                # angles.append([pitch, roll, yaw])
                timestamps.append(frame_idx)
    return np.array(timestamps), np.array(xyz), np.array(angles)
# 输出尺寸（与输入图片保持一致）
W, H = 960, 540
crop_w, crop_h = 720, 405


x_off = (W - crop_w) // 2
y_off = (H - crop_h) // 2
# 棋盘格参数（行列数）
grid_rows, grid_cols = 3, 5  # 共 96 块格子

gt_timestamps, gt_xyz, gt_angles = load_pose(ref_pose_file)
es_timestamps, es_xyz, es_angles = load_pose(query_pose_file)


ang_err = []
pos_err = []
for i, fid in enumerate(es_timestamps):
    cos = np.clip((np.trace(np.dot(gt_angles[fid].T, es_angles[i])) - 1) / 2, -1., 1.)
    e_R = np.rad2deg(np.abs(np.arccos(cos)))
    ang_err.append(e_R)
    err = np.linalg.norm(es_xyz[i] - gt_xyz[fid])
    pos_err.append(err)

# 主循环
for i in range(start_idx, end_idx + 1):
    ref_path = os.path.join(ref_dir, f"{i}_0.png")
    query_path = os.path.join(query_dir, f"{i}_0.png")
    ref = cv2.imread(ref_path)
    qry = cv2.imread(query_path)
    if qry is None :
        print(f"[跳过] 帧 {i}")
        continue
    # -----

    #------
    ref = cv2.resize(ref, (W, H))
    qry = cv2.resize(qry, (W, H))

    # 棋盘格交错融合
    ref = cv2.resize(ref, (W, H))
    qry = cv2.resize(qry, (W, H))
    ref_crop = ref[y_off:y_off+crop_h, x_off:x_off+crop_w]
    qry_crop = qry[y_off:y_off+crop_h, x_off:x_off+crop_w]

    # === 构造棋盘格混合区域 ===
    checker_crop = generate_checkerboard_overlay_transparent(ref_crop, qry_crop, grid_rows, grid_cols)
    # checker_crop = generate_checkerboard_overlay(ref_crop, qry_crop, grid_rows, grid_cols)

    # === 放入 frame 中间 ===
    frame = qry.copy()
    frame[y_off:y_off+crop_h, x_off:x_off+crop_w] = checker_crop
    # === 叠加误差信息 ===
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    shadow_offset = 1
    # 获取当前帧误差
    if i in es_timestamps:
        idx = np.where(es_timestamps == i)[0][0]
        err_pos = pos_err[idx]
        err_ang = ang_err[idx]

        pos_str = f"Pos Error: {err_pos:.2f} m"
        ang_str = f"Angle Error: {err_ang:.2f} deg"

        # 位置1
        # 位置1
        pos_xy = (30, 40)
        ang_xy = (30, 80)

        # 阴影背景
        cv2.putText(frame, pos_str, (pos_xy[0]+shadow_offset, pos_xy[1]+shadow_offset),
                    font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        # cv2.putText(frame, ang_str, (ang_xy[0]+shadow_offset, ang_xy[1]+shadow_offset),
        #             font, font_scale, (0, 0, 0), thickness , cv2.LINE_AA)
        
        # 主文字
        cv2.putText(frame, pos_str, pos_xy, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        # cv2.putText(frame, ang_str, ang_xy, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        if i % 30 ==0 and ('ORB' in query_dir or 'RAFT' in query_dir):
            font_scale_key = 0.5
            thickness_key = 2
            color_key = (0, 255, 255)  # 明黄色
            text = "Keyframe"

            # 获取文字宽高
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale_key, thickness_key)

            # 右下角坐标（偏移 20 像素）
            x = frame.shape[1] - text_w - 20
            y = frame.shape[0] - 20

            # 绘制文字
            cv2.putText(frame, text, (x, y),
                        font, font_scale_key, color_key, thickness_key, cv2.LINE_AA)
    # === 可选：小图显示 Query 全图 ===
    # pip_w, pip_h = 320, 180
    # pip_x = W - pip_w - 20
    # pip_y = H - pip_h - 20
    # qry_pip = cv2.resize(qry, (pip_w, pip_h))
    # cv2.rectangle(frame, (pip_x-2, pip_y-2), (pip_x+pip_w+2, pip_y+pip_h+2), (255,255,255), 2)
    # frame[pip_y:pip_y+pip_h, pip_x:pip_x+pip_w] = qry_pip
    name = ref_dir.split('/')[-2]
    # cv2.imwrite(f'{output_path}/{i}_0.png', frame)
    # frame = cv2.resize(frame, (480, 270))
    cv2.imwrite(f'{output_path}/{i}_0.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f'{output_path}/{i}_{name}.png')


print(f"✅ 完成：视频已保存到 {output_path}")
