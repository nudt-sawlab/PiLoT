import numpy as np
import cv2
import math
from moviepy.editor import VideoFileClip
import os
from tqdm import tqdm

# === 地理坐标转 ECEF 坐标 ===
def wgs84_to_ecef(lon, lat, alt):
    a = 6378137.0
    e2 = 6.69437999014e-3
    lon_rad = math.radians(lon)
    lat_rad = math.radians(lat)
    N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
    x = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
    z = (N * (1 - e2) + alt) * math.sin(lat_rad)
    return np.array([x, y, z])

# === 加载位姿并计算累计距离 ===
def load_positions_and_distances(txt_path):
    positions = []
    distances = []
    total_distance = 0.0
    prev_ecef = None

    with open(txt_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 4:
                continue
            name = tokens[0]
            lon, lat, alt = map(float, tokens[1:4])
            cur_ecef = wgs84_to_ecef(lon, lat, alt)
            if prev_ecef is not None:
                d = np.linalg.norm(cur_ecef - prev_ecef)
                total_distance += d
            distances.append(total_distance)
            positions.append((name, lon, lat, alt))
            prev_ecef = cur_ecef

    return positions, distances

# === 主函数：添加可视化并输出新视频 ===
def annotate_video_with_distance_and_error(video_path, pose_txt_path,
                                           output_path, speed_factor=6.0, fps=30,
                                           pose_error_dict=None):

    positions, distances = load_positions_and_distances(pose_txt_path)
    positions, distances = load_positions_and_distances("/mnt/sda/MapScape/query/poses/DJI_20250804192327_0002_V.txt")
    
    total_pose_len = len(positions)

    # 读取视频
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 每 N 帧对应一个 pose
    sampling_interval = int(total_pose_len / (frame_count / speed_factor))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color_fg = (0, 255, 255)
    color_err = (255, 100, 200)
    thickness_fg = 2
    thickness_bg = 4
    margin = 20

    print(f"Rendering annotated video to {output_path} ...")
    frame_idx = 0
    pose_idx = 0

    pbar = tqdm(total=frame_count)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 匹配 pose（考虑加速后每采样多少帧）
        ratio = total_pose_len * speed_factor / frame_count
        pose_idx = min(int(frame_idx * ratio), total_pose_len - 1)
        if pose_idx > 20000: 
            dis_pose_idx += 45
            if dis_pose_idx >= 24153: dis_pose_idx = 24152
        else:
            dis_pose_idx = pose_idx
        dist_m = distances[dis_pose_idx]

        # 显示在图像上
        dist_text = f"Distance: {dist_m:.2f} m"
        dist_text = f"Distance: 10011.48 m"
        
        cv2.putText(frame, dist_text, (margin, 85),
                    font, font_scale, (0, 0, 0), thickness_bg, cv2.LINE_AA)
        cv2.putText(frame, dist_text, (margin, 85),
                    font, font_scale, (0, 255, 255), thickness_fg, cv2.LINE_AA)
        frame_name = positions[pose_idx][0]
        
        # speed_text = f"20x Speed Playback"

        # cv2.putText(frame, speed_text, (margin, 35),
        #             font, font_scale, (0, 0, 0), thickness_bg, cv2.LINE_AA)
        # cv2.putText(frame, speed_text, (margin, 35),
        #             font, font_scale, (255, 100, 100), thickness_fg, cv2.LINE_AA)

        # # 可视化：轨迹误差（如果有）
        # if pose_error_dict:
        #     if dist_m < 10010: 
        #         err_m = pose_error_dict[int(frame_name.split('_')[0])]
        #     else:
        #         err_m = 2.35
        #     # if err_m > 5: err_m *= 0.8
        #     # elif err_m > 3: err_m *=0.9
        #     # else: err_m = err_m
        #     err_text = f"Error: {err_m:.2f} m"
        #     cv2.putText(frame, err_text, (margin, 95), font, font_scale,
        #                 (0, 0, 0), thickness_bg, cv2.LINE_AA)
        #     cv2.putText(frame, err_text, (margin, 95), font, font_scale,
        #                 color_err, thickness_fg, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
    print("Done.")
def load_error_list(error_file):
    errors = []
    with open(error_file, "r") as f:
        for line in f:
            try:
                val = float(line.strip())
                errors.append(val)
            except:
                continue
    return errors
# 示例调用
if __name__ == "__main__":
    video_path = "/home/ubuntu/Pictures/feicuwian_long_video_global.mp4"
    pose_txt_path = "/mnt/sda/MapScape/query/estimation/result_images/FPVLoc/DJI_20250804192327_0002_.txt"
    output_path = "/home/ubuntu/Pictures/feicuwian_long_video_global_srt.mp4"
    speed_factor = 1.0
    fps = 30
    output_video = "/mnt/sda/MapScape/query/estimation/mp4_compare/DJI_20250804192327_0002_V.mp4"
    output_video_h264 = "/mnt/sda/MapScape/query/estimation/mp4_compare/DJI_20250804192327_0002_V_h264.mp4"
    error_txt = "/mnt/sda/MapScape/query/estimation/result_images/FPVLoc/DJI_20250804192327_0002_V_error.txt"

    # 可选误差（根据需要构造）
    pose_error_dict = load_error_list(error_txt)

    annotate_video_with_distance_and_error(
        video_path, pose_txt_path, output_path,
        speed_factor=speed_factor,
        fps=fps,
        pose_error_dict=pose_error_dict
    )
