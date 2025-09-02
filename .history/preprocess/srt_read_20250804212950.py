import re
import math

# ==== 配置 ====
srt_file = '/mnt/sda/MapScape/sup/feicuiwan_long_video/DJI_20250804192327_0002_V.srt'
output_txt = '/mnt/sda/MapScape/query/poses/DJI_20250804192327_0002_V.txt'

# ==== 时间段和帧率 ====
start_sec = 0
fps = 29.97
target_frame_count = 100

start_frame = math.floor(start_sec * fps)

# ==== 解析 SRT 并收集位姿 ====
pose_dict = {}
with open(srt_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(len(lines)):
    line = lines[i]
    if line.startswith("FrameCnt:"):
        frame_id_match = re.search(r"FrameCnt: (\d+)", line)
        if not frame_id_match:
            continue
        frame_id = int(frame_id_match.group(1))

        info_line = lines[i + 1] if i + 1 < len(lines) else ""

        lat_match = re.search(r"latitude:\s*([-\d.]+)", info_line)
        lon_match = re.search(r"longitude:\s*([-\d.]+)", info_line)
        alt_match = re.search(r"abs_alt:\s*([-\d.]+)", info_line)
        roll_match  = re.search(r"gb_roll:\s*([-\d.]+)", info_line)
        pitch_match = re.search(r"gb_pitch:\s*([-\d.]+)", info_line)
        yaw_match   = re.search(r"gb_yaw:\s*([-\d.]+)", info_line)

        if all([lat_match, lon_match, alt_match, roll_match, pitch_match, yaw_match]):
            lat = float(lat_match.group(1))
            lon = float(lon_match.group(1))
            alt = float(alt_match.group(1))
            roll  = float(roll_match.group(1))
            pitch = float(pitch_match.group(1)) + 90
            yaw   = float(yaw_match.group(1))
            yaw = -yaw
            pose_dict[frame_id] = [lon, lat, alt, roll, pitch, yaw]

# ==== 收集从 start_frame 开始往后的 900 个有效帧 ====
sorted_ids = sorted(fid for fid in pose_dict if fid >= start_frame)

if len(sorted_ids) < target_frame_count:
    print(f"❌ 错误：从 frame {start_frame} 开始只找到 {len(sorted_ids)} 个有效帧，无法满足 900 帧")
    exit()

selected_ids = sorted_ids[:target_frame_count]

# ==== 写入 TXT ====
with open(output_txt, 'w') as f:
    for i, fid in enumerate(selected_ids):
        name = f"{i}_0.png"
        values = pose_dict[fid]
        f.write(f"{name} {' '.join(map(str, values))}\n")

print(f"✅ 成功提取 {target_frame_count} 帧位姿数据（起点：frame {start_frame}），保存至 {output_txt}")
