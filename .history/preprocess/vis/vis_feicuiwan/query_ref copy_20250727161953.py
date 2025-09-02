import os
import re
import cv2
import numpy as np
import subprocess
import os
import re
import glob
import cv2
import numpy as np
def generate_fov_traj_frames(txt_path,
                              dist_forward=5.0,
                              width=4.0,
                              height=4.0,
                              size=(240, 240)) -> list:
    """
    生成轨迹视锥图像序列，返回 PIL 图像列表
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image

    import math
    import numpy as np

    # 1. 读取轨迹数据
    data_list = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                filename = parts[0]
                lon = float(parts[1])
                lat = float(parts[2])
                alt = float(parts[3])
                roll = float(parts[4])
                pitch = float(parts[5])
                yaw = float(parts[6])
                data_list.append((lon, lat, alt, roll, pitch, yaw))

    if not data_list:
        print("❌ 轨迹文件为空或格式错误")
        return []

    # 2. 原点 (ENU 坐标系参考点)
    lon0, lat0, alt0, _, _, _ = data_list[0]

    # 3. 转为 ENU
    enu_points = []
    for (lon, lat, alt, roll, pitch, yaw) in data_list:
        R_EARTH_LAT = 111320.0
        avg_lat_rad = math.radians(lat0)
        east = (lon - lon0) * R_EARTH_LAT * math.cos(avg_lat_rad)
        north = (lat - lat0) * R_EARTH_LAT
        up = alt - alt0
        enu_points.append((east, north, up, roll, pitch, yaw))

    traj_xyz = np.array([[x, y, z] for x, y, z, _, _, _ in enu_points])
    images = []

    # 4. 每一帧绘图
    for i, (x, y, z, roll, pitch, yaw) in enumerate(enu_points):
        fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()
        ax.view_init(elev=70, azim=130)

        # 轨迹灰线
        ax.plot(traj_xyz[:, 0], traj_xyz[:, 1], traj_xyz[:, 2], color='gray', linewidth=1)

        # 计算视锥
        from scipy.spatial.transform import Rotation as R

        def camera_direction(yaw_deg, pitch_deg, roll_deg=0.0):
            pitch_eff = 90.0 - pitch_deg
            yaw_eff = yaw_deg - 90.0
            rot = R.from_euler('ZYX', [yaw_eff, pitch_eff, roll_deg], degrees=True)
            return rot.apply([1, 0, 0])

        dir_vec = camera_direction(yaw, pitch, roll)
        apex = np.array([x, y, z])
        center = apex + dist_forward * dir_vec

        # 构建底面四个点
        not_collinear = np.array([0, 0, 1])
        if abs(np.dot(dir_vec, not_collinear)) > 0.99:
            not_collinear = np.array([1, 0, 0])
        u = np.cross(dir_vec, not_collinear); u /= np.linalg.norm(u)
        v = np.cross(dir_vec, u); v /= np.linalg.norm(v)
        half_w = width / 2.0
        half_h = height / 2.0

        base_pts = [
            center + half_w*u + half_h*v,
            center + half_w*u - half_h*v,
            center - half_w*u - half_h*v,
            center - half_w*u + half_h*v
        ]

        # 画视锥边线
        for pt in base_pts:
            ax.plot([apex[0], pt[0]], [apex[1], pt[1]], [apex[2], pt[2]], 'r-')
        for j in range(4):
            a, b = base_pts[j], base_pts[(j+1)%4]
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], 'b-')

        # 设置轨迹范围
        margin = 0.1
        min_x, max_x = np.min(traj_xyz[:, 0]), np.max(traj_xyz[:, 0])
        min_y, max_y = np.min(traj_xyz[:, 1]), np.max(traj_xyz[:, 1])
        min_z, max_z = np.min(traj_xyz[:, 2]), np.max(traj_xyz[:, 2])
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        ax.set_zlim(min_z - margin, max_z + margin)

        # 转换为图像
        canvas = FigureCanvas(fig)
        buf = BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        images.append(img)
        plt.close(fig)

    return images
def get_sorted_images(folder):
    imgs = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.png')) and "init" not in f
    ]
    def sort_key(p):
        name = os.path.basename(p)
        m = re.search(r'(\d+)', name)
        return int(m.group(1)) if m else -1
    imgs.sort(key=sort_key)
    return imgs

def parallel_video_display(seq_dirs,
                           labels = (
    'GT',
    'FPVLoc  FPS: 15',
    'Pixloc  FPS: 0.83',
    'Render2loc  FPS: 0.5'
),
                           output_path=None,
                           target_size = (480, 270),
                           fps=20,
                           fourcc_str='mp4v'):   # 默认改为 H.264 的 FourCC):
    # 排序读取
    lists = [get_sorted_images(d) for d in seq_dirs]
    n = len(lists[0])  # 以第一个序列的长度为准
    W, H = target_size
    out_size = (W * 2, H * 2)

    # 初始化 writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(output_path, fourcc, fps, out_size)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # 构造“无信号”画面
    def make_no_signal_frame(text="NO SIGNAL", size=(W, H)):
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        cv2.putText(img, text, (size[0]//4, size[1]//2),
                    font, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        return img

    for i in range(n):
        frames = []
        for idx, lst in enumerate(lists):
            if i < len(lst):
                img = cv2.imread(lst[i])
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            else:
                img = make_no_signal_frame()

            label = labels[idx]
            # 添加标签文字
            cv2.putText(img, label, (10, 25), font, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, label, (10, 25), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            frames.append(img)

        # 拼接为 2×2 格式
        top = np.hstack((frames[0], frames[1]))
        bottom = np.hstack((frames[2], frames[3]))
        grid = np.vstack((top, bottom))

        if writer:
            writer.write(grid)

    if writer:
        writer.release()
        print(f'✅ 视频已保存到 {output_path}')

    cv2.destroyAllWindows()
def transcode_to_h264(input_path: str, output_path: str,
                      crf: int = 23, preset: str = 'slow'):
    """
    使用系统 ffmpeg 将 MP4 转码为 H.264 编码的 MP4。
    :param input_path: 原始视频文件路径
    :param output_path: 转码后输出文件路径
    :param crf: 压缩质量因子（0–51，越大越小越差）
    :param preset: 编码预设（ultrafast…veryslow）
    """
    cmd = [
        'ffmpeg',
        '-y',                 # 若存在则覆盖
        '-i', input_path,     # 输入文件
        '-c:v', 'libx264',    # 视频编码 H.264
        '-preset', preset,    # 编码速度/压缩率
        '-crf', str(crf),     # 质量因子
        '-c:a', 'copy',       # 音频直接复制
        output_path           # 输出文件
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':

    sequence_name = "DJI_20250612194903_0021_V"
    sequences = [
        '/mnt/sda/MapScape/query/estimation/result_images/feicuiwan/Query/'+sequence_name,
        '/mnt/sda/MapScape/query/estimation/result_images/feicuiwan/GeoPixel/'+sequence_name,
        '/mnt/sda/MapScape/query/estimation/result_images/feicuiwan/PixLoc/'+sequence_name,
        '/mnt/sda/MapScape/query/estimation/result_images/feicuiwan/Render2Loc/'+sequence_name,
        '/mnt/sda/MapScape/query/estimation/result_images/feicuiwan/Render2ORB/'+sequence_name,
        '/mnt/sda/MapScape/query/estimation/result_images/feicuiwan/Render2RAFT/'+sequence_name
    ]
    target_size = (480, 270)
    video_path = "/mnt/sda/MapScape/query/estimation/mp4_compare/"
    output_path = os.path.join(video_path, sequence_name+".mp4")
    parallel_video_display(sequences, output_path=output_path, target_size=target_size, fps=25)
