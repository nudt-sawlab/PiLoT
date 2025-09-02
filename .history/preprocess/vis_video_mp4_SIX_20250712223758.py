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
def parallel_video_display_six(seq_dirs,
                           labels = (
                               'GT (FPS: 25)',
                               'FPVLoc (FPS: 15)',
                               'Pixloc (FPS: 0.83)',
                               'Render2loc (FPS: 0.5)',
                               'Render2loc+RAFT (FPS: 10)',
                               'Render2loc+ORBSLAM3 (FPS: 25)'
                           ),
                           output_path=None,
                           target_size = (480, 270),
                           fps=20,
                           fourcc_str='mp4v'):
    # 排序读取
    lists = [get_sorted_images(d) for d in seq_dirs]
    n = len(lists[0])  # 以第一个序列的长度为准
    W, H = target_size
    out_size = (W * 3, H * 2)

    # 初始化 writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(output_path, fourcc, fps, out_size)

    font = cv2.FONT_HERSHEY_SIMPLEX

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

        # 2 × 3 拼接
        row1 = np.hstack((frames[0], frames[1], frames[2]))
        row2 = np.hstack((frames[3], frames[4], frames[5]))
        grid = np.vstack((row1, row2))

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
    # sequence_name = "USA_seq5@8@sunset@300"
    # sequences = [
    #     '/mnt/sda/MapScape/query/images/'+sequence_name,
    #     '/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/outputs/Mapsape/'+sequence_name,
    #     '/home/ubuntu/Documents/code/github/FPVLoc_dev/outputs/Mapsape/'+sequence_name+'@Pixloc',
    #     '/home/ubuntu/Documents/code/github/Target2loc_tan/fast_render2loc/datasets/switzerland/'+sequence_name+'/result_images',
    # ]
    # sequence_name = "DJI_20250612194040_0013_V"
    # sequences = [
    #     '/mnt/sda/MapScape/query/images/'+sequence_name,
    #     '/mnt/sda/MapScape/query/estimation/result_images/FPVLoc/'+sequence_name,
    #     '/mnt/sda/MapScape/query/estimation/result_images/Pixloc/'+sequence_name,
    #     '/home/ubuntu/Documents/code/github/Target2loc/datasets/翡翠湾/result_images'
    # ]
    sequence_name = "DJI_20250612194150_0014_V"
    # sequences = [
    #     '/mnt/sda/MapScape/query/images/'+sequence_name,
    #     '/mnt/sda/MapScape/query/estimation/result_images/FPVLoc/'+sequence_name,
    #     '/mnt/sda/MapScape/query/estimation/result_images/FPVLoc/'+sequence_name,
    #     '/mnt/sda/MapScape/query/estimation/result_images/FPVLoc/'+sequence_name
    # ]
    sequences = [
        '/media/ubuntu/PS20001/images/USA_seq5@8@foggy@500-400@intensity3@500',                      # GT
        '/mnt/sda/MapScape/query/estimation/result_images/FPVLoc/USA_seq5@8@sunny@500-400@500',  # FPVLoc
        '/mnt/sda/MapScape/query/estimation/result_images/Pixloc/USA_seq5@8@night@500-400@intensity3@500',                # Pixloc
        '/mnt/sda/MapScape/query/estimation/result_images/Render2loc@raft/a',       # Render2loc+RAFT
        '/mnt/sda/MapScape/query/estimation/result_images/Render2loc/USA_seq5@8@sunset@500-400@500',       # Render2loc
        '/mnt/sda/MapScape/query/estimation/result_images/ORB@per30/a'                  # ORBSLAM
    ]
    target_size = (512, 288)
    video_path = "/mnt/sda/MapScape/query/estimation/mp4_compare/"
    output_path = os.path.join(video_path, sequence_name+".mp4")
    parallel_video_display(sequences, output_path=output_path, target_size=target_size, fps=25)
    transcode_to_h264(output_path, os.path.join(video_path, sequence_name+"@h264.mp4"))
    print('save in ', output_path)
    # import ffmpeg
    # webm_path = os.path.join(video_path, "switzerland_seq7@8@cloudy@500.webm")
    # (
    # ffmpeg
    # .input(output_path)
    # .output(
    #     webm_path,
    #     vcodec='libvpx-vp9',
    #     crf=30,
    #     **{'b:v': '0', 'b:a': '64k'},
    #     acodec='libopus'
    # )
    # .run(overwrite_output=True)
# )
