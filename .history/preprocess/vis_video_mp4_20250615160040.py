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
                           labels=('GT','FPVLoc','Pixloc','Render2loc'),
                           output_path=None,
                           target_size = (480, 270),
                           fps=20,
                           fourcc_str='mp4v'):   # 默认改为 H.264 的 FourCC):
    # 排序读取
    lists = [get_sorted_images(d) for d in seq_dirs]
    n = min(len(lst) for lst in lists)
    # 统一输出分辨率
    # 如果要保存视频，初始化 writer
    writer = None
    if output_path:
        W, H = target_size
        out_size = (W*2, H*2)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(output_path, fourcc, fps, out_size)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(n):
        frames = []
        # 读取、resize、加标注
        for idx, lst in enumerate(lists):
            img = cv2.imread(lst[i])
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            text = labels[idx]
            # 在左上角写字：字号0.8，白色带黑边
            cv2.putText(img, text, (10, 25), font, 0.8,
                        (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(img, text, (10, 25), font, 0.8,
                        (255,255,255), 2, cv2.LINE_AA)
            frames.append(img)
        # 拼 2×2
        top    = np.hstack((frames[0], frames[1]))
        bottom = np.hstack((frames[2], frames[3]))
        grid   = np.vstack((top, bottom))

        # 展示
        # cv2.imshow('Parallel View', grid)
        # if cv2.waitKey(int(1000/fps)) & 0xFF == 27:
        #     break
        # 写入
        if writer:
            writer.write(grid)

    if writer:
        writer.release()
        print(f'视频已保存到 {output_path}')
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
    sequence_name = "feicuiwan_seq5"
    sequences = [
        '/mnt/sda/CityofStars/Queries/process/video/seq5/images/seq5',
        '/mnt/sda/MapScape/query/estimation/result_images/FPVLoc/DJI_20250612194040_0013_V_900',
        '/mnt/sda/MapScape/query/estimation/result_images/FPVLoc/DJI_20250612194040_0013_V_900',
        '/home/ubuntu/Documents/code/github/Target2loc/datasets/翡翠湾/result_images'
    ]
    target_size = (480, 270)
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
