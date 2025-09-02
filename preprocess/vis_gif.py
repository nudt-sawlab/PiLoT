import glob
from PIL import Image
import os
import re
def make_gif(image_folder: str,
             output_path: str,
             size: tuple,
             duration: int = 500,
             loop: int = 0):
    """
    :param image_folder: 图片序列所在文件夹
    :param output_path: 输出 GIF 路径
    :param size: 输出分辨率 (width, height)
    :param duration: 每帧时长（毫秒）
    :param loop: 循环次数，0 表示无限
    """
    def sort_key(img_path):
        # 提取文件名中"_0"之前的数字部分
        base_name = os.path.basename(img_path)
        match = re.search(r'(\d+)\_0', base_name)
        if match:
            return int(match.group(1))
        return 0
    all_files = sorted(os.listdir(image_folder))
    image_files = [f for f in all_files if '_0' in f]
    # 按自定义排序函数排序
    image_files.sort(key=sort_key)
    # files = sorted(glob.glob(f"{image_folder}/*.[pj][pn]g"))
    if not image_files:
        raise ValueError(f"No images found in {image_folder}")

    # 打开并统一缩放
    frames = []
    for f in image_files:
        img_path = os.path.join(image_folder, f)
        img = Image.open(img_path)
        img_resized = img.resize(size, Image.ANTIALIAS)
        frames.append(img_resized)

    # 保存 GIF
    frames[0].save(
        output_path,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=loop
    )
    print(f"生成 GIF：{output_path}，分辨率：{size[0]}×{size[1]}")

if __name__ == "__main__":
    # 设定输出 640×480
    image_path = "/home/ubuntu/Documents/code/github/FPVLoc_dev/outputs/Mapsape/switzerland_seq7@8@cloudy@500@VGG@LM20"
    output_gif = os.path.join(image_path, "output.gif")
    make_gif(image_path, output_gif, size=(480,270), duration=40, loop=0)
