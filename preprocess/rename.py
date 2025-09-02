import os
import re

def rename_images(image_folder):
    """
    将文件夹下所有图像文件名的前缀加上2500
    :param image_folder: 包含图像的文件夹路径
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(image_folder):
        # 检查文件名是否符合"数字_0.png"的格式
        match = re.match(r'^(\d+)_1\.png$', filename)
        if match:
            # 提取原始数字部分
            original_number = match.group(1)
            # 计算新的数字部分（加上2500）
            new_number = int(original_number) - 2080
            # 构造新的文件名
            new_filename = f"{new_number}_1.png"
            # 构造完整的文件路径
            original_path = os.path.join(image_folder, filename)
            new_path = os.path.join(image_folder, new_filename)
            # 重命名文件
            os.rename(original_path, new_path)
            print(f"文件已重命名：{filename} -> {new_filename}")

# 示例用法
image_folder = "/mnt/sda/MapScape/Train/Switzerland_seq32@300@0_30/ref"  # 替换为你的图像文件夹路径
rename_images(image_folder)

