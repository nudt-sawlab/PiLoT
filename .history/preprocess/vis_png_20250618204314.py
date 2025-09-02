import os
from PIL import Image
import numpy as np
# 输入输出文件夹路径
input_dir = '/mnt/sda/MapScape/sup/目标检测测试数据/images'    # 替换为你的输入路径
output_dir = '/mnt/sda/MapScape/sup/目标检测测试数据/images_refined'  # 替换为你的输出路径

# 创建输出文件夹（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 遍历所有 PNG 文件
for filename in os.listdir(input_dir):
    if filename.lower().endswith('.png'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 打开图像
        img = Image.open(input_path)

        # 处理透明通道
        img_rgb = img.convert('RGB')
        img_rgb.save(output_path)
        
        print(f"Saved: {output_path}")
