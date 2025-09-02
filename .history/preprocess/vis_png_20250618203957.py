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
        # 若图像是 RGBA，转为 RGB（不叠加背景，仅查看原始值）
        if img.mode == 'RGBA':
            img_np = np.array(img)
            print("原始 RGBA 值示例（前5个像素）:")
            print(img_np.reshape(-1, 4)[:5])
        else:
            img_np = np.array(img.convert('RGB'))
            print("RGB 值示例（前5个像素）:")
            print(img_np.reshape(-1, 3)[:5])
        # 处理透明通道
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))  # 可改为黑色等其他背景
            img = Image.alpha_composite(background.convert('RGBA'), img).convert('RGB')
        else:
            img = img.convert('RGB')  # 确保为 RGB 模式

        # 保存为新的 PNG 图像
        img.save(output_path)
        print(f"Saved: {output_path}")
