import os
from PIL import Image

# 输入输出文件夹路径
input_dir = '/mnt/sda/MapScape/sup/目标检测测试数据/images'    # 替换为你的输入路径
output_dir = '/path/to/output_pngs'  # 替换为你的输出路径

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
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))  # 可改为黑色等其他背景
            img = Image.alpha_composite(background.convert('RGBA'), img).convert('RGB')
        else:
            img = img.convert('RGB')  # 确保为 RGB 模式

        # 保存为新的 PNG 图像
        img.save(output_path)
        print(f"Saved: {output_path}")
