import cv2
import numpy as np
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# （如果需要，可以取消下面的注释以使用 pfm 读取函数）
def read_pfm(file):
    """ 读取 PFM 文件并返回 (image, scale) """
    with open(file, 'rb') as f:
        header = f.readline().decode('utf-8').strip()
        if header not in ['PF', 'Pf']:
            raise Exception('Not a PFM file.')
        dim_line = f.readline().decode('utf-8').strip()
        width, height = map(int, dim_line.split())
        scale = float(f.readline().decode('utf-8').strip())
        color = (header == 'PF')
        data = np.fromfile(f, dtype=np.float32)
        if color:
            image = np.reshape(data, (height, width, 3))
        else:
            image = np.reshape(data, (height, width))
        return image, scale

# 可见光图像路径（示例）
color_pth = '/mnt/sda/MapScape/query/depth/switzerland_seq2@8@cloudy@100/0_1.png'
# 深度图路径（示例），此处用 .npy 作为演示
depth_pth = '/mnt/sda/MapScape/query/depth/switzerland_seq2@8@cloudy@100/0_1.png'
depth_npy_pth = '/home/ubuntu/Documents/code/FPVLoc/src_open/preprocess/datasets/Netherland_seq1/spain_seq3@300@30_50_new/0_1.png'

# --- 1. 读取可见光图像 ---
# 1. 读取可见光图 (OpenCV)
color_img = cv2.imread(color_pth, cv2.IMREAD_COLOR)
if color_img is None:
    raise IOError(f"无法读取可见光图：{color_pth}")

# 2. 读取深度图 (PFM)
ref_depth_image = cv2.imread(depth_pth, cv2.IMREAD_UNCHANGED)
depth_img, scale = read_pfm(depth_pth)
near_cm = 10
far_cm = 2000

z_cm = near_cm * far_cm / (far_cm - (far_cm - near_cm) * depth_img)
z_m  = z_cm / 100.0
# 3. 简单处理深度图：去除无效值、滤波
#    例如：将 65504 当作无效值标记，并做中值滤波。
depth_img[depth_img == 65504] = -1
depth_img = cv2.medianBlur(depth_img, 3)

# 4. min-max 归一化到 [0, 255]
d_min, d_max = np.min(depth_img), np.max(depth_img)
depth_norm = (depth_img - d_min) / (d_max - d_min + 1e-8)
depth_norm_8u = (depth_norm * 255).astype(np.uint8)

# 5. 使用伪彩色映射 (COLORMAP_JET)
depth_color = cv2.applyColorMap(depth_norm_8u, cv2.COLORMAP_JET)

# 6. 拼接：可见光图 (左) + 深度伪彩色图 (右)
combined = np.hstack((color_img, depth_color))

# 7. 使用 Matplotlib 显示，并加 colorbar
plt.figure(figsize=(10, 6))

# OpenCV 读出来的是 BGR 排列，需要转换到 RGB 以便 Matplotlib 正常显示
combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
plt.imshow(combined_rgb)
plt.title("Visible + Depth (Jet)")

# 8. 添加色柱（colorbar）
#    让它对应原始深度 [d_min, d_max]，并与 'jet' 色图匹配
norm = mpl.colors.Normalize(vmin=d_min, vmax=d_max)
cmap = plt.cm.get_cmap('jet')
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # 需要给一个空array，才可以正常生成colorbar

cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
cbar.set_label('Depth Value')

plt.axis("off")  # 去除坐标轴
plt.show()