import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def read_pfm(file_path):
    """ 读取 PFM 文件并返回 (image, scale) """
    with open(file_path, 'rb') as f:
        header = f.readline().decode('utf-8').strip()
        if header not in ['PF', 'Pf']:
            raise IOError('不是有效的 PFM 文件。')

        # 读取图像宽高
        dim_line = f.readline().decode('utf-8').strip()
        width, height = map(int, dim_line.split())

        # 读取 scale（若为负值代表存储顺序为小端在前）
        scale_line = f.readline().decode('utf-8').strip()
        scale = float(scale_line)
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)

        # 读取图像数据
        data = np.fromfile(f, endian + 'f')
        if header == 'PF':
            # 彩色图 (height, width, 3)
            image = np.reshape(data, (height, width, 3))
        else:
            # 灰度图 (height, width)
            image = np.reshape(data, (height, width))

        # PFM 通常是从左下角到右上角，这里根据实际需要翻转
        # 注：若读出的图像上下颠倒，可在此处 image = np.flipud(image)
        return image, scale

def resize_image_keep_ratio(img, max_size=256):
    """
    将图像等比例缩放到最大边不超过 max_size。
    支持单通道或多通道的 numpy 数组。
    """
    h, w = img.shape[:2]  # 对于单通道和多通道都适用
    if max(h, w) > max_size:
        # 计算缩放比例
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        # 使用INTER_AREA插值在缩小时效果较好
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized
    else:
        return img
def write_pfm(file_path, image, scale=1.0):
    """
    将浮点图像写入 PFM 文件。
      - image: (H,W) 或 (H,W,3) 的浮点 numpy 数组
      - scale: 输出文件的 scale 值，默认1.0即可
               若实际写入时检测到系统是 little-endian，会将 scale 写为负值
    """
    if image.dtype != np.float32:
        # 若不是 float32, 先转一下
        image = image.astype(np.float32)

    if len(image.shape) == 2:
        color = False
        height, width = image.shape
    elif len(image.shape) == 3 and image.shape[2] == 3:
        color = True
        height, width, _ = image.shape
    else:
        raise ValueError("write_pfm only supports HxW or HxWx3 float images.")

    # 判断系统大小端
    import sys
    byteorder = sys.byteorder
    if byteorder == 'little':
        scale = -scale  # 小端则写负值
    mode = 'PF\n' if color else 'Pf\n'

    with open(file_path, 'wb') as f:
        # 写header
        f.write(mode.encode('utf-8'))
        f.write(f"{width} {height}\n".encode('utf-8'))
        f.write(f"{scale}\n".encode('utf-8'))
        # 如果需要 flip: image = np.flipud(image)
        image.tofile(f)
# ---------------------- #
#      主流程示例        #
# ---------------------- #
if __name__ == "__main__":
    # 请根据实际情况修改以下路径
    color_pth = "/home/ubuntu/Documents/code/FPVLoc/src_open/preprocess/datasets/Netherland_seq1/Switzerland/193_0.png"
    depth_pth = "/home/ubuntu/Documents/code/FPVLoc/src_open/preprocess/datasets/Netherland_seq1/Switzerland/193_1.png"

    # 设定输出的文件路径（自行定义你想要的存储位置和文件名）
    color_out_path = "/home/ubuntu/Documents/code/FPVLoc/src_open/preprocess/datasets/Netherland_seq1/Switzerland/0_resized_color.png"
    depth_out_path = "/home/ubuntu/Documents/code/FPVLoc/src_open/preprocess/datasets/Netherland_seq1/Switzerland/0_resized_depth.png"

    # 1. 读取可见光图像（OpenCV 默认 BGR）
    color_img = cv2.imread(color_pth, cv2.IMREAD_COLOR)
    if color_img is None:
        raise IOError(f"无法读取可见光图：{color_pth}")

    # 2. 读取深度图（PFM）
    ref_depth_image = cv2.imread(depth_pth, cv2.IMREAD_UNCHANGED)
    depth_img, scale = read_pfm(depth_pth)
    # depth_img 此时通常是 float 型，单通道 (H, W)

    # 3. 简单处理深度图：剔除无效值、滤波等
    #   （示例：将65504视作无效值）
    depth_img[depth_img == 65504] = -1
    #   中值滤波，需要先将 -1 替换为 0 或者合理值，否则滤波结果可能包含很多无效值
    depth_img[depth_img < 0] = 0
    depth_img = cv2.medianBlur(depth_img.astype(np.float32), 3)

    # 4. 等比例缩放：可见光图、深度图都缩放到最大边 <= 256
    color_resized = resize_image_keep_ratio(color_img, max_size=256)
    depth_resized = resize_image_keep_ratio(depth_img, max_size=256)

    # 5.2 保存深度图 (PFM，保留浮点数据；不做任何 min-max 归一化)
    #    注意：在pfm中不会“改变值”，但插值本身会丢失空间分辨率
    cv2.imwrite(color_out_path, color_resized)
    write_pfm(depth_out_path, depth_resized, scale=1.0)

    print(f"已将可见光图保存至: {color_out_path}")
    print(f"已将深度图(PFM)保存至: {depth_out_path}")

    # =============== 6. 重新加载已保存的缩放后 PFM 并可视化 ===============
    #    (验证一下读写是否正确)
    depth_reload, _ = read_pfm(depth_out_path)
    resize_depth_image = cv2.imread(depth_out_path, cv2.IMREAD_UNCHANGED)
    # 如果想可视化，可以做一个简单的伪彩色展示
    # 但注意，伪彩色显示会做 0~255 的转换，仅用于可视化
    dmin, dmax = depth_reload.min(), depth_reload.max()

    # 构造一个可视化用的 8位图
    depth_vis = (np.clip(depth_reload, dmin, dmax) - dmin) / (dmax - dmin + 1e-8)
    depth_vis_8u = (depth_vis * 255).astype(np.uint8)
    depth_vis_color = cv2.applyColorMap(depth_vis_8u, cv2.COLORMAP_JET)

    # 将 color_resized 与 depth_vis_color 拼接可视化
    # 先匹配尺寸
    h1, w1 = color_resized.shape[:2]
    h2, w2 = depth_vis_color.shape[:2]
    if (h1 != h2) or (w1 != w2):
        depth_vis_color = cv2.resize(depth_vis_color, (w1, h1), interpolation=cv2.INTER_LINEAR)

    combined = np.hstack((color_resized, depth_vis_color))

    # Matplotlib 显示 (BGR -> RGB)
    combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    plt.imshow(combined_rgb)
    plt.title("Resized Color (Left) + Resized Depth (Right) [Jet Vis]")
    plt.axis("off")

    # 添加 colorbar 显示深度数值范围
    norm = mpl.colors.Normalize(vmin=dmin, vmax=dmax)
    cmap = plt.cm.get_cmap('jet')
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
    cbar.set_label("Depth Value (Float)")

    plt.show()
