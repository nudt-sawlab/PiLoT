import cv2
import numpy as np
import os

def save_triangle_crop(image, triangle_pts, save_path):
    """
    裁剪三角形区域并保存为PNG，保持透明背景。
    """
    mask = np.zeros_like(image[:, :, 0])  # 单通道掩膜
    cv2.fillConvexPoly(mask, np.array(triangle_pts, dtype=np.int32), 255)

    result = cv2.bitwise_and(image, image, mask=mask)

    # 设置透明背景
    b, g, r = cv2.split(result)
    alpha = mask
    result_rgba = cv2.merge([b, g, r, alpha])

    cv2.imwrite(save_path, result_rgba)
    print(f"[Saved] {save_path}")

def main():
    img_path = "/mnt/sda/MapScape/query/images/feicuiwan_sim_seq7/609_0.png"  # ← 原始图片路径
    out_dir = "/mnt/sda/MapScape/query/bbox/feicuiwan_sim_seq7"
    os.makedirs(out_dir, exist_ok=True)
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    H, W= image.shape[:2]
    # === 图像四个角点（图像坐标系，单位：像素）===
    # 顺序建议：左上、右上、右下、左下
    p0 = [0, 0]    # 左上
    p1 = [W, 0]    # 右上
    p2 = [W, H]   # 右下
    p3 = [0, H]   # 左下

    
    if image is None:
        print(f"Error: 图像加载失败 {img_path}")
        return

    # === 裁剪两个三角形 ===
    save_triangle_crop(image, [p0, p1, p3], os.path.join(out_dir, "trapezoid_tri1.png"))
    save_triangle_crop(image, [p1, p2, p3], os.path.join(out_dir, "trapezoid_tri2.png"))

if __name__ == "__main__":
    main()
