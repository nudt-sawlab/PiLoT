import os
from PIL import Image
import numpy as np
import cv2
def read_intrinsics(camera):
    """
    计算35mm等效焦距和内参矩阵。
    
    参数:
    image_width_px -- 图像的宽度（像素）
    image_height_px -- 图像的高度（像素）
    sensor_width_mm -- 相机传感器的宽度（毫米）
    sensor_height_mm -- 相机传感器的高度（毫米）
    
    返回:
    K -- 内参矩阵，形状为3x3
    """
    print("DEBUG: camera", camera)
    if len(camera) == 5:
        image_width_px, image_height_px, sensor_width_mm, sensor_height_mm, f_mm = camera
        # 计算内参矩阵中的焦距
        focal_ratio_x = f_mm / sensor_width_mm
        focal_ratio_y = f_mm / sensor_height_mm
        
        fx = image_width_px * focal_ratio_x
        fy = image_height_px * focal_ratio_y

        # 计算主点坐标
        cx = image_width_px / 2
        cy = image_height_px / 2
    elif len(camera) == 7:
        image_width_px, image_height_px, cx, cy, sensor_width_mm, sensor_height_mm, f_mm = camera
        # 计算内参矩阵中的焦距
        focal_ratio_x = f_mm / sensor_width_mm
        focal_ratio_y = f_mm / sensor_height_mm
        
        fx = image_width_px * focal_ratio_x
        fy = image_height_px * focal_ratio_y
        
    # 构建内参矩阵 K
    K = [[fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]]
    
    return np.array(K), image_width_px, image_height_px

def make_pairwise_gif(folder1, folder2, output_dir, size=(480, 270), duration=500, loop=0):
    os.makedirs(output_dir, exist_ok=True)
    files1 = {os.path.splitext(f)[0]: f for f in os.listdir(folder1) if f.endswith('.png')}
    files2 = {os.path.splitext(f)[0]: f for f in os.listdir(folder2) if f.endswith('.jpg')}

    # 找到主干名相同的文件对
    common_keys = sorted(set(files1.keys()) & set(files2.keys()))
    if not common_keys:
        raise ValueError("两个文件夹中没有找到同名主干的 PNG/JPG 文件")

    for key in common_keys:
        fname1 = files1[key]  # png 文件名
        fname2 = files2[key]  # jpg 文件名
        # img1 = Image.open(os.path.join(folder1, fname)).resize(size, Image.Resampling.LANCZOS)
        # img2 = Image.open(os.path.join(folder2, fname)).resize(size, Image.Resampling.LANCZOS)
        img1_path = os.path.join(folder1, fname1)
        img1 = cv2.imread(img1_path)
        img2_path = os.path.join(folder2, fname2)
        img2 = cv2.imread(img2_path)
        img2 = cv2.resize(img2, size)
        
        kp = [0.0046,
            0.1294,
            -0.00021808,
            0.0012,
            -0.2037]
        query_camera = [3840,
                2160,
                1922.2,
                1070.0,
                3840,
                2160,
                2815.0]
        kp = np.array(kp).astype(np.float32)
        CameraMatrix, w, h = read_intrinsics(query_camera)
        NewCameraMatrix, _ = cv2.getOptimalNewCameraMatrix(CameraMatrix, kp, (w, h), 1,  (w, h), 0)

        # img_disort = cv2.undistort(img1, CameraMatrix, kp, None, NewCameraMatrix) # None, NewCameraMatrix
        img1 = cv2.resize(img1, size)
        # OpenCV 是 BGR → 转为 RGB，再转为 PIL.Image
        img1_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

        # 保存 GIF 动图
        gif_path = os.path.join(output_dir, fname1.replace('.png', '.gif'))
        img1_pil.save(
            gif_path,
            format='GIF',
            save_all=True,
            append_images=[img2_pil],
            duration=duration,
            loop=loop
        )
        print(f"生成 GIF：{gif_path}")

if __name__ == "__main__":
    folder2 = "/mnt/sda/MapScape/query/estimation/mp4_compare/DJI_20250612194903_0021_V/GeoPixel"
    folder1 = "/mnt/sda/MapScape/query/estimation/mp4_compare/DJI_20250612194903_0021_V/GT_distorted"
    output_dir = "/mnt/sda/MapScape/query/estimation/mp4_compare/DJI_20250612194903_0021_V/gif"

    make_pairwise_gif(folder2, folder1, output_dir, size=(960, 540), duration=500)
