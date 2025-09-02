import cv2
import numpy as np
import os
from pathlib import Path
import glob
import shutil


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

def write_intrinsics(intrinsics_path, NewCameraMatrix, w, h, name_list):
    fx, fy, cx, cy = NewCameraMatrix[0][0], NewCameraMatrix[1][1], NewCameraMatrix[0, 2], NewCameraMatrix[1,2]
    
    with open(intrinsics_path, 'w+') as f:
        for name in name_list:
            outline = name + ' ' + 'PINHOLE' +' '+ str(w) + ' ' + str(h) + ' ' + str(fx) + ' ' + str(fy) + ' ' + str(cx) + ' ' + str(cy) + '\n'
            f.write(outline)
    
def main(image,
         query_camera,
         kp,
         ):
    kp = np.array(kp).astype(np.float32)
    CameraMatrix, w, h = read_intrinsics(query_camera)
    NewCameraMatrix, _ = cv2.getOptimalNewCameraMatrix(CameraMatrix, kp, (w, h), 1,  (w, h), 0)

    img_disort = cv2.undistort(image, CameraMatrix, kp, None, NewCameraMatrix) # None, NewCameraMatrix
    print(NewCameraMatrix)
    update_query_camera = [w, h, NewCameraMatrix[0][0], NewCameraMatrix[0][2], NewCameraMatrix[1][2]]
    return img_disort, update_query_camera

       
if __name__ == "__main__":
    kp = [0.3009, -1.1082232, 0.00050171939, 0.00048351417, 1.118214837] #k1k2p1p2k3
    image_path = "/media/ubuntu/XB/DJI_20231204164034_0001_W_frames_1" 
    w_save_path = "/media/ubuntu/XB/undistort_video_frames"
    intrinsics_path = "/home/ubuntu/Documents/code/github/Render2loc/datasets/demo4/queries/query_intrinsics.txt"
    main(image_path, w_save_path, intrinsics_path, kp)      