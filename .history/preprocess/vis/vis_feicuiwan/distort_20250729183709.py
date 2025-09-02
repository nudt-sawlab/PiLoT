import numpy as np
import cv2
def main(image,
         kp,
         ):
    CameraMatrix =[[2700.0, 0, 2700.0],
        [0, 1915.7, 1075.1],
        [0, 0, 1]]
    kp = np.array([0.0651,
            -0.12010,
            -0.0006094,
            -0.0003261,
            0.0924]).astype(np.float32)
    img_disort = cv2.undistort(image, CameraMatrix, kp)
    cv2.imwrite('/home/ubuntu/Documents/code/FPVLoc/object_index/undistorted.jpg', img_disort)
    return img_disort

image_path = "/mnt/sda/MapScape/query/estimation/mp4_compare/DJI_20250612174308_0001_V/GT"