import numpy as np
import cv2
import os
def main(image_file
         ):
    image = cv2.imread(image_file)
    CameraMatrix =np.array([[2700.0, 0, 2700.0],
        [0, 1915.7, 1075.1],
        [0, 0, 1]])
    kp = np.array([0.0681,
            -0.1365,
            -0.0006094,
            -0.0003261,
            0.1177]).astype(np.float32)
    img_disort = cv2.undistort(image, CameraMatrix, kp)
    save_path = '/mnt/sda/MapScape/query/estimation/mp4_compare/DJI_20250612174308_0001_V/GT_distorted'
    save_file = os.path.join(save_path, os.path.basename(image_file))
    cv2.imwrite(save_file, img_disort)
    return img_disort

image_path = "/mnt/sda/MapScape/query/estimation/mp4_compare/DJI_20250612174308_0001_V/GT"
image_list = os.listdir(image_path)
for img in image_list:
    image_file = os.path.join(image_path, img)
    
    main(image_file)