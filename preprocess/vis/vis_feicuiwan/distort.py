import numpy as np
import cv2
import os
def main(image_file
         ):
    image = cv2.imread(image_file)
    CameraMatrix =np.array([[2700.0, 0, 1915.7],
        [0, 2700.0, 1075.1],
        [0, 0, 1]])

    kp = np.array([0.0651,
            -0.12010,
            -0.0006094,
            -0.0003261,
            0.0924]).astype(np.float32)
    kp = np.array([0.0046,0.1294,0,0.0012,-0.2037
            ]).astype(np.float32)
    img_disort = cv2.undistort(image, CameraMatrix, kp)
    save_path = '/mnt/sda/MapScape/query/estimation/mp4_compare/DJI_20250612194903_0021_V/GT_distorted'
    name = (os.path.basename(image_file)).split('.')[0] + '.jpg'
    save_file = os.path.join(save_path, name)
    # img_disort = cv2.resize(img_disort, (480, 270))
    # img_disort = cv2.resize(img_disort, (480, 270))
    
    # cv2.imwrite(save_file, img_disort)
    cv2.imwrite(save_file, img_disort, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return img_disort

image_path = "/mnt/sda/MapScape/query/estimation/mp4_compare/DJI_20250612194903_0021_V/GT"
image_list = os.listdir(image_path)

for img in image_list:
    if '340' not in img: continue
    image_file = os.path.join(image_path, img)
    if '.png' not in image_file: continue
    main(image_file)