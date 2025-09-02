import numpy as np
def main(image,
         query_camera,
         kp,
         ):
    CameraMatrix =[[fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]]
    kp = np.array(kp).astype(np.float32)
    CameraMatrix = read_intrinsics(query_camera)
    img_disort = cv2.undistort(image, CameraMatrix, kp)
    cv2.imwrite('/home/ubuntu/Documents/code/FPVLoc/object_index/undistorted.jpg', img_disort)
    return img_disort