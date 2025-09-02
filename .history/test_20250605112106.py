from pixloc.utils.get_depth import get_3D_samples_v3,generate_render_camera
from pixloc.utils.transform import euler_angles_to_matrix_ECEF
import torch
from multiprocessing import Process, Queue, Event
import numpy as np
import time
def back_project( depth_frame, euler_angles, translation, num_samples = 500, device = 'cuda'):
        # 
        if not torch.is_tensor(depth_frame):
            depth = torch.as_tensor(depth_frame, device=device)
        else:
            depth = depth_frame.to(device)

        # 2) 把 T_render_in_ECEF_c2w 也转为 GPU tensor
        T_render_in_ECEF_c2w = torch.as_tensor(
            euler_angles_to_matrix_ECEF(euler_angles, translation),
            device=device, dtype=torch.float32
        )  # shape (4,4) 或 (3,4)

        # 3) 生成随机像素坐标也用 torch
        H, W = 512,512
        # [num_samples]
        ys = torch.randint(0, H, size=(num_samples,), device=device)
        xs = torch.randint(0, W, size=(num_samples,), device=device)
        points2d = torch.stack((xs, ys), dim=1)  # (N,2)
        Points_3D_ECEF, T_render_in_ECEF_w2c_modified, T_initial_pose_candidates, dd = get_3D_samples_v3(points2d, depth, T_render_in_ECEF_c2w, render_camera,  euler_angles, translation, origin = [111110.0,111110.0,111110.0], num_init_pose=500,mul = 0.001)
        # if dd is not None:
        #     self.dd = 
        return Points_3D_ECEF, T_render_in_ECEF_w2c_modified, T_initial_pose_candidates, dd
    
depth = np.load('depth.npy')
render_camera_osg = [512,512, 256,256, 5.12,5.12, 6]
render_camera = generate_render_camera(render_camera_osg).float()
for _ in range(100):
    t1 = time.time()
    back_project(depth, euler_angles=[24.971615671291282, -0.0014066373185035689 ,-45.015328436935214], translation=[7.621656338208605, 46.74082876209364, 1100.1373342024162])
    print(time.time()-t1)
