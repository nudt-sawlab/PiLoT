import os
import glob
import copy
import time
import yaml
import logging
import threading
import numpy as np
import torch

from multiprocessing import Process, Queue, Event
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

from pixloc.utils.osg import osg_render
from pixloc.utils.get_depth import get_3D_samples_v3, pad_to_multiple, generate_render_camera
from pixloc.utils.transform import euler_angles_to_matrix_ECEF, WGS84_to_ECEF
from pixloc.localization import RenderLocalizer
from pixloc.pixlib.datasets.view import read_image_list
from pixloc.pixlib.geometry import Camera, Pose
from pixloc.utils.eval import evaluate
from pixloc.utils import video_generation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(message)s",
    datefmt="%H:%M:%S"
)

def load_poses(pose_file, origin):
    """Load poses from the pose file."""
    pose_dict = {}
    translation_list = []
    euler_angles_list = []
    name_list = []
    with open(pose_file, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line
            parts = line.strip().split()
            if parts:  # Ensure the line is not empty
                # pose_dict[parts[0]] = parts[1: ]  # Add the first element to the name list
                lon, lat, alt, roll, pitch, yaw = map(float, parts[1: ])
                translation_list.append([lon, lat, alt])
                euler_angles_list.append([pitch, roll, yaw])
                name_list.append(parts[0])
                euler_angles = [pitch, roll, yaw]
                translation = [lon, lat, alt]
                T_in_ECEF_c2w = euler_angles_to_matrix_ECEF(euler_angles, translation)
                pose_dict[parts[0]] = {}
                pose_dict[parts[0]]['T_w2c_4x4'] = copy.deepcopy(T_in_ECEF_c2w)
                T_in_ECEF_c2w[:3, 1] = -T_in_ECEF_c2w[:3, 1]  # Y轴取反，投影后二维原点在左上角
                T_in_ECEF_c2w[:3, 2] = -T_in_ECEF_c2w[:3, 2]  # Z轴取反
                T_in_ECEF_c2w[:3, 3] -= origin  # t_c2w - origin
                render_T_w2c = np.eye(4)
                render_T_w2c[:3, :3] = T_in_ECEF_c2w[:3, :3].T
                render_T_w2c[:3, 3] = -T_in_ECEF_c2w[:3, :3].T @ T_in_ECEF_c2w[:3, 3]
                
                
                pose_dict[parts[0]]['euler'] = [pitch, roll, yaw]
                pose_dict[parts[0]]['trans'] = [lon, lat, alt]
                
                
                render_T_w2c = Pose.from_Rt(render_T_w2c[:3, :3], render_T_w2c[:3, 3])
                pose_dict[parts[0]]['T_w2c'] = render_T_w2c.to_flat()
    return pose_dict

class DualProcessTask:
    def __init__(self, config):
        # —— 队列 & 全局停止事件 —— 
        self.task_q   = Queue(maxsize=2)   # 渲染 → 定位
        self.pose_q   = Queue(maxsize=3)   # 定位 → 渲染
        self.stop_evt = Event()            # 跨进程可共享的停止信号

        # —— 原有初始化不变 —— 
        self.render_config = config["render_config"]
        default_confs = config["default_confs"]
        self.conf = default_confs['from_render_test']

        folder_path = default_confs['dataset_path']
        dataset_name = default_confs['dataset_name']
        output_name  = default_confs['output_name']

        self.estimated_pose = os.path.join(
            folder_path, 'estimation', f'FPVLoc@{output_name}.txt')
        self.gt_pose = os.path.join(folder_path, 'poses', f'{dataset_name}.txt')

        # 相机参数
        self.render_camera_osg = np.array(self.render_config['render_camera'])
        self.query_resize_ratio = (self.render_camera_osg[0] /
                                   default_confs['cam_query']['width'])
        self.render_camera_osg /= self.query_resize_ratio
        self.render_camera    = generate_render_camera(
            self.render_camera_osg).float().cuda()

        # 图像列表
        img_path = os.path.join(folder_path, 'images', dataset_name)
        self.img_list = sorted(
            glob.glob(f"{img_path}/*.png") + glob.glob(f"{img_path}/*.jpg"),
            key=lambda x: int(os.path.basename(x).split('.')[0])
        )
        self.query_list = read_image_list(
            self.img_list, scale=self.query_resize_ratio)

        # 初始位姿 & origin
        self.euler_angles, self.translation = (
            self.render_config['init_rot'],
            self.render_config['init_trans']
        )
        self.origin = WGS84_to_ECEF(self.translation)
        self.gt_pose_dict = load_poses(self.gt_pose, origin=self.origin)
        self.origin = torch.tensor(self.origin, device='cuda')

        # 预推两帧避免阻塞
        self.pose_q.put_nowait((self.euler_angles, self.translation))
        self.pose_q.put_nowait((self.euler_angles, self.translation))

    def rendering_worker(self):
        """渲染线程：不断从 pose_q 取新位姿，渲染后扔到 task_q"""
        import torch
        renderer = osg_render.RenderImageProcessor(self.render_config)
        stream   = torch.cuda.Stream()
        torch.cuda.synchronize()

        fps_log = 0
        while not self.stop_evt.is_set():
            try:
                euler, trans = self.pose_q.get(timeout=0.5)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            with torch.cuda.stream(stream):
                for _ in range(5):
                    renderer.update_pose(trans, euler,
                                         ref_camera=self.render_camera_osg)
                color = renderer.get_color_image()
                depth = renderer.get_depth_image()
            torch.cuda.current_stream().synchronize()
            t_render = (time.perf_counter() - t0) * 1e3

            if pad_to_multiple:
                color = pad_to_multiple(color, 16)

            t1 = time.perf_counter()
            P3d, T_mod, T_init, dd = self.back_project(
                depth, euler, trans)
            t_back = (time.perf_counter() - t1) * 1e3

            fps_log += 1
            if fps_log == 30:
                logging.info("render: %.2f ms | back: %.2f ms",
                             t_render, t_back)
                fps_log = 0

            self.task_q.put((color, P3d, T_mod, T_init, dd,
                             euler, trans))

        logging.info("Render worker exiting")

    def localization_worker(self):
        """定位线程：不断从 task_q 取渲染结果，计算并推回 pose_q"""
        localizer = RenderLocalizer(self.conf)
        results = []

        for idx, (img_path, img_tensor) in enumerate(
                zip(self.img_list, self.query_list)):

            # 如果主进程或自身发出停止信号，则跳出
            if self.stop_evt.is_set():
                break

            color, P3d, T_mod, T_init, dd, euler, trans = self.task_q.get()

            t0 = time.perf_counter()
            ret = localizer.run_query(
                img_path,
                Camera.from_colmap(self.conf['cam_query']).to('cuda'),
                self.render_camera,
                color,
                query_T=T_init,
                render_T=T_mod,
                Points_3D_ECEF=P3d,
                dd=dd,
                gt_pose_dict=self.gt_pose_dict,
                last_frame_info=getattr(self, 'last_frame_info', None),
                image_query=img_tensor
            )
            if idx % 30 == 0:
                logging.info("loc: %.2f ms",
                             (time.perf_counter() - t0) * 1e3)

            if idx < len(self.img_list) - 1:
                self.pose_q.put((ret['euler_angles'],
                                 ret['translation']))

            name = os.path.basename(img_path)
            results.append(f"{name} " +
                           " ".join(map(str, ret['translation'])) + " " +
                           f"{ret['euler_angles'][1]} "
                           f"{ret['euler_angles'][0]} "
                           f"{ret['euler_angles'][2]}")

        # 定位结束，写文件并通知渲染线程退出
        with open(self.estimated_pose, "w") as f:
            f.write("\n".join(results))

        logging.info("Localization done, setting stop_evt")
        self.stop_evt.set()

    def back_project(self, depth_frame, euler_angles, translation, num_samples = 200, device = 'cuda'):
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
        H, W = int(self.render_camera_osg[1]), int(self.render_camera_osg[0])
        # [num_samples]
        ys = torch.randint(0, H, size=(num_samples,), device=device)
        xs = torch.randint(0, W, size=(num_samples,), device=device)
        points2d = torch.stack((xs, ys), dim=1)  # (N,2)
        Points_3D_ECEF, T_render_in_ECEF_w2c_modified, T_initial_pose_candidates, dd = get_3D_samples_v3(points2d, depth, T_render_in_ECEF_c2w, self.render_camera,  euler_angles, translation, origin = self.origin, num_init_pose=self.num_init_pose,mul = self.mul)
        # if dd is not None:
        #     self.dd = 
        return Points_3D_ECEF, T_render_in_ECEF_w2c_modified, T_initial_pose_candidates, dd

    def run(self):
        p_render = Process(target=self.rendering_worker)
        p_loc    = Process(target=self.localization_worker)

        p_render.start()
        p_loc.start()

        try:
            # 等定位进程完成或手动中断
            p_loc.join()
        except KeyboardInterrupt:
            logging.info("Main got SIGINT, setting stop_evt")
            self.stop_evt.set()

        # 渲染线程看到 stop_evt 后会退出
        p_render.join()
        logging.info("All processes exited")

    def video_save(self):
        video_generation.create_video_from_images(
            os.path.join(self.render_config['outputs'],
                         "video.mp4")
        )

    def eval(self):
        evaluate(self.estimated_pose, self.gt_pose)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/default.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    task = DualProcessTask(config)
    task.run()
    task.video_save()
    task.eval()
