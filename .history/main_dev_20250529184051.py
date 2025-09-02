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
import  argparse
from pixloc.utils.osg import osg_render
from pixloc.utils.data import Paths
from pixloc.utils.transform import euler_angles_to_matrix_ECEF, WGS84_to_ECEF
from pixloc.localization import RenderLocalizer
from pixloc.pixlib.geometry import Camera, Pose
from pixloc.utils.data import Paths
from pixloc.utils.eval import evaluate
from pixloc.utils.get_depth import get_3D_samples_v3, generate_render_camera,pad_to_multiple
from pixloc.pixlib.datasets.view import read_image_list

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
        # —— 全局队列 & 停止事件 —— 
        self.task_q   = Queue(maxsize=2)    # 渲染 → 定位
        self.pose_q   = Queue(maxsize=3)    # 定位 → 渲染
        self.stop_evt = Event()             # 全局停止标志

        # —— 其余初始化略，保持不变 —— 
        self.render_config = config["render_config"]
        default_confs = config["default_confs"]
        self.conf        = default_confs['from_render_test']
        folder_path      = default_confs['dataset_path']
        dataset_name     = default_confs['dataset_name']
        output_name      = default_confs['output_name']
        self.estimated_pose = os.path.join(folder_path, 'estimation', f'FPVLoc@{output_name}.txt')
        self.gt_pose        = os.path.join(folder_path, 'poses', dataset_name +'.txt')

        self.render_camera_osg = np.array(self.render_config['render_camera'])
        self.query_resize_ratio = self.render_config['render_camera'][0] / default_confs['cam_query']['width']
        self.render_camera_osg /= self.query_resize_ratio
        self.render_camera = generate_render_camera(self.render_camera_osg).float().cuda()

        img_path = os.path.join(folder_path, 'images', dataset_name)
        self.img_list   = sorted(glob.glob(img_path + "/*.png") + glob.glob(img_path + "/*.jpg"),
                                 key=lambda x: int(os.path.basename(x).split('.')[0]))
        self.query_list = read_image_list(self.img_list, scale=self.query_resize_ratio)

        # 初始位姿
        self.euler_angles, self.translation = \
            self.render_config['init_rot'], self.render_config['init_trans']
        self.origin = WGS84_to_ECEF(self.translation)
        self.gt_pose_dict = load_poses(self.gt_pose, origin=self.origin)
        self.origin = torch.tensor(self.origin, device='cuda')

        # 预先推两帧，保证渲染队列不空
        self.pose_q.put_nowait((self.euler_angles, self.translation))
        self.pose_q.put_nowait((self.euler_angles, self.translation))

    def rendering_worker(self):
        """渲染线程：不断从 pose_q 取新位姿，渲染后推到 task_q"""
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

            # 渲染
            t0 = time.perf_counter()
            with torch.cuda.stream(stream):
                for _ in range(5):
                    renderer.update_pose(trans, euler, ref_camera=self.render_camera_osg)
                color = renderer.get_color_image()
                depth = renderer.get_depth_image()
            torch.cuda.current_stream().synchronize()
            t_render = (time.perf_counter() - t0) * 1e3

            if pad_to_multiple:
                color = pad_to_multiple(color, 16)

            # 反投影
            t1 = time.perf_counter()
            P3d, T_mod, T_init, dd = self.back_project(depth, euler, trans)
            t_back = (time.perf_counter() - t1) * 1e3

            fps_log += 1
            if fps_log == 30:
                logging.info("render: %.2f ms | back_project: %.2f ms", t_render, t_back)
                fps_log = 0

            # 推送给定位
            self.task_q.put((color, P3d, T_mod, T_init, dd, euler, trans))

        logging.info("Render worker exiting")

    def localization_worker(self):
        """定位线程：从 task_q 取渲染结果，输出新位姿，推回 pose_q"""
        localizer = RenderLocalizer(self.conf)
        results    = []

        for idx, (img_path, img_tensor) in enumerate(zip(self.img_list, self.query_list)):
            if self.stop_evt.is_set():
                break

            # 拉渲染结果
            color, P3d, T_mod, T_init, dd, euler, trans = self.task_q.get()

            # 定位
            t0 = time.perf_counter()
            ret = localizer.run_query(
                img_path, 
                Camera.from_colmap(self.conf['cam_query']).to('cuda'),
                self.render_camera,
                color,
                query_T = T_init,
                render_T = T_mod,
                Points_3D_ECEF = P3d,
                dd = dd,
                gt_pose_dict = self.gt_pose_dict,
                last_frame_info = getattr(self, 'last_frame_info', None),
                image_query = img_tensor
            )
            if idx % 30 == 0:
                logging.info("loc: %.2f ms", (time.perf_counter()-t0)*1e3)

            # 推回下帧渲染
            if idx < len(self.img_list)-1:
                self.pose_q.put((ret['euler_angles'], ret['translation']))

            # 记录结果
            name = os.path.basename(img_path)
            results.append(f"{name} " +
                " ".join(map(str, ret['translation'])) + " " +
                f"{ret['euler_angles'][1]} {ret['euler_angles'][0]} {ret['euler_angles'][2]}"
            )

        # 定位完了，通知渲染退出
        with open(self.estimated_pose, "w") as f:
            f.write("\n".join(results))

        logging.info("Localization worker exiting, signaling stop_evt")
        self.stop_evt.set()

    def back_project(self, depth, euler, trans):
        # … 保持不变 …
        return Points_3D_ECEF, T_mod, T_init, dd

    def run(self):
        p_render = Process(target=self.rendering_worker, daemon=False)
        p_loc    = Process(target=self.localization_worker, daemon=False)

        p_render.start()
        p_loc.start()

        try:
            # 等定位完毕或中断
            p_loc.join()
        except KeyboardInterrupt:
            logging.info("Main received Ctrl+C, setting stop_evt")
            self.stop_evt.set()

        # 渲染线程见 stop_evt 再退出
        p_render.join()

        logging.info("All processes have exited")

    def video_save(self):
        # … unchanged …
        pass

    def eval(self):
        # … unchanged …
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/default.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    args.config = '/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/configs/switzerland_seq4@8@sunny@500@512.yaml'
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    task = DualProcessTask(config)
    task.run()
    task.video_save()
    task.eval()
