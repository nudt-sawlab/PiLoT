import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)     # Linux 默认 fork → 改成 spawn
from this import d
import threading
from pixloc.utils.data import Paths
import os
import queue
import glob
import cv2
import shutil
import argparse
import ast
import numpy as np
from pixloc.utils.osg import osg_render
from pixloc.utils.transform import colmap_to_osg
from tqdm import tqdm
from pprint import pformat
from pixloc.settings import DATA_PATH, LOC_PATH
from pixloc.localization import RenderLocalizer, SimpleTracker
from pixloc.pixlib.geometry import Camera, Pose
from pixloc.pixlib.datasets.view import read_image_list
from pixloc.utils.colmap import qvec2rotmat
from pixloc.utils.data import Paths
from pixloc.utils.eval import evaluate_xyz, evaluate_XYZ_EULER
from pixloc.utils.get_depth import get_3D_samples_v3, pad_to_multiple, generate_render_camera, get_3D_samples_v2
from pixloc.utils.transform import euler_angles_to_matrix_ECEF, pixloc_to_osg, WGS84_to_ECEF
from pixloc.utils import video_generation
import time
import yaml
import copy
import logging
import torch
from multiprocessing import Process, Queue, Event

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(message)s",
    datefmt="%H:%M:%S")
'''
1. config init rot, init translation, datapath
'''
def get_init(pose_file):
    with open(pose_file, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line
            parts = line.strip().split()
            if parts:  # Ensure the line is not empty
                # if '14000' in parts[0]:
                # pose_dict[parts[0]] = parts[1: ]  # Add the first element to the name list
                lon, lat, alt, roll, pitch, yaw = map(float, parts[1: ])
                # pitch, roll, yaw, lon, lat, alt,  = map(float, parts[1: ])
                
                euler_angles = [pitch, roll, yaw]
                translation = [lon, lat, alt]
                origin = WGS84_to_ECEF(translation)
                break
    return euler_angles, translation, origin
def load_poses(pose_file, origin = None):
    """Load poses from the pose file."""
    pose_dict = {}
    translation_list = []
    euler_angles_list = []
    name_list = []
    init_euler = None
    init_trans = None
    with open(pose_file, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line
            parts = line.strip().split()
            if parts:  # Ensure the line is not empty
                # pose_dict[parts[0]] = parts[1: ]  # Add the first element to the name list
                lon, lat, alt, roll, pitch, yaw = map(float, parts[1: ])
                translation_list.append([lon, lat, alt])
                euler_angles_list.append([pitch, roll, yaw])
                if '_' not in parts[0]:
                    name = parts[0][:-4] +'_0.png'
                else:
                    name = parts[0]
                name_list.append(name)
                euler_angles = [pitch, roll, yaw]
                translation = [lon, lat, alt]
                T_in_ECEF_c2w = euler_angles_to_matrix_ECEF(euler_angles, translation)
                pose_dict[name] = {}
                pose_dict[name]['T_w2c_4x4'] = copy.deepcopy(T_in_ECEF_c2w)
                T_in_ECEF_c2w[:3, 1] = -T_in_ECEF_c2w[:3, 1]  # Y轴取反，投影后二维原点在左上角
                T_in_ECEF_c2w[:3, 2] = -T_in_ECEF_c2w[:3, 2]  # Z轴取反
                T_in_ECEF_c2w[:3, 3] -= origin  # t_c2w - origin
                render_T_w2c = np.eye(4)
                render_T_w2c[:3, :3] = T_in_ECEF_c2w[:3, :3].T
                render_T_w2c[:3, 3] = -T_in_ECEF_c2w[:3, :3].T @ T_in_ECEF_c2w[:3, 3]
                
                
                pose_dict[name]['euler'] = [pitch, roll, yaw]
                pose_dict[name]['trans'] = [lon, lat, alt]
                
                
                render_T_w2c = Pose.from_Rt(render_T_w2c[:3, :3], render_T_w2c[:3, 3])
                pose_dict[name]['T_w2c'] = render_T_w2c.to_flat()
    return pose_dict

class DualProcessTask:        
    def __init__(self, config, init_euler = None, init_trans = None, name = None):
        # 用 multiprocessing 队列/事件
        self.task_q   = Queue(maxsize=2)     # 渲染 → 定位
        self.pose_q   = Queue(maxsize=3)     # 定位 → 渲染
        self.stop_evt = Event()
        self.render_config = config["render_config"]
        default_confs = config["default_confs"] 
        self.conf = default_confs['from_render_test'] # from_render_test
        # conf初始化
        folder_path = default_confs['dataset_path']
        dataset_name = default_confs['dataset_name']
        output_name = default_confs['dataset_name']
        # self.euler_angles, self.translation = self.render_config['init_rot'], self.render_config['init_trans']
        
        if name is not None:
            dataset_name = name
            output_name = name
        # if init_euler is not None:
        #     self.render_config['init_rot'], self.render_config['init_trans'] = init_euler, init_trans
        #     self.euler_angles = init_euler
        #     self.translation = init_trans
        self.refine_conf = default_confs['refine']
        self.mul = self.refine_conf['mul']
        # self.estimated_pose = os.path.join(folder_path, 'estimation', 'FPVLoc@'+output_name +'.txt') #'FPVLoc@'+ dataset_name +'.txt'
        output_folder = "/mnt/sda/MapScape/query/estimation/result_images/FPVLoc"
        self.outputs = os.path.join(output_folder, output_name)
        if not os.path.exists(self.outputs):
            os.makedirs(self.outputs)
        else:
            shutil.rmtree(self.outputs)
            os.makedirs(self.outputs)
        self.estimated_pose = os.path.join(output_folder, output_name +'.txt') #'FPVLoc@'+ dataset_name +'.txt'
        
        self.gt_pose = os.path.join(folder_path, 'poses', dataset_name +'.txt') #
        
        self.last_frame_info = {}
        self.last_frame_info['observations'] = []
        self.last_frame_info['refine_conf'] = self.refine_conf
        
        print(f'conf:\n{pformat(self.conf)}')
        # 初始化先验位姿和内参
        self.name_q = None
        # camera
        self.query_resize_ratio = default_confs['cam_query']['width'] / default_confs['cam_query']['max_size']
        fx, fy, cx, cy = default_confs['cam_query']['params'] 
        w, h = default_confs['cam_query']['width'], default_confs['cam_query']['height']
        
        raw_query_camera = np.array([w, h, cx, cy, fx, fy])
        self.render_camera_osg = raw_query_camera / self.query_resize_ratio
        
        default_confs['cam_query']['params']  = np.array(default_confs['cam_query']['params']) / self.query_resize_ratio
        default_confs['cam_query']['width'], default_confs['cam_query']['height'] = default_confs['cam_query']['width'] / self.query_resize_ratio, default_confs['cam_query']['height']/self.query_resize_ratio
        cam_query = default_confs["cam_query"]
        self.query_camera = Camera.from_colmap(cam_query) #! 2.
        
        img_path = os.path.join(folder_path, 'images', dataset_name)
        self.img_list = glob.glob(img_path + "/*.png") + glob.glob(img_path + "/*.jpg") + glob.glob(img_path + "/*.JPG")
        self.img_list = sorted(self.img_list, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        self.img_list = self.img_list[:2]
        # self.query_list = read_image_list(self.img_list, scale = self.query_resize_ratio, distortion=cam_query['distortion'], query_camera = raw_query_camera)
        
        self.dd = None
        self.name_r = None

        self.render_camera = generate_render_camera(self.render_camera_osg).float()
        self.render_config['render_camera'] = self.render_camera_osg 

        # 设置无人机起始点为geo origin
        # self.origin = WGS84_to_ECEF(self.translation) #np.array(default_confs['refine']['origin'])
        
        # 是否padding, num init  pose
        self.num_init_pose = default_confs['num_init_pose']
        self.padding = default_confs['padding']
        self.euler_angles, self.translation, self.origin = get_init(self.gt_pose)
        self.render_config['init_rot'], self.render_config['init_trans'] = self.euler_angles, self.translation
        default_confs['refine']['origin'] = self.origin
        self.gt_pose_dict = load_poses(self.gt_pose, origin = self.origin)
        
        self.device = 'cuda'
        self.origin = torch.tensor(self.origin, device=self.device)
        self.query_camera, self.render_camera = self.query_camera.to(self.device), self.render_camera.to(self.device)

        self.pose_q.put_nowait((self.euler_angles, self.translation))
        self.pose_q.put_nowait((self.euler_angles, self.translation))
    # ---------------- 渲染线程 ----------------        
    def rendering_worker(self):
        import torch
        from pixloc.utils.osg import osg_render

        renderer      = osg_render.RenderImageProcessor(self.render_config)
        render_stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        fps_log_every = 0
        idx = 0
        while True:
            try:
                item = self.pose_q.get(timeout=1)     # <—— 带超时
            except queue.Empty:
                if self.stop_evt.is_set():            # 有人喊停就走
                    break
                continue
            if item is None:                          # 对称哨兵
                break
            euler, trans = item
            # ---- 1) 测渲染耗时 ----
            t0 = time.perf_counter()
            with torch.cuda.stream(render_stream):
                for _ in range(20):  # 如果需要 Near/Far 收敛
                    renderer.update_pose(trans, euler)
                color = renderer.get_color_image()
                depth = renderer.get_depth_image()
            renderer.save_color_image(os.path.join(self.outputs, str(fps_log_every-2)+'_0.png'))
            # print(os.path.join(self.outputs, str(fps_log_every-2)+'.png'))
            torch.cuda.current_stream().synchronize()
            t_render = (time.perf_counter() - t0) * 1e3  # ms

            if self.padding:
                color = pad_to_multiple(color, 16)

            # # ---- 2) 测反投影耗时 ----
            # t1 = time.perf_counter()
            # P3d, T_w2c_mod, T_init, dd = self.back_project(depth, euler, trans)
            # t_back = (time.perf_counter() - t1) * 1e3     # ms

            # ---- 3) 打印结果 ----
            fps_log_every+=1
            if fps_log_every % 30 == 0:
                logging.info("render: %.2f ms", t_render)

            # ---- 4) 推送给定位 ----
            try:
                self.task_q.put((color,depth, euler, trans), timeout=1)
            except queue.Full:
                break                       # 定位端不再消费，直接收尾
        self.stop_evt.set()
        self.task_q.put(None)               # 给 localization_worker 的哨兵
        self.task_q.close(); self.task_q.join_thread()
        self.pose_q.close(); self.pose_q.join_thread()
        logging.info('Render process done')
    # ---------------- 定位线程 ----------------
    def localization_worker(self):
        from pixloc.localization import RenderLocalizer, SimpleTracker
        localizer = RenderLocalizer(self.conf)
        results = []   # 存盘缓冲
        fps_log_every = 30
        flag = True
        last_euler, last_trans = None, None
        for idx, (img_path, img_tensor) in enumerate(zip(self.img_list, self.img_list)):
            # 1) 拿到渲染结果（阻塞）
            item = self.task_q.get()
            if item is None:        # 渲染端提前喊停
                break
            color, depth, render_euler, render_trans = item
            # 2) 反投影
            # t1 = time.time()
            if last_trans is None:
                last_euler, last_trans = render_euler, render_trans
            P3d, T_w2c_mod, T_init, dd = self.back_project(depth, render_euler, render_trans, last_euler, last_trans)
            # P3d, T_w2c_mod, T_init, dd = self.back_project(depth, render_euler, render_trans, render_euler, render_trans)
            
            # t2 = time.time()
            # print('反投影耗时：', t2-t1)
            t0 = time.time()
            # 2) 调用定位
            ret = localizer.run_query(
                img_path, self.query_camera, self.render_camera,
                color,  # render_frame
                query_T = T_init,
                render_T= T_w2c_mod,
                Points_3D_ECEF = P3d,
                query_resize_ratio = self.query_resize_ratio,
                dd = dd,
                gt_pose_dict=self.gt_pose_dict,
                last_frame_info = self.last_frame_info,
                # image_query = img_tensor
            )
            last_euler, last_trans = ret['euler_angles'], ret['translation'] 
            # self.last_frame_info['observations'] = ret['observations']
            # self.last_frame_info['euler_angles'] = ret['euler_angles']
            # self.last_frame_info['translation'] = ret['translation']
            # cv2.imwrite(f'{self.outputs}/{idx}_query.png', img_tensor)
            if idx % fps_log_every == 0:
                logging.info("loc %.2f ms", (time.time()-t0)*1e3)
            
            # 3) 把得到的新姿态塞回 pose_q，供下一帧渲染
            if idx < len(self.img_list)-1:
                self.pose_q.put((ret['euler_angles'], ret['translation']))
            # print(f"Localization Thread calculated frames: {idx} | Pitch, Roll, Yaw: {ret['euler_angles'].tolist()} | Longitude, Latitude, Altitude: {ret['translation']}")
            # 4) 暂存结果
            qname = os.path.basename(img_path)
            results.append(f"{qname} {' '.join(map(str, ret['translation']))} "
                           f"{' '.join(map(str, [ret['euler_angles'][1], ret['euler_angles'][0], ret['euler_angles'][2]]))}")

            if self.stop_evt.is_set():
                break
        with open(self.estimated_pose, "w") as f:
            f.write("\n".join(results))
        
        # self.flush_pose_and_send_sentinel()
        # 收尾：给渲染端塞哨兵 + 关队列 + 事件
        self.pose_q.put(None)
        self.stop_evt.set()
        self.task_q.close(); self.task_q.join_thread()
        self.pose_q.close(); self.pose_q.join_thread()
        logging.info('Localization process done')
    def flush_pose_and_send_sentinel(self):
        # 1) 清空 pose_q 中所有旧条目
        try:
            while True:
                self.pose_q.get_nowait()
        except queue.Empty:
            pass

        # 2) 然后往里放一个 None，作为哨兵
        self.pose_q.put(None)
    def start_threads(self):
        # 启动定位线程 
        self.localization_thread_instance = threading.Thread(target=self.localization_thread)
        self.localization_thread_instance.start()

        # 启动渲染线程
        self.rendering_thread_instance = threading.Thread(target=self.rendering_thread)
        self.rendering_thread_instance.start()

    def stop_threads(self):
        self.localization_thread_instance.join()
        self.rendering_thread_instance.join()

    def back_project(self, depth_frame, euler_angles, translation, query_euler_angles, query_translation, num_samples = 500, device = 'cuda'):
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
        Points_3D_ECEF, T_render_in_ECEF_w2c_modified, T_initial_pose_candidates, dd = get_3D_samples_v3(points2d, depth, T_render_in_ECEF_c2w, self.render_camera,  euler_angles, translation,  query_euler_angles, query_translation, origin = self.origin, num_init_pose=self.num_init_pose,mul = self.mul)
        # if dd is not None:
        #     self.dd = 
        return Points_3D_ECEF, T_render_in_ECEF_w2c_modified, T_initial_pose_candidates, dd

    # 假设的计算位姿函数
    def calculate_pose(self, render_frame, points_3d):
        # 根据渲染图和3D点计算位姿
        return "Pose"

    def video_save(self):
        video_generation.create_video_from_images(self.outputs, self.outputs+'/video.mp4') 
    def eval(self):
        evaluate_XYZ_EULER(self.estimated_pose, self.gt_pose)
    def run(self):
        ctx = mp.get_context("spawn")        # 保持 spawn
        p_render = ctx.Process(target=self.rendering_worker, daemon=True)
        p_loc    = ctx.Process(target=self.localization_worker, daemon=True)

        p_render.start(); p_loc.start()
        p_loc.join()                         # 先等定位结束
        p_render.join(5)                    # 最多等 30 s
        if p_render.is_alive():              # 兜底：仍卡住就强退
            p_render.terminate()
            p_render.join()
def parse_args():
    parser = argparse.ArgumentParser(description="你的程序说明")

    parser.add_argument(
        "-c", "--config",
        type=str,
        default="configs/default.yaml",
        help="配置文件路径"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="default_experiment",
        help="实验名称，用于日志标记"
    )

    parser.add_argument(
        "--init_euler",
        type=str,
        default="[0.0, 0.0, 0.0]",
        help="初始欧拉角（格式如：[25.0, 0.0, 314.9993]）"
    )

    parser.add_argument(
        "--init_trans",
        type=str,
        default="[0.0, 0.0, 0.0]",
        help="初始平移（格式如：[x, y, z]）"
    )

    args = parser.parse_args()

    # 将字符串解析为列表
    # args.init_euler = ast.literal_eval(args.init_euler)
    # args.init_trans = ast.literal_eval(args.init_trans)

    return args
# 主程序入口
if __name__ == "__main__":
    args = parse_args()
    init_euler = args.init_euler
    init_trans = args.init_trans
    name = args.name
    config_file = args.config
    
    # filname = None
    # name = None
    # config_file = "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/configs/feicuiwan_m4t_test.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    render_config = config["render_config"]
    default_confs = config["default_confs"]
    default_paths = config["default_paths"]
    dual_task = DualProcessTask(config,  name=name)
    
    dual_task.run()
    # dual_task.video_save()
    dual_task.eval()
    






