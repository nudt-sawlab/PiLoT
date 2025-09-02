from this import d
import threading
from pixloc.utils.data import Paths
import os
import queue
import glob
import argparse
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
from pixloc.utils.eval import evaluate
from pixloc.utils.get_depth import get_3D_samples, pad_to_multiple, generate_render_camera, get_3D_samples_v2
from pixloc.utils.transform import euler_angles_to_matrix_ECEF, pixloc_to_osg, WGS84_to_ECEF
from pixloc.utils import video_generation
import time
import yaml
import copy
import logging
import torch
from multiprocessing import Process, Queue, Event
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)     # Linux 默认 fork → 改成 spawn
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(message)s",
    datefmt="%H:%M:%S")
'''
1. config init rot, init translation, datapath
'''
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

class DualProcessTask:          # ← 名字随意
    def __init__(self, config):
        # 用 multiprocessing 队列/事件
        self.task_q   = Queue(maxsize=2)     # 渲染 → 定位
        self.pose_q   = Queue(maxsize=3)     # 定位 → 渲染
        self.stop_evt = Event()
        self.render_config = config["render_config"]
        default_confs = config["default_confs"] 
        conf = default_confs['from_render_test'] # from_render_test
        # conf初始化
        folder_path = default_confs['dataset_path']
        dataset_name = default_confs['dataset_name']
        output_name = default_confs['output_name']
        
        self.refine_conf = default_confs['refine']
        self.mul = self.refine_conf['mul']
        # self.mul = None
        self.estimated_pose = os.path.join(folder_path, 'estimation', 'FPVLoc@'+output_name +'.txt') #'FPVLoc@'+ dataset_name +'.txt'
        self.gt_pose = os.path.join(folder_path, 'poses', dataset_name +'.txt') #
        self.render_pose = "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/outputs/FPVLoc@switzerland_seq4@8@sunny@500@512.txt"
        self.query_resize_ratio = render_config['render_camera'][0] / default_confs['cam_query']['width']
        self.render_camera_osg = np.array(render_config['render_camera'])
        self.render_camera_osg = self.render_camera_osg / self.query_resize_ratio
        
        img_path = os.path.join(folder_path, 'images', dataset_name)
        self.img_list = glob.glob(img_path + "/*.png") + glob.glob(img_path + "/*.jpg") + glob.glob(img_path + "/*.JPG")
        self.img_list = sorted(self.img_list, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        self.query_list = read_image_list(self.img_list, scale = self.query_resize_ratio)
        
        # render_img_path = "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/outputs/Mapsape/switzerland_seq4@8@sunny@500@512"
        # self.render_img_list = glob.glob(render_img_path + "/*.png") + glob.glob(render_img_path + "/*.jpg") + glob.glob(img_path + "/*.JPG")
        # self.render_img_list = sorted(self.render_img_list, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        # self.render_list, self.render_depth_list = read_render_image_list(self.render_img_list, scale = 1)
        
        # self.img_list = self.img_list[109:]
        self.last_frame_info = {}
        self.last_frame_info['observations'] = []
        self.last_frame_info['refine_conf'] = self.refine_conf
        
        print(f'conf:\n{pformat(conf)}')
        
        # 初始化先验位姿和内参
        self.euler_angles, self.translation = render_config['init_rot'], render_config['init_trans']
        self.name_q = None
        cam_query = default_confs["cam_query"]
        self.query_camera = Camera.from_colmap(cam_query) #! 2.


        
        self.dd = None
        self.name_r = None

        self.render_camera = generate_render_camera(self.render_camera_osg).float()
        render_config['render_camera'] = self.render_camera_osg 

        self.outputs = os.path.join(render_config['outputs'], output_name)
        if not os.path.exists(self.outputs):
            os.makedirs(self.outputs)
        # 是否padding, num init  pose
        self.num_init_pose = default_confs['num_init_pose']
        self.padding = default_confs['padding']

        # 设置无人机起始点为geo origin
        self.origin = WGS84_to_ECEF(self.translation) #np.array(default_confs['refine']['origin'])
        default_confs['refine']['origin'] = self.origin
        self.gt_pose_dict = load_poses(self.gt_pose, origin = self.origin)
        # self.render_pose_dict = load_poses(self.render_pose, origin = self.origin)
        # 定位初始化
        # self.localizer = RenderLocalizer(conf)
        # self.tracker = SimpleTracker(self.localizer.refiner)  # will hook & store the predictions
        # # 渲染初始化
        # self.renderer = osg_render.RenderImageProcessor(render_config)

        self.pose_q.put_nowait((render_config['init_rot'], render_config['init_trans']))
        self.pose_q.put_nowait((render_config['init_rot'], render_config['init_trans']))
        
      
    # ---------------- 渲染线程 ----------------
    def rendering_worker(self):
        # -------- 一次性初始化 (只做一次) ------------
        import torch
        from pixloc.utils.osg import osg_render

        renderer      = osg_render.RenderImageProcessor(render_config)
        render_stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        while not self.stop_evt.is_set():
            try:
                euler, trans = self.pose_q.get(timeout=1)
            except queue.Empty:
                continue

            # --- GPU warm-up 一次即可 ---
            # if not warmup_done:
            #     for _ in range(5):
            #         self.renderer.update_pose(trans, euler,
            #                                 ref_camera=self.render_camera_osg)
            #     warmup_done = True

            # --- 真正渲染：只 1 次 ---
            with torch.cuda.stream(render_stream):
                for _ in range(5):                       # 收敛 Near/Far
                    renderer.update_pose(trans, euler, ref_camera=self.render_camera_osg)
                color = renderer.get_color_image()
                depth = renderer.get_depth_image()

            torch.cuda.current_stream().synchronize()   # 等 GPU 完
            if self.padding:
                color = pad_to_multiple(color, 16)

            P3d, T_w2c_mod, T_init = self.back_project(depth, euler, trans)

            self.task_q.put((color, P3d, T_w2c_mod, T_init, euler, trans ))

    # ---------------- 定位线程 ----------------
    def localization_worker(self):
        import torch
        from pixloc.localization import RenderLocalizer, SimpleTracker
        localizer = RenderLocalizer(self.conf)
        results = []   # 存盘缓冲
        fps_log_every = 30
        for idx, (img_path, img_tensor) in enumerate(zip(self.img_list, self.query_list)):
            # 1) 拿到渲染结果（阻塞）
            color, P3d, T_w2c_mod, T_init, render_euler, render_trans = self.task_q.get()
            t0 = time.time()
            # 2) 调用定位
            ret = localizer.run_query(
                img_path, self.query_camera, self.render_camera,
                color,  # render_frame
                query_T = T_init,
                render_T= T_w2c_mod,
                Points_3D_ECEF = P3d,
                query_resize_ratio = self.query_resize_ratio,
                dd = self.dd,
                gt_pose_dict = self.gt_pose_dict,
                last_frame_info = self.last_frame_info,
                image_query = img_tensor
            )
            if idx % fps_log_every == 0:
                logging.info("loc %.2f ms", (time.time()-t0)*1e3)
            # 3) 把得到的新姿态塞回 pose_q，供下一帧渲染
            self.pose_q.put((ret['euler_angles'], ret['translation']))

            # 4) 暂存结果
            qname = os.path.basename(img_path)
            results.append(f"{qname} {' '.join(map(str, ret['translation']))} "
                           f"{' '.join(map(str, [ret['euler_angles'][1], ret['euler_angles'][0], ret['euler_angles'][2]]))}")

            if self.stop_evt.is_set():
                break

        # 写文件一次性落盘，更快
        with open(self.estimated_pose, "w") as f:
            f.write("\n".join(results))

        self.stop_evt.set()       # 通知渲染线程退出
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

    # 假设的渲染和反投影函数
    def render_scene(self, euler_angles, translation):
        # 渲染场景并返回RGB图和深度图
        
        start_time = time.time()
        for _ in range(20):
            self.renderer.update_pose(translation, euler_angles, ref_camera = self.render_camera_osg)
            color_image = self.renderer.get_color_image()
            depth_image = self.renderer.get_depth_image()
        # end_time = time.time()
        # print('---render time: ', end_time - start_time)
        # if self.name_r is not None:
        #     self.renderer.save_color_image(os.path.join(self.outputs, self.name_r.split('/')[-1]))
        #     print(os.path.join(self.outputs, self.name_r.split('/')[-1]))
        # else:
        #     self.renderer.save_color_image(os.path.join(self.outputs, 'init.png'))
        #     print(os.path.join(self.outputs, 'init.png'))
        if self.padding:
            color_image = pad_to_multiple(color_image, padd=16)  
            # depth_image = pad_to_multiple(depth_image, padd=16)
        return color_image, depth_image

    def back_project(self, depth_frame, euler_angles, translation, num_samples = 200):
        # 
        T_render_in_ECEF_c2w = euler_angles_to_matrix_ECEF(euler_angles, translation)
        # 反投影得到3D点
        # print('T render_______', euler_angles, translation)
        width, height = self.render_camera_osg[:2]
        ey = np.random.randint(0, height, size= num_samples)
        ex = np.random.randint(0, width, size= num_samples)
        points2d = np.column_stack((ex, ey))
        Points_3D_ECEF, T_render_in_ECEF_w2c_modified, T_initial_pose_candidates, dd = get_3D_samples_v2(points2d, depth_frame, T_render_in_ECEF_c2w, self.render_camera,  euler_angles, translation, origin = self.origin, num_init_pose=self.num_init_pose,mul = self.mul)
        if dd is not None:
            self.dd = dd
        return Points_3D_ECEF, T_render_in_ECEF_w2c_modified, T_initial_pose_candidates

    # 假设的计算位姿函数
    def calculate_pose(self, render_frame, points_3d):
        # 根据渲染图和3D点计算位姿
        return "Pose"

    # 主函数模拟
    def input(self):
        # 模拟传递第n-1帧渲染图和3D点
        while self.render_flag == -2:
            self.render_event.set()
            time.sleep(1)
        # self.pose = self.T_init
        self.render_event.set()
        while self.loc_flag < len(self.query_list):
            time.sleep(0.001)  # 模拟输入的延迟
            # if 
    def video_save(self):
        video_generation.create_video_from_images(self.outputs, self.outputs+'/video.mp4') 
    def eval(self):
        evaluate(self.estimated_pose, self.gt_pose)
    def run(self):
        p_render = Process(target=self.rendering_worker, daemon=True)
        p_loc    = Process(target=self.localization_worker)

        p_render.start();  p_loc.start()
        p_loc.join();      self.stop_evt.set();   p_render.join()
def parse_args():
    parser = argparse.ArgumentParser(description="你的程序说明")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="configs/default.yaml",
        help="配置文件路径"
    )
    return parser.parse_args()  
# 主程序入口
if __name__ == "__main__":
    args = parse_args()
    config_file = args.config
    config_file = '/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/configs/switzerland_seq4@8@sunny@500@512.yaml'
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    default_paths = config["default_paths"]
    # render_config = config["render_config"]
    # default_confs = config["default_confs"] 
    # conf = default_confs['from_render_test'] # from_render_test
    print("Running on Unet fusion model")
    dual_task = DualProcessTask(config)
    dual_task.run()
    dual_task.video_save()
    dual_task.eval()
    






