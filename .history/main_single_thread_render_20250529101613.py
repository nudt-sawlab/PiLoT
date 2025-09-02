from this import d
import threading
from pixloc.utils.data import Paths
import os
import queue
import glob
import argparse
import torch
import numpy as np
from pixloc.utils.osg import osg_render
from pixloc.utils.transform import colmap_to_osg
from tqdm import tqdm
from pprint import pformat
from pixloc.settings import DATA_PATH, LOC_PATH
from pixloc.localization import RenderLocalizer, SimpleTracker
from pixloc.pixlib.geometry import Camera, Pose
from pixloc.pixlib.datasets.view import read_image_list, read_render_image_list
from pixloc.utils.colmap import qvec2rotmat
from pixloc.utils.data import Paths
from pixloc.utils.eval import evaluate
from pixloc.utils.get_depth import get_3D_samples_v3, pad_to_multiple, generate_render_camera, get_3D_samples_v2
from pixloc.utils.transform import euler_angles_to_matrix_ECEF, pixloc_to_osg, WGS84_to_ECEF
from pixloc.utils import video_generation
import time
import yaml
import copy
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

class DualThreadTask:
    def __init__(self, conf):
        self.render_flag = -2
        self.loc_flag = 0
        
        # 
        # 初始化队列
        self.render_event = threading.Event()  # 渲染线程等待定位线程的位姿
        self.loc_event = threading.Event()     # 定位线程等待渲染线程的参考图
        self.stop_event = threading.Event() # 停止标志
        self.stop_flag = False  # 用于停止线程的标志
 
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
        # self.img_list = self.img_list[:50]
        # self.query_list = read_image_list(self.img_list, scale = self.query_resize_ratio)
        
        render_img_path = "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/outputs/Mapsape/switzerland_seq4@8@sunny@500@512"
        self.render_img_list = glob.glob(render_img_path + "/*.png") + glob.glob(render_img_path + "/*.jpg") + glob.glob(img_path + "/*.JPG")
        self.render_img_list = sorted(self.render_img_list, key=lambda x: int(x.split('/')[-1].split('.')[0]))
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
        self.device = 'cuda'
        # 设置无人机起始点为geo origin
        self.origin = WGS84_to_ECEF(self.translation) #np.array(default_confs['refine']['origin'])
        default_confs['refine']['origin'] = self.origin
        self.gt_pose_dict = load_poses(self.gt_pose, origin = self.origin)
        self.render_pose_dict = load_poses(self.render_pose, origin = self.origin)
        # 定位初始化
        self.localizer = RenderLocalizer(conf, device=self.device)
        
        # self.tracker = SimpleTracker(self.localizer.refiner)  # will hook & store the predictions
        # 渲染初始化
        
        self.renderer = osg_render.RenderImageProcessor(render_config)
        
    def localization_thread(self):
        with open(self.estimated_pose, "w") as f:
            # 计算并输出第n帧位姿解算结果
            for frame in range(len(self.query_list)):
                self.loc_flag = frame
                self.name_q = self.img_list[self.loc_flag]
                query_image = self.query_list[self.loc_flag]
                print("-------", self.name_q)
                if self.loc_flag < 0:
                    temp_name = '0_0.png'
                    #temp
                else:
                    temp_name = str(self.loc_flag) + '_0.png'
                self.euler_angles, self.translation = self.render_pose_dict[temp_name]['euler'], self.render_pose_dict[temp_name]['trans']
                self.euler_angles = np.array(self.euler_angles)
                #temp
                render_frame, depth_frame = self.render_list[int(temp_name.split('_')[0])], self.render_depth_list[int(temp_name.split('_')[0])]

                # 反投影得到3D点
                Points_3D_ECEF, T_render_in_ECEF_w2c_modified, T_query_pose_candidates = self.back_project(depth_frame, self.euler_angles, self.translation, device = self.device)
                self.Points_3D_ECEF = Points_3D_ECEF
                self.T_render = T_render_in_ECEF_w2c_modified
                self.T_query_pose_candidates = T_query_pose_candidates
                self.render_frame = render_frame
                start_time = time.time()
                ret = self.localizer.run_query(self.name_q , self.query_camera, self.render_camera, self.render_frame, query_T = self.T_query_pose_candidates, render_T = self.T_render, Points_3D_ECEF = self.Points_3D_ECEF, query_resize_ratio = self.query_resize_ratio, dd = self.dd, gt_pose_dict = self.gt_pose_dict, last_frame_info=self.last_frame_info, image_query = query_image) 
                
                # 将位姿传递给渲染线程
                self.euler_angles, self.translation = ret['euler_angles'], ret['translation']
                qname = self.name_q.split('/')[-1]
                #temp
                
                euler = [self.euler_angles[1], self.euler_angles[0], self.euler_angles[2]]
                euler = ' '.join(map(str, euler))
                trans = ' '.join(map(str, self.translation))
                name = qname.split('/')[-1]
                self.name_r = copy.deepcopy(self.name_q)
                f.write(f'{name} {trans} {euler}\n')
                f.flush()  # 实时写入文件
                print(f"Localization Thread calculated frames: {self.loc_flag} | Pitch, Roll, Yaw: {self.euler_angles.tolist()} | Longitude, Latitude, Altitude: {self.translation}")
                end_time = time.time()
                print("time: ",end_time-start_time)
                if 'observations' in ret.keys():
                    self.last_frame_info['observations'] = ret["observations"]
                if 'origin' in ret.keys():
                    self.last_frame_info['euler_angles'], self.last_frame_info['translation'] = ret['euler_angles'], ret['translation']
                self.loc_flag += 1
                        

    def rendering_thread(self):
        for i in range(600):
            temp_name = str(i) +'_0.png'
            euler_angles, translation = self.render_pose_dict[temp_name]['euler'], self.render_pose_dict[temp_name]['trans']
            # 渲染RGB图与Depth图
            # temp
            start_time = time.time()
            render_frame, depth_frame = self.render_scene(euler_angles, translation)
            end_time = time.time()
            print('render time: ', end_time -start_time)
            # 反投影得到3D点
            Points_3D_ECEF, T_render_in_ECEF_w2c_modified, T_query_pose_candidates = self.back_project(depth_frame, euler_angles, translation)
            # print(f"Rendering Thread back-projected 3D points: {len(Points_3D_ECEF)}")
            # print('Trender: ', T_render_in_ECEF_w2c_modified.R, T_render_in_ECEF_w2c_modified.t, euler_angles, translation)
            self.render_flag += 1
            # 传递给定位线程，参与第n+1帧定位
            self.render_frame = render_frame
            self.Points_3D_ECEF = Points_3D_ECEF
            self.T_render = T_render_in_ECEF_w2c_modified
            self.T_query_pose_candidates = T_query_pose_candidates


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
        
        start_time1 = time.time()
        for _ in range(5):
            self.renderer.update_pose(translation, euler_angles, ref_camera = self.render_camera_osg)
        color_image = self.renderer.get_color_image()
        depth_image = self.renderer.get_depth_image()
        if self.padding:
                color_image = pad_to_multiple(color_image, padd=16)

            # 2) 反投影 & 先验候选
        P3d, T_w2c_mod, T_init = self.back_project(depth_image, euler_angles, translation)
        end_time1 = time.time()
        print('---render time: ', end_time1 - start_time1)
        depth1 = np.load('/mnt/sda/MapScape/query/poses/test/10_0.npy')
        abs = depth_image - depth1
        import ipdb; ipdb.set_trace()
        # if self.name_r is not None:
        #     self.renderer.save_color_image(os.path.join(self.outputs, self.name_r.split('/')[-1]))
        #     print(os.path.join(self.outputs, self.name_r.split('/')[-1]))
        # else:
        #     self.renderer.save_color_image(os.path.join(self.outputs, 'init.png'))
        #     print(os.path.join(self.outputs, 'init.png')) 
            # depth_image = pad_to_multiple(depth_image, padd=16)
        return color_image, depth_image

    def back_project(self, depth_frame, euler_angles, translation, num_samples = 200, device = 'cuda'):
        # 
        T_render_in_ECEF_c2w = euler_angles_to_matrix_ECEF(euler_angles, translation)
        # 反投影得到3D点
        # print('T render_______', euler_angles, translation)
        width, height = self.render_camera_osg[:2]
        ey = np.random.randint(0, height, size= num_samples)
        ex = np.random.randint(0, width, size= num_samples)
        points2d = np.column_stack((ex, ey))
        Points_3D_ECEF, T_render_in_ECEF_w2c_modified, T_initial_pose_candidates, dd = get_3D_samples_v2(points2d, depth_frame, T_render_in_ECEF_c2w, self.render_camera, euler_angles, translation, origin = self.origin, num_init_pose=self.num_init_pose,mul = self.mul, device=device)

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
            time.sleep(1)  # 模拟输入的延迟
            # if 
    def video_save(self):
        video_generation.create_video_from_images(self.outputs, self.outputs+'/video.mp4') 
    def eval(self):
        evaluate(self.estimated_pose, self.gt_pose)
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
    render_config = config["render_config"]
    default_confs = config["default_confs"] 
    conf = default_confs['from_render_test'] # from_render_test
    print("Running on Unet fusion model")
    dual_thread_task = DualThreadTask(conf)
    
    # # # # # # # # # # # # # 启动线程
    # dual_thread_task.start_threads()

    # # 模拟主线程输入
    # dual_thread_task.input()
    
    # # 停止线程
    # time.sleep(1) 
    # dual_thread_task.stop_threads()
    dual_thread_task.rendering_thread()
    # dual_thread_task.localization_thread()
    dual_thread_task.video_save()
    dual_thread_task.eval()
    






