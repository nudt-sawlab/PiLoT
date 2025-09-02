from pathlib import Path
import numpy as np
import cv2
import os
import math
import sys
sys.path.append("../build/")
from ModelRenderScene import ModelRenderScene
from tqdm import tqdm
class RenderImageProcessor:
    def __init__(self, config):
        self.config = config
        self.osg_config = self.config["osg"]
        self.renderer = self._initialize_renderer()
        self.outputs = Path(self.config["output"])
        self.depths = Path("/home/ubuntu/Documents/code/github/Target2loc/fast_render2loc/3DTilesRender/python/osg_render_depths")

    def _initialize_renderer(self):
        # Construct paths for model

        model_path = self.osg_config["model_path"]
        render_camera = self.config['render_camera']
        view_width, view_height = int(render_camera[0]), int(render_camera[1])
        sensor_width, sensor_height = (render_camera[-2]), (render_camera[-1])
        self.fovy, aspectRatio = self.fov_calculate(render_camera[2], render_camera[-1]), view_width / view_height
        return ModelRenderScene(model_path, view_width, view_height, self.fovy, aspectRatio)
    
    def update_pose(self, Trans, Rot, fovy = None):
        if fovy is not None:
            self.fovy = fovy
        self.renderer.updateViewPoint(Trans, Rot)
        self.renderer.nextFrame(self.fovy) 
    def get_color_image(self):
        colorImgMat = np.array(self.renderer.getColorImage(), copy=False)
        colorImgMat = cv2.flip(colorImgMat, 0)
        colorImgMat = cv2.cvtColor(colorImgMat, cv2.COLOR_RGB2BGR)
        
        return colorImgMat
    
    def get_depth_image(self):
        depthImgMat = np.array(self.renderer.getDepthImage(), copy=False).squeeze()
        
        return depthImgMat
    
    def save_color_image(self, outputs):
        self.renderer.saveColorImage(outputs)
    
    def delay_to_load_map(self, config_web):
        self.euler_angles = config_web['euler_angles']  #相机角度
        self.translation = config_web['translation']   #位置  
        for i in range(500):
            self.update_pose(self.translation, self.euler_angles)
    def get_color_image(self):
        colorImgMat = np.array(self.renderer.getColorImage(), copy=False)
        colorImgMat = cv2.flip(colorImgMat, 0)
        colorImgMat = cv2.cvtColor(colorImgMat, cv2.COLOR_RGB2BGR)
        
        return colorImgMat
    def rendering(self, config_web, fovy):
        self.image_id = config_web['image_id']
        self.euler_angles = config_web['euler_angles']  #相机角度
        self.translation = config_web['translation']   #位置  
        print(self.euler_angles, self.translation)
        for i in range(50):
            self.update_pose(self.translation, self.euler_angles, fovy)
            color_image = self.get_color_image()
            depth_image = self.get_depth_image() 
        cv2.imwrite(str(self.outputs/self.image_id), color_image)
        # np.save(str(self.depths/(self.image_id[:-4]+'.npy')), depth_image)
        # cv2.imwrite(str(self.depths/self.image_id), np.uint8(np.around(depth_image)))
        # if not os.path.exists(self.outputs/"normalFOV_images"):
        #     os.mkdir(self.outputs/"normalFOV_images")
        
    def fov_calculate(self, f_mm, sensor_height):
        print("sensor_height: ", sensor_height, f_mm)
        fovy_radians = 2 * math.atan(sensor_height / 2 / f_mm)
        fovy_degrees = math.degrees(fovy_radians)
        
        return fovy_degrees
    
    def generate_pose(self, x, y, z, yaw, pitch, roll):
        output_path = "/home/ubuntu/Documents/code/3DTilesRender-3DTilesRender_dev/euler_gt_pose.txt"
        
        # import pdb; pdb.set_trace()
        x_list = [0]#np.arange(0, 0.05, 0.01)
        y_list = [0]#np.arange(0, 0.05, 0.01)
        z_list = np.arange(0, 1, 0.25)
        yaw_list = np.arange(-1, 1, 0.25)
        pitch_list = np.arange(-1, 1, 0.25)
        num = 0
        with open(output_path, 'w') as f:
            for i in x_list:
                for j in y_list:
                    for k in z_list:
                        for v in yaw_list:
                            for w in pitch_list:
                                infor = str(num) + '.png' + ' ' + str(pitch+w) + ' ' + \
                                    str(roll) + ' ' + str(yaw + v) + ' ' + stsensor_height(x + i) + ' ' + \
                                        str(y + j) + ' ' + str(z + k) + '\n'
                                num += 1
                                f.write(infor)
                    

              
        
if __name__ == "__main__":
    # 卫星象山
    config = {
        "osg": {
        "model_path": "http://localhost:8078/Scene/Production_6.json"
        },
        # "render_camera": [
        #     640,  # width
        #     512, #height
        #     1125, # focal length(pixel/mm)
        #     640, # sensor_width(pixel/mm)
        #     512 # sensor_height(pixel/mm)
        # ],
        #  "render_camera": [
        #     1920,
        #     1080,
        #     4.5,
        #     6.17,
        #     3.47,
        # ],
         "render_camera": [
            602,
            448,
            35,
            50.06183825012458,
            37.53780953072358,
        ],
        # "render_camera": [
        #     800,  # width
        #     600, #height
        #     1.5648, # width/height
        #     1.143, # sensor_width
        #     0.858 # sensor_height
        # ],
        "output":"/home/ubuntu/Documents/code/github/Target2loc/fast_render2loc/3DTilesRender/python/osg_render_images/"
    }

    config_web = {
        "image_id":'1.jpg',
        "euler_angles": [],
        "translation":[],
    }
    pose_save_path = "/home/ubuntu/Documents/code/github/Target2loc/fast_render2loc/3DTilesRender/python/select_.txt"
    
    renderer = RenderImageProcessor(config)

    # load map 
    with open(pose_save_path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            tokens = data.split()
            name = os.path.basename(tokens[0])
            # Split the data into the quaternion and translation parts
            e_c2w, t_c2w = np.split(np.array(tokens[1:], dtype=float), [3])
            config_web['euler_angles']  = e_c2w 
            config_web['translation']  = t_c2w
            config_web['image_id'] = name
            break
    renderer.delay_to_load_map(config_web)
    
    # render
    with open(pose_save_path, 'r') as f:
        for data in tqdm(f.read().rstrip().split('\n')):
            tokens = data.split()
            name = os.path.basename(tokens[0])
            # Split the data into the quaternion and translation parts
            e_c2w, t_c2w = np.split(np.array(tokens[1:], dtype=float), [3])
            config_web['euler_angles']  = e_c2w 
            config_web['translation']  = t_c2w
            config_web['image_id'] = name

            fovy = renderer.fov_calculate(config['render_camera'][2], config['render_camera'][-1])
            renderer.rendering(config_web, fovy)
            
            
    
    
    
    
        
