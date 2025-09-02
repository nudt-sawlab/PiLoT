from pathlib import Path
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

from ..osg.ModelRenderScene import ModelRenderScene


class RenderImageProcessor:
    def __init__(self, config, eglDpy=0):
        self.osg_config = config
        self.renderer = self._initialize_renderer(eglDpy)
        self._delay()
        # self.viewer = self.renderer.getViewer()
        print("DEBUG: RenderImageProcessor after _delay")

    def _initialize_renderer(self, eglDpy):
        # Construct paths for model
        model_path = self.osg_config["model_path"]
        render_camera = self.osg_config['render_camera']
        view_width, view_height = int(render_camera[0]), int(render_camera[1])
        # self.fovy, self.aspectRatio = 55, 1.777777777777778
        # self.fovy = self.fovy_calculate(render_camera)
        print("DEBUG: before ModelRenderScene")
        return ModelRenderScene(model_path, view_width, view_height, render_camera[-2], render_camera[-1], render_camera[2], render_camera[3])
        # return ModelRenderScene(model_path, view_width, view_height, self.fovy, self.aspectRatio)
    def shutdown(self):
        # 通知 OSG 结束渲染循环
        # self.viewer.setDone(True)
        # 再调用 C++ 层 shutdown，join 渲染线程
        self.renderer.shutdown()
    def _delay(self):
        initTrans = self.osg_config["init_trans"]
        initRot = self.osg_config["init_rot"]
        for i in range(500):
            self.update_pose(initTrans, initRot)
    def fovy_calculate(self, ref_camera):
        _,_,_,_,_, sensor_height_mm, f_mm = ref_camera
        fovy_radian = 2* math.atan(sensor_height_mm / 2 / f_mm)
        fovy_degree = math.degrees(fovy_radian)

        return fovy_degree
    def update_pose(self, Trans, Rot, ref_camera = None):
        # TODO: Fix [osgearth warning] FAILED to create a terrain engine for this map
        if ref_camera is not None:
            self.fovy = self.fovy_calculate(ref_camera)
        self.renderer.updateViewPoint(Trans, Rot)
        self.renderer.nextFrame()
    
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
    def save_depth_image(self, depth_image, outputs):
        np.save(str(self.outputs/(self.image_id[:-4]+'.npy')), depth_image)
        # depth_image = np.where(depth_image > 1000, np.nan, depth_image)
        # # 去除NaN后计算有效值范围
        # valid_min = np.nanmin(depth_image)
        # valid_max = np.nanmax(depth_image)
        
        # # 归一化深度值到0-1范围
        # depth_image = (depth_image - valid_min) / (valid_max - valid_min)
        
        # plt.figure(figsize=(10, 8))
        # plt.imshow(depth_image, cmap='viridis')  # 将图像原点设置为左下角
        # plt.colorbar(label=f'Depth ({"meters"})')
        # plt.title("Absolute Depth Map Visualization")
        # plt.axis('off')
        # plt.savefig(outputs)

    def get_EGLDisplay(self):
        return self.renderer.getEGLDisplay()
    
    
    
    
        
