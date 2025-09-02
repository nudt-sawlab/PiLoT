import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_depth_image(file_path):
    """
    加载深度图（支持16位PNG或未压缩的深度数据）
    :param file_path: 深度图文件路径
    :return: numpy数组格式的深度图
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # 使用 OpenCV 加载深度图（支持16位和未压缩数据）
    depth_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    
    if depth_image is None:
        raise ValueError(f"Failed to load image file: {file_path}")
    
    return depth_image

def preprocess_depth_image(depth_array, max_invalid_value=1000):
    """
    处理深度图：将无效值替换为NaN，便于后续归一化
    :param depth_array: 深度图数组
    :param max_invalid_value: 无效值的定义（如65535）
    :return: 处理后的深度图
    """
    # 将无效值（如65535）替换为NaN
    depth_array = np.where(depth_array > max_invalid_value, np.nan, depth_array)
    return depth_array

def visualize_depth_absolute(depth_array, colormap='viridis', normalize=True, units="meters"):
    """
    可视化绝对深度图，矫正图像原点位置
    :param depth_array: 深度图数组
    :param colormap: 颜色映射方案
    :param normalize: 是否对深度值归一化以增强可视化
    :param units: 深度值单位，默认米
    """
    if normalize:
        # 去除NaN后计算有效值范围
        valid_min = np.nanmin(depth_array)
        valid_max = np.nanmax(depth_array)
        
        # 归一化深度值到0-1范围
        depth_array = (depth_array - valid_min) / (valid_max - valid_min)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(depth_array, cmap=colormap, origin='lower')  # 将图像原点设置为左下角
    plt.colorbar(label=f'Depth ({units})')
    plt.title("Absolute Depth Map Visualization")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # 替换为你的深度图路径
    depth_image_path = '/home/ubuntu/Documents/code/FPVLoc/datasets/Aero/Reference/Jan_seq1/Depth/0_1.png'
    
    try:
        # 加载深度图
        depth_data = load_depth_image(depth_image_path)
        
        # 打印深度图信息
        print(f"Loaded depth map with shape: {depth_data.shape}, dtype: {depth_data.dtype}")
        
        # 处理深度图（移除无效值）
        depth_data_cleaned = preprocess_depth_image(depth_data)
        
        # 如果单位是毫米，转换为米
        depth_data_in_meters = depth_data_cleaned / 1000.0 if depth_data.dtype in [np.uint16, np.int32, np.float32] else depth_data_cleaned
        
        # 可视化深度图
        visualize_depth_absolute(depth_data_in_meters, colormap='viridis', units="meters")
    
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
