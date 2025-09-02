import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_image(file_path, is_depth=False):
    """
    加载图像（支持RGB或深度图）
    :param file_path: 图像文件路径
    :param is_depth: 是否为深度图，True时加载为16位无损数据
    :return: numpy数组格式的图像
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if is_depth:
        # 加载深度图（16位或未压缩数据）
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to load depth image file: {file_path}")
        # 矫正原点（垂直翻转）
        image = cv2.flip(image, 0)
    else:
        # 加载RGB图像
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load RGB image file: {file_path}")
        # OpenCV默认是BGR格式，需要转换为RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def preprocess_depth_image(depth_array, max_invalid_value=1000):
    """
    处理深度图：将无效值替换为NaN，便于后续归一化
    :param depth_array: 深度图数组
    :param max_invalid_value: 无效值的定义（如65535）
    :return: 处理后的深度图
    """
    depth_array = np.where(depth_array > max_invalid_value, np.nan, depth_array)
    return depth_array

def visualize_images(rgb_image, depth_array, colormap='viridis', normalize=True, units="meters"):
    """
    同时可视化RGB图像和绝对深度图
    :param rgb_image: RGB图像数组
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
    
    # 创建两个子图：RGB和深度图
    plt.figure(figsize=(12, 6))
    
    # 显示RGB图像
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_image)
    plt.title("RGB Image")
    plt.axis('off')
    
    # 显示深度图
    plt.subplot(1, 2, 2)
    plt.imshow(depth_array, cmap=colormap)
    plt.colorbar(label=f'Depth ({units})')
    plt.title("Depth Map")
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    # 替换为你的RGB和深度图路径
    rgb_image_path = '/home/ubuntu/Documents/code/FPVLoc/datasets/Aero/Reference/Jan_seq1/RGB/11730_0.png'
    depth_image_path = '/home/ubuntu/Documents/code/FPVLoc/datasets/Aero/Reference/Jan_seq1/Depth/11730_1.png'
    
    try:
        # 加载RGB图像
        rgb_image = load_image(rgb_image_path, is_depth=False)
        
        # 加载深度图
        depth_data = load_image(depth_image_path, is_depth=True)
        
        # 打印图像信息
        print(f"Loaded RGB image with shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")
        print(f"Loaded depth map with shape: {depth_data.shape}, dtype: {depth_data.dtype}")
        
        # 处理深度图（移除无效值）
        depth_data_cleaned = preprocess_depth_image(depth_data)
        
        # 如果单位是毫米，转换为米
        # depth_data_in_meters = depth_data_cleaned / 1000.0 if depth_data.dtype in [np.uint16, np.int32, np.float32] else depth_data_cleaned
        
        # 可视化RGB和深度图
        visualize_images(rgb_image, depth_data_cleaned, colormap='viridis', units="meters")
    
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
