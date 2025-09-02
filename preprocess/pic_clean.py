import os
import re

def sort_by_number(file_name):
    # 提取文件名中 "_" 前的数字部分
    try:
        return int(file_name.split('_')[0])
    except ValueError:
        return float('inf')  # 如果不能转换为数字，排在最后

def process_image_sequence(folder_path, frame_interval=30):
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    # 获取文件夹内所有的文件
    all_files = os.listdir(folder_path)

    # 筛选出以 .png 或 .jpg 结尾的文件
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg'))]

    # 按照 "_" 前的数字排序
    image_files.sort(key=sort_by_number)

    # 保留每隔 frame_interval 的文件
    files_to_keep = set(image_files[::frame_interval])

    # 删除不需要的文件
    for file_name in image_files:
        if file_name not in files_to_keep:
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)
            print(f"Deleted: {file_name}")

    print("Processing complete. Retained files:", files_to_keep)
# 使用示例
folder_path = "/home/ubuntu/Documents/code/FPVLoc/datasets/Aero/Raw/Jan_seq2_sunset"  # 替换为你的文件夹路径
process_image_sequence(folder_path)
