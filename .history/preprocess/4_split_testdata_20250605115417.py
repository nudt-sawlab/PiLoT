import os
import shutil
import argparse
import re
def split_photos_and_poses(input_dir, output_dir, pose_file, num_groups=3):
    # 获取输入文件夹中所有文件，并过滤出图片（支持常见图片扩展名）
    all_files = sorted(os.listdir(input_dir))
    image_files = [f for f in all_files if '_0' in f]
    
    # if len(image_files) != 4500:
    #     print(f"警告：预期有4500张照片，但实际发现 {len(image_files)} 张。请核对输入文件夹中的文件。")
    #     return
    def sort_key(img_path):
        # 提取文件名中"_0"之前的数字部分
        base_name = os.path.basename(img_path)
        match = re.search(r'(\d+)\_0.png', base_name)
        if match:
            return int(match.group(1))
        return 0

    # 按自定义排序函数排序
    image_files.sort(key=sort_key)
    # 读取 pose.txt 文件中的所有行
    pose_path = os.path.join(input_dir, pose_file)
    with open(pose_path, 'r') as f:
        pose_lines = f.readlines()
    
    # if len(pose_lines) != 4500:
    #     print(f"警告：预期 pose.txt 中有4500行数据，但实际发现 {len(pose_lines)} 行。请核对文件内容。")
    #     return

    photos_per_group = len(image_files) // num_groups  # 每组应为900张

    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 按组切分处理
    for i in range(num_groups):
        # 创建每个组的照片存放文件夹，例如 group_0, group_1, ...
        group_folder = os.path.join(output_dir, f"group_{i}")
        os.makedirs(group_folder, exist_ok=True)
        
        group_start = i * photos_per_group
        group_end = (i+1) * photos_per_group
        
        # 拷贝并重命名图片，重命名为 0～899
        for j, photo_name in enumerate(image_files[group_start:group_end]):
            ext = os.path.splitext(photo_name)[1]
            src_path = os.path.join(input_dir, photo_name)
            dest_path = os.path.join(group_folder, f"{j}_0{ext}")
            shutil.copy(src_path, dest_path)
        
        # 修改对应组的 pose 数据，使第一列与图片序号保持一致
        group_pose_lines = pose_lines[group_start:group_end]
        modified_pose_lines = []
        for j, line in enumerate(group_pose_lines):
            # 假设每行数据以空白字符分隔，第一列为名称/编号
            parts = line.strip().split()
            if parts:
                parts[0] = f"{j}_0{ext}" # 将第一列替换为当前图片序号
            modified_pose_lines.append(" ".join(parts) + "\n")
        
        # 写入修改后的 pose 文件，例如 pose_0.txt, pose_1.txt, ...
        pose_out_path = os.path.join(output_dir, f"pose_{i}.txt")
        with open(pose_out_path, 'w') as f:
            f.writelines(modified_pose_lines)
            
        print(f"组 {i} 处理完成：{group_folder} 中包含图片，{pose_out_path} 已生成。")
    
    print("所有组处理完成。")

if __name__ == "__main__":
    input_dir = "/mnt/sda/MapScape/query/USA_seq5@8@cloudy_sunset/USA_seq5@8@cloudy"
    output_dir = "/mnt/sda/MapScape/query/images/USA_seq5@8@cloudy"
    pose_file = "/mnt/sda/MapScape/query/poses/USA_seq5@8@300-100.txt"
    
    split_photos_and_poses(input_dir, output_dir, pose_file)
