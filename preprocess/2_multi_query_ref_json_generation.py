import os
import json
import numpy as np
import json
import pyproj
from scipy.spatial.transform import Rotation as R

def WGS84_to_ECEF(pos):
    lon, lat, height = pos
    transprojr = pyproj.Transformer.from_crs(
        "EPSG:4326",
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        always_xy=True,
    )
    xpjr, ypjr, zpjr = transprojr.transform(lon, lat, height, radians=False)
    return [xpjr, ypjr, zpjr]

def get_rotation_enu_in_ecef(lon, lat):
    """
    @param: lon, lat Longitude and latitude in degree
    @return: 3x3 rotation matrix of heading-pith-roll ENU in ECEF coordinate system
    Reference: https://apps.dtic.mil/dtic/tr/fulltext/u2/a484864.pdf, Section 4.3, 4.1
    Reference: https://www.fossen.biz/wiley/ed2/Ch2.pdf, p29
    """
    # 将角度转换为弧度
    latitude_rad = np.radians(lat)
    longitude_rad = np.radians(lon)
    
    # 计算向上的向量（Up Vector）
    up = np.array([
        np.cos(longitude_rad) * np.cos(latitude_rad),
        np.sin(longitude_rad) * np.cos(latitude_rad),
        np.sin(latitude_rad)
    ])
    
    # 计算向东的向量（East Vector）
    east = np.array([
        -np.sin(longitude_rad),
        np.cos(longitude_rad),
        0
    ])
    
    # 计算向北的向量（North Vector），即up向量和east向量的外积（叉积）
    north = np.cross(up, east)
    
    # 构建局部到世界坐标系的转换矩阵
    local_to_world = np.zeros((3, 3))
    local_to_world[:, 0] = east  # 东向分量
    local_to_world[:, 1] = north  # 北向分量
    local_to_world[:, 2] = up  # 上向分量
    return local_to_world
def generate_refer_info(sequence_folder, output_json, seq_name, ref_pose_dict):
    """
    Generate a JSON file containing query and reference pairings.

    :param sequence_folder: Folder containing the six sequences of query images.
    :param output_json: Path to save the generated JSON file.
    """
    query_files = []
    reference_rgb_path = "Reference/"+seq_name+"/RGB"
    reference_depth_path = "Reference/"+seq_name+"/Depth"

    query_pose_dict = load_poses(os.path.join(sequence_folder, 'pose.txt'))
    # Generate pairings
    refer_info = {}
    for pose_name, pose in query_pose_dict.items():
        # get query pose
        lon, lat, alt, roll, pitch, yaw = map(float, pose)

        euler_angles = [pitch, roll, yaw]
        translation = [lon, lat, alt]
        
        rot_pose_in_ned = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
        rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
        R_c2w = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
        t_c2w = WGS84_to_ECEF(translation)
        query_T = np.eye(4)
        query_T[:3, :3] = R_c2w
        query_T[:3, 3] = t_c2w
        query_T = query_T.tolist()
        # query_pose_euler = [135.4952986366768, 34.61647769927806, 349.72181821408884, 44.58677849397614, -0.04386775097711815, 63.959086793208705]
        # euler_angles = query_pose_euler[3:]
        # translation = query_pose_euler[0:3]
        # lon, lat, height = translation
        # rot_pose_in_ned = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
        # rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
        # R_c2w = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
        # t_c2w = WGS84_to_ECEF(translation)
        # print("drone wgs84: ", translation)
        # # (34.61566232598386,135.49318247040046)
        # T_c2w = np.eye(4)
        # T_c2w[:3, :3] = R_c2w
        # T_c2w[:3, 3] = t_c2w

        # get query intrinscis
        K_w2c = [640, 480, 1200.0, 1200.0, 320, 240]
         # get query rgb path
        query_rgb_path = os.path.join(sequence_folder, 'RGB', pose_name)
        # get reference info
        query_index = int(pose_name.split('.')[0].split('_')[-1])
        ref_indices = [
            query_index - 150,
            query_index - 300,
            query_index - 450,
            query_index + 150,
            query_index + 300,
            query_index + 450
        ]
        # Post-process indices
        ref_pose_list = []
        ref_intrisics_list = []
        for i, ref_idx in enumerate(ref_indices):
            if ref_idx < 0:
                ref_indices[i] = query_index + (i+1) * 50 + 30  # Replace with 80, 130, 180
            elif ref_idx > len(ref_pose_dict)-1:
                ref_indices[i] = query_index - (6 - i) * 50 - 30  # Replace with -180, -130, -80
            ref_pose = ref_pose_dict[str(ref_indices[i])+'.png']
            lon, lat, alt, roll, pitch, yaw = map(float, ref_pose)
            euler_angles = [pitch, roll, yaw]
            translation = [lon, lat, alt]
            lon, lat, height = translation
            rot_pose_in_ned = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
            rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
            R_c2w = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
            t_c2w = WGS84_to_ECEF(translation)
            ref_T = np.eye(4)
            ref_T[:3, :3] = R_c2w
            ref_T[:3, 3] = t_c2w
            ref_T = ref_T.tolist()

            ref_pose_list.append(ref_T)

            ref_intrisics_list.append(K_w2c)
        ref_files = [f"{ref_idx}.png" for ref_idx in ref_indices if ref_idx >= 0]

        refer_info[pose_name] = {
            "img_pose": query_T,
            "img_path": query_rgb_path,
            "img_intrisic": K_w2c,
            "img_depth": os.path.join(reference_depth_path, f"{os.path.splitext(pose_name)[0].split('_')[-1]}_1{os.path.splitext(pose_name)[-1]}"),
            "ref_info":{"ref_name": ref_files,
                        "ref_rgb": [os.path.join(reference_rgb_path, f"{os.path.splitext(ref)[0]}_0{os.path.splitext(ref)[-1]}") for ref in ref_files],
                        "ref_depth": [os.path.join(reference_depth_path, f"{os.path.splitext(ref)[0]}_1{os.path.splitext(ref)[-1]}") for ref in ref_files],
                        "ref_poses" : ref_pose_list,
                        "ref_intrinsics": ref_intrisics_list}
        }


    # Save JSON
    save_path = os.path.join(sequence_folder, output_json)
    with open(save_path, 'w') as json_file:
        json.dump(refer_info, json_file, indent=4)
def load_poses(pose_file):
    """Load poses from the pose file."""
    pose_dict = {}
    with open(pose_file, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line
            parts = line.strip().split()
            if parts:  # Ensure the line is not empty
                pose_dict[parts[0]] = parts[1: ]  # Add the first element to the name list
    return pose_dict

if __name__ == "__main__":
    folder_path = "/home/ubuntu/Documents/code/FPVLoc/datasets/Aero/Query"
    ref_path = "/home/ubuntu/Documents/code/FPVLoc/datasets/Aero/Reference"
    sequence_folder = ["Jan_seq1", "Jan_seq2"]  # Replace with your actual query folder path

    for seq in sequence_folder:
        seq_dirs = os.listdir(os.path.join(folder_path, seq))
        # get reference number
        ref_pose_dict = load_poses(os.path.join(ref_path, seq, 'pose.txt'))
        for folder_index, folder in enumerate(seq_dirs):
            seq_dir = os.path.join(folder_path, seq, folder)
            output_json = "refer_info.json"
            generate_refer_info(seq_dir, output_json, seq, ref_pose_dict)
            print(f"Refer info JSON saved to {output_json}")