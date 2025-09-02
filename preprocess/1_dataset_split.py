import os
import shutil
import random
from pathlib import Path
import math

def create_directory_structure(base_path, num_folders):
    """Create the directory structure for output folders."""
    for i in range(num_folders):
        folder_path = os.path.join(base_path, f"{i:06d}", "RGB")
        os.makedirs(folder_path, exist_ok=True)

def load_poses(pose_file):
    """Load poses from the pose file."""
    with open(pose_file, 'r') as f:
        poses = f.readlines()
    return poses

def save_poses(output_folder, poses_dict):
    """Save poses into the corresponding output folders."""
    for folder, poses in poses_dict.items():
        output_pose_file = os.path.join(output_folder, folder, 'pose.txt')
        with open(output_pose_file, 'w') as f:
            f.writelines(poses)

def main():
    # Input directories and pose file
    # .(Query path)(Raw file)
    # ├── Jan_seq2
    # │   ├── Jan_seq2_cloudy_day
    # │   ├── Jan_seq2_fog_day
    # │   ├── Jan_seq2_rain_day
    # │   ├── Jan_seq2_sand_day
    # │   ├── Jan_seq2_snow
    # │   └── Jan_seq2_sunset
    name = 'Jan_seq2'
    folder_path = "/home/ubuntu/Documents/code/FPVLoc/datasets/Aero/"

    query_raw_path = os.path.join(folder_path, "Raw", name)
    # input_dirs = os.listdir(query_raw_path)  # Query path
    input_dirs = [name for name in os.listdir(query_raw_path) if os.path.isdir(os.path.join(query_raw_path, name))]

    pose_file = os.path.join(query_raw_path, "pose.txt")  # Path to the pose file
    # Load poses
    poses = load_poses(pose_file)
    # Output base directory
    output_base = os.path.join(folder_path, "Query", name)

    # num_output_folders = math.ceil(len(poses) / 1000 ) #!
    num_output_folders = 6

    # Create output directory structure
    create_directory_structure(output_base, num_output_folders)

    

    # Map images and poses
    image_pose_mapping = []
    for idx, pose in enumerate(poses):
        if not idx % 30:
            for folder_index, folder in enumerate(input_dirs):
                image_dir = os.path.join(query_raw_path, folder)
                image_file = os.path.join(image_dir, f"{idx}_0.png")

                unique_name = f"{folder_index}_{idx}.png"
                if os.path.exists(image_file):
                    pose_parts = pose.strip().split()
                    pose_parts[0] = unique_name
                    updated_pose = ' '.join(pose_parts) + '\n'
                    image_pose_mapping.append((image_file, updated_pose, unique_name))

    # Shuffle images and poses
    random.shuffle(image_pose_mapping)

    # Distribute images and poses
    output_poses = {f"{i:06d}": [] for i in range(num_output_folders)}
    for i, (image_path, pose, unique_name) in enumerate(image_pose_mapping):
        folder_index = i % num_output_folders
        folder_name = f"{folder_index:06d}"
        output_folder = os.path.join(output_base, folder_name, "RGB")

        # Copy image with new unique name
        output_image_path = os.path.join(output_folder, unique_name)
        shutil.copy(image_path, output_image_path)

        # Save pose
        output_poses[folder_name].append(pose)

    # Save pose files
    save_poses(output_base, output_poses)

if __name__ == "__main__":
    main()
    # query_path = "/home/ubuntu/Documents/code/FPVLoc/datasets/Aero/Query"
    # seq_list = []
    # for root, dirs, _ in os.walk(query_path):
    #     for dir_name in dirs:
    #         folder_path = os.path.join(root, dir_name)
    #         index_list = os.listdir(folder_path)
    #         seq_index_list = dir_name
    #         seq_list += index_list
            # rot_pose_in_ned = R.from_euler('xyz', self.euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
            # rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
            # R_c2w = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
            # q_w2c = rotmat2qvec(R_c2w.transpose())  # return wxyz (colmap pnp return xyzw)
            # # Initialize a 4x4 identity matrix
            # R_w2c_in_enu = R_c2w.transpose() # 和enu的差异是第二行和第三行取负号
            # t_c2w = WGS84_to_ECEF(self.translation)
            # t_w2c = -R_w2c_in_enu.dot(t_c2w)
            # print("after: ", q_w2c, t_w2c)