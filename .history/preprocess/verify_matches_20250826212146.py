import numpy as np
from scipy.spatial.transform import Rotation as R
from wrapper import  Camera
import cv2
from get_depth import  get_3D_samples, transform_ecef_origin, get_points2D_ECEF_projection
from transform import  WGS84_to_ECEF, get_rotation_enu_in_ecef, visualize_matches, ECEF_to_WGS84
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
    query_files = []
    reference_rgb_path = "/mnt/sda/ycb/Newzealand_seq1@300@0_30@rainy/0_0.png"
    reference_depth_path = "/mnt/sda/ycb/Newzealand_seq1@300@0_30@rainy/0_1.png"
    query_rgb_path = "/mnt/sda/ycb/Newzealand_seq1@300@0_30@rainy/7_0.png"
    query_depth_path = "/mnt/sda/ycb/Newzealand_seq1@300@0_30@rainy/7_1.png"
    pose_txt = "/mnt/sda/ycb/Newzealand_seq1@300@0_30@rainy/Newzealand_seq1@300@0_30@sunny.txt"
    vis_save_path = "/mnt/sda/ycb/"
    pose_dict = load_poses(pose_txt)
    # Generate pairings
    origin = None
    ref_pose_name = reference_rgb_path.split('/')[-1].split('_')[0] +'.jpg'
    query_pose_name = query_rgb_path.split('/')[-1].split('_')[0] +'.jpg'
    ref_pose = pose_dict[ref_pose_name]
    query_pose = pose_dict[query_pose_name]
    # get query pose
    lon, lat, alt, roll, pitch, yaw = map(float, query_pose)

    euler_angles = [pitch, roll, yaw]
    translation = [lon, lat, alt]
    rot_pose_in_ned = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
    rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
    t_c2w = WGS84_to_ECEF(translation)
    query_T = np.eye(4)
    
    query_T[:3, :3] = R_c2w
    query_T[:3, 3] = t_c2w
    query_T[:3, 1] = -query_T[:3, 1]  # Y轴取反，投影后二维原点在左上角
    query_T[:3, 2] = -query_T[:3, 2]  # Z轴取反
    query_T = query_T.tolist()

    query_pose = euler_angles + translation
        # query_T_candidates = add_noise_to_pose(euler_angles, translation, query_T)

    # get query intrinscis
    qcamera = [1600, 1200, 1931.7, 1931.7, 800.0, 600.0]
    
    cam_query = {
            'model': 'PINHOLE',
            'width': 1600,
            'height': 1200,
            'params': [1931.7, 1931.7, 800.0, 600.0] #np.array(K[2:]
            }   
            
    qcamera = Camera.from_colmap(cam_query)
    rcamera = qcamera
    lon, lat, alt, roll, pitch, yaw = map(float, ref_pose)
    euler_angles_ref = [pitch, roll, yaw]
    translation_ref = [lon, lat, alt]
    lon, lat, _ = translation_ref
    rot_pose_in_ned = R.from_euler('xyz', euler_angles_ref, degrees=True).as_matrix()  # ZXY 东北天  
    rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
    t_c2w = WGS84_to_ECEF(translation_ref)
    ref_T = np.eye(4)
    ref_T[:3, :3] = R_c2w
    ref_T[:3, 3] = t_c2w
    ref_T[:3, 1] = -ref_T[:3, 1]  # Y轴取反，投影后二维原点在左上角
    ref_T[:3, 2] = -ref_T[:3, 2]  # Z轴取反
    ref_T = ref_T.tolist()
            

    # get query & ref rgb path/depth
    rgb_image = cv2.imread(query_rgb_path)
    ref_image = cv2.imread(reference_rgb_path)
    ref_depth_image = cv2.imread(reference_depth_path, cv2.IMREAD_UNCHANGED)
    ref_depth_image = cv2.flip(ref_depth_image, 0)
           
    num_samples = 1       
    width, height = qcamera.size
    ey = np.random.randint(0, height, size= num_samples)
    ex = np.random.randint(0, width, size= num_samples)
    points2d_ref = np.column_stack((ex, ey)) 
    points2d_ref = np.array([[1244, 541]])   

    points2d_ref_valid, point3D_from_ref, _, _ = get_3D_samples(points2d_ref, ref_depth_image, ref_T, rcamera)
    points2d_query, _, Points_3D_ECEF_origin, valid = get_points2D_ECEF_projection(np.array(query_T), qcamera, point3D_from_ref, points2d_ref_valid, use_valid = False, num_samples=20000)
    print(point3D_from_ref[0])
    print(ECEF_to_WGS84(point3D_from_ref[0]))
    visualize_matches(rgb_image, ref_image, 
                    points2d_query, 
                    points2d_ref,vis_save_path = vis_save_path)
