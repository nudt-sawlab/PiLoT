
from tqdm import tqdm
from vis_fov_view import batch_traj_two_with_axes_tour, extract_track_to_kml, generate_trajectory_js
import simplekml
# 读取轨迹文件，拼出 <coordinates> 内容
seq_name = 'DJI_20250804192327_0002_V.txt'
trajectory_file = '/mnt/sda/MapScape/query/estimation/result_images/FPVLoc/'+seq_name
gt_file = "/mnt/sda/MapScape/query/poses/" +  seq_name
kml_output = '/mnt/sda/MapScape/query/trajectory/' + seq_name.split('.')[0] + '_dyn.kml'
gt_kml = '/mnt/sda/MapScape/query/trajectory/' + seq_name.split('.')[0] + '_gt.kml'
es_kml = '/mnt/sda/MapScape/query/trajectory/' + seq_name.split('.')[0] + '_es.kml'
gt_czml = '/mnt/sda/MapScape/query/trajectory/' + seq_name.split('.')[0] + '_gt.txt'
es_czml = '/mnt/sda/MapScape/query/trajectory/' + seq_name.split('.')[0] + '_es.txt'

error_txt = "/mnt/sda/MapScape/query/estimation/result_images/FPVLoc/DJI_20250804192327_0002_V_error.txt"
# batch_traj_two_with_axes_tour(trajectory_file, gt_file, kml_output)
# print("Done. Open '.kml' in Google Earth to see multiple FOV pyramids.")

# 示例：提取 EST 和 GT
generate_trajectory_js(trajectory_file, es_czml)
generate_trajectory_js(gt_file,  gt_czml)
