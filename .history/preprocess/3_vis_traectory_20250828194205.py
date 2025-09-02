
from tqdm import tqdm
from vis_fov_view import batch_traj_two_with_axes_tour, extract_track_to_kml, batch_fov_visualization_with_error_tour
import simplekml
# 读取轨迹文件，拼出 <coordinates> 内容
seq_name = 'DJI_20250804192327_0002_V.txt'
trajectory_file = '/mnt/sda/MapScape/query/estimation/result_images/FPVLoc/'+seq_name
gt_file = "/mnt/sda/MapScape/query/poses/" +  seq_name
kml_output = '/mnt/sda/MapScape/query/trajectory/' + seq_name.split('.')[0] + '_dyn.kml'
gt_kml = '/mnt/sda/MapScape/query/trajectory/' + seq_name.split('.')[0] + '_gt.kml'
error_txt = "/mnt/sda/MapScape/query/estimation/result_images/FPVLoc/DJI_20250804192327_0002_V_error.txt"
# batch_traj_two_with_axes_tour(trajectory_file, gt_file, kml_output)
# print("Done. Open '.kml' in Google Earth to see multiple FOV pyramids.")

# 示例：提取 EST 和 GT
# extract_track_to_kml("traj_est.txt", "est_track.kml", simplekml.Color.yellow, name="EST")
extract_track_to_kml(gt_file,  gt_kml,  simplekml.Color.blue,   name="GT")
