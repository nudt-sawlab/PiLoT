
from tqdm import tqdm
from vis_fov_view import batch_traj_with_axes_tour, batch_fov_visualization_with_error, batch_fov_visualization_with_error_tour

# 读取轨迹文件，拼出 <coordinates> 内容
seq_name = 'DJI_20250804192327_0002_V.txt'
trajectory_file = '/mnt/sda/MapScape/query/estimation/result_images/FPVLoc/'+seq_name
kml_output = '/mnt/sda/MapScape/query/trajectory/' + seq_name.split('.')[0] + '.kml'
error_txt = "/mnt/sda/MapScape/query/estimation/result_images/FPVLoc/DJI_20250804192327_0002_V_error.txt"
batch_traj_with_axes_tour(trajectory_file, error_txt, kml_output)
print("Done. Open '.kml' in Google Earth to see multiple FOV pyramids.")
