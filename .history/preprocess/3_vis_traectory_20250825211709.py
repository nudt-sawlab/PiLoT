
from tqdm import tqdm
from vis_fov_view import batch_fov_visualization, batch_fov_visualization_with_error

# 读取轨迹文件，拼出 <coordinates> 内容
seq_name = 'DJI_20250804192327_0002_V.txt'
trajectory_file = '/mnt/sda/MapScape/query/estimation/result_images/FPVLoc/'+seq_name
kml_output = '/mnt/sda/MapScape/query/trajectory/' + seq_name.split('.')[0] + '.kml'
batch_fov_visualization_with_error(trajectory_file, kml_output)
print("Done. Open '.kml' in Google Earth to see multiple FOV pyramids.")
