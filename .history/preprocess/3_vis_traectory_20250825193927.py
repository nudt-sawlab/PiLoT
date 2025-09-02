
from tqdm import tqdm
from vis_fov_view import batch_fov_visualization

# 读取轨迹文件，拼出 <coordinates> 内容
seq_name = 'DJI_20250804192327_0002_V.txt'
trajectory_file = '/mnt/sda/MapScape/query/poses/'+seq_name
kml_output = '/mnt/sda/MapScape/query/trajectory/' + seq_name.split('.')[0] + '.kml'
batch_fov_visualization(trajectory_file, kml_output)
print("Done. Open '.kml' in Google Earth to see multiple FOV pyramids.")
