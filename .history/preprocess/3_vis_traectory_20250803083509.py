
from tqdm import tqdm
from vis_fov_view import batch_fov_visualization

# 读取轨迹文件，拼出 <coordinates> 内容
seq_name = 'switzerland_seq4@8@sunny@100.txt'
trajectory_file = '/mnt/sda/MapScape/query/poses/'+seq_name
kml_output = '/mnt/sda/MapScape/query/trajectory/' + seq_name.split('.')[0] + '.kml'
batch_fov_visualization(trajectory_file, kml_output)
print("Done. Open '.kml' in Google Earth to see multiple FOV pyramids.")
