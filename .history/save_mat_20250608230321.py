import numpy as np
from scipy.io import savemat

# 加载两个 .npz 文件
data_main = np.load("data_seed_losses.npz")
data_alt  = np.load("data_seed_losses_32.npz")
data_main['gt_euler'][-1] = data_main['gt_euler'][-1] - 360
# 创建一个新字典，合并所有关键数据
mat_data = {
    # 主文件字段
    'prior_euler':      data_main['prior_euler'],  # [3] [pitch, roll, yaw]
    'prior_trans':      data_main['prior_trans'],  #[lon, lat, alt]
    'gt_euler':         data_main['gt_euler'],  # [3]
    'gt_trans':         data_main['gt_trans'],
    
    'init_seed_euler':  data_main['init_seed_euler'],  # [144,  3]
    'init_seed_trans':  data_main['init_seed_trans'],
    
    'opt_euler_1':      data_main['opt_euler'][0],   # [144,  3]
    'losses_1':         data_main['losses'][0],
    
    'opt_euler_2':      data_alt['opt_euler'][0],   # [32,  3]
    'losses_2':         data_alt['losses'][0],
    'opt_euler_3':      data_alt['opt_euler'][1],   # [32,  3]
    'losses_3':         data_alt['losses'][1],
}

# 保存为 .mat 文件
savemat("merged_pose_data.mat", mat_data)

print("✅ 已保存为 merged_pose_data.mat，可在 MATLAB 中使用")
