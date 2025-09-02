import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置美观主题
sns.set_style("whitegrid")
rcParams['font.family'] = 'DejaVu Sans'  # 可替换为 'Arial' 或 'Times New Roman'
rcParams['axes.titlesize'] = 16
rcParams['axes.labelsize'] = 14
rcParams['legend.fontsize'] = 12

# 加载数据
data = np.load("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/data_seed_losses.npz")

# 初始欧拉角种子 [N,3]
init_euler = data['init_seed_euler'].copy()
gt_euler = data['gt_euler'].copy()
prior_euler = data['prior_euler'].copy()

# yaw 修正
gt_euler[2] -= 360

# 提取 pitch 和 yaw
pitch = init_euler[:, 0]
yaw   = init_euler[:, 2]

# 创建图像
plt.figure(figsize=(8, 6))
plt.scatter(pitch, yaw, color='lightgray', s=30, alpha=0.7, edgecolors='k', linewidths=0.3, label="Init Seeds")

# Prior 欧拉角（蓝色菱形）
plt.scatter(prior_euler[0], prior_euler[2],
            color='royalblue', marker='D', s=100, edgecolors='black', linewidths=0.8, label='Prior')

# GT 欧拉角（红色五角星）
plt.scatter(gt_euler[0], gt_euler[2],
            color='crimson', marker='*', s=180, edgecolors='black', linewidths=1.0, label='GT')

# 标注设置
plt.xlabel("Pitch (°)")
plt.ylabel("Yaw (°)")
plt.title("Initial Euler Seeds: Pitch-Yaw Distribution")
plt.legend(loc='upper right', frameon=True)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# 坐标轴范围自动/自设
plt.xlim(np.min(pitch)-5, np.max(pitch)+5)
plt.ylim(np.min(yaw)-5, np.max(yaw)+5)

# 保存图像
save_path = "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/init_pitch_yaw_pretty.png"
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()
print(f"✅ 漂亮图像已保存：{save_path}")
