import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载数据
data = np.load("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/data_seed_losses.npz")

# 初始欧拉角种子 [N,3]
init_euler = data['init_seed_euler'].copy()
gt_euler = data['gt_euler'].copy()
prior_euler = data['prior_euler'].copy()

# yaw 修正（如果你前面减了360）
gt_euler[2] -= 360

# 拆分
pitch = init_euler[:, 0]
roll  = init_euler[:, 1]
yaw   = init_euler[:, 2]

# 绘制
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 灰色小点（初始种子）
ax.scatter(pitch, roll, yaw, color='gray', s=30, alpha=0.7, label="Init Seeds")

# GT 欧拉角（红星）
ax.scatter(gt_euler[0], gt_euler[1], gt_euler[2],
           color='red', marker='*', s=200, label="GT Euler")

# Prior 欧拉角（蓝方）
ax.scatter(prior_euler[0], prior_euler[1], prior_euler[2],
           color='blue', marker='s', s=120, label="Prior Euler")

# 标签与图例
ax.set_xlabel("Pitch (°)")
ax.set_ylabel("Roll (°)")
ax.set_zlabel("Yaw (°)")
ax.set_title("Initial Euler Seeds Distribution")

ax.legend()
plt.tight_layout()
plt.savefig("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/init_seed_distribution.png", dpi=300)
plt.close()
print("✅ 初始种子图已保存：init_seed_distribution.png")
