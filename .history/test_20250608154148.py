import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载文件
data = np.load("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/data_seed_losses_144.npz")

# 数据加载
eulers = data['opt_euler']       # shape: [1, N, 3] 或 [N, 3]
losses = data['losses']          # shape: [N]
gt_euler = data['gt_euler']      # shape: [3]
prior_euler = data['prior_euler']  # shape: [3]

# 若为 batch 维度 [1, N, 3]，去掉 batch
if eulers.ndim == 3:
    eulers = eulers[0]

# yaw 修正（你自己添加的处理）
gt_euler[2] -= 360
prior_euler[2] -= 360  # 与 GT 一致修正

# 拆分欧拉角分量
pitch = eulers[:, 0]
roll  = eulers[:, 1]
yaw   = eulers[:, 2]

# ——— 绘制图像 ———
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 种子点 loss 散点图
sc = ax.scatter(pitch, roll, yaw, c=losses, cmap='viridis', s=40, alpha=0.9)

# GT 欧拉角：红星 *
ax.scatter(gt_euler[0], gt_euler[1], gt_euler[2],
           color='red', marker='*', s=200, label="GT Euler")

# Prior 欧拉角：蓝色方块 s
ax.scatter(prior_euler[0], prior_euler[1], prior_euler[2],
           color='blue', marker='s', s=120, label="Prior Euler")

# 标签设置
ax.set_xlabel("Pitch (°)")
ax.set_ylabel("Roll (°)")
ax.set_zlabel("Yaw (°)")
ax.set_title("Seed Loss Landscape during LM Pose Refinement")

# 颜色条 & 图例
cbar = plt.colorbar(sc, pad=0.1)
cbar.set_label("Loss")
ax.legend()

# 保存为图像
plt.tight_layout()
plt.savefig("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/euler_loss_distribution.png", dpi=300)
plt.close()
