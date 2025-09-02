import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载文件
data = np.load("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/data_seed_losses.npz")

eulers = data['opt_euler']     # [N, 3]
losses = data['losses']        # [N]
gt_euler = data['gt_euler']    # [3]
gt_euler = gt_euler -360
# 拆分欧拉角
pitch = eulers[0,:, 0]
roll = eulers[0,:, 1]
yaw = eulers[0,:, 2]

# 绘制 3D 欧拉角 loss 分布图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 所有种子点
sc = ax.scatter(pitch, roll, yaw, c=losses, cmap='viridis', s=40, alpha=0.9)

# GT 欧拉角标记（红色星号）
ax.scatter(gt_euler[0], gt_euler[1], gt_euler[2],
           color='red', marker='*', s=200, label="GT Euler")

# 坐标轴和标签
ax.set_xlabel("Pitch (°)")
ax.set_ylabel("Roll (°)")
ax.set_zlabel("Yaw (°)")
ax.set_title("LM 欧拉角种子 Loss 分布")

# 颜色条 & 图例
cbar = plt.colorbar(sc, pad=0.1)
cbar.set_label("Loss")
ax.legend()

# plt.tight_layout()
# plt.show()
# 保存为图像文件（png格式）
plt.tight_layout()
plt.savefig("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/euler_loss_distribution.png", dpi=300)  # 你可以指定文件路径
plt.close()  # 关闭图像，以释放内存