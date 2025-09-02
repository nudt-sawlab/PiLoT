import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data = np.load("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/data_seed_losses.npz")

# 初始欧拉角种子 [N,3]
init_euler = data['init_seed_euler'].copy()
gt_euler = data['gt_euler'].copy()
prior_euler = data['prior_euler'].copy()

# yaw 修正（你之前减360）
init_euler[:, 2] -= 360
gt_euler[2] -= 360
prior_euler[2] -= 360

# 提取 pitch 和 yaw
pitch = init_euler[:, 0]
yaw   = init_euler[:, 2]

# 绘图
plt.figure(figsize=(8, 6))
plt.scatter(pitch, yaw, color='gray', s=30, alpha=0.7, label="Init Seeds")

# GT 点
plt.scatter(gt_euler[0], gt_euler[2],
            color='red', marker='*', s=150, label='GT Euler')

# Prior 点
plt.scatter(prior_euler[0], prior_euler[2],
            color='blue', marker='s', s=100, label='Prior Euler')

# 坐标标签与图例
plt.xlabel("Pitch (°)")
plt.ylabel("Yaw (°)")
plt.title("Initial Euler Seeds Distribution (Pitch-Yaw Plane)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/init_pitch_yaw.png", dpi=300)
plt.close()

print("✅ 已保存：init_pitch_yaw.png")
