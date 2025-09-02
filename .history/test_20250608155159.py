import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载数据
data = np.load("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/data_seed_losses.npz")

# 数据加载
euler_seeds = data['init_seed_euler']          # [N, 3]
opt_euler = data['opt_euler']                  # [3, N, 3] → 3轮 LM 输出
losses_all = data['losses']                    # [3, N]
gt_euler = data['gt_euler'].copy()             # [3]
prior_euler = data['prior_euler'].copy()       # [3]

# yaw 修正
gt_euler[2] -= 360
# prior_euler[2] -= 360
# euler_seeds[:, 2] -= 360
# opt_euler[:, :, 2] -= 360

# 绘制每一轮
for iter_idx in range(opt_euler.shape[0]):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # === 初始种子点（灰色小点） ===
    ax.scatter(euler_seeds[:, 0], euler_seeds[:, 1], euler_seeds[:, 2],
               color='gray', s=20, alpha=0.5, label="Init Seeds")

    # === 当前轮的优化种子和 loss ===
    eulers = opt_euler[iter_idx]     # shape: [N, 3]
    losses = losses_all[iter_idx]    # shape: [N]

    ax.scatter(eulers[:, 0], eulers[:, 1], eulers[:, 2],
               c=losses, cmap='viridis', s=50, alpha=0.9, label=f"Iter {iter_idx+1}")

    # === GT & Prior 点 ===
    ax.scatter(gt_euler[0], gt_euler[1], gt_euler[2],
               color='red', marker='*', s=200, label="GT Euler")

    ax.scatter(prior_euler[0], prior_euler[1], prior_euler[2],
               color='blue', marker='s', s=120, label="Prior Euler")

    # === 轴标签 ===
    ax.set_xlabel("Pitch (°)")
    ax.set_ylabel("Roll (°)")
    ax.set_zlabel("Yaw (°)")
    ax.set_title(f"LM Iter {iter_idx+1}: Euler Seeds + Loss Landscape")

    # === 颜色条 & 图例 ===
    cbar = plt.colorbar(ax.collections[-2], pad=0.1)  # 只对当前迭代 loss 点加色条
    cbar.set_label("Loss")
    ax.legend()

    # === 保存图像 ===
    save_path = f"/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/euler_loss_iter{iter_idx+1}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved: {save_path}")
