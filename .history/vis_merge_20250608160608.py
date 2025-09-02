import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 美化配置
rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.titlesize'] = 12
rcParams['axes.labelsize'] = 11
rcParams['legend.fontsize'] = 9

# 加载数据
data = np.load("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/data_seed_losses.npz")

init_euler   = data['init_seed_euler'].copy()   # [N, 3]
opt_euler    = data['opt_euler']                # [3, N, 3]
losses_all   = data['losses']                   # [3, N]
gt_euler     = data['gt_euler'].copy()          # [3]
prior_euler  = data['prior_euler'].copy()       # [3]

# yaw 修正
gt_euler[2] -= 360

# 创建 2x2 子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# ——— 子图1：初始种子 ———
ax = axes[0]
ax.scatter(init_euler[:, 0], init_euler[:, 2], color='lightgray', s=30, alpha=0.7,
           edgecolors='k', linewidths=0.3, label="Init Seeds")
ax.scatter(prior_euler[0], prior_euler[2], color='royalblue', marker='D', s=100,
           edgecolors='black', linewidths=0.8, label='Prior')
ax.scatter(gt_euler[0], gt_euler[2], color='crimson', marker='*', s=180,
           edgecolors='black', linewidths=1.0, label='GT')
ax.set_title("Initial Seeds (Pitch-Yaw)")
ax.set_xlabel("Pitch (°)")
ax.set_ylabel("Yaw (°)")
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper right')

# ——— 子图2-4：LM优化迭代 ———
for i in range(3):
    ax = axes[i+1]
    eulers = opt_euler[i]     # [N, 3]
    losses = losses_all[i]
    pitch = eulers[:, 0]
    yaw   = eulers[:, 2]

    sc = ax.scatter(pitch, yaw, c=losses, cmap='viridis', s=40, alpha=0.9)
    ax.scatter(prior_euler[0], prior_euler[2], color='royalblue', marker='D', s=100,
               edgecolors='black', linewidths=0.8, label='Prior')
    ax.scatter(gt_euler[0], gt_euler[2], color='crimson', marker='*', s=180,
               edgecolors='black', linewidths=1.0, label='GT')
    ax.set_title(f"LM Iter {i+1}")
    ax.set_xlabel("Pitch (°)")
    ax.set_ylabel("Yaw (°)")
    ax.grid(True, linestyle='--', alpha=0.6)

    # 添加色条（仅当前子图）
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Loss")

# 调整整体布局
plt.suptitle("Euler Angle Distribution: Initial & LM Optimization", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # 给 suptitle 留空间

# 保存图像
save_path = "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/combined_pitch_yaw_grid.png"
plt.savefig(save_path, dpi=300)
plt.close()
print(f"✅ 融合图已保存为：{save_path}")
