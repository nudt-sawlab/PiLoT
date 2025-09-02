import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 美观设置
rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.titlesize'] = 12
rcParams['axes.labelsize'] = 11
rcParams['legend.fontsize'] = 9

# === 加载主文件 ===
data_main = np.load("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/data_seed_losses.npz")
data_alt  = np.load("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/data_seed_losses_32.npz")

# 通用参数
gt_euler    = data_main['gt_euler'].copy()
prior_euler = data_main['prior_euler'].copy()
gt_euler[2] -= 360
prior_euler[2] -= 360

# 创建图像
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# === 子图0：初始种子 ===
init_euler = data_main['init_seed_euler'].copy()
init_euler[:, 2] -= 360
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

# === 迭代轮次设置 ===
iters = [
    (data_main, 0, "LM Iter 1"),
    (data_alt, 1, "LM Iter 2"),
    (data_alt, 2, "LM Iter 3"),
]

# === 子图1-3：每轮 LM 优化 ===
for idx, (dset, step_idx, title) in enumerate(iters, start=1):
    ax = axes[idx]
    eulers = dset['opt_euler'][step_idx]
    losses = dset['losses'][step_idx]
    eulers[:, 2] -= 360
    pitch = eulers[:, 0]
    yaw   = eulers[:, 2]

    sc = ax.scatter(pitch, yaw, c=losses, cmap='viridis', s=40, alpha=0.9)
    ax.scatter(prior_euler[0], prior_euler[2], color='royalblue', marker='D', s=100,
               edgecolors='black', linewidths=0.8)
    ax.scatter(gt_euler[0], gt_euler[2], color='crimson', marker='*', s=180,
               edgecolors='black', linewidths=1.0)
    ax.set_title(title)
    ax.set_xlabel("Pitch (°)")
    ax.set_ylabel("Yaw (°)")
    ax.grid(True, linestyle='--', alpha=0.6)

    # 色条
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Loss")

# 总标题 & 布局保存
plt.suptitle("Euler Angle Optimization Progress (Pitch-Yaw View)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
save_path = "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/combined_pitch_yaw_iters.png"
plt.savefig(save_path, dpi=300)
plt.close()
print(f"✅ 总览图已保存到：{save_path}")
