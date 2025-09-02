import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import gaussian_kde

# —— 图像美观设置 ——
rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.titlesize'] = 12
rcParams['axes.labelsize'] = 11
rcParams['legend.fontsize'] = 9

# —— 加载数据 ——
data_main = np.load("data_seed_losses.npz", allow_pickle=True)
data_alt  = np.load("data_seed_losses_32.npz", allow_pickle=True)

gt_euler    = data_main['gt_euler'].copy()
prior_euler = data_main['prior_euler'].copy()
init_euler  = data_main['init_seed_euler'].copy()

# 修正 yaw
gt_euler[2]    -= 360

# —— 创建 2x2 子图画布 ——
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# === 子图0：初始种子 ===
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

# === 每轮迭代可视化配置 ===
iters = [
    (data_main, 0, "LM Iter 1"),
    (data_alt,  0, "LM Iter 2"),
    (data_alt,  1, "LM Iter 3"),
]

# === 子图1~3：每轮 LM 优化过程 ===
for idx, (dset, step_idx, title) in enumerate(iters, start=1):
    ax = axes[idx]
    eulers = dset['opt_euler'][step_idx].copy()
    losses = dset['losses'][step_idx].copy()
    pitch = eulers[:, 0]
    yaw   = eulers[:, 2]

    # —— jitter 抗堆叠 ——
    rng = np.random.default_rng(seed=42)
    jitter_scale = 0.2
    pitch_jitter = pitch + rng.normal(0, jitter_scale, size=pitch.shape)
    yaw_jitter   = yaw   + rng.normal(0, jitter_scale, size=yaw.shape)

    # —— 用密度排序美化点图显示层次 ——
    xy = np.vstack([pitch_jitter, yaw_jitter])
    density = gaussian_kde(xy)(xy)
    idx_sort = density.argsort()
    p_sorted = pitch_jitter[idx_sort]
    y_sorted = yaw_jitter[idx_sort]
    l_sorted = losses[idx_sort]

    sc = ax.scatter(p_sorted, y_sorted, c=l_sorted, cmap='viridis',
                    s=40, alpha=0.6, edgecolors='none')

    # 高亮 GT / Prior
    ax.scatter(prior_euler[0], prior_euler[2], color='royalblue', marker='D', s=100,
               edgecolors='black', linewidths=0.8)
    ax.scatter(gt_euler[0], gt_euler[2], color='crimson', marker='*', s=200,
               edgecolors='black', linewidths=1.2)

    ax.set_title(title)
    ax.set_xlabel("Pitch (°)")
    ax.set_ylabel("Yaw (°)")
    ax.grid(True, linestyle='--', alpha=0.6)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Loss")

# 总标题 & 保存
plt.suptitle("Euler Angle Optimization Progress (Pitch-Yaw View)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("combined_pitch_yaw_iters_pretty.png", dpi=300)
plt.close()

print("✅ 图像保存为：combined_pitch_yaw_iters_pretty.png")
