import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 图像样式美化设置
rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.titlesize'] = 15
rcParams['axes.labelsize'] = 13
rcParams['legend.fontsize'] = 11

# 加载数据
data = np.load("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/data_seed_losses.npz")

euler_seeds   = data['init_seed_euler']        # [N, 3]，未使用，但你可以加入
opt_euler     = data['opt_euler']              # [3, N, 3]
losses_all    = data['losses']                 # [3, N]
gt_euler      = data['gt_euler'].copy()        # [3]
prior_euler   = data['prior_euler'].copy()     # [3]

# yaw 修正（可选）
gt_euler[2] -= 360

# 遍历每轮优化结果
for iter_idx in range(opt_euler.shape[0]):
    eulers = opt_euler[iter_idx]      # shape: [N, 3]
    losses = losses_all[iter_idx]     # shape: [N]

    pitch = eulers[:, 0]
    yaw   = eulers[:, 2]

    # 绘图
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(pitch, yaw, c=losses, cmap='viridis', s=40, alpha=0.9, label=f"Iter {iter_idx+1}")

    # Prior 点（蓝菱形）
    plt.scatter(prior_euler[0], prior_euler[2], color='royalblue', marker='D', s=100,
                edgecolors='black', linewidths=0.8, label="Prior")

    # GT 点（红星）
    plt.scatter(gt_euler[0], gt_euler[2], color='crimson', marker='*', s=180,
                edgecolors='black', linewidths=1.0, label="GT")

    # 图例与标签
    plt.xlabel("Pitch (°)")
    plt.ylabel("Yaw (°)")
    plt.title(f"LM Iter {iter_idx+1}: Pitch-Yaw Loss Distribution")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.6)

    # 色条（对应 loss）
    cbar = plt.colorbar(sc)
    cbar.set_label("Loss")

    # 保存图像
    save_path = f"/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/euler_loss_iter{iter_idx+1}_pitch_yaw.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved: {save_path}")
