import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import platform
import matplotlib.font_manager as fm

# === 数据 ===
data = {
    "Method": ["Render2ORB", "FPVLoc", "Render2Loc", "Render2RAFT", "PixLoc"],
    "Sunny (1m/3m/5m Recall %)": ["6.75 / 27.48 / 52.50", "92.82 / 99.99 / 100.00", "69.90 / 96.86 / 99.52", "24.08 / 54.16 / 69.77", "46.64 / 89.63 / 96.29"],
    "Cloudy (1m/3m/5m Recall %)": ["3.72 / 23.98 / 55.58", "88.48 / 100.00 / 100.00", "59.22 / 96.13 / 99.04", "14.31 / 41.28 / 57.69", "44.69 / 95.37 / 99.91"],
    "Sunset (1m/3m/5m Recall %)": ["9.42 / 32.98 / 57.01", "92.43 / 99.90 / 99.93", "69.47 / 97.00 / 99.03", "21.06 / 49.90 / 65.29", "46.53 / 92.18 / 97.17"],
    "Rainy (1m/3m/5m Recall %)": ["11.12 / 32.75 / 64.19", "94.28 / 99.98 / 100.00", "77.49 / 96.57 / 98.59", "25.61 / 56.04 / 70.13", "40.21 / 73.33 / 82.99"],
    "Foggy (1m/3m/5m Recall %)": ["8.98 / 30.56 / 56.10", "92.42 / 99.98 / 100.00", "66.66 / 95.74 / 98.27", "19.54 / 47.80 / 62.64", "40.84 / 85.30 / 93.34"],
    "Night (1m/3m/5m Recall %)": ["7.70 / 31.59 / 50.84", "83.10 / 99.94 / 100.00", "55.89 / 91.97 / 96.21", "15.94 / 44.21 / 60.82", "28.37 / 71.02 / 87.75"]
}
df = pd.DataFrame(data)

# === 配置 ===
labels = ['Sunny', 'Cloudy', 'Sunset', 'Rainy', 'Foggy', 'Night']
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]
output_dir = "/mnt/sda/MapScape/query/estimation/result_images/outputs"
os.makedirs(output_dir, exist_ok=True)

# 字体设置
if platform.system() == 'Windows' and os.path.exists(r'C:\Windows\Fonts\Georgia.ttf'):
    label_fp = fm.FontProperties(fname=r'C:\Windows\Fonts\Georgia.ttf', size=16)
else:
    label_fp = fm.FontProperties(family='serif', size=16)
label_color = '#5A4A42'

# 美学参数
colors = {
    '1m': '#F7C9BD',
    '3m': '#E69D8B',
    '5m': '#8C3B2E',
}
alphas = {'1m': 0.35, '3m': 0.25, '5m': 0.15}
markers = {'1m': 'o', '3m': 's', '5m': 'D'}
marker_size = 40
z_orders = {'5m': 1, '3m': 2, '1m': 3}

# === 绘图 ===
for idx, row in df.iterrows():
    method = row["Method"]
    recall_dict = {'1m': [], '3m': [], '5m': []}

    for col in df.columns[1:]:
        vals = [float(v.strip()) for v in row[col].split('/')]
        recall_dict['1m'].append(vals[0])
        recall_dict['3m'].append(vals[1])
        recall_dict['5m'].append(vals[2])

    # 归一化
    for k in recall_dict:
        recall_dict[k] = np.array(recall_dict[k]) / 100

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_ylim(0, 1.0)

    # 网格美化
    grid = np.linspace(0, 1.0, 6)
    ax.set_yticks(grid)
    ax.set_yticklabels([f"{int(g*100)}%" for g in grid], fontsize=10, color='#8C7A6B')
    ax.yaxis.grid(True, color='#C8C5C3', linestyle='--', linewidth=2.0)
    ax.xaxis.grid(True, color='#C8C5C3', linestyle='--', linewidth=2.0)

    # 画每条曲线
    for level in ['5m', '3m', '1m']:
        data = recall_dict[level].tolist() + [recall_dict[level][0]]
        z = z_orders[level]
        ax.fill(angles, data, color=colors[level], alpha=alphas[level], zorder=z)
        ax.plot(angles, data, color=colors[level], linewidth=2, zorder=z + 0.1)
        for ang, val in zip(angles, data):
            ax.scatter(ang, val, color=colors[level], marker=markers[level], s=marker_size, zorder=z + 0.2)

    # 极坐标标签
    label_r = 1.05
    for ang, txt in zip(angles[:-1], labels):
        ha = 'left' if np.cos(ang) >= 0 else 'right'
        va = 'bottom' if np.sin(ang) >= 0 else 'top'
        ax.text(ang, label_r, txt, fontproperties=label_fp, color=label_color, ha=ha, va=va)

    ax.spines['polar'].set_visible(False)
    ax.set_xticks([])

    ax.legend(["Recall@1m", "Recall@3m", "Recall@5m"],
              loc='upper right', bbox_to_anchor=(1.3, 1.1),
              facecolor='white', framealpha=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{method}_radar.png"), dpi=300, bbox_inches='tight')
    plt.close()
