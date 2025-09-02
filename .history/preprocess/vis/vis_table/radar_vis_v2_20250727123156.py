import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import rcParams

# === 字体设置 ===
rcParams['font.family'] = 'serif'
rcParams['axes.labelweight'] = 'bold'

# === 数据表格 ===
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

# === 方法名与颜色映射 ===
methods_name = {
    "FPVLoc": "GeoPixel",
    "Render2Loc": "Render2Loc",
    "PixLoc": "PixLoc",
    "Render2RAFT": "Render2RAFT",
    "Render2ORB": "Render2ORB",
}
# methods_color = {
#     "GeoPixel": "#1b9e77",
#     "PixLoc": "#7570b3",
#     "Render2Loc": "#d95f02",
#     "Render2ORB": "#e7298a",
#     "Render2RAFT": "#66a61e"
# }
# methods_color = {
#     "GeoPixel": "#007F49",     # 深绿
#     "PixLoc": "#C0D7E9",       # 淡蓝灰
#     "Render2Loc": "#FDC3BC",   # 淡粉
#     "Render2ORB": "#E5D6E9",   # 淡紫
#     "Render2RAFT": "#FFE0B5"   # 奶油橙
# }
methods_color = {
    "GeoPixel": "#1b9e77",      # 深墨绿，比 #007F49 更柔和、稳定（ColorBrewer 绿色）
    "PixLoc": "#8da9c4",        # 蓝灰色，质感更强，不发白（比 #C0D7E9 深一阶）
    "Render2Loc": "#f4a582",    # 暖粉橘，更有设计感，比 #FDC3BC 更饱和（ColorBrewer）
    "Render2ORB": "#c2a5cf",    # 高级浅紫灰，比 #E5D6E9 更淡雅但有存在感（ColorBrewer 紫）
    "Render2RAFT": "#fee08b"    # 奶油黄橙，比 #FFE0B5 更稳重（ColorBrewer 黄橙）
}
weather_labels = ['Sunny', 'Cloudy', 'Sunset', 'Rainy', 'Foggy', 'Night']
angles = np.linspace(0, 2 * np.pi, len(weather_labels), endpoint=False).tolist()
angles += angles[:1]

output_dir = "/mnt/sda/MapScape/query/estimation/result_images/outputs"
os.makedirs(output_dir, exist_ok=True)

recall_indices = {
    "Recall@1m": 0,
    "Recall@3m": 1,
    "Recall@5m": 2
}

for recall_name, recall_idx in recall_indices.items():
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_ylim(0, 105)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    method_lines = []
    for i, row in df.iterrows():
        method_key = row["Method"]
        method_label = methods_name.get(method_key, method_key)
        color = methods_color[method_label]

        values = []
        for col in df.columns[1:]:
            parts = [float(x.strip()) for x in row[col].split('/')]
            values.append(parts[recall_idx])
        values += values[:1]
        mean_recall = np.mean(values[:-1])
        method_lines.append((mean_recall, method_label, values, color))

    # 排序：高 recall 方法先画，放底下
    method_lines.sort()

    for z, (mean_recall, method_label, values, color) in enumerate(method_lines[::-1]):
        ax.fill(angles, values, color=color, alpha=0.6, linewidth=0, zorder=z * 2)
        ax.plot(angles, values, label=method_label, color=color, linewidth=2, zorder=z * 2 + 1)

    # 设置网格虚线层级
    ax.grid(True, linestyle="--", linewidth=1, color="#AAAAAA", zorder=1000)

    # 去除默认角度刻度
    ax.set_xticks([])

    # 自定义天气标签
    for angle, label in zip(angles[:-1], weather_labels):
        ax.text(angle, 107, label, ha='center', va='center',
                fontsize=13, fontweight='bold', family='serif', zorder=2000)

    # 去除默认 yticks
    ax.set_yticklabels([])

    # 手动绘制极径标签到顶部
    for r in [20, 40, 60, 80, 100]:
        ax.text(np.pi / 2, r, f"{r}%", ha='center', va='bottom',
                fontsize=11, fontweight='bold', family='serif', zorder=2000)
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 105], color="#AAAAAA", linewidth=1, linestyle='--', zorder=1000)
    # 图标题与图例
    ax.set_title(recall_name, fontsize=18, fontweight='bold', family='serif', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12, frameon=True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"radar_{recall_name}_layered_final.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

print(f"✅ 所有雷达图已保存至：{os.path.abspath(output_dir)}")
