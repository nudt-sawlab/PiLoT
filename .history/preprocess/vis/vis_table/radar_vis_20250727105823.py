import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# 1️⃣ 数据准备：将你的表格填入或读取CSV
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

# 2️⃣ 设置输出路径
output_dir = "./radar_each_method"
os.makedirs(output_dir, exist_ok=True)

# 3️⃣ 美化设置（Seaborn + matplotlib）
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.facecolor"] = "#f5f5f5"
plt.rcParams["savefig.facecolor"] = "#ffffff"

# 4️⃣ 雷达图绘制函数（带三条曲线 + 美化）
def plot_radar_multi_lines(method_name, scores_by_level, labels):
    levels = ["1m", "3m", "5m"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # 调色板
    colors = sns.color_palette("Set2", 3)

    # 每条 Recall 曲线
    for i, level in enumerate(levels):
        values = [score[i] for score in scores_by_level]
        values += values[:1]
        ax.plot(angles, values, label=f"Recall@{level}", color=colors[i], linewidth=2)
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    # 网格美化
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], color="gray", fontsize=10)
    ax.yaxis.grid(True, linestyle="dashed", linewidth=0.5)
    ax.xaxis.grid(True, linestyle="solid", linewidth=0.8)

    ax.spines["polar"].set_visible(False)
    ax.set_title(method_name, fontsize=16, y=1.1)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"radar_{method_name}_multi.png"))
    plt.close()

# 5️⃣ 开始绘图：每个方法一张图
weather_labels = ["Sunny", "Cloudy", "Sunset", "Rainy", "Foggy", "Night"]
for idx, row in df.iterrows():
    method = row["Method"]
    scores_by_weather = []
    for col in df.columns[1:]:
        parts = row[col].split('/')
        values = [float(p.strip()) for p in parts]
        scores_by_weather.append(values)  # [1m, 3m, 5m]
    plot_radar_multi_lines(method, scores_by_weather, weather_labels)
