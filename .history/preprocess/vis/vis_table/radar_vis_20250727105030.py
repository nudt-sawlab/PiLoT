import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 构造 DataFrame（来自用户粘贴的表格内容）
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

# 创建输出文件夹
output_dir = "/mnt/sda/MapScape/query/estimation/result_images/outputs"
os.makedirs(output_dir, exist_ok=True)

# 雷达图绘制函数
def plot_radar(method_name, scores, labels):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    scores += scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, scores, 'o-', linewidth=2)
    ax.fill(angles, scores, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(f"{method_name}", y=1.08)
    ax.set_ylim(0, 100)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"radar_{method_name}.png"))
    plt.close()

# 主循环：每个方法一张雷达图，六种天气作为维度
weather_labels = ["Sunny", "Cloudy", "Sunset", "Rainy", "Foggy", "Night"]
for idx, row in df.iterrows():
    method = row["Method"]
    weather_scores = []
    for col in df.columns[1:]:
        parts = row[col].split('/')
        values = [float(p.strip()) for p in parts]
        avg = sum(values) / len(values)
        weather_scores.append(avg)
    plot_radar(method, weather_scores, weather_labels)
