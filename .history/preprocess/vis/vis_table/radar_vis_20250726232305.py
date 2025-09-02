import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 读取CSV文件
csv_path = "/mnt/sda/MapScape/query/estimation/result_images/Google/Google.csv"
df = pd.read_csv(csv_path, header=None, names=["File", "Method", "Error1", "Error2", "Recall_1m", "Recall_3m", "Recall_5m"])

# 从文件名中提取天气类型
df["Weather"] = df["File"].str.split('@').str[2]

# 所有天气类型（建议固定顺序）
weather_order = ["sunny", "cloudy", "sunset", "rainy", "foggy", "night"]
metrics = ["Recall_1m", "Recall_3m", "Recall_5m"]

# 确保输出文件夹存在
output_dir = "./radar_charts"
os.makedirs(output_dir, exist_ok=True)

# 绘图函数
def plot_radar(method_name, values, labels):
    num_vars = len(labels)
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]  # 闭合曲线
    angles += angles[:1]

    # 绘图
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='r', linewidth=2)
    ax.fill(angles, values, color='r', alpha=0.25)

    ax.set_title(f"{method_name}", size=16, y=1.1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)

    ax.set_ylim(0, 1)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{method_name}_radar.png"))
    plt.close()

# 为每种方法绘制雷达图
for method in df["Method"].unique():
    weather_means = []
    for weather in weather_order:
        sub = df[(df["Method"] == method) & (df["Weather"].str.lower() == weather)]
        if not sub.empty:
            avg_recall = sub[["Recall_1m", "Recall_3m", "Recall_5m"]].mean().mean()
        else:
            avg_recall = 0.0
        weather_means.append(avg_recall)

    plot_radar(method, weather_means, [w.capitalize() for w in weather_order])
