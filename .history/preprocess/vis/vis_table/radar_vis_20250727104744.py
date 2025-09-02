import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========== 1. 模拟输入（或从CSV读入） ==========
# 你可以用 pd.read_csv(...) 替换下面的 dataframe 读入
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

# ========== 2. 提取并绘制雷达图 ==========
output_dir = "/mnt/sda/MapScape/query/estimation/result_images/outputs"
import os
os.makedirs(output_dir, exist_ok=True)

# 三个 Recall 维度
recall_levels = ["1m", "3m", "5m"]
angles = np.linspace(0, 2 * np.pi, len(recall_levels), endpoint=False).tolist()
angles += angles[:1]  # 闭合曲线

# 遍历每种天气列
for weather_col in df.columns[1:]:
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    for i, row in df.iterrows():
        method = row["Method"]
        try:
            recall_str = row[weather_col]
            values = [float(v.strip()) for v in recall_str.split("/")]
            if len(values) != 3:
                continue
            values += values[:1]  # 闭合曲线
            ax.plot(angles, values, label=method)
            ax.fill(angles, values, alpha=0.1)
        except:
            continue  # 跳过格式错误行

    # 图设置
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(recall_levels)
    ax.set_ylim(0, 105)
    ax.set_title(weather_col.split()[0], fontsize=14, pad=20)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"radar_{weather_col.split()[0].lower()}.png"))
    plt.close()
