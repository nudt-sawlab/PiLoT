import pandas as pd

# 读取CSV文件
csv_path = "/mnt/sda/MapScape/query/estimation/result_images/Google/Google.csv"

df = pd.read_csv(csv_path, header=None, names=["File", "Method", "Error1", "Error2", "Recall_1m", "Recall_3m", "Recall_5m"])

# 从文件名中提取天气类型
df["Weather"] = df["File"].str.split('@').str[2]

# 定义指标列
metrics = ["Recall_1m", "Recall_3m", "Recall_5m"]

# 创建一个空的DataFrame来存储汇总数据
summary = pd.DataFrame(columns=["Method", "Sunny (1m, 1°, 3m, 3°, 5m, 5°)", "Cloudy (1m, 1°, 3m, 3°, 5m, 5°)", 
                                "Sunset (1m, 1°, 3m, 3°, 5m, 5°)", "Rainy (1m, 1°, 3m, 3°, 5m, 5°)", 
                                "Foggy (1m, 1°, 3m, 3°, 5m, 5°)", "Night (1m, 1°, 3m, 3°, 5m, 5°)"])

# 按Method和Weather分组并计算每个指标的平均值
for method in df["Method"].unique():
    row = {"Method": method}
    
    # 遍历每个天气条件
    for weather in df["Weather"].unique():
        weather_data = df[(df["Method"] == method) & (df["Weather"] == weather)]
        
        # 计算Recall指标的平均值
        recalls = []
        for metric in metrics:
            recalls.append(weather_data[metric].mean())
        
        # 格式化输出每个指标（1m, 3m, 5m）
        row[weather] = " / ".join([f"{recalls[i]*100:.2f}%" for i in range(len(recalls))])
    
    # 将每个方法的结果添加到汇总表中
    summary = summary.append(row, ignore_index=True)

# 将汇总表保存为新的CSV文件
summary.to_csv("summary_output.csv", index=False)

# 输出结果
print(summary)

