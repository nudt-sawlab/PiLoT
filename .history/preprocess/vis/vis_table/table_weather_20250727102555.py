import pandas as pd

# 替换为你的 CSV 文件路径
csv_path = "/mnt/sda/MapScape/query/estimation/result_images/Google/Google.csv"

# 读取 CSV（你可以根据实际情况设置 header 行数）
df = pd.read_csv(csv_path, header=None, names=[
    "File", "Method", "Error1", "Error2", "Recall_1m", "Recall_3m", "Recall_5m"
])

# 提取天气字段（防止格式异常，做容错处理）
def extract_weather(filename):
    try:
        parts = str(filename).split('@')
        if len(parts) >= 3:
            return parts[2].lower()
        else:
            return "unknown"
    except:
        return "unknown"

df["Weather"] = df["File"].apply(extract_weather)

# 设置固定天气顺序和方法列表
weather_order = ["sunny", "cloudy", "sunset", "rainy", "foggy", "night"]
methods = df["Method"].unique()

# 构建结果表格
summary_rows = []

for method in methods:
    row = {"Method": method}
    for weather in weather_order:
        sub = df[(df["Method"] == method) & (df["Weather"] == weather)]
        if not sub.empty:
            r1 = sub["Recall_1m"].mean() * 100
            r3 = sub["Recall_3m"].mean() * 100
            r5 = sub["Recall_5m"].mean() * 100
            row[weather.capitalize()] = f"{r1:.2f}% / {r3:.2f}% / {r5:.2f}%"
        else:
            row[weather.capitalize()] = "-"
    summary_rows.append(row)

# 转换为 DataFrame 并输出
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

# 如果需要保存为 CSV
summary_df.to_csv("/mnt/sda/MapScape/query/estimation/result_images/Google/weather_recall_summary.csv", index=False)
