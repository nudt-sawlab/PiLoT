import pandas as pd
import numpy as np
import tabulate

def summarize_metrics_by_weather(csv_path):
    df = pd.read_csv(csv_path, header=None, names=[
        "File", "Method", "MedianError", "StdError",
        "Recall@1m", "Recall@3m", "Recall@5m", "Completeness"
    ])

    # 提取天气（@分隔的第三段）
    def extract_weather(file):
        try:
            return str(file).split('@')[2].lower()
        except:
            return "unknown"

    df["Weather"] = df["File"].apply(extract_weather)
    print(df["Weather"].value_counts())
    weather_order = ["sunny", "cloudy", "sunset", "rainy", "foggy", "night"]
    all_methods = df['Method'].unique()
    summary = []

    for method in all_methods:
        row = {"Method": method}
        for weather in weather_order:
            sub = df[(df["Method"] == method) & (df["Weather"] == weather)]
            if not sub.empty:
                print((sub.index + 1).tolist())
                recall_1_values = pd.to_numeric(sub['Recall@1m'], errors='coerce')
                recall_3_values = pd.to_numeric(sub['Recall@3m'], errors='coerce')
                recall_5_values = pd.to_numeric(sub['Recall@5m'], errors='coerce')
                recall_1 = recall_1_values.mean() * 100
                recall_3 = recall_3_values.mean() * 100
                recall_5 = recall_5_values.mean() * 100
                recall_str = f"{recall_1:.2f} / {recall_3:.2f} / {recall_5:.2f}"
            else:
                recall_str = "-"
            row[weather.capitalize()] = recall_str
        summary.append(row)

    summary_df = pd.DataFrame(summary)

    # 打印 LaTeX 格式
    print("\n�� LaTeX Style Table:\n")
    print(tabulate.tabulate(summary_df, headers='keys', tablefmt='latex'))

    # 打印 Markdown 格式
    print("\n�� Markdown Style Table:\n")
    print(tabulate.tabulate(summary_df, headers='keys', tablefmt='github'))

    return summary_df


if __name__ == "__main__":
    csv_file = "/mnt/sda/MapScape/query/estimation/result_images/Google/Google.csv"
    df_summary = summarize_metrics_by_weather(csv_file)
    df_summary.to_csv("/mnt/sda/MapScape/query/estimation/result_images/Google/summary_by_weather.csv", index=False)
