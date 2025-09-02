import pandas as pd
import numpy as np
import tabulate

def summarize_metrics(csv_path):
    df = pd.read_csv(csv_path)

    all_methods = df['Method'].unique()
    summary = []

    for method in all_methods:
        sub = df[df['Method'] == method]
        errors = []

        for _, row in sub.iterrows():
            # 用 median ± std 估计误差分布（假设正态分布）
            med = row['MedianError']
            std = row['StdError']
            est_errors = np.random.normal(loc=med, scale=std, size=2)
            errors.extend(est_errors)

        errors = np.array(errors)

        # Recall 平均值，格式化合并为一列
        recall_1 = sub['Recall@1m'].mean() * 100
        recall_3 = sub['Recall@3m'].mean() * 100
        recall_5 = sub['Recall@5m'].mean() * 100
        recall_str = f"{recall_1:.2f} / {recall_3:.2f} / {recall_5:.2f}"

        completeness = sub['Completeness'].mean() * 100

        summary.append([
            method,
            np.median(errors),
            np.mean(errors),
            np.std(errors),
            recall_str,
            completeness
        ])

    columns = [
        "Method",
        "Median Error (m)",
        "Mean Error (m)",
        "Std Error (m)",
        "Recall@1/3/5m (%)",
        "Completeness (%)"
    ]

    summary_df = pd.DataFrame(summary, columns=columns)

    # 打印 LaTeX 格式
    print("\n📄 LaTeX Style Table:\n")
    print(tabulate.tabulate(summary_df, headers='keys', tablefmt='latex'))

    # 打印 Markdown 格式
    print("\n📋 Markdown Style Table:\n")
    print(tabulate.tabulate(summary_df, headers='keys', tablefmt='github'))

    return summary_df

if __name__ == "__main__":
    csv_file = "/mnt/sda/MapScape/query/estimation/position_result/feicuiwan.csv"
    df_summary = summarize_metrics(csv_file)
    df_summary.to_csv("/mnt/sda/MapScape/query/estimation/position_result/summary_by_method.csv", index=False)

