import pandas as pd
import numpy as np
import tabulate

def summarize_metrics(csv_path):
    df = pd.read_csv(csv_path)

    # 提取不同的环境类型（sunny, cloudy, etc.）以及每种方法
    conditions = df['Condition'].unique()
    methods = df['Method'].unique()

    summary = []

    for method in methods:
        method_data = df[df['Method'] == method]
        method_row = [method]

        # 按照环境分类计算每个环境下的Recall和误差值
        for condition in conditions:
            condition_data = method_data[method_data['Condition'] == condition]
            recall_1 = condition_data['Recall@1m'].mean() * 100
            recall_3 = condition_data['Recall@3m'].mean() * 100
            recall_5 = condition_data['Recall@5m'].mean() * 100
            recall_str = f"{recall_1:.2f} / {recall_3:.2f} / {recall_5:.2f}"

            # 计算误差
            errors = []
            for _, row in condition_data.iterrows():
                med = row['MedianError']
                std = row['StdError']
                est_errors = np.random.normal(loc=med, scale=std, size=2)  # 基于正态分布估计误差
                errors.extend(est_errors)

            errors = np.array(errors)
            median_error = np.median(errors)
            mean_error = np.mean(errors)
            std_error = np.std(errors)

            # 将结果添加到该方法行
            method_row.append(f"{median_error:.2f}% / {mean_error:.2f}% / {std_error:.2f}% / {recall_str}")
        
        summary.append(method_row)

    # 创建 DataFrame 来保存结果
    columns = ["Method"]
    columns.extend(conditions)  # 这里添加不同环境条件
    summary_df = pd.DataFrame(summary, columns=columns)

    # 打印 LaTeX 格式
    print("\n📄 LaTeX Style Table:\n")
    print(tabulate.tabulate(summary_df, headers='keys', tablefmt='latex'))

    # 打印 Markdown 格式
    print("\n📋 Markdown Style Table:\n")
    print(tabulate.tabulate(summary_df, headers='keys', tablefmt='github'))

    # 返回最终的 DataFrame
    return summary_df

if __name__ == "__main__":
    csv_file = "/mnt/sda/MapScape/query/estimation/result_images/Google/Google.csv"
    df_summary = summarize_metrics(csv_file)
    df_summary.to_csv("/mnt/sda/MapScape/query/estimation/result_images/Google/summary_by_method.csv", index=False)
