import pandas as pd
import numpy as np
import tabulate

def summarize_metrics(csv_path):
    df = pd.read_csv(csv_path)

    # 获取所有方法和场景类型
    methods = df['Method'].unique()
    environments = ['sunny', 'cloudy', 'sunset', 'rainy', 'foggy', 'night']
    
    # 汇总每个方法的结果
    summary = []

    for method in methods:
        method_data = df[df['Method'] == method]
        row = [method]  # 初始化方法的行
        
        # 遍历每种环境并计算 Recall@1m, Recall@3m, Recall@5m 和误差
        for env in environments:
            env_data = method_data[method_data['Condition'].str.contains(env)]
            
            recall_1 = env_data['Recall@1m'].mean() * 100
            recall_3 = env_data['Recall@3m'].mean() * 100
            recall_5 = env_data['Recall@5m'].mean() * 100
            recall_str = f"{recall_1:.2f} / {recall_3:.2f} / {recall_5:.2f}"
            
            # 计算误差
            errors = []
            for _, row_data in env_data.iterrows():
                med = row_data['MedianError']
                std = row_data['StdError']
                est_errors = np.random.normal(loc=med, scale=std, size=2)
                errors.extend(est_errors)

            errors = np.array(errors)
            median_error = np.median(errors)
            mean_error = np.mean(errors)
            std_error = np.std(errors)

            # 拼接结果
            row.append(f"{median_error:.2f}% / {mean_error:.2f}% / {std_error:.2f}% / {recall_str}")

        summary.append(row)
    
    # 创建 DataFrame
    columns = ['Method'] + [f"{env.capitalize()} (1m, 1°, 3m, 3°, 5m, 5°)" for env in environments]
    summary_df = pd.DataFrame(summary, columns=columns)

    # 输出 LaTeX 格式
    print("\n📄 LaTeX Style Table:\n")
    print(tabulate.tabulate(summary_df, headers='keys', tablefmt='latex'))

    # 输出 Markdown 格式
    print("\n📋 Markdown Style Table:\n")
    print(tabulate.tabulate(summary_df, headers='keys', tablefmt='github'))

    # 返回 DataFrame
    return summary_df

if __name__ == "__main__":
    csv_file = "/mnt/sda/MapScape/query/estimation/result_images/Google/Google.csv"
    df_summary = summarize_metrics(csv_file)
    df_summary.to_csv("/mnt/sda/MapScape/query/estimation/result_images/Google/summary_by_method.csv", index=False)

