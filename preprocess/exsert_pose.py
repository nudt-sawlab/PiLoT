import numpy as np

# ----------------- 参数设置 -----------------
input_filename  = "/mnt/sda/MapScape/query/poses/switzerland_seq7@8.txt"               # 原始 pose 数据文件名
output_filename = "/mnt/sda/MapScape/query/poses/switzerland_seq7@8@subsample.txt"    # 插值后结果输出文件名

# ------------------- 读取原始数据 -------------------
with open(input_filename, "r") as f:
    # 读取所有非空行
    lines = [line.strip() for line in f if line.strip()]

n = len(lines)
if n == 0:
    raise ValueError("输入文件中没有数据！")

# ------------------- 采样：减到原来的一半 -------------------
# 若希望得到固定的一半行数，例如 new_n = n//2
# 也可以根据需要保留首尾，比如 new_n = int(np.ceil(n/2))
new_n = n // 2
# 使用 np.linspace 均匀地从 0 到 n-1 选取 new_n 个下标，
# 注意这里 np.linspace 默认生成浮点数，需要转换为整数
indices = np.linspace(0, n - 1, new_n).astype(int)

# 根据采样下标选择相应的行
sampled_lines = [lines[i] for i in indices]

# ------------------- 写入新文件 -------------------
with open(output_filename, "w") as f:
    for line in sampled_lines:
        f.write(line + "\n")

print(f"采样完成，减半后的 pose 数据已保存到 {output_filename}")