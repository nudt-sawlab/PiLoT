import numpy as np

# ----------------- 参数设置 -----------------
input_filename  = "/mnt/sda/MapScape/query/poses/USA_seq5@8@300-100.txt"               # 原始 pose 数据文件名
output_filename = "/mnt/sda/MapScape/query/poses/USA_seq5@8@300-100@insert.txt"    # 插值后结果输出文件名

# ----------------- 读取原始数据 -----------------
with open(input_filename, "r") as f:
    # 读取所有非空行，并去掉行尾空白
    lines = [line.strip() for line in f if line.strip()]

# 初始化保存文件名和数值数据的容器
filenames = []
data = []

# 解析每一行，将文件名与后续的数值数据分离
for line in lines:
    tokens = line.split()
    filenames.append(tokens[0])
    # 将后续字段转换为浮点数
    numeric_vals = [float(tok) for tok in tokens[1:]]
    data.append(numeric_vals)

# 将数据转换为 NumPy 数组，形状为 (n, 6)，其中 n 为原始行数
data = np.array(data)
n = data.shape[0]

# ----------------- 插值过程 -----------------
# 原始时间标记，假设为 0, 1, 2, ..., n-1
t_old = np.arange(n)
# 新的时间标记，共 2*n - 1 个点，可以在原始数据之间插入中间点
t_new = np.linspace(0, n - 1, 2 * n - 1)

# 对每一列数据进行线性插值
new_data = np.zeros((len(t_new), data.shape[1]))
for i in range(data.shape[1]):
    new_data[:, i] = np.interp(t_new, t_old, data[:, i])

# ----------------- 文件名处理 -----------------
# 对于原始数据点（整数时刻），直接采用原文件名
# 对于插值生成的新点，按照 “interp_索引.jpg” 的格式生成新的文件名
new_filenames = []
for idx, t in enumerate(t_new):
    new_filenames.append(f"{idx}.jpg")

# ----------------- 写入新文件 -----------------
with open(output_filename, "w") as f:
    cnt = 0
    for name, row in zip(new_filenames, new_data):
        # 将每个数值格式化为小数点后 8 位，并用空格分隔
        if cnt % 2 == 1:
            row_str = " ".join([f"{v:.8f}" for v in row])
            f.write(f"{name} {row_str}\n")
        cnt += 1

print(f"插值后生成的 pose 数据已写入 {output_filename}")
