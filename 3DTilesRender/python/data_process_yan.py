# 定义一个函数来处理文件并生成新的txt文件
def process_file(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
    i = 0
    with open(output_file_path, 'w') as file:
        for line in lines:
            # 移除字符串中的换行符和多余的空格
            line = line.strip()
            # 检查行是否包含所需的数据
            if line and len(line)>1:
                # 分割字符串以获取所需的数据
                parts = line.split()
                # 提取时间戳、经度、纬度、高度、滚转角、俯仰角、偏航角
                timestamp = parts[0] + ' ' + parts[1]
                longitude = parts[3]
                latitude = parts[5]
                height = parts[7]
                pitch = parts[9]
                roll = parts[11]
                yaw = parts[13]

                # 将数据写入新的文件
                name = str(i)+'.jpg'
                i += 1
                file.write(f"{name} {pitch} {roll} {yaw} {longitude} {latitude} {height}\n")

# 设置输入和输出文件路径
input_file_path = '/mnt/sda/liujiachong/Production_3/pose/time+pose.txt'  # 这里假设你的文件名为time_pose.txt
output_file_path = '/mnt/sda/liujiachong/Production_3/pose/processed_data.txt'  # 输出文件的名称

# 调用函数处理文件
process_file(input_file_path, output_file_path)