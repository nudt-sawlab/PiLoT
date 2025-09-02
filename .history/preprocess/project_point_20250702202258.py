# === 修改为你的 txt 文件路径 ===
file_path = "/home/ubuntu/Downloads/object_index/coordinates.txt"

camera_poses = []
object_poses = []

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 跳过第一行（表头）
for line in lines[1:]:
    # 用 split() 自动去除多余空格或制表符
    tokens = line.strip().split()

    if len(tokens) < 18:
        print(f"⚠ 数据列不足，跳过行：{line}")
        continue

    # 无人机位姿
    cam_pose = {
        'file': tokens[0],
        'lon': float(tokens[1]),
        'lat': float(tokens[2]),
        'alt': float(tokens[3]),
        'roll': float(tokens[4]),
        'pitch': float(tokens[5]),
        'yaw': float(tokens[6]),
    }

    # 目标物体位姿与尺寸
    obj_pose = {
        'class': tokens[7],  # 有时为 Spline_CarFlow2.NODE_AddStaticMeshComponent-5
        'subclass': tokens[8],  # 比如 base
        'lon': float(tokens[9]),
        'lat': float(tokens[10]),
        'alt': float(tokens[11]),
        'roll': float(tokens[12]),
        'pitch': float(tokens[13]),
        'yaw': float(tokens[14]),
        'size': {
            'length': float(tokens[15]),
            'width':  float(tokens[16]),
            'height': float(tokens[17])
        }
    }

    camera_poses.append(cam_pose)
    object_poses.append(obj_pose)

# === 示例输出 ===
print("Sample Camera Pose:\n", camera_poses[0])
print("Sample Object Pose:\n", object_poses[0])
