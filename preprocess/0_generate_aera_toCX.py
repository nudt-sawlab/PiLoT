import math
from pyproj import CRS, Transformer
import os 
import yaml
import glob
import re
def get_utm_epsg_from_lonlat(lon, lat):
    """
    根据经纬度 (lon, lat) 计算其对应的 UTM 分带 EPSG 号。
    - 北半球：EPSG:326XX
    - 南半球：EPSG:327XX
    """
    if lon < -180 or lon > 180 or lat < -90 or lat > 90:
        return None
    zone = int(math.floor((lon + 180) / 6)) + 1
    if lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone

def wgs84_to_utm(lon, lat, alt, epsg):
    """
    将 WGS84 坐标 (lon, lat, alt) 转换到对应 UTM 坐标系 (x, y, z)。
    """
    crs_wgs84 = CRS.from_epsg(4326)
    crs_utm   = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return [x, y, alt]

def utm_to_wgs84(x, y, alt, epsg):
    """
    将 UTM 坐标 (x, y, z) 转换回 WGS84 坐标 (lon, lat, alt)。
    """
    crs_utm   = CRS.from_epsg(epsg)
    crs_wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_utm, crs_wgs84, always_xy=True)
    lon, lat = transformer.transform(x, y)
    return [lat, lon, alt]

def get_rectangle_vertices(lon1, lat1, lon2, lat2, alt=0, half_width=1000):
    """
    给定两个经纬度点 (lon1, lat1) 与 (lon2, lat2)，构造一个矩形区域，
    该区域以这两点连线作为中垂线（即直线的两端分别是短边中心），
    且矩形两侧长边距离中垂线 1000 米（因此整个宽度为 2000 米）。
    
    返回的四个顶点顺序为：左侧短边点（A_left）、左侧长边点（B_left）、
    右侧长边点（B_right）、右侧短边点（A_right），其中“左”、“右”是相对于从点 A 到点 B 的方向而言。
    """
    # 使用第一个点的经纬度确定UTM分带 EPSG 编号
    epsg = get_utm_epsg_from_lonlat(lon1, lat1)
    
    # 将两个点转换到 UTM 坐标系
    p1 = wgs84_to_utm(lon1, lat1, alt, epsg)  # A 点
    p2 = wgs84_to_utm(lon2, lat2, alt, epsg)  # B 点

    # 计算 A -> B 的向量及其长度
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    L = math.sqrt(dx**2 + dy**2)
    if L == 0:
        raise ValueError("两点距离为0，无法构造矩形。")
        
    # 单位向量（沿 A->B 方向）
    ux = dx / L
    uy = dy / L

    # 计算垂直于 A->B 的单位向量（右手法则：旋转90度，得到 (-uy, ux)）
    # 该方向表示向左侧（相对于 A->B）的方向
    vx = -uy
    vy = ux

    # 在 p1 和 p2 处各偏移 half_width 得到矩形边界的两个点
    # p1 左侧点
    A_left = (p1[0] + half_width * vx, p1[1] + half_width * vy)
    # p1 右侧点
    A_right = (p1[0] - half_width * vx, p1[1] - half_width * vy)
    # p2 左侧点
    B_left = (p2[0] + half_width * vx, p2[1] + half_width * vy)
    # p2 右侧点
    B_right = (p2[0] - half_width * vx, p2[1] - half_width * vy)

    # 将 UTM 坐标转换回 WGS84 坐标
    A_left_wgs = utm_to_wgs84(A_left[0], A_left[1], alt, epsg)
    A_right_wgs = utm_to_wgs84(A_right[0], A_right[1], alt, epsg)
    B_left_wgs = utm_to_wgs84(B_left[0], B_left[1], alt, epsg)
    B_right_wgs = utm_to_wgs84(B_right[0], B_right[1], alt, epsg)

    # 返回矩形顶点（顺序为：左侧短边、左侧长边、右侧长边、右侧短边）
    # 注：如果需要其它顺序，可根据实际需求调整顺序
    return A_left_wgs, B_left_wgs, B_right_wgs, A_right_wgs
def get_extended_rectangle_vertices(lon1, lat1, lon2, lat2, alt=0, half_width=1000, extension=0):
    """
    给定两个经纬度点 (lon1, lat1) 与 (lon2, lat2)，构造一个矩形区域：
      - 以这两点连线所在直线作为中垂线，直线距离矩形长边 1000 米（即矩形宽度为2000米）。
      - 原始矩形的短边中点正好在 A、B 两点处。
      - 然后将矩形长边在两端各延长 extension 米（本例 extension=1000 米）。
    
    返回的四个顶点顺序为：
      左下（A 端左侧延长后点）、左上（B 端左侧延长后点）、右上（B 端右侧延长后点）、右下（A 端右侧延长后点）。
    
    “左”、“右”方向相对于 A→B 的方向确定，“下”对应 A 端延长后的点，“上”对应 B 端延长后的点。
    """
    # 根据第一个点确定 UTM 分带 EPSG 编号
    epsg = get_utm_epsg_from_lonlat(lon1, lat1)
    
    # 将两个点转换到 UTM 坐标系
    p1 = wgs84_to_utm(lon1, lat1, alt, epsg)  # A 点
    p2 = wgs84_to_utm(lon2, lat2, alt, epsg)  # B 点

    # 计算 A -> B 的向量及其长度
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    L = math.sqrt(dx**2 + dy**2)
    if L == 0:
        raise ValueError("两点距离为0，无法构造矩形。")
        
    # 单位向量（沿 A->B 方向）
    ux = dx / L
    uy = dy / L

    # 计算垂直于 A->B 的单位向量（取 (-uy, ux) 为左侧方向）
    vx = -uy
    vy = ux

    # 原始矩形短边点（分别在 A、B 点沿垂直方向偏移 half_width）
    A_left = (p1[0] + half_width * vx, p1[1] + half_width * vy)
    A_right = (p1[0] - half_width * vx, p1[1] - half_width * vy)
    B_left = (p2[0] + half_width * vx, p2[1] + half_width * vy)
    B_right = (p2[0] - half_width * vx, p2[1] - half_width * vy)

    # 延长：在 A 端（下端）沿 -u 方向延长 extension 米，
    #       在 B 端（上端）沿 +u 方向延长 extension 米
    A_left_ext = (A_left[0] - extension * ux, A_left[1] - extension * uy)
    A_right_ext = (A_right[0] - extension * ux, A_right[1] - extension * uy)
    B_left_ext = (B_left[0] + extension * ux, B_left[1] + extension * uy)
    B_right_ext = (B_right[0] + extension * ux, B_right[1] + extension * uy)

    # 将 UTM 坐标转换回 WGS84 坐标
    A_left_wgs = utm_to_wgs84(A_left_ext[0], A_left_ext[1], alt, epsg)
    A_right_wgs = utm_to_wgs84(A_right_ext[0], A_right_ext[1], alt, epsg)
    B_left_wgs = utm_to_wgs84(B_left_ext[0], B_left_ext[1], alt, epsg)
    B_right_wgs = utm_to_wgs84(B_right_ext[0], B_right_ext[1], alt, epsg)

    # 返回顺序为：左下、左上、右上、右下
    return A_left_wgs, B_left_wgs, B_right_wgs, A_right_wgs
# 示例调用
if __name__ == "__main__":
    # 示例经纬度：点 A 和点 B
    save_path = "/home/ubuntu/Documents/code/FPVLoc/src_open/preprocess/area.txt"
    folder_path = "/home/ubuntu/Documents/code/FPVLoc/src_open/preprocess/datasets"
    yaml_path_list = []
    for file in os.listdir(folder_path):
    # 检查文件后缀是否为 .yaml 或 .yml
        if file.endswith('.yaml'):
            file_path = os.path.join(folder_path, file)
            yaml_path_list.append(file_path)
    # 获取所有 .yaml 和 .yml 文件
    yaml_files = glob.glob(os.path.join(folder_path, '*.yaml')) + glob.glob(os.path.join(folder_path, '*.yml'))

    def sort_key(file_path):
        # 提取文件名（不带路径）
        base_name = os.path.basename(file_path)
        # 用正则表达式匹配文件名中的国家和序号，假设文件名格式为：国家_seq数字@...
        pattern = r"([A-Za-z]+)_seq(\d+)"
        match = re.search(pattern, base_name)
        if match:
            # 将国家名称转换为小写，保证排序时统一
            country = match.group(1).lower()
            seq = int(match.group(2))
        else:
            # 如果不匹配，则使用文件名本身和序号 0 作为排序依据
            country = base_name.lower()
            seq = 0
        return (country, seq)
    input_lines = []
    # 按照国家和 seq 编号进行排序
    
    sorted_yaml_files = sorted(yaml_files, key=sort_key)
    with open(save_path, "w", encoding="utf-8") as f1:
        for yaml_path in sorted_yaml_files:
            values = []
            TITLE = yaml_path.split('/')[-1].split('@')[0]

            with open(yaml_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)

            params = cfg['params']

            # 从配置中读取各参数
            ABS_INIT  = params['ABS_INIT']
            lon1, lat1, lon2, lat2 = ABS_INIT

            vertices = get_extended_rectangle_vertices(lon1, lat1, lon2, lat2)

            for vertex in vertices:
                # 只取前两个数值：经度和纬度
                values.extend([str(vertex[0]), str(vertex[1])])

            # 将所有字符串用空格拼接成一行
            output_line = TITLE + " "+ " ".join(values) + "\n"
            input_lines.append(output_line)
            f1.write(output_line)  


# KML文件头与尾部
kml_header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
"""

kml_footer = """
</Document>
</kml>
"""

# 初始化所有 placemark 的内容
placemarks = ""

# 逐行解析输入数据
for line in input_lines:
    tokens = line.split()
    if not tokens:
        continue
    seq_name = tokens[0]
    # 将后面的所有数值转换为浮点型
    coords = list(map(float, tokens[1:]))
    if len(coords) % 2 != 0:
        print(f"Error: {seq_name} 的坐标数目不成对！")
        continue
    # 按 (纬度, 经度) 分组
    pairs = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
    # 每4个点构成一个长方形
    num_rectangles = len(pairs) // 4
    for i in range(num_rectangles):
        pts = pairs[i*4:(i+1)*4]
        # 保证多边形闭合：将第一个点再加入末尾
        pts.append(pts[0])
        # KML 中要求的坐标格式是 经度,纬度,高度；这里高度默认设置为 0
        coord_str = " ".join([f"{lon},{lat},0" for lat, lon in pts])
        placemark = f"""
    <Placemark>
        <name>{seq_name} Rect {i+1}</name>
        <Polygon>
            <outerBoundaryIs>
                <LinearRing>
                    <coordinates>
                        {coord_str}
                    </coordinates>
                </LinearRing>
            </outerBoundaryIs>
        </Polygon>
    </Placemark>
"""
        placemarks += placemark

# 组合完整的 KML 内容
kml_content = kml_header + placemarks + kml_footer

# 写入到 KML 文件
output_filename = "/home/ubuntu/Documents/code/FPVLoc/src_open/preprocess/rectangles.kml"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(kml_content)

print(f"KML 文件 '{output_filename}' 已生成，可以在 Google Earth Pro 中打开。")