import numpy as np
import pyproj
def get_CRS():
    wgs84 = pyproj.CRS('EPSG:4326')
    cgcs2000 = pyproj.CRS('EPSG:4529')  # 宁波
    return wgs84, cgcs2000
def wgs84tocgcs2000(trans):
    """Convert coordinates from WGS84 to CGCS2000.
    
    Args:
        trans (list): [lon, lat, height] in WGS84 format
    """
    lon, lat, height = trans  # Unpack the WGS84 coordinates
    
    wgs84, cgcs2000 = get_CRS()
    
    # Create a transformer from WGS84 to CGCS2000
    transformer = pyproj.Transformer.from_crs(wgs84, cgcs2000, always_xy=True)
    # Perform the transformation
    x, y = transformer.transform(lon, lat)
    # Return the transformed coordinates as a list
    return [x, y, height]  # Keep the original height from WGS84    
def cgcs2000towgs84(c2w_t):
    """Convert coordinates from CGCS2000 to WGS84.
    
    Args:
        c2w_t (list): [x, y, z] in CGCS2000 format
    """
    x, y = c2w_t[0], c2w_t[1]
    
    wgs84, cgcs2000 = get_CRS()
    
    transformer = pyproj.Transformer.from_crs(cgcs2000, wgs84, always_xy=True)
    lon, lat = transformer.transform(x, y)
    height = c2w_t[2]
    return [lon, lat, height]
def ECEF_to_WGS84(pos):
    xpjr, ypjr, zpjr = pos
    transprojr = pyproj.Transformer.from_crs(
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        "EPSG:4326",
        always_xy=True,
    )
    lon, lat, height = transprojr.transform(xpjr, ypjr, zpjr, radians=False)
    return [lon, lat, height]  

def WGS84_to_ECEF(pos):
    lon, lat, height = pos
    transprojr = pyproj.Transformer.from_crs(
        "EPSG:4326",
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        always_xy=True,
    )
    xpjr, ypjr, zpjr = transprojr.transform(lon, lat, height, radians=False)
    return [xpjr, ypjr, zpjr]

def get_image_points_cooordinates(o, w, h, yaw, pitch, roll):
    """
    参数:
        o: 图像矩形中心点的[经、纬、高]
        w: 图像宽度（米）
        h: 图像高度（米）
        yaw: 偏航角 (东北天坐标系）
        pitch: 俯仰角（东北天坐标系）
        roll: 偏航角（东北天坐标系）
    返回值:
        图像左上、右上、右下、左下四个顶点的[经、纬、高]
    """

    o = wgs84tocgcs2000(o.copy())

    # 四个顶点在标准坐标系下坐标
    a_point = np.array([-w/2, -h/2, 0])
    b_point = np.array([w/2, -h/2, 0])
    c_point = np.array([w/2, h/2, 0])
    d_point = np.array([-w/2, h/2, 0])

    # 定义旋转矩阵
    rotation_matrix_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    rotation_matrix_pitch = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [ 0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    rotation_matrix_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # 组合旋转矩阵，按照 yaw-pitch-roll 顺序 (np.dot是矩阵乘法)
    rotation_matrix = np.dot(rotation_matrix_roll, np.dot(rotation_matrix_pitch, rotation_matrix_yaw))

    # 执行旋转变换
    o_point = np.array([0, 0, 0])
    t = np.dot(rotation_matrix, a_point)
    o_new_point = cgcs2000towgs84(np.dot(rotation_matrix, o_point) + o)
    a_new_point = cgcs2000towgs84(np.dot(rotation_matrix, a_point) + o)
    a = np.dot(rotation_matrix, a_point) + o
    b_new_point = cgcs2000towgs84(np.dot(rotation_matrix, b_point) + o)
    b = np.dot(rotation_matrix, b_point) + o
    c_new_point = cgcs2000towgs84(np.dot(rotation_matrix, c_point) + o)
    d_new_point = cgcs2000towgs84(np.dot(rotation_matrix, d_point) + o)
    return a_new_point, b_new_point, c_new_point, d_new_point

if __name__ == '__main__':
    # 定义一个三维点
    w = 40
    h = 30
    o = np.array([121.93069801, 29.07695132, 34])  # 底面中心点坐标
    yaw = np.radians(90)    # 偏航
    pitch = np.radians(0)  # 俯仰
    roll = np.radians(0)   # 翻滚
    a_new_point, b_new_point, c_new_point, d_new_point = get_image_points_cooordinates(o, w, h, yaw, pitch, roll)
    print('a b c d\n', a_new_point, '\n', b_new_point, '\n', c_new_point, '\n', d_new_point)
