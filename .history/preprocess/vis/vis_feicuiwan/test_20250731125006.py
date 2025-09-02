import cv2
import numpy as np

def draw_arc_through_points(img, pt1, pt2, color=(0, 255, 0), thickness=2, 
                            arrow_size=8, radius_factor=1.2, min_radius=30, max_radius=500,
                            arc_direction='left'):
    """
    使用 cv2.ellipse2Poly 画一段穿过 pt1 和 pt2 的圆弧连接线
    - arc_radius 根据距离自动决定，距离越近越弯，越远越直
    """
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)
    chord = pt2 - pt1
    length = np.linalg.norm(chord)
    if length < 1e-6:
        return

    # === 自动计算合理弯曲半径 ===
    arc_radius = np.clip(length * radius_factor, min_radius, max_radius)

    # 圆心构造
    midpoint = (pt1 + pt2) / 2
    dir_unit = chord / length
    ortho = np.array([-dir_unit[1], dir_unit[0]])
    if arc_direction == 'right':
        ortho = -ortho

    half_len = length / 2
    if arc_radius <= half_len:
        arc_radius = half_len + 1e-3  # 防止 sqrt 负数

    h = np.sqrt(arc_radius**2 - half_len**2)
    center = midpoint + ortho * h

    def get_angle(p):
        v = p - center
        return np.degrees(np.arctan2(v[1], v[0]))

    angle1 = get_angle(pt1)
    angle2 = get_angle(pt2)

    if arc_direction == 'left' and angle2 < angle1:
        angle2 += 360
    if arc_direction == 'right' and angle1 < angle2:
        angle1 += 360

    # 使用 OpenCV 内置弧线采样函数
    arc_pts = cv2.ellipse2Poly(
        tuple(np.round(center).astype(int)),
        (int(arc_radius), int(arc_radius)),
        0,
        int(angle1),
        int(angle2),
        2
    )

    # 起点对齐（补 pt1）+ 画圆弧
    cv2.line(img, tuple(pt1.astype(int)), tuple(arc_pts[0]), color, thickness, cv2.LINE_AA)
    for i in range(len(arc_pts) - 1):
        cv2.line(img, tuple(arc_pts[i]), tuple(arc_pts[i + 1]), color, thickness, cv2.LINE_AA)
    cv2.line(img, tuple(arc_pts[-1]), tuple(pt2.astype(int)), color, thickness, cv2.LINE_AA)

    # 箭头
    p_tip = np.array(arc_pts[-1], dtype=np.float32)
    p_base = np.array(arc_pts[-5], dtype=np.float32)
    dir_vec = p_tip - p_base
    dir_vec /= np.linalg.norm(dir_vec) + 1e-6
    ortho = np.array([-dir_vec[1], dir_vec[0]])
    left = p_tip - dir_vec * arrow_size + ortho * arrow_size * 0.5
    right = p_tip - dir_vec * arrow_size - ortho * arrow_size * 0.5
    triangle = np.array([p_tip, left, right], dtype=np.int32)
    cv2.fillPoly(img, [triangle], color)

    # 圆圈端点
    pt1i = tuple(np.round(pt1).astype(int))
    pt2i = tuple(np.round(pt2).astype(int))
    cv2.circle(img, pt1i, 5, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, pt1i, 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pt2i, 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pt2i, 3, (0, 0, 0), -1, cv2.LINE_AA)


img = np.ones((500, 800, 3), dtype=np.uint8) * 255

# 距离很近（弯得很大）
draw_arc_through_points(img, (200, 200), (205, 205), color=(0, 0, 255))

# 中等距离（适中弯）
draw_arc_through_points(img, (100, 100), (300, 150), color=(0, 128, 0))

# 距离很远（近似直线）
draw_arc_through_points(img, (100, 400), (700, 450), color=(255, 0, 0))

cv2.imshow("Auto Arc Correct", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
