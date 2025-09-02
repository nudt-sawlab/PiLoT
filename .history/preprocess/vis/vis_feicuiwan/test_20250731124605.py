import cv2
import numpy as np

def draw_arc_through_points(img, pt1, pt2, color=(0, 255, 0), thickness=2, 
                            arrow_size=8, arc_radius=100, arc_direction='left'):
    """
    画一段穿过 pt1 和 pt2 的圆弧，并贴合起点终点。
    - arc_radius: 决定圆弧“绕多远”
    - arc_direction: 'left' or 'right'，弧线从 pt1 看向 pt2，弯向哪边
    """
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)
    chord = pt2 - pt1
    length = np.linalg.norm(chord)
    if length < 1e-6:
        return

    # 中点和法向
    midpoint = (pt1 + pt2) / 2
    dir_unit = chord / length
    ortho = np.array([-dir_unit[1], dir_unit[0]])
    if arc_direction == 'right':
        ortho = -ortho

    # 计算圆心：使 pt1 和 pt2 位于同一圆弧上，距离圆心等于 arc_radius
    # 半弦长为 l/2，弦高为 h
    half_len = length / 2
    if arc_radius <= half_len:
        arc_radius = half_len + 1e-3  # 避免 sqrt 负数

    h = np.sqrt(arc_radius**2 - half_len**2)
    center = midpoint + ortho * h

    # 计算角度范围
    def get_angle(p):
        v = p - center
        return np.degrees(np.arctan2(v[1], v[0]))

    angle1 = get_angle(pt1)
    angle2 = get_angle(pt2)

    if arc_direction == 'left' and angle2 < angle1:
        angle2 += 360
    if arc_direction == 'right' and angle1 < angle2:
        angle1 += 360

    # 使用 OpenCV 采样圆弧路径
    arc_pts = cv2.ellipse2Poly(
        center=tuple(np.round(center).astype(int)),
        axes=(int(arc_radius), int(arc_radius)),
        angle=0,
        startAngle=int(angle1),
        endAngle=int(angle2),
        delta=2
    )

    # 绘制弧线
    for i in range(len(arc_pts) - 1):
        cv2.line(img, tuple(arc_pts[i]), tuple(arc_pts[i + 1]), color, thickness, cv2.LINE_AA)

    # 箭头方向
    p_tip = np.array(arc_pts[-1], dtype=np.float32)
    p_base = np.array(arc_pts[-5], dtype=np.float32)
    dir_vec = p_tip - p_base
    dir_vec /= np.linalg.norm(dir_vec) + 1e-6
    ortho = np.array([-dir_vec[1], dir_vec[0]])

    left = p_tip - dir_vec * arrow_size + ortho * arrow_size * 0.5
    right = p_tip - dir_vec * arrow_size - ortho * arrow_size * 0.5
    triangle = np.array([p_tip, left, right], dtype=np.int32)
    cv2.fillPoly(img, [triangle], color)

    # 起终点圆圈
    pt1i = tuple(np.round(pt1).astype(int))
    pt2i = tuple(np.round(pt2).astype(int))
    cv2.circle(img, pt1i, 5, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, pt1i, 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pt2i, 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pt2i, 3, (0, 0, 0), -1, cv2.LINE_AA)


img = np.ones((500, 800, 3), dtype=np.uint8) * 255

# 距离近：绕大圈
draw_auto_arc_arrow(img, (200, 200), (205, 205), color=(255, 0, 0), arc_direction='left')

# 距离中等：中弧
draw_auto_arc_arrow(img, (100, 100), (300, 150), color=(0, 128, 0), arc_direction='right')

# 距离远：小弯
draw_auto_arc_arrow(img, (100, 300), (600, 400), color=(0, 0, 255), arc_direction='left')

cv2.imshow("Adaptive Arc Arrows", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



