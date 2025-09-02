

import cv2
import numpy as np

def draw_exaggerated_arc_arrow(img, pt1, pt2, color=(0, 255, 0), thickness=2,
                                arrow_size=10, exaggeration=4.0, min_radius=30,
                                arc_direction='left'):
    """
    绘制一段弯得很夸张的圆弧箭头，从 pt1 → pt2。
    - exaggeration: 决定圆弧有多夸张，越大越绕圈。
    - arc_direction: 'left' or 'right' 控制弯曲方向。
    """

    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)
    chord = pt2 - pt1
    length = np.linalg.norm(chord)
    if length < 1e-6:
        return

    # === 设置夸张的半径（比线长多几倍） ===
    arc_radius = max(length * exaggeration, min_radius)

    # === 计算圆心 ===
    midpoint = (pt1 + pt2) / 2
    dir_unit = chord / length
    ortho = np.array([-dir_unit[1], dir_unit[0]])
    if arc_direction == 'right':
        ortho = -ortho

    half_len = length / 2
    if arc_radius <= half_len:
        arc_radius = half_len + 1e-3

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

    # === 使用 OpenCV 采样弧线点 ===
    arc_pts = cv2.ellipse2Poly(
        tuple(np.round(center).astype(int)),
        (int(arc_radius), int(arc_radius)),
        0,
        int(angle1),
        int(angle2),
        2
    )

    # === 补起止线 ===
    cv2.line(img, tuple(pt1.astype(int)), tuple(arc_pts[0]), color, thickness, cv2.LINE_AA)
    for i in range(len(arc_pts) - 1):
        cv2.line(img, tuple(arc_pts[i]), tuple(arc_pts[i + 1]), color, thickness, cv2.LINE_AA)
    cv2.line(img, tuple(arc_pts[-1]), tuple(pt2.astype(int)), color, thickness, cv2.LINE_AA)

    # === 箭头（从终点反向找一个与其距离≥10px的点）===
    p_tip = np.array(arc_pts[-1], dtype=np.float32)
    for i in range(len(arc_pts) - 2, -1, -1):
        p_base = np.array(arc_pts[i], dtype=np.float32)
        if np.linalg.norm(p_tip - p_base) >= 10:
            break
    else:
        p_base = np.array(arc_pts[0], dtype=np.float32)

    dir_vec = p_tip - p_base
    dir_vec /= (np.linalg.norm(dir_vec) + 1e-6)
    ortho = np.array([-dir_vec[1], dir_vec[0]])
    left = p_tip - dir_vec * arrow_size + ortho * arrow_size * 0.5
    right = p_tip - dir_vec * arrow_size - ortho * arrow_size * 0.5
    triangle = np.array([p_tip, left, right], dtype=np.int32)
    cv2.fillPoly(img, [triangle], color)

    # === 起终点圆圈 ===
    for pt in [pt1, pt2]:
        p = tuple(np.round(pt).astype(int))
        cv2.circle(img, p, 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
        
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
img = np.ones((500, 800, 3), dtype=np.uint8) * 255

draw_exaggerated_arc_arrow(img, (200, 200), (201, 201), color=(0, 0, 255), exaggeration=6)
draw_exaggerated_arc_arrow(img, (100, 300), (700, 320), color=(0, 128, 0), exaggeration=3)
draw_exaggerated_arc_arrow(img, (300, 500), (320, 510), color=(255, 0, 0), exaggeration=10, arc_direction='right')

cv2.imshow("Exaggerated Arc Arrows", img)
cv2.waitKey(0)

