import cv2
import numpy as np

def draw_curved_arrow(img, pt1, pt2, color=(0, 255, 0), thickness=2, arrow_size=8, 
                   arc_radius=100, arc_direction='left'):
    """
    使用圆弧绕一圈连接 pt1 → pt2，弯曲方向可选
    - arc_radius: 圆弧的大小（越大弯越大）
    - arc_direction: 'left' / 'right'（从 pt1 看向 pt2，弯向哪边）
    """
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)
    vec = pt2 - pt1
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return

    # 基本方向
    dir_unit = vec / length
    ortho = np.array([-dir_unit[1], dir_unit[0]])
    if arc_direction == 'right':
        ortho = -ortho

    # 圆心在两个点中间偏正交方向 arc_radius 的位置
    center = (pt1 + pt2) / 2 + ortho * arc_radius

    # 极坐标角度
    def angle(p): return np.arctan2(p[1] - center[1], p[0] - center[0])
    theta1 = angle(pt1)
    theta2 = angle(pt2)

    # 保证按逆时针或顺时针方向绘制
    if arc_direction == 'left' and theta2 < theta1:
        theta2 += 2 * np.pi
    if arc_direction == 'right' and theta2 > theta1:
        theta1 += 2 * np.pi

    # 采样圆弧点
    arc_pts = []
    for t in np.linspace(theta1, theta2, 100):
        x = center[0] + arc_radius * np.cos(t)
        y = center[1] + arc_radius * np.sin(t)
        arc_pts.append((int(round(x)), int(round(y))))

    # 绘制曲线
    for i in range(len(arc_pts) - 1):
        cv2.line(img, arc_pts[i], arc_pts[i + 1], color, thickness, cv2.LINE_AA)

    # 箭头
    p_tip = np.array(arc_pts[-1], dtype=np.float32)
    p_base = np.array(arc_pts[-5], dtype=np.float32)
    dir_vec = p_tip - p_base
    dir_vec /= (np.linalg.norm(dir_vec) + 1e-6)
    ortho = np.array([-dir_vec[1], dir_vec[0]])

    left = p_tip - dir_vec * arrow_size + ortho * arrow_size * 0.5
    right = p_tip - dir_vec * arrow_size - ortho * arrow_size * 0.5
    triangle = np.array([p_tip, left, right], dtype=np.int32)
    cv2.fillPoly(img, [triangle], color)

    # 圆圈
    pt1i = tuple(np.round(pt1).astype(int))
    pt2i = tuple(np.round(pt2).astype(int))
    cv2.circle(img, pt1i, 5, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, pt1i, 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pt2i, 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pt2i, 3, (0, 0, 0), -1, cv2.LINE_AA)

img = np.ones((400, 600, 3), dtype=np.uint8) * 255
draw_curved_arrow(img, (150, 200), (151, 201))  # 距离很近
draw_curved_arrow(img, (100, 100), (500, 300))  # 距离较远
cv2.imshow("Curved Arrows", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

