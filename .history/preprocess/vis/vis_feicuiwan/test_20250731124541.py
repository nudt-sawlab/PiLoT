import cv2
import numpy as np

def draw_auto_arc_arrow(img, pt1, pt2, color=(0, 255, 0), thickness=2,
                        arrow_size=8, max_radius=150, min_radius=30, arc_direction='left'):
    """
    自动根据距离生成圆弧箭头，距离越近弯曲越大
    - max_radius: 距离非常近时的最大弯曲半径
    - min_radius: 距离很远时的最小弯曲半径
    """
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)
    vec = pt2 - pt1
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return

    # === 自动计算圆弧半径：距离越近越弯 ===
    arc_radius = np.clip(4000 / (length + 1e-6), min_radius, max_radius)

    dir_unit = vec / length
    ortho = np.array([-dir_unit[1], dir_unit[0]])
    if arc_direction == 'right':
        ortho = -ortho

    # 圆心位置
    center = (pt1 + pt2) / 2 + ortho * arc_radius

    # 起终角度
    def angle(p): return np.arctan2(p[1] - center[1], p[0] - center[0])
    theta1 = angle(pt1)
    theta2 = angle(pt2)

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

    # 箭头部分
    p_tip = np.array(arc_pts[-1], dtype=np.float32)
    p_base = np.array(arc_pts[-5], dtype=np.float32)
    dir_vec = p_tip - p_base
    dir_vec /= (np.linalg.norm(dir_vec) + 1e-6)
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



