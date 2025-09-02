import cv2
import numpy as np

def draw_curved_arrow(img, pt1, pt2, color=(0, 255, 0), thickness=2,
                      arrow_size=8, curve_height=60):
    """
    绘制从 pt1 到 pt2 的大弧度柔和箭头。
    - curve_height: 控制“弧线往外鼓出多少像素”（越大越张开）
    """
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)
    chord = pt2 - pt1
    length = np.linalg.norm(chord)
    if length < 1e-6:
        return

    dir_unit = chord / length
    ortho = np.array([-dir_unit[1], dir_unit[0]])

    # 控制点在中点处向正交方向突出一定距离（控制弧度）
    midpoint = (pt1 + pt2) / 2
    height = curve_height if length < 80 else length * 0.3
    ctrl = midpoint + ortho * height

    # 二次贝塞尔插值
    curve = []
    for t in np.linspace(0, 1, 100):
        p = (1 - t) ** 2 * pt1 + 2 * (1 - t) * t * ctrl + t ** 2 * pt2
        curve.append(tuple(np.round(p).astype(int)))

    # 绘制主曲线
    for i in range(len(curve) - 1):
        cv2.line(img, curve[i], curve[i + 1], color, thickness, cv2.LINE_AA)

    # 箭头
    p_tip = np.array(curve[-1], dtype=np.float32)
    p_base = np.array(curve[-5], dtype=np.float32)
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

img = np.ones((400, 600, 3), dtype=np.uint8) * 255
draw_curved_arrow(img, (150, 200), (151, 201))  # 距离很近
draw_curved_arrow(img, (100, 100), (500, 300))  # 距离较远
cv2.imshow("Curved Arrows", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

