import cv2
import numpy as np


def draw_curved_arrow(img, pt1, pt2, color=(0, 255, 0), thickness=2, arrow_size=8, bend_scale=0.25):
    """
    绘制从 pt1 到 pt2 的一段平滑弯曲箭头。
    即使距离很近，也会绘制较明显的弯曲。
    - pt1, pt2: 起点终点 (x, y)
    - bend_scale: 控制弯曲强度的比例
    """
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)
    direction = pt2 - pt1
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return  # 距离太近不画

    # 单位方向与法向
    dir_unit = direction / length
    ortho = np.array([-dir_unit[1], dir_unit[0]])

    # 控制点：中点向法向偏移
    midpoint = (pt1 + pt2) / 2
    bend = max(bend_scale * length, 20)  # 最小20像素偏移
    ctrl = midpoint + ortho * bend

    # 二次贝塞尔插值（pt1 → ctrl → pt2）
    curve = []
    for t in np.linspace(0, 1, 100):
        p = (
            (1 - t) ** 2 * pt1 +
            2 * (1 - t) * t * ctrl +
            t ** 2 * pt2
        )
        curve.append(tuple(np.round(p).astype(int)))

    # 画曲线
    for i in range(len(curve) - 1):
        cv2.line(img, curve[i], curve[i + 1], color, thickness, cv2.LINE_AA)

    # 箭头（从最后几个点确定方向）
    p_tip = np.array(curve[-1], dtype=np.float32)
    p_base = np.array(curve[-5], dtype=np.float32)
    dir_vec = p_tip - p_base
    dir_vec /= (np.linalg.norm(dir_vec) + 1e-6)
    ortho = np.array([-dir_vec[1], dir_vec[0]])

    left = p_tip - dir_vec * arrow_size + ortho * arrow_size * 0.5
    right = p_tip - dir_vec * arrow_size - ortho * arrow_size * 0.5
    triangle = np.array([p_tip, left, right], dtype=np.int32)
    cv2.fillPoly(img, [triangle], color)

    # 起点终点圆圈
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

