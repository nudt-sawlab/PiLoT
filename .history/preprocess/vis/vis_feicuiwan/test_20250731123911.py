import cv2
import numpy as np

def draw_curved_arrow(img, pt1, pt2, color=(0, 255, 0), thickness=2,
                      arrow_size=8, curve_offset=60):
    """
    使用三次贝塞尔曲线画柔和大弧度箭头
    - curve_offset: 控制两侧控制点向外弯曲的距离（越大越柔和）
    """
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)
    direction = pt2 - pt1
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return

    dir_unit = direction / length
    ortho = np.array([-dir_unit[1], dir_unit[0]])

    # 控制点1：pt1 附近，向外偏移
    ctrl1 = pt1 + dir_unit * (0.3 * length) + ortho * curve_offset

    # 控制点2：pt2 附近，向外偏移（同向）
    ctrl2 = pt2 - dir_unit * (0.3 * length) + ortho * curve_offset

    # 三次贝塞尔插值
    curve = []
    for t in np.linspace(0, 1, 100):
        p = (
            (1 - t)**3 * pt1 +
            3 * (1 - t)**2 * t * ctrl1 +
            3 * (1 - t) * t**2 * ctrl2 +
            t**3 * pt2
        )
        curve.append(tuple(np.round(p).astype(int)))

    # 绘制曲线
    for i in range(len(curve) - 1):
        cv2.line(img, curve[i], curve[i + 1], color, thickness, cv2.LINE_AA)

    # 箭头部分
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

