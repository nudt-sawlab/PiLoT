import cv2
import numpy as np

import cv2
import numpy as np

def draw_variable_curve_arrow(img, pt1, pt2,
                              color=(0,255,0),
                              thickness=2,
                              arrow_size=8,
                              max_dist_for_max_bend=150,
                              min_bend_frac=0.05,
                              max_bend_frac=0.5,
                              n_pts=30):
    """
    两点间画一条二次 Bézier 箭头，bend_frac * distance 决定了控制点偏移量：
      - 当 dist→0 时，bend_frac→max_bend_frac（最夸张弯曲）
      - 当 dist→max_dist_for_max_bend 及更大时，bend_frac→min_bend_frac（近似直线）
    参数：
      pt1, pt2：起点击 (x,y) 和终点 (x,y)
      max_dist_for_max_bend：超过这个距离就只保留最小弯度
      min_bend_frac：最小弯度（弯度 = 控制点偏移 / 距离）
      max_bend_frac：最大弯度
      n_pts：曲线采样点数
    """
    p1 = np.array(pt1, dtype=np.float32)
    p2 = np.array(pt2, dtype=np.float32)
    v = p2 - p1
    d = np.linalg.norm(v)
    if d < 1e-3:
        return

    # 单位方向与法向
    u = v / d
    ortho = np.array([-u[1], u[0]])

    # 计算 bend_frac
    t = np.clip(d / max_dist_for_max_bend, 0.0, 1.0)
    bend_frac = (1 - t) * (max_bend_frac - min_bend_frac) + min_bend_frac

    # 二次 Bézier 控制点
    mid = (p1 + p2) * 0.5
    ctrl = mid + ortho * (bend_frac * d)

    # 采样曲线
    curve = []
    for tt in np.linspace(0, 1, n_pts):
        pt = (1-tt)**2 * p1 + 2*(1-tt)*tt * ctrl + tt**2 * p2
        curve.append((int(pt[0]+0.5), int(pt[1]+0.5)))

    # 画主线
    for i in range(len(curve)-1):
        cv2.line(img, curve[i], curve[i+1], color, thickness, cv2.LINE_AA)

    # 箭头头部
    if len(curve) >= 4:
        p_tip  = np.array(curve[-1], dtype=np.float32)
        p_base = np.array(curve[-4], dtype=np.float32)
        dv = p_tip - p_base
        dv /= (np.linalg.norm(dv) + 1e-6)
        ort = np.array([-dv[1], dv[0]])
        left  = p_tip - dv*arrow_size + ort*arrow_size*0.5
        right = p_tip - dv*arrow_size - ort*arrow_size*0.5
        tri = np.array([p_tip, left, right], dtype=np.int32)
        cv2.fillPoly(img, [tri], color)

    # 起终点小圆
    p1i = tuple(p1.astype(int))
    p2i = tuple(p2.astype(int))
    cv2.circle(img, p1i, 5, (0,0,0), -1, cv2.LINE_AA)
    cv2.circle(img, p1i, 3, (255,255,255), -1, cv2.LINE_AA)
    cv2.circle(img, p2i, 5, (255,255,255), -1, cv2.LINE_AA)
    cv2.circle(img, p2i, 3, (0,0,0), -1, cv2.LINE_AA)

# =============== DEMO ===============
if __name__ == "__main__":
    canvas = np.full((550, 800, 3), 255, np.uint8)

    # 端点只差 1px → 整圈
    draw_variable_curve_arrow(canvas, (200, 200), (201, 201),
                             color=(0, 0, 255))

    # 中距离
    draw_variable_curve_arrow(canvas, (120, 320), (700, 350),
                             color=(0, 128, 0))

    # 小距离但 > min_separation
    draw_variable_curve_arrow(canvas, (350, 520), (370, 540),
                             color=(255, 0, 0))

    cv2.imshow("Detached Loopy Arrows (improved)", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

