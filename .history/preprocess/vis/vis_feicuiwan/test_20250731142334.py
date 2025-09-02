import cv2
import numpy as np

import cv2
import numpy as np

import numpy as np
import cv2

def draw_variable_curve_arrow(img, pt1, pt2,
                              color=(0,255,0),
                              thickness=2,
                              arrow_size=8,
                              max_dist_for_max_bend=150,
                              min_bend_frac=0.05,
                              max_bend_frac=0.5,
                              small_thresh=50,
                              small_scale=50,
                              arc_direction='left',
                              n_pts=30):
    """
    如果两点距离 < small_thresh，绘制半径 = d*small_scale 的半圆弧箭头；
    否则，按 bend_frac * d 的二次 Bézier 曲线绘制可变弯度箭头。
    参数：
      small_thresh      : 小距离阈值（像素）
      small_scale       : 半圆弧半径 = d * small_scale
      其它参数同原函数含义
    """
    p1 = np.array(pt1, dtype=np.float32)
    p2 = np.array(pt2, dtype=np.float32)
    v = p2 - p1
    d = np.linalg.norm(v)
    if d < 1e-3:
        return

    # 两点距离很小：画半圆弧
    if d < small_thresh:
        R = d * small_scale
        midpoint = (p1 + p2) * 0.5
        u = v / d
        ortho = np.array([-u[1], u[0]])
        if arc_direction != 'left':
            ortho = -ortho

        # 计算圆心
        h = np.sqrt(max(R*R - (d/2)**2, 0.0))
        center = midpoint + ortho * h

        # 计算起点/终点在圆上的角度
        def angle_of(pt):
            return np.degrees(np.arctan2(pt[1]-center[1], pt[0]-center[0]))
        ang1 = angle_of(p1)
        ang2 = angle_of(p2)
        if arc_direction=='left' and ang2 < ang1:   ang2 += 360
        if arc_direction=='right' and ang1 < ang2:  ang1 += 360

        # 在两角度间采样半圆
        angles = np.linspace(ang1, ang2, n_pts)
        arc_pts = [
            (
                int(center[0] + R * np.cos(np.radians(a))),
                int(center[1] + R * np.sin(np.radians(a)))
            )
            for a in angles
        ]
        # 画弧线
        for i in range(len(arc_pts)-1):
            cv2.line(img, arc_pts[i], arc_pts[i+1], color, thickness, cv2.LINE_AA)

        # 画箭头头部
        if len(arc_pts) >= 4:
            tip  = np.array(arc_pts[-1], dtype=np.float32)
            base = np.array(arc_pts[-4], dtype=np.float32)
            dv = tip - base
            dv /= (np.linalg.norm(dv) + 1e-6)
            ort = np.array([-dv[1], dv[0]])
            left  = tip - dv*arrow_size + ort*arrow_size*0.5
            right = tip - dv*arrow_size - ort*arrow_size*0.5
            cv2.fillPoly(img, [np.array([tip,left,right], dtype=np.int32)], color)

        # 起终点小圆
        for p in (p1, p2):
            pi = tuple(p.astype(int))
            cv2.circle(img, pi, 5, (0,0,0), -1, cv2.LINE_AA)
            cv2.circle(img, pi, 3, (255,255,255), -1, cv2.LINE_AA)
        return

    # 距离较大：继续用二次 Bézier 曲线
    # --- 计算 bend_frac ---
    t = np.clip(d / max_dist_for_max_bend, 0.0, 1.0)
    bend_frac = (1 - t) * (max_bend_frac - min_bend_frac) + min_bend_frac

    # 控制点
    u = v / d
    ortho = np.array([-u[1], u[0]])
    mid = (p1 + p2) * 0.5
    ctrl = mid + ortho * (bend_frac * d)

    # 采样并画曲线
    curve = []
    for tt in np.linspace(0, 1, n_pts):
        pt = (1-tt)**2 * p1 + 2*(1-tt)*tt * ctrl + tt**2 * p2
        curve.append((int(pt[0]+0.5), int(pt[1]+0.5)))
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
    draw_variable_curve_arrow(canvas, (200, 200), (221, 221),
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

