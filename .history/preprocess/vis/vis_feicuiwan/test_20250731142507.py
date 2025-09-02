import cv2
import numpy as np

import cv2
import numpy as np

import numpy as np
import cv2

import numpy as np
import cv2

def draw_variable_curve_arrow(img, pt1, pt2,
                              color=(0,255,0),
                              thickness=2,
                              arrow_size=8,
                              max_dist_for_max_bend=150,
                              min_bend_frac=0.05,
                              max_bend_frac=0.5,
                              small_thresh=5,
                              small_scale=10,
                              arc_direction='left',
                              n_pts=30):
    """
    小距离 (< small_thresh) 时：从 pt1 出半圆 (半径 = max(d,small_thresh)*small_scale)，
    半圆尾端再连一条直线到 pt2，并在 pt2 画箭头。
    大距离 (>= small_thresh) 时：保持原来的可变弯度二次 Bézier。
    """
    p1 = np.array(pt1, dtype=np.float32)
    p2 = np.array(pt2, dtype=np.float32)
    v = p2 - p1
    d = np.linalg.norm(v)
    if d < 1e-3:
        return

    # === 小距离：画半圆 + 直线 ===
    if d < small_thresh:
        # 半圆半径
        R = max(d, small_thresh) * small_scale
        u = v / d
        ortho = np.array([-u[1], u[0]])
        if arc_direction != 'left':
            ortho = -ortho

        # 圆心在 p1 + ortho*R
        center = p1 + ortho * R

        # 起点角度
        start_ang = np.degrees(np.arctan2(p1[1]-center[1], p1[0]-center[0]))
        # 生成半圆角度序列
        if arc_direction == 'left':
            angles = np.linspace(start_ang, start_ang + 180, n_pts)
        else:
            angles = np.linspace(start_ang, start_ang - 180, n_pts)

        # 半圆轨迹点
        arc_pts = [
            (
                int(center[0] + R * np.cos(np.radians(a))),
                int(center[1] + R * np.sin(np.radians(a)))
            ) for a in angles
        ]
        # 画半圆
        for i in range(len(arc_pts)-1):
            cv2.line(img, arc_pts[i], arc_pts[i+1], color, thickness, cv2.LINE_AA)

        # 半圆尾端到 pt2 的连线
        tail = arc_pts[-1]
        cv2.line(img, tail, tuple(p2.astype(int)), color, thickness, cv2.LINE_AA)

        # 箭头头部
        tip = np.array(p2, dtype=np.float32)
        base = np.array(tail, dtype=np.float32)
        dv = tip - base
        dv /= (np.linalg.norm(dv) + 1e-6)
        ort = np.array([-dv[1], dv[0]])
        left  = tip - dv*arrow_size + ort*arrow_size*0.5
        right = tip - dv*arrow_size - ort*arrow_size*0.5
        cv2.fillPoly(img, [np.array([tip, left, right],dtype=np.int32)], color)

        # 起点/终点高亮圆
        for p in (p1, p2):
            pi = tuple(p.astype(int))
            cv2.circle(img, pi, 5, (0,0,0), -1, cv2.LINE_AA)
            cv2.circle(img, pi, 3, (255,255,255), -1, cv2.LINE_AA)
        return

    # === 大距离：原可变弯度 Bézier ===
    t = np.clip(d / max_dist_for_max_bend, 0.0, 1.0)
    bend_frac = (1 - t) * (max_bend_frac - min_bend_frac) + min_bend_frac

    u = v / d
    ortho = np.array([-u[1], u[0]])
    mid = (p1 + p2) * 0.5
    ctrl = mid + ortho * (bend_frac * d)

    curve = []
    for tt in np.linspace(0, 1, n_pts):
        pt = (1-tt)**2 * p1 + 2*(1-tt)*tt * ctrl + tt**2 * p2
        curve.append((int(pt[0]+0.5), int(pt[1]+0.5)))
    for i in range(len(curve)-1):
        cv2.line(img, curve[i], curve[i+1], color, thickness, cv2.LINE_AA)

    if len(curve) >= 4:
        p_tip  = np.array(curve[-1], dtype=np.float32)
        p_base = np.array(curve[-4], dtype=np.float32)
        dv = p_tip - p_base
        dv /= (np.linalg.norm(dv) + 1e-6)
        ort = np.array([-dv[1], dv[0]])
        left  = p_tip - dv*arrow_size + ort*arrow_size*0.5
        right = p_tip - dv*arrow_size - ort*arrow_size*0.5
        cv2.fillPoly(img, [np.array([p_tip,left,right],dtype=np.int32)], color)

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

