import cv2
import numpy as np

import cv2
import numpy as np

import numpy as np
import cv2

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
                              small_scale=5,
                              arc_direction='left',
                              n_pts=60):
    """
    小距离 (< small_thresh) 时：绘制一个完整的 360° 圆环（半径 = max(d, small_thresh) * small_scale），
    圆环上恰好包含 pt1 和 pt2，并在 pt2 处画箭头指向圆切线方向。
    大距离 (>= small_thresh) 时：保持原来的可变弯度二次 Bézier 箭头。
    """
    p1 = np.array(pt1, dtype=np.float32)
    p2 = np.array(pt2, dtype=np.float32)
    v = p2 - p1
    d = np.linalg.norm(v)
    if d < 1e-3:
        return

    # --- 小距离：绘制完整圆环 + 箭头 ---
    if d < small_thresh:
        # 1) 计算半径和圆心
        R = min(d, small_thresh) * small_scale
        u = v / d
        ortho = np.array([-u[1], u[0]])
        if arc_direction != 'left':
            ortho = -ortho
        midpoint = (p1 + p2) * 0.5
        # 圆心在中点沿法线方向的 h 距离处，使圆经过 p1 和 p2
        h = np.sqrt(max(R*R - (d/2)**2, 0.0))
        center = midpoint + ortho * h

        # 2) 采样整个圆
        circle_pts = [
            (
                int(center[0] + R * np.cos(theta)),
                int(center[1] + R * np.sin(theta))
            )
            for theta in np.linspace(0, 2*np.pi, n_pts, endpoint=False)
        ]
        # 3) 画圆环（闭合）
        for i in range(len(circle_pts)):
            p_curr = circle_pts[i]
            p_next = circle_pts[(i+1) % len(circle_pts)]
            cv2.line(img, p_curr, p_next, color, thickness, cv2.LINE_AA)

        # 4) 在 p2 处画箭头，方向取圆切线方向
        # 计算 p2 在圆上的角度
        ang2 = np.arctan2(p2[1] - center[1], p2[0] - center[0])
        # 切线方向：derivative of (cos, sin) is (-sin, cos)
        tan = np.array([-np.sin(ang2), np.cos(ang2)])
        if arc_direction != 'left':
            tan = -tan
        tan /= (np.linalg.norm(tan) + 1e-6)
        # 构造三角箭头
        tip   = p2
        base  = tip - tan * arrow_size
        ortan = np.array([-tan[1], tan[0]])
        left  = base + ortan * (arrow_size*0.5)
        right = base - ortan * (arrow_size*0.5)
        tri = np.array([tip, left, right], dtype=np.int32)
        cv2.fillPoly(img, [tri], color)

        # 5) 起、终点高亮
        for p in (p1, p2):
            pi = tuple(p.astype(int))
            cv2.circle(img, pi, 5, (0,0,0), -1, cv2.LINE_AA)
            cv2.circle(img, pi, 3, (255,255,255), -1, cv2.LINE_AA)
        return

    # --- 大距离：原可变弯度 Bézier 箭头 ---
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
        tri = np.array([p_tip, left, right], dtype=np.int32)
        cv2.fillPoly(img, [tri], color)

    # 高亮端点
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

