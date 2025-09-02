import cv2
import numpy as np

def draw_detached_loop_arrow(img,
                             pt1, pt2,
                             color        =(0, 255, 0),
                             thickness    = 2,
                             offset_px    = 60,   # 最大外扩
                             lead_px      = 25,   # 起点先冲/终点先退
                             n_samples    = 120): # 曲线平滑度
    """
    起点➜终点画一条“先冲出去再大回环”的三次 Bézier 箭头。
    - offset_px：回环距离法向最大偏移
    - lead_px  ：起点沿切线先走的距离，保证曲线不贴着端点
    """

    p1 = np.asarray(pt1, dtype=np.float32)
    p2 = np.asarray(pt2, dtype=np.float32)
    vec = p2 - p1
    dist = np.linalg.norm(vec)

    # ==== 处理几乎重合（或非常近）的情况：改画一整圈 ====
    if dist < lead_px * 1.2:
        # 圆心在端点法向 offset_px 处
        dir_u = np.array([1.0, 0.0])   # 任意占位方向
        if dist >= 1e-3:               # 还是能算单位向量
            dir_u = vec / dist
        ortho = np.array([-dir_u[1], dir_u[0]])
        center = p1 + ortho * offset_px

        # 用半径 offset_px 画 270° 的弧
        start_ang = int(np.degrees(np.arctan2(p1[1] - center[1],
                                              p1[0] - center[0])))
        end_ang   = start_ang + 270     # 顺时针 270°
        arc_pts = cv2.ellipse2Poly(tuple(center.astype(int)),
                                   (int(offset_px), int(offset_px)),
                                   0, start_ang, end_ang, 2)

        # 连线
        for a, b in zip(arc_pts[:-1], arc_pts[1:]):
            cv2.line(img, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)
        # 箭头
        if len(arc_pts) >= 2:
            p_tip  = arc_pts[-1].astype(np.float32)
            p_prev = arc_pts[-2].astype(np.float32)
            _draw_arrow_head(img, p_tip, p_prev, color, max(12, offset_px//3))
        _draw_endpoints(img, p1, p2, color)
        return

    # ==== 正常距离：三次 Bézier ====
    dir_u   = vec / dist
    ortho_u = np.array([-dir_u[1], dir_u[0]])

    # 控制点 1：起点沿切线“冲出去” lead_px，再法向偏移 offset_px
    c1 = p1 + dir_u * lead_px + ortho_u * offset_px
    # 控制点 2：终点沿反方向“退” lead_px，再同向偏移
    c2 = p2 - dir_u * lead_px + ortho_u * offset_px

    # 三次 Bézier 采样
    t = np.linspace(0, 1, n_samples)[:, None]
    curve = ((1-t)**3 * p1 +
             3*(1-t)**2 * t * c1 +
             3*(1-t) * t**2 * c2 +
             t**3 * p2).astype(np.int32)

    for a, b in zip(curve[:-1], curve[1:]):
        cv2.line(img, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)

    # 箭头
    p_tip, p_prev = curve[-1].astype(np.float32), curve[-2].astype(np.float32)
    _draw_arrow_head(img, p_tip, p_prev, color, max(12, offset_px//3))

    _draw_endpoints(img, p1, p2, color)


# --- 小工具函数 -------------------------------------------------
def _draw_arrow_head(img, p_tip, p_prev, color, arrow_len):
    """在 p_tip 处画一个三角箭头，方向由 p_prev→p_tip"""
    tan = p_tip - p_prev
    tan /= (np.linalg.norm(tan) + 1e-6)
    ort = np.array([-tan[1], tan[0]])
    left  = p_tip - tan * arrow_len + ort * arrow_len * 0.5
    right = p_tip - tan * arrow_len - ort * arrow_len * 0.5
    cv2.fillPoly(img, [np.int32([p_tip, left, right])], color)

def _draw_endpoints(img, p1, p2, color):
    """黑边白心小圆"""
    for p in (p1, p2):
        pi = tuple(p.astype(int))
        cv2.circle(img, pi, 7, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, pi, 4, (255, 255, 255), -1, cv2.LINE_AA)


# ---------------- DEMO -----------------
if __name__ == "__main__":
    canvas = np.full((550, 800, 3), 255, np.uint8)

    # 端点只差 1px ⇒ 会自动走整圈
    draw_detached_loop_arrow(canvas, (200, 200), (201, 201),
                             color=(0, 0, 255), offset_px=70, lead_px=30)

    # 中距离：先冲出去 lead_px，再大回环
    draw_detached_loop_arrow(canvas, (120, 320), (700, 350),
                             color=(0, 128, 0), offset_px=90, lead_px=40)

    # 小距离但>lead_px：同样能明显分离
    draw_detached_loop_arrow(canvas, (350, 520), (370, 540),
                             color=(255, 0, 0), offset_px=80, lead_px=35)

    cv2.imshow("Detached Loopy Arrows", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

