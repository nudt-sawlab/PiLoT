import cv2
import numpy as np

def draw_detached_loop_arrow(
        img,
        pt1, pt2,
        color         =(0, 255, 0),
        thickness     =2,
        offset_px     =60,    # 提示值：中距离时回环外扩量
        lead_px       =25,    # 提示值：端点沿切线冲/退距离
        min_separation=12,    # 低于此距离就按“整圈”处理
        n_samples     =160):  # Bézier 采样点数，越多越平滑
    """
    端点 pt1 ➜ pt2 画“先冲出去再大回环”的箭头。
    - offset_px/lead_px：提供一个“推荐”尺度；真正用的是动态量 offset_dyn / lead_dyn
    """

    p1 = np.asarray(pt1, dtype=np.float32)
    p2 = np.asarray(pt2, dtype=np.float32)
    vec = p2 - p1
    dist = np.linalg.norm(vec) + 1e-6     # 避免除零

    # ========= 1. 极小距离：画整圈 =========
    if dist < min_separation:
        _draw_full_loop(img, p1, p2, color, thickness, offset_px)
        return

    # ========= 2. 正常距离：三次 Bézier =========
    #   根据端点距离自适应地放大/缩小 offset 和 lead
    offset_dyn = max(offset_px, dist * 1.3)   # 越近越夸张
    lead_dyn   = min(lead_px,   dist * 0.45)  # 不要超过端点间距的一半

    dir_u   = vec / dist
    ortho_u = np.array([-dir_u[1], dir_u[0]], dtype=np.float32)

    c1 = p1 + dir_u * lead_dyn + ortho_u * offset_dyn
    c2 = p2 - dir_u * lead_dyn + ortho_u * offset_dyn

    # ----- 采样 Bézier 曲线 -----
    t = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)[:, None]
    curve = ((1 - t) ** 3 * p1 +
             3 * (1 - t) ** 2 * t * c1 +
             3 * (1 - t) * t ** 2 * c2 +
             t ** 3 * p2).astype(np.int32)

    # 主体曲线
    for a, b in zip(curve[:-1], curve[1:]):
        cv2.line(img, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)

    # 箭头
    _draw_arrow_head(img, curve[-1], curve[-2], color, max(12, offset_dyn // 3))
    _draw_endpoints(img, p1, p2, color)


# ========= 工具函数 =========
def _draw_full_loop(img, p1, p2, color, thickness, offset_px):
    """极小距离时：从 p1 绕 270° 画到 p2，再补箭头"""
    # 选一条与 p1→p2 垂直（或随意）的法向
    dir_u = (p2 - p1)
    dir_u = dir_u / (np.linalg.norm(dir_u) + 1e-6)
    ortho = np.array([-dir_u[1], dir_u[0]], dtype=np.float32)
    center = p1 + ortho * offset_px         # 圆心

    # 270° 顺时针
    start_ang = int(np.degrees(np.arctan2(p1[1] - center[1],
                                          p1[0] - center[0])))
    arc_pts = cv2.ellipse2Poly(tuple(center.astype(int)),
                               (int(offset_px), int(offset_px)),
                               0, start_ang, start_ang + 270, 2)

    for a, b in zip(arc_pts[:-1], arc_pts[1:]):
        cv2.line(img, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)

    if len(arc_pts) >= 2:
        _draw_arrow_head(img, arc_pts[-1], arc_pts[-2], color,
                         max(12, offset_px // 3))
    _draw_endpoints(img, p1, p2, color)


def _draw_arrow_head(img, p_tip, p_prev, color, arrow_len):
    """在 p_tip 处画一个实心三角箭头"""
    p_tip  = p_tip.astype(np.float32)
    p_prev = p_prev.astype(np.float32)
    tan = p_tip - p_prev
    tan /= (np.linalg.norm(tan) + 1e-6)
    ort = np.array([-tan[1], tan[0]], dtype=np.float32)
    left  = p_tip - tan * arrow_len + ort * arrow_len * 0.5
    right = p_tip - tan * arrow_len - ort * arrow_len * 0.5
    cv2.fillPoly(img, [np.int32([p_tip, left, right])], color)


def _draw_endpoints(img, p1, p2, color):
    """端点：黑边+白心小圆"""
    for p in (p1, p2):
        pi = tuple(p.astype(int))
        cv2.circle(img, pi, 7, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, pi, 4, (255, 255, 255), -1, cv2.LINE_AA)


# =============== DEMO ===============
if __name__ == "__main__":
    canvas = np.full((550, 800, 3), 255, np.uint8)

    # 端点只差 1px → 整圈
    draw_detached_loop_arrow(canvas, (200, 200), (201, 201),
                             color=(0, 0, 255), offset_px=70, lead_px=30)

    # 中距离
    draw_detached_loop_arrow(canvas, (120, 320), (700, 350),
                             color=(0, 128, 0), offset_px=90, lead_px=40)

    # 小距离但 > min_separation
    draw_detached_loop_arrow(canvas, (350, 520), (370, 540),
                             color=(255, 0, 0), offset_px=80, lead_px=35)

    cv2.imshow("Detached Loopy Arrows (improved)", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

