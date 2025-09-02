import cv2
import numpy as np

import cv2
import numpy as np

def draw_detached_loop_arrow(
        img,
        pt1, pt2,
        color        =(0, 255, 0),
        thickness    =2,
        offset_px    =60,    # 在参考距离 loop_ref 时的“推荐”偏移
        lead_px      =25,    # 端点沿切线先冲/退的推荐距离
        loop_ref     =12,    # 参考距离，距离=loop_ref 时 offset_dyn==offset_px
        n_samples    =160):  # Bézier 采样点数
    """
    端点 pt1 ➜ pt2 画“先冲出去再大回环”的箭头，
    距离越远弯曲越轻，距离越近弯曲越夸张，且始终连接到起点和终点。
    """

    # -- 预处理 --
    p1 = np.asarray(pt1, dtype=np.float32)
    p2 = np.asarray(pt2, dtype=np.float32)
    vec = p2 - p1
    dist = np.linalg.norm(vec) + 1e-6  # 防除零

    dir_u   = vec / dist
    ortho_u = np.array([-dir_u[1], dir_u[0]], dtype=np.float32)

    # -- 动态偏移 --
    # 当 dist == loop_ref 时 offset_dyn == offset_px；
    # dist 增大 -> offset_dyn 减小（更平缓）；dist 减小 -> offset_dyn 增大（更夸张）
    offset_dyn = offset_px * (loop_ref / dist)
    # 起点/终点“冲/退”距离，保持原逻辑
    lead_dyn   = min(lead_px, dist * 0.45)

    # -- 计算 Bézier 控制点 --
    c1 = p1 + dir_u * lead_dyn + ortho_u * offset_dyn
    c2 = p2 - dir_u * lead_dyn + ortho_u * offset_dyn

    # -- 采样并绘制 Bézier 曲线 --
    t = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)[:, None]
    curve = ((1 - t)**3 * p1 +
             3 * (1 - t)**2 * t * c1 +
             3 * (1 - t) * t**2 * c2 +
             t**3 * p2).astype(np.int32)

    for a, b in zip(curve[:-1], curve[1:]):
        cv2.line(img, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)

    # -- 箭头和端点 --
    _draw_arrow_head(img, curve[-1], curve[-2], color, max(12, int(offset_dyn // 3)))
    _draw_endpoints(img, p1, p2, color)


# ------------ 工具函数，不动 ------------
def _draw_arrow_head(img, p_tip, p_prev, color, arrow_len):
    """在 p_tip 处画一个实心三角箭头，方向由 p_prev→p_tip"""
    p_tip  = np.asarray(p_tip, dtype=np.float32)
    p_prev = np.asarray(p_prev, dtype=np.float32)
    tan = p_tip - p_prev
    tan /= (np.linalg.norm(tan) + 1e-6)
    ort = np.array([-tan[1], tan[0]], dtype=np.float32)
    left  = p_tip - tan * arrow_len + ort * arrow_len * 0.5
    right = p_tip - tan * arrow_len - ort * arrow_len * 0.5
    cv2.fillPoly(img, [np.int32([p_tip, left, right])], color)

def _draw_endpoints(img, p1, p2, color):
    """在起点和终点画黑边白心小圆"""
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

