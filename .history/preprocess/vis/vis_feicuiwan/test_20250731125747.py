import cv2
import numpy as np

def draw_curvy_arrow(img,
                     pt1, pt2,
                     color=(0, 255, 0),
                     thickness=2,
                     arrow_len=12,
                     curvature=0.35,      # 0~1，数值越大弯得越厉害
                     min_offset_px=25,     # 端点太近时曲线至少偏离这么多
                     n_samples=60):        # 贝塞尔采样点数，越多越顺滑
    """
    画一条二次贝塞尔弯曲箭头（pt1 ➜ pt2）。
    即使两点只差 1-2 px，也能保证曲线在图上可见。

    参数
    ----
    curvature       : 相对弯曲度（相对于两点距离），0=直线，0.3~0.5 够夸张
    min_offset_px   : 当两点非常近时，曲线侧向至少偏离这么多像素
    arrow_len       : 箭头长度(px)
    n_samples       : 采样点个数
    """
    p1 = np.array(pt1, dtype=np.float32)
    p2 = np.array(pt2, dtype=np.float32)
    vec = p2 - p1
    dist = np.linalg.norm(vec)

    if dist < 1e-3:            # 几乎同一点，直接退出
        return

    # === 计算控制点（决定弯曲程度） ===
    dir_unit  = vec / dist                     # 起点指向终点的单位向量
    ortho_unit = np.array([-dir_unit[1], dir_unit[0]])  # 顺时针 90° 的法向

    # 曲线向法向方向偏移量
    offset = max(dist * curvature, min_offset_px)
    control_pt = (p1 + p2) / 2 + ortho_unit * offset

    # === 二次贝塞尔采样 ===
    t = np.linspace(0.0, 1.0, n_samples).reshape(-1, 1)
    curve = (1 - t) ** 2 * p1 + 2 * (1 - t) * t * control_pt + t ** 2 * p2
    curve = curve.astype(np.int32)

    # === 逐段连线 ===
    for i in range(len(curve) - 1):
        cv2.line(img, tuple(curve[i]), tuple(curve[i + 1]), color, thickness, cv2.LINE_AA)

    # === 箭头（取曲线最后一点的切线方向） ===
    p_tip  = curve[-1].astype(np.float32)
    p_base = curve[-2].astype(np.float32)      # 倒数第二个采样点
    tangent = p_tip - p_base
    if np.linalg.norm(tangent) < 1e-3:         # 极端情况再找前一点
        tangent = curve[-3].astype(np.float32) - p_base
    tangent_unit = tangent / (np.linalg.norm(tangent) + 1e-6)

    ortho_arrow = np.array([-tangent_unit[1], tangent_unit[0]])
    left  = p_tip - tangent_unit * arrow_len + ortho_arrow * arrow_len * 0.5
    right = p_tip - tangent_unit * arrow_len - ortho_arrow * arrow_len * 0.5
    cv2.fillPoly(img, [np.int32([p_tip, left, right])], color)

    # === 起终点圆圈 ===
    for p in (p1, p2):
        p_int = tuple(p.astype(int))
        cv2.circle(img, p_int, 6, (0, 0, 0), -1, cv2.LINE_AA)   # 黑色外圈
        cv2.circle(img, p_int, 4, (255, 255, 255), -1, cv2.LINE_AA)

# ----------------- Demo -----------------
if __name__ == "__main__":
    canvas = np.ones((600, 800, 3), dtype=np.uint8) * 255

    # 端点几乎重合（只差 1 像素）
    draw_curvy_arrow(canvas, (200, 200), (201, 201),
                     color=(0, 0, 255), curvature=0.4)

    # 中等距离
    draw_curvy_arrow(canvas, (100, 300), (700, 320),
                     color=(0, 128, 0), curvature=0.25)

    # 右弯长曲线
    draw_curvy_arrow(canvas, (350, 520), (370, 540),
                     color=(255, 0, 0), curvature=0.6, min_offset_px=40)

    cv2.imshow("Curvy Arrows", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

