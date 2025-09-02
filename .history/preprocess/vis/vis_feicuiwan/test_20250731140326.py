import cv2
import numpy as np

def draw_wide_loop_arrow(img,
                         pt1, pt2,
                         color=(0, 255, 0),
                         thickness=2,
                         offset_px=50,       # 冲出去多远（像素）
                         n_samples=100):     # 采样点数，够顺滑就行
    """
    起点➜终点画一条“先外扩再回来”的三次 Bézier 弯曲箭头。
    即使两点几乎重合，也能看得见整条曲线。

    offset_px  : 曲线在法向方向的最大偏移量
    """

    p1 = np.asarray(pt1, dtype=np.float32)
    p2 = np.asarray(pt2, dtype=np.float32)
    vec = p2 - p1
    dist = np.linalg.norm(vec)

    if dist < 1e-3:   # 同一点直接跳过
        return

    # === 基础向量 ===
    dir_u   = vec / dist                     # 起点到终点方向
    ortho_u = np.array([-dir_u[1], dir_u[0]])   # 法向

    # === 设计两个控制点，前后各“甩”出去一段 ===
    # 控制点1：起点前方稍微推进一点，再加偏移
    c1 = p1 + dir_u * (0.15 * offset_px) + ortho_u * offset_px
    # 控制点2：终点后方稍微退一点，同样偏移
    c2 = p2 - dir_u * (0.15 * offset_px) + ortho_u * offset_px

    # === 三次 Bézier 采样 ===
    t = np.linspace(0, 1, n_samples)[:, None]
    curve = ((1 - t)**3      * p1 +
             3 * (1 - t)**2 * t * c1 +
             3 * (1 - t)  * t**2 * c2 +
             t**3            * p2).astype(np.int32)

    # === 连线绘制 ===
    for a, b in zip(curve[:-1], curve[1:]):
        cv2.line(img, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)

    # === 箭头（取末两点切线） ===
    p_tip  = curve[-1].astype(np.float32)
    p_prev = curve[-2].astype(np.float32)
    tan_u  = p_tip - p_prev
    tan_u /= (np.linalg.norm(tan_u) + 1e-6)

    arrow_len = max(12, int(offset_px * 0.4))  # 自动放大
    ortho_u2  = np.array([-tan_u[1], tan_u[0]])
    left  = p_tip - tan_u * arrow_len + ortho_u2 * arrow_len * 0.5
    right = p_tip - tan_u * arrow_len - ortho_u2 * arrow_len * 0.5
    cv2.fillPoly(img, [np.int32([p_tip, left, right])], color)

    # === 起终点圆圈 ===
    for p in (p1, p2):
        pi = tuple(p.astype(int))
        cv2.circle(img, pi, 7, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, pi, 4, (255, 255, 255), -1, cv2.LINE_AA)


# ---------------- DEMO ----------------
if __name__ == "__main__":
    canvas = np.full((600, 800, 3), 255, np.uint8)

    # 只差 1 像素的例子
    draw_wide_loop_arrow(canvas, (200, 200), (201, 201),
                         color=(0, 0, 255), offset_px=60)

    # 正常距离
    draw_wide_loop_arrow(canvas, (120, 320), (700, 340),
                         color=(0, 128, 0), offset_px=80)

    # 另一条小距离右上弯
    draw_wide_loop_arrow(canvas, (350, 520), (370, 540),
                         color=(255, 0, 0), offset_px=70)

    cv2.imshow("Wide-Loop Arrows", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

