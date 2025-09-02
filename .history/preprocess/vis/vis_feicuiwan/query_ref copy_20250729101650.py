import cv2
import os
import numpy as np

# 配置
ref_dir = "/mnt/sda/MapScape/query/estimation/mp4_compare/USA_seq5_foggy_gif/GT"
query_dir = "/mnt/sda/MapScape/query/estimation/mp4_compare/USA_seq5_foggy_gif/GeoPixel"
output_path = "/mnt/sda/MapScape/query/estimation/mp4_compare/USA_seq5_foggy_gif/vis"
query_pose_file = os.path.join(query_dir, "USA_seq5@8@foggy@200.txt")
ref_pose_file = os.path.join(ref_dir, "USA_seq5@8@cloudy@300-100@200.txt")
start_idx, end_idx = 760, 790

# 输出尺寸（与输入图片保持一致）
W, H = 960, 540
W, H = 960, 540
crop_w, crop_h = 720, 405
x_off = (W - crop_w) // 2
y_off = (H - crop_h) // 2
# 棋盘格参数（行列数）
grid_rows, grid_cols = 9, 16  # 共 96 块格子


def generate_full_checkerboard(ref_img, query_img, rows, cols):
    h, w = ref_img.shape[:2]
    cell_h, cell_w = h // rows, w // cols
    output = np.zeros_like(ref_img)
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            if (r + c) % 2 == 0:
                output[y0:y1, x0:x1] = query_img[y0:y1, x0:x1]
            else:
                output[y0:y1, x0:x1] = ref_img[y0:y1, x0:x1]
    return output
def generate_checkerboard_overlay(ref_crop, query_crop, rows, cols):
    h, w = ref_crop.shape[:2]
    cell_h, cell_w = h // rows, w // cols
    output = np.zeros_like(ref_crop)
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            if (r + c) % 2 == 0:
                output[y0:y1, x0:x1] = query_crop[y0:y1, x0:x1]
            else:
                output[y0:y1, x0:x1] = ref_crop[y0:y1, x0:x1]
    return output

def draw_shadow_text(img, text, org, font_scale=1.0):
    x, y = org
    cv2.putText(img, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 1, cv2.LINE_AA)

# 主循环
for i in range(start_idx, end_idx + 1):
    ref_path = os.path.join(ref_dir, f"{i}_0.png")
    query_path = os.path.join(query_dir, f"{i}_0.png")
    ref = cv2.imread(ref_path)
    qry = cv2.imread(query_path)
    if qry is None :
        print(f"[跳过] 帧 {i}")
        continue

    ref = cv2.resize(ref, (W, H))
    qry = cv2.resize(qry, (W, H))

    # 棋盘格交错融合
    ref = cv2.resize(ref, (W, H))
    qry = cv2.resize(qry, (W, H))
    ref_crop = ref[y_off:y_off+crop_h, x_off:x_off+crop_w]
    qry_crop = qry[y_off:y_off+crop_h, x_off:x_off+crop_w]

    # === 构造棋盘格混合区域 ===
    checker_crop = generate_checkerboard_overlay(ref_crop, qry_crop, grid_rows, grid_cols)

    # === 放入 frame 中间 ===
    frame = ref.copy()
    frame[y_off:y_off+crop_h, x_off:x_off+crop_w] = checker_crop

    # === 可选：小图显示 Query 全图 ===
    # pip_w, pip_h = 320, 180
    # pip_x = W - pip_w - 20
    # pip_y = H - pip_h - 20
    # qry_pip = cv2.resize(qry, (pip_w, pip_h))
    # cv2.rectangle(frame, (pip_x-2, pip_y-2), (pip_x+pip_w+2, pip_y+pip_h+2), (255,255,255), 2)
    # frame[pip_y:pip_y+pip_h, pip_x:pip_x+pip_w] = qry_pip
    name = ref_dir.split('/')[-2]
    cv2.imwrite(f'{output_path}/{i}_0.png', frame)
    print(f'{output_path}/{i}_{name}.png')


print(f"✅ 完成：视频已保存到 {output_path}")
