import cv2
import os
import numpy as np

# 配置
ref_dir = "/mnt/sda/MapScape/query/estimation/result_images/feicuiwan/Render_vis/GeoPixel/DJI_20250612194903_0021_V"
query_dir = "/mnt/sda/MapScape/query/estimation/result_images/feicuiwan/Render_vis/Query/DJI_20250612194903_0021_V"
output_path = "/mnt/sda/MapScape/query/estimation/result_images/feicuiwan/outputs"
start_idx, end_idx = 300, 350

# 输出尺寸（与输入图片保持一致）
W, H = 1920, 1080

# 棋盘格参数（行列数）
grid_rows, grid_cols = 4, 6  # 共 96 块格子


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
    if ref is None :
        print(f"[跳过] 帧 {i}")

    ref = cv2.resize(ref, (W, H))
    qry = cv2.resize(qry, (W, H))

    # 棋盘格交错融合
    grid_mix = generate_full_checkerboard(ref, qry, grid_rows, grid_cols)

    name = ref_dir.split('/')[-2]
    cv2.imwrite(f'{output_path}/{i}_{name}.png', grid_mix)
    print(f'{output_path}/{i}_{name}.png')


print(f"✅ 完成：视频已保存到 {output_path}")
