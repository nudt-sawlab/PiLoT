#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量读取形如  xxx_0.(png/jpg) 和 xxx_1.(png/pfm) 的配套文件，
分别缩放到 (512, 512)，再保存到指定文件夹。
"""

import os
import sys
import cv2
import numpy as np

# ---------- PFM 工具 ----------
def read_pfm(file_path):
    with open(file_path, 'rb') as f:
        head = f.readline().decode().strip()
        if head not in ('PF', 'Pf'):
            raise IOError(f'{file_path} 不是合法 PFM 文件')

        w, h = map(int, f.readline().decode().strip().split())
        scale = float(f.readline().decode().strip())
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)

        data = np.fromfile(f, endian + 'f')
        img  = data.reshape((h, w, 3)) if head == 'PF' else data.reshape((h, w))
        return img, scale

def write_pfm(file_path, image, scale=1.0):
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    color  = image.ndim == 3 and image.shape[2] == 3
    h, w   = image.shape[:2]
    mode   = 'PF\n' if color else 'Pf\n'

    # 小端机器 scale 写负值是 PF 格式的约定
    if sys.byteorder == 'little':
        scale = -scale

    with open(file_path, 'wb') as f:
        f.write(mode.encode())
        f.write(f'{w} {h}\n'.encode())
        f.write(f'{scale}\n'.encode())
        # 按 PF 习惯，行序需自下而上；若遇到颠倒请改用 np.flipud(image)
        image.tofile(f)

# ---------- 主流程 ----------
def process_folder(src_root: str,
                   dst_root: str,
                   size=(512, 512)):
    os.makedirs(dst_root, exist_ok=True)
    rgb_suffix  = ('_0.png', '_0.jpg', '_0.jpeg')
    depth_suf   = ('_1.png', '_1.pfm')        

    # 建立 RGB -> 深度 的对应表
    rgb_files = [f for f in os.listdir(src_root)
                 if f.lower().endswith(rgb_suffix)]

    for rgb_name in rgb_files:
        prefix, ext = os.path.splitext(rgb_name)       # 例：193_0, .png
        depth_name  = rgb_name.replace('_0', '_1')

        rgb_path   = os.path.join(src_root,  rgb_name)
        # depth_path = os.path.join(src_root, depth_name)
        # if not os.path.exists(depth_path):
        #     print(f'⚠️  未找到深度文件: {depth_name}，跳过。')
        #     continue

        # 1) 读取 RGB
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            print(f'⚠️  无法读取 {rgb_path}')
            continue

        # # 2) 读取深度（png / pfm）
        # if depth_path.lower().endswith('.pfm'):
        #     depth, _ = read_pfm(depth_path)
        # else:
        #     depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        # 3) 缩放
        rgb_resized   = cv2.resize(rgb,   size, interpolation=cv2.INTER_AREA)
        # 深度用最近邻，避免插值破坏取值
        # depth_resized = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)

        # 4) 保存
        rgb_out   = os.path.join(dst_root, rgb_name)
        # depth_out = os.path.join(dst_root, depth_name)

        cv2.imwrite(rgb_out, rgb_resized)

        # if depth_path.lower().endswith('.pfm'):
        #     write_pfm(depth_out + '.pfm', depth_resized, scale=1.0)
        # else:
        #     # png / tiff 等保持原格式
        #     depth_ext = os.path.splitext(depth_name)[1]
        #     cv2.imwrite(depth_out, depth_resized)

        print(f'✅ 已处理 {rgb_name}  →  {os.path.basename(rgb_out)}')

# ------------------ 入口 ------------------
if __name__ == '__main__':
    # 修改为你自己的路径
    SRC_DIR = '/mnt/sda/CityofStars/feicuiwan_3dtiles/overlay/resize'
    DST_DIR = '/mnt/sda/CityofStars/feicuiwan_3dtiles/overlay/resized'

    process_folder(SRC_DIR, DST_DIR, size=(512, 288))