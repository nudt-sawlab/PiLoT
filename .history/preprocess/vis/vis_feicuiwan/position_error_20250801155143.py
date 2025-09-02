import os

import numpy as np
import os
import argparse
import csv
from transform import WGS84_to_ECEF

def evaluate_xyz(results, gt, only_localized=False):
    """
    Compare predicted positions (in WGS84) with GT positions (in WGS84),
    convert to ECEF, compute 2D XY distance error.
    Return error statistics in a dict.
    """
    predictions = {}
    test_names = []
    total_num = 0
    test_num = 0

    if not os.path.exists(results):
        print(f"[Warning] 预测文件不存在: {results}")
        return None
    if not os.path.exists(gt):
        print(f"[Warning] GT文件不存在: {gt}")
        return None

    with open(results, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 4:
                continue
            name = tokens[0].split('/')[-1]
            try:
                t = np.array(tokens[1:4], dtype=float)
                t_ecef = WGS84_to_ECEF(t)
                predictions[name] = t_ecef
                test_names.append(name)
                test_num += 1
            except Exception as e:
                print(f"[Error] 预测读取失败 {name}: {e}")
                continue

    gts = {}
    with open(gt, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 4:
                continue
            name = tokens[0].split('/')[-1]
            try:
                t = np.array(tokens[1:4], dtype=float)
                t_ecef = WGS84_to_ECEF(t)
                gts[name] = t_ecef
                total_num += 1
            except Exception as e:
                print(f"[Error] GT读取失败 {name}: {e}")
                continue

    errors_t = {}
    for name in test_names:
        gt_name = name.split('_')[0] + '_0.png'
        if name not in predictions or gt_name not in gts:
            if only_localized:
                continue
            errors_t[name] = np.inf
        else:
            t_gt = np.array(gts[gt_name])
            t_pred = np.array(predictions[name])
            e_t = np.linalg.norm(t_pred - t_gt)
            errors_t[name] = e_t

    if len(errors_t) == 0:
        return None

    return errors_t
 
def get_xy(save_xy_path):
    xy_dict = {}
    with open(save_xy_path, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line
            parts = line.strip().split()
            if parts:  # Ensure the line is not empty
                x, y = map(float, parts[1: ])
                xy_dict[parts[0]] = [[x, y]] # Add the first element to the name list
    return xy_dict 

def main():
    base_path = "/mnt/sda/MapScape/query/estimation/position_result"
    
    methods = {
        'ORB@per30': 'Render2ORB',
        'FPVLoc': 'Geopixel',
        'Render2loc': 'Render2Loc',
        'Render2loc@raft': 'Render2RAFT',
        'Pixloc': 'PixLoc',
    }

    results_list = []
    gt_path = os.path.join(base_path, "GT")
    render2orb_path = os.path.join(base_path, "ORB@per30")
    txt_files = [f for f in os.listdir(render2orb_path) if f.endswith('.txt') and 'DJI_' in f]

    for txt_file in txt_files:
        # if 'DJI_20250612194622_0018_V' not in txt_file: continue
        seq = txt_file.split('.')[0]
        gt_file = os.path.join(gt_path, f"{seq}_RTK.txt")
        save_xy_path = os.path.join("/mnt/sda/MapScape/query/bbox",seq, seq + '_xy.txt')
        xy_dict = get_xy(save_xy_path)
        print(f"\n-------- Sequence: {seq} --------")

        for method_name, subfolder in methods.items():
            result_file = os.path.join(base_path, method_name, f"{seq}.txt")
            print(f"== Method: {method_name} ==")
            metrics = evaluate_xyz(result_file, gt_file)

            


