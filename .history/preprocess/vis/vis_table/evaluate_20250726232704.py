import numpy as np
import os
import argparse
import csv
from lib.transform import qvec2rotmat, rotmat2qvec, convert_quaternion_to_euler, get_rotation_enu_in_ecef,WGS84_to_ECEF


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

    errors_t = []
    for name in test_names:
        gt_name = name.split('_')[0] + '_0.png'
        if name not in predictions or gt_name not in gts:
            if only_localized:
                continue
            errors_t.append(np.inf)
        else:
            t_gt = np.array(gts[gt_name])
            t_pred = np.array(predictions[name])
            e_t = np.linalg.norm(t_pred - t_gt)
            errors_t.append(e_t)

    if len(errors_t) == 0:
        return None

    errors_t = np.array(errors_t)
    finite_errors = errors_t[np.isfinite(errors_t)]

    med_t = np.median(finite_errors)
    std_t = np.std(finite_errors)
    recall_1m = np.mean(errors_t < 1)
    recall_3m = np.mean(errors_t < 3)
    recall_5m = np.mean(errors_t < 5)
    completeness = test_num / total_num if total_num > 0 else 0
    print({
        'MedianError': med_t,
        'StdError': std_t,
        'Recall@1m': recall_1m,
        'Recall@3m': recall_3m,
        'Recall@5m': recall_5m,
        'Completeness': completeness
    })
    return {
        'MedianError': med_t,
        'StdError': std_t,
        'Recall@1m': recall_1m,
        'Recall@3m': recall_3m,
        'Recall@5m': recall_5m,
        'Completeness': completeness
    }

def main():
    base_path = "/mnt/sda/MapScape/query/estimation/result_images/Google_foggy"
    output_csv = "/mnt/sda/MapScape/query/estimation/result_images/Google_foggy/Google_foggy.csv"
    methods = {
        'Render2ORB': 'Render2ORB',
        'FPVLoc': 'GeoPixel',
        'Render2Loc': 'Render2Loc',
        'Render2RAFT': 'Render2RAFT',
        'PixLoc': 'PixLoc',
    }

    results_list = []
    gt_path = os.path.join(base_path, "GT")
    render2orb_path = os.path.join(base_path, "GeoPixel")
    txt_files = [f for f in os.listdir(render2orb_path) if f.endswith('.txt') ]

    for txt_file in txt_files:
        seq = txt_file.split('.')[0]
        gt_file = os.path.join(gt_path, f"{seq}.txt")
        print(f"\n-------- Sequence: {seq} --------")

        for method_name, subfolder in methods.items():
            result_file = os.path.join(base_path, subfolder, f"{seq}.txt")
            print(f"== Method: {method_name} ==")
            metrics = evaluate_xyz(result_file, gt_file)

            if metrics is not None:
                results_list.append({
                    'Sequence': seq,
                    'Method': method_name,
                    'MedianError': metrics['MedianError'],
                    'StdError': metrics['StdError'],
                    'Recall@1m': metrics['Recall@1m'],
                    'Recall@3m': metrics['Recall@3m'],
                    'Recall@5m': metrics['Recall@5m'],
                    'Completeness': metrics['Completeness'],
                })

    # 保存为 CSV
    csv_output_path = output_csv
    with open(csv_output_path, 'w', newline='') as csvfile:
        fieldnames = ['Sequence', 'Method', 'MedianError', 'StdError', 'Recall@1m', 'Recall@3m', 'Recall@5m', 'Completeness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_list)

    print(f"\n✅ 已保存统计结果至 CSV: {csv_output_path}")

if __name__ == "__main__":
    main()

