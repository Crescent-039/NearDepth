#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import cv2
import random
import argparse
import numpy as np
from tqdm import tqdm


def scan_pairs(root_dir, mode='train'):
    pairs = []
    train_folders = ['train_1', 'train_2', 'train_3', 'train_4']

    for t in train_folders:
        tpath = os.path.join(root_dir, t)
        if not os.path.exists(tpath):
            continue

        for scene_folder in sorted(os.listdir(tpath)):
            if not scene_folder.startswith('scene_'):
                continue

            scene_id = int(scene_folder.split('_')[1])
            if mode == 'train' and scene_id >= 90:
                continue
            if mode == 'val' and scene_id < 90:
                continue

            base = os.path.join(tpath, scene_folder, 'realsense')
            rgb_dir = os.path.join(base, 'rgb')
            gt_dir = os.path.join(base, 'depth')
            s_dir = os.path.join(base, 'synthetic_depth')

            if not (os.path.exists(rgb_dir) and os.path.exists(gt_dir) and os.path.exists(s_dir)):
                continue

            for rgb_p in sorted(glob.glob(os.path.join(rgb_dir, '*.png'))):
                fn = os.path.basename(rgb_p)
                gt_p = os.path.join(gt_dir, fn)
                s_p = os.path.join(s_dir, fn)
                if os.path.exists(gt_p) and os.path.exists(s_p):
                    pairs.append((rgb_p, gt_p, s_p))
    return pairs


def median_align(pred, gt, mask, eps=1e-6):
    """每图用 median(gt/pred) 做尺度对齐"""
    s = np.median(gt[mask] / (pred[mask] + eps))
    return pred * s, s


def percentile_mask(ref, base_mask, low=1, high=99):
    """在 base_mask 内，按 ref 的分位数截尾"""
    vals = ref[base_mask]
    if vals.size < 100:
        return base_mask, None, None
    lo = np.percentile(vals, low)
    hi = np.percentile(vals, high)
    m = base_mask & (ref >= lo) & (ref <= hi)
    return m, lo, hi


def metric_stats(pred, gt, mask, eps=1e-6):
    """返回可累计的统计量（像素级加权）"""
    n = int(mask.sum())
    if n < 10:
        return None

    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)
    p = np.clip(p, eps, None)

    abs_rel_sum = np.sum(np.abs(g - p) / (g + eps))
    sq_err_sum = np.sum((g - p) ** 2)

    ratio = np.maximum(p / (g + eps), g / (p + eps))
    d1_sum = np.sum(ratio < 1.25)
    d2_sum = np.sum(ratio < (1.25 ** 2))
    d3_sum = np.sum(ratio < (1.25 ** 3))

    return {
        "n": n,
        "abs_rel_sum": float(abs_rel_sum),
        "sq_err_sum": float(sq_err_sum),
        "d1_sum": int(d1_sum),
        "d2_sum": int(d2_sum),
        "d3_sum": int(d3_sum),
    }


def init_acc():
    return {"n": 0, "abs_rel_sum": 0.0, "sq_err_sum": 0.0, "d1_sum": 0, "d2_sum": 0, "d3_sum": 0}


def update_acc(acc, st):
    if st is None:
        return
    acc["n"] += st["n"]
    acc["abs_rel_sum"] += st["abs_rel_sum"]
    acc["sq_err_sum"] += st["sq_err_sum"]
    acc["d1_sum"] += st["d1_sum"]
    acc["d2_sum"] += st["d2_sum"]
    acc["d3_sum"] += st["d3_sum"]


def finalize_acc(acc):
    if acc["n"] == 0:
        return None
    n = acc["n"]
    abs_rel = acc["abs_rel_sum"] / n
    rmse = np.sqrt(acc["sq_err_sum"] / n)
    d1 = acc["d1_sum"] / n
    d2 = acc["d2_sum"] / n
    d3 = acc["d3_sum"] / n
    return abs_rel, rmse, d1, d2, d3


def fmt_metrics(m):
    if m is None:
        return "N/A"
    a, r, d1, d2, d3 = m
    return f"AbsRel={a:.4f} | RMSE={r:.4f} | δ1={d1*100:.2f}% | δ2={d2*100:.2f}% | δ3={d3*100:.2f}%"


def main():
    ap = argparse.ArgumentParser("GT vs SyntheticDepth 诊断（Raw/P1-P99/P5-P95）")
    ap.add_argument('--root_dir', type=str, default='../autodl-tmp/graspnet_dataset')
    ap.add_argument('--mode', type=str, default='train', choices=['train', 'val'])
    ap.add_argument('--num', type=int, default=400, help='随机抽样数')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min_depth', type=float, default=0.01)
    ap.add_argument('--max_depth', type=float, default=3.0)
    ap.add_argument('--trim_ref', type=str, default='both', choices=['gt', 'syn', 'both'],
                    help='分位数截尾参考：gt/syn/both(交集)')
    ap.add_argument('--use_align', action='store_true', help='额外输出 median 对齐后的结果')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    pairs = scan_pairs(args.root_dir, args.mode)
    print(f'扫描到配对样本: {len(pairs)}')
    if len(pairs) == 0:
        print("❌ 没找到数据，请检查 root_dir")
        return

    random.shuffle(pairs)
    pairs = pairs[:min(args.num, len(pairs))]
    print(f'参与统计样本: {len(pairs)}')

    # 统计容器：raw syn
    acc_raw = init_acc()
    acc_p1p99 = init_acc()
    acc_p5p95 = init_acc()

    # 统计容器：aligned syn（可选）
    acc_raw_aln = init_acc()
    acc_p1p99_aln = init_acc()
    acc_p5p95_aln = init_acc()

    bad_dtype, bad_shape, bad_channel, bad_read = 0, 0, 0, 0
    gt_zero_list, s_zero_list = [], []
    gt_p50, s_p50, scales = [], [], []

    for _, gt_p, s_p in tqdm(pairs, desc='诊断中'):
        gt_u16 = cv2.imread(gt_p, cv2.IMREAD_UNCHANGED)
        s_u16 = cv2.imread(s_p, cv2.IMREAD_UNCHANGED)

        if gt_u16 is None or s_u16 is None:
            bad_read += 1
            continue
        if gt_u16.ndim != 2 or s_u16.ndim != 2:
            bad_channel += 1
            continue
        if gt_u16.dtype != np.uint16 or s_u16.dtype != np.uint16:
            bad_dtype += 1
            continue
        if gt_u16.shape != s_u16.shape:
            bad_shape += 1
            continue

        gt_zero_list.append((gt_u16 == 0).mean())
        s_zero_list.append((s_u16 == 0).mean())

        gt = gt_u16.astype(np.float32) / 1000.0
        syn = s_u16.astype(np.float32) / 1000.0

        if np.any(gt > 0):
            gt_p50.append(np.median(gt[gt > 0]))
        if np.any(syn > 0):
            s_p50.append(np.median(syn[syn > 0]))

        base = (gt > args.min_depth) & (gt < args.max_depth) & (syn > 0)
        if base.sum() < 100:
            continue

        # --- Raw ---
        st_raw = metric_stats(syn, gt, base)
        update_acc(acc_raw, st_raw)

        # --- P1-P99 / P5-P95 ---
        if args.trim_ref == 'gt':
            m1, _, _ = percentile_mask(gt, base, 1, 99)
            m5, _, _ = percentile_mask(gt, base, 5, 95)
        elif args.trim_ref == 'syn':
            m1, _, _ = percentile_mask(syn, base, 1, 99)
            m5, _, _ = percentile_mask(syn, base, 5, 95)
        else:  # both
            m1_gt, _, _ = percentile_mask(gt, base, 1, 99)
            m1_sy, _, _ = percentile_mask(syn, base, 1, 99)
            m1 = m1_gt & m1_sy

            m5_gt, _, _ = percentile_mask(gt, base, 5, 95)
            m5_sy, _, _ = percentile_mask(syn, base, 5, 95)
            m5 = m5_gt & m5_sy

        update_acc(acc_p1p99, metric_stats(syn, gt, m1))
        update_acc(acc_p5p95, metric_stats(syn, gt, m5))

        # --- 对齐结果（可选）---
        if args.use_align:
            syn_aln, sc = median_align(syn, gt, base)
            scales.append(sc)

            update_acc(acc_raw_aln, metric_stats(syn_aln, gt, base))

            if args.trim_ref == 'gt':
                m1a, _, _ = percentile_mask(gt, base, 1, 99)
                m5a, _, _ = percentile_mask(gt, base, 5, 95)
            elif args.trim_ref == 'syn':
                m1a, _, _ = percentile_mask(syn_aln, base, 1, 99)
                m5a, _, _ = percentile_mask(syn_aln, base, 5, 95)
            else:
                m1_gt, _, _ = percentile_mask(gt, base, 1, 99)
                m1_sy, _, _ = percentile_mask(syn_aln, base, 1, 99)
                m1a = m1_gt & m1_sy

                m5_gt, _, _ = percentile_mask(gt, base, 5, 95)
                m5_sy, _, _ = percentile_mask(syn_aln, base, 5, 95)
                m5a = m5_gt & m5_sy

            update_acc(acc_p1p99_aln, metric_stats(syn_aln, gt, m1a))
            update_acc(acc_p5p95_aln, metric_stats(syn_aln, gt, m5a))

    print('\n===== 数据质量 =====')
    print(f'bad_read: {bad_read}')
    print(f'bad_channel(非单通道): {bad_channel}')
    print(f'bad_dtype(非uint16): {bad_dtype}')
    print(f'bad_shape(尺寸不一致): {bad_shape}')
    if len(gt_zero_list) > 0:
        print(f'GT零值比例均值: {np.mean(gt_zero_list):.3f}')
    if len(s_zero_list) > 0:
        print(f'SYN零值比例均值: {np.mean(s_zero_list):.3f}')
    if len(gt_p50) > 0:
        print(f'GT中位深度(米)均值: {np.mean(gt_p50):.3f}')
    if len(s_p50) > 0:
        print(f'SYN中位深度(米)均值: {np.mean(s_p50):.3f}')

    print('\n===== SYN 原始结果 =====')
    print(f'[Raw]     {fmt_metrics(finalize_acc(acc_raw))}')
    print(f'[P1-P99]  {fmt_metrics(finalize_acc(acc_p1p99))}')
    print(f'[P5-P95]  {fmt_metrics(finalize_acc(acc_p5p95))}')
    print(f'（trim_ref = {args.trim_ref}）')

    if args.use_align:
        print('\n===== SYN 尺度对齐后结果（每图 median 对齐） =====')
        print(f'[Raw]     {fmt_metrics(finalize_acc(acc_raw_aln))}')
        print(f'[P1-P99]  {fmt_metrics(finalize_acc(acc_p1p99_aln))}')
        print(f'[P5-P95]  {fmt_metrics(finalize_acc(acc_p5p95_aln))}')
        if len(scales) > 0:
            print(f'scale 因子: mean={np.mean(scales):.4f}, std={np.std(scales):.4f}')


if __name__ == '__main__':
    main()

# python debug.py --root_dir ../autodl-tmp/graspnet_dataset --mode train --num 50

    
    