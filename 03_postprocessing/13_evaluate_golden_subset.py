#!/usr/bin/env python3
"""
Light-Sheet Fluorescence Microscopy (LSFM) Post-Processing Pipeline.
Step 13: Curated Golden Cohort Target Miner and Performance Profiler.
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)

def load_config(path: str = "config.yaml") -> dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)['evaluation_suite']
    except (FileNotFoundError, KeyError) as e:
        logging.error(f"Centralized validation architecture mapping key error: {e}")
        sys.exit(1)

CFG = load_config()
CLASSES = {0: "Neuron", 1: "Glia", 2: "Background"}

def parse_boxes(txt_path: Path) -> np.ndarray:
    boxes = []
    if not txt_path.exists(): return np.array(boxes)
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2, cls_id])
    return np.array(boxes)

def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return float(interArea / (boxAArea + boxBArea - interArea + 1e-6))

def evaluate_slice_cm(gt_boxes: np.ndarray, pred_boxes: np.ndarray, iou_thresh: float) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=int)
    if len(gt_boxes) == 0 and len(pred_boxes) == 0: return cm
    if len(gt_boxes) == 0:
        for p in pred_boxes: cm[2, int(p[4])] += 1
        return cm
    if len(pred_boxes) == 0:
        for g in gt_boxes: cm[int(g[4]), 2] += 1
        return cm

    matched_gt, matched_pred = set(), set()
    for p_idx, p_box in enumerate(pred_boxes):
        best_iou, best_gt_idx = 0.0, -1
        for g_idx, g_box in enumerate(gt_boxes):
            if g_idx in matched_gt: continue
            iou = calculate_iou(p_box[:4], g_box[:4])
            if iou > best_iou: best_iou, best_gt_idx = iou, g_idx
        if best_iou >= iou_thresh:
            matched_pred.add(p_idx)
            matched_gt.add(best_gt_idx)
            cm[int(gt_boxes[best_gt_idx][4]), int(p_box[4])] += 1
        else:
            cm[2, int(p_box[4])] += 1

    for g_idx, g_box in enumerate(gt_boxes):
        if g_idx not in matched_gt: cm[int(g_box[4]), 2] += 1
    return cm

def calculate_slice_metrics(cm: np.ndarray) -> dict:
    metrics = {}
    for cls_idx in [0, 1]:
        tp = cm[cls_idx, cls_idx]
        fp = cm[2, cls_idx] + cm[1 - cls_idx, cls_idx]
        fn = cm[cls_idx, 2] + cm[cls_idx, 1 - cls_idx]
        acc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        metrics[CLASSES[cls_idx]] = {'Accuracy': acc, 'Precision': p, 'Recall': r, 'F1': f1, 'TP': tp, 'FP': fp, 'FN': fn}
    return metrics

def plot_combined_confusion_matrix(master_cms: dict, methods: list, out_dir: Path, dpi: int, limit: int):
    logging.info("Generating 1x4 Golden Subset Combined Confusion Matrix panel...")
    vmax = max(int(np.delete(master_cms[m], 8).max()) for m in methods)
    fig, axes = plt.subplots(1, len(methods) + 1, figsize=(25, 6.5), gridspec_kw={'width_ratios': [1]*len(methods) + [0.05], 'wspace': 0.05})
    labels = ["Neuron", "Glia", "BG"]
    
    for i, method in enumerate(methods):
        ax = axes[i]
        cm = master_cms[method]
        annot_cm = cm.astype(str)
        annot_cm[2, 2] = ""
        mask = np.zeros_like(cm, dtype=bool)
        mask[2, 2] = True
        
        sns.heatmap(cm, annot=annot_cm, fmt="", cmap="Blues", xticklabels=labels, yticklabels=labels if i == 0 else False,
                    mask=mask, vmin=0, vmax=vmax, cbar=(i == len(methods)-1), cbar_ax=axes[-1] if i == len(methods)-1 else None, ax=ax,
                    annot_kws={"weight": "bold", "size": 22})
        ax.set_title(f"{method} (Top {limit})", fontsize=18, fontweight='bold', pad=10)
        if i == 0: plt.setp(ax.get_yticklabels(), rotation=90, va="center")
            
    fig.supxlabel('Predicted Class (YOLO Output)', fontsize=24, fontweight='bold', y=-0.02)
    fig.supylabel('True Class (Ground Truth)', fontsize=24, fontweight='bold', x=0.08)
    axes[-1].tick_params(labelsize=16)
    axes[-1].set_ylabel('Box Count', fontsize=18, fontweight='bold', rotation=270, labelpad=25)
    plt.savefig(out_dir / "06_Golden_Combined_CM.png", dpi=dpi, bbox_inches='tight')
    plt.close()

def main():
    subset_dir = Path(CFG['paths']['subset_root'])
    out_dir = Path(CFG['paths']['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    gt_folder = CFG['structure']['gt_folder_name']
    gt_file = CFG['structure']['gt_file_name']
    methods = GLOBAL_CFG['structure']['ACQUISITION_MODES'] + ['HDPR_Early', 'HDPR_Late']
    
    iou_thresh = CFG['parameters']['iou_threshold']
    top_k = CFG['parameters']['top_k_limit']
    rank_method = CFG['parameters']['ranking_method']
    rank_class = CFG['parameters']['ranking_class']
    rank_metric = CFG['parameters']['ranking_metric']
    
    slice_folders = [d for d in subset_dir.iterdir() if d.is_dir()]
    slice_pool = []

    for slice_folder in slice_folders:
        gt_path = slice_folder / gt_folder / gt_file
        if not gt_path.exists(): continue
            
        has_all = True
        for method in methods:
            if not (slice_folder / "labels" / f"{method}.txt").exists():
                has_all = False
                break
        if not has_all: continue

        gt_boxes = parse_boxes(gt_path)
        slice_cms = {}
        for method in methods:
            pred_boxes = parse_boxes(slice_folder / "labels" / f"{method}.txt")
            slice_cms[method] = evaluate_slice_cm(gt_boxes, pred_boxes, iou_thresh)
            
        mets = calculate_slice_metrics(slice_cms[rank_method])
        score = (mets['Neuron'][rank_metric] + mets['Glia'][rank_metric]) / 2.0 if rank_class == "Average" else mets[rank_class][rank_metric]
            
        slice_pool.append({'folder': slice_folder.name, 'score': score, 'cms': slice_cms})

    if not slice_pool:
        logging.error("No complete validation sandboxes matched golden subset criteria constraints.")
        return

    slice_pool.sort(key=lambda x: x['score'], reverse=True)
    golden_slices = slice_pool[:top_k]
    logging.info(f"Curated Top {len(golden_slices)} Golden Slices (Ranked by {rank_method} {rank_class} {rank_metric})")
    
    master_cms = {m: np.zeros((3, 3), dtype=int) for m in methods}
    for item in golden_slices:
        for method in methods:
            master_cms[method] += item['cms'][method]

    # Export Curated Report Logs
    with open(out_dir / "05_Golden_Cohort_Report.txt", "w") as f:
        f.write(f"--- GOLDEN COHORT ANALYTICAL REPORT ---\nCuration Limit: Top {len(golden_slices)} based on {rank_method} {rank_class} {rank_metric}\nMatching IoU Threshold: {iou_thresh}\n\nSelected Cohort Slices:\n")
        for i, item in enumerate(golden_slices, 1):
            f.write(f"  {i}. {item['folder']} (Curation Anchor Score: {item['score']:.3f})\n")
        f.write("\n")
        
        for method in methods:
            f.write(f"========================================\nMETHOD: {method}\n========================================\n")
            metrics = calculate_slice_metrics(master_cms[method])
            for cls_name, mets in metrics.items():
                f.write(f"{cls_name} -> Acc: {mets['Accuracy']:.3f} | P: {mets['Precision']:.3f} | R: {mets['Recall']:.3f} | F1: {mets['F1']:.3f} (TP:{mets['TP']} FP:{mets['FP']})\n")
            f.write("\n")

    plot_combined_confusion_matrix(master_cms, methods, out_dir, CFG['parameters']['dpi'], top_k)

    # Render Summary Bar Plot
    plot_data = []
    for method in methods:
        metrics = calculate_slice_metrics(master_cms[method])
        plot_data.append({'Method': method, 'Class': 'Neuron', 'F1': metrics['Neuron']['F1']})
        plot_data.append({'Method': method, 'Class': 'Glia', 'F1': metrics['Glia']['F1']})
        
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid", font_scale=CFG['parameters']['font_scale'])
    ax = sns.barplot(data=pd.DataFrame(plot_data), x='Method', y='F1', hue='Class', palette=['#4C72B0', '#DD8452'])
    plt.title(f"Golden Cohort Peak Performance (Top {len(golden_slices)})", fontsize=20, pad=20, fontweight='bold')
    plt.ylabel("F1-Score", fontsize=16, fontweight='bold')
    plt.ylim(0, 1.05)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "07_Golden_F1_Summary.png", dpi=CFG['parameters']['dpi'])
    plt.close()
    logging.info(f"--- Golden Cohort Mining Extraction Complete. Artifacts deployed to: {out_dir} ---")

if __name__ == '__main__':
    main()