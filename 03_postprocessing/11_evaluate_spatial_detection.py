#!/usr/bin/env python3
"""
Light-Sheet Fluorescence Microscopy (LSFM) Post-Processing Pipeline.
Step 11: Comprehensive Spatial Detection Evaluator (Micro/Macro Stats & Confusion Matrices).
"""

import os
import sys
import time
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
        logging.error(f"Centralized validation configuration block missing: {e}")
        sys.exit(1)

GLOBAL_CFG = load_config()
CFG = GLOBAL_CFG
CLASSES = {0: "Neuron", 1: "Glia", 2: "Background"}

def parse_boxes(txt_path: Path) -> np.ndarray:
    boxes = []
    if not txt_path.exists():
        return np.array(boxes)
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
        if g_idx not in matched_gt:
            cm[int(g_box[4]), 2] += 1
    return cm

def calculate_metrics(cm: np.ndarray) -> dict:
    metrics = {}
    for cls_idx in [0, 1]:
        tp = cm[cls_idx, cls_idx]
        fp = cm[2, cls_idx] + cm[1 - cls_idx, cls_idx]
        fn = cm[cls_idx, 2] + cm[cls_idx, 1 - cls_idx]
        acc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        metrics[CLASSES[cls_idx]] = {'Acc': acc, 'Precision': p, 'Recall': r, 'F1': f1, 'TP': tp, 'FP': fp, 'FN': fn}
    return metrics

def plot_combined_confusion_matrix(master_cms: dict, methods: list, out_dir: Path, dpi: int):
    logging.info("Generating 1xN Combined Confusion Matrix Figure panel ...")
    vmax = max(int(np.delete(master_cms[m], 8).max()) for m in methods)
    
    fig, axes = plt.subplots(
        1, len(methods) + 1, 
        figsize=(25, 6.5), 
        gridspec_kw={'width_ratios': [1]*len(methods) + [0.05], 'wspace': 0.05}
    )
    labels = ["Neuron", "Glia", "BG"]
    
    for i, method in enumerate(methods):
        ax = axes[i]
        cm = master_cms[method]
        annot_cm = cm.astype(str)
        annot_cm[2, 2] = ""
        mask = np.zeros_like(cm, dtype=bool)
        mask[2, 2] = True
        
        sns.heatmap(
            cm, annot=annot_cm, fmt="", cmap="Blues", 
            xticklabels=labels, yticklabels=labels if i == 0 else False,
            mask=mask, vmin=0, vmax=vmax, 
            cbar=(i == len(methods)-1), cbar_ax=axes[-1] if i == len(methods)-1 else None, 
            ax=ax, annot_kws={"weight": "bold", "size": 42}
        )
        
        ax.set_title(method, fontsize=22, fontweight='bold', pad=15)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(axis='both', which='major', labelsize=30)
        if i == 0: 
            plt.setp(ax.get_yticklabels(), rotation=90, va="center")
            
    fig.supxlabel('Predicted Class (YOLO Output)', fontsize=36, fontweight='bold', y=-0.06)
    fig.supylabel('True Class (Ground Truth)', fontsize=36, fontweight='bold', x=0.05)
    
    axes[-1].tick_params(labelsize=26)
    axes[-1].set_ylabel('Box Count', fontsize=30, fontweight='bold', rotation=270, labelpad=35)
    
    # UPDATED INDEX: Prefixed layout figure path cleanly for ordered sorting
    plt.savefig(out_dir / "01_02_Combined_CM_Figure.png", dpi=dpi, bbox_inches='tight')
    plt.close()

def main():
    subset_dir = Path(CFG['paths']['subset_root'])
    out_dir = Path(CFG['paths']['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    gt_folder = CFG['structure']['gt_folder_name']
    gt_file = CFG['structure']['gt_file_name']
    methods = yaml.safe_load(open("config.yaml", "r"))['structure']['ACQUISITION_MODES'] + ['HDPR_Early', 'HDPR_Late']
    iou_thresh = CFG['parameters']['iou_threshold']
    
    slice_folders = [d for d in subset_dir.iterdir() if d.is_dir()]
    master_cms = {m: np.zeros((3, 3), dtype=int) for m in methods}
    macro_data = {m: {0: {'f1': []}, 1: {'f1': []}} for m in methods}
    plot_raw_f1 = []
    slices_evaluated = 0

    for slice_folder in slice_folders:
        gt_path = slice_folder / gt_folder / gt_file
        if not gt_path.exists(): continue
            
        gt_boxes = parse_boxes(gt_path)
        for method in methods:
            pred_boxes = parse_boxes(slice_folder / "labels" / f"{method}.txt")
            cm = evaluate_slice_cm(gt_boxes, pred_boxes, iou_thresh)
            master_cms[method] += cm
            
            mets = calculate_metrics(cm)
            macro_data[method][0]['f1'].append(mets['Neuron']['F1'])
            macro_data[method][1]['f1'].append(mets['Glia']['F1'])
            plot_raw_f1.append({'Method': method, 'Class': 'Neuron', 'F1': mets['Neuron']['F1']})
            plot_raw_f1.append({'Method': method, 'Class': 'Glia', 'F1': mets['Glia']['F1']})
        slices_evaluated += 1

    if slices_evaluated == 0:
        logging.error("Zero standard validation ground-truth matrix coordinates found.")
        return

    logging.info(f"Successfully processed {slices_evaluated} slices for Spatial Detection validation.")
    
    # UPDATED INDEX: Prefixed report metrics file cleanly for ordered sorting
    with open(out_dir / "01_00_Spatial_Detection_Report.txt", "w") as f:
        f.write(f"--- GLOBAL SPATIAL DETECTION PERFORMANCE REPORT ---\nTotal Slices Evaluated: {slices_evaluated}\nMatching IoU Threshold: {iou_thresh}\n\n")
        for method in methods:
            f.write(f"========================================\nMETHOD: {method}\n========================================\n")
            micro_metrics = calculate_metrics(master_cms[method])
            f.write("[MICRO-AVERAGED METRICS (Global Pool)]\n")
            for cls_name, mets in micro_metrics.items():
                f.write(f"  {cls_name} -> Acc: {mets['Acc']:.3f} | P: {mets['Precision']:.3f} | R: {mets['Recall']:.3f} | F1: {mets['F1']:.3f} (TP:{mets['TP']} FP:{mets['FP']})\n")
            
            f.write("\n[MACRO-AVERAGED METRICS (Slice-by-Slice Variance)]\n")
            for cls_idx, cls_name in enumerate(["Neuron", "Glia"]):
                f1_list = macro_data[method][cls_idx]['f1']
                f.write(f"  {cls_name} F1 -> Mean: {np.mean(f1_list):.3f} | Median: {np.median(f1_list):.3f} | Variance: {np.var(f1_list):.4f}\n")
            f.write("\n")

    plot_combined_confusion_matrix(master_cms, methods, out_dir, CFG['parameters']['dpi'])
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid", font_scale=CFG['parameters']['font_scale'])
    ax = sns.barplot(data=pd.DataFrame(plot_raw_f1), x='Method', y='F1', hue='Class', palette=['#4C72B0', '#DD8452'], errorbar='sd', capsize=0.1)
    plt.title("Macro-Averaged Detection Performance (F1-Score $\pm$ SD)", fontsize=20, pad=20, fontweight='bold')
    plt.ylabel("F1-Score", fontsize=16, fontweight='bold')
    plt.ylim(0, 1.05)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=15, fontsize=11)
    plt.tight_layout()
    
    # Prefixed macro validation plot filename cleanly for ordered sorting
    plt.savefig(out_dir / "01_01_Macro_F1_Summary_with_Variance.png", dpi=CFG['parameters']['dpi'])
    plt.close()
    logging.info(f"--- Spatial Detection Validation Complete. Artifacts stored in: {out_dir} ---")

if __name__ == '__main__':
    main()