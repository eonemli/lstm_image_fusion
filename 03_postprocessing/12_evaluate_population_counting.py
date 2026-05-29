#!/usr/bin/env python3
"""
Light-Sheet Fluorescence Microscopy (LSFM) Post-Processing Pipeline.
Step 12: Unified Population Counting and Density Regression Evaluator.
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)

def load_config(path: str = "config.yaml") -> dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)['evaluation_suite']
    except (FileNotFoundError, KeyError) as e:
        logging.error(f"Centralized configuration path exception: {e}")
        sys.exit(1)

CFG = load_config()

def count_classes_in_txt(txt_path: Path) -> Dict[int, int]:
    counts = {0: 0, 1: 0}
    if not txt_path.exists(): return counts
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                cls_id = int(parts[0])
                if cls_id in counts: counts[cls_id] += 1
    return counts

def render_table_image(df: pd.DataFrame, title: str, out_path: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(11, df.shape[0] * 0.6 + 1.5))
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.2, 2.0)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white', size=14)
            cell.set_facecolor('#4C72B0')
        elif row == 1:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#EAECEE')
        else:
            cell.set_facecolor('#F8F9FA' if row % 2 == 0 else 'white')
            
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_agreement_scatter(df: pd.DataFrame, class_name: str, out_dir: Path, dpi: int, font_scale: float):
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid", font_scale=font_scale)
    df_class = df[df['Class'] == class_name]
    
    max_val = max(df_class['GT_Count'].max(), df_class['Pred_Count'].max())
    max_val += (max_val * 0.1) # Add 10% structural viewport padding
    
    ax = sns.scatterplot(data=df_class, x='GT_Count', y='Pred_Count', hue='Method', style='Method', s=170, alpha=0.8)
    ax.plot([0, max_val], [0, max_val], color='black', linestyle='--', linewidth=2.5, label="Perfect Agreement ($y = x$)")
    
    plt.title(f"{class_name} Population Density Linear Regression", fontsize=22, pad=20, fontweight='bold')
    plt.xlabel('Ground Truth (Manual Count)', fontsize=18, fontweight='bold')
    plt.ylabel('Predicted (YOLO Output)', fontsize=18, fontweight='bold')
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.legend(title="Method Tracker", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / f"04_{class_name}_Counting_Scatter.png", dpi=dpi, bbox_inches='tight')
    plt.close()

def main():
    subset_dir = Path(CFG['paths']['subset_root'])
    out_dir = Path(CFG['paths']['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    gt_folder = CFG['structure']['gt_folder_name']
    gt_file = CFG['structure']['gt_file_name']
    methods = GLOBAL_CFG['structure']['ACQUISITION_MODES'] + ['HDPR_Early', 'HDPR_Late']
    
    slice_folders = [d for d in subset_dir.iterdir() if d.is_dir()]
    raw_data = []
    
    total_gt_neurons, total_gt_glia = 0, 0
    method_totals = {m: {'Neurons': 0, 'Glia': 0} for m in methods}
    slices_processed = 0

    for slice_folder in slice_folders:
        gt_path = slice_folder / gt_folder / gt_file
        if not gt_path.exists(): continue
            
        gt_counts = count_classes_in_txt(gt_path)
        total_gt_neurons += gt_counts[0]
        total_gt_glia += gt_counts[1]
        
        for method in methods:
            pred_counts = count_classes_in_txt(slice_folder / "labels" / f"{method}.txt")
            method_totals[method]['Neurons'] += pred_counts[0]
            method_totals[method]['Glia'] += pred_counts[1]
            
            raw_data.append({'Slice': slice_folder.name, 'Method': method, 'Class': 'Neuron', 'GT_Count': gt_counts[0], 'Pred_Count': pred_counts[0]})
            raw_data.append({'Slice': slice_folder.name, 'Method': method, 'Class': 'Glia', 'GT_Count': gt_counts[1], 'Pred_Count': pred_counts[1]})
        slices_processed += 1

    if slices_processed == 0:
        logging.error("No valid ground truth counting entries located.")
        return

    df = pd.DataFrame(raw_data)
    df.to_csv(out_dir / "03_Raw_Counting_Data.csv", index=False)

    # Generate Mathematical Metrics Report
    with open(out_dir / "00_Counting_Regression_Report.txt", "w") as f:
        f.write(f"--- GLOBAL POPULATION COUNTING ERROR REPORT ---\nProcessed: {slices_processed} validation slices\n\n")
        for cls_name in ["Neuron", "Glia"]:
            f.write(f"=== {cls_name.upper()} REGRESSION METRICS ===\n")
            df_cls = df[df['Class'] == cls_name]
            for method in methods:
                df_method = df_cls[df_cls['Method'] == method]
                gt, pred = df_method['GT_Count'], df_method['Pred_Count']
                mae = mean_absolute_error(gt, pred)
                rmse = np.sqrt(mean_squared_error(gt, pred))
                r2 = r2_score(gt, pred) if len(gt) > 1 else 0.0
                bias = ((pred.sum() - gt.sum()) / gt.sum()) * 100 if gt.sum() > 0 else 0
                f.write(f"{method.ljust(15)} | MAE: {mae:>6.2f} | RMSE: {rmse:>6.2f} | R2: {r2:>6.3f} | Systematic Bias: {bias:>+6.1f}%\n")
            f.write("\n")

    # Generate Unified Population Table Data
    table_rows = [{'Method': 'Ground Truth (Manual Consensus)', 'Total Neurons': total_gt_neurons, 'Neuron Bias': '-', 'Total Glia': total_gt_glia, 'Glia Bias': '-'}]
    for method in methods:
        n_p, g_p = method_totals[method]['Neurons'], method_totals[method]['Glia']
        n_bias = ((n_p - total_gt_neurons) / total_gt_neurons) * 100 if total_gt_neurons > 0 else 0
        g_bias = ((g_p - total_gt_glia) / total_gt_glia) * 100 if total_gt_glia > 0 else 0
        table_rows.append({'Method': method, 'Total Neurons': n_p, 'Neuron Bias': f"{n_bias:+.1f}%", 'Total Glia': g_p, 'Glia Bias': f"{g_bias:+.1f}%"})
        
    df_table = pd.DataFrame(table_rows)
    df_table.to_csv(out_dir / "01_Population_Totals_Summary.csv", index=False)
    render_table_image(df_table, f"Global Population Affiliation Matrix ({slices_processed} Slices)", out_dir / "02_Total_Population_Table.png", CFG['parameters']['dpi'])

    # Render Scatter Charts
    plot_agreement_scatter(df, "Neuron", out_dir, CFG['parameters']['dpi'], CFG['parameters']['font_scale'])
    plot_agreement_scatter(df, "Glia", out_dir, CFG['parameters']['dpi'], CFG['parameters']['font_scale'])
    logging.info(f"--- Population Counting Analytics Complete. Figures deployed to: {out_dir} ---")

if __name__ == '__main__':
    main()