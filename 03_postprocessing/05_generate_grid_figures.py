#!/usr/bin/env python3
"""
Light-Sheet Fluorescence Microscopy (LSFM) Post-Processing Pipeline.
Step 05: Generate Comparative Grid Panels (Adapts dynamically to RAW/STRETCHED availability).
"""

import os
import sys
import time
import yaml
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union

import cv2
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configure unbuffered log stream optimized for HPC clusters and standard out redirects
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout
)

def load_config(path: str = "config.yaml") -> dict:
    """Loads configuration blocks cleanly from the centralized workspace yaml configuration."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Centralized configuration file not found at: {path}")
        sys.exit(1)

GLOBAL_CFG = load_config()
CFG = GLOBAL_CFG['code_05_grid_generation']
STRUCT = GLOBAL_CFG['structure']

def parse_boxes(txt_path: str, img_w: int, img_h: int) -> List[Tuple[float, float, float, float, int]]:
    """Reads YOLO text file and converts relative coordinates to absolute pixel boxes."""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                x1 = (xc - w/2) * img_w
                y1 = (yc - h/2) * img_h
                box_w = w * img_w
                box_h = h * img_h
                boxes.append((x1, y1, box_w, box_h, cls_id))
    return boxes

def get_image_path(base_path: str, hdpr_path: str, method: str, x_folder: str, tile: str, z_slice: str) -> Path:
    """Constructs the correct path for either raw powers or HDPR fusions from configuration definitions."""
    channel = STRUCT['CHANNEL']
    hdpr_dir = STRUCT.get('hdpr_folder', 'HDPR')
    
    if "HDPR" in method:
        return Path(hdpr_path) / hdpr_dir / channel / x_folder / tile / f"{z_slice}.tiff"
    else:
        return Path(base_path) / method / channel / x_folder / tile / f"{z_slice}.tiff"

def draw_figure(slice_info: pd.Series, dataset_type: str, out_folder: Path) -> bool:
    """Draws a high-resolution, custom layout 2xN grid matching manuscript configurations."""
    x_folder = str(slice_info['X_Folder'])
    tile = str(slice_info['Tile'])
    z_slice = str(slice_info['Z_Slice'])
    
    methods = STRUCT['ACQUISITION_MODES'] + ['HDPR_Early', 'HDPR_Late']
    num_cols = len(methods)
    
    # Dynamic Path Validation Layer
    if dataset_type == "RAW":
        img_root = CFG['paths'].get('raw_root')
        hdpr_root = CFG['paths'].get('raw_hdpr_root')
    else:
        img_root = CFG['paths'].get('stretched_root')
        hdpr_root = CFG['paths'].get('stretched_hdpr_root')
        
    if not img_root or not hdpr_root:
        logging.warning(f"Skipping rendering for branch [{dataset_type}]: Target paths missing from configuration.")
        return False
        
    fig, axes = plt.subplots(2, num_cols, figsize=(5 * num_cols, 10))
    fig.suptitle(f"{dataset_type} | Optical Slice: {z_slice}", fontsize=22, y=0.98, fontweight='bold')
    
    for i, method in enumerate(methods):
        ax_img = axes[0, i]
        ax_hist = axes[1, i]
        
        img_path = get_image_path(img_root, hdpr_root, method, x_folder, tile, z_slice)
        txt_path = Path(CFG['paths']['txt_root']) / dataset_type / method / x_folder / tile / f"{z_slice}.txt"
        
        if img_path.exists():
            img_16 = tifffile.imread(str(img_path))
            h, w = img_16.shape
            
            img_8 = cv2.normalize(img_16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            ax_img.imshow(img_8, cmap='gray')
            ax_img.axis('off')
            
            boxes = parse_boxes(str(txt_path), w, h)
            for box in boxes:
                x1, y1, bw, bh, cls = box
                color = 'lime' if cls == 0 else 'yellow'
                rect = patches.Rectangle((x1, y1), bw, bh, linewidth=1, edgecolor=color, facecolor='none', alpha=0.8)
                ax_img.add_patch(rect)
                
            ax_img.set_title(f"{method}\n(Boxes: {len(boxes)})", fontsize=15, fontweight='bold', pad=10)

            pixel_data = img_16.flatten()
            pixel_data = pixel_data[pixel_data > 0] 
            
            if pixel_data.size > 0:
                mean_val = np.mean(pixel_data)
                median_val = np.median(pixel_data)
                
                ax_hist.hist(pixel_data, bins=100, log=True, color='royalblue', alpha=0.7)
                ax_hist.axvline(mean_val, color='red', linestyle='dashed', linewidth=2.5, alpha=0.8, label=f'Mean: {mean_val:.0f}')
                ax_hist.axvline(median_val, color='green', linestyle='dotted', linewidth=2.5, alpha=0.8, label=f'Med: {median_val:.0f}')
                
                ax_hist.set_title("Intensity Distribution", fontsize=13, fontweight='bold', pad=8)
                ax_hist.set_xlabel("Pixel Intensity (16-bit)", fontsize=13, fontweight='bold')
                ax_hist.set_ylabel("Frequency (Log Scale)", fontsize=13, fontweight='bold')
                ax_hist.tick_params(axis='both', which='major', labelsize=12)
                ax_hist.legend(loc='upper right', fontsize=12, framealpha=0.9)
            else:
                ax_hist.text(0.5, 0.5, "Noise Core Block", ha='center', va='center', fontsize=14, fontweight='bold')
                ax_hist.axis('off')
        else:
            ax_img.text(0.5, 0.5, "Image Target Offline", ha='center', va='center', color='red', fontsize=14, fontweight='bold')
            ax_img.axis('off')
            ax_hist.axis('off')

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    out_path = Path(out_folder) / f"{z_slice}_{dataset_type}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True

def main():
    if not CFG.get('enabled', True):
        logging.info("--- FIGURES PANEL COMPILATION PIPELINE BYPASSED VIA CONFIG ---")
        return

    ledger_path = Path(CFG['paths']['ledger_path'])
    if not ledger_path.exists():
        logging.error(f"Target text ledger file missing: {ledger_path}. Run step 04 first.")
        return
        
    df = pd.read_csv(ledger_path, sep='\t')
    
    # Dynamic Branch Discovery Layer
    available_datasets = df['Dataset'].unique().tolist()
    if not available_datasets:
        logging.error("The provided text ledger contains no data lines.")
        return
        
    # Safely select an anchor branch for calculating categorical variations
    anchor_dataset = "STRETCHED" if "STRETCHED" in available_datasets else available_datasets[0]
    logging.info(f"Detected processing tracks: {available_datasets}. Setting reference anchor to: [{anchor_dataset}]")
    
    df_anchor = df[df['Dataset'] == anchor_dataset].copy()
    
    # Pivot using whichever dataset branch actually exists
    pivot = df_anchor.pivot_table(
        index=['X_Folder', 'Tile', 'Z_Slice'], 
        columns='Method', 
        values='Raw_Boxes_Total', 
        fill_value=0
    ).reset_index()
    
    method_cols = [m for m in STRUCT['ACQUISITION_MODES'] + ['HDPR_Early', 'HDPR_Late'] if m in pivot.columns]
    pivot['Max_Boxes'] = pivot[method_cols].max(axis=1)
    
    pivot = pivot[pivot['Max_Boxes'] > 5].copy() 
    pivot['Variance'] = pivot[method_cols].max(axis=1) - pivot[method_cols].min(axis=1)
    
    pivot['Detection_Class'] = 'Mixed'
    pivot.loc[pivot['Variance'] <= 3, 'Detection_Class'] = 'Consistent'
    pivot.loc[pivot['Variance'] >= 10, 'Detection_Class'] = 'Different'
    
    coords = pivot['Tile'].str.split('_', expand=True).astype(float)
    pivot['X'] = coords[0]
    pivot['Y'] = coords[1]
    
    center_X = (pivot['X'].max() + pivot['X'].min()) / 2.0
    center_Y = (pivot['Y'].max() + pivot['Y'].min()) / 2.0
    pivot['Distance'] = np.sqrt((pivot['X'] - center_X)**2 + (pivot['Y'] - center_Y)**2)
    
    dist_33 = pivot['Distance'].quantile(0.33)
    dist_66 = pivot['Distance'].quantile(0.66)
    
    pivot['Location_Class'] = 'Mid'
    pivot.loc[pivot['Distance'] <= dist_33, 'Location_Class'] = 'Center'
    pivot.loc[pivot['Distance'] >= dist_66, 'Location_Class'] = 'Edge'

    categories = {
        "Center_Consistent": pivot[(pivot['Location_Class'] == 'Center') & (pivot['Detection_Class'] == 'Consistent')],
        "Center_Different": pivot[(pivot['Location_Class'] == 'Center') & (pivot['Detection_Class'] == 'Different')],
        "Edge_Consistent": pivot[(pivot['Location_Class'] == 'Edge') & (pivot['Detection_Class'] == 'Consistent')],
        "Edge_Different": pivot[(pivot['Location_Class'] == 'Edge') & (pivot['Detection_Class'] == 'Different')]
    }
    
    if CFG['parameters'].get('generate_unbiased_random', False):
        categories["Unbiased_Random"] = pivot.copy()
    
    out_root = Path(CFG['paths']['output_dir'])
    target_samples = CFG['parameters'].get('target_samples_per_category', 10)
    random_seed = CFG['parameters'].get('random_seed', 42)
    
    total_generated = 0
    for cat_name, df_cat in categories.items():
        if df_cat.empty:
            continue
            
        cat_folder = out_root / cat_name
        cat_folder.mkdir(parents=True, exist_ok=True)
        
        n_samples = min(target_samples, len(df_cat))
        sampled = df_cat.sample(n=n_samples, random_state=random_seed)
        
        logging.info(f"Processing branch: [{cat_name}] -> Generating {n_samples} matrices.")
        
        for _, row in sampled.iterrows():
            # Adaptive generation block: only renders tracks that exist in your pipeline data
            for dataset_type in available_datasets:
                success = draw_figure(row, dataset_type, cat_folder)
                if success:
                    total_generated += 1
            
    logging.info(f"--- PROCESS SUCCESSFUL | Rendered {total_generated} panel grids inside: {out_root} ---")

if __name__ == '__main__':
    main()