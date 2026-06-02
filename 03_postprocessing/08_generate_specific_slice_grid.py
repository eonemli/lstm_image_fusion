#!/usr/bin/env python3
"""
Light-Sheet Fluorescence Microscopy (LSFM) Post-Processing Pipeline.
Step 08: High-Resolution Visual Dashboard Generation for a Targeted List of Optical Z-Slices.
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
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configure unbuffered log stream explicitly optimized for direct cluster bsub tracking
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
CFG = GLOBAL_CFG['code_08_specific_slice']
STRUCT = GLOBAL_CFG['structure']

def parse_boxes(txt_path: Path, img_w: int, img_h: int) -> List[Tuple[float, float, float, float, int]]:
    """Reads YOLO text file and converts relative coordinates to absolute pixel boxes."""
    boxes = []
    if not txt_path or not txt_path.exists() or txt_path.is_dir():
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

def draw_figure(x_folder: str, tile: str, z_slice: str, dataset_type: str, out_folder: Path, dpi: int) -> bool:
    """Draws a high-resolution, custom layout 2xN grid matching manuscript configurations."""
    methods = STRUCT['ACQUISITION_MODES'] + ['HDPR_Early', 'HDPR_Late']
    num_cols = len(methods)
    
    # Lazy Path Resolution Layer with Safety Checks
    if dataset_type == "RAW":
        img_root = CFG['paths'].get('raw_root')
        hdpr_root = CFG['paths'].get('raw_hdpr_root')
    else:
        img_root = CFG['paths'].get('stretched_root')
        hdpr_root = CFG['paths'].get('stretched_hdpr_root')
        
    if not img_root or not hdpr_root:
        logging.warning(f"Bypassing branch [{dataset_type}]: Required path pointers missing from config.yaml.")
        return False

    # --- UPSTREAM FILE INTEGRITY GUARD ---
    # Verify at least one target image path exists before spawning the matplotlib canvas
    slice_exists_somewhere = False
    for method in methods:
        img_path = get_image_path(img_root, hdpr_root, method, x_folder, tile, z_slice)
        if img_path.exists():
            slice_exists_somewhere = True
            break
            
    if not slice_exists_somewhere:
        logging.error(f"  [PATH ERROR] Target slice folder/images completely offline on disk volume: {tile}/{z_slice}")
        return False
        
    # FIX: Applied identical tight layout dimensions matching Step 05 & Step 07 grids
    fig, axes = plt.subplots(
        2, num_cols, 
        figsize=(4.0 * num_cols, 7.2), 
        squeeze=False,
        gridspec_kw={'height_ratios': [1, 0.65]}
    )
    
    fig.suptitle(f"{dataset_type} | Optical Slice: {z_slice}", fontsize=20, y=0.97, fontweight='bold')
    
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
            
            boxes = parse_boxes(txt_path, w, h)
            for box in boxes:
                x1, y1, bw, bh, cls = box
                color = 'lime' if cls == 0 else 'yellow'
                rect = patches.Rectangle((x1, y1), bw, bh, linewidth=1, edgecolor=color, facecolor='none', alpha=0.8)
                ax_img.add_patch(rect)
                
            ax_img.set_title(f"{method}\n(Boxes: {len(boxes)})", fontsize=14, fontweight='bold', pad=8)

            pixel_data = img_16.flatten()
            pixel_data = pixel_data[pixel_data > 0] 
            
            if pixel_data.size > 0:
                mean_val = np.mean(pixel_data)
                median_val = np.median(pixel_data)
                
                ax_hist.hist(pixel_data, bins=100, log=True, color='royalblue', alpha=0.7)
                ax_hist.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, alpha=0.8, label=f'Mean: {mean_val:.0f}')
                ax_hist.axvline(median_val, color='green', linestyle='dotted', linewidth=2, alpha=0.8, label=f'Med: {median_val:.0f}')
                
                ax_hist.set_title("Intensity Distribution", fontsize=11, fontweight='bold', pad=6)
                ax_hist.set_xlabel("Pixel Intensity (16-bit)", fontsize=11, fontweight='bold')
                ax_hist.set_ylabel("Frequency (Log)", fontsize=11, fontweight='bold')
                ax_hist.tick_params(axis='both', which='major', labelsize=10)
                ax_hist.legend(loc='upper right', fontsize=10, framealpha=0.9)
            else:
                ax_hist.text(0.5, 0.5, "Noise Core Block", ha='center', va='center', fontsize=12, fontweight='bold')
                ax_hist.axis('off')
        else:
            ax_img.text(0.5, 0.5, "Image Offline", ha='center', va='center', color='red', fontsize=12, fontweight='bold')
            ax_img.axis('off')
            ax_hist.axis('off')

    # Applied fine-tuned h_pad=1.8 spacer parameters to guarantee no title masking overlaps
    plt.tight_layout(
        rect=[0.01, 0.01, 0.99, 0.94], 
        h_pad=1.8,                    
        w_pad=0.6                     
    )
    out_path = Path(out_folder) / f"{z_slice}_{dataset_type}_specific.png"
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    
    logging.info(f"  Target panel matrix exported successfully: {out_path.name}")
    return True

def main():
    if not CFG.get('enabled', True):
        logging.info("--- TARGETED FIGURE EXTRACTOR PIPELINE BYPASSED VIA CONFIG ---")
        return
        
    out_dir = Path(CFG['paths']['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    params = CFG['parameters']
    target_z_slices = params.get('target_z_slices', [])
    
    if isinstance(target_z_slices, str):
        target_z_slices = [target_z_slices]
        
    if not target_z_slices:
        logging.error("No entries identified inside the target_z_slices parameter array block.")
        return
        
    dataset_type = str(params.get('dataset_type', 'BOTH')).upper()
    dpi = params.get('dpi', 150)
    
    logging.info(f"--- TARGETED MULTI-SLICE EXTRACTION MODULE INITIALIZED ---")
    logging.info(f"Indexed {len(target_z_slices)} specific coordinate arrays to look up.")
    print("-" * 80 + "\n")
    
    for idx, slice_token in enumerate(target_z_slices, 1):
        slice_token = str(slice_token).strip()
        parts = slice_token.split('_')
        
        if len(parts) < 3:
            logging.error(f"[{idx}] Skipping incompatible coordinate layout string: '{slice_token}'. Needs 'X_Y_Z' template.")
            continue
            
        x_folder = parts[0]
        tile = f"{parts[0]}_{parts[1]}"
        
        logging.info(f"[{idx}/{len(target_z_slices)}] Querying volume for path signature: {slice_token}")
        
        if dataset_type == "BOTH":
            draw_figure(x_folder, tile, slice_token, "RAW", out_dir, dpi)
            draw_figure(x_folder, tile, slice_token, "STRETCHED", out_dir, dpi)
        elif dataset_type in ["RAW", "STRETCHED"]:
            draw_figure(x_folder, tile, slice_token, dataset_type, out_dir, dpi)
        else:
            logging.error(f"Unknown dataset_type token context parameter constraint: '{dataset_type}'.")
            return

    logging.info(f"--- PROCESS EXITED SUCCESSFULLY | Figure panel matrix pipeline complete ---")

if __name__ == '__main__':
    main()