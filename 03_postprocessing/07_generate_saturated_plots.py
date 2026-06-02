#!/usr/bin/env python3
"""
Light-Sheet Fluorescence Microscopy (LSFM) Post-Processing Pipeline.
Step 07: High-Resolution 2xN Overexposure Grid Visualization Dashboard.
"""

import os
import sys
import time
import yaml
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configure unbuffered log stream explicitly optimized for HPC cluster log tracing
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
CFG = GLOBAL_CFG['code_07_saturated_comparisons']
STRUCT = GLOBAL_CFG['structure']

def format_time(seconds: float) -> str:
    """Formats raw seconds into an elegant execution clock readout string."""
    return time.strftime("%H:%M:%S", time.gmtime(max(0.0, seconds)))

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

def find_txt_path(txt_root: Path, dataset_type: str, method: str, z_slice: str) -> Path:
    """Recursively crawls prediction directories to locate specific target label coordinates."""
    branch_dir = txt_root / dataset_type / method
    if not branch_dir.exists():
        return Path("")
    matches = list(branch_dir.rglob(f"{z_slice}.txt"))
    return matches[0] if matches else Path("")

def draw_saturated_grid(folder: Path, dataset_type: str, z_slice: str, out_dir: Path, out_dpi: int) -> bool:
    """Generates a comprehensive 2xN publication panel matching standard layout metrics."""
    methods = STRUCT['ACQUISITION_MODES'] + ['HDPR_Early', 'HDPR_Late']
    num_cols = len(methods)
    txt_root = Path(CFG['paths']['txt_root'])
    
    # Pre-flight Path Verification Guard
    # Check if files exist for the requested type before building matplotlib containers
    files_exist = False
    for method in methods:
        file_alias = "HDPR" if "HDPR" in method else method
        if (folder / f"{file_alias}_{dataset_type}.tiff").exists():
            files_exist = True
            break
            
    if not files_exist:
        return False

    fig, axes = plt.subplots(
        2, num_cols, 
        figsize=(4.0 * num_cols, 7.2),
        gridspec_kw={'height_ratios': [1, 0.65]}
    )
    fig.suptitle(f"Saturated Outlier Profile ({dataset_type}) | Optical Slice: {z_slice}", fontsize=20, y=0.97, fontweight='bold')
    
    for i, method in enumerate(methods):
        ax_img = axes[0, i]
        ax_hist = axes[1, i]
        
        file_alias = "HDPR" if "HDPR" in method else method
        tiff_path = folder / f"{file_alias}_{dataset_type}.tiff"
        txt_path = find_txt_path(txt_root, dataset_type, method, z_slice)
        
        if tiff_path.exists():
            img_16 = tifffile.imread(str(tiff_path))
            h, w = img_16.shape
            
            img_8 = cv2.normalize(img_16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            ax_img.imshow(img_8, cmap='gray')
            ax_img.axis('off')
            
            if txt_path != Path(""):
                boxes = parse_boxes(txt_path, w, h)
                for box in boxes:
                    x1, y1, bw, bh, cls = box
                    color = 'lime' if cls == 0 else 'yellow'
                    rect = patches.Rectangle((x1, y1), bw, bh, linewidth=1, edgecolor=color, facecolor='none', alpha=0.8)
                    ax_img.add_patch(rect)
            else:
                boxes = []
                
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

    plt.tight_layout(
        rect=[0.01, 0.01, 0.99, 0.94], 
        h_pad=1.8,                    
        w_pad=0.6                     
    )
    out_path = out_dir / f"{z_slice}_{dataset_type}_grid.png"
    plt.savefig(out_path, dpi=out_dpi)
    plt.close()
    return True

def main():
    if not CFG.get('enabled', True):
        logging.info("--- OVEREXPOSURE ANALYSIS PLOTTER BYPASSED VIA CONFIG ---")
        return

    saturated_root = Path(CFG['paths']['saturated_root'])
    out_dir = Path(CFG['paths']['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not saturated_root.exists():
        logging.error(f"Extracted overexposure dataset root missing at: {saturated_root}")
        return
        
    folders = [d for d in saturated_root.iterdir() if d.is_dir()]
    total_folders = len(folders)
    if total_folders == 0:
        logging.error("No saturated slice subsets discovered. Verify step 06 parameters.")
        return
        
    params = CFG['parameters']
    out_dpi = params.get('dpi', 150)
    
    # Centralized Configuration Resolution Block
    dataset_config = str(params.get('dataset_type', 'BOTH')).upper().strip()
    if dataset_config == "BOTH":
        target_branches = ["RAW", "STRETCHED"]
    elif dataset_config in ["RAW", "STRETCHED"]:
        target_branches = [dataset_config]
    else:
        logging.error(f"Unknown dataset_type attribute in config: '{dataset_config}'. Use RAW, STRETCHED, or BOTH.")
        return

    logging.info(f"--- SATURATED ANALYSIS GRAPHICS ENGINE STARTED ---")
    logging.info(f"Processing Track Selection Constraints: {target_branches}")
    logging.info(f"Scanning target index counts: {total_folders} directories found.")
    print("-" * 80 + "\n")
    
    start_time = time.time()
    generated_count = 0
    folders_processed = 0
    
    for folder in folders:
        z_slice = folder.name
        
        # Execute plotting runs restricted exclusively to your targeted config selection
        for branch in target_branches:
            success = draw_saturated_grid(folder, branch, z_slice, out_dir, out_dpi)
            if success:
                generated_count += 1
            
        folders_processed += 1
        elapsed = time.time() - start_time
        avg_time_per_folder = elapsed / folders_processed
        remaining_time = avg_time_per_folder * (total_folders - folders_processed)
        percentage = (folders_processed / total_folders) * 100
        
        logging.info(
            f"[{percentage:6.2f}%] Generated: {folders_processed}/{total_folders} | "
            f"Slice: {z_slice.ljust(25)} | "
            f"Remaining: {format_time(remaining_time)}"
        )
        sys.stdout.flush()
        
    logging.info(f"\n--- PROCESS SUCCESSFUL | Compiled {generated_count} overexposure figures inside: {out_dir} ---")

if __name__ == '__main__':
    main()