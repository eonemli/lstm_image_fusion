#!/usr/bin/env python3
"""
Light-Sheet Fluorescence Microscopy (LSFM) Post-Processing Pipeline.
Step 09: Unbiased Validation Dataset Cohort Extraction & Figure Grid Panel Generator.
"""

import os
import sys
import time
import yaml
import random
import shutil
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
CFG = GLOBAL_CFG['code_09_validation_extraction']
STRUCT = GLOBAL_CFG['structure']

def format_time(seconds: float) -> str:
    """Converts raw seconds into an elegant execution clock readout string."""
    return time.strftime("%H:%M:%S", time.gmtime(max(0.0, seconds)))

def count_cells_in_txt(txt_path: Path) -> int:
    """Quick thread-safe line count calculation to identify text-line boundaries."""
    try:
        with open(txt_path, 'r') as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0

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

def get_image_path(method: str, x_folder: str, tile: str, z_slice: str, dataset_type: str) -> Path:
    """Constructs the correct path for either raw powers or HDPR fusions from configuration definitions."""
    channel = STRUCT['CHANNEL']
    hdpr_dir = STRUCT.get('hdpr_folder', 'HDPR')
    
    if dataset_type == "RAW":
        base_root = Path(CFG['paths']['raw_root'])
        hdpr_root = Path(CFG['paths']['raw_hdpr_root'])
    else:
        base_root = Path(CFG['paths'].get('stretched_root', ''))
        hdpr_root = Path(CFG['paths'].get('stretched_hdpr_root', ''))
    
    if "HDPR" in method:
        return hdpr_root / hdpr_dir / channel / x_folder / tile / f"{z_slice}.tiff"
    else:
        mapping = STRUCT.get('raw_image_mapping', {})
        actual_img_folder = mapping.get(method, method)
        return base_root / actual_img_folder / channel / x_folder / tile / f"{z_slice}.tiff"

def generate_comparison_grid(x_folder: str, tile: str, z_slice: str, dataset_type: str, out_folder: Path, dpi: int):
    """Draws a high-resolution, custom layout 2xN grid matching manuscript configurations."""
    methods = STRUCT['ACQUISITION_MODES'] + ['HDPR_Early', 'HDPR_Late']
    num_cols = len(methods)
    
    # Configured row height distribution specs [1, 0.65] matching prior tools
    fig, axes = plt.subplots(
        2, num_cols, 
        figsize=(4.0 * num_cols, 7.2), 
        squeeze=False,
        gridspec_kw={'height_ratios': [1, 0.65]}
    )
    
    fig.suptitle(f"{dataset_type} Validation Profile | Optical Slice: {z_slice}", fontsize=20, y=0.97, fontweight='bold')
    
    for i, method in enumerate(methods):
        ax_img = axes[0, i]
        ax_hist = axes[1, i]
        
        img_path = get_image_path(method, x_folder, tile, z_slice, dataset_type)
        if not img_path.exists() and img_path.suffix == '.tiff':
            img_path = img_path.with_suffix('.tif')
            
        txt_path = Path(CFG['paths']['txt_root']) / dataset_type / method / x_folder / tile / f"{z_slice}.txt"
        
        if img_path.exists():
            img_16 = tifffile.imread(str(img_path))
            h, w = img_16.shape
            
            img_8 = cv2.normalize(img_16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            ax_img.imshow(img_8, cmap='gray')
            ax_img.axis('off')
            
            # Safeguard Guard: Check if file tracking links point correctly to labels on disk
            if txt_path.exists() and not txt_path.is_dir():
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

    # Configured the fine-tuned h_pad=1.8 and side border overrides
    plt.tight_layout(
        rect=[0.01, 0.01, 0.99, 0.94], 
        h_pad=1.8,                    
        w_pad=0.6                     
    )
    # file string syntax to output clean names without x_y string duplication
    out_path = out_folder / f"{z_slice}_Comparison_Grid.png"
    plt.savefig(out_path, dpi=dpi)
    plt.close()

def main():
    if not CFG.get('enabled', True):
        logging.info("--- VALIDATION COHORT EXTRACTION PIPELINE BYPASSED VIA CONFIG ---")
        return
        
    out_root = Path(CFG['paths']['output_dir'])
    txt_root = Path(CFG['paths']['txt_root'])
    
    params = CFG['parameters']
    dataset_type = params['dataset_type']
    baseline_method = params['selection_method']
    min_cells = params['min_detected_cells']
    target_count = params['total_slices_to_select']
    random.seed(params['random_seed'])
    
    target_txt_dir = txt_root / dataset_type / baseline_method
    
    if not target_txt_dir.exists():
        logging.error(f"Target dataset directory path missing: {target_txt_dir}")
        return
        
    logging.info(f"--- VALIDATION COHORT EXTRACTION ENGINE ENGAGED ---")
    logging.info(f"Scanning labels branch: [{dataset_type}/{baseline_method}] for tissue slices containing >= {min_cells} cells...")
    
    all_txt_files = list(target_txt_dir.rglob("*.txt"))
    valid_files_by_tile = {}
    total_valid = 0
    
    for txt_path in all_txt_files:
        if count_cells_in_txt(txt_path) >= min_cells:
            tile_name = txt_path.parent.name
            if tile_name not in valid_files_by_tile:
                valid_files_by_tile[tile_name] = []
            valid_files_by_tile[tile_name].append(txt_path)
            total_valid += 1
            
    if total_valid == 0:
        logging.error("Zero candidate coordinates matched your minimum density filter constraint thresholds.")
        return
        
    actual_target = min(target_count, total_valid)
    logging.info(f"Discovered {total_valid} candidates. Extracting {actual_target} matrices uniformly across {len(valid_files_by_tile)} physical tiles.")
    print("-" * 80 + "\n")

    # Round-Robin Tile Balancing selection algorithm
    selected_txt_paths = []
    available_tiles = list(valid_files_by_tile.keys())
    random.shuffle(available_tiles) 
    
    while len(selected_txt_paths) < actual_target and available_tiles:
        for tile in list(available_tiles): 
            if len(selected_txt_paths) >= actual_target: 
                break
            choices = valid_files_by_tile[tile]
            if not choices:
                available_tiles.remove(tile)
                continue
            selected_txt_paths.append(choices.pop(random.randrange(len(choices))))
            if not choices: 
                available_tiles.remove(tile)
                
    methods = STRUCT['ACQUISITION_MODES'] + ['HDPR_Early', 'HDPR_Late']
    start_time = time.time()
    slices_processed = 0
    
    for i, txt_path in enumerate(selected_txt_paths, 1):
        rel_parts = txt_path.relative_to(target_txt_dir).parts
        if len(rel_parts) < 3: 
            continue
            
        x_folder = rel_parts[0]
        tile_folder = rel_parts[1]
        z_slice = rel_parts[2].replace(".txt", "")
        
        # Build clean validation cohort subfolders architecture without duplicating coords
        slice_out_dir = out_root / z_slice
        img_dir = slice_out_dir / "images"
        lbl_dir = slice_out_dir / "labels"
        gt_dir = slice_out_dir / "labels_gt"
        
        for folder_path in [img_dir, lbl_dir, gt_dir]:
            folder_path.mkdir(parents=True, exist_ok=True)
        
        for method in methods:
            src_txt = txt_root / dataset_type / method / x_folder / tile_folder / f"{z_slice}.txt"
            if src_txt.exists():
                shutil.copy2(str(src_txt), lbl_dir / f"{method}.txt")
            
            if method != 'HDPR_Late':
                src_img = get_image_path(method, x_folder, tile_folder, z_slice, dataset_type)
                if not src_img.exists() and src_img.suffix == '.tiff':
                    src_img = src_img.with_suffix('.tif')
                    
                if src_img.exists():
                    shutil.copy2(str(src_img), img_dir / f"{method}.tiff")
        
        if params.get('generate_annotated_grids', False):
            generate_comparison_grid(x_folder, tile_folder, z_slice, dataset_type, slice_out_dir, params.get('dpi', 150))
            
        # Telemetry Engine Updates: Progress tracking block for stable cluster bsub reviews
        slices_processed += 1
        elapsed = time.time() - start_time
        avg_time = elapsed / slices_processed
        remaining = avg_time * (actual_target - slices_processed)
        percentage = (slices_processed / actual_target) * 100
        
        logging.info(
            f"[{percentage:6.2f}%] Extracted: {slices_processed}/{actual_target} | "
            f"Sandbox Slice: {z_slice.ljust(25)} | "
            f"Remaining: {format_time(remaining)}"
        )
        sys.stdout.flush()

    logging.info(f"\n--- PROCESS SUCCESSFUL | Saved {actual_target} sandboxes inside output directory: {out_root} ---")

if __name__ == '__main__':
    main()