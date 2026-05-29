#!/usr/bin/env python3
"""
Light-Sheet Fluorescence Microscopy (LSFM) Post-Processing Pipeline.
Step 10: Ground-Truth Multi-Exposure Label Fusion and Deduplication Engine.
"""

import os
import sys
import time
import yaml
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from torchvision.ops import nms

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
CFG = GLOBAL_CFG['code_10_gt_label_fusion']
STRUCT = GLOBAL_CFG['structure']

def main():
    if not CFG.get('enabled', True):
        logging.info("--- GROUND TRUTH LABEL FUSION PIPELINE BYPASSED VIA CONFIG ---")
        return
        
    subset_dir = Path(CFG['paths']['subset_root'])
    gt_folder_name = CFG['parameters'].get('gt_folder_name', 'labels_gt')
    iou_thresh = CFG['parameters'].get('nms_iou_threshold', 0.45)
    
    # Dynamically bind to global multi-exposure/power lists
    acquisition_modes = STRUCT['ACQUISITION_MODES']
    
    if not subset_dir.exists():
        logging.error(f"Validation cohort sandbox root missing at: {subset_dir}")
        return
        
    slice_folders = [d for d in subset_dir.iterdir() if d.is_dir()]
    
    logging.info("--- GROUND TRUTH CONSENSUS EXTRACTION MATRIX OPERATIONAL ---")
    logging.info(f"Scanning {len(slice_folders)} validation sandboxes for manual '{gt_folder_name}' tracks...")
    print("-" * 80 + "\n")
    
    processed_count = 0
    start_time = time.time()
    
    for slice_folder in slice_folders:
        labels_gt_dir = slice_folder / gt_folder_name
        
        # Guard clause: Skip gracefully if the annotator hasn't provided folders for this slice yet
        if not labels_gt_dir.exists():
            continue
            
        all_boxes = []
        
        # 1. Aggregate coordinates across all listed hardware acquisition modes
        for mode in acquisition_modes:
            txt_file = labels_gt_dir / f"{mode}.txt"
            if not txt_file.exists():
                continue
                
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        # Fallback confidence mapping to satisfy standard NMS sorting rules
                        score = float(parts[5]) if len(parts) > 5 else 1.0
                        
                        # Transform centers to standard min/max corner bounds for tensor computations
                        x1 = cx - (w / 2.0)
                        y1 = cy - (h / 2.0)
                        x2 = cx + (w / 2.0)
                        y2 = cy + (h / 2.0)
                        
                        all_boxes.append([x1, y1, x2, y2, score, cls_id])
                        
        output_txt = labels_gt_dir / "HDPR_Late.txt"
        
        # 2. Handle empty or un-annotated slices cleanly
        if len(all_boxes) == 0:
            open(output_txt, 'w').close()
            logging.info(f"Processed: {slice_folder.name.ljust(35)} | (Empty annotation layer generated)")
            processed_count += 1
            continue
            
        # 3. Class-Aware Non-Maximum Suppression (NMS) Engine (Executed on CPU for OS stability)
        all_boxes_np = np.array(all_boxes)
        final_boxes = []
        unique_classes = np.unique(all_boxes_np[:, 5])
        
        # Segment arrays by class token ID to prevent cross-biological feature suppression
        for cls in unique_classes:
            cls_mask = all_boxes_np[:, 5] == cls
            cls_boxes = all_boxes_np[cls_mask]
            
            b_tensor = torch.tensor(cls_boxes[:, :4], dtype=torch.float32)
            s_tensor = torch.tensor(cls_boxes[:, 4], dtype=torch.float32)
            
            # Extract independent survival index references
            keep_indices = nms(b_tensor, s_tensor, iou_thresh).numpy()
            
            for k in keep_indices:
                x1, y1, x2, y2, _, c_id = cls_boxes[k]
                
                # Reverse transformations back to standard YOLO normalized dimensions
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1
                
                final_boxes.append((int(c_id), cx, cy, w, h))
                
        # 4. Export Consensus Ground Truth text layer
        with open(output_txt, 'w') as f:
            for box in final_boxes:
                f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
                
        logging.info(f"Processed: {slice_folder.name.ljust(35)} | Consolidated down to {len(final_boxes)} cell markers.")
        processed_count += 1

    elapsed_time = time.time() - start_time
    logging.info(f"\n--- PROCESS SUCCESSFUL | Compiled ground-truth consensus layers for {processed_count} sandboxes in {elapsed_time:.1f}s ---")

if __name__ == '__main__':
    main()