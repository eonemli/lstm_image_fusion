#!/usr/bin/env python3
"""
Light-Sheet Fluorescence Microscopy (LSFM) Post-Processing Pipeline.
Step 06: High-Speed Parallel Saturation Matrix Outlier Miner and Extractor Engine.
"""

import os
import sys
import time
import yaml
import shutil
import random
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tifffile

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
CFG = GLOBAL_CFG['code_06_saturation_finder']
STRUCT = GLOBAL_CFG['structure']

def format_time(seconds: float) -> str:
    """Converts raw seconds into an elegant execution clock readout."""
    return time.strftime("%H:%M:%S", time.gmtime(max(0.0, seconds)))

def evaluate_single_frame(args: Tuple[str, str, int]) -> Dict[str, Any]:
    """Worker core function executed in multiprocessing pools to calculate saturation ratios."""
    tiff_path_str, raw_dir_str, saturation_threshold = args
    tiff_path = Path(tiff_path_str)
    raw_dir = Path(raw_dir_str)
    
    try:
        img_16 = tifffile.imread(str(tiff_path))
        img_sub = img_16[::8, ::8] # 8x8 stride to protect CPU cache lines
        
        saturated_count = np.sum(img_sub >= saturation_threshold)
        score = float(saturated_count / img_sub.size)
        
        if score > 0:
            rel_path = tiff_path.relative_to(raw_dir)
            return {
                'score': score,
                'rel_path': str(rel_path),
                'tile': rel_path.parent.name,
                'slice': rel_path.stem
            }
    except Exception as e:
        pass
    return {}

def copy_corresponding_files(rel_path_str: str, out_folder: Path) -> int:
    """Gathers all variant configurations of a target Z-slice dynamically based on active path definitions."""
    paths = CFG['paths']
    channel = STRUCT['CHANNEL']
    hdpr_sub = STRUCT.get('hdpr_folder', 'HDPR')
    modes = STRUCT['ACQUISITION_MODES']
    rel_path = Path(rel_path_str)
    
    copy_manifest = []
    
    if 'raw_root' in paths:
        for mode in modes:
            src = Path(paths['raw_root']) / mode / channel / rel_path
            copy_manifest.append((src, f"{mode}_RAW.tiff"))
    if 'raw_hdpr_root' in paths:
        src = Path(paths['raw_hdpr_root']) / hdpr_sub / channel / rel_path
        copy_manifest.append((src, "HDPR_RAW.tiff"))
        
    if 'stretched_root' in paths:
        for mode in modes:
            src = Path(paths['stretched_root']) / mode / channel / rel_path
            copy_manifest.append((src, f"{mode}_STRETCHED.tiff"))
    if 'stretched_hdpr_root' in paths:
        src = Path(paths['stretched_hdpr_root']) / hdpr_sub / channel / rel_path
        copy_manifest.append((src, "HDPR_STRETCHED.tiff"))
    
    copied_count = 0
    for src_file, new_name in copy_manifest:
        if src_file.exists():
            shutil.copy2(str(src_file), out_folder / new_name)
            copied_count += 1
            
    return copied_count

def main():
    if not CFG.get('enabled', True):
        logging.info("--- SATURATION AUTOMATED FINDER PIPELINE BYPASSED VIA CONFIG ---")
        return

    out_root = Path(CFG['paths']['output_dir'])
    out_root.mkdir(parents=True, exist_ok=True)
    
    highest_mode = STRUCT['ACQUISITION_MODES'][-1]
    channel = STRUCT['CHANNEL']
    raw_dir = Path(CFG['paths']['raw_root']) / highest_mode / channel
    
    logging.info(f"--- PARALLEL VOLUMETRIC SATURATION OUTLIER MINER ENGAGED ---")
    logging.info(f"Target Baseline Mode: [{highest_mode}]")
    
    tiffs = list(raw_dir.rglob("*.tiff")) + list(raw_dir.rglob("*.tif"))
    total_files = len(tiffs)
    
    if total_files == 0:
        logging.error(f"No source frames encountered at location: {raw_dir}.")
        return
        
    params = CFG['parameters']
    search_pool_size = min(params['search_pool_size'], total_files) 
    logging.info(f"Indexed {total_files} assets. Allocating randomized search pool size: {search_pool_size}")
    
    random.seed(params['random_seed']) 
    search_pool = random.sample(tiffs, search_pool_size)
    
    # Pack parallel worker arguments
    thresh = params['saturation_threshold']
    task_args = [(str(p), str(raw_dir), thresh) for p in search_pool]
    
    # Dynamic resource allocation matching your cluster configurations
    num_cores = GLOBAL_CFG.get('code_04_analytics', {}).get('system', {}).get('num_cores', 16)
    num_cores = min(num_cores, os.cpu_count() or 1)
    logging.info(f"Spawning multiprocessing orchestration pool using {num_cores} cores...")
    print("-" * 80 + "\n")
    
    scored_images = []
    start_time = time.time()
    frames_processed = 0
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(evaluate_single_frame, arg) for arg in task_args]
        
        for future in as_completed(futures):
            res = future.result()
            if res:
                scored_images.append(res)
                
            frames_processed += 1
            if frames_processed % 100 == 0 or frames_processed == search_pool_size:
                elapsed = time.time() - start_time
                avg_time = elapsed / frames_processed
                remaining = avg_time * (search_pool_size - frames_processed)
                percentage = (frames_processed / search_pool_size) * 100
                logging.info(
                    f"[{percentage:6.2f}%] Evaluated: {frames_processed}/{search_pool_size} frames | "
                    f"Elapsed: {format_time(elapsed)} | Remaining: {format_time(remaining)}"
                )
                sys.stdout.flush()

    logging.info("\n--- SCAN COMPLETE. SORTING AND EXTRACTING OUTLIERS ---")
    
    # Sort dataset collections from highest pixel saturation volume down to minimum baseline boundary
    scored_images.sort(key=lambda x: x['score'], reverse=True)
    
    target_count = params['target_count']
    max_per_tile = params['max_per_tile']
    
    found_count = 0
    used_tiles = {}
    
    for item in scored_images:
        if found_count >= target_count:
            break
            
        tile_name = item['tile']
        if used_tiles.get(tile_name, 0) >= max_per_tile:
            continue
            
        found_count += 1
        used_tiles[tile_name] = used_tiles.get(tile_name, 0) + 1
        
        slice_name = item['slice']
        score_percent = item['score'] * 100
        
        slice_folder = out_root / slice_name
        slice_folder.mkdir(parents=True, exist_ok=True)
        
        copied = copy_corresponding_files(item['rel_path'], slice_folder)
        logging.info(
            f"Rank {str(found_count).rjust(3)}: Sub-grid: {tile_name}/{slice_name} | "
            f"Overexposed Ratio: {score_percent:6.2f}% | Copied: {copied} assets"
        )
        
    logging.info(f"\n--- PROCESS SUCCESSFUL | Saturated subsets deployed ({found_count} targets) inside: {out_root} ---")

if __name__ == '__main__':
    main()