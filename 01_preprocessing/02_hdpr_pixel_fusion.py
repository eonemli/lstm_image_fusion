#!/usr/bin/env python3
"""
Light-Sheet Fluorescence Microscopy (LSFM) Preprocessing Pipeline.
Step 02: High Dynamic Power Range (HDPR) Pixel-Level (Early) Fusion Engine.
"""

import os
import sys
import time
import yaml
import shutil
import logging
from pathlib import Path
from typing import List, Tuple, Union
from concurrent.futures import ProcessPoolExecutor

import cv2
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
    """Loads configuration values safely from a YAML file."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {path}")
        sys.exit(1)

def format_time(seconds: float) -> str:
    """Formats raw seconds into an elegant clock string."""
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def fuse_task(args: Tuple[str, str, dict, dict, str, str]) -> bool:
    """
    Worker function executed inside multiprocessing pools.
    Blends a single Z-slice across all configured power and exposure modes.
    """
    filename, tile_subpath, cfg_struct, cfg_fuse, root_in, root_out = args
    
    images_16bit = []
    modes = cfg_struct['ACQUISITION_MODES'] # General loop handles any permutation grid
    channel = cfg_struct['CHANNEL']
    min_thresh = cfg_fuse.get('min_signal_threshold', 1)

    # Load images across all available power/exposure domains
    for mode in modes:
        img_path = Path(root_in) / mode / channel / tile_subpath / filename
        if not img_path.exists():
            return False
            
        img = tifffile.imread(str(img_path))
        images_16bit.append(img)

    if not images_16bit:
        return False

    out_dir = Path(root_out) / "HDPR" / channel / tile_subpath
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = str(out_dir / filename)

    # --- PADDING BYPASS LOGIC ---
    # If baseline acquisition has no structure, bypass calculations to protect processing limits
    if images_16bit[0].max() <= min_thresh:
        # Save completely uncompressed to maintain strict volumetric shape and file size consistency
        tifffile.imwrite(out_file, images_16bit[0], compression=None)
        return True

    # --- CORE PYRAMID EXPOSURE FUSION MATH ---
    # Convert integer matrices to [0.0, 1.0] floats for OpenCV operators
    images_norm = [img.astype(np.float32) / 65535.0 for img in images_16bit]

    w = cfg_fuse['weights']
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=w['contrast'], 
        saturation_weight=w['saturation'], 
        exposure_weight=w['exposure']
    )

    # Execute Laplacian and Gaussian pyramid multi-exposure blending
    fusion = merge_mertens.process(images_norm)
    
    # --- OPTIONAL CONTRAST STRETCHING (Requirement 1) ---
    if cfg_fuse.get('contrast_stretching', False):
        f_min = np.min(fusion)
        f_max = np.max(fusion)
        fusion = (fusion - f_min) / (f_max - f_min + 1e-8)
    
    # Clip extreme ranges safely and map back to uint16 bit space
    fusion = np.clip(fusion, 0.0, 1.0)
    fusion_16bit = (fusion * 65535.0).astype(np.uint16)
    
    tifffile.imwrite(out_file, fusion_16bit, compression=None)
    return True

def discover_tiles(input_root: Path, modes: List[str], channel: str, tiles_config: Union[str, list]) -> List[str]:
    """Dynamically parses and builds a sorted list of tile subdirectories."""
    if isinstance(tiles_config, list): 
        return tiles_config
    
    tiles = set()
    ref_dir = input_root / modes[0] / channel
    if not ref_dir.exists():
        logging.error(f"Reference directory does not exist: {ref_dir}")
        return []
        
    tiffs = list(ref_dir.rglob("*.tiff")) + list(ref_dir.rglob("*.tif"))
    for f in tiffs:
        tiles.add(str(f.parent.relative_to(ref_dir)))
    return sorted(list(tiles))

def main():
    cfg = load_config()
    struct = cfg['structure']
    fuse_cfg = cfg['code_02_fusion']
    
    root_in = Path(fuse_cfg['input_dir'])
    root_out = Path(fuse_cfg['output_dir'])
    modes = struct['ACQUISITION_MODES']
    channel = struct['CHANNEL']
    
    tiles = discover_tiles(root_in, modes, channel, struct['TILES'])
    if not tiles:
        logging.error("No valid processing subdirectories found. Check path keys.")
        return

    total_tiles = len(tiles)
    start_time = time.time()

    # --- PIPELINE STEP ACTIVE TOGGLE CHECK ---
    if not fuse_cfg.get('enabled', True):
        logging.info("--- HDPR FUSION PIPELINE BYPASSED VIA CONFIG ---")
        return

    logging.info(f"--- HDPR MULTI-EXPOSURE IMAGING FUSION ENGINE STARTED ---")
    logging.info(f"Input Data Source: {root_in.name}")
    logging.info(f"Dynamic Contrast Stretching Windowing: {fuse_cfg.get('contrast_stretching', False)}")
    logging.info(f"Acquisition Modes Input Counts: {len(modes)} subdomains")
    logging.info(f"Allocated Core Infrastructure: {fuse_cfg['num_cores']} processes")
    print("-" * 80 + "\n")

    for i, tile in enumerate(tiles):
        ref_mode_dir = root_in / modes[0] / channel / tile
        filenames = [f.name for f in ref_mode_dir.glob("*.tif*")]
        
        if not filenames:
            continue

        # Package parallel tasks cleanly
        task_args = [
            (fname, tile, struct, fuse_cfg, str(root_in), str(root_out)) 
            for fname in filenames
        ]

        # Multi-core thread distribution per tile
        with ProcessPoolExecutor(max_workers=fuse_cfg['num_cores']) as executor:
            executor.map(fuse_task, task_args)

        # Progress reporting metrics calculation
        done_tiles = i + 1
        elapsed = time.time() - start_time
        avg_per_tile = elapsed / done_tiles
        remaining = avg_per_tile * (total_tiles - done_tiles)
        percentage = (done_tiles / total_tiles) * 100

        logging.info(
            f"[{percentage:6.2f}%] FUSED TILE: {tile.ljust(15)} | "
            f"Slices: {str(len(filenames)).rjust(4)} | "
            f"Elapsed: {format_time(elapsed)} | "
            f"Remaining: {format_time(remaining)}"
        )
        sys.stdout.flush()

    logging.info(f"--- FUSION PROCESSING ARCHITECTURE MATRIX METRIC REBUILT | Target Location: {root_out} ---")

if __name__ == '__main__':
    main()