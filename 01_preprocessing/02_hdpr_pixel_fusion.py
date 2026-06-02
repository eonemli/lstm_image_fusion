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
    
    modes = cfg_struct['ACQUISITION_MODES']
    channel = cfg_struct['CHANNEL']
    min_thresh = cfg_fuse.get('min_signal_threshold', 1)

    out_dir = Path(root_out) / "HDPR" / channel / tile_subpath
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / filename

    # Isolate the first exposure path to inspect the slice profile
    first_img_path = Path(root_in) / modes[0] / channel / tile_subpath / filename
    if not first_img_path.exists():
        return False

    # --- PADDING BYPASS VIA DIRECT FILESYSTEM COPY ---
    try:
        baseline_img = tifffile.imread(str(first_img_path))
        if baseline_img.max() <= min_thresh:
            # Directly copy the original file byte-for-byte to keep filesystem compression intact
            shutil.copy2(str(first_img_path), str(out_file))
            return True
    except Exception as e:
        logging.warning(f"Failed to process baseline check for {first_img_path}: {e}")
        return False

    # Load the remaining active exposure frames for this slice position
    images_16bit = [baseline_img]
    for mode in modes[1:]:
        img_path = Path(root_in) / mode / channel / tile_subpath / filename
        if not img_path.exists():
            return False
        images_16bit.append(tifffile.imread(str(img_path)))

    # --- TARGETED GROUP MAX INTENSITY RESCALING ---
    # Identify the real upper intensity boundary across this specific multi-exposure slice group
    group_max = float(max(img.max() for img in images_16bit))
    if group_max == 0:
        group_max = 1.0  # Safe guard against zero division

    # Normalize relative to the group maximum so values center correctly around Mertens' 0.5 exposure well
    images_norm = [img.astype(np.float32) / group_max for img in images_16bit]

    w = cfg_fuse['weights']
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=w['contrast'], 
        saturation_weight=w['saturation'], 
        exposure_weight=w['exposure']
    )

    # Execute multi-exposure blending across the normalized float spaces
    fusion = merge_mertens.process(images_norm)
    
    # --- OUTPUT RANGE CONTROL REGIMES ---
    if cfg_fuse.get('contrast_stretching', False):
        f_min = np.min(fusion)
        f_max = np.max(fusion)
        fusion = (fusion - f_min) / (f_max - f_min + 1e-8)
        fusion_16bit = np.clip(fusion * 65535.0, 0, 65535).astype(np.uint16)
    else:
        # Re-scale back up using the exact group maximum to protect global tile-to-tile intensity consistency
        fusion_16bit = np.clip(fusion * group_max, 0, 65535).astype(np.uint16)
    
    tifffile.imwrite(str(out_file), fusion_16bit, compression=None)
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

        task_args = [
            (fname, tile, struct, fuse_cfg, str(root_in), str(root_out)) 
            for fname in filenames
        ]

        with ProcessPoolExecutor(max_workers=fuse_cfg['num_cores']) as executor:
            executor.map(fuse_task, task_args)

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