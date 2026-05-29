#!/usr/bin/env python3
"""
Light-Sheet Fluorescence Microscopy (LSFM) Preprocessing Pipeline.
Step 01: Dynamic range windowing and percentile-based contrast stretching.
"""

import os
import sys
import time
import yaml
import shutil
import logging
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader

# Configure logging stream explicitly optimized for direct cluster bsub tracking
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

class LSMDataset(Dataset):
    """Efficient PyTorch Dataset for loading 16-bit volumetric microscopy images."""
    def __init__(self, file_list: List[str]):
        self.file_list = file_list

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str, float]:
        src_path = self.file_list[idx]
        try:
            img = tifffile.imread(src_path)
            max_val = np.max(img)
            return img.astype(np.float32), src_path, float(max_val)
        except Exception as e:
            logging.warning(f"Failed to read image {src_path}: {e}")
            return np.zeros((1, 1), dtype=np.float32), src_path, 0.0

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
    c01 = cfg['code_01_preprocessing']
    
    root_in = Path(c01['input_dir'])
    root_out = Path(c01['output_dir'])
    modes = struct['ACQUISITION_MODES']
    channel = struct['CHANNEL']
    
    tiles = discover_tiles(root_in, modes, channel, struct['TILES'])
    if not tiles:
        logging.error("No valid processing targets found. Verify path structure inside config.yaml.")
        return

    # Count total files to establish an accurate global progress metric tracker
    total_files = 0
    files_per_mode = {}
    for mode_folder in modes:
        files_per_mode[mode_folder] = []
        for tile in tiles:
            src_dir = root_in / mode_folder / channel / tile
            if src_dir.exists():
                matched_files = [str(f) for f in src_dir.glob("*.tif*")]
                files_per_mode[mode_folder].extend(matched_files)
                total_files += len(matched_files)

    start_time = time.time()
    global_files_processed = 0
    total_copied = 0

    # --- HANDLING PIPELINE TOGGLE: DIRECT CLUSTER COPY ---
    if not c01.get('enabled', True):
        logging.info("--- PREPROCESSING STEP BYPASSED VIA CONFIG ---")
        logging.info(f"Direct copy mode initiated for {total_files} indexed frames...")
        
        for mode_folder in modes:
            for tile in tiles:
                src_dir = root_in / mode_folder / channel / tile
                dst_dir = root_out / mode_folder / channel / tile
                if src_dir.exists():
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    for f in src_dir.glob("*.tif*"):
                        shutil.copy2(f, dst_dir / f.name)
                        total_copied += 1
                        global_files_processed += 1
                        
                if global_files_processed % 500 == 0 or global_files_processed == total_files:
                    percent = (global_files_processed / total_files) * 100
                    logging.info(f"[COPY PROGRESS] Deployed: {global_files_processed}/{total_files} files ({percent:6.2f}%)")
                    sys.stdout.flush()
                    
        logging.info(f"--- Direct Copy Complete. Total raw frames deployed: {total_copied} ---")
        return

    # --- STANDARD ACTIVE STREAM PROCESSING WORKFLOW ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode = c01['mode']
    p_params = c01['percentile_params']
    stride = p_params['stride']
    min_thresh = c01.get('min_signal_threshold', 1)

    logging.info(f"--- HDPR PRE-PROCESSING STREAM ENGINE STARTED ---")
    logging.info(f"Target Compute Unit Hardware: [{device.type.upper()}]")
    logging.info(f"Total Volumetric Files Indexed: {total_files}")
    logging.info(f"Isolating empty background volumes at Max Signal < {min_thresh}")
    print("-" * 80 + "\n")

    for mode_folder in modes:
        mode_files = files_per_mode[mode_folder]
        if not mode_files:
            continue
            
        logging.info(f"Initializing continuous parallel stream loader for track: [{mode_folder}] ({len(mode_files)} slices)")
        
        # Instantiate DataLoader ONCE per modality branch to prevent worker spawning overhead
        loader = DataLoader(
            LSMDataset(mode_files), 
            batch_size=c01['dataloader']['batch_size'], 
            num_workers=c01['dataloader']['num_workers'],
            shuffle=False,
            drop_last=False
        )
        
        for images, paths, max_vals in loader:
            batch_size = len(paths)
            to_process_mask = max_vals >= min_thresh
            
            # 1. Background Masking (Zero Overlap Copy to bypass calculations)
            for i in range(batch_size):
                if not to_process_mask[i]:
                    rel_path = os.path.relpath(paths[i], str(root_in))
                    dest_file = Path(root_out) / rel_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(paths[i], dest_file)
                    total_copied += 1

            # 2. Parallel Intensity Normalization & Contrast Stretching Engine
            if to_process_mask.any():
                proc_images = images[to_process_mask].to(device)
                proc_paths = [paths[i] for i, m in enumerate(to_process_mask) if m]
                
                if mode == "percentile":
                    # Subsample using spatial stride matrix to conserve compute unit VRAM
                    samples = proc_images[:, ::stride, ::stride].reshape(proc_images.shape[0], -1)
                    vmin = torch.quantile(samples, p_params['p_low'] / 100.0, dim=1).view(-1, 1, 1)
                    vmax = torch.quantile(samples, p_params['p_high'] / 100.0, dim=1).view(-1, 1, 1)
                    proc_images = torch.clamp(proc_images, min=vmin, max=vmax)
                    proc_images = (proc_images - vmin) / (vmax - vmin + 1e-8) * 65535.0
                else:
                    limit = c01['limit_params']['clip_limit']
                    proc_images = torch.clamp(proc_images, max=limit) * (65535.0 / limit)

                proc_np = proc_images.cpu().numpy().astype(np.uint16)
                for i, s_path in enumerate(proc_paths):
                    rel_path = os.path.relpath(s_path, str(root_in))
                    dest_file = Path(root_out) / rel_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(str(dest_file), proc_np[i])

            # Global Metrics Calculation Update
            global_files_processed += batch_size
            if global_files_processed % 100 == 0 or global_files_processed == total_files:
                elapsed = time.time() - start_time
                avg_time = elapsed / global_files_processed
                remaining = avg_time * (total_files - global_files_processed)
                percentage = (global_files_processed / total_files) * 100

                logging.info(
                    f"[{percentage:6.2f}%] Streamed: {global_files_processed}/{total_files} | "
                    f"Current Mode Branch: {mode_folder.ljust(15)} | "
                    f"Padded Copies: {str(total_copied).rjust(5)} | "
                    f"Remaining: {format_time(remaining)}"
                )
                sys.stdout.flush() 

    logging.info(f"--- PREPROCESSING STEP COMPLETE | Total Background Frames Protected: {total_copied} ---")

if __name__ == "__main__":
    main()