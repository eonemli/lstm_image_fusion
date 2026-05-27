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

# Configure logging style for terminal and HPC cluster output files
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout  # <--- Directs all logs to stdout instead of stderr
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, float]:
        src_path = self.file_list[idx]
        try:
            img = tifffile.imread(src_path)
            max_val = np.max(img)
            return img.astype(np.float32), str(src_path), float(max_val)
        except Exception as e:
            logging.warning(f"Failed to read image {src_path}: {e}")
            return torch.zeros(1), str(src_path), 0.0

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

    total_tasks = len(modes) * len(tiles)
    tasks_done = 0
    total_copied = 0
    start_time = time.time()

    # --- HANDLING PIPELINE TOGGLE ---
    if not c01.get('enabled', True):
        logging.info("--- PREPROCESSING STEP BYPASSED VIA CONFIG ---")
        logging.info(f"Bypassing normalization. Direct copy mode initiated for {total_tasks} subdirectories...")
        
        for mode_folder in modes:
            for tile in tiles:
                src_dir = root_in / mode_folder / channel / tile
                dst_dir = root_out / mode_folder / channel / tile
                if src_dir.exists():
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    for f in src_dir.glob("*.tif*"):
                        shutil.copy(f, dst_dir / f.name)
                        total_copied += 1
                tasks_done += 1
                percent = (tasks_done / total_tasks) * 100
                print(f"[{percent:6.2f}%] Copied raw folder: {mode_folder} | {tile}", end="\r")
        logging.info(f"\n--- Direct Copy Complete. Total raw frames deployed: {total_copied} ---")
        return

    # --- STANDARD ACTIVE PROCESSING WORKFLOW ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode = c01['mode']
    p_params = c01['percentile_params']
    stride = p_params['stride']
    min_thresh = c01.get('min_signal_threshold', 1)

    logging.info(f"--- HDPR PRE-PROCESSING ACCELERATION ENGINE STARTED ---")
    logging.info(f"Target Compute Unit: [{device.type.upper()}]")
    logging.info(f"Total Folders to Normalize: {total_tasks}")
    logging.info(f"Isolating empty background volumes at Max Signal < {min_thresh}")
    print("-" * 80 + "\n")

    for mode_folder in modes:
        for tile in tiles:
            src_dir = root_in / mode_folder / channel / tile
            dst_dir = root_out / mode_folder / channel / tile
            
            tile_files = [str(f) for f in src_dir.glob("*.tif*")]
            if not tile_files:
                tasks_done += 1
                continue
                
            dst_dir.mkdir(parents=True, exist_ok=True)
            tile_copied = 0
            
            loader = DataLoader(
                LSMDataset(tile_files), 
                batch_size=c01['dataloader']['batch_size'], 
                num_workers=c01['dataloader']['num_workers'],
                shuffle=False
            )
            
            for images, paths, max_vals in loader:
                to_process_mask = max_vals >= min_thresh
                
                # 1. Background Masking (Zero Overlap Copy)
                for i in range(len(paths)):
                    if not to_process_mask[i]:
                        rel_path = os.path.relpath(paths[i], str(root_in))
                        shutil.copy(paths[i], os.path.join(str(root_out), rel_path))
                        tile_copied += 1

                # 2. Intensity Normalization & Contrast Stretching Engine
                if to_process_mask.any():
                    proc_images = images[to_process_mask].to(device)
                    proc_paths = [paths[i] for i, m in enumerate(to_process_mask) if m]
                    
                    if mode == "percentile":
                        # Subsample using spatial stride matrix to conserve GPU VRAM
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
                        tifffile.imwrite(os.path.join(str(root_out), rel_path), proc_np[i])

            # Progress Metrics Generation
            tasks_done += 1
            total_copied += tile_copied
            elapsed = time.time() - start_time
            avg_time = elapsed / tasks_done
            remaining = avg_time * (total_tasks - tasks_done)
            percentage = (tasks_done / total_tasks) * 100

            logging.info(
                f"[{percentage:6.2f}%] Mode: {mode_folder.ljust(15)} | Tile: {tile.ljust(15)} | "
                f"Padded: {str(tile_copied).rjust(3)} | Remaining: {format_time(remaining)}"
            )
            sys.stdout.flush() 

    logging.info(f"--- PROCESSING STRETCH MATRIX COMPLETE | Total Background Frames Protected: {total_copied} ---")

if __name__ == "__main__":
    main()