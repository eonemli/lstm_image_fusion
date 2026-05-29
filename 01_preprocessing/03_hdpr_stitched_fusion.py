#!/usr/bin/env python3
"""
Light-Sheet Fluorescence Microscopy (LSFM) Preprocessing Pipeline.
Step 03: Global Stitched Whole-Organ Macro-Volume HDPR Fusion & Histogram Profiler.
"""

import os
import sys
import time
import yaml
import shutil
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import cv2
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configure unbuffered log stream explicitly optimized for direct cluster bsub tracking
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout
)

def load_config(path: str = "config.yaml") -> dict:
    """Loads configuration blocks safely from the centralized workspace yaml configuration."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)['code_03_stitched_fusion']
    except FileNotFoundError:
        logging.error(f"Centralized configuration file not found at: {path}")
        sys.exit(1)
    except KeyError:
        logging.error(f"Configuration key 'code_03_stitched_fusion' is missing from {path}")
        sys.exit(1)

CFG = load_config()

def format_time(seconds: float) -> str:
    """Converts raw seconds into an elegant execution clock readout."""
    return time.strftime("%H:%M:%S", time.gmtime(max(0.0, seconds)))

def generate_histogram(img_16: np.ndarray, img_name: str, method_name: str, out_folder: Path, dpi: int, bins: int):
    """Generates and exports high-impact statistical intensity histograms using log frequency charts."""
    logging.info(f"Computing bit-depth frequency distribution statistics for modality: [{method_name}]")
    
    # Flatten array matrix to calculate analytical distributions
    pixel_data = img_16.flatten()
    pixel_data = pixel_data[pixel_data > 0]  # Disregard background dark noise masks
    
    if pixel_data.size == 0:
        logging.warning(f"Aborting plot tracking: Modality image [{method_name}] contains no valid tissue signal (>0).")
        return

    # Extract whole-slice structural statistical anchors
    mean_val = np.mean(pixel_data)
    median_val = np.median(pixel_data)
    p05_val = np.percentile(pixel_data, 5)
    p95_val = np.percentile(pixel_data, 95)

    plt.figure(figsize=(12, 8))
    plt.hist(pixel_data, bins=bins, log=True, color='royalblue', alpha=0.7, edgecolor='black', linewidth=0.3)
    
    # Draw clear indicator lines for mathematical reference frames
    plt.axvline(mean_val, color='red', linestyle='solid', linewidth=2.5, label=f'Mean: {mean_val:.0f}')
    plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2.5, label=f'Median: {median_val:.0f}')
    plt.axvline(p05_val, color='orange', linestyle='dotted', linewidth=2.5, label=f'5th Pct: {p05_val:.0f}')
    plt.axvline(p95_val, color='purple', linestyle='dotted', linewidth=2.5, label=f'95th Pct: {p95_val:.0f}')
    
    # Apply publication-scale title and text adjustments
    plt.title(f"16-Bit Intensity Distribution Profile - Configuration: {method_name}\nTarget: {img_name}", fontsize=15, pad=15, fontweight='bold')
    plt.xlabel('Absolute Pixel Intensity Spectrum (16-bit Bit-Depth)', fontsize=13, fontweight='bold')
    plt.ylabel('Volumetric Pixel Allocation Frequency (Log Scale)', fontsize=13, fontweight='bold')
    plt.xlim(0, 65535) 
    plt.tick_params(axis='both', which='major', labelsize=11)
    
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9, facecolor='white', edgecolor='none')
    plt.grid(axis='y', alpha=0.2, linestyle='--')
    plt.tight_layout()
    
    plot_name = f"{method_name}_{img_name.replace('.tif', '').replace('.tiff', '')}_histogram.png"
    plot_path = out_folder / plot_name
    plt.savefig(plot_path, dpi=dpi)
    plt.close()
    
    logging.info(f"Statistical plot profile successfully exported: {plot_name}")

def main():
    if not CFG.get('enabled', True):
        logging.info("--- STITCHED IMAGE HIGH-RESOLUTION BLENDING PIPELINE BYPASSED VIA CONFIG ---")
        return

    input_paths = CFG.get('input_paths', {})
    out_dir = Path(CFG.get('output_dir', ''))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_paths:
        logging.error("No input stitched paths defined inside config.yaml.")
        return

    # Validate all file paths exist prior to starting heavy RAM calculations
    validated_paths = {}
    for label, path_str in input_paths.items():
        p = Path(path_str)
        if not p.exists():
            logging.error(f"Failed to locate macro-volume for [{label}] at path: {p}")
            return
        validated_paths[label] = p

    # Pull the baseline filename from the first indexed tracking file path
    first_label = list(validated_paths.keys())[0]
    base_name = validated_paths[first_label].name
    start_time = time.time()
    
    params = CFG.get('parameters', {})
    fused_label = params.get('fused_label', '640_HDPR')
    dpi = params.get('histogram_dpi', 300)
    bins = params.get('histogram_bins', 256)

    # 1. Load micro-volumes sequentially into memory
    images_16bit = {}
    logging.info(f"--- ADAPTIVE MULTI-POWER STITCHED FUSION ENGINE STARTED ---")
    for label, path in validated_paths.items():
        logging.info(f"Loading raw multi-gigabyte matrix for profile [{label}]...")
        images_16bit[label] = tifffile.imread(str(path))
    
    # Verify spatial resolution match uniformity across inputs
    shapes = [img.shape for img in images_16bit.values()]
    if not all(s == shapes[0] for s in shapes):
        logging.error(f"Dimension mismatch detected across input volumes: {shapes}. Aborting.")
        return
    logging.info(f"All dimensions verified. Global tracking layout size: {shapes[0]}")
    
    # 2. Deploy original control targets to the destination directory
    logging.info("Deploying raw control files to output directory tracking vectors...")
    for label, path in validated_paths.items():
        out_target = out_dir / f"{label}_{base_name}"
        shutil.copy2(str(path), out_target)
        logging.info(f"  Saved Image Copy: {out_target.name}")

    # 3. Mathematical Normalization and Multi-Exposure Fusion
    logging.info("Converting integer channels to 32-bit floating scales for matrix operations...")
    images_float = [img.astype(np.float32) / 65535.0 for img in images_16bit.values()]
    
    logging.info(f"Executing Laplacian pyramid exposure blending across all {len(images_float)} inputs...")
    merge_mertens = cv2.createMergeMertens()
    fused_float = merge_mertens.process(images_float)
    
    logging.info("Normalizing raw floating results and performing linear Min-Max dynamic range stretching...")
    fused_norm = cv2.normalize(fused_float, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    fused_16 = fused_norm.astype(np.uint16)
    
    # 4. Save Fused Output
    out_fused_tiff = out_dir / f"{fused_label}_{base_name}"
    logging.info(f"Exporting high-dynamic-range fused whole-organ macro matrix: {out_fused_tiff.name}")
    tifffile.imwrite(str(out_fused_tiff), fused_16)

    # 5. Generate Statistical Distribution Histograms
    logging.info("--- INITIATING STATISTICAL FREQUENCY MATRIX SPECTRUM CHECKS ---")
    for label, img in images_16bit.items():
        generate_histogram(img, base_name, label, out_dir, dpi, bins)
    generate_histogram(fused_16, base_name, fused_label, out_dir, dpi, bins)
    
    logging.info(f"--- PROCESS SUCCESSFUL | Whole-organ stitched volume assets compiled: {format_time(time.time() - start_time)} ---")

if __name__ == '__main__':
    main()