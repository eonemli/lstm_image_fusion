#!/usr/bin/env python3
"""
Light-Sheet Fluorescence Microscopy (LSFM) Post-Processing Pipeline.
Step 04: Compile Un-Aggregated Slice-Level Metadata Ledger.
"""

import os
import sys
import time
import yaml
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

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
            return yaml.safe_load(f)['code_04_analytics']
    except FileNotFoundError:
        logging.error(f"Centralized configuration file not found at: {path}")
        sys.exit(1)

CFG = load_config()
CLASSES = CFG['classes']

def format_time(seconds: float) -> str:
    """Converts raw seconds into an elegant execution clock readout."""
    return time.strftime("%H:%M:%S", time.gmtime(max(0.0, seconds)))

def parse_chunk(args: Tuple[List[str], str]) -> List[Dict[str, Any]]:
    """
    Worker core function. Processes a large chunk list of text labels in memory
    to eliminate inter-process communication serialization bottlenecks.
    """
    chunk_paths_str, txt_root_str = args
    txt_root = Path(txt_root_str)
    chunk_data = []
    
    for txt_path_str in chunk_paths_str:
        txt = Path(txt_path_str)
        
        # Parse nested directories: Dataset / Method / X_Folder / Tile(X_Y) / Z_Slice.txt
        rel_parts = txt.relative_to(txt_root).parts
        if len(rel_parts) < 5:
            continue
            
        dataset_type = rel_parts[0]     
        method = rel_parts[1]           
        x_folder = rel_parts[2]         
        tile_folder = rel_parts[3]      
        filename = rel_parts[-1]        
        z_slice = filename.replace('.txt', '')
        
        counts = {class_name: 0 for class_name in CLASSES.values()}
        
        # High-speed tally of detected bounding box instances per file
        if txt.stat().st_size > 0:
            with open(txt, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_name = CLASSES.get(class_id, "Unknown")
                        if class_name in counts:
                            counts[class_name] += 1
                            
        row_data = {
            "Dataset": dataset_type,
            "Method": method,
            "X_Folder": x_folder,
            "Tile": tile_folder,
            "Z_Slice": z_slice
        }
        
        raw_boxes = 0
        for class_name, count in counts.items():
            row_data[f"Raw_Boxes_{class_name}"] = count
            raw_boxes += count
            
        row_data["Raw_Boxes_Total"] = raw_boxes
        chunk_data.append(row_data)
        
    return chunk_data

def main():
    if not CFG.get('enabled', True):
        logging.info("--- SLICE METADATA LEDGER PIPELINE BYPASSED VIA CONFIG ---")
        return

    txt_root = Path(CFG['paths']['txt_root'])
    out_dir = Path(CFG['paths']['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"--- BATCH-PARALLEL VOLUMETRIC SLICE LEDGER ENGINE STARTED ---")
    logging.info(f"Scanning raw target prediction logs: {txt_root}")
    
    txt_files = list(txt_root.rglob("*.txt"))
    total_files = len(txt_files)
    
    if total_files == 0:
        logging.error("No annotation text files discovered. Verify paths inside config.yaml.")
        return

    num_cores = CFG.get('system', {}).get('num_cores', 16)
    num_cores = min(num_cores, os.cpu_count() or 1)
    logging.info(f"Indexed {total_files} slices. Spawning core parsing pool size: {num_cores}")
    
    # Pack paths sequentially into execution chunks
    chunk_size = 5000 
    chunks = [txt_files[i:i + chunk_size] for i in range(0, total_files, chunk_size)]
    task_args = [([str(p) for p in chunk], str(txt_root)) for chunk in chunks]
    
    data = []
    start_time = time.time()
    chunks_done = 0

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(parse_chunk, arg) for arg in task_args]
        
        for future in as_completed(futures):
            result_list = future.result()
            if result_list:
                data.extend(result_list)
                
            chunks_done += 1
            files_processed = min(chunks_done * chunk_size, total_files)
            
            elapsed = time.time() - start_time
            avg_time = elapsed / files_processed
            remaining = avg_time * (total_files - files_processed)
            percentage = (files_processed / total_files) * 100
            
            logging.info(
                f"[{percentage:6.2f}%] Logged: {str(files_processed).rjust(6)}/{total_files} files | "
                f"Elapsed: {format_time(elapsed)} | Remaining: {format_time(remaining)}"
            )
            sys.stdout.flush()

    # Convert parsed metadata array directly into spreadsheet dataframe
    df = pd.DataFrame(data)
    if df.empty:
        logging.error("Metadata parsing failure encountered. Inspect folder alignment maps.")
        return

    # Sort modalities dynamically to clean position baseline domains before HDPR fusion tracks
    unique_methods = df['Method'].unique().tolist()
    method_order = [m for m in unique_methods if "HDPR" not in m] + [m for m in unique_methods if "HDPR" in m]
    df['Method'] = pd.Categorical(df['Method'], categories=method_order, ordered=True)
    df = df.sort_values(by=["Dataset", "Method", "X_Folder", "Tile", "Z_Slice"])

    # Export out the single, comprehensive un-aggregated text ledger
    ledger_txt_path = out_dir / "01_slice_level_comparison.txt"
    logging.info(f"Compiling complete slice ledger track target...")
    
    # Save as a clean, tab-separated text file (.txt)
    df.to_csv(ledger_txt_path, sep='\t', index=False)
    
    logging.info(f"--- PROCESS SUCCESSFUL | Slice metadata ledger deployed: {ledger_txt_path} ---")

if __name__ == "__main__":
    main()