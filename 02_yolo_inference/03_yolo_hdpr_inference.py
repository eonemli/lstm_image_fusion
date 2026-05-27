#!/usr/bin/env python3
"""
Light-Sheet Fluorescence Microscopy (LSFM) Inference Pipeline.
Step 03: On-the-Fly Matrix Patching, YOLO Object Detection, and HDPR Late Fusion.
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
import torch
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from torchvision.ops import nms

# Configure logging stream explicitly optimized for direct cluster bsub tracking
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout
)

# Prevent internal threading clashes inside spawned subprocess sub-pools
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def load_config(path: str = "config.yaml") -> dict:
    """Loads configuration blocks cleanly from the centralized workspace yaml configuration."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)['code_03_inference']
    except FileNotFoundError:
        logging.error(f"Centralized configuration file not found at: {path}")
        sys.exit(1)

CFG = load_config()

COLOR_MAP = {
    0: (0, 255, 0),   # Neuron -> Green
    1: (255, 255, 0), # Glia -> Yellow/Cyan
}

def format_time(seconds: float) -> str:
    """Converts seconds into an elegant execution clock readout."""
    return time.strftime("%H:%M:%S", time.gmtime(max(0.0, seconds)))

def run_tiled_inference(img_16bit: np.ndarray, model: Any, device: str) -> List[List[float]]:
    """Slices raw 16-bit frames into overlapping patch arrays on-the-fly inside RAM."""
    h, w = img_16bit.shape
    min_val, max_val = img_16bit.min(), img_16bit.max()
    
    if max_val > min_val:
        img_8bit = ((img_16bit - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    else:
        img_8bit = np.zeros((h, w), dtype=np.uint8)

    if CFG['yolo'].get('color_mode', 'gray').lower() == 'green':
        blank = np.zeros_like(img_8bit)
        img_bgr = cv2.merge([blank, img_8bit, blank])
    else:
        img_bgr = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)

    tile_size = CFG['yolo']['tile_size']
    step = CFG['yolo']['step']
    tiles, coords = [], []
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            y_end, x_end = min(y + tile_size, h), min(x + tile_size, w)
            y_start, x_start = y, x
            
            if (y_end - y_start) < tile_size and h >= tile_size:
                y_start, y_end = h - tile_size, h
            if (x_end - x_start) < tile_size and w >= tile_size:
                x_start, x_end = w - tile_size, w
                
            tile = img_bgr[y_start:y_end, x_start:x_end]
            
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                tile = cv2.copyMakeBorder(
                    tile, 0, tile_size - tile.shape[0], 0, tile_size - tile.shape[1], 
                    cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )

            tiles.append(tile)
            coords.append((x_start, y_start))

    all_boxes = []
    batch_size = CFG['system']['batch_size']
    conf_thresh = CFG['yolo']['conf_thresh']
    
    for i in range(0, len(tiles), batch_size):
        batch_tiles = tiles[i:i+batch_size]
        batch_coords = coords[i:i+batch_size]
        
        results = model.predict(batch_tiles, verbose=False, conf=conf_thresh, device=device)
        
        for res, (x_start, y_start) in zip(results, batch_coords):
            if res.boxes:
                boxes = res.boxes.data.cpu().numpy()
                for box in boxes:
                    box[0] += x_start
                    box[1] += y_start
                    box[2] += x_start
                    box[3] += y_start
                    all_boxes.append(box)

    if all_boxes:
        all_boxes = np.array(all_boxes)
        boxes_tensor = torch.tensor(all_boxes[:, :4], dtype=torch.float32)
        scores_tensor = torch.tensor(all_boxes[:, 4], dtype=torch.float32)
        keep = nms(boxes_tensor, scores_tensor, CFG['yolo']['nms_iou_tiles']).cpu().numpy()
        return all_boxes[keep].tolist()
    return []

def save_txt_labels(boxes: List[list], txt_path: str, img_w: int, img_h: int):
    """Writes boxes safely out using normalized standard YOLO textual formats."""
    Path(txt_path).parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, 'w') as f:
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            xc = ((x1 + x2) / 2.0) / img_w
            yc = ((y1 + y2) / 2.0) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            f.write(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")

def draw_boxes(img: np.ndarray, boxes: List[list]):
    """Draws custom overlay bounding geometry directly onto display grids."""
    classes = CFG['yolo']['classes']
    for box in boxes:
        x1, y1, x2, y2, conf, cls = map(float, box)
        color = COLOR_MAP.get(int(cls), (255, 255, 255))
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{classes.get(int(cls), str(int(cls)))} {conf:.2f}"
        (t_w, t_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (int(x1), int(y1)-t_h-4), (int(x1)+t_w, int(y1)), color, -1)
        cv2.putText(img, label, (int(x1), int(y1)-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

def process_z_slice(args: Tuple[str, int, dict]) -> Tuple[str, str]:
    """Subprocess execution core. Handles lazy-loaded branch evaluation."""
    rel_path_str, gpu_id, global_struct = args
    device = f"cuda:{gpu_id}"
    rel_path = Path(rel_path_str)
    
    # Inject explicit model directory structure path securely from the YAML file
    model_root = CFG['paths'].get('model_root', '')
    if model_root and model_root not in sys.path:
        sys.path.append(model_root)
        
    from ultralytics import YOLO
    model = YOLO(CFG['paths']['model_path'])
    model.to(device)

    channel = global_struct['CHANNEL']
    modes = global_struct['ACQUISITION_MODES']
    hdpr_sub = global_struct.get('hdpr_folder', 'HDPR')
    
    active_branches = CFG['system'].get('process_branches', ["RAW", "STRETCHED"])
    resume_mode = CFG['system'].get('resume', False)
    min_thresh = CFG['system'].get('min_signal_threshold', 1)
    
    # --- LAZY PATH POPULATION GRID (Prevents KeyErrors for disabled tracks) ---
    processing_branches = {}
    if "RAW" in active_branches:
        processing_branches["RAW"] = {
            "power_root": Path(CFG['paths']['raw_root']),
            "hdpr_root": Path(CFG['paths']['raw_hdpr_root'])
        }
    if "STRETCHED" in active_branches:
        processing_branches["STRETCHED"] = {
            "power_root": Path(CFG['paths']['stretched_root']),
            "hdpr_root": Path(CFG['paths']['stretched_hdpr_root'])
        }
        
    branches_resumed, branches_padded = 0, 0
    
    for dataset_type, roots in processing_branches.items():
        late_txt_out = Path(CFG['paths']['output_txt_root']) / dataset_type / "HDPR_Late" / rel_path.parent / f"{rel_path.stem}.txt"
        
        if resume_mode and late_txt_out.exists():
            branches_resumed += 1
            continue
            
        images, all_boxes = {}, {}
        h_global, w_global = 0, 0
        has_signal = False 
        
        # 1. Process Individual Acquisition Modes
        for mode_folder in modes:
            img_path = roots['power_root'] / mode_folder / channel / rel_path
            if img_path.exists():
                img_16 = tifffile.imread(str(img_path))
                if img_16.max() <= min_thresh: continue
                
                has_signal = True
                if h_global == 0: h_global, w_global = img_16.shape
                
                boxes = run_tiled_inference(img_16, model, device)
                all_boxes[mode_folder] = boxes
                
                img_8 = cv2.normalize(img_16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                if CFG['yolo'].get('color_mode', 'gray').lower() == 'green':
                    blank = np.zeros_like(img_8)
                    images[mode_folder] = cv2.merge([blank, img_8, blank])
                else:
                    images[mode_folder] = cv2.cvtColor(img_8, cv2.COLOR_GRAY2BGR)
                
                txt_out = Path(CFG['paths']['output_txt_root']) / dataset_type / mode_folder / rel_path.parent / f"{rel_path.stem}.txt"
                save_txt_labels(boxes, str(txt_out), w_global, h_global)

        # 2. Process Image-Level Early Fusion Target Tracks
        hdpr_path = roots['hdpr_root'] / hdpr_sub / channel / rel_path
        if hdpr_path.exists():
            img_16 = tifffile.imread(str(hdpr_path))
            if img_16.max() > min_thresh:
                has_signal = True
                if h_global == 0: h_global, w_global = img_16.shape
                
                boxes = run_tiled_inference(img_16, model, device)
                all_boxes['HDPR_Early'] = boxes
                
                img_8 = cv2.normalize(img_16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                if CFG['yolo'].get('color_mode', 'gray').lower() == 'green':
                    blank = np.zeros_like(img_8)
                    images['HDPR_Early'] = cv2.merge([blank, img_8, blank])
                else:
                    images['HDPR_Early'] = cv2.cvtColor(img_8, cv2.COLOR_GRAY2BGR)
                
                txt_out = Path(CFG['paths']['output_txt_root']) / dataset_type / "HDPR_Early" / rel_path.parent / f"{rel_path.stem}.txt"
                save_txt_labels(boxes, str(txt_out), w_global, h_global)

        # 3. Handle Empty Background Protective Exit
        if not has_signal:
            branches_padded += 1
            continue

        # 4. Compute Late Fusion 
        late_boxes = []
        for mode_folder in modes:
            if mode_folder in all_boxes:
                late_boxes.extend(all_boxes[mode_folder])
                
        if late_boxes:
            lb = np.array(late_boxes)
            b_t = torch.tensor(lb[:, :4], dtype=torch.float32)
            s_t = torch.tensor(lb[:, 4], dtype=torch.float32)
            keep = nms(b_t, s_t, CFG['yolo']['nms_iou_fusion']).cpu().numpy()
            all_boxes['HDPR_Late'] = lb[keep].tolist()
        else:
            all_boxes['HDPR_Late'] = [] 
            
        save_txt_labels(all_boxes['HDPR_Late'], str(late_txt_out), w_global if w_global > 0 else 1, h_global if h_global > 0 else 1)

        # 5. Output Verification Grids if Flag Enabled
        if CFG['system'].get('generate_grids', False) and images:
            slice_out_dir = Path(CFG['paths']['output_grid_root']) / rel_path.parent / rel_path.stem
            slice_out_dir.mkdir(parents=True, exist_ok=True)
            
            for mode_folder in modes:
                if mode_folder in images:
                    img_to_save = images[mode_folder].copy()
                    draw_boxes(img_to_save, all_boxes[mode_folder])
                    cv2.imwrite(str(slice_out_dir / f"{mode_folder}_{dataset_type}.png"), img_to_save)

            for key in ['HDPR_Early', 'HDPR_Late']:
                if key in images or (key == 'HDPR_Late' and images):
                    img_to_save = images['HDPR_Early'].copy() if 'HDPR_Early' in images else list(images.values())[0].copy()
                    draw_boxes(img_to_save, all_boxes.get(key, []))
                    cv2.imwrite(str(slice_out_dir / f"{key}_{dataset_type}.png"), img_to_save)
        
    if branches_resumed == len(processing_branches):
        status = "Skipped (Resume)"
    elif branches_padded + branches_resumed == len(processing_branches):
        status = "Skipped (Padding)"
    else:
        status = "Evaluated"
        
    return rel_path.name, status

def main():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    global_cfg = yaml.safe_load(open("config.yaml", "r"))
    struct = global_cfg['structure']
    
    logging.info(f"--- HDPR ON-THE-FLY YOLO EVALUATION MODULE INITIALIZED ---")
    logging.info(f"Processing Track Branches Selectors: {CFG['system']['process_branches']}")
    
    blueprint_dir = Path(CFG['paths']['raw_root']) / struct['ACQUISITION_MODES'][0] / struct['CHANNEL']
    tiffs = list(blueprint_dir.rglob("*.tiff")) + list(blueprint_dir.rglob("*.tif"))
    
    rel_paths = [str(f.relative_to(blueprint_dir)) for f in tiffs]
    total_tasks = len(rel_paths)
    logging.info(f"Target Scan Scope Found: {total_tasks} volumetric Z-slices indexed.")

    tasks_done = 0
    start_time = time.time()
    
    num_cores = CFG['system']['num_cores']
    num_gpus = CFG['system'].get('num_gpus', 1)
    
    task_args = [(path, i % num_gpus, struct) for i, path in enumerate(rel_paths)]
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(process_z_slice, arg) for arg in task_args]
        
        for future in as_completed(futures):
            try:
                filename, status = future.result()
                tasks_done += 1
                elapsed = time.time() - start_time
                avg_time = elapsed / tasks_done
                remaining = avg_time * (total_tasks - tasks_done)
                percentage = (tasks_done / total_tasks) * 100
                
                logging.info(
                    f"[{percentage:6.2f}%] Status: {status.ljust(18)} | Slice: {filename.ljust(25)} | "
                    f"Progress: {tasks_done}/{total_tasks} | Remaining: {format_time(remaining)}"
                )
                sys.stdout.flush()
            except Exception as e:
                logging.error(f"Execution panic triggered on tracking thread: {e}")
                sys.stdout.flush()
                
    logging.info(f"--- PIPELINE STEP EXITED SUCCESSFULLY | Total Deployment Window: {format_time(time.time() - start_time)} ---")

if __name__ == '__main__':
    main()