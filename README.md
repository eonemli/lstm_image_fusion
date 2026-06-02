# High Dynamic Range Power (HDPR) Light-Sheet Microscopy Post-Processing Pipeline

An enterprise-grade, high-performance computing (HPC) computational suite engineered to automate multi-exposure pixel fusion, deep learning cell mapping (YOLOv8), and multi-expert consensus validation for Light-Sheet Fluorescence Microscopy (LSFM) whole-brain imaging datasets.

---

# Table of Contents

1. Global Workspace System Architecture
2. Operational Core Working Principles
   - 2.1 The Photometric Bottleneck
   - 2.2 Mathematical Pixel Fusion Framework
   - 2.3 Deep Learning Inference Windowing
   - 2.4 Blind Consensus Graph Mapping
3. Automated Multi-Step Pipeline Execution Guide
   - Phase I: Preprocessing & Exposure Fusion
   - Phase II: Deep Learning Inference
   - Phase III: Diagnostics & Post-Processing Visualization
   - Phase IV: Validation & Performance Metrics Suite
4. Centralized Configuration Reference (`config.yaml`)
5. Critical Infrastructure Safeguards & Best Practices
6. Getting Started

---

# 1. Global Workspace System Architecture

The project directory should follow the structure below:

```text
lstm_image_fusion/
│
├── config.yaml
├── bsub.sh
├── .gitignore
├── directory.txt
│
├── 01_preprocessing/
│   ├── 01_preprocess_clip_scale.py
│   ├── 02_hdpr_pixel_fusion.py
│   └── 03_hdpr_stitched_fusion.py
│
├── 02_yolo_inference/
│   └── 03_yolo_hdpr_inference.py
│
└── 03_postprocessing/
    ├── 04_compile_slice_ledger.py
    ├── 05_generate_grid_figures.py
    ├── 06_find_saturated_images.py
    ├── 07_generate_saturated_plots.py
    ├── 08_generate_specific_slice_grid.py
    ├── 09_extract_val_set.py
    ├── 10_fuse_gt_labels.py
    ├── 11_evaluate_spatial_detection.py
    ├── 12_evaluate_population_counting.py
    └── 13_evaluate_golden_subset.py
```

---

# 2. Operational Core Working Principles

## 2.1 The Photometric Bottleneck

Standard Light-Sheet Fluorescence Microscopy (LSFM) is fundamentally constrained by the tradeoff between underexposure and overexposure caused by tissue scattering and non-uniform fluorophore expression.

### Low Laser Power / Short Exposure

- Preserves fine cellular boundaries.
- Reduces saturation artifacts.
- Weak fluorescent structures may remain below the detection threshold.

### High Laser Power / Long Exposure

- Reveals dim cellular structures.
- Improves visibility of weak fluorescent signals.
- Saturates bright regions and merges neighboring cells into large pixel blobs that degrade object detection performance.

---

## 2.2 Mathematical Pixel Fusion Framework

To overcome the photometric bottleneck, the HDPR workflow acquires multiple images of the same optical slice at different laser powers.

For a two-power acquisition:

```text
I_low  = low-power image
I_high = high-power image

α = P_high / P_low
```

A Gaussian weighting function is applied:

```text
W(I) = exp(-(I - I_mid)^2 / (2σ²))
```

where:

```text
I_mid = midpoint intensity
σ     = weighting width parameter
```

The fused image is computed as:

```text
If I_high(x,y) ≥ τ:

    I_fused(x,y) = α · I_low(x,y)

Else:

    I_fused(x,y) =
        [ W(I_low) · (α · I_low(x,y))
          + W(I_high) · I_high(x,y) ]
        ------------------------------------------------
             W(I_low) + W(I_high)
```

where:

```text
τ = saturation threshold
```

This process smoothly transitions from the non-saturated low-power exposure to the scaled high-power exposure, producing a unified 16-bit image that preserves information across both dim and bright tissue regions.

---

## 2.3 Deep Learning Inference Windowing

The object detection pipeline uses a customized YOLOv8 architecture optimized for multi-class cellular detection.

```text
Class 0 = Neuron
Class 1 = Glia
```

Because YOLO operates on 8-bit inputs (0–255), fused 16-bit images must undergo dynamic range compression before inference.

Instead of using global normalization, the pipeline applies adaptive percentile-based min-max normalization:

```text
I_norm(x,y) =
    min(
        255,
        max(
            0,
            ((I16(x,y) - P_min) /
             (P_max - P_min)) * 255
        )
    )
```

where:

```text
I16   = fused 16-bit image
P_min = lower percentile bound
P_max = upper percentile bound
```

This adaptive normalization preserves local cellular contrast and fine structural details before tensors are passed into the neural network backbone.

---

## 2.4 Blind Consensus Graph Mapping

To establish an unbiased validation dataset, multiple experts independently annotate the same image slices.

The consensus engine constructs a graph where:

- Bounding boxes are represented as graph nodes.
- Edges connect boxes whose overlap exceeds a predefined threshold.

Intersection-over-Union (IoU) is computed as:

```text
IoU =
Area(B_Expert_A ∩ B_Expert_B)
--------------------------------
Area(B_Expert_A ∪ B_Expert_B)
```

An edge is created when:

```text
IoU ≥ IoU_threshold
```

Connected components or cliques represent agreed-upon cellular detections.

Class disagreements are resolved using majority voting, producing a consensus ground-truth dataset (`labels_gt`) that minimizes individual annotator bias.

---

# 3. Automated Multi-Step Pipeline Execution Guide

# Phase I: Preprocessing & Exposure Fusion

## 01_preprocess_clip_scale.py

### Purpose

Performs percentile clipping and converts 16-bit microscopy images into 8-bit representations.

### Input

```text
/raw_root/{ACQUISITION_MODE}/{CHANNEL}/{X_Folder}/{Tile}/*.tiff
```

### Output

```text
/stretched_root/{ACQUISITION_MODE}/{CHANNEL}/{X_Folder}/{Tile}/*.tiff
```

### Key Parameters

```yaml
clip_percentiles: [0.1, 99.9]
```

### Validation

- Rejects empty image files.
- Rejects corrupted image files.
- Logs missing metadata entries.

---

## 02_hdpr_pixel_fusion.py

### Purpose

Performs multi-exposure HDPR image fusion.

### Input

Reads multiple acquisition-power directories under:

```text
/raw_root/
```

### Output

```text
/raw_hdpr_root/{hdpr_folder}/{CHANNEL}/{X_Folder}/{Tile}/*.tiff
```

### Key Parameters

```yaml
saturation_threshold: 62000-65200
```

### Validation

- Verifies coordinate alignment between power levels.
- Flags stage shifts greater than 0.5 µm.

---

## 03_hdpr_stitched_fusion.py

### Purpose

Stitches neighboring tiles into whole-organ reconstructions.

### Method

- Normalized cross-correlation
- Linear alpha blending

### Input

```text
/Tile_00_00/
/Tile_00_01/
/Tile_00_02/
...
```

### Output

Whole-organ stitched image volumes.

### Validation

- Detects stitching artifacts.
- Detects spatial tearing and misalignment.

---

# Phase II: Deep Learning Inference

## 03_yolo_hdpr_inference.py

### Purpose

Runs YOLOv8 inference on HDPR images.

### Input

Supports:

- RAW images
- STRETCHED images

### Output

```text
/txt_root/{DATASET_TYPE}/{METHOD}/{X_Folder}/{Tile}/{Z_Slice}.txt
```

### Parameters

```yaml
confidence_threshold: 0.25
nms_iou_threshold: 0.45
```

### Output Format

```text
<class_id> <x_center> <y_center> <width> <height>
```

### Validation

Missing YOLO model weights (`.pt`) trigger immediate script termination.

---

# Phase III: Diagnostics & Post-Processing Visualization

## 04_compile_slice_ledger.py

### Purpose

Builds a tabular summary of detection counts across all models and acquisition conditions.

### Output

```text
/output_dir/sample_slice_ledger.txt
```

### Classification Rules

```text
Consistent Slices:

MaxBoxes - MinBoxes ≤ variance_threshold_consistent

Different Slices:

MaxBoxes - MinBoxes ≥ variance_threshold_different
```

### Validation

Slices containing fewer than five baseline detections are excluded.

---

## 05_generate_grid_figures.py

### Purpose

Generates multi-panel comparison figures.

### Output

```text
/{Cohort_Class}/{Z_Slice}_{Dataset_Type}.png
```

### Validation

- Automatically skips missing frames.
- Prevents failures caused by unavailable images.

---

## 06_find_saturated_images.py

### Purpose

Identifies highly saturated images from the highest laser-power acquisition.

### Output

```text
/{Z_Slice}/
```

### Parameters

```text
search_pool_size
target_count
max_per_tile
```

---

## 07_generate_saturated_plots.py

### Purpose

Generates visual comparison panels for saturated image examples.

### Output

```text
/{Z_Slice}_{Dataset_Type}_grid.png
```

### Supported Dataset Types

```text
RAW
STRETCHED
BOTH
```

### Features

- Runtime telemetry
- Error protection
- Progress estimation

---

## 08_generate_specific_slice_grid.py

### Purpose

Generates targeted visual inspection grids for user-specified coordinates.

### Output

```text
/{Z_Slice}_{Dataset_Type}_specific.png
```

### Configuration

Target slices are specified directly in `config.yaml`.

---

# Phase IV: Validation & Performance Metrics Suite

## 09_extract_val_set.py

### Purpose

Creates an unbiased validation subset.

### Output Structure

```text
/{Z_Slice}/
├── images/
├── labels/
└── labels_gt/
```

### Parameters

```yaml
min_detected_cells
total_slices_to_select
```

---

## 10_fuse_gt_labels.py

### Purpose

Creates consensus annotations from multiple expert label sets.

### Method

- IoU-based graph construction
- Majority-vote class assignment

### Output

```text
/{Z_Slice}/labels_gt/HDPR_Late.txt
```

---

## 11_evaluate_spatial_detection.py

### Purpose

Computes spatial detection performance metrics.

### Metrics

- Precision
- Recall
- F1 Score

### Outputs

```text
01_00_Spatial_Detection_Report.txt
01_01_Macro_F1_Summary_with_Variance.png
```

### Parameter

```yaml
iou_threshold: 0.45
```

---

## 12_evaluate_population_counting.py

### Purpose

Computes population-counting performance metrics.

### Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Bias

### Outputs

```text
02_00_Counting_Regression_Report.txt
02_04_Neuron_Counting_Scatter.png
```

---

## 13_evaluate_golden_subset.py

### Purpose

Identifies the highest-performing validation slices.

### Outputs

```text
03_00_Golden_Cohort_Report.txt
03_01_Golden_Combined_CM.png
```

### Parameters

```yaml
top_k_limit
ranking_class
ranking_metric
```

---

# 4. Centralized Configuration Reference (`config.yaml`)

All scripts read configuration values dynamically from a single master `config.yaml` file located at the repository root.

```yaml
structure:
  CHANNEL: "640nm"
  hdpr_folder: "HDPR"

  ACQUISITION_MODES:
    - "power3_exp50"
    - "power15_exp50"

code_05_grid_generation:
  enabled: true

  paths:
    ledger_path: "/path/to/ledger/sample_slice_ledger.txt"
    txt_root: "/path/to/output/txts"
    output_dir: "/path/to/output/grid_figures"

  parameters:
    target_samples_per_category: 10
    random_seed: 42
    variance_threshold_consistent: 3
    variance_threshold_different: 10

code_07_saturated_comparisons:
  enabled: true

  paths:
    saturated_root: "/path/to/output/sample_saturated_high"
    txt_root: "/path/to/output/sample_txts"
    output_dir: "/path/to/output/sample_saturated_figs"

  parameters:
    dpi: 300
    dataset_type: "BOTH"

evaluation_suite:
  paths:
    subset_root: "/path/to/output/sample_validation_subset"
    output_dir: "/path/to/output/sample_evaluation_results"

  structure:
    gt_folder_name: "labels_gt"
    gt_file_name: "HDPR_Late.txt"

  parameters:
    iou_threshold: 0.45
    dpi: 300
    font_scale: 1.2
    top_k_limit: 10
    ranking_method: "HDPR_Late"
    ranking_class: "Neuron"
    ranking_metric: "F1"
```

---

# 5. Critical Infrastructure Safeguards & Best Practices

## Avoid Hardcoded Paths

Never hardcode filesystem paths inside Python scripts.

All directory mappings should be managed through:

```text
config.yaml
```

---

## Python Path Resolution

An empty path:

```python
Path("")
```

resolves to:

```text
.
```

Always verify paths before opening files:

```python
if txt_path != Path(""):
    ...
```

This prevents errors such as:

```text
IsADirectoryError
```

---

## HPC Logging and Output Buffering

For long-running HPC jobs:

```python
logging.basicConfig(
    ...,
    stream=sys.stdout
)
```

Combined with:

```python
sys.stdout.flush()
```

This ensures real-time log updates and accurate progress reporting.

---

# 6. Getting Started

Clone the repository into your cluster workspace.

Update `config.yaml` with:

- Input image locations
- Output directories
- Acquisition channels
- Laser power settings
- YOLO model paths

Run an interactive diagnostic script:

```bash
python3 03_postprocessing/05_generate_grid_figures.py
```

For batch execution, configure resources in:

```text
bsub.sh
```

Then submit:

```bash
bsub < bsub.sh
```

---

# Pipeline Summary

This pipeline provides a complete workflow for:

- Multi-exposure HDPR image fusion
- YOLOv8-based neuron and glia detection
- Expert-consensus ground-truth generation
- Quantitative validation and benchmarking
- Large-scale HPC deployment for whole-brain LSFM datasets
