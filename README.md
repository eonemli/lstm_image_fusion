# High Dynamic Range Power (HDPR) Light-Sheet Microscopy Post-Processing Pipeline

An enterprise-grade, high-performance computing (HPC) computational suite engineered to automate multi-exposure pixel fusion, deep learning cell mapping (YOLOv8), and multi-expert consensus validation for Light-Sheet Fluorescence Microscopy (LSFM) whole-brain imaging datasets.

---

# Table of Contents

1. [Global Workspace System Architecture](#1-global-workspace-system-architecture)
2. [Operational Core Working Principles](#2-operational-core-working-principles)
   - [2.1 The Photometric Bottleneck](#21-the-photometric-bottleneck)
   - [2.2 Mathematical Pixel Fusion Framework](#22-mathematical-pixel-fusion-framework)
   - [2.3 Deep Learning Inference Windowing](#23-deep-learning-inference-windowing)
   - [2.4 Blind Consensus Graph Mapping](#24-blind-consensus-graph-mapping)
3. [Automated Multi-Step Pipeline Execution Guide](#3-automated-multi-step-pipeline-execution-guide)
   - [Phase I: Preprocessing & Fusion](#phase-i-preprocessing--exposure-fusion)
   - [Phase II: Deep Learning Inference](#phase-ii-deep-learning-inference)
   - [Phase III: Diagnostics & Post-Processing Visualization](#phase-iii-diagnostics--post-processing-visualization)
   - [Phase IV: Validation & Performance Metrics Suite](#phase-iv-validation--performance-metrics-suite)
4. [Centralized Configuration Reference (config.yaml)](#4-centralized-configuration-reference-configyaml)
5. [Critical Infrastructure Safeguards & Best Practices](#5-critical-infrastructure-safeguards--best-practices)
6. [Getting Started](#6-getting-started)

---

# 1. Global Workspace System Architecture

The project directory must match the structural organization below to guarantee robust cross-module path parsing and unhindered SLURM/LSF cluster execution:

```text
lstm_image_fusion/
│
├── config.yaml                     # Single master workspace configuration for all steps
├── bsub.sh                         # LSF HPC batch cluster submission script template
├── .gitignore                      # Git exclusion rules for large microscopy TIFF tracks
├── directory.txt                   # Automated repository structural layout mapping
│
├── 01_preprocessing/
│   ├── 01_preprocess_clip_scale.py # Multi-channel percentile clipping & 16-to-8-bit scaling
│   ├── 02_hdpr_pixel_fusion.py     # Multi-power exposure alignment & HDR linear fusion engine
│   └── 03_hdpr_stitched_fusion.py  # Spatial tile stitching matrix compiler for fused ranges
│
├── 02_yolo_inference/
│   └── 03_yolo_hdpr_inference.py   # Multi-track YOLO deep learning batch prediction pipeline
│
└── 03_postprocessing/
    ├── 04_compile_slice_ledger.py  # Master database indexer tracking raw box variance
    ├── 05_generate_grid_figures.py # Curated 2xN comparative evaluation panel generator
    ├── 06_find_saturated_images.py # Parallel overexposure matrix miner & validation extractor
    ├── 07_generate_saturated_plots.py # Publication-ready dashboard for overexposed outliers
    ├── 08_generate_specific_slice_grid.py # Targeted multi-slice coordinate dashboard generator
    ├── 09_extract_val_set.py       # Unbiased validation cohort sampler & sandbox builder
    ├── 10_fuse_gt_labels.py        # Multi-expert consensus ground truth label fusion resolver
    ├── 11_evaluate_spatial_detection.py # Micro/Macro F1 stats & 1xN combined confusion matrix plot
    ├── 12_evaluate_population_counting.py # Population density regression tracker & summary table image
    └── 13_evaluate_golden_subset.py # Top-K peak performance golden cohort miner & visual profiler

```

---

# 2. Operational Core Working Principles

## 2.1 The Photometric Bottleneck

Standard Light-Sheet Fluorescence Microscopy (LSFM) is fundamentally constrained by the tradeoff between underexposure and overexposure caused by tissue scattering and non-uniform fluorophore expression across large whole-organ clearances.

### Low Laser Power / Short Exposure

* Preserves fine cellular boundaries.
* Reduces saturation artifacts within dense structural cores.
* Weak fluorescent structures deep within tissues remain below the detection threshold, dropping true positives.

### High Laser Power / Long Exposure

* Reveals dim cellular structures and weak fluorescent signals.
* Improves visibility across heavily scattered regions.
* Over-saturates bright regions and merges neighboring cells into large, unresolvable pixel blobs that degrade object detection performance.

---

## 2.2 Mathematical Pixel Fusion Framework

To overcome the photometric bottleneck, the HDPR workflow acquires multiple sequential physical images of the identical optical slice at changing laser intensities ($P_1, P_2, \dots, P_n$). The fusion module maps these arrays to recover low-signal structures while suppressing overexposure artifacts.

For a two-power acquisition system ($P_{\text{low}}, P_{\text{high}}$) with linear scaling factor $\alpha = P_{\text{high}} / P_{\text{low}}$, a Gaussian soft weighting function is evaluated over non-saturated regions:

$$W(I) = \exp \left( -\frac{(I - I_{\text{mid}})^2}{2\sigma^2} \right)$$

where $I_{\text{mid}}$ is the midpoint intensity and $\sigma$ represents the weighting width parameter. The unified, single 16-bit fused linear high-dynamic range array frame $I_{\text{fused}}(x,y)$ is computed as:

$$I_{\text{fused}}(x,y) = \begin{cases} 
\alpha \cdot I_{\text{low}}(x,y) & \text{if } I_{\text{high}}(x,y) \ge \tau \\
\frac{W(I_{\text{low}}) \cdot (\alpha \cdot I_{\text{low}}(x,y)) + W(I_{\text{high}}) \cdot I_{\text{high}}(x,y)}{W(I_{\text{low}}) + W(I_{\text{high}})} & \text{if } I_{\text{high}}(x,y) < \tau 
\end{cases}$$

where $\tau$ is the strict saturation threshold (typically between $62000$ and $65200$ for 16-bit CCD cameras). This process smoothly transitions from the non-saturated low-power exposure to the scaled high-power data, outputting a 16-bit array that preserves cellular morphological details across both ultra-dim and highly intense tissue targets.

---

## 2.3 Deep Learning Inference Windowing

The object detection pipeline uses a customized YOLOv8 architecture optimized for multi-class cellular object recognition (Class 0: **Neuron**, Class 1: **Glia**).

Because convolutional neural network native layers process 8-bit inputs ($[0, 255]$), 16-bit volumetric arrays must undergo dynamic range compression before inference. Instead of a naive global normalization that clips fine intensity changes, the pipeline applies an adaptive min-max normalization targeted to a rolling $k$-th percentile window of the non-zero background array:

$$I_{\text{norm}}(x,y) = \min \left( 255, \max \left( 0, \frac{I_{16}(x,y) - P_{\text{min}}}{P_{\text{max}} - P_{\text{min}}} \times 255 \right) \right)$$

where $I_{16}$ is the fused 16-bit image, $P_{\text{min}}$ is the lower percentile bound, and $P_{\text{max}}$ is the upper percentile bound. This adaptive normalization preserves local cellular contrast and fine structural details before tensors are passed into the neural network backbone.

---

## 2.4 Blind Consensus Graph Mapping

To establish an unbiased validation dataset, multiple experts independently annotate the same image slices. The consensus engine constructs a spatial graph where bounding boxes are represented as graph nodes, and edges connect boxes whose overlap exceeds a predefined threshold.

Intersection-over-Union (IoU) is computed as:

$$\text{IoU} = \frac{\text{Area}(B_{\text{Expert } A} \cap B_{\text{Expert } B})}{\text{Area}(B_{\text{Expert } A} \cup B_{\text{Expert } B})}$$

An edge is created in the spatial graph when the computed overlap meets the barrier criteria:

$$\text{IoU} \ge \text{IoU Threshold}$$

Connected components or cliques within this spatial graph represent agreed-upon cellular detections. Class disagreements are resolved using majority voting logic, producing a unified consensus ground-truth dataset (`labels_gt`) that minimizes individual annotator bias.

---

# 3. Automated Multi-Step Pipeline Execution Guide

# Phase I: Preprocessing & Exposure Fusion

## 01_preprocess_clip_scale.py

### Purpose

Performs multi-channel percentile clipping and converts 16-bit microscopy images into 8-bit representations optimized for standard viewing screens.

### Input Structure

```text
/raw_root/{ACQUISITION_MODE}/{CHANNEL}/{X_Folder}/{Tile}/*.tiff

```

### Output Structure

```text
/stretched_root/{ACQUISITION_MODE}/{CHANNEL}/{X_Folder}/{Tile}/*.tiff

```

### Key Parameters

```yaml
clip_percentiles: [0.1, 99.9]

```

### Validation Checkpoints

* Rejects empty or corrupt image files.
* Logs missing metadata entries.

---

## 02_hdpr_pixel_fusion.py

### Purpose

Performs multi-exposure HDPR image fusion by combining low-signal structures and suppressing overexposure artifacts.

### Input Structure

Reads multiple acquisition-power directories concurrently under:

```text
/raw_root/

```

### Output Structure

```text
/raw_hdpr_root/{hdpr_folder}/{CHANNEL}/{X_Folder}/{Tile}/*.tiff

```

### Key Parameters

```yaml
saturation_threshold: 62000-65200

```

### Validation Checkpoints

* Verifies physical stage coordinate alignment between power levels.
* Flags stage shifts greater than 0.5 µm as geometric alignment faults.

---

## 03_hdpr_stitched_fusion.py

### Purpose

Stitches neighboring physical micro-tiles into seamless macroscopic whole-organ reconstructions.

### Method

* Normalized cross-correlation for translational overlap matrix computation.
* Linear alpha blending over physical margins.

### Input Structure

```text
/Tile_00_00/
/Tile_00_01/
/Tile_00_02/
...

```

### Output Structure

Whole-organ stitched macroscopic image volumes.

### Validation Checkpoints

* Monitors stitching boundaries to prevent spatial tearing and misalignment.

---

# Phase II: Deep Learning Inference

## 03_yolo_hdpr_inference.py

### Purpose

Runs multi-track YOLOv8 deep learning batch inference on raw and fused image arrays.

### Input Structure

Supports mixed processing tracks of:

* `RAW` images
* `STRETCHED` images

### Output Structure

```text
/txt_root/{DATASET_TYPE}/{METHOD}/{X_Folder}/{Tile}/{Z_Slice}.txt

```

### Key Parameters

```yaml
confidence_threshold: 0.25
nms_iou_threshold: 0.45

```

### Output Format

Standardized YOLO text strings:

```text
<class_id> <x_center> <y_center> <width> <height>

```

### Validation Checkpoints

Missing YOLO model weights (`.pt`) trigger immediate script termination.

---

# Phase III: Diagnostics & Post-Processing Visualization

## 04_compile_slice_ledger.py

### Purpose

Builds a master database indexer tracking cell counts and cross-method object detection variance.

### Output Structure

Centralized tab-delimited metrics ledger saved to:

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

### Validation Checkpoints

Slices containing fewer than five baseline detections are excluded to filter out tissue dead zones.

---

## 05_generate_grid_figures.py

### Purpose

Generates high-resolution multi-column 2xN publication panels featuring raw image viewports stacked directly over log-scale pixel intensity histograms.

### Output Structure

```text
/{Cohort_Class}/{Z_Slice}_{Dataset_Type}.png

```

### Layout Adjustments

Tight grid styling parameters (`rect=[0.01, 0.01, 0.99, 0.94]`, `h_pad=1.8`, `w_pad=0.6`) eliminate wide margins and row overlapping.

### Validation Checkpoints

Automatically safe-skips missing frames without crashing global loop arrays.

---

## 06_find_saturated_images.py

### Purpose

Parallelized chunk evaluation over raw image folders using an 8x8 matrix sub-sampling stride to isolate highly saturated frames from the highest laser-power acquisition.

### Output Structure

Dedicated verification folders named clearly after their coordinate signatures:

```text
/{Z_Slice}/

```

### Key Parameters

```text
search_pool_size
target_count
max_per_tile

```

---

## 07_generate_saturated_plots.py

### Purpose

Generates layout-harmonized 2xN visual comparison panels for saturated image examples.

### Output Structure

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

* Unbuffered log stream telemetry tracking execution runtime.
* Progress estimation clock outputs (`Remaining: HH:MM:SS`).

---

## 08_generate_specific_slice_grid.py

### Purpose

Generates targeted 2xN comparative visual inspection grids for a manual list of user-specified coordinates.

### Output Structure

```text
/{Z_Slice}_{Dataset_Type}_specific.png

```

### Configuration

Target coordinates are specified directly inside the parameters block of `config.yaml`.

### Validation Checkpoints

Integrated upstream file integrity layer skips missing disk frames cleanly instead of rendering empty plots.

---

# Phase IV: Validation & Performance Metrics Suite

## 09_extract_val_set.py

### Purpose

Creates an unbiased validation cohort subset using a round-robin balancing selection block across physical tiles.

### Output Structure

Self-contained sandbox folders standardizing output filenames to remove coordinate duplication:

```text
/{Z_Slice}/
├── images/
├── labels/
└── labels_gt/

```

### Key Parameters

```yaml
min_detected_cells
total_slices_to_select

```

---

## 10_fuse_gt_labels.py

### Purpose

Creates a unified consensus validation standard from multiple expert label sets.

### Method

* IoU-based spatial graph construction to isolate overlapping bounding boxes.
* Majority-vote class assignment to resolve inter-operator variation.

### Output Structure

```text
/{Z_Slice}/labels_gt/HDPR_Late.txt

```

---

## 11_evaluate_spatial_detection.py

### Purpose

Computes spatial detection performance metrics by comparing model predictions against the consensus ground truth.

### Metrics

* Precision
* Recall
* F1 Score (Micro/Macro averages)

### Outputs

```text
01_00_Spatial_Detection_Report.txt
01_01_Macro_F1_Summary_with_Variance.png
01_02_Combined_CM_Figure.png

```

### Key Parameters

```yaml
iou_threshold: 0.45
dpi: 300

```

### Layout Adjustments

Heatmap text overlays scale to an ultra-clear size 42 bold font with axis tick labels set to size 30.

---

## 12_evaluate_population_counting.py

### Purpose

Computes global population-counting performance metrics and density regression tracking.

### Metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score
* Systematic Model Bias Percentage

### Outputs

```text
02_00_Counting_Regression_Report.txt
02_01_Population_Totals_Summary.csv
02_02_Total_Population_Table.png
02_03_Raw_Counting_Data.csv
02_04_Neuron_Counting_Scatter.png
02_05_Glia_Counting_Scatter.png

```

---

## 13_evaluate_golden_subset.py

### Purpose

Mines the validation sandboxes to isolate and profile the highest-performing "Golden Cohort" slices based on your prioritized statistical parameter.

### Outputs

```text
03_00_Golden_Cohort_Report.txt
03_01_Golden_Combined_CM.png
03_02_Golden_F1_Summary.png

```

### Key Parameters

```yaml
top_k_limit
ranking_class
ranking_metric     # Supported keys: Accuracy, Precision, Recall, F1

```

### Layout Adjustments

Heatmap formatting grid rules are explicitly matched to Step 11 properties (font size 42 bold, tick labels size 30) for visual uniformity in publication.

---

## 4. Centralized Configuration Reference (`config.yaml`)

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

## 5. Critical Infrastructure Safeguards & Best Practices

### Avoid Hardcoded Paths

Never hardcode absolute linux file path pointers inside the individual Python modules. All system directory mappings must be managed through the centralized root `config.yaml` block.

### Python Path Resolution

An empty string configuration assignment:

```python
Path("")

```

resolves to Python's current working directory indicator:

```text
.

```

Always verify path handles before executing down-stream file operations:

```python
if txt_path != Path(""):
    ...

```

This prevents unhandled folder-reading runtime failures:

```text
IsADirectoryError

```

### HPC Logging and Output Buffering

When scaling scripts across massive volumetric image runs inside shared compute cluster nodes, standard terminal outputs can lag due to network stream buffering. Ensure real-time log tracing and accurate progress updates by utilizing an unbuffered stream block:

```python
logging.basicConfig(
    ...,
    stream=sys.stdout
)

```

Combined with explicit log flushes:

```python
sys.stdout.flush()

```

---

# 6. Getting Started

Clone the repository into your cluster workspace environment. Update `config.yaml` with your target image locations, output directories, acquisition channels, laser power settings, and YOLO model paths.

To execute an interactive diagnostic run on a single login node, call the target module directly:

```bash
python3 03_postprocessing/05_generate_grid_figures.py

```

For large-scale pipeline execution via an automated batch scheduler, configure your resource limits inside the HPC shell file:

```text
bsub.sh

```

Then submit the job wrapper to the cluster:

```bash
bsub < bsub.sh

```

---

# Pipeline Summary

This integrated pipeline provides an automated, end-to-end framework for:

* Multi-exposure 16-bit linear HDPR image fusion.
* Adaptive percentile-driven 8-bit dynamic range windowing.
* YOLOv8-based multi-class neuron and glia whole-brain cell mapping.
* Graph-theoretic multi-expert consensus ground-truth label generation.
* Rigorous micro/macro spatial validation and population counting regression benchmarking.
* Robust, unbuffered telemetry logging optimized for large-scale HPC cluster deployments.

```

```
