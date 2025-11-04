# DRAFTS++: Radio Transient Search Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com/)
[![Research](https://img.shields.io/badge/Research-Academic-green.svg)](https://github.com/Kodamonkey/DRAFTS-UC)
[![arXiv](https://img.shields.io/badge/arXiv-2410.03200-b31b1b.svg)](https://arxiv.org/abs/2410.03200)

![DRAFTS WorkFlow](WorkFlow.png)

> **Original repository:** [DRAFTS](https://github.com/SukiYume/DRAFTS) - Deep learning-based RAdio Fast Transient Search pipeline

## Project Overview

**DRAFTS++** is an advanced pipeline for detecting **Fast Radio Bursts (FRBs)** in radio astronomy data using deep learning. It builds upon the original **DRAFTS** (Deep Learning-based RAdio Fast Transient Search) framework, integrating modern neural networks to overcome challenges like radio-frequency interference (RFI) and propagation dispersion that hinder traditional search algorithms. In DRAFTS++, a **deep-learning object detector** (CenterNet-based) localizes burst candidates in dedispersed time–DM space, and a **binary classifier** (ResNet-based) verifies each candidate to distinguish real FRBs from noise/RFI. This two-stage approach greatly improves detection accuracy and reduces false positives compared to classical methods (e.g., PRESTO/Heimdall).

> **What's DRAFTS-UC?**  
> DRAFTS++ (a.k.a. _DRAFTS-UC_) is our maintained fork/extension. It keeps the original DRAFTS ideas and models, adds modern engineering (logging, chunking, GPU/CPU fallbacks), and streamlines configuration for easy, reproducible runs.

---

## Features

- **CUDA-accelerated dedispersion** for near real-time DM sweeps.
- **CenterNet object detection** to infer **arrival time & DM** directly from time–DM "bow-ties".
- **ResNet binary classification** to confirm candidates and **reduce false positives** dramatically.
- **Command-line configuration**: flexible parameter control via CLI arguments with sensible defaults.
- **Chunked processing** of large files with automatic memory-aware slicing.
- **PSRFITS & SIGPROC (.fil)** input support; optional multi-band analysis.
- **Rich outputs**: CSV summaries, annotated plots (waterfalls, DM curves, S/N traces), and logs.
- **Trainable**: scripts to (re)train detection and classification models on your own data.
- **Docker support**: Reproducible environments for CPU and GPU with full documentation.

---

## Quick Start

### Option 1: Docker (Recommended)

Docker provides an isolated and reproducible environment without the need to manually install dependencies.

```bash
# 1) Clone the repository
git clone https://github.com/Kodamonkey/DRAFTS-UC.git
cd DRAFTS-UC

# 2) Verify Docker Desktop is running
docker ps

# 3) Build the image (CPU or GPU)
docker-compose build drafts-cpu    # For systems without GPU
docker-compose build drafts-gpu    # For systems with NVIDIA GPU

# 4) Place .fits/.fil files in Data/raw/

# 5) Run the pipeline
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "FRB20180301_0001"
```

**Build time:** 10-15 minutes first time, then instantaneous (uses cache)

### Option 2: Local Installation

```bash
# 1) Clone the repository
git clone https://github.com/Kodamonkey/DRAFTS-UC.git
cd DRAFTS-UC

# 2) Create and activate virtual environment (Python 3.8+)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Place .fits/.fil files in Data/raw/

# 5) Run the pipeline with required arguments
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" --target "FRB20180301_0001"
```

When finished, inspect `Results/` to view plots and CSV detection summaries.

---

## Prerequisites

### For Local Installation

- **OS:** Linux/macOS recommended (Windows works too).
- **Python:** 3.8+ (use a virtualenv/Conda).
- **GPU:** NVIDIA GPU with CUDA 11+ **recommended** (CPU works but is slow).
- **Drivers/Toolkit:** Matching NVIDIA driver + CUDA toolkit; install a **PyTorch** build that matches your CUDA.
- **RAM/VRAM:** Several GB suggested for large observations (the pipeline chunks automatically).
- **Git** to clone the repository.

Verify PyTorch & CUDA after install:

```bash
python -c "import torch; print(f'PyTorch={torch.__version__} CUDA={torch.cuda.is_available()}')"
```

### For Docker Installation

- **Docker Desktop** installed and running ([Download](https://www.docker.com/get-started))
- **NVIDIA Docker** (for GPU) - Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **10+ GB** free disk space
- Model weights in `src/models/`

Verify Docker:

```bash
docker ps  # Should work without error
```

---

## Repository Layout

```
DRAFTS-UC/
├── Data/
│   ├── raw/             # put your .fits/.fil here
│   └── processed/       # temp/intermediate (generated)
├── Results/             # figures, CSVs, logs (generated)
├── src/
│   ├── config/          # configuration defaults (overridden by CLI args)
│   ├── core/            # pipeline orchestrator
│   ├── input/           # FITS/.fil readers, chunking
│   ├── preprocessing/   # GPU/CPU dedispersion, filters
│   ├── detection/       # model I/O & inference utils
│   ├── models/          # .pth weights + training code
│   │   ├── cent_resnet18.pth      # Detection model
│   │   ├── class_resnet18.pth     # Classification model
│   │   ├── ObjectDet/             # Training code for detector
│   │   └── BinaryClass/           # Training code for classifier
│   ├── analysis/        # S/N, stats
│   ├── visualization/   # plotting & figure export
│   ├── output/          # save candidates, CSVs
│   ├── logging/         # logging utilities
│   └── scripts/         # helper/utility scripts
├── main.py              # entry point (CLI)
├── requirements.txt
├── README.md
├── Dockerfile           # Multi-stage Docker build (CPU/GPU)
└── docker-compose.yml   # Docker Compose orchestration
```

---

## Configuration

The pipeline is configured exclusively via **command-line arguments**. All arguments are passed when executing `main.py`.

### Command-Line Arguments Reference

#### Required Arguments

These arguments are mandatory for every execution:

| Argument        | Type             | Description                                                                                              |
| --------------- | ---------------- | -------------------------------------------------------------------------------------------------------- |
| `--data-dir`    | `str`            | **[REQUIRED]** Directory containing input files (`.fits`, `.fil`). Can be relative or absolute path.     |
| `--results-dir` | `str`            | **[REQUIRED]** Directory where all results will be saved (CSVs, plots, logs). Created if doesn't exist.  |
| `--target`      | `str` (multiple) | **[REQUIRED]** Pattern(s) to search for files. Can specify multiple patterns. Supports partial matching. |

**Example:**

```bash
python main.py \
  --data-dir "./Data/raw/" \
  --results-dir "./Results/" \
  --target "FRB20180301_0001"
```

---

#### Temporal Analysis Parameters

| Argument           | Type    | Default | Description                                                                               |
| ------------------ | ------- | ------- | ----------------------------------------------------------------------------------------- |
| `--slice-duration` | `float` | `300.0` | Duration of each temporal window in milliseconds. Controls the size of analysis segments. |

**Example:**

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "FRB20180301" --slice-duration 500.0
```

---

#### Dispersion Measure (DM) Configuration

| Argument   | Type  | Default | Description                                                            |
| ---------- | ----- | ------- | ---------------------------------------------------------------------- |
| `--dm-min` | `int` | `0`     | Minimum dispersion measure in pc cm⁻³. Lower bound of DM search range. |
| `--dm-max` | `int` | `1024`  | Maximum dispersion measure in pc cm⁻³. Upper bound of DM search range. |

**Example:**

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "FRB20180301" --dm-min 100 --dm-max 600
```

---

#### Detection and Classification Thresholds

| Argument       | Type    | Default | Description                                                                                                                              |
| -------------- | ------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `--det-prob`   | `float` | `0.3`   | Minimum probability (0.0-1.0) for CenterNet detection to be considered valid. Lower values = more detections (but more false positives). |
| `--class-prob` | `float` | `0.5`   | Minimum probability (0.0-1.0) for ResNet classifier to label as BURST. Higher values = more conservative classification.                 |
| `--snr-thresh` | `float` | `5.0`   | Signal-to-noise ratio threshold used for highlighting candidates in visualizations. Does not affect detection.                           |

**Example:**

```bash
# More conservative detection (fewer false positives)
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "FRB20180301" --det-prob 0.5 --class-prob 0.7

# More inclusive detection (more candidates)
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "FRB20180301" --det-prob 0.2 --class-prob 0.4
```

---

#### Downsampling Configuration

| Argument           | Type  | Default | Description                                                                             |
| ------------------ | ----- | ------- | --------------------------------------------------------------------------------------- |
| `--down-freq-rate` | `int` | `1`     | Frequency reduction factor. `1` = no reduction, `2` = half the frequency channels, etc. |
| `--down-time-rate` | `int` | `8`     | Time reduction factor. `1` = no reduction, `8` = 1/8th temporal resolution, etc.        |

**Example:**

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "FRB20180301" --down-freq-rate 2 --down-time-rate 4
```

---

#### Multi-Band Analysis

| Argument       | Type   | Default | Description                                                                                                                                                              |
| -------------- | ------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--multi-band` | `flag` | `False` | Enable multi-band analysis. Processes data in three bands: Full Band, Low Band, and High Band. Increases processing time but provides better frequency characterization. |

**Example:**

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "FRB20180301" --multi-band
```

---

#### High-Frequency Pipeline Configuration

| Argument                | Type    | Default  | Description                                                                                                              |
| ----------------------- | ------- | -------- | ------------------------------------------------------------------------------------------------------------------------ |
| `--auto-high-freq`      | `flag`  | `True`   | Automatically activates high-frequency processing pipeline when central frequency exceeds threshold. Enabled by default. |
| `--no-auto-high-freq`   | `flag`  | —        | Explicitly disable automatic high-frequency pipeline.                                                                    |
| `--high-freq-threshold` | `float` | `8000.0` | Central frequency threshold in MHz to trigger high-frequency mode. Default is 8 GHz (e.g., for ALMA observations).       |

**Example:**

```bash
# Process ALMA data (typical freq > 8 GHz)
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "ALMA_obs" --auto-high-freq --high-freq-threshold 7500.0

# Disable high-freq mode explicitly
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "FRB20180301" --no-auto-high-freq
```

---

#### Polarization Configuration (PSRFITS only)

| Argument               | Type  | Default     | Options                                                           | Description                                                                |
| ---------------------- | ----- | ----------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `--polarization-mode`  | `str` | `intensity` | `intensity`, `linear`, `circular`, `pol0`, `pol1`, `pol2`, `pol3` | Polarization processing mode for PSRFITS files with IQUV data.             |
| `--polarization-index` | `int` | `0`         | `0-3`                                                             | Default polarization index when IQUV is not available (e.g., AABB format). |

**Polarization Modes:**

- `intensity`: Stokes I (total intensity)
- `linear`: √(Q² + U²) (linear polarization)
- `circular`: |V| (circular polarization)
- `pol0`, `pol1`, `pol2`, `pol3`: Select specific polarization index directly

**Example:**

```bash
# Use linear polarization
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "pulsar_obs" --polarization-mode linear

# Use second polarization channel
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "pulsar_obs" --polarization-mode pol1
```

---

#### Visualization and Debug Parameters

| Argument            | Type   | Default | Description                                                                                                   |
| ------------------- | ------ | ------- | ------------------------------------------------------------------------------------------------------------- |
| `--force-plots`     | `flag` | `False` | Always generate plots, even when no candidates are detected. Useful for debugging data issues.                |
| `--no-force-plots`  | `flag` | —       | Explicitly disable forced plot generation (default behavior).                                                 |
| `--debug-frequency` | `flag` | `False` | Show detailed frequency ordering and file information during processing. Useful for diagnosing data problems. |

**Example:**

```bash
# Debug mode with forced plots
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "test_file" --force-plots --debug-frequency
```

---

#### Candidate Filtering and Output

| Argument            | Type   | Default | Description                                                                          |
| ------------------- | ------ | ------- | ------------------------------------------------------------------------------------ |
| `--save-only-burst` | `flag` | `True`  | Save only candidates classified as BURST by the ResNet classifier. Default behavior. |
| `--save-all`        | `flag` | —       | Save all detected candidates, regardless of classification (BURST and non-BURST).    |

**Example:**

```bash
# Save all candidates for inspection
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "FRB20180301" --save-all

# Explicitly save only BURST candidates (default)
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "FRB20180301" --save-only-burst
```

---

### Getting Help

To see all available arguments with descriptions:

```bash
python main.py --help
```

### Complete Example with All Parameters

```bash
python main.py \
  --data-dir "./Data/raw/" \
  --results-dir "./Results/" \
  --target "FRB20180301_0001" \
  --slice-duration 400.0 \
  --dm-min 100 \
  --dm-max 600 \
  --det-prob 0.4 \
  --class-prob 0.6 \
  --snr-thresh 6.0 \
  --down-freq-rate 1 \
  --down-time-rate 8 \
  --multi-band \
  --auto-high-freq \
  --high-freq-threshold 8000.0 \
  --polarization-mode intensity \
  --polarization-index 0 \
  --save-only-burst \
  --debug-frequency
```

### Configuration Priority

1. **Command-line arguments** (highest priority) - Override all defaults
2. **`src/config/user_config.py`** (lowest priority) - Only used if CLI args not provided

> **Important:** The three required arguments (`--data-dir`, `--results-dir`, `--target`) must always be provided via CLI. The pipeline will not run without them.

---

## Running the Pipeline

### Execution Methods

The pipeline can be run in two ways:

1. **Docker (Recommended)** - Isolated, reproducible environment
2. **Local Python** - Direct execution on your system

---

### Method 1: Running with Docker (Recommended)

#### Basic Docker Execution

**From outside the container (host machine):**

```bash
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "FRB20180301_0001"
```

**From inside the container (interactive shell):**

```bash
# First, enter the container
docker-compose run --rm --entrypoint /bin/bash drafts-gpu

# Now you're inside the container at /app
# Run the pipeline:
python main.py \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "FRB20180301_0001"
```

#### Docker Usage Examples

**Process with custom thresholds (inside container):**

```bash
# Enter container
docker-compose run --rm --entrypoint /bin/bash drafts-gpu

# Execute from inside
python main.py \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "2017-04-03-08_55_22" \
  --det-prob 0.5 \
  --class-prob 0.6
```

**Multi-band analysis (from host):**

```bash
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "FRB20201124" \
  --multi-band \
  --slice-duration 3000.0
```

**Process multiple files (inside container):**

```bash
# Enter container
docker-compose run --rm --entrypoint /bin/bash drafts-gpu

# List available files
ls /app/Data/raw/

# Process multiple patterns
python main.py \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "2017-04-03" "FRB" "B0355"
```

**Custom DM range (from host):**

```bash
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "FRB20180301" \
  --dm-min 100 \
  --dm-max 600
```

**View help (inside container):**

```bash
docker-compose run --rm --entrypoint /bin/bash drafts-gpu
python main.py --help
```

---

### Method 2: Running Locally (Without Docker)

#### Basic Local Execution

**From repository root:**

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" --target "filename"
```

#### Local Usage Examples

**Simple processing with default values:**

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" --target "FRB20180301_0001"
```

**With custom thresholds:**

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "2017-04-03-08_55_22" --det-prob 0.5 --class-prob 0.6
```

**Enable multi-band analysis:**

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "FRB20201124" --multi-band --slice-duration 3000.0
```

**Process multiple files:**

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "2017-04-03" "FRB" "B0355"
```

**Configure custom DM range:**

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "FRB20180301" --dm-min 100 --dm-max 512
```

**View all available parameters:**

```bash
python main.py --help
```

---

## Advanced Usage Examples

This section provides practical examples for common use cases, with both Docker and local execution methods.

### Working Inside Docker Container (Interactive Mode)

The most flexible way to work with Docker is using an interactive shell:

```bash
# Enter the container
docker-compose run --rm --entrypoint /bin/bash drafts-gpu

# Now you're inside at /app directory
# You can explore and run multiple commands:

# List available data files
ls -lh /app/Data/raw/

# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Run pipeline with any parameters
python main.py \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "FRB20180301_0001"

# Check results
ls -lh /app/Results/

# Exit when done
exit
```

### Common Use Cases

#### 1. Batch Processing Multiple Files

**From host (docker-compose run):**

```bash
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "FRB20180301" "FRB20201124" "2017-04-03"
```

**Inside container:**

```bash
# Enter container
docker-compose run --rm --entrypoint /bin/bash drafts-gpu

# Process all FRB files at once
python main.py \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "FRB20180301" "FRB20201124" "2017-04-03"
```

#### 2. Conservative Detection (High Confidence)

**From host (docker-compose run):**

```bash
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "FRB20180301_0001" \
  --det-prob 0.6 \
  --class-prob 0.8 \
  --snr-thresh 7.0
```

**Inside container:**

```bash
# First enter container
docker-compose run --rm --entrypoint /bin/bash drafts-gpu

# Then run
python main.py \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "FRB20180301_0001" \
  --det-prob 0.6 \
  --class-prob 0.8 \
  --snr-thresh 7.0
```

#### 3. Sensitive Detection (More Candidates)

**From host:**

```bash
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "FRB20180301_0001" \
  --det-prob 0.2 \
  --class-prob 0.3 \
  --save-all
```

**Inside container:**

```bash
python main.py \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "FRB20180301_0001" \
  --det-prob 0.2 \
  --class-prob 0.3 \
  --save-all
```

#### 4. Multi-Band Analysis with Custom Temporal Window

**From host:**

```bash
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "FRB20201124" \
  --multi-band \
  --slice-duration 3000.0 \
  --dm-min 200 \
  --dm-max 800
```

**Inside container:**

```bash
python main.py \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "FRB20201124" \
  --multi-band \
  --slice-duration 3000.0 \
  --dm-min 200 \
  --dm-max 800
```

#### 5. High-Frequency Observations (ALMA/mm-wave)

**From host:**

```bash
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "ALMA_obs" \
  --auto-high-freq \
  --high-freq-threshold 7500.0 \
  --down-freq-rate 2
```

**Inside container:**

```bash
python main.py \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "ALMA_obs" \
  --auto-high-freq \
  --high-freq-threshold 7500.0 \
  --down-freq-rate 2
```

#### 6. Filterbank Files Processing

**From host:**

```bash
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "3096_0001_00_8bit" "3097_0001_00_8bit" \
  --dm-min 0 \
  --dm-max 512
```

**Inside container:**

```bash
python main.py \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "3096_0001_00_8bit" "3097_0001_00_8bit" \
  --dm-min 0 \
  --dm-max 512
```

#### 7. Debug Mode with Forced Plots

**From host:**

```bash
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "test_file" \
  --force-plots \
  --debug-frequency \
  --save-all
```

**Inside container:**

```bash
python main.py \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "test_file" \
  --force-plots \
  --debug-frequency \
  --save-all
```

#### 8. Pulsar Observations with Polarization

**From host:**

```bash
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "B0355" \
  --polarization-mode linear \
  --dm-min 50 \
  --dm-max 100
```

**Inside container:**

```bash
python main.py \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "B0355" \
  --polarization-mode linear \
  --dm-min 50 \
  --dm-max 100
```

---

## Pipeline Execution Flow

For each file matching the pattern specified in `--target`:

1. **Load & chunk**: Efficiently loads the dynamic spectrum (.fits/.fil) into memory
2. **Dedisperse**: Dedisperses over the range `[DM_min, DM_max]` (uses GPU if available)
3. **Detect candidates**: Detects candidates in time–DM using CenterNet → boxes + scores
4. **Classify**: Classifies each candidate with ResNet → FRB probability vs non-FRB
5. **Save outputs**: Saves annotated figures, CSV per file, and logs in the results directory

> **Tip:** If thresholds are too strict, you will see fewer detections but with higher confidence. Relax `--det-prob` or `--class-prob` to be more inclusive.

---

## Additional Examples (Local Execution)

### Batch Processing Multiple Sessions

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "session1_" "session2_" "2024-10-05"
```

### High Frequency with Custom Threshold

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "high_freq_obs" --auto-high-freq --high-freq-threshold 7500.0
```

### Ad-hoc Plotting (Utility Script)

Generate plots for specific time segments without running full pipeline:

**Local:**

```bash
python src/scripts/absolute_segment_plots.py \
  --filename FRB20180301_0001.fits --start 10.0 --duration 5.0 --dm 565
```

**Inside Docker:**

```bash
docker-compose run --rm --entrypoint /bin/bash drafts-gpu
python /app/src/scripts/absolute_segment_plots.py \
  --filename /app/Data/raw/FRB20180301_0001.fits --start 10.0 --duration 5.0 --dm 565
```

### Integration into Python Applications

For programmatic use, you can import and run the pipeline directly:

```python
from pathlib import Path
from src.core.pipeline import run_pipeline

# Custom configuration dictionary
config_dict = {
    "DATA_DIR": Path("./Data/raw/"),
    "RESULTS_DIR": Path("./Results/"),
    "FRB_TARGETS": ["FRB20180301_0001"],
    "DM_min": 100,
    "DM_max": 600,
    "DET_PROB": 0.4,
    "CLASS_PROB": 0.6,
    "USE_MULTI_BAND": False,
    "AUTO_HIGH_FREQ_PIPELINE": True,
}

# Execute pipeline
run_pipeline(config_dict=config_dict)
```

---

## Docker Deployment Guide

### Docker Execution Modes

The pipeline can be executed via Docker in two ways:

#### Direct Execution (Single Command)

Execute the pipeline directly from the host system:

```bash
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "FRB20180301_0001"
```

**Use when:**

- Processing a single file with known parameters
- Running automated scripts or batch jobs
- Quick one-time executions

**Behavior:** Container starts, executes pipeline, terminates automatically.

---

#### Interactive Shell (Multiple Commands)

Enter the container environment for iterative work:

```bash
# Enter container shell
docker-compose run --rm --entrypoint /bin/bash drafts-gpu

# Execute commands inside container
ls /app/Data/raw/
python main.py --help
python main.py --data-dir /app/Data/raw --results-dir /app/Results --target "file1"
python main.py --data-dir /app/Data/raw --results-dir /app/Results --target "file2"
exit
```

**Use when:**

- Processing multiple files sequentially
- Exploring data directory contents
- Testing different parameter configurations
- Debugging or development work

**Behavior:** Container remains active until explicit exit, allowing multiple command executions.

---

### Execution Mode Comparison

| Feature             | Direct Execution                            | Interactive Shell                                           |
| ------------------- | ------------------------------------------- | ----------------------------------------------------------- |
| Command             | `docker-compose run --rm drafts-gpu [ARGS]` | `docker-compose run --rm --entrypoint /bin/bash drafts-gpu` |
| Session Persistence | No                                          | Yes                                                         |
| Multiple Commands   | No                                          | Yes                                                         |
| Python Invocation   | Arguments passed to docker-compose          | `python main.py` called directly                            |
| Optimal For         | Single runs, automation                     | Exploration, debugging, batch processing                    |

---

### Docker Command Reference

```bash
# Build image (required once)
docker-compose build drafts-gpu

# Direct execution
docker-compose run --rm drafts-gpu [PIPELINE_ARGS]

# Interactive shell
docker-compose run --rm --entrypoint /bin/bash drafts-gpu
```

### Prerequisites

Before running Docker commands:

1. Docker Desktop must be running (verify: `docker ps`)
2. Model weights must exist in `src/models/`:
   - `cent_resnet18.pth` (detection model)
   - `class_resnet18.pth` (classification model)
3. Data files must be placed in `Data/raw/`

### Docker Setup and Execution

#### Step 1: Build Image

Build the Docker image (required once, approximately 10-15 minutes):

```bash
# GPU-enabled (requires NVIDIA GPU and drivers)
docker-compose build drafts-gpu

# CPU-only (no GPU required)
docker-compose build drafts-cpu
```

Subsequent builds utilize layer caching and complete in approximately 30 seconds.

#### Step 2: Verify Configuration

Verify GPU availability:

```bash
docker-compose run --rm --entrypoint python drafts-gpu \
  -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output (GPU systems):

```
PyTorch: 2.x.x
CUDA: True
```

#### Step 3: Execute Pipeline

**Direct execution (single command):**

```bash
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "FRB20180301_0001"
```

**Interactive shell (multiple commands):**

```bash
# Enter container
docker-compose run --rm --entrypoint /bin/bash drafts-gpu

# Execute commands
ls /app/Data/raw/
python main.py --help
python main.py --data-dir /app/Data/raw --results-dir /app/Results --target "FRB20180301_0001"
exit
```

**Directory mounting:**

- Host `./Data/raw/` maps to `/app/Data/raw/`
- Host `./Results/` maps to `/app/Results/`
- Host `./src/models/` maps to `/app/src/models/`

#### View Available Arguments

```bash
# From host
docker-compose run --rm drafts-gpu --help

# From container shell
docker-compose run --rm --entrypoint /bin/bash drafts-gpu
python main.py --help
```

### Docker Usage Examples

**Process specific file with custom parameters:**

```bash
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "2017-04-03-08_55_22" \
  --det-prob 0.5 \
  --class-prob 0.6 \
  --dm-min 100 \
  --dm-max 600
```

**Process multiple files matching a pattern:**

```bash
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "2017-04-03" "FRB" "B0355"
```

**Multi-band analysis with extended temporal window:**

```bash
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "FRB20201124" \
  --multi-band \
  --slice-duration 3000.0
```

**Process filterbank files (.fil):**

```bash
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "3096_0001_00_8bit"
```

**High-frequency observations (ALMA data):**

```bash
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "ALMA_observation" \
  --auto-high-freq \
  --high-freq-threshold 7500.0
```

**Debug mode with forced plot generation:**

```bash
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "test_file" \
  --force-plots \
  --debug-frequency
```

**Save all candidates (not just classified as BURST):**

```bash
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "FRB20180301" \
  --save-all
```

**Check GPU availability:**

```bash
docker-compose run --rm --entrypoint python drafts-gpu \
  -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**List files in data directory:**

```bash
docker-compose run --rm --entrypoint ls drafts-gpu /app/Data/raw/
```

**View results directory:**

```bash
docker-compose run --rm --entrypoint ls drafts-gpu -lh /app/Results/
```

### Advanced Docker Usage

**Interactive shell for debugging:**

```bash
# GPU container
docker-compose run --rm --entrypoint /bin/bash drafts-gpu

# Once inside, you can:
cd /app
python main.py --help
ls Data/raw/
```

**View all available CLI arguments:**

```bash
docker-compose run --rm drafts-gpu --help
```

**Check container disk usage:**

```bash
docker system df
```

**View running containers:**

```bash
docker ps
```

**Stop all running containers:**

```bash
docker stop $(docker ps -aq)
```

**Complete cleanup (remove everything):**

```bash
# Stop all containers and remove images/volumes
docker-compose down --rmi all --volumes

# Or use the aggressive cleanup command
docker system prune -a --volumes -f
```

### Docker Volume Mounts

Docker automatically mounts these directories:

- **`./Data/raw/`** → `/app/Data/raw/` (input data files)
- **`./Results/`** → `/app/Results/` (pipeline outputs)
- **`./src/models/`** → `/app/src/models/` (model weights)

Files you place in these local directories are immediately accessible inside the container.

### Troubleshooting Docker Issues

**"Cannot connect to Docker daemon":**

```bash
# Make sure Docker Desktop is running
# On Windows: Check system tray for Docker icon
# On Linux: sudo systemctl start docker
```

**"Models not found" error:**

```bash
# Verify models exist locally
ls -lh src/models/*.pth

# Expected output:
# cent_resnet18.pth
# class_resnet18.pth
```

**"Permission denied" on Results directory (Windows):**

```bash
icacls Results /grant Everyone:F /T
```

**"CUDA out of memory":**

```bash
# Use CPU version instead
docker-compose run --rm drafts-cpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "your-file"
```

**Rebuild image from scratch (if issues persist):**

```bash
docker-compose build --no-cache drafts-gpu
```

---

## Data Requirements

- **Inputs:** single-beam **PSRFITS (.fits)** or **SIGPROC filterbank (.fil)** containing time–frequency power.
- **Headers:** should include frequency axis, sample time, and shape (number of channels, number of samples).
- **Placement:** files go under `Data/raw/` (or your custom `DATA_DIR`).
- **Size:** large files are fine—processing is chunked automatically. Ensure free disk for `Results/` and temporary intermediates under `Data/processed/`.

---

## Model Weights

Place pre-trained weights under `src/models/` with these names:

- **Detection (CenterNet):** `cent_resnet18.pth` or `cent_resnet50.pth`
- **Classification (ResNet):** `class_resnet18.pth` or `class_resnet50.pth`

The pipeline configuration in `src/config/config.py` defaults to `resnet18` for balance of speed/accuracy. To use `resnet50`, edit the `MODEL_NAME` and `CLASS_MODEL_NAME` variables in config.

> **Custom models:** If you train your own, copy your `best_model.pth` to `src/models/` and rename accordingly or update the model names in config.

---

## Model Architecture

The pipeline uses two pre-trained deep learning models:

### Detection Model (CenterNet)

- **Architecture:** ResNet18-based CenterNet
- **File:** `src/models/cent_resnet18.pth`
- **Purpose:** Localizes burst candidates in time-DM space
- **Implementation:** `src/models/ObjectDet/centernet_model.py`

### Classification Model (ResNet)

- **Architecture:** ResNet18 binary classifier
- **File:** `src/models/class_resnet18.pth`
- **Purpose:** Distinguishes real FRBs from RFI/noise
- **Implementation:** `src/models/BinaryClass/binary_model.py`

> **Note:** Training scripts for these models are not currently included in this repository. The pre-trained weights must be obtained separately and placed in `src/models/`.

---

## Outputs

- **Per-file CSV** with all candidates that pass thresholds (arrival time, DM, scores, S/N).
- **Figures**: annotated waterfalls, time–DM "bow-ties", and S/N/DM curves.
- **Logs**: detailed progress and timing (helpful for profiling and debugging).

---

## Tips & Troubleshooting

### General Issues

- **No detections?** Reduce `--det-prob` or `--class-prob`, expand the DM range, or verify data quality and polarization mode.
- **Too many false positives?** Increase thresholds, enable `--multi-band`, or restrict the DM range.
- **Slow execution on CPU?** Install PyTorch with CUDA support and appropriate NVIDIA drivers to enable GPU.
- **Inverted frequency axis?** Use `--debug-frequency` and verify the reader output.
- **High-frequency data (ALMA/mm-wave):** Keep `--auto-high-freq` enabled (default) for automatic parameter adaptation.
- **Error "required arguments"?** Remember that `--data-dir`, `--results-dir` and `--target` are required.
- **View complete help?** Run `python main.py --help` to see all available parameters.

### Docker-Specific Issues

- **"Docker daemon not running"?** Open Docker Desktop and wait for it to fully initialize.
- **"Models not found"?** Verify that `src/models/cent_resnet18.pth` and `src/models/class_resnet18.pth` exist.
- **Build very slow?** Normal first time (10-15 min). Subsequent builds use cache (~30 sec).
- **"Permission denied" when writing results?** On Windows: `icacls Results /grant Everyone:F /T`
- **"CUDA out of memory"?** Use CPU version: `docker-compose run --rm drafts-cpu ...`

---

## Citation & Acknowledgements

If you use DRAFTS++ in research, please cite the original DRAFTS paper and this repository fork:

```bibtex
@article{zhang2024drafts,
  title={DRAFTS: A Deep Learning-Based Radio Fast Transient Search Pipeline},
  author={Zhang, Y.-K. and others},
  journal={arXiv preprint arXiv:2410.03200},
  year={2024}
}
```

- **Zhang, Y.-K., et al. (2024)**, _DRAFTS: A Deep Learning-Based Radio Fast Transient Search Pipeline_ ([arXiv:2410.03200](https://arxiv.org/abs/2410.03200)).
- **DRAFTS-UC / DRAFTS++**: this repository and documentation.

---

## Author

**Sebastian Salgado Polanco**

---

## License

This project is available for academic and research purposes. Please cite the original DRAFTS paper when using this software in your research.
