# DRAFTS++: Radio Transient Search Pipeline 

![DRAFTS WorkFlow](WorkFlow.png)

> **Original repository:** [DRAFTS](https://github.com/SukiYume/DRAFTS) - Deep learning-based RAdio Fast Transient Search pipeline

## Project Overview

**DRAFTS++** is an advanced pipeline for detecting **Fast Radio Bursts (FRBs)** in radio astronomy data using deep learning. It builds upon the original **DRAFTS** (Deep Learning‑based RAdio Fast Transient Search) framework, integrating modern neural networks to overcome challenges like radio‑frequency interference (RFI) and propagation dispersion that hinder traditional search algorithms. In DRAFTS++, a **deep‑learning object detector** (CenterNet‑based) localizes burst candidates in dedispersed time–DM space, and a **binary classifier** (ResNet‑based) verifies each candidate to distinguish real FRBs from noise/RFI. This two‑stage approach greatly improves detection accuracy and reduces false positives compared to classical methods (e.g., PRESTO/Heimdall).

> **What’s DRAFTS‑UC?**  
> DRAFTS++ (a.k.a. _DRAFTS‑UC_) is our maintained fork/extension. It keeps the original DRAFTS ideas and models, adds modern engineering (logging, chunking, GPU/CPU fallbacks), and streamlines configuration for easy, reproducible runs.

---

## Features

- **CUDA‑accelerated dedispersion** for near real‑time DM sweeps.
- **CenterNet object detection** to infer **arrival time & DM** directly from time–DM “bow‑ties”.
- **ResNet binary classification** to confirm candidates and **reduce false positives** dramatically.
- **Single‑config operation**: one place to set data paths, DM range, thresholds, and options.
- **Chunked processing** of large files with automatic memory‑aware slicing.
- **PSRFITS & SIGPROC (.fil)** input support; optional multi‑band analysis.
- **Rich outputs**: CSV summaries, annotated plots (waterfalls, DM curves, S/N traces), and logs.
- **Trainable**: scripts to (re)train detection and classification models on your own data.

---

## Quick Start

```bash
# 1) Clone and enter the repo
git clone https://github.com/Kodamonkey/DRAFTS-UC.git
cd DRAFTS-UC

# 2) Create & activate a virtual environment (Python 3.8+)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Put your .fits/.fil files under Data/raw/
# 5) Edit src/config/user_config.py (DATA_DIR, RESULTS_DIR, FRB_TARGETS, DM range, thresholds)
# 6) Run the full pipeline
python main.py
```

When it finishes, inspect `Results/` for plots and a CSV of detections.

---

## Prerequisites

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

---

## Repository Layout 

```
DRAFTS-UC/
├── Data/
│   ├── raw/             # put your .fits/.fil here
│   └── processed/       # temp/intermediate (generated)
├── Results/             # figures, CSVs, logs (generated)
├── models/              # .pth weights (detector/classifier)
│   ├── cent_resnet18.pth
│   ├── cent_resnet50.pth
│   ├── class_resnet18.pth
│   └── class_resnet50.pth
├── src/
│   ├── config/          # **start here** (user_config.py)
│   ├── core/            # pipeline orchestrator
│   ├── input/           # FITS/.fil readers, chunking
│   ├── preprocessing/   # GPU/CPU dedispersion, filters
│   ├── detection/       # model I/O & inference utils
│   ├── models/          # training code (ObjectDet/BinaryClass)
│   ├── analysis/        # S/N, stats
│   ├── visualization/   # plotting & figure export
│   ├── output/          # save candidates, CSVs
│   └── scripts/         # helper/utility scripts
├── main.py              # entry point
├── requirements.txt
└── README.md
```

---

## Configuration (single place)

All user‑facing settings live in **`src/config/user_config.py`**. Open it and set:

- **Paths**
  - `DATA_DIR`: where your `.fits/.fil` live (default `./Data/raw`)
  - `RESULTS_DIR`: where results will be written (default `./Results`)
- **Targets**
  - `FRB_TARGETS = ["pattern1", "pattern2"]`: substrings to match filenames in `DATA_DIR`.
    - Example: if you have `FRB121102_0001.fits`, set `FRB_TARGETS = ["FRB121102_0001"]`.
- **Time slicing**
  - `SLICE_DURATION_MS`: size of each time window (ms) analyzed at once.
- **DM search**
  - `DM_min`, `DM_max`: dispersion measure range for dedispersion.
- **Decision thresholds**
  - `DET_PROB`: min CenterNet confidence to accept a candidate (e.g., 0.30).
  - `CLASS_PROB`: min ResNet probability to call it a true FRB (e.g., 0.50).
  - `SNR_THRESH`: for highlighting in plots (visual only).
- **Optional knobs**
  - `DOWN_FREQ_RATE`, `DOWN_TIME_RATE` (downsampling for speed),
  - `USE_MULTI_BAND` (split band into full/low/high),
  - `POLARIZATION_MODE` (total intensity, linear, circular, or specific pol index),
  - `AUTO_HIGH_FREQ_PIPELINE` and `HIGH_FREQ_THRESHOLD_MHZ` (auto tuning at mm‑wave),
  - `FORCE_PLOTS`, `DEBUG_FREQUENCY_ORDER`, logging verbosity, etc.

> **Tip:** Defaults are sensible. Start with paths + `FRB_TARGETS`, then run. Tighten `DM_max` to speed up if you know the DM.

---

## Running the Pipeline

From the repo root:

```bash
python main.py
```

For each matching file in `DATA_DIR`:

1. **Load & chunk** the dynamic spectrum (.fits/.fil), memory‑aware.
2. **Dedisperse** across `[DM_min, DM_max]` (GPU if available).
3. **Detect candidates** in time–DM (CenterNet) → boxes + scores.
4. **Classify** each candidate (ResNet) → FRB vs non‑FRB probability.
5. **Save outputs**: annotated figures, per‑file CSV, logs in `Results/`.

If thresholds are strict, you may see fewer but higher‑confidence detections. Relax `DET_PROB`/`CLASS_PROB` to be more inclusive.

---

## Usage Examples

**Single file (simple):**

```python
# src/config/user_config.py
DATA_DIR = Path("./Data/raw")
RESULTS_DIR = Path("./Results")
FRB_TARGETS = ["FRB121102_0001"]  # will match *FRB121102_0001*.fits/.fil
# keep other defaults
```

```bash
python main.py
```

**Batch multiple observations:**

```python
FRB_TARGETS = ["session1_", "session2_", "2024-10-05"]
```

```bash
python main.py
```

**Ad‑hoc plotting (debug/inspection):**

```bash
python src/scripts/absolute_segment_plots.py   --filename FRB121102_0001.fits --start 10.0 --duration 5.0 --dm 565
```

**Embed in your own Python app:**

```python
from src.core.pipeline import run_pipeline
# (Optionally edit config programmatically)
run_pipeline()
```

---

## Data Requirements

- **Inputs:** single‑beam **PSRFITS (.fits)** or **SIGPROC filterbank (.fil)** containing time–frequency power.
- **Headers:** should include frequency axis, sample time, and shape (#chans, #samples).
- **Placement:** files go under `Data/raw/` (or your custom `DATA_DIR`).
- **Size:** large files are fine—processing is chunked automatically. Ensure free disk for `Results/` and temporary intermediates under `Data/processed/`.

---

## Model Weights

Place pre‑trained weights under `./models/` with these names (or adjust in config):

- **Detection (CenterNet):** `cent_resnet18.pth`, `cent_resnet50.pth`
- **Classification (ResNet):** `class_resnet18.pth`, `class_resnet50.pth`

By default, the pipeline loads the `resnet50` variants for best accuracy. Switch to `resnet18` for speed (edit the model names in config if applicable).

> **Custom models:** If you train your own, copy your `best_model.pth` here and rename accordingly or point the config to the new filenames.

---

## Training (optional)

### Object detection (CenterNet)

Training code lives in `src/models/ObjectDet/`.

```bash
cd src/models/ObjectDet/
python centernet_train.py resnet18    # or: resnet50
# outputs logs_resnet18/ (checkpoints incl. best_model.pth)
```

**Data:** 2D time–DM arrays (e.g., 512×512) + labels (boxes for each burst). A `data_label.txt`/CSV listing files and boxes is expected by the script (adapt paths as needed).

### Binary classification (ResNet)

Training code lives in `src/models/BinaryClass/`.

```bash
cd src/models/BinaryClass/
python binary_train.py resnet18 BinaryNet   # or: resnet50
# outputs logs_resnet18/ (checkpoints incl. best_model.pth)
```

**Data:** burst cutouts vs non‑bursts (two folders or an index file).

After training, place your `best_model.pth` under `models/` with the expected name (or update config) and re‑run the pipeline.

---

## Outputs

- **Per‑file CSV** with all candidates that pass thresholds (arrival time, DM, scores, S/N).
- **Figures**: annotated waterfalls, time–DM “bow‑ties”, and S/N/DM curves.
- **Logs**: detailed progress and timing (helpful for profiling and debugging).

---

## Tips & Troubleshooting

- **No detections?** Lower `DET_PROB`/`CLASS_PROB`, widen DM range, or verify data quality/polarization mode.
- **Too many false positives?** Raise thresholds, enable multi‑band, or restrict DM range.
- **Slow run on CPU?** Install CUDA‑enabled PyTorch and a proper NVIDIA driver; enable GPU.
- **Frequency axis inverted?** Set `DEBUG_FREQUENCY_ORDER = True` and check reader output.
- **mm‑wave data (ALMA/hi‑freq):** Keep `AUTO_HIGH_FREQ_PIPELINE=True` so defaults adapt.

---

## Citation & Acknowledgements

If you use DRAFTS++ in research, please cite the original DRAFTS paper and this repository fork:

- **Zhang, Y.‑K., et al. (2024)**, _DRAFTS: A Deep Learning‑Based Radio Fast Transient Search Pipeline_ (arXiv:2410.03200).
- **DRAFTS‑UC / DRAFTS++**: this repository and documentation.

**Made by Sebastian Salgado Polanco**
