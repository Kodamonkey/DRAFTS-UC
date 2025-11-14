# DRAFTS++: Deep Learning Radio Transient Search

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/Kodamonkey/DRAFTS-UC/releases)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2410.03200-b31b1b.svg)](https://arxiv.org/abs/2410.03200)

![DRAFTS WorkFlow](WorkFlow.png)

## What is DRAFTS++?

**DRAFTS++** is a deep learning pipeline for detecting Fast Radio Bursts (FRBs) in radio astronomy data. It uses a two-stage approach:

1. **CenterNet (Detection)** - Localizes burst candidates in time-DM space from dedispersed dynamic spectra
2. **ResNet (Classification)** - Distinguishes real FRBs from RFI and noise

The pipeline processes `.fits` or `.fil` files and outputs candidate detections with their properties (DM, time, SNR, classification probability).

> Based on [DRAFTS](https://github.com/SukiYume/DRAFTS) by Zhang et al. | [Paper](https://arxiv.org/abs/2410.03200)

---

## Prerequisites

Before starting, ensure you have:

- **Model weights**: Place `cent_resnet18.pth` and `class_resnet18.pth` in `src/models/`
  - These are pre-trained models required for detection and classification
  - If not included in the repository, they should be obtained from the original DRAFTS repository or trained using the provided training scripts
- **Data files**: `.fits` or `.fil` files ready to process
- **Docker (recommended)**: Docker Desktop + NVIDIA Docker (for GPU) or CPU version
- **Local installation (alternative)**: Python 3.8+, CUDA 11+ (optional, for GPU acceleration)

---

## Configuration

**Important:** Configure paths and parameters before installation, especially if using Docker.

### Step 1: Set Data Path

**For Docker** - Edit `docker-compose.yml`:

Replace only the left side of the path (your local data directory). **Do not change** `/app/Data/raw:ro`:

```yaml
volumes:
  - D:/Your/Data/Path:/app/Data/raw:ro
  #     ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^
  #     EDIT THIS       DO NOT CHANGE THIS
```

Example:

```yaml
volumes:
  - /home/user/observations:/app/Data/raw:ro # Linux/Mac
  - D:\Seba - Dev\TESIS\Data:/app/Data/raw:ro # Windows
```

**For Local** - Edit directly `config.yaml`:

```yaml
data:
  input_dir: "D:/Your/Data/Path" # Direct path to your data
```

### Step 2: Configure Processing Parameters

Edit `config.yaml` with your observation parameters:

```yaml
data:
  input_dir: "/app/Data/raw/" # Docker path (or local path if not using Docker)
  targets:
    - "2017-04-03-08_16_13_142_0006" # File patterns to process
    - "2017-04-03-08_55_22_153_0006"

dispersion:
  dm_min: 0 # Minimum DM (pc cm⁻³)
  dm_max: 1024 # Maximum DM (pc cm⁻³) - adjust for your source

thresholds:
  detection_probability: 0.3 # Lower = more detections (0.1-0.7)
  classification_probability: 0.5 # Higher = fewer false positives (0.3-0.9)

output:
  save_only_burst: true # true = only BURST candidates, false = all detections
```

**Key parameters to adjust:**

- `dm_min/dm_max`: Set based on your source (nearby pulsars: 0-100, extragalactic FRBs: 100-2000)
- `detection_probability`: Lower values (0.1-0.2) for sensitive searches, higher (0.5-0.7) for conservative
- `classification_probability`: Higher values (0.7-0.9) reduce false positives

---

## Installation

### Option 1: Docker (Recommended)

**For GPU:**

```bash
git clone https://github.com/Kodamonkey/DRAFTS-UC.git
cd DRAFTS-UC
docker-compose build drafts-gpu
```

**For CPU (beta):**

```bash
docker-compose build drafts-cpu
```

### Option 2: Local Installation

```bash
git clone https://github.com/Kodamonkey/DRAFTS-UC.git
cd DRAFTS-UC
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### Docker

```bash
docker-compose run --rm drafts-gpu
# or for CPU:
docker-compose run --rm drafts-cpu
```

### Local

```bash
python main.py
```

### Command-Line Overrides (Optional)

You can override config parameters without editing the file:

```bash
python main.py --dm-max 512 --det-prob 0.5 --save-all-detections
python main.py --help  # See all available options
```

---

## Output

Results are saved in `./Results/`:

- **CSV files**: Candidate detections with DM, time, SNR, classification probability
- **Plots**: Dynamic spectra with detected candidates highlighted
- **Logs**: Processing information and timestamps

**Output modes:**

- `save_only_burst: true` → Only candidates classified as BURST (recommended for production)
- `save_only_burst: false` → All detections including NON-BURST (useful for analysis)

---

## Key Parameters Reference

| Parameter                    | Location    | Typical Values | Effect                                  |
| ---------------------------- | ----------- | -------------- | --------------------------------------- |
| `input_dir`                  | config.yaml | Path string    | Location of your data files             |
| `targets`                    | config.yaml | File patterns  | Which files to process                  |
| `dm_min/dm_max`              | config.yaml | 0-2000         | DM search range (pc cm⁻³)               |
| `detection_probability`      | config.yaml | 0.1-0.7        | Lower = more detections                 |
| `classification_probability` | config.yaml | 0.3-0.9        | Higher = fewer false positives          |
| `save_only_burst`            | config.yaml | true/false     | Filter output by classification         |
| `slice_duration_ms`          | config.yaml | 150-500        | Temporal resolution (ms)                |
| `downsampling.time_rate`     | config.yaml | 4-64           | Time reduction factor (higher = faster) |

---

## Troubleshooting

| Issue                        | Solution                                                            |
| ---------------------------- | ------------------------------------------------------------------- |
| **No files found**           | Check `input_dir` path in config.yaml or docker-compose.yml volumes |
| **No detections**            | Lower `detection_probability` to 0.2 or enable `force_plots: true`  |
| **Too many false positives** | Increase `classification_probability` to 0.7-0.9                    |
| **File truncated/corrupted** | Verify file integrity, use complete files                           |
| **Models not found**         | Place `cent_resnet18.pth` and `class_resnet18.pth` in `src/models/` |
| **Slow processing**          | Use GPU version or increase `downsampling.time_rate`                |
| **Docker won't start**       | Ensure Docker Desktop is running                                    |

---

## Advanced Features

### High-Frequency Pipeline (≥8 GHz)

For ALMA or high-frequency observations, the pipeline automatically enables:

- Multi-polarization analysis (IQUV)
- Linear polarization validation
- Dual-polarization SNR checks

Configure in `config.yaml`:

```yaml
high_frequency:
  auto_enable: true
  threshold_mhz: 8000.0
  enable_linear_validation: false # true = stricter validation
```

### Multi-Band Analysis

Enable frequency-dependent analysis:

```yaml
multiband:
  enabled: true # Note: ~3x slower processing
```

### Debug Mode

For troubleshooting:

```yaml
debug:
  show_frequency_info: true
  force_plots: true # Generate plots even without detections
```

---

## Important Notes

**Docker paths:**

- Data location: Edit `docker-compose.yml` volumes, not `config.yaml`
- Config changes: No rebuild needed, just re-run
- Unused volumes create folders: Comment them out in `docker-compose.yml`

**Performance:**

- GPU recommended for large files (>1 GB)
- CPU version available but significantly slower
- Adjust `downsampling.time_rate` to balance speed vs. resolution

---

## Citation

```bibtex
@article{zhang2024drafts,
  title={DRAFTS: A Deep Learning-Based Radio Fast Transient Search Pipeline},
  author={Zhang, Y.-K. and others},
  journal={arXiv preprint arXiv:2410.03200},
  year={2024}
}
```

**Author:** Sebastian Salgado Polanco
