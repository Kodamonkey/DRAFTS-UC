# DRAFTS++: Deep Learning Radio Transient Search

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com/)
[![arXiv](https://img.shields.io/badge/arXiv-2410.03200-b31b1b.svg)](https://arxiv.org/abs/2410.03200)

![DRAFTS WorkFlow](WorkFlow.png)

## What is DRAFTS++?

**DRAFTS++** is an enhanced and production-ready implementation of the original DRAFTS pipeline. It uses a two-stage deep learning approach:

1. **CenterNet (Detection)** - Localizes burst candidates in time-DM space from dedispersed dynamic spectra
2. **ResNet (Classification)** - Distinguishes real FRBs from RFI and noise

### Key Improvements over Original DRAFTS

- **Configuration via YAML** - Simple `config.yaml` instead of hardcoded values
- **Docker support** - Reproducible CPU and GPU environments (CPU is beta)
- **Smart chunking** - Handles large files with automatic memory management
- **Logging** - Timestamped logs with color-coded console output
- **CLI flexibility** - Optional command-line overrides for quick experiments
- **Multi-polarization support** - High-frequency pipeline with IQUV analysis
- **Better error handling** - Graceful fallbacks (GPU→CPU, Torch→Numba→CPU)

> Based on [DRAFTS](https://github.com/SukiYume/DRAFTS) by Zhang et al. | [Paper](https://arxiv.org/abs/2410.03200)

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Kodamonkey/DRAFTS-UC.git
cd DRAFTS-UC

# 2. Configure
nano config.yaml  # Set input_dir and targets

# 3. Run
docker-compose build drafts-gpu && docker-compose run --rm drafts-gpu

# OR locally
pip install -r requirements.txt && python main.py
```

Results in `./Results/`

---

## Prerequisites

- Model weights: `src/models/cent_resnet18.pth` and `class_resnet18.pth`
- Data files: `.fits` or `.fil` files
- **Docker:** Docker Desktop + NVIDIA Docker (GPU) or CPU version
- **Local:** Python 3.8+, CUDA 11+ (optional)

---

## Configuration

### For Docker

**1. Mount your data** in `docker-compose.yml`:

```yaml
volumes:
  - D:/Your/Data/Path:/app/Data/raw:ro # Change to your path
```

**2. Configure** in `config.yaml`:

```yaml
data:
  input_dir: "/app/Data/raw/" # Keep for Docker
  targets: ["FRB20201124"] # Your file pattern

thresholds:
  detection_probability: 0.3 # Lower = more detections
  classification_probability: 0.5 # Higher = more conservative

output:
  save_only_burst: true # false = save all detections (BURST + NON-BURST)
```

**3. Run:**

```bash
docker-compose build drafts-gpu
docker-compose run --rm drafts-gpu
```

### For Local

**1. Configure** `config.yaml`:

```yaml
data:
  input_dir: "D:/Your/Data/Path" # Direct path
  targets: ["FRB20201124"]
```

**2. Run:**

```bash
python3 -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### CLI Overrides (Optional)

```bash
python main.py --dm-max 512 --det-prob 0.5 --save-all-detections
python main.py --help  # See all options
```

Full config options in `config.yaml`.

---

## Key Parameters

| Parameter                    | Location    | Values        | Purpose                                           |
| ---------------------------- | ----------- | ------------- | ------------------------------------------------- |
| `input_dir`                  | config.yaml | Path string   | Data location                                     |
| `targets`                    | config.yaml | File patterns | Which files to process                            |
| `detection_probability`      | config.yaml | 0.1-0.7       | CenterNet threshold (lower = more detections)     |
| `classification_probability` | config.yaml | 0.3-0.9       | ResNet threshold (higher = fewer false positives) |
| `dm_min/dm_max`              | config.yaml | 0-2000        | DM search range (pc cm⁻³)                         |
| `save_only_burst`            | config.yaml | true/false    | Save only BURST or all detections                 |

---

## Troubleshooting

| Issue                        | Solution                                                            |
| ---------------------------- | ------------------------------------------------------------------- |
| **No files found**           | Check `input_dir` path in config.yaml or docker-compose.yml volumes |
| **No detections**            | Lower `detection_probability` to 0.2 or enable `force_plots: true`  |
| **File truncated/corrupted** | Verify file integrity, use complete files                           |
| **Models not found**         | Place `cent_resnet18.pth` and `class_resnet18.pth` in `src/models/` |
| **Slow on CPU**              | Use GPU version or increase `downsampling.time_rate`                |
| **Docker won't start**       | Ensure Docker Desktop is running                                    |

---

## Important Notes

**Docker paths:**

- Data location: Edit `docker-compose.yml` volumes, not `config.yaml`
- Config changes: No rebuild needed, just re-run
- Unused volumes create folders: Comment them out in `docker-compose.yml`

**Output modes:**

- `save_only_burst: true` → Only BURST candidates (production)
- `save_only_burst: false` → All detections including NON-BURST (analysis)

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
