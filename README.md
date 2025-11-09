# DRAFTS++: Deep Learning Radio Transient Search

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com/)
[![Research](https://img.shields.io/badge/Research-Academic-green.svg)](https://github.com/Kodamonkey/DRAFTS-UC)
[![arXiv](https://img.shields.io/badge/arXiv-2410.03200-b31b1b.svg)](https://arxiv.org/abs/2410.03200)

![DRAFTS WorkFlow](WorkFlow.png)

> **Original work:** [DRAFTS](https://github.com/SukiYume/DRAFTS) | [Paper](https://arxiv.org/abs/2410.03200)

## Overview

**DRAFTS++** detects Fast Radio Bursts (FRBs) using deep learning: a **CenterNet detector** localizes candidates in time-DM space, and a **ResNet classifier** verifies them as real FRBs vs RFI/noise. This two-stage approach outperforms classical methods (PRESTO/Heimdall).

> **DRAFTS-UC** is our maintained fork with modern engineering: logging, chunking, GPU/CPU fallbacks, and simplified configuration.

---

## Getting Started

### Prerequisites

| Component                | Required                                                |
| ------------------------ | ------------------------------------------------------- |
| **Model weights**        | `src/models/cent_resnet18.pth` and `class_resnet18.pth` |
| **Data files**           | `.fits` or `.fil` files                                 |
| **Docker** (recommended) | Docker Desktop + NVIDIA Docker (GPU)                    |
| **Local** (alternative)  | Python 3.8+, CUDA 11+ (optional)                        |

### Installation & Setup

**1. Clone repository:**

```bash
git clone https://github.com/Kodamonkey/DRAFTS-UC.git
cd DRAFTS-UC
```

**2. Configure paths:**

Choose your method:

<details>
<summary><b>Docker Setup (Recommended)</b></summary>

Edit `docker-compose.yml` to mount your data:

```yaml
volumes:
  - D:/Your/Data/Path:/app/Data/raw:ro # Your data location
```

Edit `config.yaml` with container paths:

```yaml
data:
  input_dir: "/app/Data/raw/" # Keep as-is for Docker
  targets:
    - "your-file-pattern" # Match your files

thresholds:
  detection_probability: 0.3 # Lower = more detections
  classification_probability: 0.5 # Higher = more conservative
```

**Run:**

```bash
# Build once
docker-compose build drafts-gpu

# Run (no rebuild needed after config changes)
docker-compose run --rm drafts-gpu

# CPU version: replace drafts-gpu with drafts-cpu
```

</details>

<details>
<summary><b>Local Setup</b></summary>

Edit `config.yaml` with direct paths:

```yaml
data:
  input_dir: "D:/Your/Data/Path" # Direct path to your data
  targets:
    - "your-file-pattern"

thresholds:
  detection_probability: 0.3
  classification_probability: 0.5
```

**Run:**

```bash
# Setup environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Execute
python main.py
```

</details>

**Results appear in:** `./Results/`

### Quick CLI Overrides (Optional)

For quick parameter adjustments without editing `config.yaml`:

```bash
# Override thresholds
python main.py --dm-max 512 --det-prob 0.5

# Enable features
python main.py --multi-band --force-plots

# See all options
python main.py --help
```

**Note:** CLI args override `config.yaml` values. For permanent changes, edit `config.yaml`.

### Configuration Tips

**All parameters** are in `config.yaml`:

```yaml
temporal:
  slice_duration_ms: 300.0 # Analysis window (150-500 ms)

dispersion:
  dm_min: 0 # DM search range
  dm_max: 1024

downsampling:
  frequency_rate: 1 # Frequency reduction (1 = none)
  time_rate: 8 # Time reduction

multiband:
  enabled: false # Multi-band analysis (Full/Low/High)

high_frequency:
  auto_enable: true # Auto-activate for freq ≥ 8 GHz

polarization:
  mode: "intensity" # intensity | linear | circular | pol0-3

output:
  save_only_burst: true # Keep only BURST candidates
```

See `config.yaml` for full documentation.

---

## Important Notes

### Docker Considerations

⚠️ **Volume behavior:**

- Docker creates local folders for mounted volumes that don't exist
- To prevent: comment out unused volumes in `docker-compose.yml`
- **To change data location:** Edit `docker-compose.yml` volumes, NOT `config.yaml`
- **After config changes:** No rebuild needed, just re-run

✅ **Path mapping:**

```yaml
# docker-compose.yml defines HOST → CONTAINER mapping
- D:/My/Data:/app/Data/raw:ro

# config.yaml uses CONTAINER path
input_dir: "/app/Data/raw/"
```

### File Integrity

✅ **Verify before processing:**

- Pipeline warns if FITS files are truncated/corrupted
- Check: actual file size matches expected size
- Use complete, non-corrupted files for accurate results

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

---

## Author

**Sebastian Salgado Polanco**
