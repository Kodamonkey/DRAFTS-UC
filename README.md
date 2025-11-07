# DRAFTS++: Deep Learning Radio Transient Search

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com/)
[![arXiv](https://img.shields.io/badge/arXiv-2410.03200-b31b1b.svg)](https://arxiv.org/abs/2410.03200)

Deep learning pipeline for detecting Fast Radio Bursts using CenterNet + ResNet.

![DRAFTS WorkFlow](WorkFlow.png)

**Based on:** [DRAFTS](https://github.com/SukiYume/DRAFTS) | [Paper](https://arxiv.org/abs/2410.03200)

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Kodamonkey/DRAFTS-UC.git
cd DRAFTS-UC

# 2. Configure
nano config.yaml  # Edit input_dir and targets

# 3. Run
docker-compose build drafts-gpu
docker-compose run --rm drafts-gpu
```

Results in `./Results/`

---

## Prerequisites

| Component                | Required                                                |
| ------------------------ | ------------------------------------------------------- |
| **Model weights**        | `src/models/cent_resnet18.pth` and `class_resnet18.pth` |
| **Data files**           | `.fits` or `.fil` files in `Data/raw/`                  |
| **Docker** (recommended) | Docker Desktop + NVIDIA Docker (GPU)                    |
| **Local** (alternative)  | Python 3.8+, CUDA 11+ (optional)                        |

---

## Configuration

Edit `config.yaml`:

```yaml
data:
  input_dir: "./Data/raw/" # Your data location
  results_dir: "./Results/" # Output location
  targets:
    - "FRB20201124_0009" # File patterns to process

dispersion:
  dm_min: 0
  dm_max: 1024

thresholds:
  detection_probability: 0.3 # Lower = more detections
  classification_probability: 0.5
```

Full options in `config.yaml` file.

---

## Running the Pipeline

### Option 1: Docker (Recommended)

```bash
# Build
docker-compose build drafts-gpu

# Run
docker-compose run --rm drafts-gpu
```

**CPU version:** Replace `drafts-gpu` with `drafts-cpu`

### Option 2: Local

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run
python main.py
```

---

## Project Structure

```
DRAFTS-UC/
├── config.yaml           # ← Configure here
├── main.py              # Entry point
├── Data/raw/            # ← Put data here
├── Results/             # ← Outputs here
└── src/models/          # ← Model weights here
    ├── cent_resnet18.pth
    └── class_resnet18.pth
```

---

## Troubleshooting

| Problem                  | Solution                                                    |
| ------------------------ | ----------------------------------------------------------- |
| No files found           | Verify `data.input_dir` and `data.targets` in `config.yaml` |
| No detections            | Lower `thresholds.detection_probability` to 0.2             |
| Too many false positives | Increase `thresholds.classification_probability` to 0.7     |
| Models not found         | Place weights in `src/models/`                              |
| Docker won't start       | Check Docker Desktop is running                             |

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

Academic and research use only. Please cite the original DRAFTS paper.

---

## Links

- [Original DRAFTS Repository](https://github.com/SukiYume/DRAFTS)
- [DRAFTS Paper](https://arxiv.org/abs/2410.03200)
