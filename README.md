# DRAFTS++: Deep Learning Radio Transient Search

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com/)
[![Research](https://img.shields.io/badge/Research-Academic-green.svg)](https://github.com/Kodamonkey/DRAFTS-UC)
[![arXiv](https://img.shields.io/badge/arXiv-2410.03200-b31b1b.svg)](https://arxiv.org/abs/2410.03200)

![DRAFTS WorkFlow](WorkFlow.png)

> **Original work:** [DRAFTS](https://github.com/SukiYume/DRAFTS) | [Paper](https://arxiv.org/abs/2410.03200)

## Project Overview

**DRAFTS++** is an advanced pipeline for detecting **Fast Radio Bursts (FRBs)** in radio astronomy data using deep learning. It builds upon the original **DRAFTS** (Deep Learning-based RAdio Fast Transient Search) framework, integrating modern neural networks to overcome challenges like radio-frequency interference (RFI) and propagation dispersion that hinder traditional search algorithms. In DRAFTS++, a **deep-learning object detector** (CenterNet-based) localizes burst candidates in dedispersed time–DM space, and a **binary classifier** (ResNet-based) verifies each candidate to distinguish real FRBs from noise/RFI. This two-stage approach greatly improves detection accuracy and reduces false positives compared to classical methods (e.g., PRESTO/Heimdall).

> **What's DRAFTS-UC?**  
> DRAFTS++ (a.k.a. _DRAFTS-UC_) is our maintained fork/extension. It keeps the original DRAFTS ideas and models, adds modern engineering (logging, chunking, GPU/CPU fallbacks), and streamlines configuration for easy, reproducible runs.

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

### Path Configuration

**Local execution:** Edit paths directly in `config.yaml`:

```yaml
data:
  input_dir: "D:/MyData/FRB/" # Any valid path
  results_dir: "./Results/"
```

**Docker execution:** Paths in `config.yaml` must match volume mounts in `docker-compose.yml`:

```yaml
# docker-compose.yml - Define your data location here
volumes:
  - D:/Seba - Dev/TESIS/Data/raw/:/app/Data/raw:ro # Host : Container

# config.yaml - Use the container path
data:
  input_dir: "/app/Data/raw/" # Must match container mount point
```

**To change data location in Docker:** Edit `docker-compose.yml` volumes, not `config.yaml`.

### Pipeline Parameters

Edit `config.yaml`:

```yaml
data:
  targets:
    - "FRB20201124_0009" # File patterns to process

temporal:
  slice_duration_ms: 300.0

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

---
