# DRAFTS++: Deep Learning Radio Transient Search

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/Kodamonkey/DRAFTS-UC/releases)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2410.03200-b31b1b.svg)](https://arxiv.org/abs/2410.03200)

![DRAFTS WorkFlow (LF)](WorkFlow-LF.png)

![High-frequency pipeline: frequency integration and ResNet18 classification](Workflow-HF.png)

*Above: low-frequency / general pipeline (LF); below: high-frequency (≥8 GHz) pipeline (HF) — two execution modes.*

## What is DRAFTS++?

**DRAFTS++** is a deep learning pipeline for detecting Fast Radio Bursts (FRBs) in radio astronomy data. It uses two stages: **CenterNet** localizes burst candidates in time–DM space from dedispersed dynamic spectra, and **ResNet** classifies them as BURST vs non-BURST (RFI/noise). Input: `.fits` or `.fil` files; output: candidates with DM, time, SNR, and classification probability.

For high-frequency observations (≥8 GHz), the pipeline uses matched-filter detection, frequency integration, and ResNet18 to classify candidates from SNR peaks.

> Based on [DRAFTS](https://github.com/SukiYume/DRAFTS) by Zhang et al. | [Paper](https://arxiv.org/abs/2410.03200)

---

## How to run the pipeline

1. **Prerequisites**  
   Place `cent_resnet18.pth` and `class_resnet18.pth` in `src/models/`. Have `.fits` or `.fil` data ready. Docker is recommended; otherwise use Python 3.8+ and optionally CUDA.

2. **Configure paths**  
   **Docker:** Edit `volumes` in `docker-compose.yml` (only the left side — your local data path). Leave `/app/Data/raw:ro` unchanged.  
   **Local:** Set `data.input_dir` in `config.yaml` to your data directory.

   ```yaml
   # docker-compose.yml (Docker)
   volumes:
     - /path/to/your/data:/app/Data/raw:ro

   # config.yaml (Local)
   data:
     input_dir: "/path/to/your/data"
   ```

3. **Key parameters**  
   In `config.yaml` set: `targets` (file patterns to process), `dm_min` / `dm_max`, `detection_probability`, and `classification_probability`.

4. **Install**  
   **Docker (GPU):** `docker-compose build drafts-gpu`  
   **Docker (CPU):** `docker-compose build drafts-cpu`  
   **Local:** `python3 -m venv .venv`, activate it, then `pip install -r requirements.txt`.

5. **Run**  
   **Docker:** `docker-compose run --rm drafts-gpu` (or `drafts-cpu`)  
   **Local:** `python main.py`  
   Use `python main.py --help` for command-line overrides.

Results are written to `./Results/` (CSV, plots, logs).

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

**DRAFTS++ developer:** Sebastian Salgado Polanco
