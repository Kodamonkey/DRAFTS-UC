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

- **CenterNet object detection** to infer **arrival time & DM** directly from time–DM "bow‑ties".
- **ResNet binary classification** to confirm candidates and **reduce false positives** dramatically.
- **Command-line configuration**: flexible parameter control via argumentos CLI con valores por defecto sensibles.
- **Chunked processing** of large files with automatic memory‑aware slicing.
- **PSRFITS & SIGPROC (.fil)** input support; optional multi‑band analysis.
- **Rich outputs**: CSV summaries, annotated plots (waterfalls, DM curves, S/N traces), and logs.
- **Trainable**: scripts to (re)train detection and classification models on your own data.
- **🐳 Docker support**: Reproducible environments for CPU and GPU with full documentation.

---

## Quick Start

### Opción 1: Docker (Recomendado) 🐳

Docker proporciona un entorno aislado y reproducible sin necesidad de instalar dependencias manualmente.

```bash
# 1) Clonar el repositorio
git clone https://github.com/Kodamonkey/DRAFTS-UC.git
cd DRAFTS-UC

# 2) Verificar que Docker Desktop esté corriendo
docker ps

# 3) Construir la imagen (CPU o GPU)
docker-compose build drafts-cpu    # Para sistemas sin GPU
docker-compose build drafts-gpu    # Para sistemas con GPU NVIDIA

# 4) Colocar archivos .fits/.fil en Data/raw/

# 5) Ejecutar el pipeline
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "tu-archivo"
```

⏱️ **Build:** 10-15 min primera vez, luego instantáneo (usa caché)

### Opción 2: Instalación Local

```bash
# 1) Clonar el repositorio
git clone https://github.com/Kodamonkey/DRAFTS-UC.git
cd DRAFTS-UC

# 2) Crear y activar entorno virtual (Python 3.8+)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Instalar dependencias
pip install -r requirements.txt

# 4) Colocar archivos .fits/.fil en Data/raw/

# 5) Ejecutar el pipeline con argumentos requeridos
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" --target "FRB121102_0001"
```

Cuando finalice, inspecciona `Results/` para ver los gráficos y CSV de detecciones.

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
│   ├── config/          # **start here** (user_config.py)
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
├── Dockerfile           # 🐳 Multi-stage Docker build (CPU/GPU)
└── docker-compose.yml   # Docker Compose orchestration
```

---

## Configuration

El pipeline se configura mediante **argumentos de línea de comandos**. Los siguientes parámetros son **obligatorios**:

### Parámetros Obligatorios

- `--data-dir`: Directorio con archivos de entrada (`.fits`, `.fil`)
- `--results-dir`: Directorio donde se guardarán los resultados
- `--target`: Patrón(es) para buscar archivos (puede especificar múltiples)

### Parámetros Opcionales Principales

- `--slice-duration`: Duración de cada ventana temporal en ms (default: 300.0)
- `--dm-min`: DM mínimo en pc cm⁻³ (default: 0)
- `--dm-max`: DM máximo en pc cm⁻³ (default: 1024)
- `--det-prob`: Probabilidad mínima de detección CenterNet (default: 0.3)
- `--class-prob`: Probabilidad mínima de clasificación ResNet (default: 0.5)
- `--snr-thresh`: Umbral SNR para visualizaciones (default: 5.0)

### Parámetros de Análisis Avanzado

- `--multi-band`: Activar análisis multi-banda (Full/Low/High)
- `--down-freq-rate`: Factor de reducción en frecuencia (default: 1)
- `--down-time-rate`: Factor de reducción en tiempo (default: 8)
- `--auto-high-freq`: Activar pipeline de alta frecuencia automático (default: True)
- `--high-freq-threshold`: Umbral de frecuencia en MHz para alta frecuencia (default: 8000.0)

### Parámetros de Polarización (PSRFITS)

- `--polarization-mode`: Modo de polarización: intensity/linear/circular/pol0-3 (default: intensity)
- `--polarization-index`: Índice cuando IQUV no está disponible (default: 0)

### Parámetros de Visualización y Debug

- `--force-plots`: Generar gráficos siempre (incluso sin candidatos)
- `--debug-frequency`: Mostrar información detallada de frecuencias
- `--save-only-burst`: Guardar solo candidatos BURST (default: True)
- `--save-all`: Guardar todos los candidatos (BURST y no-BURST)

> **Nota:** También existe `src/config/user_config.py` con valores por defecto, pero los argumentos de línea de comandos son obligatorios y tienen prioridad.

---

## Running the Pipeline

### Comando Básico

Desde la raíz del repositorio:

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" --target "nombre_archivo"
```

### Ejemplos de Uso

**Procesamiento simple con valores por defecto:**

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" --target "FRB121102_0001"
```

**Con umbrales personalizados:**

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "FRB121102" --det-prob 0.5 --class-prob 0.6
```

**Activar análisis multi-banda:**

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "FRB121102" --multi-band --slice-duration 3000.0
```

**Procesar múltiples archivos:**

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "2017-04-03" "FRB" "B0355"
```

**Configurar rango DM personalizado:**

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "FRB121102" --dm-min 100 --dm-max 512
```

**Ver todos los parámetros disponibles:**

```bash
python main.py --help
```

### Proceso de Ejecución

Para cada archivo que coincida con el patrón especificado en `--target`:

1. **Load & chunk**: Carga el espectro dinámico (.fits/.fil) de forma eficiente en memoria
2. **Dedisperse**: Dedispersa en el rango `[DM_min, DM_max]` (usa GPU si está disponible)
3. **Detect candidates**: Detecta candidatos en tiempo–DM usando CenterNet → cajas + scores
4. **Classify**: Clasifica cada candidato con ResNet → probabilidad FRB vs no-FRB
5. **Save outputs**: Guarda figuras anotadas, CSV por archivo, y logs en el directorio de resultados

> **Tip:** Si los umbrales son muy estrictos, verás menos detecciones pero con mayor confianza. Relaja `--det-prob` o `--class-prob` para ser más inclusivo.

---

## Ejemplos Adicionales

**Procesamiento de observaciones por lote:**

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "session1_" "session2_" "2024-10-05"
```

**Alta frecuencia con umbral personalizado:**

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "ALMA_observation" --auto-high-freq --high-freq-threshold 7500.0
```

**Modo debug con gráficos forzados:**

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "test_file" --force-plots --debug-frequency
```

**Guardar todos los candidatos (no solo BURST):**

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "FRB" --save-all
```

**Plotting ad-hoc (inspección/debug):**

```bash
python src/scripts/absolute_segment_plots.py \
  --filename FRB121102_0001.fits --start 10.0 --duration 5.0 --dm 565
```

**Integración en tu propia aplicación Python:**

```python
from src.core.pipeline import run_pipeline

# Configuración personalizada
config_dict = {
    "DATA_DIR": Path("./Data/raw/"),
    "RESULTS_DIR": Path("./Results/"),
    "FRB_TARGETS": ["FRB121102"],
    "DM_min": 100,
    "DM_max": 600,
    "DET_PROB": 0.4,
    "CLASS_PROB": 0.6,
}

# Ejecutar con configuración personalizada
run_pipeline(config_dict=config_dict)
```

---

## Running with Docker 🐳

### Quick Docker Commands

**Build image (first time only, 10-15 min):**

```bash
# GPU (recommended if you have NVIDIA)
docker-compose build drafts-gpu

# CPU (no GPU required)
docker-compose build drafts-cpu
```

**Run pipeline:**

```bash
# GPU
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "FRB121102"

# CPU
docker-compose run --rm drafts-cpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "FRB121102"
```

**Docker examples with custom parameters:**

```bash
# Multi-band analysis
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "FRB121102" \
  --multi-band --slice-duration 3000.0

# Custom thresholds
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "FRB" \
  --det-prob 0.5 --class-prob 0.6 --dm-min 100 --dm-max 600

# Batch processing
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "2017-04-03" "FRB" "B0355"
```

**Useful Docker commands:**

```bash
# Interactive shell (for debugging)
docker-compose run --rm --entrypoint /bin/bash drafts-gpu

# View help
docker-compose run --rm drafts-gpu --help

# Verify GPU availability
docker-compose run --rm --entrypoint python drafts-gpu \
  -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Clean up
docker-compose down --rmi all --volumes
```

**Note:** Docker automatically mounts:
- `./Data/raw/` → input data
- `./Results/` → pipeline outputs  
- `./src/models/` → model weights (.pth files)

---

## Ejemplos de Uso Avanzado

Basados en la ayuda integrada del pipeline (`python main.py --help`):

**Ejecutar con archivos específicos y umbrales personalizados:**

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "2017-04-03-08_55_22" --det-prob 0.5 --class-prob 0.6
```

**Cambiar directorios y configurar rango DM:**

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "observation" --dm-max 512
```

**Activar análisis multi-banda:**

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "FRB" --multi-band --slice-duration 3000.0
```

**Procesamiento de alta frecuencia con umbral personalizado:**

```bash
python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
  --target "high_freq_obs" --auto-high-freq --high-freq-threshold 7500.0
```

---

## Data Requirements

- **Inputs:** single‑beam **PSRFITS (.fits)** or **SIGPROC filterbank (.fil)** containing time–frequency power.
- **Headers:** should include frequency axis, sample time, and shape (#chans, #samples).
- **Placement:** files go under `Data/raw/` (or your custom `DATA_DIR`).
- **Size:** large files are fine—processing is chunked automatically. Ensure free disk for `Results/` and temporary intermediates under `Data/processed/`.

---

## Model Weights

Place pre‑trained weights under `src/models/` with these names:

- **Detection (CenterNet):** `cent_resnet18.pth` or `cent_resnet50.pth`
- **Classification (ResNet):** `class_resnet18.pth` or `class_resnet50.pth`

The pipeline configuration in `src/config/config.py` defaults to `resnet18` for balance of speed/accuracy. To use `resnet50`, edit the `MODEL_NAME` and `CLASS_MODEL_NAME` variables in config.

> **Custom models:** If you train your own, copy your `best_model.pth` to `src/models/` and rename accordingly or update the model names in config.

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

After training, place your `best_model.pth` under `src/models/` with the expected name (or update config) and re‑run the pipeline.

---

## Outputs

- **Per‑file CSV** with all candidates that pass thresholds (arrival time, DM, scores, S/N).
- **Figures**: annotated waterfalls, time–DM “bow‑ties”, and S/N/DM curves.
- **Logs**: detailed progress and timing (helpful for profiling and debugging).

---

## Tips & Troubleshooting

### General Issues

- **No detections?** Reduce `--det-prob` o `--class-prob`, amplía el rango DM, o verifica la calidad de los datos y el modo de polarización.
- **Demasiados falsos positivos?** Aumenta los umbrales, activa `--multi-band`, o restringe el rango DM.
- **Ejecución lenta en CPU?** Instala PyTorch con soporte CUDA y drivers NVIDIA adecuados para habilitar GPU.
- **Eje de frecuencia invertido?** Usa `--debug-frequency` y verifica la salida del lector.
- **Datos de alta frecuencia (ALMA/mm-wave):** Mantén `--auto-high-freq` activado (default) para que los parámetros se adapten automáticamente.
- **Error "required arguments"?** Recuerda que `--data-dir`, `--results-dir` y `--target` son obligatorios.
- **Ver ayuda completa?** Ejecuta `python main.py --help` para ver todos los parámetros disponibles.

### Docker-Specific Issues

- **"Docker daemon not running"?** Abre Docker Desktop y espera a que inicie completamente.
- **"Modelos no encontrados"?** Verifica que existan `src/models/cent_resnet18.pth` y `src/models/class_resnet18.pth`.
- **Build muy lento?** Normal la primera vez (10-15 min). Siguientes builds usan caché (~30 seg).
- **"Permission denied" al escribir resultados?** En Windows: `icacls Results /grant Everyone:F /T`
- **"CUDA out of memory"?** Usa la versión CPU: `docker-compose run --rm drafts-cpu ...`

---

## Citation & Acknowledgements

If you use DRAFTS++ in research, please cite the original DRAFTS paper and this repository fork:

- **Zhang, Y.‑K., et al. (2024)**, _DRAFTS: A Deep Learning‑Based Radio Fast Transient Search Pipeline_ (arXiv:2410.03200).
- **DRAFTS‑UC / DRAFTS++**: this repository and documentation.

**Made by Sebastian Salgado Polanco**
