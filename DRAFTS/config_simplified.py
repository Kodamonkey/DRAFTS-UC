"""
Configuración del Pipeline de Detección de FRB - Effelsberg
===========================================================

Este archivo contiene solo los parámetros esenciales para un astrónomo.
La visualización dinámica se ajusta automáticamente.

GUÍA RÁPIDA PARA ASTRÓNOMOS:
- Cambie DM_min y DM_max según su rango de búsqueda
- Ajuste DET_PROB para sensibilidad (0.05 = muy sensible, 0.2 = poco sensible)
- Modifique DATA_DIR y RESULTS_DIR según sus carpetas
- Todo lo demás se configura automáticamente
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
try:
    import torch
except ImportError:
    torch = None

# =============================================================================
# CONFIGURACIÓN ASTRONÓMICA - Solo estas variables necesita modificar
# =============================================================================

# --- Directorios ---
DATA_DIR = Path("./Data")                        # Carpeta con sus archivos .fits/.fil
RESULTS_DIR = Path("./Results/ObjectDetection")  # Carpeta para guardar resultados

# --- Rango de búsqueda de DM ---
DM_min: int = 0                                  # DM mínimo para búsqueda (pc cm⁻³)
DM_max: int = 1024                               # DM máximo para búsqueda (pc cm⁻³)

# --- Sensibilidad de detección ---
DET_PROB: float = 0.1                            # 0.05=muy sensible, 0.1=normal, 0.2=poco sensible

# --- Lista de archivos a procesar ---
FRB_TARGETS = ["B0355+54"]                       # Nombres de archivos (sin extensión)

# =============================================================================
# CONFIGURACIÓN AUTOMÁTICA - No tocar (funciona automáticamente)
# =============================================================================

# --- Zoom dinámico en plots (automático cuando hay candidatos) ---
DM_DYNAMIC_RANGE_ENABLE: bool = True             # Zoom automático en candidatos
DM_ZOOM_FACTOR: float = 0.3                      # Nivel de zoom (30% alrededor del candidato)

# --- Procesamiento ---
USE_MULTI_BAND: bool = True                      # Análisis en 3 bandas de frecuencia
ENABLE_CHUNK_PROCESSING: bool = True             # Para archivos grandes
MAX_SAMPLES_LIMIT: int = 2000000                 # Tamaño máximo por chunk

# --- Modelos de IA ---
MODEL_NAME = "resnet50"
MODEL_PATH = Path(f"./models/cent_{MODEL_NAME}.pth")
CLASS_MODEL_NAME = "resnet18"
CLASS_MODEL_PATH = Path(f"./models/class_{CLASS_MODEL_NAME}.pth")
CLASS_PROB: float = 0.5

# =============================================================================
# PARÁMETROS INTERNOS - Se configuran automáticamente
# =============================================================================

# --- Slice temporal ---
SLICE_DURATION_SECONDS: float = 0.032
SLICE_LEN_AUTO: bool = True
SLICE_LEN_INTELLIGENT: bool = True
SLICE_LEN_OVERRIDE_MANUAL: bool = True
SLICE_LEN_MIN: int = 16
SLICE_LEN_MAX: int = 512
SLICE_LEN: int = 32

# --- SNR ---
SNR_THRESH: float = 3.0
SNR_OFF_REGIONS = [(-250, -150), (-100, -50), (50, 100), (150, 250)]
SNR_COLORMAP = "viridis"
SNR_HIGHLIGHT_COLOR = "red"

# --- RFI ---
RFI_FREQ_SIGMA_THRESH = 5.0
RFI_TIME_SIGMA_THRESH = 5.0
RFI_ZERO_DM_SIGMA_THRESH = 4.0
RFI_IMPULSE_SIGMA_THRESH = 6.0
RFI_POLARIZATION_THRESH = 0.8
RFI_CHANNEL_DETECTION_METHOD = "mad"
RFI_TIME_DETECTION_METHOD = "mad"
RFI_ENABLE_ALL_FILTERS = False
RFI_INTERPOLATE_MASKED = False
RFI_SAVE_DIAGNOSTICS = False

# --- Chunking ---
CHUNK_OVERLAP_SAMPLES: int = 1000

# --- Metadatos del archivo (se cargan automáticamente) ---
FREQ: np.ndarray | None = None
FREQ_RESO: int = 0
TIME_RESO: float = 0.0
FILE_LENG: int = 0
DOWN_FREQ_RATE: int = 1
DOWN_TIME_RATE: int = 1
DATA_NEEDS_REVERSAL: bool = False

# --- Configuración DM dinámico interno (conectado a DM_ZOOM_FACTOR) ---
DM_RANGE_FACTOR: float = DM_ZOOM_FACTOR          # Mismo valor que DM_ZOOM_FACTOR
DM_PLOT_MARGIN_FACTOR: float = 0.2               # Margen extra para que no quede pegado
DM_RANGE_MIN_WIDTH: float = 80.0                 # Ancho mínimo del zoom
DM_RANGE_MAX_WIDTH: float = 300.0                # Ancho máximo del zoom
DM_PLOT_MIN_RANGE: float = 120.0
DM_PLOT_MAX_RANGE: float = 400.0
DM_PLOT_DEFAULT_RANGE: float = 250.0
DM_RANGE_ADAPTIVE: bool = True
DM_RANGE_DEFAULT_VISUALIZATION: str = "detailed"

# --- Sistema ---
if torch is not None:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = "cpu"

# =============================================================================
# NOTAS PARA EL ASTRÓNOMO
# =============================================================================
"""
FUNCIONAMIENTO DEL ZOOM DINÁMICO:
1. El pipeline busca candidatos en el rango DM_min a DM_max
2. Cuando encuentra un candidato (ej: DM = 300 pc cm⁻³):
   - El plot general sigue mostrando 0-1024 pc cm⁻³
   - Los plots de detección hacen zoom automático alrededor de DM=300
   - El candidato aparece centrado y grande en el plot
   - No necesita configurar nada más

AJUSTES SIMPLES:
- Para más sensibilidad: DET_PROB = 0.05
- Para menos falsos positivos: DET_PROB = 0.2
- Para más zoom en candidatos: DM_ZOOM_FACTOR = 0.4
- Para menos zoom: DM_ZOOM_FACTOR = 0.2

¡Es así de simple!
"""
