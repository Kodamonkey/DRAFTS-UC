"""Global configuration and runtime parameters for the Effelsberg pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

# Configuracion del dispositivo ---------------------------------------------------
if torch is not None:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = "cpu"

# Parametros de observacion -------------------------------------------------------
FREQ: np.ndarray | None = None # Frecuencia de observación, puede ser None si no se especifica.
FREQ_RESO: int = 0 # Resolución de frecuencia, en MHz.
TIME_RESO: float = 0.0 # Resolución de tiempo, en segundos.
FILE_LENG: int = 0 # Longitud del archivo, en muestras.
DOWN_FREQ_RATE: int = 1 # Tasa de reducción de frecuencia, factor por el cual se reduce la frecuencia.
DOWN_TIME_RATE: int = 1 # Tasa de reducción de tiempo, factor por el cual se reduce el tiempo. 
DATA_NEEDS_REVERSAL: bool = False # Indica si los datos necesitan ser revertidos (invertidos) en el eje de frecuencia.

# Configuracion del pipeline  ------------------------------------------------------
USE_MULTI_BAND: bool = False # Indica si se utiliza procesamiento de múltiples bandas.
SLICE_LEN: int = 512  # Longitud de cada slice, en muestras.
DET_PROB: float = 0.1 # Probabilidad de detección mínima para considerar un evento como válido.
DM_min: int = 0 # DM mínimo, en pc cm⁻³. 
DM_max: int = 1025 # DM máximo, en pc cm⁻³.

# Rutas de archivos y modelos ---------------------------------------------------
DATA_DIR = Path("./Data") # Directorio donde se almacenan los datos de entrada.
RESULTS_DIR = Path("./Results/ObjectDetection") # Directorio donde se guardan los resultados del procesamiento.
MODEL_NAME = "resnet50" # Nombre del modelo utilizado para la detección de eventos.
MODEL_PATH = Path(f"./models/cent_{MODEL_NAME}.pth") # Ruta al modelo preentrenado para la detección de eventos.

# Configuración del modelo de clasificación binaria
CLASS_MODEL_NAME = "resnet18"
CLASS_MODEL_PATH = Path(f"./models/class_{CLASS_MODEL_NAME}.pth")
# Probabilidad mínima para considerar que un parche corresponde a un burst
CLASS_PROB = 0.5

# Configuración de SNR y visualización -------------------------------------------
SNR_THRESH = 3.0  # Umbral de SNR para resaltar en visualizaciones
SNR_OFF_REGIONS = [(-200, -100), (-50, 50), (100, 200)]  # Regiones off para calcular ruido (en bins)
SNR_COLORMAP = "viridis"  # Mapa de colores para waterfalls
SNR_HIGHLIGHT_COLOR = "red"  # Color para resaltar detecciones por encima del umbral
 
# Default FRB targets --------------------------------------------------------
#Objetivos de FRB predeterminados. Esta lista se utiliza para buscar archivos FITS
FRB_TARGETS = ["B0355+54"] # "B0355+54", "FRB20121102", "FRB20201124", "FRB20180301", "3096_0001_00_8bit"