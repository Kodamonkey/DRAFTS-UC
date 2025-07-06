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
USE_MULTI_BAND: bool = True # Indica si se utiliza procesamiento de múltiples bandas.
# Bandas generadas automáticamente:
# banda[0] = Full Band  (suma completa de frecuencias)
# banda[1] = Low Band   (mitad inferior del espectro)  
# banda[2] = High Band  (mitad superior del espectro)


# Configuración dinámica de SLICE_LEN basada en duración temporal
SLICE_DURATION_SECONDS: float = 0.032  # Duración deseada por slice en segundos (32ms por defecto)
SLICE_LEN_AUTO: bool = True  # Si True, calcula SLICE_LEN automáticamente basado en SLICE_DURATION_SECONDS
SLICE_LEN_MIN: int = 16      # Valor mínimo permitido para SLICE_LEN
SLICE_LEN_MAX: int = 512     # Valor máximo permitido para SLICE_LEN

# Configuración avanzada de SLICE_LEN automático (basado en metadatos del archivo)
SLICE_LEN_INTELLIGENT: bool = True  # Si True, usa análisis automático completo de metadatos del archivo
SLICE_LEN_OVERRIDE_MANUAL: bool = True  # Si True, el sistema inteligente anula configuración manual

# SLICE_LEN se calculará dinámicamente o usará valor manual
SLICE_LEN: int = 32 # Valor manual (usado solo si SLICE_LEN_AUTO = False y SLICE_LEN_INTELLIGENT = False)
DET_PROB: float = 0.1 # Probabilidad de detección mínima para considerar un evento como válido.
DM_min: int = 0 # DM mínimo, en pc cm⁻³. 
DM_max: int = 1024 # DM máximo, en pc cm⁻³.

# Configuración de rangos DM dinámicos para visualización centrada en candidatos
DM_DYNAMIC_RANGE_ENABLE: bool = True  # Habilita cálculo automático del rango DM para plots
DM_RANGE_FACTOR: float = 0.2  # Factor de rango como fracción del DM óptimo (0.2 = ±20%)
DM_RANGE_MIN_WIDTH: float = 50.0  # Ancho mínimo del rango DM en pc cm⁻³
DM_RANGE_MAX_WIDTH: float = 200.0  # Ancho máximo del rango DM en pc cm⁻³
DM_RANGE_ADAPTIVE: bool = True  # Ajusta el rango basado en la confianza de detección
DM_RANGE_DEFAULT_VISUALIZATION: str = "detailed"  # Tipo de visualización por defecto: 'composite', 'patch', 'detailed', 'overview'

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

# Regiones off optimizadas para pulsos centralizados
# Asume que el pulso está en el centro, usa los bordes para estimar ruido
SNR_OFF_REGIONS = [(-250, -150), (-100, -50), (50, 100), (150, 250)]  # Regiones off simétricas

# Configuración alternativa más conservadora
# SNR_OFF_REGIONS = [(-200, -100), (100, 200)]  # Solo bordes izquierdo y derecho

SNR_COLORMAP = "viridis"  # Mapa de colores para waterfalls
SNR_HIGHLIGHT_COLOR = "red"  # Color para resaltar detecciones por encima del umbral

# Configuración de Mitigación de RFI ------------------------------------------
RFI_FREQ_SIGMA_THRESH = 5.0      # Umbral sigma para enmascarado de canales
RFI_TIME_SIGMA_THRESH = 5.0      # Umbral sigma para enmascarado temporal  
RFI_ZERO_DM_SIGMA_THRESH = 4.0   # Umbral sigma para filtro Zero-DM
RFI_IMPULSE_SIGMA_THRESH = 6.0   # Umbral sigma para filtrado de impulsos
RFI_POLARIZATION_THRESH = 0.8    # Umbral para filtrado de polarización (0-1)
RFI_ENABLE_ALL_FILTERS = False    # Habilita todos los filtros de RFI
RFI_INTERPOLATE_MASKED = False    # Interpola valores enmascarados
RFI_SAVE_DIAGNOSTICS = False      # Guarda gráficos de diagnóstico de RFI
RFI_CHANNEL_DETECTION_METHOD = "mad"  # Método para detectar canales malos: "mad", "std", "kurtosis"
RFI_TIME_DETECTION_METHOD = "mad"     # Método para detectar muestras temporales malas: "mad", "std", "outlier"
 
# Default FRB targets --------------------------------------------------------
#Objetivos de FRB predeterminados. Esta lista se utiliza para buscar archivos FITS
FRB_TARGETS = ["3100_0001_00_8bit"] # "B0355+54", "FRB20121102", "FRB20201124", "FRB20180301", "3097_0001_00_8bit"

# Configuración de límites de procesamiento -------------------------------------
MAX_SAMPLES_LIMIT: int = 2000000  # Límite por chunk para evitar problemas de memoria
# Con 4.4 GB disponibles, podemos procesar ~2M muestras por chunk (≈2 GB por chunk)
# El archivo completo se procesará en múltiples chunks automáticamente
CHUNK_OVERLAP_SAMPLES: int = 1000  # Solapamiento entre chunks para no perder detecciones en bordes
ENABLE_CHUNK_PROCESSING: bool = True  # Habilita procesamiento automático por chunks


'''
100,000 muestras ≈ 512 MB RAM
500,000 muestras ≈ 2.5 GB RAM
1,000,000 muestras ≈ 5 GB RAM
Archivo completo ≈ 33+ GB RAM
'''