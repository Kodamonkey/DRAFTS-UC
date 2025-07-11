"""
Configuración del Pipeline de Detección de FRB
===========================================================

Este archivo contiene todos los parámetros configurables del pipeline de detección
de Fast Radio Bursts (FRB). Los parámetros están organizados por categorías para
facilitar la configuración según las necesidades astronómicas.

GUÍA DE USO:
- Modifique los SWITCHES DE CONTROL para elegir modo manual o automático
- Configure los parámetros de cada sección según sus necesidades
- Los parámetros avanzados generalmente no requieren modificación.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
try:
    import torch
except ImportError:  # Si torch no está instalado, lo dejamos como None
    torch = None

# =============================================================================
# SWITCHES DE CONTROL - Activar/Desactivar configuraciones manuales
# =============================================================================

# --- Control de Rango DM Dinámico ---
DM_DYNAMIC_RANGE_ENABLE: bool = False       # True = zoom automático, False = rango fijo
DM_RANGE_ADAPTIVE: bool = False             # True = adaptar según confianza, False = factor fijo

# --- Control de RFI ---
RFI_ENABLE_ALL_FILTERS: bool = False        # True = todos los filtros, False = solo básicos
RFI_INTERPOLATE_MASKED: bool = False        # True = interpolar valores, False = mantener enmascarados
RFI_SAVE_DIAGNOSTICS: bool = False          # True = guardar gráficos, False = no guardar

# =============================================================================
# CONFIGURACIÓN PRINCIPAL - Parámetros que típicamente se modifican
# =============================================================================

# --- Rutas de archivos y datos ---
DATA_DIR = Path("./Data")                        # Directorio con archivos de entrada (.fits, .fil)
RESULTS_DIR = Path("./Results/ObjectDetection")  # Directorio para guardar resultados
FRB_TARGETS = ["3101_0001_00_8bit"]                       # Lista de targets FRB a procesar

# --- Configuración de Slice Temporal ---
SLICE_DURATION_MS: float = 196.0            # Duración deseada de cada slice en milisegundos 
                                            # El sistema calculará automáticamente SLICE_LEN según:
                                            # SLICE_LEN = round(SLICE_DURATION_MS / (TIME_RESO × DOWN_TIME_RATE × 1000))
                                            # Valores típicos: 16ms (rápido), 32ms (normal), 64ms (estándar), 128ms (lento)

SLICE_LEN_MIN: int = 32                     # Límite inferior de seguridad para el cálculo automático de SLICE_LEN
SLICE_LEN_MAX: int = 2048                   # Límite superior de seguridad para el cálculo automático de SLICE_LEN
                                            # Estos valores actúan como guardias: SLICE_LEN = max(MIN, min(calculado, MAX))
                                            # Evitan slices muy pequeños (<32) o muy grandes (>2048) que causen problemas

# --- Rango de Dispersion Measure (DM) ---
DM_min: int = 0                             # DM mínimo en pc cm⁻³
DM_max: int = 1028                           # DM máximo en pc cm⁻³

# --- Umbrales de detección ---
DET_PROB: float = 0.2                       # Probabilidad mínima para considerar una detección válida
CLASS_PROB: float = 0.5                     # Probabilidad mínima para clasificar como burst
SNR_THRESH: float = 3.0                     # Umbral de SNR para resaltar en visualizaciones

# --- Configuración de procesamiento ---
USE_MULTI_BAND: bool = False                 # Usar análisis multi-banda (Full/Low/High)
ENABLE_CHUNK_PROCESSING: bool = True        # Procesar archivos grandes en chunks
MAX_SAMPLES_LIMIT: int = 2000000            # Límite de muestras por chunk (memoria)

# =============================================================================
# CONFIGURACIÓN MANUAL - Variables que se modifican cuando switches = False
# =============================================================================

# --- Rango DM Manual (cuando DM_DYNAMIC_RANGE_ENABLE = False) ---

DM_RANGE_FACTOR: float = 0.3                # Factor de rango (0.3 = ±30% del DM óptimo)
DM_RANGE_MIN_WIDTH: float = 80.0            # Ancho mínimo del rango DM en pc cm⁻³
DM_RANGE_MAX_WIDTH: float = 300.0           # Ancho máximo del rango DM en pc cm⁻³

# --- Rango DM para Plots ---
DM_PLOT_MARGIN_FACTOR: float = 0.25         # Margen adicional para evitar bordes (25%)
DM_PLOT_MIN_RANGE: float = 120.0            # Rango mínimo del plot en pc cm⁻³
DM_PLOT_MAX_RANGE: float = 400.0            # Rango máximo del plot en pc cm⁻³
DM_PLOT_DEFAULT_RANGE: float = 250.0        # Rango por defecto sin candidatos

# --- Tipo de visualización por defecto ---
DM_RANGE_DEFAULT_VISUALIZATION: str = "detailed"  # Tipo de visualización por defecto

# --- RFI Manual (cuando RFI_ENABLE_ALL_FILTERS = False) ---
RFI_FREQ_SIGMA_THRESH = 5.0                # Umbral sigma para enmascarado de canales
RFI_TIME_SIGMA_THRESH = 5.0                # Umbral sigma para enmascarado temporal
RFI_ZERO_DM_SIGMA_THRESH = 4.0             # Umbral sigma para filtro Zero-DM
RFI_IMPULSE_SIGMA_THRESH = 6.0             # Umbral sigma para filtrado de impulsos
RFI_POLARIZATION_THRESH = 0.8              # Umbral para filtrado de polarización (0-1)
RFI_CHANNEL_DETECTION_METHOD = "mad"       # Método detección canales: "mad", "std", "kurtosis"
RFI_TIME_DETECTION_METHOD = "mad"          # Método detección temporal: "mad", "std", "outlier"

# =============================================================================
# CONFIGURACIÓN DE MODELOS Y SISTEMA - No dependen de switches
# =============================================================================

# --- Modelo de detección ---
MODEL_NAME = "resnet50"                     # Arquitectura del modelo de detección
MODEL_PATH = Path(f"./models/cent_{MODEL_NAME}.pth")  # Ruta al modelo entrenado

# --- Modelo de clasificación binaria ---
CLASS_MODEL_NAME = "resnet18"               # Arquitectura del modelo de clasificación
CLASS_MODEL_PATH = Path(f"./models/class_{CLASS_MODEL_NAME}.pth")  # Ruta al modelo

# --- Dispositivo de cómputo ---
if torch is not None:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = "cpu"

# =============================================================================
# CONFIGURACIÓN AUTOMÁTICA - Variables que se configuran automáticamente
# =============================================================================

# --- Metadatos del archivo (se configuran automáticamente) ---
FREQ: np.ndarray | None = None              # Array de frecuencias de observación (MHz)
FREQ_RESO: int = 0                          # Resolución de frecuencia (canales)
TIME_RESO: float = 0.0                      # Resolución temporal (segundos)
FILE_LENG: int = 0                          # Longitud del archivo (muestras)

# --- Configuración de Slice Temporal (calculada dinámicamente) ---
SLICE_LEN: int = 512                        # Número de muestras por slice (calculado automáticamente desde SLICE_DURATION_MS)

# --- Parámetros de decimación ---
DOWN_FREQ_RATE: int = 1                     # Factor de reducción en frecuencia
DOWN_TIME_RATE: int = 1                     # Factor de reducción en tiempo
DATA_NEEDS_REVERSAL: bool = False           # Invertir eje de frecuencia si es necesario

# --- Configuración de SNR y visualización ---
SNR_OFF_REGIONS = [(-250, -150), (-100, -50), (50, 100), (150, 250)]  # Regiones simétricas
SNR_COLORMAP = "viridis"                    # Mapa de colores para waterfalls
SNR_HIGHLIGHT_COLOR = "red"                 # Color para resaltar detecciones

# --- Configuración de chunking ---
CHUNK_OVERLAP_SAMPLES: int = 1000           # Solapamiento entre chunks
# Referencia de memoria:
# 100,000 muestras ≈ 512 MB RAM
# 500,000 muestras ≈ 2.5 GB RAM
# 1,000,000 muestras ≈ 5 GB RAM
# 2,000,000 muestras ≈ 10 GB RAM

# =============================================================================
# INFORMACIÓN ADICIONAL Y NOTAS
# =============================================================================

# --- GUÍA DE USO SIMPLIFICADA ---
"""
CONFIGURACIÓN SIMPLIFICADA:

1. SLICE TEMPORAL:
   - Solo configura SLICE_DURATION_MS con la duración deseada en milisegundos
   - El sistema calcula automáticamente SLICE_LEN según los metadatos del archivo
   - Ejemplos: 32.0 ms (rápido), 64.0 ms (normal), 128.0 ms (lento), 256.0 ms (detallado)

2. RANGO DM PARA PLOTS:
   Para usar configuración MANUAL:
   - DM_DYNAMIC_RANGE_ENABLE = False
   - Configurar: DM_RANGE_FACTOR, DM_PLOT_MARGIN_FACTOR, etc.
   
   Para usar configuración AUTOMÁTICA:
   - DM_DYNAMIC_RANGE_ENABLE = True

3. RFI:
   Para procesamiento BÁSICO:
   - RFI_ENABLE_ALL_FILTERS = False
   
   Para procesamiento COMPLETO:
   - RFI_ENABLE_ALL_FILTERS = True

CASOS DE USO TÍPICOS:
- Análisis rápido: SLICE_DURATION_MS = 32.0
- Análisis estándar: SLICE_DURATION_MS = 64.0  
- Análisis detallado: SLICE_DURATION_MS = 128.0
- Análisis ultra-detallado: SLICE_DURATION_MS = 256.0
"""

# --- Bandas de frecuencia automáticas ---
# El sistema genera automáticamente 3 bandas:
# - banda[0] = Full Band  (suma completa de frecuencias)
# - banda[1] = Low Band   (mitad inferior del espectro)  
# - banda[2] = High Band  (mitad superior del espectro)

# --- Notas sobre memoria y chunking ---
# El procesamiento en chunks es esencial para archivos grandes:
# - Archivos típicos de FRB pueden ser >30 GB
# - MAX_SAMPLES_LIMIT controla el tamaño de cada chunk
# - CHUNK_OVERLAP_SAMPLES evita perder detecciones en bordes

# --- Configuración recomendada para diferentes casos ---
# Para detecciones de alta precisión:
#   - DET_PROB = 0.05 (más sensible)
#   - SNR_THRESH = 2.5 (umbral más bajo)
#   - DM_RANGE_FACTOR = 0.2 (rango más estrecho)
#   - SLICE_DURATION_MS = 32.0 (slices más cortos)
#
# Para búsqueda exploratoria:
#   - DET_PROB = 0.1 (balanced)
#   - SNR_THRESH = 3.0 (estándar)
#   - DM_RANGE_FACTOR = 0.3 (rango más amplio)
#   - SLICE_DURATION_MS = 64.0 (duración estándar)
#
# Para procesamiento rápido:
#   - ENABLE_CHUNK_PROCESSING = True
#   - MAX_SAMPLES_LIMIT = 1000000 (chunks más pequeños)
#   - USE_MULTI_BAND = False (si no es necesario)
#   - SLICE_DURATION_MS = 128.0 (slices más largos)