"""
Configuración Simple para Detección de FRB
==========================================

Este archivo contiene solo los parámetros esenciales que un astrónomo
necesita modificar para usar el pipeline de detección de FRB.

IMPORTANTE: 
- Solo modifique los valores que aparecen a continuación
- El rango de DM que configure aquí se usará para la BÚSQUEDA completa
- Los plots se enfocarán automáticamente en los candidatos detectados
"""

from pathlib import Path

# =============================================================================
# CONFIGURACIÓN ESENCIAL - Solo estos parámetros necesitan modificarse
# =============================================================================

# --- Datos de entrada ---
DATA_DIR = Path("./Data")                   # Carpeta con archivos .fits o .fil
RESULTS_DIR = Path("./Results")             # Carpeta donde guardar resultados
FRB_TARGETS = ["B0355+54"]                  # Lista de targets a procesar

# --- Rango de búsqueda DM ---
DM_min = 0                                  # DM mínimo de búsqueda (pc cm⁻³)
DM_max = 1024                               # DM máximo de búsqueda (pc cm⁻³)

# --- Sensibilidad de detección ---
DET_PROB = 0.3                             # Probabilidad mínima de detección (0.05 = más sensible, 0.2 = menos sensible)
CLASS_PROB = 0.5                           # Probabilidad mínima para clasificar como burst

# =============================================================================
# CONFIGURACIÓN AUTOMÁTICA - No modificar
# =============================================================================

# Estas variables se configuran automáticamente basándose en los parámetros de arriba
# y en la información extraída de los archivos de datos

# --- Configuración de slice temporal (automática) ---
SLICE_DURATION_SECONDS = 0.032             # 32ms por slice (recomendado para FRB)
SLICE_LEN_AUTO = True                       # Calcular automáticamente desde metadatos
SLICE_LEN_INTELLIGENT = True                # Usar análisis inteligente
SLICE_LEN_OVERRIDE_MANUAL = True            # El sistema anula configuración manual

# --- Configuración de visualización (automática) ---
DM_DYNAMIC_RANGE_ENABLE = True              # Plots se centran automáticamente en candidatos
DM_RANGE_FACTOR = 0.3                       # Factor de zoom automático (±30% del DM óptimo)
SNR_THRESH = 3.0                           # Umbral SNR para resaltar en plots

# --- Configuración de procesamiento (automática) ---
USE_MULTI_BAND = True                       # Usar análisis multi-banda
ENABLE_CHUNK_PROCESSING = True              # Procesar archivos grandes en chunks
MAX_SAMPLES_LIMIT = 2000000                # Límite de memoria por chunk

# --- Configuración de modelos (automática) ---
MODEL_NAME = "resnet50"
MODEL_PATH = Path(f"./models/cent_{MODEL_NAME}.pth")

# =============================================================================
# NOTAS IMPORTANTES
# =============================================================================

# RANGO DE DM:
# El rango DM_min a DM_max define el espacio completo de búsqueda.
# Los plots se centrarán automáticamente en los candidatos encontrados.
# No necesita configurar parámetros de visualización adicionales.

# SENSIBILIDAD:
# DET_PROB = 0.05  -> Muy sensible (más candidatos, más falsos positivos)
# DET_PROB = 0.10  -> Balanceado (recomendado)
# DET_PROB = 0.20  -> Conservador (menos candidatos, menos falsos positivos)

# DATOS:
# Coloque sus archivos .fits o .fil en la carpeta DATA_DIR
# Los resultados aparecerán en RESULTS_DIR
# Modifique FRB_TARGETS para especificar qué archivos procesar
