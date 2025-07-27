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
# CONFIGURACIÓN DE GENERACIÓN DE PLOTS
# =============================================================================

# Si ``PLOT_CONTROL_DEFAULT`` es ``True`` se mantiene el comportamiento
# estándar de generar los distintos plots únicamente cuando se detecta un
# candidato.  Al ponerlo en ``False`` se activan los switches individuales para
# controlar qué visualizaciones se generan independientemente de que haya o no
# candidatos detectados.

PLOT_CONTROL_DEFAULT: bool = True

PLOT_WATERFALL_DISPERSION: bool = False      # Waterfalls dispersados
PLOT_WATERFALL_DEDISPERSION: bool = False    # Waterfalls de-dispersados
PLOT_COMPOSITE: bool = False                 # Figura composite por slice
PLOT_DETECTION_DM_TIME: bool = True          # Plots de detección DM-tiempo
PLOT_PATCH_CANDIDATE: bool = False           # Patch del candidato clasificado

# =============================================================================    
    

# =============================================================================
# SWITCHES DE CONTROL ESENCIALES - Solo configuraciones principales
# =============================================================================


# --- Control de Debug (ESENCIAL) ---
DEBUG_FREQUENCY_ORDER: bool = False         # True = debug orden de frecuencias y dedispersión, False = sin debug para ahorrar memoria

# =============================================================================
# CONFIGURACIÓN PRINCIPAL - Parámetros que típicamente se modifican
# =============================================================================

# --- Rutas de archivos y datos ---
DATA_DIR = Path("./Data")                        # Directorio con archivos de entrada (.fits, .fil)
RESULTS_DIR = Path("./Results/ObjectDetection")  # Directorio para guardar resultados
# --- Lista de targets optimizada para múltiples archivos ---

FRB_TARGETS = ["3100_0001_00_8bit"]          # Lista de targets FRB a procesar - Reducida para pruebas

# Para procesar todos: ["FRB20201124_0009", "FRB20180301_0002", "B0355+54_FB_20220918"]
# Nota: FRB20180301_0002.fits parece estar corrupto - revisar archivo

# --- Configuración de Slice Temporal (ESENCIAL) ---
SLICE_DURATION_MS: float = 1000.0             # Duración deseada de cada slice en milisegundos
                                            # Ajustado para obtener todos los slices
                                            # El sistema calculará automáticamente SLICE_LEN según:
                                            # SLICE_LEN = round(SLICE_DURATION_MS / (TIME_RESO × DOWN_TIME_RATE × 1000))
                                            # Valores típicos: 16ms (rápido), 32ms (normal), 64ms (estándar), 128ms (lento)

# --- Rango de Dispersion Measure (DM) ---
DM_min: int = 0                             # DM mínimo en pc cm⁻³
DM_max: int = 1024                           # DM máximo en pc cm⁻³

# --- Umbrales de detección ---
DET_PROB: float = 0.4                       # Probabilidad mínima para considerar una detección válida
CLASS_PROB: float = 0.5                     # Probabilidad mínima para clasificar como burst
SNR_THRESH: float = 3.0                     # Umbral de SNR para resaltar en visualizaciones

# --- Configuración de procesamiento ---
USE_MULTI_BAND: bool = False                 # Usar análisis multi-banda (Full/Low/High)

# =============================================================================
# CONFIGURACIÓN MANUAL - Solo configuraciones esenciales
# =============================================================================

# (Las configuraciones avanzadas se movieron a la sección de análisis arriba)

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

# --- Parámetros de downsampling ---
DOWN_FREQ_RATE: int = 1                     # Factor de reducción en frecuencia
DOWN_TIME_RATE: int = 1                     # Factor de reducción en tiempo
DATA_NEEDS_REVERSAL: bool = False           # Invertir eje de frecuencia si es necesario

# --- Límites de seguridad ---
MAX_SAMPLES_LIMIT: int = 100_000_000        # Límite máximo de muestras para archivos .fil (100M)

# --- Configuración de SNR y visualización (SIMPLIFICADA) ---
# Solo configuraciones esenciales, las estéticas están en la sección de análisis arriba

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


4. VISUALIZACIÓN SNR:
   Para MOSTRAR líneas rojas del SNR en composite:
   - SNR_SHOW_PEAK_LINES = True
   
   Para OCULTAR líneas rojas del SNR en composite:
   - SNR_SHOW_PEAK_LINES = False

5. DEBUG DE ARCHIVOS Y FRECUENCIAS:
   Para HABILITAR debugs detallados de archivos:
   - DEBUG_FREQUENCY_ORDER = True
   
   Para DESHABILITAR debugs (producción):
   - DEBUG_FREQUENCY_ORDER = False
   
   Los debugs muestran:
   - Información completa del archivo (.fits/.fil)
   - Orden y valores de frecuencias
   - Parámetros de decimación
   - Datos cargados en memoria
   - Dirección de dedispersión

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
#   - USE_MULTI_BAND = False (si no es necesario)
#   - SLICE_DURATION_MS = 128.0 (slices más largos)

# --- CONFIGURACIONES RECOMENDADAS PARA DIFERENTES ESCENARIOS ---

# Para ARCHIVO ÚNICO (análisis detallado):
#   - DEBUG_FREQUENCY_ORDER = True
#   - GENERATE_WATERFALLS = True
#
# Para MÚLTIPLES ARCHIVOS (procesamiento en lote):
#   - DEBUG_FREQUENCY_ORDER = False  ← Configuración actual
#   - GENERATE_WATERFALLS = True
#   - SKIP_CORRUPTED_FILES = True
#   - FORCE_GARBAGE_COLLECTION = True
#
# Para ARCHIVOS MUY GRANDES (>5GB):
#   - REDUCE_VISUALIZATION_QUALITY = True
#   - USE_MULTI_BAND = False
#
# Para ANÁLISIS RÁPIDO (solo detección):
#   - GENERATE_WATERFALLS = False
#   - GENERATE_PATCHES = False
#   - GENERATE_COMPOSITES = False

"""
SOLUCIÓN PARA EL PROBLEMA ACTUAL:

1. El archivo FRB20180301_0002.fits está corrupto ("buffer is too small")
2. B0355+54_FB_20220918.fits es muy grande (1.09 GB en memoria + procesamiento)
3. La configuración actual intenta cargar demasiado en memoria

PASOS PARA RESOLVER:

1. Procesar archivos de uno en uno:
   FRB_TARGETS = ["FRB20201124_0009"]  # Solo uno por vez

2. Una vez confirmado que funciona, procesar los archivos grandes:
   FRB_TARGETS = ["B0355+54_FB_20220918"]

3. Investigar y reparar el archivo corrupto:
   FRB_TARGETS = ["FRB20180301_0002"]  # Este requiere atención especial

4. La configuración actual está optimizada para múltiples archivos pequeños-medianos
"""

# =============================================================================
# CONFIGURACIONES OPCIONALES/AVANZADAS - Candidatas para eliminación
# =============================================================================
"""
ANÁLISIS DE CONFIGURACIONES OPCIONALES:

Las siguientes configuraciones pueden no ser esenciales para el funcionamiento básico
del pipeline. Cada una se analiza con su propósito y ubicación de uso.
"""

# --- 1. CONFIGURACIONES DE RANGO DM DINÁMICO (POSIBLEMENTE INNECESARIAS) ---
# UBICACIÓN: Se usan en astro_conversions.py y visualización
# PROPÓSITO: Ajustar automáticamente el rango DM según candidatos detectados
# IMPACTO: Solo afecta visualizaciones, no detección
DM_DYNAMIC_RANGE_ENABLE: bool = False       # True = zoom automático, False = rango fijo
DM_RANGE_ADAPTIVE: bool = False             # True = adaptar según confianza, False = factor fijo
DM_RANGE_FACTOR: float = 0.3                # Factor de rango (0.3 = ±30% del DM óptimo)
DM_RANGE_MIN_WIDTH: float = 80.0            # Ancho mínimo del rango DM en pc cm⁻³
DM_RANGE_MAX_WIDTH: float = 300.0           # Ancho máximo del rango DM en pc cm⁻³

# --- 2. CONFIGURACIONES DE PLOTS ESTÉTICAS (POSIBLEMENTE INNECESARIAS) ---
# UBICACIÓN: Se usan en visualization.py para ajustar plots
# PROPÓSITO: Controlar márgenes y rangos de visualización
# IMPACTO: Solo estética, no afecta detección
DM_PLOT_MARGIN_FACTOR: float = 0.25         # Margen adicional para evitar bordes (25%)
DM_PLOT_MIN_RANGE: float = 120.0            # Rango mínimo del plot en pc cm⁻³
DM_PLOT_MAX_RANGE: float = 400.0            # Rango máximo del plot en pc cm⁻³
DM_PLOT_DEFAULT_RANGE: float = 250.0        # Rango por defecto sin candidatos
DM_RANGE_DEFAULT_VISUALIZATION: str = "detailed"  # Tipo de visualización por defecto


# --- 4. CONFIGURACIONES DE VISUALIZACIÓN SNR (PURAMENTE ESTÉTICAS) ---
# UBICACIÓN: Se usan en visualization.py y image_utils.py
# PROPÓSITO: Control de apariencia de plots
# IMPACTO: Solo estética, no afecta detección
SNR_SHOW_PEAK_LINES: bool = False            # True = mostrar líneas rojas del SNR peak, False = ocultar líneas
SNR_OFF_REGIONS = [(-250, -150), (-100, -50), (50, 100), (150, 250)]  # Regiones simétricas
SNR_COLORMAP = "viridis"                    # Mapa de colores para waterfalls
SNR_HIGHLIGHT_COLOR = "red"                 # Color para resaltar detecciones

# --- 5. CONFIGURACIONES DE SLICE AVANZADAS (POSIBLEMENTE REDUNDANTES) ---
# UBICACIÓN: Se usan en slice_len_utils.py
# PROPÓSITO: Límites de seguridad para cálculo automático
# IMPACTO: Solo seguridad, valores por defecto suelen funcionar
SLICE_LEN_MIN: int = 32                      # Límite inferior de seguridad para el cálculo automático de SLICE_LEN
SLICE_LEN_MAX: int = 2048                    # Límite superior de seguridad para el cálculo automático de SLICE_LEN

# --- 6. CONFIGURACIONES DE CHUNKING AVANZADAS (POSIBLEMENTE INNECESARIAS) ---
# =============================================================================
# RECOMENDACIONES PARA ELIMINACIÓN:
# =============================================================================
"""
CONFIGURACIONES QUE SE PUEDEN ELIMINAR FÁCILMENTE:

1. ALTA PRIORIDAD PARA ELIMINAR (Solo estética):
   - SNR_OFF_REGIONS, SNR_COLORMAP, SNR_HIGHLIGHT_COLOR
   - DM_PLOT_MARGIN_FACTOR, DM_PLOT_MIN_RANGE, DM_PLOT_MAX_RANGE
   - DM_RANGE_DEFAULT_VISUALIZATION
   - SNR_SHOW_PEAK_LINES

2. MEDIA PRIORIDAD PARA ELIMINAR (Funcionalidad avanzada poco usada):
   - DM_DYNAMIC_RANGE_ENABLE, DM_RANGE_ADAPTIVE, DM_RANGE_FACTOR
   - DM_RANGE_MIN_WIDTH, DM_RANGE_MAX_WIDTH
   - DM_DYNAMIC_RANGE_ENABLE, DM_RANGE_ADAPTIVE, DM_RANGE_FACTOR
   - DM_RANGE_MIN_WIDTH, DM_RANGE_MAX_WIDTH

3. BAJA PRIORIDAD PARA ELIMINAR (Podrían ser útiles):
   - SLICE_LEN_MIN, SLICE_LEN_MAX (seguridad)

CONFIGURACIONES ESENCIALES QUE NO SE DEBEN ELIMINAR:
- FRB_TARGETS, DATA_DIR, RESULTS_DIR
- SLICE_DURATION_MS (configura duración temporal)
- DM_min, DM_max (rango de dispersión)
- DET_PROB, CLASS_PROB, SNR_THRESH (umbrales de detección)
- USE_MULTI_BAND
 - MODEL_NAME, MODEL_PATH, CLASS_MODEL_NAME, CLASS_MODEL_PATH
 - DEBUG_FREQUENCY_ORDER
"""
