"""
Configuración del Sistema para el Pipeline de Detección de FRB
==============================================================

Este archivo contiene las configuraciones del sistema que se configuran automáticamente
o que son específicas del funcionamiento interno del pipeline.

IMPORTANTE: 
- NO modifique este archivo directamente
- Para configurar parámetros del usuario, modifique user_config.py
- Este archivo mantiene compatibilidad con el código existente
"""

from __future__ import annotations

# Importar configuraciones del usuario
try:
    from .user_config import *
except ImportError:
    # Si se ejecuta como script independiente
    from user_config import *

import numpy as np
from pathlib import Path
try:
    import torch
except ImportError:  # Si torch no está instalado, lo dejamos como None
    torch = None

# =============================================================================
# CONFIGURACIÓN DE MODELOS Y SISTEMA
# =============================================================================

# Modelo de detección
MODEL_NAME = "resnet50"                     # Arquitectura del modelo de detección
MODEL_PATH = Path(f"./models/cent_{MODEL_NAME}.pth")  # Ruta al modelo entrenado

# Modelo de clasificación binaria
CLASS_MODEL_NAME = "resnet50"               # Arquitectura del modelo de clasificación
CLASS_MODEL_PATH = Path(f"./models/class_{CLASS_MODEL_NAME}.pth")  # Ruta al modelo

# Dispositivo de cómputo
if torch is not None:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = "cpu"

# =============================================================================
# CONFIGURACIÓN AUTOMÁTICA - Variables que se configuran automáticamente
# =============================================================================

# Metadatos del archivo (se configuran automáticamente)
FREQ: np.ndarray | None = None              # Array de frecuencias de observación (MHz)
FREQ_RESO: int = 0                          # Resolución de frecuencia (canales)
TIME_RESO: float = 0.0                      # Resolución temporal (segundos)
FILE_LENG: int = 0                          # Longitud del archivo (muestras)

# Configuración de Slice Temporal (calculada dinámicamente)
SLICE_LEN: int = 512                        # Número de muestras por slice (calculado automáticamente desde SLICE_DURATION_MS)

# Configuración de datos
DATA_NEEDS_REVERSAL: bool = False           # Invertir eje de frecuencia si es necesario

# =============================================================================
# CONFIGURACIONES AVANZADAS DEL SISTEMA
# =============================================================================

# Configuraciones de rango DM dinámico (sistema)
DM_RANGE_ADAPTIVE: bool = False             # True = adaptar según confianza, False = factor fijo
DM_RANGE_MIN_WIDTH: float = 80.0            # Ancho mínimo del rango DM en pc cm⁻³
DM_RANGE_MAX_WIDTH: float = 300.0           # Ancho máximo del rango DM en pc cm⁻³

# Configuraciones de plots estéticas (sistema)
DM_PLOT_MARGIN_FACTOR: float = 0.25         # Margen adicional para evitar bordes (25%)
DM_PLOT_MIN_RANGE: float = 120.0            # Rango mínimo del plot en pc cm⁻³
DM_PLOT_MAX_RANGE: float = 400.0            # Rango máximo del plot en pc cm⁻³
DM_PLOT_DEFAULT_RANGE: float = 250.0        # Rango por defecto sin candidatos
DM_RANGE_DEFAULT_VISUALIZATION: str = "detailed"  # Tipo de visualización por defecto

# Configuraciones de SNR (sistema)
SNR_OFF_REGIONS = [(-250, -150), (-100, -50), (50, 100), (150, 250)]  # Regiones simétricas
SNR_HIGHLIGHT_COLOR = "red"                 # Color para resaltar detecciones

# Configuraciones de slice avanzadas (sistema)
SLICE_LEN_MIN: int = 32                     # Límite inferior de seguridad para el cálculo automático de SLICE_LEN
SLICE_LEN_MAX: int = 2048                   # Límite superior de seguridad para el cálculo automático de SLICE_LEN

# =============================================================================
# CONFIGURACIONES AVANZADAS DE VISUALIZACIÓN (SISTEMA)
# =============================================================================

# Configuraciones de rango DM dinámico (solo para visualización)
DM_DYNAMIC_RANGE_ENABLE: bool = False       # True = ajustar automáticamente rango DM en plots
DM_RANGE_FACTOR: float = 0.3                # Factor de rango para plots (±30% del DM óptimo)

# Configuraciones de visualización SNR
SNR_SHOW_PEAK_LINES: bool = False           # True = mostrar líneas rojas del SNR peak en plots
SNR_COLORMAP = "viridis"                    # Mapa de colores para waterfalls

# =============================================================================
# FUNCIONES DE CONFIGURACIÓN DE BANDAS
# =============================================================================

def get_band_configs():
    """Retorna la configuración de bandas según USE_MULTI_BAND"""
    return [
        (0, "fullband", "Full Band"),
        (1, "lowband", "Low Band"),
        (2, "highband", "High Band"),
    ] if USE_MULTI_BAND else [(0, "fullband", "Full Band")]