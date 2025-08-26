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
    from .user_config import (
        DATA_DIR,
        DEBUG_FREQUENCY_ORDER,
        DM_max,
        DM_min,
        DOWN_FREQ_RATE,
        DOWN_TIME_RATE,
        DET_PROB,
        CLASS_PROB,
        FORCE_PLOTS,
        FRB_TARGETS,
        RESULTS_DIR,
        SLICE_DURATION_MS,
        SNR_THRESH,
        USE_MULTI_BAND,
        SAVE_ONLY_BURST
    )
except ImportError:
    # Si se ejecuta como script independiente
    from user_config import (
        DATA_DIR,
        DEBUG_FREQUENCY_ORDER,
        DM_max,
        DM_min,
        DOWN_FREQ_RATE,
        DOWN_TIME_RATE,
        DET_PROB,
        CLASS_PROB,
        FORCE_PLOTS,
        FRB_TARGETS,
        RESULTS_DIR,
        SLICE_DURATION_MS,
        SNR_THRESH,
        USE_MULTI_BAND,
        SAVE_ONLY_BURST
    )

# Standard library imports
from pathlib import Path

# Third-party imports
import numpy as np

# Optional third-party imports
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

# Configuraciones de datos
DATA_NEEDS_REVERSAL: bool = False           # Invertir eje de frecuencia si es necesario

# Límite de muestras para archivos grandes
MAX_SAMPLES_LIMIT: int = 10_000_000         # Límite máximo de muestras para archivos .fil (10M muestras)

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
# CONFIGURACIONES AVANZADAS DE CHUNKING Y SLICING (SISTEMA)
# =============================================================================

# Sistema de chunking contiguo y slicing ajustado a divisores
USE_PLANNED_CHUNKING: bool = True           # Usar planificador para tamaños de chunk y slices uniformes

# Límites de memoria para chunking
MAX_CHUNK_BYTES: int | None = None          # Límite duro en bytes para un chunk (recomendado); None para usar fracción de RAM
MAX_RAM_FRACTION: float = 0.25              # Fracción de RAM disponible a utilizar si MAX_CHUNK_BYTES es None
OVERHEAD_FACTOR: float = 1.3                # Factor de overhead por copias/buffers temporales

# Tolerancias y límites de slicing
TIME_TOL_MS: float = 0.1                    # Tolerancia temporal al ajustar la duración objetivo del slice
MAX_SLICE_COUNT: int = 5000                 # Límite superior de slices por chunk

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
# CONFIGURACIONES AVANZADAS DE LOGGING Y DEBUG (SISTEMA)
# =============================================================================

# Configuración de logging del sistema
LOG_LEVEL: str = "INFO"                     # Nivel de logging: DEBUG, INFO, WARNING, ERROR
LOG_COLORS: bool = True                     # Usar colores en la consola
LOG_FILE: bool = False                      # Guardar logs en archivo
GPU_VERBOSE: bool = False                   # Mostrar mensajes detallados de GPU
SHOW_PROGRESS: bool = True                  # Mostrar barras de progreso

# =============================================================================
# FLAGS AVANZADOS PARA MITIGAR ARTEFACTOS Y MEJORAR ESTABILIDAD (SISTEMA)
# =============================================================================

# Pre-whitening por canal antes de construir el cubo DM–tiempo (z-score por canal)
PREWHITEN_BEFORE_DM: bool = True

# Sombrear visualmente la cola inválida en los waterfalls cuando no hay solapamiento
SHADE_INVALID_TAIL: bool = True

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


# =============================================================================
# VALIDACIÓN DE CONFIGURACIÓN CON MENSAJES INFORMATIVOS
# =============================================================================

def validate_configuration():
    """
    Valida la configuración del sistema y genera mensajes de error informativos.
    
    Esta función verifica que todos los parámetros críticos estén configurados
    correctamente antes de ejecutar el pipeline.
    """
    errors = []
    
    # Validar parámetros de frecuencia
    if FREQ_RESO <= 0:
        errors.append(
            f"FREQ_RESO={FREQ_RESO} es inválido\n"
            f"  → FREQ_RESO debe ser > 0 para procesar datos de frecuencia\n"
            f"  → Este valor se extrae del header del archivo FITS/FIL\n"
            f"  → Recomendación: Verificar que el archivo de datos sea válido"
        )
    
    if TIME_RESO <= 0:
        errors.append(
            f"TIME_RESO={TIME_RESO} es inválido\n"
            f"  → TIME_RESO debe ser > 0 para procesar datos temporales\n"
            f"  → Este valor se extrae del header del archivo FITS/FIL\n"
            f"  → Recomendación: Verificar que el archivo de datos sea válido"
        )
    
    if FILE_LENG <= 0:
        errors.append(
            f"FILE_LENG={FILE_LENG} es inválido\n"
            f"  → FILE_LENG debe ser > 0 para procesar datos\n"
            f"  → Este valor indica el número total de muestras temporales\n"
            f"  → Recomendación: Verificar que el archivo de datos sea válido"
        )
    
    # Validar configuración de slice
    if SLICE_LEN < SLICE_LEN_MIN or SLICE_LEN > SLICE_LEN_MAX:
        errors.append(
            f"SLICE_LEN={SLICE_LEN} está fuera del rango válido [{SLICE_LEN_MIN}, {SLICE_LEN_MAX}]\n"
            f"  → SLICE_LEN debe estar entre {SLICE_LEN_MIN} y {SLICE_LEN_MAX} muestras\n"
            f"  → Este valor se calcula automáticamente desde SLICE_DURATION_MS\n"
            f"  → Recomendación: Ajustar SLICE_DURATION_MS en user_config.py"
        )
    
    # Validar rangos DM
    if DM_RANGE_MIN_WIDTH <= 0 or DM_RANGE_MAX_WIDTH <= 0:
        errors.append(
            f"Rangos DM inválidos: MIN={DM_RANGE_MIN_WIDTH}, MAX={DM_RANGE_MAX_WIDTH}\n"
            f"  → Ambos valores deben ser > 0\n"
            f"  → Estos valores definen los límites del rango DM dinámico\n"
            f"  → Recomendación: Verificar configuración de rangos DM"
        )
    
    if DM_RANGE_MIN_WIDTH >= DM_RANGE_MAX_WIDTH:
        errors.append(
            f"Rangos DM inconsistentes: MIN={DM_RANGE_MIN_WIDTH} >= MAX={DM_RANGE_MAX_WIDTH}\n"
            f"  → DM_RANGE_MIN_WIDTH debe ser < DM_RANGE_MAX_WIDTH\n"
            f"  → Recomendación: Ajustar los valores de rango DM"
        )
    
    # Validar configuración de modelos
    if not MODEL_PATH.exists():
        errors.append(
            f"Modelo de detección no encontrado: {MODEL_PATH}\n"
            f"  → El archivo del modelo no existe en la ruta especificada\n"
            f"  → Verificar que el modelo esté entrenado y guardado\n"
            f"  → Recomendación: Entrenar el modelo o verificar la ruta"
        )
    
    if not CLASS_MODEL_PATH.exists():
        errors.append(
            f"Modelo de clasificación no encontrado: {CLASS_MODEL_PATH}\n"
            f"  → El archivo del modelo no existe en la ruta especificada\n"
            f"  → Verificar que el modelo esté entrenado y guardado\n"
            f"  → Recomendación: Entrenar el modelo o verificar la ruta"
        )
    
    # Validar configuración de dispositivo
    if torch is not None and torch.cuda.is_available():
        try:
            # Verificar que CUDA funcione correctamente
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
        except Exception as e:
            errors.append(
                f"Error con CUDA: {e}\n"
                f"  → CUDA está disponible pero no funciona correctamente\n"
                f"  → Verificar drivers de NVIDIA y instalación de PyTorch\n"
                f"  → Recomendación: Reinstalar PyTorch con soporte CUDA o usar CPU"
            )
    
    # Si hay errores, lanzar excepción con todos los errores
    if errors:
        error_message = "Configuración del sistema inválida:\n\n"
        for i, error in enumerate(errors, 1):
            error_message += f"{i}. {error}\n\n"
        error_message += "Corrija estos errores antes de ejecutar el pipeline."
        
        raise ValueError(error_message)
    
    return True


def check_model_files():
    """
    Verifica que los archivos de modelo existan y sean accesibles.
    
    Returns:
        dict: Diccionario con el estado de cada modelo
    """
    model_status = {}
    
    # Verificar modelo de detección
    if MODEL_PATH.exists():
        try:
            # Intentar cargar el modelo para verificar que no esté corrupto
            if torch is not None:
                state = torch.load(MODEL_PATH, map_location='cpu')
                model_status['detection'] = {
                    'exists': True,
                    'size_mb': MODEL_PATH.stat().st_size / (1024 * 1024),
                    'state_dict_keys': len(state.keys()) if isinstance(state, dict) else 0
                }
            else:
                model_status['detection'] = {
                    'exists': True,
                    'size_mb': MODEL_PATH.stat().st_size / (1024 * 1024),
                    'note': 'PyTorch no disponible para verificación completa'
                }
        except Exception as e:
            model_status['detection'] = {
                'exists': True,
                'error': f"Modelo corrupto: {e}",
                'recommendation': 'Reentrenar el modelo'
            }
    else:
        model_status['detection'] = {
            'exists': False,
            'error': f"Archivo no encontrado: {MODEL_PATH}",
            'recommendation': 'Entrenar el modelo o verificar la ruta'
        }
    
    # Verificar modelo de clasificación
    if CLASS_MODEL_PATH.exists():
        try:
            if torch is not None:
                state = torch.load(CLASS_MODEL_PATH, map_location='cpu')
                model_status['classification'] = {
                    'exists': True,
                    'size_mb': CLASS_MODEL_PATH.stat().st_size / (1024 * 1024),
                    'state_dict_keys': len(state.keys()) if isinstance(state, dict) else 0
                }
            else:
                model_status['classification'] = {
                    'exists': True,
                    'size_mb': CLASS_MODEL_PATH.stat().st_size / (1024 * 1024),
                    'note': 'PyTorch no disponible para verificación completa'
                }
        except Exception as e:
            model_status['classification'] = {
                'exists': True,
                'error': f"Modelo corrupto: {e}",
                'recommendation': 'Reentrenar el modelo'
            }
    else:
        model_status['classification'] = {
            'exists': False,
            'error': f"Archivo no encontrado: {CLASS_MODEL_PATH}",
            'recommendation': 'Entrenar el modelo o verificar la ruta'
        }
    
    return model_status