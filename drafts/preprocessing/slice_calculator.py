"""
Módulo de Cálculo de Slices - Cálculo Dinámico de Segmentación Temporal
======================================================================

Este módulo se encarga de calcular dinámicamente los tamaños óptimos
de slices y chunks para el procesamiento de datos astronómicos.

Responsabilidades:
- Calcular SLICE_LEN basado en SLICE_DURATION_MS y metadatos
- Calcular tamaños óptimos de chunks para procesamiento
- Optimizar uso de memoria del sistema
- Validar parámetros de procesamiento
- Actualizar dinámicamente configuraciones

Para astrónomos:
- Usar calculate_slice_len_from_duration() para calcular slices
- Usar calculate_optimal_chunk_size() para optimizar memoria
- Usar get_processing_parameters() para obtener configuración completa
- Los parámetros se configuran en config.SLICE_DURATION_MS
"""

import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np
import psutil
import os

from .. import config

logger = logging.getLogger(__name__)


def calculate_slice_len_from_duration() -> Tuple[int, float]:
    """
    Calcula SLICE_LEN dinámicamente basado en SLICE_DURATION_MS y metadatos del archivo.
    
    Fórmula inversa: SLICE_LEN = round(SLICE_DURATION_MS / (TIME_RESO × DOWN_TIME_RATE × 1000))
    
    Returns
    -------
    Tuple[int, float]
        (slice_len_calculado, duracion_real_ms)
    """
    if config.TIME_RESO <= 0:
        logger.warning("TIME_RESO no está configurado, usando SLICE_LEN_MIN")
        return config.SLICE_LEN_MIN, config.SLICE_DURATION_MS
    
    # SLICE_LEN se calcula para datos YA decimados
    # Por lo tanto, usar TIME_RESO * DOWN_TIME_RATE
    target_duration_s = config.SLICE_DURATION_MS / 1000.0
    calculated_slice_len = round(target_duration_s / (config.TIME_RESO * config.DOWN_TIME_RATE))
    
    # Aplicar límites mín/máx
    slice_len = max(config.SLICE_LEN_MIN, min(config.SLICE_LEN_MAX, calculated_slice_len))
    
    # Calcular duración real obtenida (para datos decimados)
    real_duration_s = slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
    real_duration_ms = real_duration_s * 1000.0
    
    # Actualizar config.SLICE_LEN con el valor calculado
    config.SLICE_LEN = slice_len
    
    # Solo mostrar información esencial en INFO
    if abs(real_duration_ms - config.SLICE_DURATION_MS) > 5.0:
        logger.warning(f"Diferencia significativa entre objetivo ({config.SLICE_DURATION_MS:.1f} ms) "
                      f"y obtenido ({real_duration_ms:.1f} ms)")
    
    return slice_len, real_duration_ms


def calculate_optimal_chunk_size() -> int:
    """
    Calcula el tamaño óptimo de chunk basado en SLICE_DURATION_MS y optimización de memoria.
    
    Estrategia:
    1. Calcular cuántos slices caben en un chunk razonable (ej: 100-500 slices)
    2. Considerar memoria disponible del sistema
    3. Optimizar para evitar fragmentación de memoria
    4. Asegurar que el chunk sea múltiplo del slice_len
    
    Returns
    -------
    int
        Tamaño óptimo de chunk en muestras
    """
    if config.TIME_RESO <= 0 or config.FILE_LENG <= 0:
        logger.warning("Metadatos del archivo no disponibles, usando chunk por defecto")
        return 2_097_152  # 2MB por defecto
    
    # Obtener slice_len actualizado
    slice_len, _ = calculate_slice_len_from_duration()
    
    # Estrategia 1: Chunk que contenga ~200-300 slices (balance entre memoria y eficiencia)
    target_slices_per_chunk = 250 # numero de slices por chunk
    chunk_samples_from_slices = slice_len * target_slices_per_chunk # muestras por chunk segun numero de slices
    
    # Estrategia 2: Basado en duración de chunk (ej: 30-60 segundos)
    target_chunk_duration_sec = 45.0  # duracion de chunk en segundos
    chunk_samples_from_duration = int(target_chunk_duration_sec / (config.TIME_RESO * config.DOWN_TIME_RATE)) # muestras por chunk segun duracion
    
    # Estrategia 3: Basado en memoria disponible
    try:
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        # Usar máximo 25% de memoria disponible para un chunk
        max_memory_per_chunk_gb = available_memory_gb * 0.25 # memoria disponible por chunk
        
        # Estimar memoria por muestra (asumiendo float32 = 4 bytes)
        bytes_per_sample = 4  # float32
        max_samples_from_memory = int((max_memory_per_chunk_gb * 1024**3) / bytes_per_sample) # muestras por chunk segun memoria disponible
        
        # Ajustar por número de canales de frecuencia
        if config.FREQ_RESO > 0:
            max_samples_from_memory = max_samples_from_memory // config.FREQ_RESO
        
        chunk_samples_from_memory = max_samples_from_memory
    except Exception as e:
        logger.warning(f"No se pudo obtener información de memoria: {e}")
        chunk_samples_from_memory = 10_000_000  # 10M muestras por defecto
    
    # Estrategia 4: Límites prácticos
    min_chunk_samples = slice_len * 50   # Mínimo 50 slices por chunk
    max_chunk_samples = slice_len * 1000 # Máximo 1000 slices por chunk
    
    # Seleccionar el valor más conservador (menor) entre las estrategias
    chunk_samples = min(
        chunk_samples_from_slices,
        chunk_samples_from_duration,
        chunk_samples_from_memory,
        max_chunk_samples
    )
    
    # Asegurar que sea al menos el mínimo
    chunk_samples = max(chunk_samples, min_chunk_samples)
    
    # Asegurar que sea múltiplo del slice_len para evitar fragmentación
    chunk_samples = (chunk_samples // slice_len) * slice_len
    
    logger.debug(f"Chunk size calculado: {chunk_samples} muestras ({chunk_samples/slice_len:.1f} slices)")
    return chunk_samples


def get_processing_parameters() -> Dict[str, Any]:
    """
    Obtiene todos los parámetros de procesamiento calculados dinámicamente.
    
    Returns
    -------
    Dict[str, Any]
        Diccionario con todos los parámetros de procesamiento
    """
    # Calcular slice_len y chunk_size
    slice_len, real_duration_ms = calculate_slice_len_from_duration()
    chunk_size = calculate_optimal_chunk_size()
    
    # Calcular número de slices por chunk
    slices_per_chunk = chunk_size // slice_len
    
    # Calcular duración del chunk
    chunk_duration_sec = chunk_size * config.TIME_RESO * config.DOWN_TIME_RATE
    chunk_duration_ms = chunk_duration_sec * 1000.0
    
    # Calcular número total de chunks
    total_chunks = (config.FILE_LENG + chunk_size - 1) // chunk_size
    
    return {
        "slice_len": slice_len,
        "slice_duration_ms": real_duration_ms,
        "chunk_size": chunk_size,
        "chunk_duration_ms": chunk_duration_ms,
        "slices_per_chunk": slices_per_chunk,
        "total_chunks": total_chunks,
        "time_resolution": config.TIME_RESO * config.DOWN_TIME_RATE,
        "frequency_resolution": config.FREQ_RESO // config.DOWN_FREQ_RATE,
        "downsampling_time_rate": config.DOWN_TIME_RATE,
        "downsampling_freq_rate": config.DOWN_FREQ_RATE
    }


def update_slice_len_dynamic():
    """
    Actualiza dinámicamente SLICE_LEN basado en metadatos del archivo.
    
    Esta función se llama automáticamente cuando se cargan nuevos datos
    para asegurar que SLICE_LEN sea consistente con los metadatos.
    """
    try:
        old_slice_len = config.SLICE_LEN
        new_slice_len, real_duration_ms = calculate_slice_len_from_duration()
        
        if old_slice_len != new_slice_len:
            logger.info(f"SLICE_LEN actualizado: {old_slice_len} → {new_slice_len} "
                       f"(duración: {real_duration_ms:.1f} ms)")
        else:
            logger.debug(f"SLICE_LEN sin cambios: {new_slice_len} "
                        f"(duración: {real_duration_ms:.1f} ms)")
            
    except Exception as e:
        logger.error(f"Error al actualizar SLICE_LEN dinámicamente: {e}")
        # Mantener valor anterior si hay error
        pass


def validate_processing_parameters(parameters: Dict[str, Any]) -> bool:
    """
    Valida que los parámetros de procesamiento sean correctos.
    
    Parameters
    ----------
    parameters : Dict[str, Any]
        Diccionario con parámetros de procesamiento
        
    Returns
    -------
    bool
        True si los parámetros son válidos, False en caso contrario
    """
    try:
        # Validar slice_len
        if parameters.get("slice_len", 0) <= 0:
            logger.error("slice_len debe ser mayor que 0")
            return False
        
        if parameters.get("slice_len", 0) < config.SLICE_LEN_MIN:
            logger.error(f"slice_len ({parameters['slice_len']}) menor que SLICE_LEN_MIN ({config.SLICE_LEN_MIN})")
            return False
        
        if parameters.get("slice_len", 0) > config.SLICE_LEN_MAX:
            logger.error(f"slice_len ({parameters['slice_len']}) mayor que SLICE_LEN_MAX ({config.SLICE_LEN_MAX})")
            return False
        
        # Validar chunk_size
        if parameters.get("chunk_size", 0) <= 0:
            logger.error("chunk_size debe ser mayor que 0")
            return False
        
        # Validar que chunk_size sea múltiplo de slice_len
        if parameters.get("chunk_size", 0) % parameters.get("slice_len", 1) != 0:
            logger.error("chunk_size debe ser múltiplo de slice_len")
            return False
        
        # Validar duración del slice
        slice_duration_ms = parameters.get("slice_duration_ms", 0)
        if slice_duration_ms <= 0:
            logger.error("slice_duration_ms debe ser mayor que 0")
            return False
        
        # Validar que la duración esté dentro de límites razonables
        if slice_duration_ms < 1.0 or slice_duration_ms > 1000.0:
            logger.warning(f"slice_duration_ms ({slice_duration_ms:.1f} ms) fuera del rango típico (1-1000 ms)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error al validar parámetros de procesamiento: {e}")
        return False


def get_memory_usage_info() -> Dict[str, Any]:
    """
    Obtiene información sobre el uso de memoria del sistema.
    
    Returns
    -------
    Dict[str, Any]
        Información sobre memoria disponible y uso
    """
    try:
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent_used": memory.percent,
            "recommended_chunk_memory_gb": memory.available * 0.25 / (1024**3)
        }
    except Exception as e:
        logger.warning(f"No se pudo obtener información de memoria: {e}")
        return {
            "error": str(e),
            "recommended_chunk_memory_gb": 1.0  # Valor por defecto
        }