"""Slice length calculator for FRB pipeline - dynamically calculates optimal temporal segmentation."""
import logging
from typing import Tuple, Optional
import numpy as np
import psutil
import os

from .. import config

logger = logging.getLogger(__name__)


def calculate_slice_len_from_duration() -> Tuple[int, float]:
    """
    Calcula SLICE_LEN dinámicamente basado en SLICE_DURATION_MS y metadatos del archivo.
    
    Fórmula inversa: SLICE_LEN = round(SLICE_DURATION_MS / (TIME_RESO × DOWN_TIME_RATE × 1000))
    
    Returns:
        Tuple[int, float]: (slice_len_calculado, duracion_real_ms)
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
    
    Returns:
        int: Tamaño óptimo de chunk en muestras
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
    candidate_chunk_sizes = [
        chunk_samples_from_slices,
        chunk_samples_from_duration,
        chunk_samples_from_memory,
        max_chunk_samples
    ]
    
    # Filtrar valores válidos y seleccionar el menor
    valid_chunk_sizes = [size for size in candidate_chunk_sizes if min_chunk_samples <= size <= max_chunk_samples]
    
    if not valid_chunk_sizes:
        logger.warning("No se encontraron tamaños de chunk válidos, usando valor por defecto")
        return slice_len * 200  # 200 slices por defecto
    
    optimal_chunk_size = min(valid_chunk_sizes)
    
    # Asegurar que sea múltiplo del slice_len para evitar fragmentación
    slices_in_chunk = optimal_chunk_size // slice_len
    optimal_chunk_size = slices_in_chunk * slice_len
    
    # Log informativo
    chunk_duration_sec = optimal_chunk_size * config.TIME_RESO * config.DOWN_TIME_RATE
    slices_per_chunk = optimal_chunk_size // slice_len
    
    logger.info(f"Chunk óptimo calculado: {optimal_chunk_size:,} muestras "
                f"({chunk_duration_sec:.1f}s, {slices_per_chunk} slices)")
    
    return optimal_chunk_size


def get_processing_parameters() -> dict:
    """
    Calcula todos los parámetros de procesamiento basados en SLICE_DURATION_MS.
    
    Returns:
        dict: Diccionario con todos los parámetros calculados
    """
    # Calcular slice_len
    slice_len, real_duration_ms = calculate_slice_len_from_duration()
    
    # Calcular chunk_size óptimo
    chunk_samples = calculate_optimal_chunk_size()
    
    # Calcular parámetros adicionales
    if config.FILE_LENG > 0:
        total_slices = (config.FILE_LENG + slice_len - 1) // slice_len
        total_chunks = (config.FILE_LENG + chunk_samples - 1) // chunk_samples
        total_duration_sec = config.FILE_LENG * config.TIME_RESO
    else:
        total_slices = 0
        total_chunks = 0
        total_duration_sec = 0
    
    chunk_duration_sec = chunk_samples * config.TIME_RESO * config.DOWN_TIME_RATE
    slices_per_chunk = chunk_samples // slice_len
    
    parameters = {
        'slice_len': slice_len,
        'slice_duration_ms': real_duration_ms,
        'chunk_samples': chunk_samples,
        'chunk_duration_sec': chunk_duration_sec,
        'slices_per_chunk': slices_per_chunk,
        'total_slices': total_slices,
        'total_chunks': total_chunks,
        'total_duration_sec': total_duration_sec,
        'memory_optimized': True
    }
    
    return parameters


def update_slice_len_dynamic():
    """
    Actualiza config.SLICE_LEN basado en SLICE_DURATION_MS.
    Debe llamarse después de cargar metadatos del archivo.
    """
    slice_len, real_duration_ms = calculate_slice_len_from_duration()
    
    # Usar el logger global si está disponible
    try:
        from ..logging.logging_config import get_global_logger
        global_logger = get_global_logger()
        global_logger.slice_config({
            'target_ms': config.SLICE_DURATION_MS,
            'slice_len': slice_len,
            'real_ms': real_duration_ms
        })
    except ImportError:
        # Fallback al logger local
        logger.info(f"Slice configurado: {slice_len} muestras = {real_duration_ms:.1f} ms")
    
    return slice_len, real_duration_ms


def validate_processing_parameters(parameters: dict) -> bool:
    """
    Valida que los parámetros calculados sean razonables.
    
    Args:
        parameters: Diccionario con parámetros de procesamiento
        
    Returns:
        bool: True si los parámetros son válidos
    """
    errors = []
    
    # Validar slice_len
    if parameters['slice_len'] < config.SLICE_LEN_MIN:
        errors.append(f"slice_len ({parameters['slice_len']}) < mínimo ({config.SLICE_LEN_MIN})")
    
    if parameters['slice_len'] > config.SLICE_LEN_MAX:
        errors.append(f"slice_len ({parameters['slice_len']}) > máximo ({config.SLICE_LEN_MAX})")
    
    # Validar chunk_samples
    if parameters['chunk_samples'] < parameters['slice_len'] * 10:
        errors.append(f"chunk_samples ({parameters['chunk_samples']}) muy pequeño para {parameters['slice_len']} slice_len")
    
    if parameters['chunk_samples'] > 50_000_000:  # 50M muestras máximo
        errors.append(f"chunk_samples ({parameters['chunk_samples']}) muy grande")
    
    # Validar slices_per_chunk
    if parameters['slices_per_chunk'] < 10:
        errors.append(f"slices_per_chunk ({parameters['slices_per_chunk']}) muy pequeño")
    
    if parameters['slices_per_chunk'] > 2000:
        errors.append(f"slices_per_chunk ({parameters['slices_per_chunk']}) muy grande")
    
    if errors:
        logger.error("Parámetros de procesamiento inválidos:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    return True

