"""Parámetros centralizados del pipeline para evitar duplicación de lógica."""
from __future__ import annotations

import numpy as np
from ..config import config
from ..preprocessing.slice_len_calculator import update_slice_len_dynamic


def calculate_frequency_downsampled() -> np.ndarray:
    """Calcula las frecuencias decimadas del pipeline.
    
    Returns:
        np.ndarray: Array de frecuencias decimadas
    """
    return np.mean(
        config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE),
        axis=1,
    )


def calculate_dm_height() -> int:
    """Calcula la altura del cubo DM-tiempo.
    
    Returns:
        int: Número de valores DM (height)
    """
    return config.DM_max - config.DM_min + 1


def calculate_width_total() -> int:
    """Calcula el ancho total decimado del archivo.
    
    Returns:
        int: Ancho total en muestras decimadas
    """
    return config.FILE_LENG // config.DOWN_TIME_RATE


def calculate_slice_parameters() -> tuple[int, float]:
    """Calcula los parámetros de slicing dinámicamente.
    
    Returns:
        tuple[int, float]: (slice_len, real_duration_ms)
    """
    return update_slice_len_dynamic()


def calculate_time_slice(width_total: int, slice_len: int) -> int:
    """Calcula el número de slices para un ancho dado.
    
    Args:
        width_total: Ancho total del bloque
        slice_len: Longitud de cada slice
        
    Returns:
        int: Número de slices
    """
    return (width_total + slice_len - 1) // slice_len


def calculate_slice_duration(slice_len: int) -> float:
    """Calcula la duración de un slice en segundos.
    
    Args:
        slice_len: Longitud del slice en muestras
        
    Returns:
        float: Duración en segundos
    """
    return slice_len * config.TIME_RESO * config.DOWN_TIME_RATE


def get_pipeline_parameters() -> dict:
    """Obtiene todos los parámetros del pipeline en un solo lugar.
    
    Returns:
        dict: Diccionario con todos los parámetros calculados
    """
    freq_down = calculate_frequency_downsampled()
    height = calculate_dm_height()
    width_total = calculate_width_total()
    slice_len, real_duration_ms = calculate_slice_parameters()
    time_slice = calculate_time_slice(width_total, slice_len)
    slice_duration = calculate_slice_duration(slice_len)
    
    return {
        'freq_down': freq_down,
        'height': height,
        'width_total': width_total,
        'slice_len': slice_len,
        'real_duration_ms': real_duration_ms,
        'time_slice': time_slice,
        'slice_duration': slice_duration,
    }


def calculate_overlap_decimated(overlap_left_raw: int, overlap_right_raw: int) -> tuple[int, int]:
    """Calcula el solapamiento en muestras decimadas.
    
    Args:
        overlap_left_raw: Solapamiento izquierdo en muestras originales
        overlap_right_raw: Solapamiento derecho en muestras originales
        
    Returns:
        tuple[int, int]: (overlap_left_ds, overlap_right_ds)
    """
    R = int(config.DOWN_TIME_RATE)
    overlap_left_ds = (overlap_left_raw + R - 1) // R
    overlap_right_ds = (overlap_right_raw + R - 1) // R
    return overlap_left_ds, overlap_right_ds


def calculate_absolute_slice_time(chunk_start_time_sec: float, start_idx: int, dt_ds: float) -> float:
    """Calcula el tiempo absoluto de inicio de un slice.
    
    Args:
        chunk_start_time_sec: Tiempo de inicio del chunk
        start_idx: Índice de inicio del slice
        dt_ds: Resolución temporal decimada
        
    Returns:
        float: Tiempo absoluto del slice en segundos
    """
    return chunk_start_time_sec + (start_idx * dt_ds)
