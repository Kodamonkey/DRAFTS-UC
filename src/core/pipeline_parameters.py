# This module calculates derived parameters for pipeline execution.

"""Parámetros centralizados del pipeline para evitar duplicación de lógica."""
from __future__ import annotations

import numpy as np
from ..config import config
from ..preprocessing.slice_len_calculator import update_slice_len_dynamic


# This function calculates the downsampled frequency array.
def calculate_frequency_downsampled() -> np.ndarray:
    """Calcula las frecuencias decimadas del pipeline."""

    if config.FREQ is None or getattr(config, "FREQ_RESO", 0) <= 0:
        raise ValueError("Los metadatos de frecuencia no han sido cargados")

    down_rate = int(getattr(config, "DOWN_FREQ_RATE", 1))
    if down_rate <= 0:
        raise ValueError("DOWN_FREQ_RATE debe ser mayor que cero")

    total_channels = len(config.FREQ)
    usable = total_channels - (total_channels % down_rate)
    if usable == 0:
        raise ValueError("DOWN_FREQ_RATE es mayor que el número de canales disponibles")

    trimmed = config.FREQ[:usable]
    return trimmed.reshape(-1, down_rate).mean(axis=1)


# This function calculates DM height.
def calculate_dm_height() -> int:
    """Calcula la altura del cubo DM-tiempo."""

    dm_max = int(getattr(config, "DM_max", 0))
    dm_min = int(getattr(config, "DM_min", 0))
    return max(0, dm_max - dm_min + 1)


# This function calculates the total decimated width.
def calculate_width_total(total_samples: int | None = None) -> int:
    """Calcula el ancho total decimado del archivo."""

    samples = int(total_samples) if total_samples is not None else int(getattr(config, "FILE_LENG", 0))
    down_rate = int(getattr(config, "DOWN_TIME_RATE", 1))
    if samples <= 0 or down_rate <= 0:
        return 0
    return samples // down_rate


# This function calculates slice parameters.
def calculate_slice_parameters() -> tuple[int, float]:
    """Calcula los parámetros de slicing dinámicamente.
    
    Returns:
        tuple[int, float]: (slice_len, real_duration_ms)
    """
    return update_slice_len_dynamic()


# This function calculates time slice.
def calculate_time_slice(width_total: int, slice_len: int) -> int:
    """Calcula el número de slices para un ancho dado.
    
    Args:
        width_total: Ancho total del bloque
        slice_len: Longitud de cada slice
        
    Returns:
        int: Número de slices
    """
    return (width_total + slice_len - 1) // slice_len


# This function calculates slice duration.
def calculate_slice_duration(slice_len: int) -> float:
    """Calcula la duración de un slice en segundos."""

    if slice_len <= 0:
        return 0.0
    time_reso = float(getattr(config, "TIME_RESO", 0.0))
    down_rate = int(getattr(config, "DOWN_TIME_RATE", 1))
    return slice_len * time_reso * max(down_rate, 1)


# This function gets pipeline parameters.
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


# This function calculates overlap decimated.
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


# This function calculates absolute slice time.
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
