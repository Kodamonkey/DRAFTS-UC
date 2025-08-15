"""
DM-Time Transformation with CUDA Acceleration
============================================

Este módulo implementa la transformación DM-Time (Dispersión de Medida)
con aceleración CUDA para corrección de dispersión temporal de señales de radio.

Basado en los algoritmos de d-center-main.py y CheckRes/
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Intentar importar dependencias opcionales
try:
    from numba import cuda, njit, prange
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    logger.warning("Numba no disponible. Funciones CUDA deshabilitadas.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV no disponible. Algunas funciones deshabilitadas.")


def d_dm_time_gpu(
    data: np.ndarray,
    freq_array: np.ndarray,
    time_reso: float,
    down_time_rate: int = 1,
    height: int = 2048,
    width: Optional[int] = None
) -> np.ndarray:
    """
    Transformación DM-Time con aceleración GPU usando CUDA.
    
    Args:
        data: Array de datos (muestras, canales)
        freq_array: Array de frecuencias (MHz)
        time_reso: Resolución temporal (segundos)
        down_time_rate: Factor de downsampling temporal
        height: Rango de DM (altura del array de salida)
        width: Ancho del array de salida (opcional)
        
    Returns:
        Array con transformación DM-Time aplicada (3, height, width)
    """
    if not CUDA_AVAILABLE:
        logger.warning("CUDA no disponible, usando versión CPU")
        return d_dm_time_cpu(data, freq_array, time_reso, down_time_rate, height, width)
    
    if width is None:
        width = data.shape[0]
    
    # Preparar datos para GPU
    freq_gpu = cuda.to_device(np.mean(freq_array.reshape(-1, 1), axis=1))
    index_gpu = cuda.to_device(np.arange(len(freq_array)))
    dm_time_gpu = cuda.to_device(np.zeros((3, height, width)).astype(np.float32))
    data_gpu = cuda.to_device(data.astype(np.float32))
    
    # Configurar grid y bloques CUDA
    nthreads = (8, 128)
    nblocks = (height // nthreads[0] + 1, width // nthreads[1] + 1)
    
    # Ejecutar kernel CUDA
    _de_disp_kernel[nblocks, nthreads](dm_time_gpu, data_gpu, freq_gpu, index_gpu, time_reso, down_time_rate)
    dm_time = dm_time_gpu.copy_to_host()
    
    return dm_time


def d_dm_time_cpu(
    data: np.ndarray,
    freq_array: np.ndarray,
    time_reso: float,
    down_time_rate: int = 1,
    height: int = 2048,
    width: Optional[int] = None
) -> np.ndarray:
    """
    Transformación DM-Time usando CPU (versión de respaldo).
    
    Args:
        data: Array de datos (muestras, canales)
        freq_array: Array de frecuencias (MHz)
        time_reso: Resolución temporal (segundos)
        down_time_rate: Factor de downsampling temporal
        height: Rango de DM (altura del array de salida)
        width: Ancho del array de salida (opcional)
        
    Returns:
        Array con transformación DM-Time aplicada (3, height, width)
    """
    if width is None:
        width = data.shape[0]
    
    new_data = np.zeros((3, height, width))
    freq_index = np.arange(len(freq_array))
    
    for DM in range(height):
        dds = (4.15 * DM * (freq_array**-2 - freq_array.max()**-2) * 1e3 / time_reso / down_time_rate).astype(np.int64)
        time_series = np.zeros(width)
        
        for i in freq_index:
            if dds[i] + width <= data.shape[0]:
                time_series += data[dds[i]:dds[i] + width, i]
                if i == len(freq_array) // 2:
                    new_data[1, DM] = time_series
        
        new_data[0, DM] = time_series
        new_data[2, DM] = time_series - new_data[1, DM]
    
    return new_data


@njit(parallel=True)
def d_dm_time_numba(
    data: np.ndarray,
    freq_array: np.ndarray,
    time_reso: float,
    down_time_rate: int = 1,
    height: int = 2048,
    width: Optional[int] = None
) -> np.ndarray:
    """
    Transformación DM-Time usando Numba para aceleración CPU.
    
    Args:
        data: Array de datos (muestras, canales)
        freq_array: Array de frecuencias (MHz)
        time_reso: Resolución temporal (segundos)
        down_time_rate: Factor de downsampling temporal
        height: Rango de DM (altura del array de salida)
        width: Ancho del array de salida (opcional)
        
    Returns:
        Array con transformación DM-Time aplicada (3, height, width)
    """
    if width is None:
        width = data.shape[0]
    
    new_data = np.zeros((3, height, width))
    freq_index = np.arange(len(freq_array))
    
    for DM in prange(height):
        dds = (4.15 * DM * (freq_array**-2 - freq_array.max()**-2) * 1e3 / time_reso / down_time_rate).astype(np.int64)
        time_series = np.zeros(width)
        
        for i in prange(len(freq_index)):
            if dds[i] + width <= data.shape[0]:
                time_series += data[dds[i]:dds[i] + width, i]
                if i == len(freq_array) // 2:
                    new_data[1, DM] = time_series
        
        new_data[0, DM] = time_series
        new_data[2, DM] = time_series - new_data[1, DM]
    
    return new_data


if CUDA_AVAILABLE:
    @cuda.jit
    def _de_disp_kernel(dm_time, data, freq, index, time_reso, down_time_rate):
        """
        Kernel CUDA para transformación DM-Time.
        """
        x, y = cuda.grid(2)
        if x < dm_time.shape[1] and y < dm_time.shape[2]:
            td_i, DM = 0.0, x
            for i in range(len(index)):
                dds = int(4.15 * DM * (freq[i]**-2 - freq[-1]**-2) * 1e3 / time_reso / down_time_rate + y)
                if dds < data.shape[0]:
                    td_i += data[dds, i]
                    if i == len(index) // 2:
                        dm_time[1, x, y] = td_i
            dm_time[2, x, y] = td_i - dm_time[1, x, y]
            dm_time[0, x, y] = td_i


def calculate_dm_delays(
    freq_array: np.ndarray,
    dm: float,
    time_reso: float,
    down_time_rate: int = 1
) -> np.ndarray:
    """
    Calcula los retrasos temporales para un DM específico.
    
    Args:
        freq_array: Array de frecuencias (MHz)
        dm: Valor de DM
        time_reso: Resolución temporal (segundos)
        down_time_rate: Factor de downsampling temporal
        
    Returns:
        Array de retrasos temporales en muestras
    """
    dds = (4.15 * dm * (freq_array**-2 - freq_array.max()**-2) * 1e3 / time_reso / down_time_rate).astype(np.int64)
    return dds 