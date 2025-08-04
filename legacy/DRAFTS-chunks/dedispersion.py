"""Dedispersion helpers using GPU/CPU."""
from __future__ import annotations

import logging
import numpy as np
from numba import cuda, njit, prange

from . import config

logger = logging.getLogger(__name__)

@cuda.jit
def _de_disp_gpu(dm_time, data, freq, index, start_offset, mid_channel):
    x, y = cuda.grid(2)
    if x < dm_time.shape[1] and y < dm_time.shape[2]:
        td_i = 0.0
        DM = x + start_offset
        for idx in index:
            delay = (
                4.15
                * DM
                * ((freq[idx]) ** -2 - (freq[-1] ** -2))
                * 1e3
                / config.TIME_RESO
                / config.DOWN_TIME_RATE
            )
            pos = int(delay + y)
            if 0 <= pos < data.shape[0]:
                td_i += data[pos, idx]
                if idx == mid_channel:
                    dm_time[1, x, y] = td_i
        dm_time[2, x, y] = td_i - dm_time[1, x, y]
        dm_time[0, x, y] = td_i


@njit(parallel=True)
def _d_dm_time_cpu(data, height: int, width: int) -> np.ndarray:
    """CPU fallback for dedispersion without Numba optimizations."""
    out = np.zeros((3, height, width), dtype=np.float32)
    nchan_ds = config.FREQ_RESO // config.DOWN_FREQ_RATE
    freq_index = np.arange(0, nchan_ds)
    mid_channel = nchan_ds // 2

    # Asegurarse de que config.FREQ sea válido
    if config.FREQ is None or config.FREQ.size == 0:
        logger.error("config.FREQ es inválido en _d_dm_time_cpu")
        return out
    
    # Usar frecuencias downsampled consistentemente
    freq_ds = np.mean(config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE), axis=1)
    
    # Sin usar prange para evitar problemas de Numba
    for DM in range(height):
        delays = (
            4.15
            * DM
            * (freq_ds ** -2 - freq_ds.max() ** -2)
            * 1e3
            / config.TIME_RESO
            / config.DOWN_TIME_RATE
        ).astype(np.int64)
        
        time_series = np.zeros(width, dtype=np.float32)
        
        for j in freq_index:
            start_idx = delays[j]
            end_idx = start_idx + width
            
            # Verificaciones simples
            if start_idx >= 0 and end_idx <= data.shape[0] and j < data.shape[1]:
                time_series += data[start_idx:end_idx, j]
                if j == mid_channel:
                    out[1, DM] = time_series.copy()
                    
        out[0, DM] = time_series.copy()
        out[2, DM] = time_series - out[1, DM]
    return out


def d_dm_time_g(data: np.ndarray, height: int, width: int, chunk_size: int = 128) -> np.ndarray:
    result = np.zeros((3, height, width), dtype=np.float32)
    try:
        print("[INFO] Intentando usar GPU para dedispersión...")
        
        # Verificar que config.FREQ y sus dimensiones sean válidas
        if config.FREQ is None:
            raise ValueError("config.FREQ is None during dedispersion")
        if config.FREQ.size == 0:
            raise ValueError("config.FREQ is empty during dedispersion")
        if config.FREQ_RESO == 0 or config.DOWN_FREQ_RATE == 0:
            raise ValueError(f"Invalid frequency parameters during dedispersion: FREQ_RESO={config.FREQ_RESO}, DOWN_FREQ_RATE={config.DOWN_FREQ_RATE}")
        
        # Verificar que el reshape es válido
        expected_size = config.FREQ_RESO // config.DOWN_FREQ_RATE
        if expected_size * config.DOWN_FREQ_RATE != config.FREQ_RESO:
            logger.warning("FREQ_RESO (%d) no es divisible por DOWN_FREQ_RATE (%d) en dedispersión", 
                          config.FREQ_RESO, config.DOWN_FREQ_RATE)
            # Ajustar para que sea divisible
            config.FREQ_RESO = expected_size * config.DOWN_FREQ_RATE
            config.FREQ = config.FREQ[:config.FREQ_RESO]
        
        freq_values = np.mean(config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE), axis=1)
        freq_gpu = cuda.to_device(freq_values)
        nchan_ds = config.FREQ_RESO // config.DOWN_FREQ_RATE
        index_values = np.arange(0, nchan_ds)
        mid_channel = nchan_ds // 2
        index_gpu = cuda.to_device(index_values)
        data_gpu = cuda.to_device(data)
        for start_dm in range(0, height, chunk_size):
            end_dm = min(start_dm + chunk_size, height)
            current_height = end_dm - start_dm
            dm_time_gpu = cuda.to_device(np.zeros((3, current_height, width), dtype=np.float32))
            nthreads = (8, 128)
            nblocks = (current_height // nthreads[0] + 1, width // nthreads[1] + 1)
            _de_disp_gpu[nblocks, nthreads](dm_time_gpu, data_gpu, freq_gpu, index_gpu, start_dm, mid_channel)
            cuda.synchronize()
            result[:, start_dm:end_dm, :] = dm_time_gpu.copy_to_host()
            del dm_time_gpu
        print("[INFO] Dedispersión GPU completada exitosamente")
        return result
    except (cuda.cudadrv.driver.CudaAPIError, Exception) as e:
        print(f"[WARNING] Error GPU ({e}), cambiando a CPU...")
        return _d_dm_time_cpu(data, height, width)

def dedisperse_patch(
    data: np.ndarray,
    freq_down: np.ndarray,
    dm: float,
    sample: int,
    patch_len: int = 512,
) -> tuple[np.ndarray, int]:
    """Dedisperse ``data`` at ``dm`` around ``sample`` and return the patch.

    Returns
    -------
    patch : np.ndarray
        Dedispersed patch of shape (patch_len, n_freq).
    start : int
        Start sample used on the original data array.
    """
    delays = (
        4.15
        * dm
        * (freq_down ** -2 - freq_down.max() ** -2)
        * 1e3
        / config.TIME_RESO
        / config.DOWN_TIME_RATE
    ).astype(np.int64)
    max_delay = int(delays.max())
    start = sample - patch_len // 2
    if start < 0:
        start = 0
    if start + patch_len + max_delay > data.shape[0]:
        start = max(0, data.shape[0] - (patch_len + max_delay))
    segment = data[start : start + patch_len + max_delay]
    patch = np.zeros((patch_len, freq_down.size), dtype=np.float32)
    for idx in range(freq_down.size):
        patch[:, idx] = segment[delays[idx] : delays[idx] + patch_len, idx]
    return patch, start

def dedisperse_patch_centered(
    data: np.ndarray,
    freq_down: np.ndarray,
    dm: float,
    sample: int,
    patch_len: int = 512,
) -> tuple[np.ndarray, int]:
    """Dedisperse ``data`` at ``dm`` around ``sample`` and return a centered patch.
    
    This function creates a patch that is centered on the candidate's peak,
    similar to the original implementation in d-center-main.py.

    Returns
    -------
    patch : np.ndarray
        Dedispersed patch of shape (patch_len, n_freq) centered on the candidate.
    start : int
        Start sample used on the original data array.
    """
    # Calculate dispersion delays
    delays = (
        4.15
        * dm
        * (freq_down ** -2 - freq_down.max() ** -2)
        * 1e3
        / config.TIME_RESO
        / config.DOWN_TIME_RATE
    ).astype(np.int64)
    
    max_delay = int(delays.max())
    
    # Calculate the center position for the patch
    # This ensures the candidate's peak is at the center of the patch
    center_sample = sample
    start = center_sample - patch_len // 2
    
    # Ensure we don't go out of bounds
    if start < 0:
        start = 0
    if start + patch_len + max_delay > data.shape[0]:
        start = max(0, data.shape[0] - (patch_len + max_delay))
    
    # Extract the segment with extra samples for dedispersion
    segment = data[start : start + patch_len + max_delay]
    
    # Create the dedispersed patch
    patch = np.zeros((patch_len, freq_down.size), dtype=np.float32)
    for idx in range(freq_down.size):
        patch[:, idx] = segment[delays[idx] : delays[idx] + patch_len, idx]
    
    return patch, start

def dedisperse_block(
    data: np.ndarray,
    freq_down: np.ndarray,
    dm: float,
    start: int,
    block_len: int,
) -> np.ndarray:
    """Dedisperse a continuous block of data.

    Parameters
    ----------
    data : np.ndarray
        Time--frequency array already downsampled in frequency.
    freq_down : np.ndarray
        Array of downsampled frequency values.
    dm : float
        Dispersion measure used for the correction.
    start : int
        Starting sample of the block within ``data``.
    block_len : int
        Number of samples to include in the output block.

    Returns
    -------
    np.ndarray
        Dedispersed block of shape ``(block_len, n_freq)`` whose time span
        matches that of the original slice.
    """

    delays = (
        4.15
        * dm
        * (freq_down ** -2 - freq_down.max() ** -2)
        * 1e3
        / config.TIME_RESO
        / config.DOWN_TIME_RATE
    ).astype(np.int64)

    # DEBUG: Verificar dedispersión
    if config.DEBUG_FREQUENCY_ORDER:
        print(f"🔍 [DEBUG DEDISPERSIÓN] DM: {dm:.2f} pc cm⁻³")
        print(f"🔍 [DEBUG DEDISPERSIÓN] freq_down shape: {freq_down.shape}")
        print(f"🔍 [DEBUG DEDISPERSIÓN] Primeras 3 freq_down: {freq_down[:3]}")
        print(f"🔍 [DEBUG DEDISPERSIÓN] Últimas 3 freq_down: {freq_down[-3:]}")
        print(f"🔍 [DEBUG DEDISPERSIÓN] freq_down.max(): {freq_down.max():.2f} MHz")
        print(f"🔍 [DEBUG DEDISPERSIÓN] Primeros 3 delays: {delays[:3]} muestras")
        print(f"🔍 [DEBUG DEDISPERSIÓN] Últimos 3 delays: {delays[-3:]} muestras")
        print(f"🔍 [DEBUG DEDISPERSIÓN] max_delay: {delays.max()} muestras")
        print(f"🔍 [DEBUG DEDISPERSIÓN] Dedispersión esperada: freq ALTAS llegan primero (delay=0), freq BAJAS llegan después (delay>0)")
        if freq_down[0] < freq_down[-1]:  # ascendente
            expected_delay_pattern = "delays DECRECIENTES (de max a 0)"
        else:  # descendente
            expected_delay_pattern = "delays CRECIENTES (de 0 a max)"
        print(f"🔍 [DEBUG DEDISPERSIÓN] Patrón esperado de delays: {expected_delay_pattern}")
        print("🔍 [DEBUG DEDISPERSIÓN] " + "="*60)

    max_delay = int(delays.max())
    if start + block_len + max_delay > data.shape[0]:
        start = max(0, data.shape[0] - (block_len + max_delay))

    segment = data[start : start + block_len + max_delay]
    block = np.zeros((block_len, freq_down.size), dtype=np.float32)
    for idx in range(freq_down.size):
        block[:, idx] = segment[delays[idx] : delays[idx] + block_len, idx]
    return block
