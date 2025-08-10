"""Dedispersion helpers using GPU/CPU."""
from __future__ import annotations

import logging
import numpy as np
from numba import cuda, njit, prange

from .. import config

logger = logging.getLogger(__name__)

@cuda.jit
def _de_disp_gpu(dm_time, data, freq, index, dm_values, mid_channel):
    x, y = cuda.grid(2)
    if x < dm_time.shape[1] and y < dm_time.shape[2]:
        # Suma total y contador de contribuciones válidas (normalización por exposición)
        total_val = 0.0
        total_cnt = 0

        # Valor del canal medio (para segundo plano)
        mid_val = 0.0
        DM = dm_values[x] 

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
                total_val += data[pos, idx]
                total_cnt += 1
                if idx == mid_channel:
                    mid_val = data[pos, idx]

        # Normalizar por #canales que realmente contribuyeron
        if total_cnt > 0:
            dm_time[0, x, y] = total_val / total_cnt
        else:
            dm_time[0, x, y] = 0.0

        dm_time[1, x, y] = mid_val
        dm_time[2, x, y] = dm_time[0, x, y] - mid_val


@njit(parallel=False)
def _d_dm_time_cpu(data, height: int, width: int, dm_min: float, dm_max: float) -> np.ndarray:
    """CPU fallback para dedispersión con normalización por exposición y bordes.

    Implementa suma con manejo de bordes por canal y normaliza cada
    punto (DM,t) por el número de canales que aportaron.
    """
    out = np.zeros((3, height, width), dtype=np.float32)
    nchan_ds = config.FREQ_RESO // config.DOWN_FREQ_RATE
    mid_channel = nchan_ds // 2

    # Validación de frecuencias
    if config.FREQ is None or config.FREQ.size == 0:
        logger.error("config.FREQ es inválido en _d_dm_time_cpu")
        return out

    # Frecuencias decimadas
    freq_ds = np.mean(
        config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE),
        axis=1,
    )

    # Calcular valores exactos de DM para cada píxel
    dm_values = np.linspace(dm_min, dm_max, height, dtype=np.float32)

    for i, DM in enumerate(dm_values):
        delays = (
            4.15
            * DM
            * (freq_ds ** -2 - freq_ds.max() ** -2)
            * 1e3
            / config.TIME_RESO
            / config.DOWN_TIME_RATE
        ).astype(np.int64)

        total_series = np.zeros(width, dtype=np.float32)
        count_series = np.zeros(width, dtype=np.int32)
        mid_series = np.zeros(width, dtype=np.float32)
        mid_filled = np.zeros(width, dtype=np.bool_)

        for j in range(nchan_ds):
            d = delays[j]
            # Rango fuente y destino con manejo de bordes
            if d >= 0:
                src_lo = d
                dst_lo = 0
            else:
                src_lo = 0
                dst_lo = -d
            src_hi = min(data.shape[0], d + width) if d + width > 0 else 0
            length = src_hi - src_lo
            if length <= 0 or j >= data.shape[1]:
                continue
            dst_hi = dst_lo + length

            total_series[dst_lo:dst_hi] += data[src_lo:src_hi, j]
            count_series[dst_lo:dst_hi] += 1

            if j == mid_channel:
                mid_series[dst_lo:dst_hi] = data[src_lo:src_hi, j]
                mid_filled[dst_lo:dst_hi] = True

        # Normalización por #canales válidos
        norm = np.maximum(count_series.astype(np.float32), 1.0)
        out[0, i] = total_series / norm
        # Si mid no estuvo disponible en algún punto, dejar 0 allí
        out[1, i] = mid_series
        out[2, i] = out[0, i] - out[1, i]

    return out


def d_dm_time_g(data: np.ndarray, height: int, width: int, chunk_size: int = 128, dm_min: float = None, dm_max: float = None) -> np.ndarray:
    result = np.zeros((3, height, width), dtype=np.float32)
    
    # Si no se proporcionan dm_min y dm_max, usar config
    if dm_min is None:
        dm_min = config.DM_min
    if dm_max is None:
        dm_max = config.DM_max
    
    # Pre-whitening por canal opcional para mitigar bandpass/RFI leve
    try:
        if getattr(config, 'PREWHITEN_BEFORE_DM', False):
            # Z-score por canal (eje 0 = tiempo)
            eps = 1e-6
            mean_ch = np.mean(data, axis=0)
            std_ch = np.std(data, axis=0)
            std_ch = np.where(std_ch < eps, 1.0, std_ch)
            data = (data - mean_ch) / std_ch
    except Exception as e:
        print(f"[WARNING] Error en prewhitening: {e}")
    try:
        print("[INFO] Intentando usar GPU para dedispersión...")
        
        # Verificar que config.FREQ y sus dimensiones sean válidas
        if config.FREQ is None:
            raise ValueError("config.FREQ is None during dedispersión")
        if config.FREQ.size == 0:
            raise ValueError("config.FREQ is empty during dedispersión")
        if config.FREQ_RESO == 0 or config.DOWN_FREQ_RATE == 0:
            raise ValueError(f"Invalid frequency parameters during dedispersión: FREQ_RESO={config.FREQ_RESO}, DOWN_FREQ_RATE={config.DOWN_FREQ_RATE}")
        
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
            
            # Calcular valores exactos de DM para este chunk
            chunk_dm_min = dm_min + (start_dm * (dm_max - dm_min) / (height - 1))
            chunk_dm_max = dm_min + (end_dm * (dm_max - dm_min) / (height - 1))
            dm_values = np.linspace(chunk_dm_min, chunk_dm_max, current_height, dtype=np.float32)
            dm_values_gpu = cuda.to_device(dm_values)
            
            dm_time_gpu = cuda.to_device(np.zeros((3, current_height, width), dtype=np.float32))
            nthreads = (8, 128)
            nblocks = (current_height // nthreads[0] + 1, width // nthreads[1] + 1)
            _de_disp_gpu[nblocks, nthreads](dm_time_gpu, data_gpu, freq_gpu, index_gpu, dm_values_gpu, mid_channel)
            cuda.synchronize()
            result[:, start_dm:end_dm, :] = dm_time_gpu.copy_to_host()
            del dm_time_gpu, dm_values_gpu
        print("[INFO] Dedispersión GPU completada exitosamente")
        return result
    except (cuda.cudadrv.driver.CudaAPIError, Exception) as e:
        print(f"[WARNING] Error GPU ({e}), cambiando a CPU...")
        return _d_dm_time_cpu(data, height, width, dm_min, dm_max)

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
