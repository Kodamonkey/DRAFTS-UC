"""Dedispersion helpers using GPU/CPU."""
from __future__ import annotations

import numpy as np
from numba import cuda, njit, prange

from . import config


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
    out = np.zeros((3, height, width), dtype=np.float32)
    nchan_ds = config.FREQ_RESO // config.DOWN_FREQ_RATE
    freq_index = np.arange(0, nchan_ds)
    mid_channel = nchan_ds // 2

    for DM in prange(height):
        delays = (
            4.15
            * DM
            * (config.FREQ ** -2 - config.FREQ.max() ** -2)
            * 1e3
            / config.TIME_RESO
            / config.DOWN_TIME_RATE
        ).astype(np.int64)
        time_series = np.zeros(width, dtype=np.float32)
        for j in freq_index:
            time_series += data[delays[j] : delays[j] + width, j]
            if j == mid_channel:
                out[1, DM] = time_series
        out[0, DM] = time_series
        out[2, DM] = time_series - out[1, DM]
    return out


def d_dm_time_g(data: np.ndarray, height: int, width: int, chunk_size: int = 128) -> np.ndarray:
    result = np.zeros((3, height, width), dtype=np.float32)
    try:
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
        return result
    except cuda.cudadrv.driver.CudaAPIError:
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

    max_delay = int(delays.max())
    if start + block_len + max_delay > data.shape[0]:
        start = max(0, data.shape[0] - (block_len + max_delay))

    segment = data[start : start + block_len + max_delay]
    block = np.zeros((block_len, freq_down.size), dtype=np.float32)
    for idx in range(freq_down.size):
        block[:, idx] = segment[delays[idx] : delays[idx] + block_len, idx]
    return block
