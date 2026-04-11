# This module performs temporal and spectral downsampling.

"""Data downsampler for FRB pipeline - reduces temporal and frequency resolution.

Uses a single-pass Numba kernel to avoid creating 4 intermediate arrays
that can consume 3-4x the input size in memory.
"""
from __future__ import annotations

import logging

import numpy as np

from ..config import config

logger = logging.getLogger(__name__)

try:
    from numba import njit, prange

    @njit(parallel=True, cache=True, fastmath=True)
    def _downsample_kernel(
        data: np.ndarray,
        n_time_ds: int,
        n_freq_ds: int,
        down_time: int,
        down_freq: int,
        n_pol: int,
    ) -> np.ndarray:
        """Single-pass downsampling: sum over time bins, mean over pol and freq.

        Input shape:  (time, pol, freq)  — trimmed to divisible sizes.
        Output shape: (n_time_ds, n_freq_ds), float32.
        """
        inv_pol_freq = np.float32(1.0 / (n_pol * down_freq))
        out = np.empty((n_time_ds, n_freq_ds), dtype=np.float32)

        for t in prange(n_time_ds):
            t_base = t * down_time
            for f in range(n_freq_ds):
                f_base = f * down_freq
                acc = np.float32(0.0)
                for dt in range(down_time):
                    for p in range(n_pol):
                        for df in range(down_freq):
                            acc += np.float32(data[t_base + dt, p, f_base + df])
                # sum over time bins (not mean), mean over pol and freq
                out[t, f] = acc * inv_pol_freq
        return out

    _HAS_NUMBA = True

except ImportError:
    _HAS_NUMBA = False


def _downsample_numpy(data: np.ndarray, down_time: int, down_freq: int) -> np.ndarray:
    """Fallback NumPy implementation (original logic, kept for non-Numba envs)."""
    n_time = (data.shape[0] // down_time) * down_time
    n_freq = (data.shape[2] // down_freq) * down_freq
    n_pol = data.shape[1]
    d = data[:n_time, :, :n_freq]
    d = d.reshape(
        n_time // down_time,
        down_time,
        n_pol,
        n_freq // down_freq,
        down_freq,
    )
    d = d.sum(axis=1)    # sum over time bins
    d = d.mean(axis=1)   # mean over polarisations
    d = d.mean(axis=2)   # mean over frequency bins
    return d.astype(np.float32)


def downsample_data(data: np.ndarray) -> np.ndarray:
    """Down-sample time-frequency data using :mod:`config` rates.

    - Temporal: sum over windows of size ``DOWN_TIME_RATE`` (PRESTO style).
    - Frequency: average across groups of ``DOWN_FREQ_RATE`` channels.
    - Polarization: average (Stokes I already selected during load if available).
    """
    down_time = int(config.DOWN_TIME_RATE)
    down_freq = int(config.DOWN_FREQ_RATE)

    n_time_ds = data.shape[0] // down_time
    n_freq_ds = data.shape[2] // down_freq
    n_pol = data.shape[1]

    if _HAS_NUMBA and n_time_ds > 0 and n_freq_ds > 0:
        # Trim to divisible sizes in a view (no copy)
        trimmed = data[: n_time_ds * down_time, :, : n_freq_ds * down_freq]
        return _downsample_kernel(trimmed, n_time_ds, n_freq_ds, down_time, down_freq, n_pol)

    return _downsample_numpy(data, down_time, down_freq)
