"""Dedispersion helpers tailored for ALMA data."""
from __future__ import annotations

import numpy as np

from . import config


def d_dm_time(data: np.ndarray, height: int, width: int) -> np.ndarray:
    """Return DM-time cubes using fractional-delay dedispersion."""
    # Average frequency according to current downsampling factor
    freq_values = np.mean(
        config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE),
        axis=1,
    )
    nchan_ds = freq_values.size
    mid_channel = nchan_ds // 2

    out = np.zeros((3, height, width), dtype=np.float32)
    base_idx = np.arange(width, dtype=np.float64)
    max_len = data.shape[0]

    for dmi in range(height):
        DM = dmi + config.DM_min
        delays = (
            4.15
            * DM
            * (freq_values**-2 - freq_values[-1] ** -2)
            * 1e3
            / config.TIME_RESO
            / config.DOWN_TIME_RATE
        )
        series = np.zeros(width, dtype=np.float64)
        mid_series = np.zeros(width, dtype=np.float64)
        for j in range(nchan_ds):
            shifted = np.interp(
                base_idx + delays[j],
                np.arange(max_len, dtype=np.float64),
                data[:, j],
                left=0.0,
                right=0.0,
            )
            series += shifted
            if j == mid_channel:
                mid_series = series.copy()
        out[0, dmi] = series
        out[1, dmi] = mid_series
        out[2, dmi] = series - mid_series

    return out.astype(np.float32)
