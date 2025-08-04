from __future__ import annotations

import numpy as np

from . import config


def downsample_data(data: np.ndarray) -> np.ndarray:
    """Down-sample time-frequency data using :mod:`config` rates."""
    n_time = (data.shape[0] // config.DOWN_TIME_RATE) * config.DOWN_TIME_RATE
    n_freq = (data.shape[2] // config.DOWN_FREQ_RATE) * config.DOWN_FREQ_RATE
    n_pol = data.shape[1]
    data = data[:n_time, :, :n_freq]
    data = data.reshape(
        n_time // config.DOWN_TIME_RATE,
        config.DOWN_TIME_RATE,
        n_pol,
        n_freq // config.DOWN_FREQ_RATE,
        config.DOWN_FREQ_RATE,
    )
    data = data.mean(axis=(1, 4, 2)).astype(np.float32)
    return data
