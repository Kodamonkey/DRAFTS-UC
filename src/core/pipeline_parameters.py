# This module calculates derived parameters for pipeline execution.

"""Central helpers that compute derived parameters for the pipeline."""
from __future__ import annotations

import numpy as np

from ..config import config
from ..preprocessing.slice_len_calculator import update_slice_len_dynamic


def calculate_frequency_downsampled() -> np.ndarray:
    """Return the decimated frequency axis used throughout the pipeline."""

    if config.FREQ is None or getattr(config, "FREQ_RESO", 0) <= 0:
        raise ValueError("Frequency metadata has not been loaded")

    down_rate = int(getattr(config, "DOWN_FREQ_RATE", 1))
    if down_rate <= 0:
        raise ValueError("DOWN_FREQ_RATE must be greater than zero")

    total_channels = len(config.FREQ)
    usable = total_channels - (total_channels % down_rate)
    if usable == 0:
        raise ValueError("DOWN_FREQ_RATE exceeds the number of available channels")

    trimmed = config.FREQ[:usable]
    return trimmed.reshape(-1, down_rate).mean(axis=1)


def calculate_dm_height() -> int:
    """Return the DM cube height derived from the configured DM range."""

    dm_max = int(getattr(config, "DM_max", 0))
    dm_min = int(getattr(config, "DM_min", 0))
    return max(0, dm_max - dm_min + 1)


def calculate_width_total(total_samples: int | None = None) -> int:
    """Return the total number of decimated time samples for a file."""

    samples = int(total_samples) if total_samples is not None else int(getattr(config, "FILE_LENG", 0))
    down_rate = int(getattr(config, "DOWN_TIME_RATE", 1))
    if samples <= 0 or down_rate <= 0:
        return 0
    return samples // down_rate


def calculate_slice_parameters() -> tuple[int, float]:
    """Return slice length and expected duration in milliseconds."""

    return update_slice_len_dynamic()


def calculate_time_slice(width_total: int, slice_len: int) -> int:
    """Return how many slices are required to cover ``width_total`` samples."""

    return (width_total + slice_len - 1) // slice_len


def calculate_overlap_decimated(overlap_left_raw: int, overlap_right_raw: int) -> tuple[int, int]:
    """Return the overlap expressed in decimated samples."""

    rate = int(config.DOWN_TIME_RATE)
    overlap_left_ds = (overlap_left_raw + rate - 1) // rate
    overlap_right_ds = (overlap_right_raw + rate - 1) // rate
    return overlap_left_ds, overlap_right_ds


def calculate_absolute_slice_time(chunk_start_time_sec: float, start_idx: int, dt_ds: float) -> float:
    """Return the absolute start time of a slice in seconds."""

    return chunk_start_time_sec + (start_idx * dt_ds)
