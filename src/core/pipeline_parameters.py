# This module calculates derived parameters for pipeline execution.

"""Central helpers that compute derived parameters for the pipeline."""
from __future__ import annotations

import numpy as np

from ..config import config
from ..domain.physics import dispersion_delay_ms
from ..preprocessing.slice_len_calculator import update_slice_len_dynamic

# Re-exported so existing callers keep importing them from here. They are
# defined one layer down because preprocessing and output need them too, and
# reaching back up into core for them is what created the hidden cycle (REF-09).
from ..config.derived import (  # noqa: F401
    calculate_dm_height,
    calculate_dm_values,
    calculate_frequency_downsampled,
)


def should_use_hf_pipeline(
    freq_low_mhz: float,
    freq_high_mhz: float,
    dm_max: float,
    time_reso_s: float,
    down_time_rate: int,
    collapse_ratio: float = 2.0,
) -> tuple[bool, str]:
    """Decide LF vs HF based on bow-tie collapse physics.

    The bow-tie collapses when the dispersive sweep across the band
    becomes smaller than *collapse_ratio* × effective time resolution.

    Returns
    -------
    use_hf : bool
        True when the HF (SNR-based) pipeline should be used.
    reason : str
        Human-readable explanation of the decision.
    """
    dt_disp_ms = dispersion_delay_ms(dm_max, freq_low_mhz, freq_high_mhz)
    dt_res_ms = time_reso_s * down_time_rate * 1000.0

    if dt_res_ms <= 0:
        return False, "invalid time resolution — falling back to standard pipeline"

    ratio = dt_disp_ms / dt_res_ms
    collapsed = ratio < collapse_ratio

    reason = (
        f"bow-tie {'collapsed' if collapsed else 'resolved'}: "
        f"\u0394t_disp={dt_disp_ms:.4f} ms, \u0394t_res={dt_res_ms:.4f} ms, "
        f"ratio={ratio:.2f} {'<' if collapsed else '>='} {collapse_ratio:.1f} "
        f"(DM_max={dm_max}, band=[{freq_low_mhz:.1f}\u2013{freq_high_mhz:.1f}] MHz)"
    )
    return collapsed, reason


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
