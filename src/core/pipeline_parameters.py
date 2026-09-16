# This module calculates derived parameters for pipeline execution.

"""Central helpers that compute derived parameters for the pipeline."""
from __future__ import annotations

import numpy as np

from ..config import config
from ..analysis.science_metrics import dm_step_for_smearing, dispersion_delay_ms
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


def calculate_dm_height() -> int:
    """Return the DM cube height derived from the configured DM range."""

    return int(calculate_dm_values().size)


def calculate_dm_values(dm_min: float | None = None, dm_max: float | None = None) -> np.ndarray:
    """Return DM trials for the configured search mode."""

    dm_max = float(getattr(config, "DM_max", 0)) if dm_max is None else float(dm_max)
    dm_min = float(getattr(config, "DM_min", 0)) if dm_min is None else float(dm_min)
    if dm_max < dm_min:
        return np.asarray([], dtype=np.float32)

    mode = str(getattr(config, "DM_GRID_MODE", "legacy_uniform")).lower()
    if mode == "legacy_uniform":
        n = int(round(dm_max - dm_min)) + 1
        return np.linspace(dm_min, dm_max, max(1, n), dtype=np.float32)

    if mode == "coarse_to_fine":
        # Backend placeholder: use physically-spaced trials until the second pass is wired.
        mode = "smear_limited"

    if mode == "smear_limited":
        freq = getattr(config, "FREQ", None)
        if freq is None or len(freq) < 2:
            step = 1.0
        else:
            dt_ms = float(getattr(config, "TIME_RESO", 0.0)) * max(1, int(getattr(config, "DOWN_TIME_RATE", 1))) * 1000.0
            smear_cfg = getattr(config, "MAX_DM_SMEARING_MS", "auto")
            if isinstance(smear_cfg, str) and smear_cfg.lower() == "auto":
                max_smear_ms = max(dt_ms, 0.001)
            else:
                max_smear_ms = max(float(smear_cfg), 0.001)
            step = dm_step_for_smearing(max_smear_ms, float(np.min(freq)), float(np.max(freq)))
        n = int(np.floor((dm_max - dm_min) / step)) + 1
        vals = dm_min + np.arange(max(1, n), dtype=np.float32) * np.float32(step)
        if vals[-1] < dm_max:
            vals = np.append(vals, np.float32(dm_max))
        return vals.astype(np.float32)

    n = int(round(dm_max - dm_min)) + 1
    return np.linspace(dm_min, dm_max, max(1, n), dtype=np.float32)


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
