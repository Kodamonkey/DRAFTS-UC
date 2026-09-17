# This module computes the parameters other layers derive from configuration.

"""Parameters derived directly from configuration, and from nothing else.

These three used to live in ``src/core/pipeline_parameters.py``, which meant
``preprocessing`` and ``output`` had to reach back up into ``core`` to get them.
They did it with imports placed inside functions, so the cycle never raised --
it was simply invisible (audit REF-09). Moving them to the configuration layer
makes the dependency graph acyclic without moving any scientific logic: nothing
imported here imports this module back.

``pipeline_parameters`` re-exports all three, so callers that already ask it for
them keep working.
"""
from __future__ import annotations

import numpy as np

from . import config
from ..analysis.science_metrics import dm_step_for_smearing

__all__ = [
    "calculate_frequency_downsampled",
    "calculate_dm_height",
    "calculate_dm_values",
]


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
