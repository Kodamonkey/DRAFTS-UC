# This module converts detection boxes into DM-time estimates.

"""DM candidate extractor for FRB pipeline - extracts DM and time information from detection boxes."""
from __future__ import annotations

import numpy as np

from ..config import config


def extract_candidate_dm(
    px: float,
    py: float,
    slice_len: int,
    img_height: int = 512,
    img_width: int = 512,
    dm_values=None,
) -> tuple[float, float, int]:
    """Map a CNN detection box (px, py) back to (DM, time_seconds, time_sample).

    SPEC-DM-005: the DM axis spans ``[DM_min, DM_max]`` linearly across the image
    rows. Row index maps as ``dm = DM_min + (py / (H - 1)) * (DM_max - DM_min)``,
    aligned with ``_dm_from_image_at_time`` in the HF pipeline. This never exceeds
    ``DM_max`` (the previous ``DM_max - DM_min + 1`` scaling produced ``DM_max + 1``
    at the top row). When ``dm_values`` is provided, the row index is resolved
    against the real DM grid instead of assuming a uniform spacing.
    """
    dm_min = float(config.DM_min)
    dm_max = float(config.DM_max)

    if dm_values is not None:
        dm_arr = np.asarray(dm_values, dtype=np.float64)
        if dm_arr.size > 0:
            denom = max(int(img_height) - 1, 1)
            frac = min(max(float(py) / denom, 0.0), 1.0)
            row = int(round(frac * (dm_arr.size - 1)))
            dm_val = float(dm_arr[row])
        else:
            dm_val = dm_min
    else:
        denom = max(int(img_height) - 1, 1)
        dm_val = dm_min + (float(py) / denom) * (dm_max - dm_min)

    # Clamp to the configured DM range (guards against boxes touching the border).
    dm_val = float(min(max(dm_val, dm_min), dm_max))

    scale_time = slice_len / float(max(int(img_width), 1))
    sample_off = px * scale_time
    t_sample = int(sample_off)
    t_seconds = float(sample_off) * config.TIME_RESO * config.DOWN_TIME_RATE
    return dm_val, t_seconds, t_sample
