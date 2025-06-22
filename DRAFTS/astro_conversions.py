from __future__ import annotations

from . import config


def pixel_to_physical(px: float, py: float, slice_len: int) -> tuple[float, float, int]:
    """Translate network pixel coordinates to DM (pc cm⁻³) and time.

    Parameters
    ----------
    px, py : float
        Pixel coordinates from the detection output.
    slice_len : int
        Length of the time slice in samples.

    Returns
    -------
    tuple of (dm_val, t_seconds, t_sample)
    """
    dm_range = config.DM_max - config.DM_min + 1
    scale_dm = dm_range / 512.0
    scale_time = slice_len / 512.0
    dm_val = config.DM_min + py * scale_dm
    sample_off = px * scale_time
    t_sample = int(sample_off)
    t_seconds = t_sample * config.TIME_RESO * config.DOWN_TIME_RATE
    return dm_val, t_seconds, t_sample
