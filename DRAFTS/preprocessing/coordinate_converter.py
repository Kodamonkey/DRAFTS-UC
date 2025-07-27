"""Coordinate converter for FRB pipeline - translates pixel coordinates to physical values."""
from __future__ import annotations

from .. import config


def pixel_to_physical(px: float, py: float, slice_len: int) -> tuple[float, float, int]:
    """Translate network pixel coordinates to DM (pc cm⁻³) and time.
    
    EXACTLY compatible with DRAFTS original implementation where:
    DM = (left_y + right_y) / 2 * (DM_range / 512)
    
    Note: DRAFTS original does NOT add DM_min offset, it uses absolute DM values.

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
    # ✅ EXACTA COMPATIBILIDAD CON DRAFTS ORIGINAL
    # DRAFTS original: DM = (left_y + right_y) / 2 * (DM_range / 512)
    # Donde DM_range es el rango total desde DM_min hasta DM_max
    dm_range = config.DM_max - config.DM_min + 1
    scale_dm = dm_range / 512.0
    scale_time = slice_len / 512.0
    
    # ✅ DRAFTS ORIGINAL NO AGREGA DM_min - usa valores absolutos
    # py ya representa el centro del bounding box (left_y + right_y) / 2
    dm_val = py * scale_dm
    
    sample_off = px * scale_time
    t_sample = int(sample_off)
    t_seconds = t_sample * config.TIME_RESO * config.DOWN_TIME_RATE
    return dm_val, t_seconds, t_sample
