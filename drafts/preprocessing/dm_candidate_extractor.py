from __future__ import annotations

from .. import config


def extract_candidate_dm(px: float, py: float, slice_len: int) -> tuple[float, float, int]:
    """Extract candidate DM (Dispersion Measure) and time from detection pixel coordinates.
    
    Converts network pixel coordinates to physical DM (pc cm⁻³) and time values.
    Compatible with DRAFTS original implementation where:
    DM = (center_y) * (DM_range / 512)

    Parameters
    ----------
    px, py : float
        Pixel coordinates from the detection output.
    slice_len : int
        Length of the time slice in samples.

    Returns
    -------
    tuple of (dm_val, t_seconds, t_sample)
        dm_val: Dispersion Measure in pc cm⁻³
        t_seconds: Time in seconds
        t_sample: Time sample index
    """
    # ✅ CORRECCIÓN: Usar la misma fórmula que DRAFTS original
    # DRAFTS original: DM = center_y * (DM_range / 512)
    # Donde DM_range es el rango total desde DM_min hasta DM_max
    dm_range = config.DM_max - config.DM_min + 1
    scale_dm = dm_range / 512.0
    scale_time = slice_len / 512.0
    
    # DRAFTS original usa: DM = py * scale_dm (sin offset DM_min)
    # Pero necesitamos agregar DM_min si el rango no empieza en 0
    dm_val = config.DM_min + py * scale_dm
    
    sample_off = px * scale_time
    t_sample = int(sample_off)
    t_seconds = t_sample * config.TIME_RESO * config.DOWN_TIME_RATE
    return dm_val, t_seconds, t_sample 