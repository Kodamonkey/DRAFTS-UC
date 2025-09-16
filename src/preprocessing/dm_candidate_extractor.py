# This module converts detection boxes into DM-time estimates.

"""DM candidate extractor for FRB pipeline - extracts DM and time information from detection boxes."""
from __future__ import annotations

               
from ..config import config


# This function extracts candidate DM.
def extract_candidate_dm(px: float, py: float, slice_len: int) -> tuple[float, float, int]:
    dm_range = config.DM_max - config.DM_min + 1
    scale_dm = dm_range / 512.0
    scale_time = slice_len / 512.0
    
    dm_val = config.DM_min + py * scale_dm
    
    sample_off = px * scale_time
                                                                         
    t_sample = int(sample_off)
                                                                
    t_seconds = float(sample_off) * config.TIME_RESO * config.DOWN_TIME_RATE
    return dm_val, t_seconds, t_sample 
