# This module plans slice layouts for each chunk.

"""Chunk planner for FRB pipeline - plans optimal chunk and slice sizes."""
from __future__ import annotations

                          
import logging
from dataclasses import dataclass
from typing import Dict, List

              
logger = logging.getLogger(__name__)


@dataclass
class SlicePlan:
    start_idx: int
    end_idx: int             
    length: int
    duration_ms: float


def plan_slices_for_chunk(
    num_samples_decimated: int,
    target_duration_ms: float,
    time_reso_decimated_s: float,
    max_slice_count: int = 5000,
    time_tol_ms: float = 0.1,
) -> Dict:
    """Compute contiguous slice plans for a time-decimated chunk.

    - Adjust the slice count to the closest divisor of the chunk size.
    - Generate contiguous 0-based boundaries without overlaps or gaps.

    Args:
        num_samples_decimated: Samples in the chunk after temporal downsampling.
        target_duration_ms: Target duration per slice (ms).
        time_reso_decimated_s: Effective temporal resolution (``TIME_RESO * DOWN_TIME_RATE``) in seconds.
        max_slice_count: Maximum number of slices per chunk.
        time_tol_ms: Tolerance used in adjustment reports.

    Returns:
        dict with:
          - n_slices
          - avg_ms, delta_ms
          - slices: List[SlicePlan]
    """
    if num_samples_decimated <= 0:
        raise ValueError("num_samples_decimated must be > 0")
    if time_reso_decimated_s <= 0:
        raise ValueError("time_reso_decimated_s must be > 0")
    if target_duration_ms <= 0:
                           
        target_duration_ms = num_samples_decimated * time_reso_decimated_s * 1000

    target_samples_float = (target_duration_ms / 1000.0) / time_reso_decimated_s
    if target_samples_float <= 0:
        target_samples_float = 1.0

    n_slices = max(1, min(max_slice_count, int(round(num_samples_decimated / target_samples_float))))
    n_slices = min(n_slices, num_samples_decimated)                              

    base = num_samples_decimated // n_slices
    r = num_samples_decimated % n_slices

    lengths: List[int] = []
    for i in range(n_slices):
        length = base + (1 if i < r else 0)
        lengths.append(max(1, length))

                                
    slices: List[SlicePlan] = []
    s0 = 0
    for length in lengths:
        s1 = s0 + length             
        duration_ms = length * time_reso_decimated_s * 1000.0
        slices.append(SlicePlan(start_idx=s0, end_idx=s1, length=length, duration_ms=duration_ms))
        s0 = s1

                                             
    assert slices[0].start_idx == 0
    assert slices[-1].end_idx == num_samples_decimated
    for a, b in zip(slices, slices[1:]):
        assert a.end_idx == b.start_idx

    avg_ms = sum(sl.duration_ms for sl in slices) / len(slices)
    delta_ms = avg_ms - target_duration_ms

    return {
        "n_slices": n_slices,
        "avg_ms": avg_ms,
        "delta_ms": delta_ms,
        "time_tol_ms": time_tol_ms,
        "slices": slices,
    }


