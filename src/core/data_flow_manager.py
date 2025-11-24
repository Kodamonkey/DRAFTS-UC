# This module manages chunk-level data flow operations.

"""Manage chunks, slices, and scheduling of the processing workflow."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..config import config
from ..preprocessing.chunk_planner import plan_slices_for_chunk
from ..preprocessing.dedispersion import d_dm_time_g
from .pipeline_parameters import (
    calculate_dm_height,
    calculate_frequency_downsampled,
    calculate_overlap_decimated,
    calculate_slice_parameters,
    calculate_time_slice,
    calculate_width_total,
)

              
logger = logging.getLogger(__name__)


def downsample_chunk(block: np.ndarray) -> tuple[np.ndarray, float]:
    """Apply temporal and frequency downsampling to the full chunk."""
    from ..preprocessing.data_downsampler import downsample_data
    block_ds = downsample_data(block)
    dt_ds = config.TIME_RESO * config.DOWN_TIME_RATE
    return block_ds, dt_ds


def build_dm_time_cube(block_ds: np.ndarray, height: int, dm_min: float, dm_max: float) -> np.ndarray:
    """Build the DM–time cube for the decimated block."""
    width = block_ds.shape[0]
    from ..preprocessing.dedispersion import d_dm_time_g
    return d_dm_time_g(block_ds, height=height, width=width, dm_min=dm_min, dm_max=dm_max)


def trim_valid_window(
    block_ds: np.ndarray,
    dm_time_full: np.ndarray,
    overlap_left_ds: int,
    overlap_right_ds: int
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Extract the valid window, discarding overlap-contaminated edges."""
    valid_start_ds = max(0, overlap_left_ds)
                                                         
    valid_end_ds = block_ds.shape[0]
    if valid_end_ds <= valid_start_ds:
        valid_start_ds, valid_end_ds = 0, block_ds.shape[0]
    
    dm_time = dm_time_full[:, :, valid_start_ds:valid_end_ds]
    block_valid = block_ds[valid_start_ds:valid_end_ds]
    return block_valid, dm_time, valid_start_ds, valid_end_ds


def plan_slices(block_valid: np.ndarray, slice_len: int, chunk_idx: int) -> list[tuple[int, int, int]]:
    """Return slice boundaries for the valid portion of the block."""
    if getattr(config, 'USE_PLANNED_CHUNKING', False):
        time_reso_ds = config.TIME_RESO * config.DOWN_TIME_RATE
        plan = plan_slices_for_chunk(
            num_samples_decimated=block_valid.shape[0],
            target_duration_ms=config.SLICE_DURATION_MS,
            time_reso_decimated_s=time_reso_ds,
            max_slice_count=getattr(config, 'MAX_SLICE_COUNT', 5000),
            time_tol_ms=getattr(config, 'TIME_TOL_MS', 0.1),
        )
        try:
            from ..logging.chunking_logging import log_slice_plan_summary
            log_slice_plan_summary(chunk_idx, plan)
        except Exception:
            pass
        return [(idx, sl.start_idx, sl.end_idx) for idx, sl in enumerate(plan["slices"])]
    else:
        time_slice = (block_valid.shape[0] + slice_len - 1) // slice_len
        return [(j, j * slice_len, min((j + 1) * slice_len, block_valid.shape[0])) for j in range(time_slice)]





def validate_slice_indices(
    start_idx: int,
    end_idx: int,
    block_shape: int,
    slice_len: int,
    j: int,
    chunk_idx: int
) -> tuple[bool, int, int, str]:
    """Validate and adjust slice indices if needed."""
                                                       
    if start_idx >= block_shape:
        return False, start_idx, end_idx, "Slice out of bounds - no data to process"
    
    if end_idx > block_shape:
                                                       
        end_idx_ajustado = block_shape
        
                                                                 
        if end_idx_ajustado - start_idx < slice_len // 2:
            return False, start_idx, end_idx_ajustado, "Slice too small for effective processing"

        return True, start_idx, end_idx_ajustado, "Slice adjusted - last chunk slice with residual data"

    return True, start_idx, end_idx, "Valid slice"


def create_chunk_directories(
    save_dir: Path,
    fits_path: Path,
    chunk_idx: int
) -> tuple[Path, Path, Path, Path]:
    """Return the directories used to store results for a given chunk.
    
    Returns:
        composite_dir: Directory for composite plots
        detections_dir: Directory for detection images
        patches_dir: Directory for patch images
        summary_dir: Directory for CSV summary files
    """
    file_folder_name = fits_path.stem
    chunk_folder_name = f"chunk{chunk_idx:03d}"
    
    composite_dir = save_dir / "Composite" / file_folder_name / chunk_folder_name
    detections_dir = save_dir / "Detections" / file_folder_name / chunk_folder_name
    patches_dir = save_dir / "Patches" / file_folder_name / chunk_folder_name
    summary_dir = save_dir / "Summary" / file_folder_name
    
    return composite_dir, detections_dir, patches_dir, summary_dir


def get_chunk_processing_parameters(metadata: dict) -> dict:
    """Return derived parameters required to process a chunk."""
                                 
    chunk_samples = int(metadata.get("actual_chunk_size", config.FILE_LENG))
    freq_down = calculate_frequency_downsampled()
    height = calculate_dm_height()
    width_total = calculate_width_total(chunk_samples)
    slice_len, real_duration_ms = calculate_slice_parameters()
    time_slice = calculate_time_slice(width_total, slice_len)
    
                                    
    _ol_raw = int(metadata.get("overlap_left", 0))
    _or_raw = int(metadata.get("overlap_right", 0))
    overlap_left_ds, overlap_right_ds = calculate_overlap_decimated(_ol_raw, _or_raw)
    
    return {
        'freq_down': freq_down,
        'height': height,
        'width_total': width_total,
        'slice_len': slice_len,
        'real_duration_ms': real_duration_ms,
        'time_slice': time_slice,
        'overlap_left_ds': overlap_left_ds,
        'overlap_right_ds': overlap_right_ds,
    }
