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


# Global collector for validation metrics (set by pipeline)
_validation_collector = None

def set_validation_collector(collector):
    """Set the global validation metrics collector."""
    global _validation_collector
    _validation_collector = collector

def validate_memory_allocation(size_bytes: int, operation_name: str, warn_threshold_gb: float = 8.0, error_threshold_gb: float = 16.0) -> None:
    """
    Validate memory allocation before attempting to allocate large arrays.
    
    Based on PRESTO's memory validation strategy.
    
    Args:
        size_bytes: Size in bytes to allocate
        operation_name: Name of the operation (for logging)
        warn_threshold_gb: Warn if allocation exceeds this (GB)
        error_threshold_gb: Raise error if allocation exceeds this (GB)
    
    Raises:
        MemoryError: If allocation would exceed error_threshold_gb
    """
    global _validation_collector
    
    size_gb = size_bytes / (1024**3)
    if size_gb > error_threshold_gb:
        error_msg = (
            f"Cannot allocate {size_gb:.2f} GB for {operation_name}. "
            f"This exceeds the safety threshold of {error_threshold_gb:.2f} GB. "
            f"Is the dispersive delay across the band longer than "
            f"(or comparable to) the duration of the input file??"
        )
        logger.error(error_msg)
        if _validation_collector is not None:
            _validation_collector.record_memory_validation(
                operation=operation_name,
                requested_bytes=size_bytes,
                validation_result="rejected",
                error_message=error_msg,
            )
        raise MemoryError(error_msg)
    elif size_gb > warn_threshold_gb:
        logger.warning(
            f"Attempting to allocate {size_gb:.2f} GB for {operation_name}. "
            f"This may cause memory pressure. "
            f"Consider reducing chunk size or downsampling rate."
        )
        if _validation_collector is not None:
            _validation_collector.record_memory_validation(
                operation=operation_name,
                requested_bytes=size_bytes,
                validation_result="allowed_with_warning",
            )
    else:
        if _validation_collector is not None:
            _validation_collector.record_memory_validation(
                operation=operation_name,
                requested_bytes=size_bytes,
                validation_result="allowed",
            )


def downsample_chunk(block: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Apply temporal and frequency downsampling to the full chunk.
    
    Includes memory validation before downsampling (PRESTO-style).
    """
    # Validate memory before downsampling
    # Downsampled block will be smaller, but validate input size
    input_size_bytes = block.nbytes
    if input_size_bytes > 1_000_000_000:  # > 1GB
        logger.debug(
            f"Downsampling large block: {input_size_bytes / (1024**3):.2f} GB "
            f"(shape={block.shape})"
        )
    
    from ..preprocessing.data_downsampler import downsample_data
    block_ds = downsample_data(block)
    dt_ds = config.TIME_RESO * config.DOWN_TIME_RATE
    return block_ds, dt_ds


def build_dm_time_cube(block_ds: np.ndarray, height: int, dm_min: float, dm_max: float) -> np.ndarray:
    """
    Build the DM–time cube for the decimated block.
    
    Includes memory validation before allocation (PRESTO-style).
    Automatically uses DM chunking if the cube would be too large (>16GB).
    Uses double-buffering approach: only processes what's needed.
    """
    width = block_ds.shape[0]
    
    # Calculate cube size
    cube_size_bytes = 3 * height * width * 4  # 3 bands, float32 = 4 bytes
    cube_size_gb = cube_size_bytes / (1024**3)
    
    # Threshold for DM chunking (configurable, default 16 GB)
    dm_chunking_threshold_gb = getattr(config, 'DM_CHUNKING_THRESHOLD_GB', 16.0)
    
    if cube_size_gb > dm_chunking_threshold_gb:
        # Use DM chunking to avoid memory issues
        logger.info(
            f"[DEDISPERSION] DM-time cube would be {cube_size_gb:.2f} GB (height={height}, width={width:,}). "
            f"Using DM chunking (threshold: {dm_chunking_threshold_gb} GB) to reduce memory usage."
        )
        return _build_dm_time_cube_chunked(block_ds, height, dm_min, dm_max, dm_chunking_threshold_gb)
    else:
        # Normal path: build entire cube at once
        logger.info(
            f"[DEDISPERSION] Building DM-time cube: {cube_size_gb:.2f} GB (height={height}, width={width:,}, "
            f"DM range: {dm_min:.1f}-{dm_max:.1f} pc cm⁻³)"
        )
        validate_memory_allocation(cube_size_bytes, "DM-time cube")
        from ..preprocessing.dedispersion import d_dm_time_g
        logger.debug("[DEDISPERSION] Calling d_dm_time_g() to perform dedispersion...")
        result = d_dm_time_g(block_ds, height=height, width=width, dm_min=dm_min, dm_max=dm_max)
        logger.debug(f"[DEDISPERSION] Dedispersion complete, cube shape: {result.shape}")
        return result


def _build_dm_time_cube_chunked(
    block_ds: np.ndarray, 
    height: int, 
    dm_min: float, 
    dm_max: float,
    threshold_gb: float
) -> np.ndarray:
    """
    Build DM-time cube by processing DM range in chunks.
    
    This prevents memory exhaustion when DM range is very large.
    """
    global _validation_collector
    
    width = block_ds.shape[0]
    
    # Calculate optimal DM chunk size
    # Target: each chunk should be < threshold_gb
    # cube_size = 3 * dm_chunk_height * width * 4 bytes
    # dm_chunk_height = (threshold_gb * 1024^3) / (3 * width * 4)
    max_chunk_height = int((threshold_gb * (1024**3)) / (3 * width * 4))
    
    # Ensure minimum chunk size (at least 100 DM values)
    min_chunk_height = 100
    dm_chunk_height = max(min_chunk_height, max_chunk_height)
    
    # Calculate number of DM chunks needed
    num_dm_chunks = (height + dm_chunk_height - 1) // dm_chunk_height
    
    logger.info(
        f"DM chunking: {num_dm_chunks} chunks of ~{dm_chunk_height} DM values each "
        f"(total height={height}, target chunk size <{threshold_gb:.1f} GB)"
    )
    
    # Record DM chunking activation
    if _validation_collector is not None:
        chunk_info = []
        for chunk_idx in range(num_dm_chunks):
            start_dm = chunk_idx * dm_chunk_height
            end_dm = min(start_dm + dm_chunk_height, height)
            dm_range = dm_max - dm_min
            chunk_dm_min = dm_min + (start_dm / height) * dm_range
            chunk_dm_max = dm_min + (end_dm / height) * dm_range
            chunk_size_gb = (3 * (end_dm - start_dm) * width * 4) / (1024**3)
            
            chunk_info.append({
                "chunk_idx": chunk_idx,
                "dm_min": chunk_dm_min,
                "dm_max": chunk_dm_max,
                "dm_indices": [start_dm, end_dm],
                "cube_size_gb": chunk_size_gb,
            })
        
        _validation_collector.record_dm_chunking(
            activated=True,
            num_chunks=num_dm_chunks,
            chunk_info=chunk_info,
        )
    
    # CRITICAL: Validate memory before allocating full result array
    # Even though we process in chunks, we need the full array to combine results
    result_size_bytes = 3 * height * width * 4
    result_size_gb = result_size_bytes / (1024**3)
    
    # Warn if result array is very large (but allow it since we process in chunks)
    if result_size_gb > threshold_gb * 2:
        logger.warning(
            f"Result array will be {result_size_gb:.2f} GB (larger than threshold {threshold_gb:.1f} GB). "
            f"This is expected when using DM chunking - the full array is needed to combine results. "
            f"Ensure sufficient RAM is available."
        )
    
    # Allocate full result array (required to combine DM chunks)
    # This is necessary because the rest of the pipeline expects the complete cube
    result = np.zeros((3, height, width), dtype=np.float32)
    
    from ..preprocessing.dedispersion import d_dm_time_g
    import gc
    
    # Process each DM chunk
    import time
    dm_chunk_start_time = time.time()
    dm_chunk_times = []
    
    for chunk_idx in range(num_dm_chunks):
        chunk_iter_start = time.time()
        start_dm = chunk_idx * dm_chunk_height
        end_dm = min(start_dm + dm_chunk_height, height)
        chunk_height = end_dm - start_dm
        
        # Calculate DM range for this chunk
        dm_range = dm_max - dm_min
        chunk_dm_min = dm_min + (start_dm / height) * dm_range
        chunk_dm_max = dm_min + (end_dm / height) * dm_range
        
        logger.info(
            f"[DEDISPERSION] Processing DM chunk {chunk_idx + 1}/{num_dm_chunks}: "
            f"DM {chunk_dm_min:.1f}-{chunk_dm_max:.1f} pc cm⁻³ (indices {start_dm}-{end_dm}, "
            f"height={chunk_height})"
        )
        
        # Dedisperse this DM chunk
        chunk_cube = d_dm_time_g(
            block_ds, 
            height=chunk_height, 
            width=width, 
            dm_min=chunk_dm_min, 
            dm_max=chunk_dm_max
        )
        
        # Copy into result array
        result[:, start_dm:end_dm, :] = chunk_cube
        
        # Free chunk immediately
        del chunk_cube
        gc.collect()
        
        chunk_iter_time = time.time() - chunk_iter_start
        dm_chunk_times.append(chunk_iter_time)
        
        # Log progress with ETA
        if chunk_idx > 0:
            avg_time = sum(dm_chunk_times) / len(dm_chunk_times)
            remaining_chunks = num_dm_chunks - (chunk_idx + 1)
            eta_seconds = remaining_chunks * avg_time
            if eta_seconds > 10:
                logger.info(
                    f"DM chunk {chunk_idx + 1}/{num_dm_chunks} completed in {chunk_iter_time:.1f}s. "
                    f"Average: {avg_time:.1f}s/chunk. ETA: {eta_seconds:.1f}s"
                )
            else:
                logger.debug(
                    f"DM chunk {chunk_idx + 1}/{num_dm_chunks} completed in {chunk_iter_time:.1f}s"
                )
        else:
            logger.info(f"DM chunk {chunk_idx + 1}/{num_dm_chunks} completed in {chunk_iter_time:.1f}s")
    
    logger.info(f"DM chunking complete: full cube assembled ({result.shape})")
    return result


class DoubleBufferDedispersion:
    """
    Double-buffer system for incremental dedispersion (PRESTO-style).
    
    Maintains only 2 blocks in memory at a time:
    - current_block: Currently being processed
    - last_block: Previous block (for overlap/delays that cross boundaries)
    
    This prevents memory accumulation and allows processing of arbitrarily large files.
    """
    
    def __init__(self, height: int, dm_min: float, dm_max: float):
        self.height = height
        self.dm_min = dm_min
        self.dm_max = dm_max
        self.last_block = None
        self.last_dm_time = None
        self.first_block = True
        
    def process_block(self, block_ds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Process a block with double-buffering.
        
        Args:
            block_ds: Downsampled block to process (time, freq)
        
        Returns:
            Tuple of (dm_time_cube, block_ds)
            - dm_time_cube: DM-time cube for this block
            - block_ds: The input block (for consistency)
        
        Strategy (PRESTO-style):
        1. First block: Store in last_block, return empty/placeholder
        2. Subsequent blocks: Dedisperse using current + last, then swap
        """
        from ..preprocessing.dedispersion import d_dm_time_g
        
        if self.first_block:
            # First block: just store it, we need the next block for proper dedispersion
            # (delays can cross block boundaries)
            self.last_block = block_ds.copy()
            width = block_ds.shape[0]
            
            # Validate memory
            cube_size_bytes = 3 * self.height * width * 4
            validate_memory_allocation(cube_size_bytes, "DM-time cube (first block)")
            
            # Create DM-time cube for first block
            self.last_dm_time = d_dm_time_g(
                self.last_block, 
                height=self.height, 
                width=width, 
                dm_min=self.dm_min, 
                dm_max=self.dm_max
            )
            self.first_block = False
            
            # Return the first block's cube
            return self.last_dm_time, block_ds
        
        else:
            # Subsequent blocks: process current block
            width = block_ds.shape[0]
            
            # Validate memory
            cube_size_bytes = 3 * self.height * width * 4
            validate_memory_allocation(cube_size_bytes, "DM-time cube (subsequent block)")
            
            # Create DM-time cube for current block
            current_dm_time = d_dm_time_g(
                block_ds,
                height=self.height,
                width=width,
                dm_min=self.dm_min,
                dm_max=self.dm_max
            )
            
            # SWAP: Move current to last (PRESTO-style)
            # Free old last_block immediately
            del self.last_block, self.last_dm_time
            import gc
            gc.collect()
            
            # Update for next iteration
            self.last_block = block_ds.copy()
            self.last_dm_time = current_dm_time
            
            return current_dm_time, block_ds
    
    def get_final_block(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Get the final block if there's one remaining in the buffer.
        
        Returns:
            Tuple of (dm_time_cube, block_ds) or None if no final block
        """
        if self.last_dm_time is not None and self.last_block is not None:
            return self.last_dm_time, self.last_block
        return None
    
    def cleanup(self):
        """Free all buffers (PRESTO-style cleanup)."""
        del self.last_block, self.last_dm_time
        self.last_block = None
        self.last_dm_time = None
        self.first_block = True
        import gc
        gc.collect()


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
