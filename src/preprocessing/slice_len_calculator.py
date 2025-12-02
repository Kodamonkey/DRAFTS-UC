# This module calculates chunking and slice length parameters.

"""Slice length calculator for FRB pipeline - dynamically calculates optimal temporal segmentation."""
                          
import logging
import math
from typing import Optional, Tuple

                     
import numpy as np
import psutil

               
from ..config import config

              
logger = logging.getLogger(__name__)


def calculate_slice_len_from_duration() -> Tuple[int, float]:
    """
    Calculates SLICE_LEN dynamically based on SLICE_DURATION_MS and file metadata.
    
    Inverse formula (decimated domain):
        dt_ds = TIME_RESO × DOWN_TIME_RATE
        SLICE_LEN = floor( (SLICE_DURATION_MS/1000) / dt_ds + 0.5 )  # stable round half up
    
    CRITICAL: Ensures slice_len is at least 512 samples to support patch extraction
    for ResNet classification (which requires 512x512 patches).
    
    Returns:
        Tuple[int, float]: (calculated_slice_len, real_duration_ms)
    """
    if config.TIME_RESO <= 0 or config.DOWN_TIME_RATE < 1:
        logger.warning("TIME_RESO not configured, using SLICE_LEN_MIN")
        return config.SLICE_LEN_MIN, config.SLICE_DURATION_MS
    
    # Minimum slice_len required for patch extraction (ResNet expects 512 samples)
    # We add a safety margin to account for dedispersion delays
    PATCH_LEN_REQUIRED = 512
    DEDISPERSION_DELAY_MARGIN = 100  # Extra samples for dispersion correction
    MIN_SLICE_LEN_FOR_PATCHES = PATCH_LEN_REQUIRED + DEDISPERSION_DELAY_MARGIN
    
                                                  
                                                   
    target_duration_s = config.SLICE_DURATION_MS / 1000.0
    dt_ds = config.TIME_RESO * config.DOWN_TIME_RATE
                                                                                   
    calculated_slice_len = int(math.floor((target_duration_s / dt_ds) + 0.5))
    
    # CRITICAL: Ensure slice_len is at least MIN_SLICE_LEN_FOR_PATCHES
    # This guarantees we always have enough data for patch extraction
    calculated_slice_len = max(calculated_slice_len, MIN_SLICE_LEN_FOR_PATCHES)
                             
    slice_len = max(config.SLICE_LEN_MIN, min(config.SLICE_LEN_MAX, calculated_slice_len))
    
    # Warn if we had to increase slice_len beyond the target duration
    if slice_len > calculated_slice_len and calculated_slice_len < MIN_SLICE_LEN_FOR_PATCHES:
        logger.info(
            f"Increased slice_len from {calculated_slice_len} to {slice_len} "
            f"to ensure sufficient data for patch extraction (minimum {MIN_SLICE_LEN_FOR_PATCHES} samples)"
        )
                                                            
    real_duration_s = slice_len * dt_ds
    real_duration_ms = real_duration_s * 1000.0
    
                                                        
    config.SLICE_LEN = slice_len
    
                                               
    if abs(real_duration_ms - config.SLICE_DURATION_MS) > 5.0:
        # This is expected when minimum slice_len requirement (612 samples) exceeds target duration
        # The minimum ensures we always have enough data for 512-sample patches required by ResNet
        if real_duration_ms > config.SLICE_DURATION_MS:
            logger.info(
                f"Slice duration adjusted: target={config.SLICE_DURATION_MS:.1f} ms → "
                f"actual={real_duration_ms:.1f} ms (slice_len={slice_len} samples). "
                f"This ensures sufficient data for patch extraction (minimum 612 samples required)."
            )
        else:
            logger.warning(
                f"Significant difference between target ({config.SLICE_DURATION_MS:.1f} ms) "
                f"and obtained ({real_duration_ms:.1f} ms). "
                f"This may be due to minimum slice_len requirement for patch extraction."
            )
    
    return slice_len, real_duration_ms


def calculate_memory_safe_chunk_size(
    slice_len: Optional[int] = None,
    safety_margin: float = 0.8,
) -> tuple[int, dict]:
    """
    Calculate memory-safe chunk size using adaptive budgeting strategy.
    
    This implements the "Dynamic Intelligent Budgeting" system:
    - Phase A: Calculate cost per temporal sample (based on DM cube size)
    - Phase B: Determine optimal time chunk based on available RAM
    - Phase C: Validate against physical constraints (overlap + slice_len)
    
    Decision Logic:
    - Scenario 1 (Ideal): max_samples > required_min_size
      → Use max_samples (cube fits in RAM, no DM chunking needed)
    - Scenario 2 (Extreme): max_samples < required_min_size
      → Use required_min_size (DM chunking will activate automatically)
    
    Args:
        slice_len: Number of samples in a slice. If None, derived from config.
        safety_margin: Fraction of available memory to use (default 0.8 = 80%).
    
    Returns:
        tuple[int, dict]: (safe_chunk_samples, diagnostics_dict)
    """
    if slice_len is None:
        slice_len, _ = calculate_slice_len_from_duration()
    
    if config.FILE_LENG <= 0 or config.FREQ_RESO <= 0 or config.TIME_RESO <= 0:
        logger.warning("File metadata not available, using default chunk size")
        return slice_len * 200, {"reason": "metadata_unavailable"}
    
    # ===== PHASE A: Calculate Cost and Budget =====
    from ..core.pipeline_parameters import calculate_dm_height, calculate_frequency_downsampled
    
    # Get DM cube height
    height_dm = calculate_dm_height()
    
    # Get frequency range for overlap calculation
    try:
        freq_ds = calculate_frequency_downsampled()
        nu_min = float(freq_ds.min())
        nu_max = float(freq_ds.max())
    except Exception:
        logger.warning("Could not calculate frequency range, using defaults")
        nu_min = 1000.0  # MHz
        nu_max = 2000.0  # MHz
    
    # Calculate overlap required (maximum dispersion delay)
    dt_max_sec = 4.1488e3 * config.DM_max * (nu_min**-2 - nu_max**-2)
    overlap_raw = max(0, int(np.ceil(dt_max_sec / config.TIME_RESO)))
    overlap_decimated = overlap_raw // max(1, config.DOWN_TIME_RATE)
    
    # Calculate cost per temporal sample
    # DM-time cube: (3, height_dm, width) where width = chunk_samples
    # Cost = 3 * height_dm * 4 bytes per sample
    cost_per_sample_bytes = 3 * height_dm * 4  # float32 = 4 bytes
    
    # ===== PHASE B: Determine Optimal Time Chunk =====
    # Get available system RAM
    vm = psutil.virtual_memory()
    available_ram_bytes = vm.available
    
    # Get available GPU VRAM if available
    available_vram_bytes = 0
    gpu_available = False
    try:
        import torch
        if torch is not None and torch.cuda.is_available():
            gpu_available = True
            vram_total = torch.cuda.get_device_properties(0).total_memory
            vram_reserved = torch.cuda.memory_reserved(0)
            available_vram_bytes = vram_total - vram_reserved
    except Exception:
        pass
    
    # Calculate usable memory with safety margin
    max_ram_fraction = getattr(config, 'MAX_RAM_FRACTION', 0.25)
    overhead_factor = getattr(config, 'OVERHEAD_FACTOR', 1.3)
    
    if gpu_available and available_vram_bytes > 0:
        usable_ram = (available_ram_bytes * max_ram_fraction * safety_margin) / overhead_factor
        usable_vram = (available_vram_bytes * 0.7 * safety_margin) / overhead_factor
        usable_bytes = usable_ram + usable_vram
    else:
        usable_bytes = (available_ram_bytes * max_ram_fraction * safety_margin) / overhead_factor
    
    # Calculate maximum samples that fit in available memory
    if cost_per_sample_bytes <= 0:
        logger.warning("Invalid cost per sample, using fallback")
        max_samples = slice_len * 100
    else:
        max_samples = int(usable_bytes / cost_per_sample_bytes)
    
    # ===== PHASE C: Validate Against Physical Constraints =====
    # Required minimum size = overlap + slice_len
    required_min_size = overlap_decimated + slice_len
    
    # Decision: Choose between max_samples (ideal) and required_min_size (fallback)
    # Note: max_samples and required_min_size are in DECIMATED domain
    
    # Calculate required RAW samples to produce required_min_size decimated samples
    # We need enough RAW samples so that after downsampling we have at least required_min_size
    required_min_raw = required_min_size * max(1, config.DOWN_TIME_RATE)
    
    if max_samples > required_min_size:
        # Scenario 1: Ideal case - we can fit the cube in RAM
        # Convert decimated capacity to raw samples
        safe_chunk_samples_raw = max_samples * max(1, config.DOWN_TIME_RATE)
        scenario = "ideal"
        reason = (
            f"Memory allows {max_samples:,} decimated samples "
            f"(>{required_min_size:,} required). "
            f"DM-time cube will fit in RAM."
        )
    else:
        # Scenario 2: Extreme case - overlap is too large, use minimum
        safe_chunk_samples_raw = required_min_raw
        scenario = "extreme"
        reason = (
            f"Memory allows {max_samples:,} decimated samples, but physical constraint "
            f"requires {required_min_size:,} (overlap={overlap_decimated:,} + slice_len={slice_len}). "
            f"DM chunking will activate automatically."
        )
    
    # Align to slice_len * DOWN_TIME_RATE (to be safe with downsampling boundaries)
    # Actually, usually we align to a convenient block size, but aligning to slice_len is fine
    # provided we consider the downsampling factor.
    
    # IMPORTANT: The pipeline expects this value to be the RAW chunk size.
    # The chunking logic often aligns to slice_len, but slice_len is usually defined in decimated domain 
    # if calculated from duration, OR raw domain if manual.
    # Let's assume slice_len is in DECIMATED domain (since it's used for the neural net input).
    
    # Align to a multiple of (slice_len * down_rate) to ensure integer slices after downsampling
    alignment_block = slice_len * max(1, config.DOWN_TIME_RATE)
    
    safe_chunk_samples = (safe_chunk_samples_raw // alignment_block) * alignment_block
    if safe_chunk_samples < required_min_raw:
        # If alignment reduced it below minimum, add one block
        safe_chunk_samples += alignment_block
        
    # Safety: enforce maximum chunk size limit from config.yaml
    # Use MAX_CHUNK_SAMPLES from config (default 1M if not set)
    max_chunk_limit = getattr(config, 'MAX_CHUNK_SAMPLES', 1_000_000)
    if safe_chunk_samples > max_chunk_limit:
        safe_chunk_samples = (max_chunk_limit // alignment_block) * alignment_block
        logger.warning(
            f"Chunk size limited to {safe_chunk_samples:,} samples "
            f"(max limit from config: {max_chunk_limit:,})"
        )
    
    # CRITICAL: Additional safety limit based on DM-time cube size to prevent OOM
    # The chunk that arrives at build_dm_time_cube includes overlap on both sides
    # We must ensure the resulting cube (with overlap) doesn't exceed max_dm_cube_size_gb
    max_cube_size_gb = getattr(config, 'MAX_DM_CUBE_SIZE_GB', 2.0)  # Default 2 GB
    max_result_size_gb = max_cube_size_gb * 4  # Allow up to 4x threshold for result array
    
    # Calculate max decimated samples that fit in the result array limit
    # The chunk that arrives includes: chunk_samples_decimated + overlap_total_decimated
    max_decimated_with_overlap = int((max_result_size_gb * 1024**3) / (3 * height_dm * 4))
    
    # The overlap_total_decimated is 2 * overlap_decimated (left + right)
    overlap_total_decimated = 2 * overlap_decimated
    
    # Calculate max chunk size (valid samples, without overlap)
    max_chunk_decimated = max_decimated_with_overlap - overlap_total_decimated
    
    # Ensure minimum size (must be at least slice_len)
    if max_chunk_decimated < slice_len:
        logger.warning(
            f"DM cube size limit ({max_cube_size_gb} GB) is too restrictive. "
            f"Minimum chunk size ({slice_len} decimated) would exceed limit. "
            f"Consider increasing max_dm_cube_size_gb or reducing DM range."
        )
        max_chunk_decimated = slice_len
    else:
        max_chunk_decimated = (max_chunk_decimated // slice_len) * slice_len  # Align to slice_len
    
    # Convert to RAW samples
    max_chunk_by_cube_raw = max_chunk_decimated * max(1, config.DOWN_TIME_RATE)
    max_chunk_by_cube_raw = (max_chunk_by_cube_raw // alignment_block) * alignment_block  # Align
    
    # Apply the cube size limit
    if safe_chunk_samples > max_chunk_by_cube_raw:
        safe_chunk_samples = max_chunk_by_cube_raw
        # Calculate expected cube size for logging
        expected_decimated = safe_chunk_samples // max(1, config.DOWN_TIME_RATE)
        expected_decimated_with_overlap = expected_decimated + overlap_total_decimated
        expected_cube_gb = (3 * height_dm * expected_decimated_with_overlap * 4) / (1024**3)
        logger.warning(
            f"Chunk size further limited to {safe_chunk_samples:,} RAW samples "
            f"({expected_decimated:,} decimated, ~{expected_decimated_with_overlap:,} with overlap) "
            f"to keep DM-time cube result array < {max_result_size_gb:.1f} GB "
            f"(DM height={height_dm:,}, overlap_total={overlap_total_decimated:,}, expected cube={expected_cube_gb:.2f} GB, "
            f"max_dm_cube_size_gb={max_cube_size_gb:.1f} GB)"
        )
    
    # Calculate expected cube size (using DECIMATED samples count)
    # safe_chunk_samples is RAW, so divide by down_rate
    safe_samples_decimated = safe_chunk_samples // max(1, config.DOWN_TIME_RATE)
    # Account for overlap in expected size (chunk arrives with overlap)
    expected_decimated_with_overlap = safe_samples_decimated + overlap_total_decimated
    expected_cube_gb = (expected_decimated_with_overlap * cost_per_sample_bytes) / (1024**3)
    dm_chunking_threshold_gb = getattr(config, 'DM_CHUNKING_THRESHOLD_GB', 16.0)
    will_use_dm_chunking = expected_cube_gb > dm_chunking_threshold_gb
    
    # Build diagnostics
    diagnostics = {
        "scenario": scenario,
        "reason": reason,
        "safe_chunk_samples": safe_chunk_samples,
        "max_samples": max_samples, # This is decimated capacity
        "required_min_size": required_min_size, # This is decimated requirement
        "overlap_decimated": overlap_decimated,
        "overlap_total_decimated": overlap_total_decimated,
        "slice_len": slice_len,
        "height_dm": height_dm,
        "cost_per_sample_bytes": cost_per_sample_bytes,
        "expected_cube_gb": expected_cube_gb,
        "max_cube_size_gb": max_cube_size_gb,
        "max_result_size_gb": max_result_size_gb,
        "max_chunk_by_cube_raw": max_chunk_by_cube_raw,
        "will_use_dm_chunking": will_use_dm_chunking,
        "available_ram_gb": available_ram_bytes / (1024**3),
        "available_vram_gb": available_vram_bytes / (1024**3) if gpu_available else 0,
        "usable_bytes_gb": usable_bytes / (1024**3),
    }
    
    logger.info(
        f"Memory-safe chunk size: {safe_chunk_samples:,} samples "
        f"(scenario={scenario}, cube={expected_cube_gb:.2f} GB, "
        f"DM_chunking={'yes' if will_use_dm_chunking else 'no'})"
    )
    logger.debug(f"Budget details: {reason}")
    
    return safe_chunk_samples, diagnostics


def calculate_optimal_chunk_size(slice_len: Optional[int] = None) -> int:
    """Determine how many samples a chunk can contain based on available hardware.

    This function considers:
    - Available RAM (system memory)
    - Available VRAM (GPU memory if available)
    - Memory overhead from dedispersion (DM-time cube is much larger than input)
    - Safety margins to prevent OOM

    The calculation ensures chunks are always processable regardless of file size.

    Args:
        slice_len: Number of samples in a slice. If ``None`` it is derived from
            :data:`config.SLICE_DURATION_MS`.

    Returns:
        int: Samples per chunk.
    """
    if slice_len is None:
        slice_len, _ = calculate_slice_len_from_duration()

    if config.FILE_LENG <= 0 or config.FREQ_RESO <= 0 or config.TIME_RESO <= 0:
        logger.warning("File metadata not available, using default chunk size")
        return slice_len * 200

    # Calculate downsampled dimensions
    total_samples = config.FILE_LENG // max(1, config.DOWN_TIME_RATE)                           
    n_channels = max(1, config.FREQ_RESO // max(1, config.DOWN_FREQ_RATE))                    
    bytes_per_sample = 4 * n_channels  # float32 = 4 bytes
    
    # Get available system RAM
    vm = psutil.virtual_memory()
    available_ram_bytes = vm.available
    
    # Get available GPU VRAM if available
    available_vram_bytes = 0
    gpu_available = False
    try:
        import torch
        if torch is not None and torch.cuda.is_available():
            gpu_available = True
            # Get free VRAM (reserved memory is already allocated)
            vram_total = torch.cuda.get_device_properties(0).total_memory
            vram_allocated = torch.cuda.memory_allocated(0)
            vram_reserved = torch.cuda.memory_reserved(0)
            available_vram_bytes = vram_total - vram_reserved
            logger.debug(
                f"GPU VRAM: {vram_total/(1024**3):.2f} GB total, "
                f"{available_vram_bytes/(1024**3):.2f} GB available"
            )
    except Exception:
        pass
    
    # Calculate DM cube overhead
    # DM-time cube size: (3, height_dm, width) where width = chunk_samples
    from ..core.pipeline_parameters import calculate_dm_height
    height_dm = calculate_dm_height()
    dm_cube_overhead = 3 * height_dm  # 3 channels (total, mid, diff) × DM height
    
    # Memory requirements per chunk sample:
    # 1. Input block: (time, freq) = 1 × n_channels
    # 2. DM-time cube: (3, height_dm, time) = 3 × height_dm × 1
    # 3. Intermediate buffers: ~20% overhead
    memory_per_sample = bytes_per_sample * (1 + dm_cube_overhead) * 1.2
    
    # Use combined memory (RAM + VRAM) with safety factor
    max_ram_fraction = getattr(config, 'MAX_RAM_FRACTION', 0.25)
    overhead_factor = getattr(config, 'OVERHEAD_FACTOR', 1.3)
    
    # Calculate usable memory
    if gpu_available and available_vram_bytes > 0:
        # Use both RAM and VRAM, but prioritize RAM for input, VRAM for processing
        # Assume 70% of processing can use VRAM, 30% needs RAM
        usable_ram = (available_ram_bytes * max_ram_fraction) / overhead_factor
        usable_vram = (available_vram_bytes * 0.7) / overhead_factor  # Use 70% of VRAM
        usable_bytes = usable_ram + usable_vram
        logger.debug(
            f"Using combined memory: RAM={usable_ram/(1024**3):.2f} GB, "
            f"VRAM={usable_vram/(1024**3):.2f} GB, total={usable_bytes/(1024**3):.2f} GB"
        )
    else:
        # CPU-only: use only RAM
        usable_bytes = (available_ram_bytes * max_ram_fraction) / overhead_factor
        logger.debug(f"CPU-only mode: usable RAM={usable_bytes/(1024**3):.2f} GB")
    
    # Calculate maximum chunk size based on memory
    max_chunk_samples = int(usable_bytes / memory_per_sample)
    
    # Ensure minimum chunk size (at least 10 slices)
    min_chunk_samples = slice_len * 10
    
    # For very large files, always chunk (don't try to load entire file)
    file_bytes = total_samples * bytes_per_sample
    if file_bytes > available_ram_bytes * 0.5:  # File > 50% of RAM
        chunk_samples = max(min_chunk_samples, min(max_chunk_samples, total_samples))
        logger.info(
            f"Large file detected ({file_bytes/(1024**3):.2f} GB), "
            f"using chunked processing"
        )
    else:
        # Small file: can process in single chunk if it fits
        if max_chunk_samples >= total_samples:
            chunk_samples = total_samples
            logger.info("File fits in memory, processing in single chunk")
        else:
            chunk_samples = max(min_chunk_samples, max_chunk_samples)

    # Align to slice_len
    chunk_samples = (chunk_samples // slice_len) * slice_len
    if chunk_samples == 0:
        chunk_samples = slice_len

    # Safety: enforce maximum chunk size limit from config.yaml
    # Use MAX_CHUNK_SAMPLES from config (default 1M if not set)
    max_chunk_limit = getattr(config, 'MAX_CHUNK_SAMPLES', 1_000_000)
    if chunk_samples > max_chunk_limit:
        chunk_samples = (max_chunk_limit // slice_len) * slice_len
        logger.warning(
            f"Chunk size limited to {chunk_samples:,} samples "
            f"(max limit from config: {max_chunk_limit:,})"
        )
    
    # Additional safety: limit based on DM-time cube size to prevent OOM
    # Calculate maximum chunk size based on DM cube size limit
    # IMPORTANT: The limit must be calculated for DECIMATED samples, not RAW
    # because the DM-time cube is built from the decimated block
    # CRITICAL: The chunk that arrives includes overlap, so we must account for that
    from ..core.pipeline_parameters import calculate_dm_height
    height_dm = calculate_dm_height()
    max_cube_size_gb = getattr(config, 'MAX_DM_CUBE_SIZE_GB', 2.0)  # Default 2 GB
    
    # The result array can be up to 4x the threshold, so we need to be more conservative
    # Limit to ensure result array stays within 4x threshold
    max_result_size_gb = max_cube_size_gb * 4
    
    # Calculate max decimated samples that fit in the result array limit
    # CRITICAL: The chunk that arrives at build_dm_time_cube includes overlap on both sides
    # The block emitted has: chunk_samples_raw + overlap_left_raw + overlap_right_raw
    # After downsampling: (chunk_samples_raw + overlap_total_raw) / DOWN_TIME_RATE
    #                    = chunk_samples_decimated + overlap_total_decimated
    # 
    # We need: (chunk_samples_decimated + overlap_total_decimated) <= max_decimated_with_overlap
    # Therefore: chunk_samples_decimated <= max_decimated_with_overlap - overlap_total_decimated
    
    # Calculate max decimated samples including overlap
    max_decimated_with_overlap = int((max_result_size_gb * 1024**3) / (3 * height_dm * 4))
    
    # The overlap_total_decimated is approximately 2 * overlap_decimated (left + right)
    # But be conservative and use the actual calculated overlap
    overlap_total_decimated = 2 * overlap_decimated  # Left + right overlap
    
    # Calculate max chunk size (valid samples, without overlap)
    max_chunk_decimated = max_decimated_with_overlap - overlap_total_decimated
    
    # Ensure minimum size (must be at least slice_len)
    max_chunk_decimated = max(max_chunk_decimated, slice_len)
    max_chunk_decimated = (max_chunk_decimated // slice_len) * slice_len  # Align to slice_len
    
    # Convert to RAW samples (multiply by downsampling rate)
    max_chunk_by_cube_raw = max_chunk_decimated * max(1, config.DOWN_TIME_RATE)
    alignment = slice_len * max(1, config.DOWN_TIME_RATE)
    max_chunk_by_cube_raw = (max_chunk_by_cube_raw // alignment) * alignment  # Align
    
    if chunk_samples > max_chunk_by_cube_raw:
        chunk_samples = max_chunk_by_cube_raw
        # Calculate expected cube size in decimated domain for logging
        expected_decimated = chunk_samples // max(1, config.DOWN_TIME_RATE)
        # Account for overlap in expected size (chunk arrives with overlap)
        expected_decimated_with_overlap = expected_decimated + overlap_total_decimated
        expected_cube_gb = (3 * height_dm * expected_decimated_with_overlap * 4) / (1024**3)
        logger.warning(
            f"Chunk size further limited to {chunk_samples:,} RAW samples "
            f"({expected_decimated:,} decimated, ~{expected_decimated_with_overlap:,} with overlap) "
            f"to keep DM-time cube result array < {max_result_size_gb:.1f} GB "
            f"(DM height={height_dm:,}, overlap_total={overlap_total_decimated:,}, expected cube={expected_cube_gb:.2f} GB, "
            f"max_dm_cube_size_gb={max_cube_size_gb:.1f} GB)"
        )

    chunk_duration_sec = chunk_samples * config.TIME_RESO * config.DOWN_TIME_RATE
    slices_per_chunk = chunk_samples // slice_len
    chunk_size_gb = (chunk_samples * memory_per_sample) / (1024**3)
    
    logger.info(
        f"Optimal chunk computed: {chunk_samples:,} samples "
        f"({chunk_duration_sec:.1f}s, {slices_per_chunk} slices, "
        f"~{chunk_size_gb:.2f} GB memory, DM height={height_dm})"
    )

    return chunk_samples


def get_processing_parameters() -> dict:
    """Automatically compute all chunking and slicing parameters.

    Using :data:`config.SLICE_DURATION_MS`, the function determines ``slice_len``
    and the maximum number of samples a chunk can handle without exhausting
    memory. It also calculates the total number of chunks and slices along with
    residual samples that do not form a complete slice.

    Returns:
        dict: Calculated processing parameters.
    """
    slice_len, real_duration_ms = calculate_slice_len_from_duration()                                        
                                                                   
    if getattr(config, 'USE_PLANNED_CHUNKING', False):
                                                                         
        down_time_rate = max(1, config.DOWN_TIME_RATE)
        down_freq_rate = max(1, config.DOWN_FREQ_RATE)
        total_channels_downsampled = max(1, config.FREQ_RESO // down_freq_rate)
        bytes_per_sample = 4 * total_channels_downsampled           

                                
        # Calculate memory considering dedispersion overhead
        from ..core.pipeline_parameters import calculate_dm_height
        height_dm = calculate_dm_height()
        dm_cube_overhead = 3 * height_dm  # 3 channels × DM height
        memory_per_sample = bytes_per_sample * (1 + dm_cube_overhead) * 1.2  # Include 20% overhead
                                
        if getattr(config, 'MAX_CHUNK_BYTES', None):
            usable_bytes = config.MAX_CHUNK_BYTES / max(1.0, getattr(config, 'OVERHEAD_FACTOR', 1.3))
        else:
            vm = psutil.virtual_memory()
            available_ram_bytes = vm.available
            
            # Get GPU VRAM if available
            available_vram_bytes = 0
            try:
                import torch
                if torch is not None and torch.cuda.is_available():
                    vram_total = torch.cuda.get_device_properties(0).total_memory
                    vram_reserved = torch.cuda.memory_reserved(0)
                    available_vram_bytes = vram_total - vram_reserved
            except Exception:
                pass
            
            max_ram_fraction = getattr(config, 'MAX_RAM_FRACTION', 0.25)
            overhead_factor = getattr(config, 'OVERHEAD_FACTOR', 1.3)
            
            if available_vram_bytes > 0:
                # Combined RAM + VRAM
                usable_ram = (available_ram_bytes * max_ram_fraction) / overhead_factor
                usable_vram = (available_vram_bytes * 0.7) / overhead_factor
                usable_bytes = usable_ram + usable_vram
            else:
                # CPU-only
                usable_bytes = (available_ram_bytes * max_ram_fraction) / overhead_factor

        nsamp_max = max(1, int(usable_bytes // memory_per_sample))

                                                                         
        if nsamp_max < slice_len:
            chunk_samples = slice_len
        else:
                                                                       
            chunk_samples = (nsamp_max // slice_len) * slice_len
            if chunk_samples == 0:
                chunk_samples = slice_len

                                        
        try:
            from ..logging.chunking_logging import log_chunk_budget
            import psutil as _ps
            vm = _psutil = psutil.virtual_memory()
            log_chunk_budget({
                'bytes_per_sample': bytes_per_sample,
                'available_bytes': vm.available,
                'usable_bytes': usable_bytes,
                'nsamp_max_raw': nsamp_max,
                'nsamp_max_aligned': (nsamp_max // slice_len) * slice_len,
                'slice_len': slice_len,
                'chunk_samples': chunk_samples,
                'down_time_rate': down_time_rate,
                'down_freq_rate': down_freq_rate,
            })
        except Exception:
            pass
    else:
        # Use the new adaptive budgeting system for more accurate chunk size calculation
        # This ensures we never exceed available RAM, even with large DM ranges
        try:
            safe_chunk_samples, _ = calculate_memory_safe_chunk_size(slice_len)
            chunk_samples = safe_chunk_samples
            logger.debug(
                f"Using adaptive budgeting for chunk size calculation: {chunk_samples:,} samples"
            )
        except Exception as e:
            # Fallback to old method if new method fails
            logger.warning(
                f"Adaptive budgeting failed ({e}), falling back to calculate_optimal_chunk_size"
            )
        chunk_samples = calculate_optimal_chunk_size(slice_len)

                                                                                   
                                                 
                                                                                   
    
                                     
    total_samples_original = config.FILE_LENG if config.FILE_LENG > 0 else 0
    total_channels_original = config.FREQ_RESO if config.FREQ_RESO > 0 else 0
    time_reso_original = config.TIME_RESO if config.TIME_RESO > 0 else 0.000064
    
                                         
    down_time_rate = max(1, config.DOWN_TIME_RATE)
    down_freq_rate = max(1, config.DOWN_FREQ_RATE)
    total_samples_downsampled = total_samples_original // down_time_rate
    total_channels_downsampled = total_channels_original // down_freq_rate
    time_reso_downsampled = time_reso_original * down_time_rate
    
                          
    total_duration_sec = total_samples_original * time_reso_original
    total_duration_min = total_duration_sec / 60.0
    
                                              
    samples_per_slice = slice_len
    slices_per_chunk = chunk_samples // slice_len
    total_slices = total_samples_downsampled // slice_len
    leftover_samples = total_samples_downsampled % slice_len
    
                                            
    total_chunks = (total_samples_downsampled + chunk_samples - 1) // chunk_samples
    chunk_duration_sec = chunk_samples * time_reso_downsampled
    
                       
    bytes_per_sample = 4 * total_channels_downsampled           
    total_file_size_gb = (total_samples_downsampled * bytes_per_sample) / (1024**3)
    chunk_size_gb = (chunk_samples * bytes_per_sample) / (1024**3)
    
                        
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    
                                                  
    can_load_full_file = total_file_size_gb <= available_memory_gb * 0.8

                                                        
                                                                                      
                                                                                         
                                                                                          
    stream_chunk_samples_original = chunk_samples
    stream_chunk_samples_decimated = max(1, stream_chunk_samples_original // down_time_rate)
    slices_per_chunk_stream_estimate = stream_chunk_samples_decimated // slice_len
    chunk_duration_stream_sec = stream_chunk_samples_original * time_reso_original
    
    parameters = {
                                
        'slice_duration_ms_target': config.SLICE_DURATION_MS,
        'down_time_rate': down_time_rate,
        'down_freq_rate': down_freq_rate,
        
                                         
        'total_samples_original': total_samples_original,
        'total_channels_original': total_channels_original,
        'time_reso_original': time_reso_original,
        'total_duration_sec': total_duration_sec,
        'total_duration_min': total_duration_min,
        
                                             
        'total_samples_downsampled': total_samples_downsampled,
        'total_channels_downsampled': total_channels_downsampled,
        'time_reso_downsampled': time_reso_downsampled,
        
                               
        'slice_len': slice_len,
        'slice_duration_ms_real': real_duration_ms,
        'samples_per_slice': samples_per_slice,
                                                                
                                                                
        'chunk_samples_stream_original': stream_chunk_samples_original,
        'chunk_samples_stream_decimated': stream_chunk_samples_decimated,
        'slices_per_chunk_stream_estimate': slices_per_chunk_stream_estimate,
        'chunk_duration_stream_sec': chunk_duration_stream_sec,
        'chunk_samples': chunk_samples,
        'chunk_duration_sec': chunk_duration_sec,
        'slices_per_chunk': min(slices_per_chunk, total_slices),
        'total_chunks': total_chunks,
        'total_slices': total_slices,
        'leftover_samples': leftover_samples,
        
                                
        'total_file_size_gb': total_file_size_gb,
        'chunk_size_gb': chunk_size_gb,
        'available_memory_gb': available_memory_gb,
        'total_memory_gb': total_memory_gb,
        'can_load_full_file': can_load_full_file,
        
                         
        'memory_optimized': True,
        'has_leftover_samples': leftover_samples > 0,
        
                                                               
        'slice_duration_ms': real_duration_ms,                                            
        'total_duration_sec': total_duration_sec,                                            
    }

    return parameters


def update_slice_len_dynamic():
    """Update ``config.SLICE_LEN`` based on ``SLICE_DURATION_MS``.

    Must be called after file metadata is loaded.
    """
    slice_len, real_duration_ms = calculate_slice_len_from_duration()
    
                                              
    try:
        from ..logging.logging_config import get_global_logger
        global_logger = get_global_logger()
        global_logger.slice_config({
            'target_ms': config.SLICE_DURATION_MS,
            'slice_len': slice_len,
            'real_ms': real_duration_ms
        })
    except ImportError:
                                  
        logger.info(f"Slice configured: {slice_len} samples = {real_duration_ms:.1f} ms")
    
    return slice_len, real_duration_ms


def validate_processing_parameters(parameters: dict) -> bool:
    """Validate that the calculated parameters are reasonable.

    Args:
        parameters: Dictionary with processing parameters

    Returns:
        bool: True if parameters are valid
    """
    errors = []
    
                       
    if parameters['slice_len'] < config.SLICE_LEN_MIN:
        errors.append(f"slice_len ({parameters['slice_len']}) < minimum ({config.SLICE_LEN_MIN})")
    
    if parameters['slice_len'] > config.SLICE_LEN_MAX:
        errors.append(f"slice_len ({parameters['slice_len']}) > maximum ({config.SLICE_LEN_MAX})")
    
                           
    # Allow chunk_samples = slice_len in extreme memory-constrained scenarios
    # but warn if it's less than 2 slices (which would be problematic)
    if parameters['chunk_samples'] < parameters['slice_len'] * 2:
        errors.append(f"chunk_samples ({parameters['chunk_samples']}) too small for {parameters['slice_len']} slice_len (minimum: {parameters['slice_len'] * 2})")
    elif parameters['chunk_samples'] < parameters['slice_len'] * 10:
        # Warn but don't error: this is acceptable in memory-constrained scenarios
        logger.warning(f"chunk_samples ({parameters['chunk_samples']}) is small (only {parameters['slices_per_chunk']:.1f} slices per chunk). This may impact efficiency but is acceptable in memory-constrained scenarios.")
    
    if parameters['chunk_samples'] > 50_000_000:                       
        errors.append(f"chunk_samples ({parameters['chunk_samples']}) too large")
    
    # Allow slices_per_chunk >= 1 in extreme scenarios, but warn if < 2
    if parameters['slices_per_chunk'] < 1:
        errors.append(f"slices_per_chunk ({parameters['slices_per_chunk']}) too small (minimum: 1)")
    elif parameters['slices_per_chunk'] < 2:
        logger.warning(f"slices_per_chunk ({parameters['slices_per_chunk']}) is very small. This may impact efficiency but is acceptable in memory-constrained scenarios.")
    
    if parameters['slices_per_chunk'] > 5000:                                            
        errors.append(f"slices_per_chunk ({parameters['slices_per_chunk']}) too large")

    leftover = parameters.get('leftover_samples', 0)
    if leftover >= parameters['slice_len']:
        errors.append(
            f"leftover_samples ({leftover}) should be less than slice_len ({parameters['slice_len']})"
        )
    
    if errors:
        logger.error("Invalid processing parameters:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    return True

