# This module aggregates the core pipeline interfaces.

"""Core building blocks used to assemble the FRB detection pipeline."""

from .detection_engine import process_slice_with_multiple_bands
from .pipeline_parameters import (
    calculate_frequency_downsampled,
    calculate_dm_height,
    calculate_width_total,
    calculate_slice_parameters,
    calculate_time_slice,
    calculate_overlap_decimated,
    calculate_absolute_slice_time,
)
from .data_flow_manager import (
    downsample_chunk,
    build_dm_time_cube,
    trim_valid_window,
    plan_slices,
    validate_slice_indices,
    create_chunk_directories,
    get_chunk_processing_parameters,
)

__all__ = [
                           
    'process_slice_with_multiple_bands',
    
                             
    'calculate_frequency_downsampled',
    'calculate_dm_height', 
    'calculate_width_total',
    'calculate_slice_parameters',
    'calculate_time_slice',
    'calculate_overlap_decimated',
    'calculate_absolute_slice_time',
    
                               
    'downsample_chunk',
    'build_dm_time_cube',
    'trim_valid_window',
    'plan_slices',
    'validate_slice_indices',
    'create_chunk_directories',
    'get_chunk_processing_parameters',
]
