"""Módulo core del pipeline de detección de FRBs.

Este módulo contiene la lógica principal del pipeline:

- pipeline: Orquestación principal del flujo de procesamiento
- detection_engine: Motor de detección, clasificación y coordinación de visualizaciones
- pipeline_parameters: Cálculos centralizados de parámetros del pipeline
- data_flow_manager: Gestión de chunks, slices y flujo de datos
"""

from .detection_engine import process_slice_with_multiple_bands
from .pipeline_parameters import (
    calculate_frequency_downsampled,
    calculate_dm_height,
    calculate_width_total,
    calculate_slice_parameters,
    calculate_time_slice,
    calculate_slice_duration,
    get_pipeline_parameters,
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
    # Funciones principales
    'process_slice_with_multiple_bands',
    
    # Parámetros del pipeline
    'calculate_frequency_downsampled',
    'calculate_dm_height', 
    'calculate_width_total',
    'calculate_slice_parameters',
    'calculate_time_slice',
    'calculate_slice_duration',
    'get_pipeline_parameters',
    'calculate_overlap_decimated',
    'calculate_absolute_slice_time',
    
    # Gestión de flujo de datos
    'downsample_chunk',
    'build_dm_time_cube',
    'trim_valid_window',
    'plan_slices',
    'validate_slice_indices',
    'create_chunk_directories',
    'get_chunk_processing_parameters',
]
