"""
Módulo de Preprocesamiento - PASO 2 del Pipeline DRAFTS
=======================================================

Este módulo se encarga de preparar y transformar los datos astronómicos
para el procesamiento de detección de FRB.

Responsabilidades separadas:
- downsampling: Reducir resolución temporal y frecuencial de datos
- dedispersion: Aplicar dedispersión GPU/CPU a los datos
- slice_calculator: Calcular tamaños óptimos de slices y chunks
- dm_calculator: Calcular rangos DM para procesamiento

Para astrónomos:
- Usar downsample_data() para reducir resolución de datos
- Usar d_dm_time_g() para aplicar dedispersión
- Usar calculate_slice_len_from_duration() para calcular slices
- Usar get_processing_parameters() para obtener parámetros óptimos
"""

# Imports principales para uso directo
from .downsampling import (
    downsample_data, 
    downsample_data_legacy,
    validate_downsampling_parameters,
    get_downsampling_info
)
from .dedispersion import (
    d_dm_time_g, 
    dedisperse_patch, 
    dedisperse_block,
    validate_dedispersion_parameters,
    d_dm_time_g_legacy,
    dedisperse_patch_legacy,
    dedisperse_block_legacy
)
from .slice_calculator import (
    calculate_slice_len_from_duration,
    calculate_optimal_chunk_size,
    get_processing_parameters,
    update_slice_len_dynamic,
    validate_processing_parameters,
    get_memory_usage_info,
    calculate_slice_len_from_duration_legacy,
    calculate_optimal_chunk_size_legacy,
    get_processing_parameters_legacy
)
from .dm_calculator import (
    calculate_dm_range,
    validate_dm_parameters,
    get_dm_processing_config,
    optimize_dm_range_for_frb_detection
)

__all__ = [
    # Funciones principales
    'downsample_data',
    'd_dm_time_g',
    'dedisperse_patch',
    'dedisperse_block',
    'calculate_slice_len_from_duration',
    'calculate_optimal_chunk_size',
    'get_processing_parameters',
    'update_slice_len_dynamic',
    'validate_processing_parameters',
    'calculate_dm_range',
    'validate_dm_parameters',
    'get_dm_processing_config',
    'optimize_dm_range_for_frb_detection',
    'validate_downsampling_parameters',
    'get_downsampling_info',
    'validate_dedispersion_parameters',
    'get_memory_usage_info',
    
    # Funciones legacy para compatibilidad
    'downsample_data_legacy',
    'd_dm_time_g_legacy',
    'dedisperse_patch_legacy',
    'dedisperse_block_legacy',
    'calculate_slice_len_from_duration_legacy',
    'calculate_optimal_chunk_size_legacy',
    'get_processing_parameters_legacy',
] 
