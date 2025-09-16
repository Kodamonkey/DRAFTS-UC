# This module collects logging utilities for the pipeline.

"""
Paquete de Logging para DRAFTS
=============================

Este paquete contiene todos los módulos relacionados con el sistema de logging
del pipeline DRAFTS, incluyendo configuración, formateadores y manejo de GPU.
"""

from .logging_config import (
    DRAFTSLogger,
    DRAFTSFormatter,
    Colors,
    setup_logging,
    get_logger,
    get_global_logger,
    set_global_logger
)

from .gpu_logging import (
    set_gpu_verbose,
    gpu_context,
    log_gpu_operation,
    log_gpu_memory_operation,
    filter_cuda_messages
)

from .chunking_logging import (
    display_detailed_chunking_info,
    log_chunk_processing_start,
    log_chunk_processing_end,
    log_file_processing_summary,
    log_memory_optimization,
    log_slice_configuration,
    log_chunk_budget,
    log_slice_plan_summary
)

from .data_loader_logging import (
    log_stream_fil_parameters,
    log_stream_fil_block_generation,
    log_stream_fil_summary,
    log_stream_fits_parameters,
    log_stream_fits_load_strategy,
    log_stream_fits_block_generation,
    log_stream_fits_summary
)

from .pipeline_logging import (
    log_streaming_parameters,
    log_block_processing,
    log_processing_summary,
    log_pipeline_file_processing,
    log_pipeline_file_completion
)

__all__ = [
    'DRAFTSLogger',
    'DRAFTSFormatter', 
    'Colors',
    'setup_logging',
    'get_logger',
    'get_global_logger',
    'set_global_logger',
    'set_gpu_verbose',
    'gpu_context',
    'log_gpu_operation',
    'log_gpu_memory_operation',
    'filter_cuda_messages',
    'display_detailed_chunking_info',
    'log_chunk_processing_start',
    'log_chunk_processing_end',
    'log_file_processing_summary',
    'log_memory_optimization',
    'log_slice_configuration',
    'log_chunk_budget',
    'log_slice_plan_summary',
    'log_stream_fil_parameters',
    'log_stream_fil_block_generation',
    'log_stream_fil_summary',
    'log_stream_fits_parameters',
    'log_stream_fits_load_strategy',
    'log_stream_fits_block_generation',
    'log_stream_fits_summary',
    'log_streaming_parameters',
    'log_block_processing',
    'log_processing_summary',
    'log_pipeline_file_processing',
    'log_pipeline_file_completion'
]
