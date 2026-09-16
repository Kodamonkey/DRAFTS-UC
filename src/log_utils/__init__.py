# This module collects logging utilities for the pipeline.

"""
DRAFTS Logging Package
======================

This package contains all modules related to the DRAFTS pipeline logging
system, including configuration, formatters, and GPU handling.
"""

from .logging_config import (
    DRAFTSLogger,
    DRAFTSFormatter,
    Colors,
    setup_logging,
    get_global_logger,
    set_global_logger
)


from .chunking_logging import (
    display_detailed_chunking_info,
    log_chunk_budget,
    log_slice_plan_summary
)

from .data_loader_logging import (
    log_stream_fil_parameters,
    log_stream_fil_block_generation,
    log_stream_fil_summary,
    log_stream_fits_parameters,
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
    'get_global_logger',
    'set_global_logger',
    'display_detailed_chunking_info',
    'log_chunk_budget',
    'log_slice_plan_summary',
    'log_stream_fil_parameters',
    'log_stream_fil_block_generation',
    'log_stream_fil_summary',
    'log_stream_fits_parameters',
    'log_stream_fits_block_generation',
    'log_stream_fits_summary',
    'log_streaming_parameters',
    'log_block_processing',
    'log_processing_summary',
    'log_pipeline_file_processing',
    'log_pipeline_file_completion'
]
