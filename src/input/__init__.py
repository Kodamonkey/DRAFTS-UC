"""Data input module for astronomical files.

This module provides functions to stream and process FITS, PSRFITS, and
filterbank files.  All data access uses streaming (chunked) I/O to avoid
loading entire files into RAM.
"""

from .data_loader import (
    get_obparams,
    stream_fits,
    get_obparams_fil,
    stream_fil,
)

from .utils import (
    safe_float,
    safe_int,
    auto_config_downsampling,
    print_debug_frequencies,
    save_file_debug_info
)

from .fits_handler import (
    get_obparams,
    stream_fits
)

from .filterbank_handler import (
    get_obparams_fil,
    stream_fil
)

from .file_detector import (
    detect_file_type,
    validate_file_compatibility,
)

from .parameter_extractor import (
    extract_parameters_auto,
)

from .streaming_orchestrator import (
    get_streaming_function,
    get_streaming_info,
    validate_streaming_parameters,
)

from .file_finder import (
    find_data_files,
)

__all__ = [
    'get_obparams',
    'stream_fits',
    'get_obparams_fil',
    'stream_fil',
    'detect_file_type',
    'validate_file_compatibility',
    'extract_parameters_auto',
    'get_streaming_function',
    'get_streaming_info',
    'validate_streaming_parameters',
    'find_data_files',
    'safe_float',
    'safe_int',
    'auto_config_downsampling',
    'print_debug_frequencies',
    'save_file_debug_info',
]
