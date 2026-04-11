# This module provides compatibility data loading helpers.

from __future__ import annotations

                          
import csv
import gc
import logging
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Tuple, Type

                     
import numpy as np
from astropy.io import fits

                              
try:
    import fitsio
except ImportError:
    fitsio = None

               
from ..config import config
from ..logging import (
    log_stream_fil_block_generation,
    log_stream_fil_parameters,
    log_stream_fil_summary,
    log_stream_fits_block_generation,
    log_stream_fits_load_strategy,
    log_stream_fits_parameters,
    log_stream_fits_summary
)
# JSON summary functionality disabled - import removed
from ..preprocessing.data_downsampler import downsample_data

                                                  
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
    _read_int,
    _read_double,
    _read_string,
    _read_header,
    _read_non_standard_header,
    get_obparams_fil,
    stream_fil
)

                                            
from .file_detector import detect_file_type, validate_file_compatibility
from .streaming_orchestrator import get_streaming_function

              
logger = logging.getLogger(__name__)



# Legacy aliases — use the canonical versions from .utils directly
_safe_float = safe_float
_safe_int = safe_int
_auto_config_downsampling = auto_config_downsampling
_print_debug_frequencies = print_debug_frequencies
_save_file_debug_info = save_file_debug_info

                                                                               
                                 
                                                                               

