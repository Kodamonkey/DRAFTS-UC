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
from ..output.summary_manager import _update_summary_with_file_debug
from ..preprocessing.data_downsampler import downsample_data

                                                  
from .utils import (
    safe_float,
    safe_int,
    auto_config_downsampling,
    print_debug_frequencies,
    save_file_debug_info
)
from .fits_handler import (
    load_fits_file,
    get_obparams,
    stream_fits
)
from .filterbank_handler import (
    _read_int,
    _read_double,
    _read_string,
    _read_header,
    _read_non_standard_header,
    load_fil_file,
    get_obparams_fil,
    stream_fil
)

                                            
from .file_detector import detect_file_type, validate_file_compatibility
from .streaming_orchestrator import get_streaming_function

              
logger = logging.getLogger(__name__)


def _safe_float(value, default=0.0):
    """Return ``value`` as ``float`` or ``default`` if conversion fails."""
    return safe_float(value, default)


def _safe_int(value, default=0):
    """Return ``value`` as ``int`` or ``default`` if conversion fails."""
    return safe_int(value, default)


def _auto_config_downsampling() -> None:
    """Configure ``DOWN_FREQ_RATE`` and ``DOWN_TIME_RATE`` when not provided."""
    auto_config_downsampling()


def _print_debug_frequencies(prefix: str, file_name: str, freq_axis_inverted: bool) -> None:
    """Print the standard frequency debug block with a prefixed label."""
    print_debug_frequencies(prefix, file_name, freq_axis_inverted)


def _save_file_debug_info(file_name: str, debug_info: dict) -> None:
    """Persist file debug information in ``summary.json`` immediately."""
    save_file_debug_info(file_name, debug_info)

                                                                               
                                 
                                                                               

