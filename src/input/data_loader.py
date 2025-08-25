
from __future__ import annotations

# Standard library imports
import csv
import gc
import logging
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Tuple, Type

# Third-party imports
import numpy as np
from astropy.io import fits

# Optional third-party imports
try:
    import fitsio
except ImportError:
    fitsio = None

# Local imports
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

# Importar funciones de los módulos especializados
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

# Setup logger
logger = logging.getLogger(__name__)

def _safe_float(value, default=0.0):
    """Return ``value`` as ``float`` or ``default`` if conversion fails."""
    return safe_float(value, default)


def _safe_int(value, default=0):
    """Return ``value`` as ``int`` or ``default`` if conversion fails."""
    return safe_int(value, default)


def _auto_config_downsampling() -> None:
    """Configura DOWN_FREQ_RATE y DOWN_TIME_RATE si no fueron fijados por el usuario."""
    auto_config_downsampling()


def _print_debug_frequencies(prefix: str, file_name: str, freq_axis_inverted: bool) -> None:
    """Imprime bloque estándar de depuración de frecuencias con un prefijo dado."""
    print_debug_frequencies(prefix, file_name, freq_axis_inverted)


def _save_file_debug_info(file_name: str, debug_info: dict) -> None:
    """Guarda debug info (FITS o FIL) en summary.json inmediatamente (unificado)."""
    save_file_debug_info(file_name, debug_info)