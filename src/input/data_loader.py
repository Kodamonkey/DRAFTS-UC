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

                                                                               
                                                           
                                                                               

# This function returns a safe float value.
def _safe_float(value, default=0.0):
    """Return ``value`` as ``float`` or ``default`` if conversion fails."""
    return safe_float(value, default)


# This function returns a safe integer value.
def _safe_int(value, default=0):
    """Return ``value`` as ``int`` or ``default`` if conversion fails."""
    return safe_int(value, default)


# This function auto-configures downsampling parameters.
def _auto_config_downsampling() -> None:
    """Configura DOWN_FREQ_RATE y DOWN_TIME_RATE si no fueron fijados por el usuario."""
    auto_config_downsampling()


# This function prints debug frequencies.
def _print_debug_frequencies(prefix: str, file_name: str, freq_axis_inverted: bool) -> None:
    """Imprime bloque estándar de depuración de frecuencias con un prefijo dado."""
    print_debug_frequencies(prefix, file_name, freq_axis_inverted)


# This function saves file debug info.
def _save_file_debug_info(file_name: str, debug_info: dict) -> None:
    """Guarda debug info (FITS o FIL) en summary.json inmediatamente (unificado)."""
    save_file_debug_info(file_name, debug_info)

                                                                               
                                 
                                                                               
"""
ESTRUCTURA REFACTORIZADA:

1. MÓDULO UTILS (src/input/utils.py):
   - safe_float() → _safe_float() (compatibilidad)
   - safe_int() → _safe_int() (compatibilidad)
   - auto_config_downsampling() → _auto_config_downsampling() (compatibilidad)
   - print_debug_frequencies() → _print_debug_frequencies() (compatibilidad)
   - save_file_debug_info() → _save_file_debug_info() (compatibilidad)

2. MÓDULO FITS_HANDLER (src/input/fits_handler.py):
   - load_fits_file() → importado directamente
   - get_obparams() → importado directamente
   - stream_fits() → importado directamente

3. MÓDULO FILTERBANK_HANDLER (src/input/filterbank_handler.py):
   - _read_int() → importado directamente
   - _read_double() → importado directamente
   - _read_string() → importado directamente
   - _read_header() → importado directamente
   - _read_non_standard_header() → importado directamente
   - load_fil_file() → importado directamente
   - get_obparams_fil() → importado directamente
   - stream_fil() → importado directamente

4. NUEVOS MÓDULOS INTELIGENTES:
   - file_detector.py: Detección y validación de tipos de archivo
   - parameter_extractor.py: Extracción automática de parámetros
   - streaming_orchestrator.py: Orquestación inteligente de streaming

5. WRAPPER PRINCIPAL (src/input/data_loader.py):
   - Mantiene todas las funciones públicas para compatibilidad
   - Importa funciones de los módulos especializados
   - Proporciona funciones de compatibilidad con prefijo _
   - Expone nuevas funciones inteligentes

BENEFICIOS DE LA REFACTORIZACIÓN:
- Código más modular y mantenible
- Separación clara de responsabilidades
- Fácil testing y debugging por módulo
- Compatibilidad total con el pipeline existente
- Mejor organización del código
- Reutilización de funciones comunes
- Funciones inteligentes para detección automática

NOTA: Todas las funciones públicas del módulo original siguen disponibles
con la misma interfaz, por lo que el pipeline existente no se verá afectado.
"""
