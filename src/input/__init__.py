"""Módulo de entrada de datos para archivos astronómicos.

Este módulo proporciona funcionalidades para cargar y procesar archivos FITS, PSRFITS y filterbank.
"""

# Importar funciones principales para facilitar el acceso
from .data_loader import (
    # Funciones FITS
    load_fits_file,
    get_obparams,
    stream_fits,
    
    # Funciones Filterbank
    load_fil_file,
    get_obparams_fil,
    stream_fil,
    
    # Funciones de utilidad (compatibilidad)
    _safe_float,
    _safe_int,
    _auto_config_downsampling,
    _print_debug_frequencies,
    _save_file_debug_info
)

# Importar funciones de utilidad directamente
from .utils import (
    safe_float,
    safe_int,
    auto_config_downsampling,
    print_debug_frequencies,
    save_file_debug_info
)

# Importar funciones específicas de cada handler
from .fits_handler import (
    load_fits_file as load_fits,
    get_obparams,
    stream_fits
)

from .filterbank_handler import (
    load_fil_file as load_filterbank,
    get_obparams_fil,
    stream_fil
)

# Importar las nuevas funciones inteligentes del sistema refactorizado
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
    # Funciones principales del data_loader (compatibilidad)
    'load_fits_file',
    'get_obparams',
    'stream_fits',
    'load_fil_file',
    'get_obparams_fil',
    'stream_fil',
    
    # Nuevas funciones inteligentes del sistema refactorizado
    'detect_file_type',
    'validate_file_compatibility',
    'extract_parameters_auto',
    'get_streaming_function',
    'get_streaming_info',
    'validate_streaming_parameters',
    'find_data_files',
    
    # Funciones de utilidad
    'safe_float',
    'safe_int',
    'auto_config_downsampling',
    'print_debug_frequencies',
    'save_file_debug_info',
    
    # Funciones de compatibilidad
    '_safe_float',
    '_safe_int',
    '_auto_config_downsampling',
    '_print_debug_frequencies',
    '_save_file_debug_info'
] 
