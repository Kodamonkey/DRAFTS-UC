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
    get_file_info,
    log_file_detection
)

from .parameter_extractor import (
    extract_parameters_auto,
    get_parameters_function,
    extract_parameters_for_target,
    validate_extracted_parameters
)

from .streaming_orchestrator import (
    get_streaming_function,
    stream_file_auto,
    get_streaming_info,
    validate_streaming_parameters,
    log_streaming_start
)

from .file_finder import (
    find_data_files,
    find_files_by_pattern,
    get_file_summary,
    validate_file_list
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
    'get_file_info',
    'log_file_detection',
    'extract_parameters_auto',
    'get_parameters_function',
    'extract_parameters_for_target',
    'validate_extracted_parameters',
    'get_streaming_function',
    'stream_file_auto',
    'get_streaming_info',
    'validate_streaming_parameters',
    'log_streaming_start',
    'find_data_files',
    'find_files_by_pattern',
    'get_file_summary',
    'validate_file_list',
    
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
