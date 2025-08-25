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

# Importar funciones específicas de FITS
from .fits_handler import (
    load_fits_file as load_fits,
    get_obparams as get_fits_params,
    stream_fits as stream_fits_data
)

# Importar funciones específicas de Filterbank
from .filterbank_handler import (
    load_fil_file as load_filterbank,
    get_obparams_fil as get_filterbank_params,
    stream_fil as stream_filterbank_data
)

__all__ = [
    # Funciones principales (compatibilidad)
    'load_fits_file',
    'get_obparams', 
    'stream_fits',
    'load_fil_file',
    'get_obparams_fil',
    'stream_fil',
    
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
    '_save_file_debug_info',
    
    # Alias para mayor claridad
    'load_fits',
    'get_fits_params',
    'stream_fits_data',
    'load_filterbank',
    'get_filterbank_params', 
    'stream_filterbank_data'
] 
