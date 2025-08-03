"""
Módulo de Carga de Datos - PASO 1 del Pipeline DRAFTS
=====================================================

Este módulo se encarga de cargar y validar archivos de datos astronómicos
(FITS y FIL) para el procesamiento de detección de FRB.

Responsabilidades separadas:
- fits_loader: Cargar archivos FITS
- fil_loader: Cargar archivos FIL  
- data_validator: Validar datos cargados
- metadata_extractor: Extraer metadatos de archivos
- stream_processor: Procesamiento en chunks/streaming
- header_parser: Parsing de headers específicos
- data_preprocessor: Preprocesamiento básico
- debug_logger: Logging y debugging
- format_detector: Detección automática de formatos

Para astrónomos:
- Usar load_fits_data() o load_fil_data() para cargar archivos
- Usar validate_data() para verificar que los datos son válidos
- Usar get_metadata() para obtener información del archivo
"""

# Imports principales para uso directo
from .fits_loader import load_fits_data, get_fits_metadata, validate_fits_file
from .fil_loader import load_fil_data, get_fil_metadata, validate_fil_file, stream_fil_data
from .data_validator import validate_data, validate_metadata, validate_file_for_processing
from .metadata_extractor import get_obparams, get_obparams_fil
from .stream_processor import stream_fil
from .data_preprocessor import load_and_preprocess_data
from .format_detector import detect_file_format, validate_format_compatibility, get_format_handler
from .header_parser import _read_header, _read_non_standard_header

# Funciones legacy para compatibilidad total
from .fits_loader import load_fits_file_legacy
from .fil_loader import load_fil_file_legacy, get_obparams_fil_legacy, stream_fil_legacy

__all__ = [
    # Funciones principales
    'load_fits_data',
    'get_fits_metadata', 
    'validate_fits_file',
    'load_fil_data',
    'get_fil_metadata',
    'validate_fil_file',
    'stream_fil_data',
    'validate_data',
    'validate_metadata',
    'validate_file_for_processing',
    'get_obparams',
    'get_obparams_fil',
    'stream_fil',
    'load_and_preprocess_data',
    'detect_file_format',
    'validate_format_compatibility',
    'get_format_handler',
    
    # Funciones legacy para compatibilidad
    'load_fits_file_legacy',
    'load_fil_file_legacy',
    'get_obparams_fil_legacy',
    'stream_fil_legacy',
] 