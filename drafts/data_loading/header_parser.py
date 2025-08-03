"""
Parser de Headers - Parsing de headers específicos de archivos astronómicos
=========================================================================

Este módulo proporciona funciones para parsear headers de archivos FIL y FITS,
manejando diferentes formatos y estructuras de datos.

Funciones principales:
- _read_header: Leer headers FIL estándar
- _read_non_standard_header: Manejar formatos no estándar
- _read_int, _read_double, _read_string: Utilidades de parsing

Para astrónomos:
- Maneja automáticamente diferentes formatos de archivos
- Detecta y adapta a formatos no estándar
"""

import logging
import struct
from typing import Dict, Tuple, Any

logger = logging.getLogger(__name__)


def _read_int(f) -> int:
    """
    Leer un entero de 32 bits desde el archivo.
    
    Args:
        f: Archivo abierto en modo binario
        
    Returns:
        int: Valor entero leído
    """
    return struct.unpack("<i", f.read(4))[0]


def _read_double(f) -> float:
    """
    Leer un double de 64 bits desde el archivo.
    
    Args:
        f: Archivo abierto en modo binario
        
    Returns:
        float: Valor double leído
    """
    return struct.unpack("<d", f.read(8))[0]


def _read_string(f) -> str:
    """
    Leer una cadena de caracteres desde el archivo.
    
    Args:
        f: Archivo abierto en modo binario
        
    Returns:
        str: Cadena leída
    """
    length = _read_int(f)
    return f.read(length).decode('utf-8', errors='ignore')


def _read_header(f) -> Tuple[Dict[str, Any], int]:
    """
    Leer header de archivo filterbank, manejando formatos estándar y no estándar.
    
    Args:
        f: Archivo abierto en modo binario
        
    Returns:
        Tuple[Dict, int]: (header_dict, header_length)
        
    Ejemplo:
        >>> with open("archivo.fil", "rb") as f:
        >>>     header, header_len = _read_header(f)
        >>>     print(f"Header length: {header_len}")
    """
    original_pos = f.tell()
    
    try:
        # Intentar leer como formato SIGPROC estándar
        start = _read_string(f)
        if start != "HEADER_START":
            # Si no es formato estándar, resetear y usar enfoque alternativo
            f.seek(original_pos)
            return _read_non_standard_header(f)

        header = {}
        while True:
            try:
                key = _read_string(f)
                if key == "HEADER_END":
                    break
                if key in {"rawdatafile", "source_name"}:
                    header[key] = _read_string(f)
                elif key in {
                    "telescope_id",
                    "machine_id",
                    "data_type",
                    "barycentric",
                    "pulsarcentric",
                    "nbits",
                    "nchans",
                    "nifs",
                    "nbeams",
                    "ibeam",
                    "nsamples",
                }:
                    header[key] = _read_int(f)
                elif key in {
                    "az_start",
                    "za_start",
                    "src_raj",
                    "src_dej",
                    "tstart",
                    "tsamp",
                    "fch1",
                    "foff",
                    "refdm",
                }:
                    header[key] = _read_double(f)
                else:
                    # Leer campo desconocido como entero por defecto
                    header[key] = _read_int(f)
            except (struct.error, UnicodeDecodeError) as e:
                logger.warning(f"Error leyendo campo de header '{key}': {e}")
                continue
        return header, f.tell()
    except Exception as e:
        logger.error(f"Error leyendo header filterbank estándar: {e}")
        f.seek(original_pos)
        return _read_non_standard_header(f)


def _read_non_standard_header(f) -> Tuple[Dict[str, Any], int]:
    """
    Manejar archivos filterbank con formato no estándar usando parámetros estimados.
    
    Args:
        f: Archivo abierto en modo binario
        
    Returns:
        Tuple[Dict, int]: (header_dict, header_length)
        
    Ejemplo:
        >>> with open("archivo_no_estandar.fil", "rb") as f:
        >>>     header, header_len = _read_non_standard_header(f)
        >>>     print(f"Parámetros estimados: {header}")
    """
    logger.info("Detectado archivo .fil con formato no estándar, usando parámetros estimados")
    
    # Obtener tamaño del archivo para estimar parámetros
    current_pos = f.tell()
    f.seek(0, 2)  # Ir al final
    file_size = f.tell()
    f.seek(current_pos)  # Volver a posición original
    
    # Parámetros comunes para muchos archivos filterbank
    header = {
        "nchans": 512,
        "tsamp": 8.192e-5,
        "fch1": 1500.0,
        "foff": -1.0,
        "nbits": 8,
        "nifs": 1,
    }
    
    # Estimar número de muestras basado en tamaño del archivo
    bytes_per_sample = header["nifs"] * header["nchans"] * (header["nbits"] // 8)
    estimated_samples = (file_size - 512) // bytes_per_sample
    
    # Usar límite máximo de muestras si está configurado
    try:
        from .. import config
        max_samples = getattr(config, 'MAX_SAMPLES_LIMIT', 1000000)
        header["nsamples"] = min(estimated_samples, max_samples)
    except ImportError:
        header["nsamples"] = min(estimated_samples, 1000000)
    
    logger.info(f"Parámetros estimados para archivo no estándar:")
    logger.info(f"  - Tamaño de archivo: {file_size / (1024**2):.1f} MB")
    logger.info(f"  - Muestras estimadas: {estimated_samples}")
    logger.info(f"  - Muestras a usar: {header['nsamples']}")
    
    return header, 512


def parse_fits_header(hdu) -> Dict[str, Any]:
    """
    Parsear header de archivo FITS.
    
    Args:
        hdu: HDU (Header Data Unit) de FITS
        
    Returns:
        Dict: Header parseado
        
    Ejemplo:
        >>> with fits.open("archivo.fits") as hdul:
        >>>     header = parse_fits_header(hdul[0])
        >>>     print(f"NAXIS: {header.get('NAXIS')}")
    """
    header = {}
    
    try:
        for key, value in hdu.header.items():
            # Convertir valores a tipos apropiados
            if isinstance(value, (int, float, str)):
                header[key] = value
            else:
                # Convertir otros tipos a string
                header[key] = str(value)
                
    except Exception as e:
        logger.error(f"Error parseando header FITS: {e}")
    
    return header


def detect_header_format(file_path: str) -> str:
    """
    Detectar el formato del header de un archivo.
    
    Args:
        file_path: Ruta al archivo
        
    Returns:
        str: Formato detectado ('standard', 'non_standard', 'fits', 'unknown')
        
    Ejemplo:
        >>> format_type = detect_header_format("archivo.fil")
        >>> print(f"Formato detectado: {format_type}")
    """
    file_path = str(file_path)
    
    try:
        with open(file_path, "rb") as f:
            # Intentar leer como string para detectar formato
            try:
                start = _read_string(f)
                if start == "HEADER_START":
                    return "standard"
                else:
                    return "non_standard"
            except:
                # Si no es FIL, verificar si es FITS
                f.seek(0)
                magic = f.read(4)
                if magic.startswith(b'SIMPLE') or magic.startswith(b'XTENSION'):
                    return "fits"
                else:
                    return "unknown"
    except Exception as e:
        logger.error(f"Error detectando formato de header: {e}")
        return "unknown"


def validate_header_parameters(header: Dict[str, Any]) -> bool:
    """
    Validar parámetros del header.
    
    Args:
        header: Diccionario con parámetros del header
        
    Returns:
        bool: True si los parámetros son válidos
        
    Ejemplo:
        >>> if validate_header_parameters(header):
        >>>     print("Header válido")
    """
    required_keys = ["nchans", "tsamp", "fch1", "foff", "nbits", "nifs"]
    
    for key in required_keys:
        if key not in header:
            logger.error(f"Header falta clave requerida: {key}")
            return False
    
    # Validar valores
    if header.get("nchans", 0) <= 0:
        logger.error(f"nchans inválido: {header.get('nchans')}")
        return False
    
    if header.get("tsamp", 0) <= 0:
        logger.error(f"tsamp inválido: {header.get('tsamp')}")
        return False
    
    if header.get("nbits", 0) not in [8, 16, 32, 64]:
        logger.error(f"nbits inválido: {header.get('nbits')}")
        return False
    
    if header.get("nifs", 0) <= 0:
        logger.error(f"nifs inválido: {header.get('nifs')}")
        return False
    
    logger.info("Parámetros de header válidos")
    return True


# Exportar funciones
__all__ = [
    '_read_int',
    '_read_double', 
    '_read_string',
    '_read_header',
    '_read_non_standard_header',
    'parse_fits_header',
    'detect_header_format',
    'validate_header_parameters',
] 