"""
Cargador de Archivos FITS - Para archivos de datos astronómicos estándar
=======================================================================

Este módulo proporciona funciones para cargar archivos FITS (.fits) y extraer
sus metadatos para el procesamiento de detección de FRB.

Funciones principales:
- load_fits_data: Cargar datos desde archivo FITS
- get_fits_metadata: Extraer metadatos del archivo FITS
- validate_fits_file: Validar archivo FITS

Para astrónomos:
- Usar load_fits_data() para cargar sus archivos FITS
- Usar get_fits_metadata() para obtener información del archivo
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Importar funciones originales para mantener compatibilidad
from ..input.data_loader import load_fits_file, get_obparams

logger = logging.getLogger(__name__)


def load_fits_data(file_path: str | Path) -> np.ndarray:
    """
    Cargar datos desde un archivo FITS.
    
    Args:
        file_path: Ruta al archivo FITS (.fits)
        
    Returns:
        np.ndarray: Datos cargados con forma (tiempo, frecuencia)
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el archivo no es un FITS válido
        
    Ejemplo:
        >>> data = load_fits_data("observacion.fits")
        >>> print(f"Datos cargados: {data.shape}")
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo FITS no encontrado: {file_path}")
    
    if not file_path.suffix.lower() in ['.fits', '.fit']:
        raise ValueError(f"Archivo debe ser FITS (.fits/.fit): {file_path}")
    
    logger.info(f"Cargando archivo FITS: {file_path}")
    
    try:
        # Usar función original para mantener compatibilidad
        data = load_fits_file(str(file_path))
        
        if data is None or data.size == 0:
            raise ValueError(f"Datos vacíos en archivo FITS: {file_path}")
        
        logger.info(f"Datos FITS cargados exitosamente: {data.shape}")
        return data
        
    except Exception as e:
        logger.error(f"Error cargando archivo FITS {file_path}: {e}")
        raise


def get_fits_metadata(file_path: str | Path) -> Dict[str, Any]:
    """
    Extraer metadatos de un archivo FITS.
    
    Args:
        file_path: Ruta al archivo FITS
        
    Returns:
        Dict con metadatos del archivo:
        - frequency_resolution: Resolución de frecuencia
        - time_resolution: Resolución temporal (segundos)
        - file_length: Número de muestras temporales
        - frequencies: Array de frecuencias (MHz)
        - file_size: Tamaño del archivo en bytes
        
    Ejemplo:
        >>> metadata = get_fits_metadata("observacion.fits")
        >>> print(f"Resolución temporal: {metadata['time_resolution']}s")
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo FITS no encontrado: {file_path}")
    
    logger.info(f"Extrayendo metadatos de FITS: {file_path}")
    
    try:
        # Usar función original para mantener compatibilidad
        metadata = get_obparams(str(file_path))
        
        # Agregar información adicional
        metadata['file_size'] = file_path.stat().st_size
        metadata['file_path'] = str(file_path)
        metadata['file_type'] = 'FITS'
        
        logger.info(f"Metadatos FITS extraídos: {list(metadata.keys())}")
        return metadata
        
    except Exception as e:
        logger.error(f"Error extrayendo metadatos FITS {file_path}: {e}")
        raise


def validate_fits_file(file_path: str | Path) -> bool:
    """
    Validar que un archivo FITS es válido y puede ser procesado.
    
    Args:
        file_path: Ruta al archivo FITS
        
    Returns:
        bool: True si el archivo es válido
        
    Ejemplo:
        >>> if validate_fits_file("observacion.fits"):
        >>>     data = load_fits_data("observacion.fits")
    """
    file_path = Path(file_path)
    
    # Verificar que existe
    if not file_path.exists():
        logger.error(f"Archivo FITS no existe: {file_path}")
        return False
    
    # Verificar extensión
    if not file_path.suffix.lower() in ['.fits', '.fit']:
        logger.error(f"Archivo no es FITS: {file_path}")
        return False
    
    # Verificar que no esté vacío
    if file_path.stat().st_size == 0:
        logger.error(f"Archivo FITS está vacío: {file_path}")
        return False
    
    try:
        # Intentar cargar metadatos para verificar que es FITS válido
        metadata = get_fits_metadata(file_path)
        
        # Verificar metadatos críticos
        required_keys = ['frequency_resolution', 'time_resolution', 'file_length']
        for key in required_keys:
            if key not in metadata or metadata[key] <= 0:
                logger.error(f"Metadato FITS inválido {key}: {metadata.get(key)}")
                return False
        
        logger.info(f"Archivo FITS válido: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Archivo FITS inválido {file_path}: {e}")
        return False


# Mantener compatibilidad con código existente
def load_fits_file_legacy(file_name: str) -> np.ndarray:
    """
    Función legacy para compatibilidad con código existente.
    """
    return load_fits_data(file_name)


# Exportar funciones para compatibilidad
__all__ = [
    'load_fits_data',
    'get_fits_metadata', 
    'validate_fits_file',
    'load_fits_file_legacy',
] 