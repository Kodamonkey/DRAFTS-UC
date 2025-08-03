"""
Extractor de Metadatos - Extraer información de archivos astronómicos
===================================================================

Este módulo proporciona funciones para extraer metadatos de archivos FITS y FIL,
incluyendo parámetros de observación, frecuencias, y configuración.

Funciones principales:
- get_obparams: Extraer parámetros de observación FITS
- get_obparams_fil: Extraer parámetros de observación FIL

Para astrónomos:
- Usar get_obparams() para obtener información de archivos FITS
- Usar get_obparams_fil() para obtener información de archivos FIL
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Importar funciones originales para mantener compatibilidad
from ..input.data_loader import get_obparams as _get_obparams_original
from ..input.data_loader import get_obparams_fil as _get_obparams_fil_original

logger = logging.getLogger(__name__)


def get_obparams(file_name: str) -> Dict[str, Any]:
    """
    Extraer parámetros de observación de un archivo FITS.
    
    Args:
        file_name: Ruta al archivo FITS
        
    Returns:
        Dict con parámetros de observación:
        - frequency_resolution: Resolución de frecuencia
        - time_resolution: Resolución temporal (segundos)
        - file_length: Número de muestras temporales
        - frequencies: Array de frecuencias (MHz)
        
    Ejemplo:
        >>> params = get_obparams("observacion.fits")
        >>> print(f"Resolución temporal: {params['time_resolution']}s")
    """
    logger.info(f"Extrayendo parámetros de observación FITS: {file_name}")
    
    try:
        # Usar función original para mantener compatibilidad
        metadata = _get_obparams_original(file_name)
        
        # Agregar información adicional
        file_path = Path(file_name)
        if file_path.exists():
            metadata['file_size'] = file_path.stat().st_size
            metadata['file_path'] = str(file_path)
            metadata['file_type'] = 'FITS'
        
        logger.info(f"Parámetros FITS extraídos: {list(metadata.keys())}")
        return metadata
        
    except Exception as e:
        logger.error(f"Error extrayendo parámetros FITS {file_name}: {e}")
        raise


def get_obparams_fil(file_name: str) -> Dict[str, Any]:
    """
    Extraer parámetros de observación de un archivo FIL.
    
    Args:
        file_name: Ruta al archivo FIL
        
    Returns:
        Dict con parámetros de observación:
        - frequency_resolution: Resolución de frecuencia
        - time_resolution: Resolución temporal (segundos)
        - file_length: Número de muestras temporales
        - frequencies: Array de frecuencias (MHz)
        
    Ejemplo:
        >>> params = get_obparams_fil("observacion.fil")
        >>> print(f"Resolución temporal: {params['time_resolution']}s")
    """
    logger.info(f"Extrayendo parámetros de observación FIL: {file_name}")
    
    try:
        # Usar función original para mantener compatibilidad
        metadata = _get_obparams_fil_original(file_name)
        
        # Agregar información adicional
        file_path = Path(file_name)
        if file_path.exists():
            metadata['file_size'] = file_path.stat().st_size
            metadata['file_path'] = str(file_path)
            metadata['file_type'] = 'FIL'
        
        logger.info(f"Parámetros FIL extraídos: {list(metadata.keys())}")
        return metadata
        
    except Exception as e:
        logger.error(f"Error extrayendo parámetros FIL {file_name}: {e}")
        raise


def extract_frequency_info(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extraer información específica de frecuencias de los metadatos.
    
    Args:
        metadata: Metadatos del archivo
        
    Returns:
        Dict con información de frecuencias:
        - freq_min: Frecuencia mínima (MHz)
        - freq_max: Frecuencia máxima (MHz)
        - freq_span: Rango de frecuencias (MHz)
        - freq_resolution: Resolución de frecuencia (MHz)
        - n_channels: Número de canales
        
    Ejemplo:
        >>> freq_info = extract_frequency_info(metadata)
        >>> print(f"Rango: {freq_info['freq_min']:.1f} - {freq_info['freq_max']:.1f} MHz")
    """
    freq_info = {}
    
    try:
        frequencies = metadata.get('frequencies')
        if frequencies is not None and len(frequencies) > 0:
            freq_info['freq_min'] = float(frequencies.min())
            freq_info['freq_max'] = float(frequencies.max())
            freq_info['freq_span'] = freq_info['freq_max'] - freq_info['freq_min']
            freq_info['n_channels'] = len(frequencies)
            
            # Calcular resolución de frecuencia
            if len(frequencies) > 1:
                freq_info['freq_resolution'] = abs(frequencies[1] - frequencies[0])
            else:
                freq_info['freq_resolution'] = 0.0
        else:
            logger.warning("No se encontraron frecuencias en los metadatos")
            
    except Exception as e:
        logger.error(f"Error extrayendo información de frecuencias: {e}")
    
    return freq_info


def extract_time_info(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extraer información específica temporal de los metadatos.
    
    Args:
        metadata: Metadatos del archivo
        
    Returns:
        Dict con información temporal:
        - time_resolution: Resolución temporal (segundos)
        - total_duration: Duración total (segundos)
        - n_samples: Número de muestras temporales
        - time_span: Rango temporal (segundos)
        
    Ejemplo:
        >>> time_info = extract_time_info(metadata)
        >>> print(f"Duración total: {time_info['total_duration']:.1f}s")
    """
    time_info = {}
    
    try:
        time_resolution = metadata.get('time_resolution', 0.0)
        file_length = metadata.get('file_length', 0)
        
        time_info['time_resolution'] = float(time_resolution)
        time_info['n_samples'] = int(file_length)
        time_info['total_duration'] = time_resolution * file_length
        time_info['time_span'] = time_info['total_duration']
        
    except Exception as e:
        logger.error(f"Error extrayendo información temporal: {e}")
    
    return time_info


def extract_observation_info(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extraer información de observación de los metadatos.
    
    Args:
        metadata: Metadatos del archivo
        
    Returns:
        Dict con información de observación:
        - telescope: Telescopio usado
        - source: Fuente observada
        - observer: Observador
        - date: Fecha de observación
        
    Ejemplo:
        >>> obs_info = extract_observation_info(metadata)
        >>> print(f"Telescopio: {obs_info.get('telescope', 'N/A')}")
    """
    obs_info = {}
    
    # Mapear claves comunes de observación
    key_mapping = {
        'telescope_id': 'telescope',
        'src_name': 'source',
        'source_name': 'source',
        'observer': 'observer',
        'date': 'date',
        'tstart': 'start_time',
        'obs_mode': 'observation_mode',
        'project_id': 'project'
    }
    
    for old_key, new_key in key_mapping.items():
        if old_key in metadata:
            obs_info[new_key] = metadata[old_key]
    
    return obs_info


def get_comprehensive_metadata(file_path: str | Path) -> Dict[str, Any]:
    """
    Obtener metadatos completos de un archivo (FITS o FIL).
    
    Args:
        file_path: Ruta al archivo
        
    Returns:
        Dict con metadatos completos organizados por categorías
        
    Ejemplo:
        >>> metadata = get_comprehensive_metadata("observacion.fits")
        >>> print(f"Información completa: {list(metadata.keys())}")
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    # Obtener metadatos básicos según el tipo de archivo
    if file_path.suffix.lower() in ['.fits', '.fit']:
        basic_metadata = get_obparams(str(file_path))
    elif file_path.suffix.lower() == '.fil':
        basic_metadata = get_obparams_fil(str(file_path))
    else:
        raise ValueError(f"Tipo de archivo no soportado: {file_path.suffix}")
    
    # Organizar metadatos por categorías
    comprehensive_metadata = {
        'file_info': {
            'file_path': str(file_path),
            'file_type': basic_metadata.get('file_type', 'UNKNOWN'),
            'file_size': basic_metadata.get('file_size', 0),
        },
        'frequency_info': extract_frequency_info(basic_metadata),
        'time_info': extract_time_info(basic_metadata),
        'observation_info': extract_observation_info(basic_metadata),
        'raw_metadata': basic_metadata  # Metadatos originales completos
    }
    
    logger.info(f"Metadatos completos extraídos para {file_path}")
    return comprehensive_metadata


# Exportar funciones
__all__ = [
    'get_obparams',
    'get_obparams_fil',
    'extract_frequency_info',
    'extract_time_info',
    'extract_observation_info',
    'get_comprehensive_metadata',
] 