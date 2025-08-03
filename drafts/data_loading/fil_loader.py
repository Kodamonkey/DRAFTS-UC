"""
Cargador de Archivos FIL - Para archivos filterbank
==================================================

Este módulo proporciona funciones para cargar archivos FIL (.fil) y extraer
sus metadatos para el procesamiento de detección de FRB.

Funciones principales:
- load_fil_data: Cargar datos desde archivo FIL
- get_fil_metadata: Extraer metadatos del archivo FIL
- stream_fil_data: Cargar archivo FIL en chunks para archivos grandes

Para astrónomos:
- Usar load_fil_data() para archivos FIL pequeños
- Usar stream_fil_data() para archivos FIL grandes
- Usar get_fil_metadata() para obtener información del archivo
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Generator, Tuple, Optional

# Importar funciones originales para mantener compatibilidad
from ..input.data_loader import load_fil_file, get_obparams_fil, stream_fil

logger = logging.getLogger(__name__)


def load_fil_data(file_path: str | Path) -> np.ndarray:
    """
    Cargar datos desde un archivo FIL (filterbank).
    
    Args:
        file_path: Ruta al archivo FIL (.fil)
        
    Returns:
        np.ndarray: Datos cargados con forma (tiempo, frecuencia)
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el archivo no es un FIL válido
        
    Ejemplo:
        >>> data = load_fil_data("observacion.fil")
        >>> print(f"Datos cargados: {data.shape}")
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo FIL no encontrado: {file_path}")
    
    if not file_path.suffix.lower() == '.fil':
        raise ValueError(f"Archivo debe ser FIL (.fil): {file_path}")
    
    logger.info(f"Cargando archivo FIL: {file_path}")
    
    try:
        # Usar función original para mantener compatibilidad
        data = load_fil_file(str(file_path))
        
        if data is None or data.size == 0:
            raise ValueError(f"Datos vacíos en archivo FIL: {file_path}")
        
        logger.info(f"Datos FIL cargados exitosamente: {data.shape}")
        return data
        
    except Exception as e:
        logger.error(f"Error cargando archivo FIL {file_path}: {e}")
        raise


def get_fil_metadata(file_path: str | Path) -> Dict[str, Any]:
    """
    Extraer metadatos de un archivo FIL.
    
    Args:
        file_path: Ruta al archivo FIL
        
    Returns:
        Dict con metadatos del archivo:
        - frequency_resolution: Resolución de frecuencia
        - time_resolution: Resolución temporal (segundos)
        - file_length: Número de muestras temporales
        - frequencies: Array de frecuencias (MHz)
        - file_size: Tamaño del archivo en bytes
        
    Ejemplo:
        >>> metadata = get_fil_metadata("observacion.fil")
        >>> print(f"Resolución temporal: {metadata['time_resolution']}s")
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo FIL no encontrado: {file_path}")
    
    logger.info(f"Extrayendo metadatos de FIL: {file_path}")
    
    try:
        # Usar función original para mantener compatibilidad
        metadata = get_obparams_fil(str(file_path))
        
        # Agregar información adicional
        metadata['file_size'] = file_path.stat().st_size
        metadata['file_path'] = str(file_path)
        metadata['file_type'] = 'FIL'
        
        logger.info(f"Metadatos FIL extraídos: {list(metadata.keys())}")
        return metadata
        
    except Exception as e:
        logger.error(f"Error extrayendo metadatos FIL {file_path}: {e}")
        raise


def stream_fil_data(file_path: str | Path, chunk_samples: int = 2_097_152) -> Generator[Tuple[np.ndarray, Dict[str, Any]], None, None]:
    """
    Cargar archivo FIL en chunks para archivos grandes.
    
    Args:
        file_path: Ruta al archivo FIL
        chunk_samples: Número de muestras por chunk (default: ~2M muestras)
        
    Yields:
        Tuple[np.ndarray, Dict]: (datos_chunk, metadatos_chunk)
        
    Ejemplo:
        >>> for chunk_data, chunk_metadata in stream_fil_data("archivo_grande.fil"):
        >>>     print(f"Procesando chunk: {chunk_data.shape}")
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo FIL no encontrado: {file_path}")
    
    if not file_path.suffix.lower() == '.fil':
        raise ValueError(f"Archivo debe ser FIL (.fil): {file_path}")
    
    logger.info(f"Streaming archivo FIL: {file_path} (chunk_size={chunk_samples})")
    
    try:
        # Usar función original para mantener compatibilidad
        for chunk_data, chunk_metadata in stream_fil(str(file_path), chunk_samples):
            # Agregar información adicional al metadata
            chunk_metadata['file_path'] = str(file_path)
            chunk_metadata['file_type'] = 'FIL'
            chunk_metadata['chunk_samples'] = chunk_samples
            
            yield chunk_data, chunk_metadata
            
    except Exception as e:
        logger.error(f"Error streaming archivo FIL {file_path}: {e}")
        raise


def validate_fil_file(file_path: str | Path) -> bool:
    """
    Validar que un archivo FIL es válido y puede ser procesado.
    
    Args:
        file_path: Ruta al archivo FIL
        
    Returns:
        bool: True si el archivo es válido
        
    Ejemplo:
        >>> if validate_fil_file("observacion.fil"):
        >>>     data = load_fil_data("observacion.fil")
    """
    file_path = Path(file_path)
    
    # Verificar que existe
    if not file_path.exists():
        logger.error(f"Archivo FIL no existe: {file_path}")
        return False
    
    # Verificar extensión
    if not file_path.suffix.lower() == '.fil':
        logger.error(f"Archivo no es FIL: {file_path}")
        return False
    
    # Verificar que no esté vacío
    if file_path.stat().st_size == 0:
        logger.error(f"Archivo FIL está vacío: {file_path}")
        return False
    
    try:
        # Intentar cargar metadatos para verificar que es FIL válido
        metadata = get_fil_metadata(file_path)
        
        # Verificar metadatos críticos
        required_keys = ['frequency_resolution', 'time_resolution', 'file_length']
        for key in required_keys:
            if key not in metadata or metadata[key] <= 0:
                logger.error(f"Metadato FIL inválido {key}: {metadata.get(key)}")
                return False
        
        logger.info(f"Archivo FIL válido: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Archivo FIL inválido {file_path}: {e}")
        return False


def estimate_fil_chunk_size(file_path: str | Path, target_chunk_duration_ms: float = 300.0) -> int:
    """
    Estimar el tamaño de chunk óptimo para un archivo FIL.
    
    Args:
        file_path: Ruta al archivo FIL
        target_chunk_duration_ms: Duración objetivo del chunk en milisegundos
        
    Returns:
        int: Número de muestras recomendado para el chunk
        
    Ejemplo:
        >>> chunk_size = estimate_fil_chunk_size("archivo.fil", 500.0)
        >>> print(f"Chunk size recomendado: {chunk_size} muestras")
    """
    file_path = Path(file_path)
    
    if not validate_fil_file(file_path):
        raise ValueError(f"Archivo FIL inválido: {file_path}")
    
    try:
        metadata = get_fil_metadata(file_path)
        time_resolution = metadata['time_resolution']
        
        # Calcular muestras necesarias para la duración objetivo
        target_duration_seconds = target_chunk_duration_ms / 1000.0
        chunk_samples = int(target_duration_seconds / time_resolution)
        
        # Asegurar que sea un valor razonable
        chunk_samples = max(1024, min(chunk_samples, 10_000_000))
        
        logger.info(f"Chunk size estimado para {file_path}: {chunk_samples} muestras "
                   f"({target_chunk_duration_ms}ms)")
        
        return chunk_samples
        
    except Exception as e:
        logger.error(f"Error estimando chunk size para {file_path}: {e}")
        # Valor por defecto seguro
        return 2_097_152


# Mantener compatibilidad con código existente
def load_fil_file_legacy(file_name: str) -> np.ndarray:
    """
    Función legacy para compatibilidad con código existente.
    """
    return load_fil_data(file_name)


def get_obparams_fil_legacy(file_name: str) -> Dict[str, Any]:
    """
    Función legacy para compatibilidad con código existente.
    """
    return get_fil_metadata(file_name)


def stream_fil_legacy(file_name: str, chunk_samples: int = 2_097_152):
    """
    Función legacy para compatibilidad con código existente.
    """
    return stream_fil_data(file_name, chunk_samples)


# Exportar funciones para compatibilidad
__all__ = [
    'load_fil_data',
    'get_fil_metadata',
    'stream_fil_data',
    'validate_fil_file',
    'estimate_fil_chunk_size',
    'load_fil_file_legacy',
    'get_obparams_fil_legacy',
    'stream_fil_legacy',
] 