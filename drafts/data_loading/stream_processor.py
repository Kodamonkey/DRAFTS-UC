"""
Procesador de Streaming - Procesamiento en chunks para archivos grandes
=====================================================================

Este módulo proporciona funciones para procesar archivos grandes en chunks,
permitiendo el procesamiento eficiente de archivos que no caben en memoria.

Funciones principales:
- stream_fil: Cargar archivo FIL en chunks para archivos grandes

Para astrónomos:
- Usar stream_fil() para procesar archivos FIL muy grandes
- Permite procesar archivos que no caben en memoria RAM
"""

import logging
import gc
import os
import struct
import numpy as np
from pathlib import Path
from typing import Dict, Generator, Tuple, Type, Any

# Importar funciones originales para mantener compatibilidad
from ..input.data_loader import stream_fil as _stream_fil_original
from .. import config

logger = logging.getLogger(__name__)


def stream_fil(file_name: str, chunk_samples: int = 2_097_152) -> Generator[Tuple[np.ndarray, Dict], None, None]:
    """
    Generador que lee un archivo .fil en bloques sin cargar todo en RAM.
    
    Args:
        file_name: Ruta al archivo .fil
        chunk_samples: Número de muestras por bloque (default: 2M)
    
    Yields:
        Tuple[data_block, metadata]: Bloque de datos (time, pol, chan) y metadatos
        
    Ejemplo:
        >>> for chunk_data, chunk_metadata in stream_fil("archivo_grande.fil"):
        >>>     print(f"Procesando chunk: {chunk_data.shape}")
    """
    logger.info(f"Iniciando streaming de archivo FIL: {file_name} (chunk_size={chunk_samples})")
    
    try:
        # Usar función original para mantener compatibilidad
        for chunk_data, chunk_metadata in _stream_fil_original(file_name, chunk_samples):
            # Agregar información adicional al metadata
            chunk_metadata['file_path'] = file_name
            chunk_metadata['file_type'] = 'FIL'
            chunk_metadata['chunk_samples'] = chunk_samples
            
            yield chunk_data, chunk_metadata
            
    except Exception as e:
        logger.error(f"Error en streaming de archivo FIL {file_name}: {e}")
        raise


def estimate_optimal_chunk_size(file_size_bytes: int, available_memory_gb: float = 4.0) -> int:
    """
    Estimar el tamaño de chunk óptimo basado en el tamaño del archivo y memoria disponible.
    
    Args:
        file_size_bytes: Tamaño del archivo en bytes
        available_memory_gb: Memoria disponible en GB
        
    Returns:
        int: Tamaño de chunk recomendado en muestras
        
    Ejemplo:
        >>> chunk_size = estimate_optimal_chunk_size(1024**3, 8.0)  # 1GB archivo, 8GB RAM
        >>> print(f"Chunk size recomendado: {chunk_size}")
    """
    # Convertir memoria disponible a bytes
    available_memory_bytes = available_memory_gb * (1024**3)
    
    # Usar máximo 25% de la memoria disponible para un chunk
    max_chunk_memory = available_memory_bytes * 0.25
    
    # Estimar bytes por muestra (asumiendo float32, 1 polarización, 512 canales)
    bytes_per_sample = 4 * 1 * 512  # float32 * nifs * nchans
    
    # Calcular muestras máximas por chunk
    max_samples_per_chunk = int(max_chunk_memory / bytes_per_sample)
    
    # Asegurar límites razonables
    min_chunk_size = 1024
    max_chunk_size = 10_000_000
    
    optimal_chunk_size = max(min_chunk_size, min(max_samples_per_chunk, max_chunk_size))
    
    logger.info(f"Chunk size estimado: {optimal_chunk_size} muestras "
               f"(archivo: {file_size_bytes/(1024**3):.1f}GB, RAM: {available_memory_gb}GB)")
    
    return optimal_chunk_size


def get_streaming_progress(file_path: str, chunk_samples: int) -> Dict[str, Any]:
    """
    Obtener información de progreso para streaming de archivo.
    
    Args:
        file_path: Ruta al archivo
        chunk_samples: Tamaño de chunk en muestras
        
    Returns:
        Dict con información de progreso
        
    Ejemplo:
        >>> progress = get_streaming_progress("archivo.fil", 1000000)
        >>> print(f"Total de chunks: {progress['total_chunks']}")
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    file_size = file_path.stat().st_size
    
    # Estimar parámetros del archivo
    # Asumiendo header de 512 bytes y datos float32
    header_size = 512
    data_size = file_size - header_size
    bytes_per_sample = 4 * 1 * 512  # float32 * nifs * nchans estimado
    
    total_samples = data_size // bytes_per_sample
    total_chunks = (total_samples + chunk_samples - 1) // chunk_samples
    
    progress_info = {
        'file_path': str(file_path),
        'file_size_bytes': file_size,
        'file_size_gb': file_size / (1024**3),
        'estimated_total_samples': total_samples,
        'chunk_samples': chunk_samples,
        'total_chunks': total_chunks,
        'estimated_duration_hours': (total_samples * 0.001) / 3600,  # Asumiendo 1ms por muestra
    }
    
    logger.info(f"Información de streaming: {total_chunks} chunks estimados "
               f"para {file_size/(1024**3):.1f}GB")
    
    return progress_info


def validate_chunk_data(chunk_data: np.ndarray, chunk_metadata: Dict) -> bool:
    """
    Validar datos de un chunk durante el streaming.
    
    Args:
        chunk_data: Datos del chunk
        chunk_metadata: Metadatos del chunk
        
    Returns:
        bool: True si el chunk es válido
        
    Ejemplo:
        >>> if validate_chunk_data(chunk_data, chunk_metadata):
        >>>     process_chunk(chunk_data)
    """
    if chunk_data is None:
        logger.error("Chunk data es None")
        return False
    
    if not isinstance(chunk_data, np.ndarray):
        logger.error(f"Chunk data no es numpy array: {type(chunk_data)}")
        return False
    
    if chunk_data.size == 0:
        logger.error("Chunk data está vacío")
        return False
    
    # Verificar dimensiones esperadas
    expected_shape = chunk_metadata.get('shape')
    if expected_shape and chunk_data.shape != expected_shape:
        logger.error(f"Forma de chunk incorrecta: {chunk_data.shape}, esperado: {expected_shape}")
        return False
    
    # Verificar que no hay valores NaN o Inf
    if np.any(np.isnan(chunk_data)):
        logger.error("Chunk contiene valores NaN")
        return False
    
    if np.any(np.isinf(chunk_data)):
        logger.error("Chunk contiene valores Inf")
        return False
    
    logger.debug(f"Chunk válido: {chunk_data.shape}, rango: [{chunk_data.min():.3f}, {chunk_data.max():.3f}]")
    return True


def process_chunks_with_progress(file_path: str, chunk_samples: int = 2_097_152, 
                                process_func=None) -> Generator[Tuple[np.ndarray, Dict, Dict], None, None]:
    """
    Procesar chunks con información de progreso.
    
    Args:
        file_path: Ruta al archivo
        chunk_samples: Tamaño de chunk
        process_func: Función opcional para procesar cada chunk
        
    Yields:
        Tuple[chunk_data, chunk_metadata, progress_info]
        
    Ejemplo:
        >>> for chunk_data, chunk_metadata, progress in process_chunks_with_progress("archivo.fil"):
        >>>     print(f"Progreso: {progress['current_chunk']}/{progress['total_chunks']}")
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    # Obtener información de progreso
    progress_info = get_streaming_progress(str(file_path), chunk_samples)
    total_chunks = progress_info['total_chunks']
    
    logger.info(f"Iniciando procesamiento de {total_chunks} chunks")
    
    current_chunk = 0
    
    try:
        for chunk_data, chunk_metadata in stream_fil(str(file_path), chunk_samples):
            current_chunk += 1
            
            # Validar chunk
            if not validate_chunk_data(chunk_data, chunk_metadata):
                logger.warning(f"Chunk {current_chunk} inválido, saltando...")
                continue
            
            # Procesar chunk si se proporciona función
            if process_func is not None:
                try:
                    chunk_data = process_func(chunk_data, chunk_metadata)
                except Exception as e:
                    logger.error(f"Error procesando chunk {current_chunk}: {e}")
                    continue
            
            # Actualizar información de progreso
            progress_info.update({
                'current_chunk': current_chunk,
                'progress_percent': (current_chunk / total_chunks) * 100,
                'chunks_remaining': total_chunks - current_chunk
            })
            
            yield chunk_data, chunk_metadata, progress_info
            
    except Exception as e:
        logger.error(f"Error en procesamiento de chunks: {e}")
        raise


# Exportar funciones
__all__ = [
    'stream_fil',
    'estimate_optimal_chunk_size',
    'get_streaming_progress',
    'validate_chunk_data',
    'process_chunks_with_progress',
] 