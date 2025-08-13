"""
Downsampling Temporal y Frecuencial para DRAFTS Pipeline
=======================================================

Este módulo contiene funcionalidad para realizar downsampling de archivos astronómicos:
- Downsampling temporal (reducción de muestras temporales)
- Downsampling frecuencial (reducción de canales de frecuencia)
- Procesamiento por chunks para archivos grandes
- Optimización con numba para operaciones intensivas

Basado en los algoritmos de d-center-main.py y d-resnet-main.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from numba import njit, prange

logger = logging.getLogger(__name__)

# Funcion principal para downsampling
def downsample_astronomical_data(
    data: np.ndarray,
    time_downsample_factor: int = 1,
    freq_downsample_factor: int = 1,
    method: str = 'mean'
) -> np.ndarray:
    """
    Realiza downsampling temporal y frecuencial de datos astronómicos.
    
    Args:
        data: Array de datos (muestras, polarizaciones, canales)
        time_downsample_factor: Factor de downsampling temporal (1 = sin cambio)
        freq_downsample_factor: Factor de downsampling frecuencial (1 = sin cambio)
        method: Método de downsampling ('mean', 'median', 'max', 'min')
        
    Returns:
        Array con datos downsampled
    """
    if time_downsample_factor == 1 and freq_downsample_factor == 1:
        logger.info("Sin downsampling aplicado")
        return data
    
    logger.info(f"Aplicando downsampling: temporal={time_downsample_factor}x, frecuencial={freq_downsample_factor}x")
    
    # Validar parámetros
    if time_downsample_factor < 1:
        raise ValueError("time_downsample_factor debe ser >= 1")
    if freq_downsample_factor < 1:
        raise ValueError("freq_downsample_factor debe ser >= 1")
    
    # Normalizar la forma de los datos
    original_shape = data.shape
    logger.info(f"Forma original de datos: {original_shape}")
    
    # Manejar diferentes formatos de datos
    if len(original_shape) == 1:
        # Datos 1D: (muestras,) -> (muestras, 1, 1)
        data = data.reshape(-1, 1, 1)
        logger.info(f"Datos 1D normalizados a: {data.shape}")
    elif len(original_shape) == 2:
        # Datos 2D: (muestras, canales) -> (muestras, 1, canales)
        data = data.reshape(original_shape[0], 1, original_shape[1])
        logger.info(f"Datos 2D normalizados a: {data.shape}")
    elif len(original_shape) == 3:
        # Datos 3D: (muestras, polarizaciones, canales) - formato esperado
        pass
    else:
        raise ValueError(f"Forma de datos no soportada: {original_shape}. Se espera 1D, 2D o 3D")
    
    # Aplicar downsampling temporal
    if time_downsample_factor > 1:
        data = _downsample_temporal(data, time_downsample_factor, method)
    
    # Aplicar downsampling frecuencial
    if freq_downsample_factor > 1:
        # Verificar si hay suficientes canales para el downsampling
        if data.shape[2] >= freq_downsample_factor:
            data = _downsample_frequency(data, freq_downsample_factor, method)
        else:
            logger.warning(f"No se puede aplicar downsampling frecuencial con factor {freq_downsample_factor} en datos con {data.shape[2]} canales")
    
    logger.info(f"Downsampling completado. Nueva forma: {data.shape}")
    return data


def _downsample_temporal(data: np.ndarray, factor: int, method: str) -> np.ndarray:
    """
    Realiza downsampling temporal de los datos.
    
    Args:
        data: Array (muestras, polarizaciones, canales)
        factor: Factor de downsampling
        method: Método de agregación
        
    Returns:
        Array con downsampling temporal aplicado
    """
    nsamples, nifs, nchans = data.shape
    
    # Calcular nuevas dimensiones
    new_nsamples = nsamples // factor
    if new_nsamples == 0:
        raise ValueError(f"Factor de downsampling temporal {factor} es muy grande para {nsamples} muestras")
    
    # Reshape para facilitar el downsampling
    # (new_nsamples, factor, nifs, nchans)
    reshaped = data[:new_nsamples * factor].reshape(new_nsamples, factor, nifs, nchans)
    
    # Aplicar método de agregación
    if method == 'mean':
        result = np.mean(reshaped, axis=1)
    elif method == 'median':
        result = np.median(reshaped, axis=1)
    elif method == 'max':
        result = np.max(reshaped, axis=1)
    elif method == 'min':
        result = np.min(reshaped, axis=1)
    else:
        raise ValueError(f"Método no soportado: {method}")
    
    return result


def _downsample_frequency(data: np.ndarray, factor: int, method: str) -> np.ndarray:
    """
    Realiza downsampling frecuencial de los datos.
    
    Args:
        data: Array (muestras, polarizaciones, canales)
        factor: Factor de downsampling
        method: Método de agregación
        
    Returns:
        Array con downsampling frecuencial aplicado
    """
    nsamples, nifs, nchans = data.shape
    
    # Calcular nuevas dimensiones
    new_nchans = nchans // factor
    if new_nchans == 0:
        raise ValueError(f"Factor de downsampling frecuencial {factor} es muy grande para {nchans} canales")
    
    # Reshape para facilitar el downsampling
    # (nsamples, nifs, new_nchans, factor)
    reshaped = data[:, :, :new_nchans * factor].reshape(nsamples, nifs, new_nchans, factor)
    
    # Aplicar método de agregación
    if method == 'mean':
        result = np.mean(reshaped, axis=3)
    elif method == 'median':
        result = np.median(reshaped, axis=3)
    elif method == 'max':
        result = np.max(reshaped, axis=3)
    elif method == 'min':
        result = np.min(reshaped, axis=3)
    else:
        raise ValueError(f"Método no soportado: {method}")
    
    return result


@njit(parallel=True)
def downsample_temporal_numba(data: np.ndarray, factor: int) -> np.ndarray:
    """
    Versión optimizada con numba para downsampling temporal.
    
    Args:
        data: Array (muestras, polarizaciones, canales)
        factor: Factor de downsampling
        
    Returns:
        Array con downsampling temporal aplicado
    """
    nsamples, nifs, nchans = data.shape
    new_nsamples = nsamples // factor
    
    result = np.zeros((new_nsamples, nifs, nchans), dtype=data.dtype)
    
    for i in prange(new_nsamples):
        start_idx = i * factor
        end_idx = start_idx + factor
        result[i] = np.mean(data[start_idx:end_idx], axis=0)
    
    return result


@njit(parallel=True)
def downsample_frequency_numba(data: np.ndarray, factor: int) -> np.ndarray:
    """
    Versión optimizada con numba para downsampling frecuencial.
    
    Args:
        data: Array (muestras, polarizaciones, canales)
        factor: Factor de downsampling
        
    Returns:
        Array con downsampling frecuencial aplicado
    """
    nsamples, nifs, nchans = data.shape
    new_nchans = nchans // factor
    
    result = np.zeros((nsamples, nifs, new_nchans), dtype=data.dtype)
    
    for i in prange(new_nchans):
        start_idx = i * factor
        end_idx = start_idx + factor
        result[:, :, i] = np.mean(data[:, :, start_idx:end_idx], axis=2)
    
    return result


def downsample_file(
    file_path: Path,
    time_downsample_factor: int = 1,
    freq_downsample_factor: int = 1,
    method: str = 'mean',
    max_samples: Optional[int] = None,
    save_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Lee un archivo astronómico y aplica downsampling.
    
    Args:
        file_path: Ruta al archivo .fits o .fil
        time_downsample_factor: Factor de downsampling temporal
        freq_downsample_factor: Factor de downsampling frecuencial
        method: Método de downsampling
        max_samples: Número máximo de muestras a procesar
        save_path: Ruta para guardar el resultado (opcional)
        
    Returns:
        Dict con datos procesados y metadata
    """
    # Intentar importación relativa primero
    try:
        from ..reader_files import astronomical_reader
    except ImportError:
        # Fallback a importación absoluta
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from reader_files import astronomical_reader
    
    logger.info(f"Procesando archivo: {file_path.name}")
    
    # Leer archivo
    file_data = astronomical_reader.read_astronomical_file(file_path, max_samples)
    
    # Extraer datos y metadata
    data = file_data['data']
    if data is None:
        raise ValueError("No se pudieron cargar los datos del archivo")
    
    logger.info(f"Datos originales: {data.shape}")
    
    # Aplicar downsampling
    downsampled_data = downsample_astronomical_data(
        data, time_downsample_factor, freq_downsample_factor, method
    )
    
    # Actualizar metadata
    result = file_data.copy()
    result['data'] = downsampled_data
    result['nsamples'] = downsampled_data.shape[0]
    result['nchans'] = downsampled_data.shape[2]
    
    # Actualizar resolución temporal si se aplicó downsampling temporal
    if time_downsample_factor > 1:
        result['time_reso'] = file_data['time_reso'] * time_downsample_factor
        logger.info(f"Resolución temporal actualizada: {file_data['time_reso']:.6f}s → {result['time_reso']:.6f}s")
    
    # Actualizar array de frecuencias si es necesario
    if freq_downsample_factor > 1:
        freq_array = file_data['freq_array']
        new_freq_array = _downsample_frequency_array(freq_array, freq_downsample_factor)
        result['freq_array'] = new_freq_array
        logger.info(f"Canales de frecuencia actualizados: {len(freq_array)} → {len(new_freq_array)}")
    
    # Actualizar tamaño de archivo estimado
    if 'file_size_gb' in result:
        original_size = result['file_size_gb']
        # Estimar nuevo tamaño basado en la reducción de datos
        size_reduction = (time_downsample_factor * freq_downsample_factor)
        estimated_new_size = original_size / size_reduction
        result['file_size_gb'] = estimated_new_size
        logger.info(f"Tamaño estimado actualizado: {original_size:.2f}GB → {estimated_new_size:.2f}GB")
    
    # Guardar si se especifica
    if save_path:
        _save_downsampled_data(result, save_path)
    
    logger.info(f"Downsampling completado: {downsampled_data.shape}")
    return result


def _downsample_frequency_array(freq_array: np.ndarray, factor: int) -> np.ndarray:
    """
    Aplica downsampling al array de frecuencias.
    
    Args:
        freq_array: Array de frecuencias original
        factor: Factor de downsampling
        
    Returns:
        Array de frecuencias downsampled
    """
    nchans = len(freq_array)
    new_nchans = nchans // factor
    
    # Usar las frecuencias centrales de cada grupo
    indices = np.arange(factor//2, nchans, factor)[:new_nchans]
    return freq_array[indices]


def _save_downsampled_data(data: Dict[str, Any], save_path: Path) -> None:
    """
    Guarda los datos downsampled.
    
    Args:
        data: Dict con datos y metadata
        save_path: Ruta para guardar
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Guardar como .npy
    np.save(save_path.with_suffix('.npy'), data['data'])
    
    # Guardar metadata como .json (excluyendo funciones y objetos no serializables)
    import json
    metadata = {}
    
    for k, v in data.items():
        if k == 'data':
            continue  # Los datos se guardan por separado
        elif k == 'read_chunk':
            continue  # Excluir funciones
        elif callable(v):
            continue  # Excluir cualquier función
        elif isinstance(v, np.ndarray):
            metadata[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            metadata[k] = v.item()
        else:
            try:
                # Intentar serializar
                json.dumps(v)
                metadata[k] = v
            except (TypeError, ValueError):
                # Si no se puede serializar, convertir a string
                metadata[k] = str(v)
    
    with open(save_path.with_suffix('.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Datos guardados en: {save_path}")


def calculate_downsample_factors(
    target_time_reso: float,
    target_nchans: int,
    current_time_reso: float,
    current_nchans: int
) -> Tuple[int, int]:
    """
    Calcula los factores de downsampling necesarios para alcanzar
    la resolución temporal y número de canales objetivo.
    
    Args:
        target_time_reso: Resolución temporal objetivo (segundos)
        target_nchans: Número de canales objetivo
        current_time_reso: Resolución temporal actual (segundos)
        current_nchans: Número de canales actual
        
    Returns:
        Tuple[time_factor, freq_factor]
    """
    time_factor = max(1, int(target_time_reso / current_time_reso))
    freq_factor = max(1, int(current_nchans / target_nchans))
    
    return time_factor, freq_factor


def downsample_chunked(
    file_path: Path,
    time_downsample_factor: int = 1,
    freq_downsample_factor: int = 1,
    method: str = 'mean',
    chunk_size: int = 100_000,
    save_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Aplica downsampling a archivos grandes procesando por chunks.
    
    Args:
        file_path: Ruta al archivo
        time_downsample_factor: Factor de downsampling temporal
        freq_downsample_factor: Factor de downsampling frecuencial
        method: Método de downsampling
        chunk_size: Tamaño de cada chunk
        save_path: Ruta para guardar resultado
        
    Returns:
        Dict con metadata del procesamiento (sin datos completos para archivos muy grandes)
    """
    # Intentar importación relativa primero
    try:
        from ..reader_files import astronomical_reader
    except ImportError:
        # Fallback a importación absoluta
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from reader_files import astronomical_reader
    
    logger.info(f"Procesando archivo grande por chunks: {file_path.name}")
    
    # Configurar lectura por chunks
    chunked_data = astronomical_reader.read_astronomical_file_chunked(file_path, chunk_size)
    
    total_samples = chunked_data['nsamples']
    read_chunk = chunked_data['read_chunk']
    
    # Calcular número de chunks
    n_chunks = (total_samples + chunk_size - 1) // chunk_size
    
    logger.info(f"Total de chunks: {n_chunks}")
    
    # Procesar primer chunk para obtener la forma de salida
    logger.info(f"Procesando chunk 1/{n_chunks} para determinar forma de salida...")
    first_chunk = read_chunk(0, min(chunk_size, total_samples))
    first_processed = downsample_astronomical_data(
        first_chunk, time_downsample_factor, freq_downsample_factor, method
    )
    
    # Calcular forma final y número total de muestras procesadas
    final_shape = (first_processed.shape[0], first_processed.shape[1], first_processed.shape[2])
    total_processed_samples = 0
    
    # Calcular el número total de muestras procesadas
    for i in range(n_chunks):
        start_sample = i * chunk_size
        end_sample = min(start_sample + chunk_size, total_samples)
        actual_samples = end_sample - start_sample
        
        # Calcular cuántas muestras tendrá este chunk después del downsampling temporal
        if time_downsample_factor > 1:
            processed_samples = actual_samples // time_downsample_factor
        else:
            processed_samples = actual_samples
        
        total_processed_samples += processed_samples
    
    logger.info(f"Forma final esperada: ({total_processed_samples}, {final_shape[1]}, {final_shape[2]})")
    
    # Verificar si el tamaño final sería demasiado grande para la memoria
    estimated_size_gb = (total_processed_samples * final_shape[1] * final_shape[2] * 8) / (1024**3)  # 8 bytes por float64
    
    if estimated_size_gb > 10:  # Si es mayor a 10GB, no cargar en memoria
        logger.warning(f"Tamaño estimado muy grande ({estimated_size_gb:.1f}GB). Procesando en modo streaming sin cargar datos completos en memoria.")
        
        # Procesar solo el primer chunk para mostrar el resultado
        logger.info("Procesando solo el primer chunk como ejemplo...")
        
        # Crear resultado con metadata pero sin datos completos
        result = chunked_data.copy()
        result['data'] = first_processed  # Solo el primer chunk procesado como ejemplo
        result['nsamples'] = total_processed_samples
        result['nchans'] = final_shape[2]
        result['processing_mode'] = 'streaming'
        result['estimated_final_size_gb'] = estimated_size_gb
        result['chunks_processed'] = 1
        result['total_chunks'] = n_chunks
        
        # Actualizar resolución temporal si se aplicó downsampling temporal
        if time_downsample_factor > 1:
            result['time_reso'] = chunked_data['time_reso'] * time_downsample_factor
            logger.info(f"Resolución temporal actualizada: {chunked_data['time_reso']:.6f}s → {result['time_reso']:.6f}s")
        
        # Actualizar frecuencias si es necesario
        if freq_downsample_factor > 1:
            freq_array = chunked_data['freq_array']
            new_freq_array = _downsample_frequency_array(freq_array, freq_downsample_factor)
            result['freq_array'] = new_freq_array
            logger.info(f"Canales de frecuencia actualizados: {len(freq_array)} → {len(new_freq_array)}")
        
        # Actualizar tamaño de archivo estimado
        if 'file_size_gb' in result:
            original_size = result['file_size_gb']
            # Estimar nuevo tamaño basado en la reducción de datos
            size_reduction = (time_downsample_factor * freq_downsample_factor)
            estimated_new_size = original_size / size_reduction
            result['file_size_gb'] = estimated_new_size
            logger.info(f"Tamaño estimado actualizado: {original_size:.2f}GB → {estimated_new_size:.2f}GB")
        
        # Guardar si se especifica (incluso en modo streaming)
        if save_path:
            logger.info(f"Guardando datos del primer chunk como ejemplo en modo streaming...")
            _save_downsampled_data(result, save_path)
        
        logger.info(f"Procesamiento en modo streaming completado. Solo se procesó 1/{n_chunks} chunks como ejemplo.")
        return result
    
    else:
        # Si el tamaño es manejable, procesar normalmente
        logger.info(f"Tamaño estimado manejable ({estimated_size_gb:.1f}GB). Procesando todos los chunks...")
        
        # Crear array final para almacenar todos los datos procesados
        final_data = np.zeros((total_processed_samples, final_shape[1], final_shape[2]), dtype=first_processed.dtype)
        
        # Copiar el primer chunk procesado
        final_data[:first_processed.shape[0]] = first_processed
        current_position = first_processed.shape[0]
        
        # Procesar el resto de chunks
        for i in range(1, n_chunks):
            start_sample = i * chunk_size
            end_sample = min(start_sample + chunk_size, total_samples)
            actual_samples = end_sample - start_sample
            
            logger.info(f"Procesando chunk {i+1}/{n_chunks}: muestras {start_sample:,}-{end_sample-1:,}")
            
            # Leer chunk
            chunk_data = read_chunk(start_sample, actual_samples)
            
            # Aplicar downsampling
            downsampled_chunk = downsample_astronomical_data(
                chunk_data, time_downsample_factor, freq_downsample_factor, method
            )
            
            # Agregar al array final
            chunk_size_processed = downsampled_chunk.shape[0]
            final_data[current_position:current_position + chunk_size_processed] = downsampled_chunk
            current_position += chunk_size_processed
            
            # Limpiar memoria
            del chunk_data, downsampled_chunk
        
        # Crear resultado
        result = chunked_data.copy()
        result['data'] = final_data
        result['nsamples'] = final_data.shape[0]
        result['nchans'] = final_data.shape[2]
        
        # Actualizar resolución temporal si se aplicó downsampling temporal
        if time_downsample_factor > 1:
            result['time_reso'] = chunked_data['time_reso'] * time_downsample_factor
            logger.info(f"Resolución temporal actualizada: {chunked_data['time_reso']:.6f}s → {result['time_reso']:.6f}s")
        
        # Actualizar frecuencias si es necesario
        if freq_downsample_factor > 1:
            freq_array = chunked_data['freq_array']
            new_freq_array = _downsample_frequency_array(freq_array, freq_downsample_factor)
            result['freq_array'] = new_freq_array
            logger.info(f"Canales de frecuencia actualizados: {len(freq_array)} → {len(new_freq_array)}")
        
        # Actualizar tamaño de archivo estimado
        if 'file_size_gb' in result:
            original_size = result['file_size_gb']
            # Estimar nuevo tamaño basado en la reducción de datos
            size_reduction = (time_downsample_factor * freq_downsample_factor)
            estimated_new_size = original_size / size_reduction
            result['file_size_gb'] = estimated_new_size
            logger.info(f"Tamaño estimado actualizado: {original_size:.2f}GB → {estimated_new_size:.2f}GB")
        
        # Guardar si se especifica
        if save_path:
            _save_downsampled_data(result, save_path)
        
        logger.info(f"Procesamiento por chunks completado: {final_data.shape}")
        return result 