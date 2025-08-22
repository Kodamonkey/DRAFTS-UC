"""
FIL/SIGPROC File Reader for DRAFTS Pipeline
===========================================

Este módulo contiene únicamente la funcionalidad de lectura de archivos FIL:
- Lectura de archivos SIGPROC filterbank (.fil)
- Extracción de parámetros críticos para procesamiento posterior
- Lectura por chunks para archivos grandes
- Conversión a arrays numpy

NO incluye preprocesamiento, downsampling o transformaciones.
"""

from __future__ import annotations

import os
import struct
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Constantes para manejo de archivos grandes
MAX_MEMORY_GB = 8  # Máximo 8GB de RAM
CHUNK_SIZE_SAMPLES = 100_000  # 100k muestras por chunk


def read_fil_file(file_path: Path, max_samples: Optional[int] = None) -> Dict[str, Any]:
    """
    Lee un archivo FIL y extrae los parámetros críticos necesarios
    para el procesamiento posterior.
    
    Args:
        file_path: Ruta al archivo .fil
        max_samples: Número máximo de muestras a leer (None = automático)
        
    Returns:
        Dict con los parámetros críticos:
            - time_reso: Resolución temporal (segundos)
            - nsamples: Número de muestras
            - nchans: Número de canales
            - nifs: Número de polarizaciones
            - freq_array: Array de frecuencias (MHz)
            - data: Datos brutos (muestras, polarizaciones, canales) o None si archivo muy grande
            - header_names: Mapeo de nombres de headers específicos del formato
            - file_size_gb: Tamaño del archivo en GB
            - is_large_file: True si el archivo es muy grande
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    if file_path.suffix.lower() != ".fil":
        raise ValueError(f"El archivo debe ser .fil, no {file_path.suffix}")
    
    return _read_fil_file(file_path, max_samples)


def _read_fil_file(file_path: Path, max_samples: Optional[int] = None) -> Dict[str, Any]:
    """Lee un archivo FIL y extrae parámetros críticos para procesamiento posterior."""
    # Mapeo de tipos de datos
    dtype_map = {
        8: np.uint8,
        16: np.int16,
        32: np.float32,
        64: np.float64
    }
    
    # Obtener tamaño del archivo
    file_size_bytes = os.path.getsize(file_path)
    file_size_gb = file_size_bytes / (1024**3)
    
    logger.info(f"Archivo FIL: {file_path.name}")
    logger.info(f"Tamaño: {file_size_gb:.2f} GB")
    
    with open(file_path, "rb") as f:
        # Leer header
        header, hdr_len = _read_sigproc_header(f)
        
        # Extraer los parámetros críticos para procesamiento
        # 1. Resolución temporal (tsamp)
        time_reso = header.get('tsamp', 0.000064)
        
        # 2. Número de muestras (nsamples)
        nsamples = header.get('nsamples')
        if nsamples is None:
            nchans = header.get("nchans", 512)
            nifs = header.get("nifs", 1)
            nbits = header.get("nbits", 8)
            bytes_per_sample = nifs * nchans * (nbits // 8)
            data_size = file_size_bytes - hdr_len
            nsamples = data_size // bytes_per_sample if bytes_per_sample > 0 else 1000
        
        # 3. Número de canales (nchans)
        nchans = header.get("nchans", 512)
        
        # 4. Número de polarizaciones (nifs)
        nifs = header.get("nifs", 1)
        
        # 5. Array de frecuencias (fch1 + foff)
        fch1 = header.get('fch1', 1500.0)  # Frecuencia inicial (MHz)
        foff = header.get('foff', -1.0)    # Ancho de canal (MHz)
        freq_array = np.linspace(fch1, fch1 + nchans * foff, nchans)
        
        # 6. Determinar si el archivo es muy grande
        nbits = header.get("nbits", 8)
        dtype = dtype_map.get(nbits, np.uint8)
        bytes_per_sample = nifs * nchans * (nbits // 8)
        total_memory_gb = (nsamples * bytes_per_sample) / (1024**3)
        
        is_large_file = total_memory_gb > MAX_MEMORY_GB
        
        logger.info(f"Parámetros del archivo:")
        logger.info(f"  - Muestras: {nsamples:,}")
        logger.info(f"  - Canales: {nchans}")
        logger.info(f"  - Polarizaciones: {nifs}")
        logger.info(f"  - Memoria requerida: {total_memory_gb:.2f} GB")
        logger.info(f"  - Archivo grande: {is_large_file}")
        
        # 7. Leer datos según el tamaño del archivo
        if is_large_file:
            logger.warning(f"Archivo muy grande ({total_memory_gb:.2f} GB). Solo extrayendo metadata.")
            data = None
            
            # Si se especifica max_samples, leer solo esa cantidad
            if max_samples is not None and max_samples > 0:
                samples_to_read = min(max_samples, nsamples)
                logger.info(f"Leyendo solo {samples_to_read:,} muestras de {nsamples:,}")
                data = _read_fil_chunk(file_path, hdr_len, samples_to_read, nifs, nchans, dtype)
        else:
            # Archivo pequeño, leer todo
            logger.info("Leyendo archivo completo...")
            data = np.fromfile(file_path, dtype=dtype, offset=hdr_len)
            data = data.reshape(nsamples, nifs, nchans)
        
        return {
            'time_reso': time_reso,      # Resolución temporal (segundos)
            'nsamples': nsamples,        # Número de muestras
            'nchans': nchans,            # Número de canales
            'nifs': nifs,                # Número de polarizaciones
            'freq_array': freq_array,    # Array de frecuencias (MHz)
            'data': data,                # Datos brutos (None si archivo muy grande)
            'header_names': {
                'time_reso': 'tsamp',
                'nsamples': 'nsamples',
                'nchans': 'nchans',
                'nifs': 'nifs',
                'freq_start': 'fch1',
                'freq_bw': 'foff'
            },
            'file_size_gb': file_size_gb,
            'is_large_file': is_large_file,
            'total_memory_gb': total_memory_gb
        }


def _read_fil_chunk(file_path: Path, hdr_len: int, samples: int, nifs: int, nchans: int, dtype) -> np.ndarray:
    """
    Lee un chunk específico de un archivo FIL.
    
    Args:
        file_path: Ruta al archivo
        hdr_len: Longitud del header en bytes
        samples: Número de muestras a leer
        nifs: Número de polarizaciones
        nchans: Número de canales
        dtype: Tipo de datos
        
    Returns:
        Array con los datos del chunk
    """
    bytes_per_sample = nifs * nchans * dtype().itemsize
    bytes_to_read = samples * bytes_per_sample
    
    with open(file_path, "rb") as f:
        f.seek(hdr_len)  # Saltar header
        data = np.fromfile(f, dtype=dtype, count=samples * nifs * nchans)
        data = data.reshape(samples, nifs, nchans)
    
    logger.info(f"Chunk leído: {data.shape}")
    return data


def read_fil_file_chunked(file_path: Path, chunk_size: int = CHUNK_SIZE_SAMPLES) -> Dict[str, Any]:
    """
    Lee un archivo FIL grande por chunks.
    
    Args:
        file_path: Ruta al archivo .fil
        chunk_size: Tamaño de cada chunk en muestras
        
    Returns:
        Dict con metadata y función para leer chunks
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    if file_path.suffix.lower() != ".fil":
        raise ValueError(f"El archivo debe ser .fil, no {file_path.suffix}")
    
    # Mapeo de tipos de datos
    dtype_map = {
        8: np.uint8,
        16: np.int16,
        32: np.float32,
        64: np.float64
    }
    
    with open(file_path, "rb") as f:
        # Leer header
        header, hdr_len = _read_sigproc_header(f)
        
        # Extraer parámetros
        time_reso = header.get('tsamp', 0.000064)
        nsamples = header.get('nsamples')
        nchans = header.get("nchans", 512)
        nifs = header.get("nifs", 1)
        nbits = header.get("nbits", 8)
        dtype = dtype_map.get(nbits, np.uint8)
        
        if nsamples is None:
            bytes_per_sample = nifs * nchans * (nbits // 8)
            file_size = os.path.getsize(file_path) - hdr_len
            nsamples = file_size // bytes_per_sample if bytes_per_sample > 0 else 1000
        
        fch1 = header.get('fch1', 1500.0)
        foff = header.get('foff', -1.0)
        freq_array = np.linspace(fch1, fch1 + nchans * foff, nchans)
        
        def read_chunk(start_sample: int, num_samples: int) -> np.ndarray:
            """Función para leer un chunk específico."""
            if start_sample >= nsamples:
                raise ValueError(f"start_sample ({start_sample}) >= nsamples ({nsamples})")
            
            end_sample = min(start_sample + num_samples, nsamples)
            actual_samples = end_sample - start_sample
            
            return _read_fil_chunk(file_path, hdr_len, actual_samples, nifs, nchans, dtype)
        
        return {
            'time_reso': time_reso,
            'nsamples': nsamples,
            'nchans': nchans,
            'nifs': nifs,
            'freq_array': freq_array,
            'data': None,  # No se cargan todos los datos
            'header_names': {
                'time_reso': 'tsamp',
                'nsamples': 'nsamples',
                'nchans': 'nchans',
                'nifs': 'nifs',
                'freq_start': 'fch1',
                'freq_bw': 'foff'
            },
            'read_chunk': read_chunk,
            'chunk_size': chunk_size,
            'is_large_file': True
        }


def _read_sigproc_header(f) -> tuple[Dict[str, Any], int]:
    """
    Lee el header de un archivo SIGPROC filterbank.
    
    Args:
        f: Archivo abierto en modo binario
        
    Returns:
        Tuple[header, header_length]: Header como diccionario y longitud en bytes
    """
    original_pos = f.tell()
    
    try:
        # Intentar leer como formato SIGPROC estándar
        start = _read_string(f)
        if start != "HEADER_START":
            f.seek(original_pos)
            return _read_non_standard_header(f)

        header = {}
        while True:
            try:
                key = _read_string(f)
                if key == "HEADER_END":
                    break
                    
                # Campos de texto
                if key in {"rawdatafile", "source_name"}:
                    header[key] = _read_string(f)
                    
                # Campos enteros
                elif key in {"telescope_id", "machine_id", "data_type", "barycentric", 
                           "pulsarcentric", "nbits", "nchans", "nifs", "nbeams", 
                           "ibeam", "nsamples"}:
                    header[key] = _read_int(f)
                    
                # Campos double
                elif key in {"az_start", "za_start", "src_raj", "src_dej", "tstart", 
                           "tsamp", "fch1", "foff", "refdm"}:
                    header[key] = _read_double(f)
                    
                # Campos desconocidos (asumir entero)
                else:
                    header[key] = _read_int(f)
                    
            except (struct.error, UnicodeDecodeError) as e:
                logger.warning(f"Error leyendo campo '{key}': {e}")
                continue
                
        return header, f.tell()
        
    except Exception as e:
        logger.warning(f"Error leyendo header SIGPROC estándar: {e}")
        f.seek(original_pos)
        return _read_non_standard_header(f)


def _read_non_standard_header(f) -> tuple[Dict[str, Any], int]:
    """
    Maneja archivos FIL con formato no estándar.
    
    Args:
        f: Archivo abierto en modo binario
        
    Returns:
        Tuple[header, header_length]: Header estimado y longitud
    """
    logger.info("Detectado archivo .fil con formato no estándar, usando parámetros estimados")
    
    # Obtener tamaño del archivo
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
    
    # Límite máximo de muestras para evitar problemas de memoria
    max_samples = 100_000_000  # 100M muestras
    header["nsamples"] = min(estimated_samples, max_samples)
    
    logger.info(f"Parámetros estimados para archivo no estándar:")
    logger.info(f"  - Tamaño de archivo: {file_size / (1024**2):.1f} MB")
    logger.info(f"  - Muestras estimadas: {estimated_samples:,}")
    logger.info(f"  - Muestras a usar: {header['nsamples']:,}")
    
    return header, 512


def _read_int(f) -> int:
    """Lee un entero de 4 bytes."""
    return struct.unpack('i', f.read(4))[0]


def _read_double(f) -> float:
    """Lee un double de 8 bytes."""
    return struct.unpack('d', f.read(8))[0]


def _read_string(f) -> str:
    """Lee una cadena de caracteres."""
    length = struct.unpack('i', f.read(4))[0]
    return f.read(length).decode('utf-8').strip('\x00') 