"""
Astronomical Data Reader for DRAFTS Pipeline
============================================

Este módulo contiene únicamente la funcionalidad de input pura:
- Lectura de archivos FITS y FIL
- Extracción de parámetros críticos para procesamiento posterior
- Lectura por chunks para archivos grandes
- Conversión a arrays numpy

NO incluye preprocesamiento, downsampling o transformaciones.

Este módulo actúa como unificador de los módulos especializados:
- fits_reader.py: Para archivos FITS
- fil_reader.py: Para archivos FIL (SIGPROC)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from . import fits_reader
from . import fil_reader

logger = logging.getLogger(__name__)


def read_astronomical_file(file_path: Path, max_samples: Optional[int] = None) -> Dict[str, Any]:
    """
    Lee un archivo astronómico (FITS o FIL) y extrae los parámetros críticos
    necesarios para el procesamiento posterior.
    
    Esta función unifica la lectura de archivos astronómicos en el pipeline DRAFTS,
    independientemente del formato.
    
    Args:
        file_path: Ruta al archivo .fits o .fil
        max_samples: Número máximo de muestras a leer (solo para archivos grandes)
        
    Returns:
        Dict con los parámetros críticos unificados:
            - time_reso: Resolución temporal (segundos)
            - nsamples: Número de muestras
            - nchans: Número de canales
            - nifs: Número de polarizaciones
            - freq_array: Array de frecuencias (MHz)
            - data: Datos brutos (muestras, polarizaciones, canales) o None si archivo muy grande
            - header_names: Mapeo de nombres de headers específicos del formato
            - file_size_gb: Tamaño del archivo en GB (solo para archivos grandes)
            - is_large_file: True si el archivo es muy grande
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == ".fits":
        return fits_reader.read_fits_file(file_path)
    elif suffix == ".fil":
        return fil_reader.read_fil_file(file_path, max_samples)
    else:
        raise ValueError(f"Formato de archivo no soportado: {suffix}")


def read_astronomical_file_chunked(file_path: Path, chunk_size: int = 100_000) -> Dict[str, Any]:
    """
    Lee un archivo astronómico grande por chunks.
    
    Args:
        file_path: Ruta al archivo .fits o .fil
        chunk_size: Tamaño de cada chunk en muestras
        
    Returns:
        Dict con metadata y función para leer chunks
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == ".fits":
        # Para archivos FITS, crear una función de lectura por chunks
        data = fits_reader.read_fits_file(file_path)
        
        # Crear función para leer chunks (siempre disponible)
        def read_chunk(start_sample: int, n_samples: int) -> np.ndarray:
            """Lee un chunk de datos."""
            if data['data'] is None:
                raise ValueError("No hay datos disponibles para leer")
            end_sample = min(start_sample + n_samples, data['nsamples'])
            return data['data'][start_sample:end_sample]
        
        # Agregar la función de lectura por chunks
        data['read_chunk'] = read_chunk
        return data
        
    elif suffix == ".fil":
        return fil_reader.read_fil_file_chunked(file_path, chunk_size)
    else:
        raise ValueError(f"Formato de archivo no soportado: {suffix}")


def get_file_info(file_path: Path) -> Dict[str, Any]:
    """
    Obtiene información básica de un archivo astronómico sin cargar los datos.
    
    Args:
        file_path: Ruta al archivo .fits o .fil
        
    Returns:
        Dict con información básica del archivo
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == ".fits":
        # Para FITS, usar astropy para obtener info básica
        from astropy.io import fits
        with fits.open(file_path) as hdul:
            header = hdul[0].header
            file_size_gb = file_path.stat().st_size / (1024**3)
            
            return {
                'file_type': 'FITS',
                'file_size_gb': file_size_gb,
                'nchans': header.get('NCHAN', 'Unknown'),
                'time_reso': header.get('TBIN', 'Unknown'),
                'data_shape': hdul[0].data.shape if hdul[0].data is not None else 'Unknown'
            }
    elif suffix == ".fil":
        # Para FIL, usar solo la función de lectura de header
        return fil_reader.read_fil_file(file_path, max_samples=0)  # Solo metadata
    else:
        raise ValueError(f"Formato de archivo no soportado: {suffix}") 