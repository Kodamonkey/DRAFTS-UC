"""
FITS File Reader for DRAFTS Pipeline
====================================

Este módulo contiene únicamente la funcionalidad de lectura de archivos FITS:
- Lectura de archivos FITS
- Extracción de parámetros críticos para procesamiento posterior
- Conversión a arrays numpy

NO incluye preprocesamiento, downsampling o transformaciones.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


def read_fits_file(file_path: Path) -> Dict[str, Any]:
    """
    Lee un archivo FITS y extrae los parámetros críticos necesarios
    para el procesamiento posterior.
    
    Args:
        file_path: Ruta al archivo .fits
        
    Returns:
        Dict con los parámetros críticos:
            - time_reso: Resolución temporal (segundos)
            - nsamples: Número de muestras
            - nchans: Número de canales
            - nifs: Número de polarizaciones
            - freq_array: Array de frecuencias (MHz)
            - data: Datos brutos (muestras, polarizaciones, canales)
            - header_names: Mapeo de nombres de headers específicos del formato
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    if file_path.suffix.lower() != ".fits":
        raise ValueError(f"El archivo debe ser .fits, no {file_path.suffix}")
    
    return _read_fits_file(file_path)


def _read_fits_file(file_path: Path) -> Dict[str, Any]:
    """Lee un archivo FITS y extrae parámetros críticos para procesamiento posterior."""
    try:
        from astropy.io import fits
    except ImportError:
        raise ImportError("astropy no está instalado. Instale con: pip install astropy")
    
    with fits.open(file_path) as hdul:
        # Buscar la extensión con datos
        data = None
        header = None
        data_ext = None
        
        for i, hdu in enumerate(hdul):
            if hdu.data is not None and hdu.data.size > 0:
                # Preferir datos con más dimensiones o más elementos
                if data is None or hdu.data.size > data.size:
                    data = hdu.data
                    header = hdu.header
                    data_ext = i
                    logger.info(f"Usando datos de extensión {i}: {data.shape}")
        
        if data is None:
            raise ValueError("No se encontraron datos en el archivo FITS")
        
        # Normalizar la forma de los datos
        original_shape = data.shape
        logger.info(f"Forma original de datos: {original_shape}")
        
        # Convertir a array numpy si no lo es
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Si es un FITS_rec o array de records, convertirlo a array numpy simple
        if hasattr(data, '_coldefs') or data.dtype.names is not None:  # Es un FITS_rec o array de records
            logger.info("Convirtiendo FITS_rec/records a numpy array")
            try:
                # Intentar convertir a float
                data = np.array(data, dtype=float)
            except (ValueError, TypeError):
                # Si falla, intentar con la primera columna o elemento
                try:
                    if hasattr(data, 'dtype') and data.dtype.names:
                        # Es un array de records, tomar la primera columna
                        first_col = data.dtype.names[0]
                        data = np.array(data[first_col], dtype=float)
                    else:
                        # Intentar convertir a lista y luego a array
                        data = np.array([float(x) for x in data.flatten()])
                except (ValueError, TypeError):
                    # Si aún falla, usar una conversión más robusta
                    logger.warning("Usando conversión robusta para datos complejos")
                    data_list = []
                    for item in data.flatten():
                        try:
                            if isinstance(item, (list, tuple)):
                                data_list.extend([float(x) for x in item])
                            else:
                                data_list.append(float(item))
                        except (ValueError, TypeError):
                            data_list.append(0.0)  # Valor por defecto
                    data = np.array(data_list)
        
        # Normalizar la forma para el procesamiento
        if len(original_shape) == 1:
            # Datos 1D: (muestras,) -> (muestras, 1, 1)
            data = data.reshape(-1, 1, 1)
            nsamples = original_shape[0]
            nchans = 1
            nifs = 1
        elif len(original_shape) == 2:
            # Datos 2D: (muestras, canales) -> (muestras, 1, canales)
            data = data.reshape(original_shape[0], 1, original_shape[1])
            nsamples = original_shape[0]
            nchans = original_shape[1]
            nifs = 1
        elif len(original_shape) == 3:
            # Datos 3D: (muestras, polarizaciones, canales) - formato esperado
            nsamples = original_shape[0]
            nifs = original_shape[1]
            nchans = original_shape[2]
        else:
            raise ValueError(f"Forma de datos no soportada: {original_shape}")
        
        # Extraer los parámetros críticos para procesamiento
        # 1. Resolución temporal (TBIN)
        time_reso = header.get('TBIN', 0.000064)
        
        # 2. Número de muestras (ya calculado arriba)
        if nsamples <= 0:
            nsamples = data.shape[0]
        
        # 3. Número de canales (ya calculado arriba)
        if nchans <= 0:
            nchans = data.shape[2] if len(data.shape) >= 3 else 1
        
        # 4. Número de polarizaciones (ya calculado arriba)
        if nifs <= 0:
            nifs = data.shape[1] if len(data.shape) >= 3 else 1
        
        # 5. Array de frecuencias (FCH1 + FCHANBW)
        fch1 = header.get('FCH1', 1500.0)  # Frecuencia inicial (MHz)
        fchanbw = header.get('FCHANBW', -1.0)  # Ancho de canal (MHz)
        freq_array = np.linspace(fch1, fch1 + nchans * fchanbw, nchans)
        
        logger.info(f"Datos normalizados: {data.shape} (muestras={nsamples}, polarizaciones={nifs}, canales={nchans})")
        
        return {
            'time_reso': time_reso,      # Resolución temporal (segundos)
            'nsamples': nsamples,        # Número de muestras
            'nchans': nchans,            # Número de canales
            'nifs': nifs,                # Número de polarizaciones
            'freq_array': freq_array,    # Array de frecuencias (MHz)
            'data': data,                # Datos brutos (muestras, polarizaciones, canales)
            'header_names': {
                'time_reso': 'TBIN',
                'nsamples': 'NSAMPS',
                'nchans': 'NCHAN',
                'nifs': 'NPOL',
                'freq_start': 'FCH1',
                'freq_bw': 'FCHANBW'
            }
        } 