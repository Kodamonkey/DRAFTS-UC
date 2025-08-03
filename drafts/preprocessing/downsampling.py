"""
Módulo de Downsampling - Reducción de Resolución de Datos
========================================================

Este módulo se encarga de reducir la resolución temporal y frecuencial
de los datos astronómicos para optimizar el procesamiento.

Responsabilidades:
- Reducir resolución temporal usando DOWN_TIME_RATE
- Reducir resolución frecuencial usando DOWN_FREQ_RATE
- Aplicar promediado para mantener la integridad de los datos
- Optimizar el uso de memoria durante el procesamiento

Para astrónomos:
- Usar downsample_data() para reducir resolución de datos FITS/FIL
- Los parámetros se configuran en config.DOWN_TIME_RATE y config.DOWN_FREQ_RATE
"""

from __future__ import annotations

import numpy as np
import logging

from .. import config

logger = logging.getLogger(__name__)


def downsample_data(data: np.ndarray) -> np.ndarray:
    """
    Reduce la resolución temporal y frecuencial de datos time-frequency.
    
    Aplica downsampling usando las tasas configuradas en config:
    - DOWN_TIME_RATE: Factor de reducción temporal
    - DOWN_FREQ_RATE: Factor de reducción frecuencial
    
    Parameters
    ----------
    data : np.ndarray
        Datos de entrada con forma (tiempo, polarización, frecuencia)
        
    Returns
    -------
    np.ndarray
        Datos downsampled con forma (tiempo_reducido, frecuencia_reducida)
        
    Notes
    -----
    - Los datos se promedian en bloques de DOWN_TIME_RATE × DOWN_FREQ_RATE
    - Se mantiene la polarización promediada
    - El resultado es float32 para optimizar memoria
    """
    if data is None or data.size == 0:
        logger.error("Datos de entrada vacíos o None en downsample_data")
        return np.array([])
    
    if len(data.shape) != 3:
        logger.error(f"Forma de datos incorrecta: {data.shape}, esperado (tiempo, pol, freq)")
        return data
    
    # Asegurar que las dimensiones sean divisibles por las tasas de downsampling
    n_time = (data.shape[0] // config.DOWN_TIME_RATE) * config.DOWN_TIME_RATE
    n_freq = (data.shape[2] // config.DOWN_FREQ_RATE) * config.DOWN_FREQ_RATE
    n_pol = data.shape[1]
    
    # Recortar datos a dimensiones válidas
    data = data[:n_time, :, :n_freq]
    
    # Reshape para aplicar downsampling
    data = data.reshape(
        n_time // config.DOWN_TIME_RATE,
        config.DOWN_TIME_RATE,
        n_pol,
        n_freq // config.DOWN_FREQ_RATE,
        config.DOWN_FREQ_RATE,
    )
    
    # Aplicar promediado en tiempo, frecuencia y polarización
    data = data.mean(axis=(1, 4, 2)).astype(np.float32)
    
    logger.debug(f"Downsampling aplicado: {data.shape[0]}x{data.shape[1]} muestras")
    return data


def validate_downsampling_parameters() -> bool:
    """
    Valida que los parámetros de downsampling sean correctos.
    
    Returns
    -------
    bool
        True si los parámetros son válidos, False en caso contrario
    """
    if config.DOWN_TIME_RATE <= 0:
        logger.error("DOWN_TIME_RATE debe ser mayor que 0")
        return False
    
    if config.DOWN_FREQ_RATE <= 0:
        logger.error("DOWN_FREQ_RATE debe ser mayor que 0")
        return False
    
    return True


def get_downsampling_info(data_shape: tuple) -> dict:
    """
    Obtiene información sobre el downsampling que se aplicará.
    
    Parameters
    ----------
    data_shape : tuple
        Forma original de los datos (tiempo, pol, freq)
        
    Returns
    -------
    dict
        Información sobre el downsampling
    """
    if len(data_shape) != 3:
        return {"error": "Forma de datos incorrecta"}
    
    n_time, n_pol, n_freq = data_shape
    
    # Calcular dimensiones después del downsampling
    n_time_ds = (n_time // config.DOWN_TIME_RATE) * config.DOWN_TIME_RATE
    n_freq_ds = (n_freq // config.DOWN_FREQ_RATE) * config.DOWN_FREQ_RATE
    
    n_time_out = n_time_ds // config.DOWN_TIME_RATE
    n_freq_out = n_freq_ds // config.DOWN_FREQ_RATE
    
    return {
        "input_shape": data_shape,
        "output_shape": (n_time_out, n_freq_out),
        "time_reduction": config.DOWN_TIME_RATE,
        "freq_reduction": config.DOWN_FREQ_RATE,
        "time_samples_lost": n_time - n_time_ds,
        "freq_samples_lost": n_freq - n_freq_ds,
        "compression_ratio": (n_time * n_freq) / (n_time_out * n_freq_out)
    }