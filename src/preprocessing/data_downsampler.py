# This module performs temporal and spectral downsampling.

"""Data downsampler for FRB pipeline - reduces temporal and frequency resolution."""
from __future__ import annotations

                     
import numpy as np

               
from ..config import config


def downsample_data(data: np.ndarray) -> np.ndarray:
    """Down-sample time-frequency data usando las tasas de :mod:`config`.

    - Temporal: suma sobre ventanas de tamaño ``DOWN_TIME_RATE`` (estilo PRESTO).
    - Frecuencia: promedio sobre grupos de ``DOWN_FREQ_RATE`` canales.
    - Polarización: promedio (Stokes I ya seleccionado en carga si existe).
    """
    n_time = (data.shape[0] // config.DOWN_TIME_RATE) * config.DOWN_TIME_RATE
    n_freq = (data.shape[2] // config.DOWN_FREQ_RATE) * config.DOWN_FREQ_RATE
    n_pol = data.shape[1]
    data = data[:n_time, :, :n_freq]
    data = data.reshape(
        n_time // config.DOWN_TIME_RATE,
        config.DOWN_TIME_RATE,
        n_pol,
        n_freq // config.DOWN_FREQ_RATE,
        config.DOWN_FREQ_RATE,
    )
                                                                                              
                                                     
    data = data.sum(axis=1)                                 
    data = data.mean(axis=1)                             
    data = data.mean(axis=2)                       
    data = data.astype(np.float32)
    return data
