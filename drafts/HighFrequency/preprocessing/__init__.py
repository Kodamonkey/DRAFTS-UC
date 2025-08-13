"""
Preprocessing Module for DRAFTS Pipeline
=======================================

Este módulo contiene funcionalidades de preprocesamiento para datos astronómicos:
- Downsampling temporal y frecuencial
- Normalización de datos
- Filtrado y limpieza
- Transformaciones de datos

Basado en los algoritmos de d-center-main.py y d-resnet-main.py
"""

from .downsampling import (
    downsample_astronomical_data,
    downsample_file,
    downsample_chunked,
    calculate_downsample_factors,
    downsample_temporal_numba,
    downsample_frequency_numba
)

__all__ = [
    'downsample_astronomical_data',
    'downsample_file', 
    'downsample_chunked',
    'calculate_downsample_factors',
    'downsample_temporal_numba',
    'downsample_frequency_numba'
]

__version__ = "1.0.0"
__author__ = "DRAFTS Team"
__description__ = "Preprocessing module for astronomical data" 