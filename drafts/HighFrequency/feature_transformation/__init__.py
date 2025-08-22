"""
Feature Transformation Module for DRAFTS Pipeline
================================================

Este módulo contiene transformaciones de características específicas para ML/DL:
- Transformación DM-Time (Dispersión de Medida) con aceleración CUDA
- Normalización y estandarización de características
- Transformaciones para modelos pre-entrenados
- Mapeo de colores y redimensionamiento

Dependencias especiales:
- CUDA (numba.cuda)
- PyTorch
- OpenCV
- Matplotlib
"""

from .dm_time_transformation import (
    d_dm_time_gpu,
    d_dm_time_cpu,
    d_dm_time_numba
)

from .feature_normalization import (
    preprocess_img,
    preprocess_data,
    normalize_features
)

__all__ = [
    'd_dm_time_gpu',
    'd_dm_time_cpu', 
    'd_dm_time_numba',
    'preprocess_img',
    'preprocess_data',
    'normalize_features'
]

__version__ = "1.0.0"
__author__ = "DRAFTS Team"
__description__ = "Feature transformation module for astronomical data" 