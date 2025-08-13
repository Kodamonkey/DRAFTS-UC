"""
Feature Normalization and Transformation
=======================================

Este módulo implementa transformaciones de características para ML/DL:
- Normalización y estandarización
- Transformaciones para modelos pre-entrenados
- Mapeo de colores y redimensionamiento
- Recorte de outliers

Basado en los algoritmos de d-center-main.py y d-resnet-main.py
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

# Intentar importar dependencias opcionales
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV no disponible. Algunas funciones deshabilitadas.")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib no disponible. Algunas funciones deshabilitadas.")


def preprocess_img(
    img: np.ndarray,
    target_size: tuple = (512, 512),
    normalize_method: str = 'imagenet',
    clip_percentiles: tuple = (0.1, 99.9)
) -> np.ndarray:
    """
    Preprocesa una imagen para modelos de ML/DL.
    
    Args:
        img: Array de imagen (height, width)
        target_size: Tamaño objetivo (height, width)
        normalize_method: Método de normalización ('imagenet', 'standard', 'min-max')
        clip_percentiles: Percentiles para recorte (min, max)
        
    Returns:
        Array preprocesado (channels, height, width) para PyTorch
    """
    img = img.copy()
    
    # 1. Normalización min-max inicial
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    
    # 2. Normalización estándar
    img = (img - np.mean(img)) / np.std(img)
    
    # 3. Redimensionamiento
    if CV2_AVAILABLE and target_size != img.shape:
        img = cv2.resize(img, target_size)
    elif target_size != img.shape:
        logger.warning("OpenCV no disponible, saltando redimensionamiento")
    
    # 4. Recorte por percentiles
    vmin, vmax = np.percentile(img, clip_percentiles)
    img = np.clip(img, vmin, vmax)
    
    # 5. Normalización final
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    
    # 6. Mapeo de colores (opcional)
    if MATPLOTLIB_AVAILABLE:
        img = plt.get_cmap('mako')(img)
        img = img[..., :3]  # Solo RGB
    
    # 7. Normalización específica del modelo
    if normalize_method == 'imagenet':
        # Normalización ImageNet
        img -= np.array([0.485, 0.456, 0.406])
        img /= np.array([0.229, 0.224, 0.225])
    elif normalize_method == 'standard':
        # Normalización estándar
        img = (img - np.mean(img)) / np.std(img)
    elif normalize_method == 'min-max':
        # Normalización min-max
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    
    # 8. Transponer para PyTorch (channels, height, width)
    if len(img.shape) == 3:
        img = img.transpose(2, 0, 1)
    else:
        img = img[np.newaxis, ...]
    
    return img.astype(np.float32)


def preprocess_data(
    data: np.ndarray,
    exp_cut: int = 5,
    normalize_method: str = 'standard'
) -> np.ndarray:
    """
    Preprocesa datos astronómicos para ML/DL.
    
    Args:
        data: Array de datos (height, width)
        exp_cut: Percentil para recorte de outliers
        normalize_method: Método de normalización
        
    Returns:
        Array preprocesado
    """
    data = data.copy()
    
    # 1. Offset
    data = data + 1
    
    # 2. Normalización por canal
    if len(data.shape) == 2:
        data /= np.mean(data, axis=0)
    else:
        data /= np.mean(data, axis=0, keepdims=True)
    
    # 3. Recorte por percentiles
    vmin, vmax = np.nanpercentile(data, [exp_cut, 100-exp_cut])
    data = np.clip(data, vmin, vmax)
    
    # 4. Normalización final
    data = (data - data.min()) / (data.max() - data.min())
    
    return data


def normalize_features(
    data: np.ndarray,
    method: str = 'standard',
    axis: Optional[int] = None
) -> np.ndarray:
    """
    Normaliza características usando diferentes métodos.
    
    Args:
        data: Array de datos
        method: Método de normalización ('standard', 'min-max', 'robust', 'quantile')
        axis: Eje para la normalización (None = global)
        
    Returns:
        Array normalizado
    """
    data = data.copy()
    
    if method == 'standard':
        # Normalización estándar (z-score)
        if axis is not None:
            mean = np.mean(data, axis=axis, keepdims=True)
            std = np.std(data, axis=axis, keepdims=True)
        else:
            mean = np.mean(data)
            std = np.std(data)
        data = (data - mean) / (std + 1e-8)
        
    elif method == 'min-max':
        # Normalización min-max
        if axis is not None:
            min_val = np.min(data, axis=axis, keepdims=True)
            max_val = np.max(data, axis=axis, keepdims=True)
        else:
            min_val = np.min(data)
            max_val = np.max(data)
        data = (data - min_val) / (max_val - min_val + 1e-8)
        
    elif method == 'robust':
        # Normalización robusta usando medianas
        if axis is not None:
            median = np.median(data, axis=axis, keepdims=True)
            mad = np.median(np.abs(data - median), axis=axis, keepdims=True)
        else:
            median = np.median(data)
            mad = np.median(np.abs(data - median))
        data = (data - median) / (mad + 1e-8)
        
    elif method == 'quantile':
        # Normalización por cuantiles
        q25 = np.percentile(data, 25, axis=axis, keepdims=True if axis is not None else False)
        q75 = np.percentile(data, 75, axis=axis, keepdims=True if axis is not None else False)
        data = (data - q25) / (q75 - q25 + 1e-8)
    
    return data


def postprocess_img(img: np.ndarray) -> np.ndarray:
    """
    Postprocesa una imagen para visualización.
    
    Args:
        img: Array de imagen (channels, height, width) de PyTorch
        
    Returns:
        Array postprocesado (height, width, channels) para visualización
    """
    img = img.copy()
    
    # 1. Transponer de PyTorch a formato de visualización
    if len(img.shape) == 3 and img.shape[0] in [1, 3, 4]:
        img = img.transpose(1, 2, 0)
    
    # 2. Desnormalización ImageNet si es necesario
    if img.shape[-1] == 3:
        img *= np.array([0.229, 0.224, 0.225])
        img += np.array([0.485, 0.456, 0.406])
    
    # 3. Escalar a [0, 255]
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    # 4. Convertir BGR a RGB si es necesario
    if CV2_AVAILABLE and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


def apply_color_map(
    data: np.ndarray,
    cmap_name: str = 'mako',
    normalize: bool = True
) -> np.ndarray:
    """
    Aplica un mapa de colores a datos 2D.
    
    Args:
        data: Array 2D de datos
        cmap_name: Nombre del colormap
        normalize: Si normalizar los datos antes de aplicar el colormap
        
    Returns:
        Array con colormap aplicado (height, width, 3)
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib no disponible, retornando datos originales")
        return data
    
    if normalize:
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    cmap = plt.get_cmap(cmap_name)
    colored_data = cmap(data)
    
    return colored_data[..., :3]  # Solo RGB 