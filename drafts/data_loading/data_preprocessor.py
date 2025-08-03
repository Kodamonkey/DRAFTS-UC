"""
Preprocesador de Datos - Preprocesamiento básico de datos astronómicos
====================================================================

Este módulo proporciona funciones para preprocesar datos cargados,
incluyendo downsampling y preparación para análisis.

Funciones principales:
- load_and_preprocess_data: Cargar y preprocesar datos

Para astrónomos:
- Usar load_and_preprocess_data() para cargar y preparar datos
- Aplica downsampling automático para optimizar procesamiento
"""

import logging
import numpy as np
from pathlib import Path
from typing import Union

# Importar funciones originales para mantener compatibilidad
from ..input.data_loader import load_and_preprocess_data as _load_and_preprocess_data_original
from ..preprocessing import downsample_data

logger = logging.getLogger(__name__)


def load_and_preprocess_data(fits_path: Union[str, Path]) -> np.ndarray:
    """
    Cargar y preprocesar datos del archivo FITS o FIL.
    
    Args:
        fits_path: Ruta al archivo FITS o FIL
        
    Returns:
        np.ndarray: Datos preprocesados
        
    Ejemplo:
        >>> data = load_and_preprocess_data("observacion.fits")
        >>> print(f"Datos preprocesados: {data.shape}")
    """
    fits_path = Path(fits_path)
    
    if not fits_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {fits_path}")
    
    logger.info(f"Cargando y preprocesando datos: {fits_path}")
    
    try:
        # Usar función original para mantener compatibilidad
        data = _load_and_preprocess_data_original(fits_path)
        
        if data is None or data.size == 0:
            raise ValueError(f"Datos vacíos después del preprocesamiento: {fits_path}")
        
        logger.info(f"Datos preprocesados exitosamente: {data.shape}")
        return data
        
    except Exception as e:
        logger.error(f"Error en preprocesamiento de datos {fits_path}: {e}")
        raise


def preprocess_data(data: np.ndarray, apply_downsampling: bool = True) -> np.ndarray:
    """
    Preprocesar datos numpy aplicando transformaciones básicas.
    
    Args:
        data: Array numpy con los datos
        apply_downsampling: Si aplicar downsampling automático
        
    Returns:
        np.ndarray: Datos preprocesados
        
    Ejemplo:
        >>> processed_data = preprocess_data(raw_data)
        >>> print(f"Datos procesados: {processed_data.shape}")
    """
    if data is None or data.size == 0:
        raise ValueError("Datos vacíos para preprocesamiento")
    
    logger.info(f"Preprocesando datos: {data.shape}")
    
    # Copiar datos para evitar modificar el original
    processed_data = data.copy()
    
    # Aplicar downsampling si se solicita
    if apply_downsampling:
        try:
            processed_data = downsample_data(processed_data)
            logger.info(f"Datos después de downsampling: {processed_data.shape}")
        except Exception as e:
            logger.warning(f"Error en downsampling, usando datos originales: {e}")
    
    # Verificar que los datos siguen siendo válidos
    if processed_data is None or processed_data.size == 0:
        raise ValueError("Datos se volvieron vacíos después del preprocesamiento")
    
    # Verificar que no hay valores NaN o Inf
    if np.any(np.isnan(processed_data)):
        logger.warning("Datos contienen NaN después del preprocesamiento")
        processed_data = np.nan_to_num(processed_data, nan=0.0)
    
    if np.any(np.isinf(processed_data)):
        logger.warning("Datos contienen Inf después del preprocesamiento")
        processed_data = np.nan_to_num(processed_data, posinf=0.0, neginf=0.0)
    
    logger.info(f"Preprocesamiento completado: {processed_data.shape}")
    return processed_data


def normalize_data(data: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalizar datos usando diferentes métodos.
    
    Args:
        data: Array numpy con los datos
        method: Método de normalización ('zscore', 'minmax', 'robust')
        
    Returns:
        np.ndarray: Datos normalizados
        
    Ejemplo:
        >>> normalized_data = normalize_data(data, 'zscore')
        >>> print(f"Datos normalizados: {normalized_data.shape}")
    """
    if data is None or data.size == 0:
        raise ValueError("Datos vacíos para normalización")
    
    logger.info(f"Normalizando datos usando método: {method}")
    
    normalized_data = data.copy()
    
    if method == 'zscore':
        # Normalización Z-score
        mean_val = np.mean(normalized_data)
        std_val = np.std(normalized_data)
        if std_val > 0:
            normalized_data = (normalized_data - mean_val) / std_val
        else:
            logger.warning("Desviación estándar es 0, usando datos originales")
            
    elif method == 'minmax':
        # Normalización Min-Max
        min_val = np.min(normalized_data)
        max_val = np.max(normalized_data)
        if max_val > min_val:
            normalized_data = (normalized_data - min_val) / (max_val - min_val)
        else:
            logger.warning("Máximo y mínimo son iguales, usando datos originales")
            
    elif method == 'robust':
        # Normalización robusta usando percentiles
        p25 = np.percentile(normalized_data, 25)
        p75 = np.percentile(normalized_data, 75)
        iqr = p75 - p25
        if iqr > 0:
            normalized_data = (normalized_data - p25) / iqr
        else:
            logger.warning("IQR es 0, usando datos originales")
    else:
        logger.warning(f"Método de normalización '{method}' no reconocido, usando datos originales")
    
    logger.info(f"Normalización completada: rango [{normalized_data.min():.3f}, {normalized_data.max():.3f}]")
    return normalized_data


def apply_data_transformations(data: np.ndarray, transformations: list) -> np.ndarray:
    """
    Aplicar múltiples transformaciones a los datos.
    
    Args:
        data: Array numpy con los datos
        transformations: Lista de transformaciones a aplicar
        
    Returns:
        np.ndarray: Datos transformados
        
    Ejemplo:
        >>> transforms = ['downsample', 'normalize']
        >>> transformed_data = apply_data_transformations(data, transforms)
    """
    if data is None or data.size == 0:
        raise ValueError("Datos vacíos para transformaciones")
    
    logger.info(f"Aplicando {len(transformations)} transformaciones")
    
    transformed_data = data.copy()
    
    for i, transform in enumerate(transformations):
        try:
            if transform == 'downsample':
                transformed_data = downsample_data(transformed_data)
                logger.info(f"Transformación {i+1}: downsampling -> {transformed_data.shape}")
                
            elif transform == 'normalize':
                transformed_data = normalize_data(transformed_data, 'zscore')
                logger.info(f"Transformación {i+1}: normalización completada")
                
            elif transform == 'minmax_normalize':
                transformed_data = normalize_data(transformed_data, 'minmax')
                logger.info(f"Transformación {i+1}: normalización minmax completada")
                
            elif transform == 'robust_normalize':
                transformed_data = normalize_data(transformed_data, 'robust')
                logger.info(f"Transformación {i+1}: normalización robusta completada")
                
            else:
                logger.warning(f"Transformación '{transform}' no reconocida, saltando...")
                
        except Exception as e:
            logger.error(f"Error en transformación '{transform}': {e}")
            # Continuar con las siguientes transformaciones
    
    logger.info(f"Todas las transformaciones aplicadas: {transformed_data.shape}")
    return transformed_data


def get_preprocessing_summary(data: np.ndarray, original_shape: tuple = None) -> dict:
    """
    Obtener resumen del preprocesamiento aplicado.
    
    Args:
        data: Datos procesados
        original_shape: Forma original de los datos (opcional)
        
    Returns:
        Dict con resumen del preprocesamiento
        
    Ejemplo:
        >>> summary = get_preprocessing_summary(processed_data, original_shape)
        >>> print(f"Reducción de tamaño: {summary['size_reduction']:.1%}")
    """
    summary = {
        'current_shape': data.shape if data is not None else None,
        'current_size': data.size if data is not None else 0,
        'data_range': [float(data.min()), float(data.max())] if data is not None else [0, 0],
        'data_mean': float(np.mean(data)) if data is not None else 0,
        'data_std': float(np.std(data)) if data is not None else 0,
    }
    
    if original_shape is not None:
        original_size = np.prod(original_shape)
        current_size = data.size if data is not None else 0
        summary['original_shape'] = original_shape
        summary['original_size'] = original_size
        summary['size_reduction'] = 1 - (current_size / original_size) if original_size > 0 else 0
    
    logger.info(f"Resumen de preprocesamiento: {summary}")
    return summary


# Exportar funciones
__all__ = [
    'load_and_preprocess_data',
    'preprocess_data',
    'normalize_data',
    'apply_data_transformations',
    'get_preprocessing_summary',
] 