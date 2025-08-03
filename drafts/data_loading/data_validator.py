"""
Validador de Datos - Verificar que los datos cargados son válidos
================================================================

Este módulo proporciona funciones para validar datos astronómicos cargados
y sus metadatos antes del procesamiento.

Funciones principales:
- validate_data: Validar datos numpy
- validate_metadata: Validar metadatos de archivos
- validate_frequency_range: Validar rango de frecuencias
- validate_time_range: Validar rango temporal

Para astrónomos:
- Usar validate_data() para verificar que los datos son válidos
- Usar validate_metadata() para verificar metadatos
- Usar las funciones específicas para validar frecuencias y tiempos
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


def validate_data(data: np.ndarray, expected_shape: Optional[Tuple[int, ...]] = None) -> bool:
    """
    Validar que los datos numpy son válidos para procesamiento.
    
    Args:
        data: Array numpy con los datos
        expected_shape: Forma esperada (tiempo, frecuencia) o None
        
    Returns:
        bool: True si los datos son válidos
        
    Ejemplo:
        >>> if validate_data(data, (1000, 512)):
        >>>     print("Datos válidos para procesamiento")
    """
    if data is None:
        logger.error("Datos son None")
        return False
    
    if not isinstance(data, np.ndarray):
        logger.error(f"Datos no son numpy array: {type(data)}")
        return False
    
    if data.size == 0:
        logger.error("Datos están vacíos")
        return False
    
    if data.ndim != 2:
        logger.error(f"Datos deben ser 2D, got {data.ndim}D")
        return False
    
    # Verificar que no hay valores NaN o Inf
    if np.any(np.isnan(data)):
        logger.error("Datos contienen valores NaN")
        return False
    
    if np.any(np.isinf(data)):
        logger.error("Datos contienen valores Inf")
        return False
    
    # Verificar forma si se especifica
    if expected_shape is not None:
        if data.shape != expected_shape:
            logger.error(f"Forma de datos incorrecta: {data.shape}, esperado: {expected_shape}")
            return False
    
    # Verificar que las dimensiones son razonables
    if data.shape[0] < 10 or data.shape[1] < 10:
        logger.error(f"Dimensiones de datos muy pequeñas: {data.shape}")
        return False
    
    logger.info(f"Datos válidos: {data.shape}, rango: [{data.min():.3f}, {data.max():.3f}]")
    return True


def validate_metadata(metadata: Dict[str, Any], required_keys: Optional[List[str]] = None) -> bool:
    """
    Validar metadatos de archivo astronómico.
    
    Args:
        metadata: Diccionario con metadatos
        required_keys: Lista de claves requeridas o None para usar valores por defecto
        
    Returns:
        bool: True si los metadatos son válidos
        
    Ejemplo:
        >>> required = ['frequency_resolution', 'time_resolution', 'file_length']
        >>> if validate_metadata(metadata, required):
        >>>     print("Metadatos válidos")
    """
    if metadata is None:
        logger.error("Metadatos son None")
        return False
    
    if not isinstance(metadata, dict):
        logger.error(f"Metadatos no son diccionario: {type(metadata)}")
        return False
    
    # Claves requeridas por defecto
    if required_keys is None:
        required_keys = [
            'frequency_resolution',
            'time_resolution', 
            'file_length',
            'frequencies'
        ]
    
    # Verificar claves requeridas
    missing_keys = [key for key in required_keys if key not in metadata]
    if missing_keys:
        logger.error(f"Metadatos faltan claves requeridas: {missing_keys}")
        return False
    
    # Validar valores críticos
    if metadata.get('frequency_resolution', 0) <= 0:
        logger.error(f"frequency_resolution inválido: {metadata.get('frequency_resolution')}")
        return False
    
    if metadata.get('time_resolution', 0) <= 0:
        logger.error(f"time_resolution inválido: {metadata.get('time_resolution')}")
        return False
    
    if metadata.get('file_length', 0) <= 0:
        logger.error(f"file_length inválido: {metadata.get('file_length')}")
        return False
    
    # Validar frecuencias si están presentes
    frequencies = metadata.get('frequencies')
    if frequencies is not None:
        if not validate_frequency_range(frequencies):
            return False
    
    logger.info(f"Metadatos válidos: {list(metadata.keys())}")
    return True


def validate_frequency_range(frequencies: np.ndarray) -> bool:
    """
    Validar rango de frecuencias.
    
    Args:
        frequencies: Array con frecuencias en MHz
        
    Returns:
        bool: True si las frecuencias son válidas
        
    Ejemplo:
        >>> if validate_frequency_range(freq_array):
        >>>     print("Rango de frecuencias válido")
    """
    if frequencies is None:
        logger.error("Frecuencias son None")
        return False
    
    if not isinstance(frequencies, np.ndarray):
        logger.error(f"Frecuencias no son numpy array: {type(frequencies)}")
        return False
    
    if frequencies.size == 0:
        logger.error("Array de frecuencias vacío")
        return False
    
    if frequencies.ndim != 1:
        logger.error(f"Frecuencias deben ser 1D, got {frequencies.ndim}D")
        return False
    
    # Verificar que no hay valores NaN o Inf
    if np.any(np.isnan(frequencies)):
        logger.error("Frecuencias contienen valores NaN")
        return False
    
    if np.any(np.isinf(frequencies)):
        logger.error("Frecuencias contienen valores Inf")
        return False
    
    # Verificar rango razonable (MHz)
    freq_min, freq_max = frequencies.min(), frequencies.max()
    if freq_min < 100 or freq_max > 10000:
        logger.warning(f"Rango de frecuencias inusual: {freq_min:.1f} - {freq_max:.1f} MHz")
    
    # Verificar que están ordenadas (de mayor a menor frecuencia típico en radioastronomía)
    if not np.all(np.diff(frequencies) <= 0):
        logger.warning("Frecuencias no están ordenadas de mayor a menor")
    
    logger.info(f"Frecuencias válidas: {frequencies.size} canales, "
               f"rango: {freq_min:.1f} - {freq_max:.1f} MHz")
    return True


def validate_time_range(time_resolution: float, file_length: int) -> bool:
    """
    Validar rango temporal de los datos.
    
    Args:
        time_resolution: Resolución temporal en segundos
        file_length: Número de muestras temporales
        
    Returns:
        bool: True si el rango temporal es válido
        
    Ejemplo:
        >>> if validate_time_range(0.001, 1000000):
        >>>     print("Rango temporal válido")
    """
    if time_resolution <= 0:
        logger.error(f"time_resolution inválido: {time_resolution}")
        return False
    
    if file_length <= 0:
        logger.error(f"file_length inválido: {file_length}")
        return False
    
    # Calcular duración total
    total_duration = time_resolution * file_length
    
    # Verificar duración razonable (entre 1 segundo y 1 día)
    if total_duration < 1.0:
        logger.warning(f"Duración muy corta: {total_duration:.3f} segundos")
    elif total_duration > 86400:  # 24 horas
        logger.warning(f"Duración muy larga: {total_duration:.1f} segundos ({total_duration/3600:.1f} horas)")
    
    # Verificar resolución temporal razonable
    if time_resolution < 1e-6:  # 1 microsegundo
        logger.warning(f"Resolución temporal muy alta: {time_resolution:.2e} segundos")
    elif time_resolution > 1.0:  # 1 segundo
        logger.warning(f"Resolución temporal muy baja: {time_resolution:.3f} segundos")
    
    logger.info(f"Rango temporal válido: {file_length} muestras, "
               f"resolución: {time_resolution:.6f}s, "
               f"duración total: {total_duration:.1f}s")
    return True


def validate_file_for_processing(file_path: str | Path) -> Tuple[bool, Dict[str, Any]]:
    """
    Validar archivo completo para procesamiento (datos + metadatos).
    
    Args:
        file_path: Ruta al archivo
        
    Returns:
        Tuple[bool, Dict]: (es_válido, información_del_archivo)
        
    Ejemplo:
        >>> is_valid, info = validate_file_for_processing("observacion.fits")
        >>> if is_valid:
        >>>     print(f"Archivo válido: {info['file_type']}")
    """
    file_path = Path(file_path)
    
    validation_info = {
        'file_path': str(file_path),
        'file_exists': file_path.exists(),
        'file_size': file_path.stat().st_size if file_path.exists() else 0,
        'file_type': None,
        'validation_errors': []
    }
    
    # Verificar que el archivo existe
    if not validation_info['file_exists']:
        validation_info['validation_errors'].append("Archivo no existe")
        return False, validation_info
    
    # Verificar que no está vacío
    if validation_info['file_size'] == 0:
        validation_info['validation_errors'].append("Archivo está vacío")
        return False, validation_info
    
    # Determinar tipo de archivo
    if file_path.suffix.lower() in ['.fits', '.fit']:
        validation_info['file_type'] = 'FITS'
        from .fits_loader import validate_fits_file, get_fits_metadata
        
        # Validar archivo FITS
        if not validate_fits_file(file_path):
            validation_info['validation_errors'].append("Archivo FITS inválido")
            return False, validation_info
        
        # Obtener y validar metadatos
        try:
            metadata = get_fits_metadata(file_path)
            if not validate_metadata(metadata):
                validation_info['validation_errors'].append("Metadatos FITS inválidos")
                return False, validation_info
            validation_info['metadata'] = metadata
        except Exception as e:
            validation_info['validation_errors'].append(f"Error obteniendo metadatos FITS: {e}")
            return False, validation_info
            
    elif file_path.suffix.lower() == '.fil':
        validation_info['file_type'] = 'FIL'
        from .fil_loader import validate_fil_file, get_fil_metadata
        
        # Validar archivo FIL
        if not validate_fil_file(file_path):
            validation_info['validation_errors'].append("Archivo FIL inválido")
            return False, validation_info
        
        # Obtener y validar metadatos
        try:
            metadata = get_fil_metadata(file_path)
            if not validate_metadata(metadata):
                validation_info['validation_errors'].append("Metadatos FIL inválidos")
                return False, validation_info
            validation_info['metadata'] = metadata
        except Exception as e:
            validation_info['validation_errors'].append(f"Error obteniendo metadatos FIL: {e}")
            return False, validation_info
    else:
        validation_info['validation_errors'].append(f"Tipo de archivo no soportado: {file_path.suffix}")
        return False, validation_info
    
    logger.info(f"Archivo válido para procesamiento: {file_path} ({validation_info['file_type']})")
    return True, validation_info


def get_validation_summary(validation_results: List[Tuple[bool, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Generar resumen de validación para múltiples archivos.
    
    Args:
        validation_results: Lista de resultados de validación
        
    Returns:
        Dict con resumen de validación
        
    Ejemplo:
        >>> results = [validate_file_for_processing(f) for f in files]
        >>> summary = get_validation_summary(results)
        >>> print(f"Archivos válidos: {summary['valid_count']}/{summary['total_count']}")
    """
    total_count = len(validation_results)
    valid_count = sum(1 for is_valid, _ in validation_results if is_valid)
    
    file_types = {}
    total_size = 0
    errors = []
    
    for is_valid, info in validation_results:
        file_type = info.get('file_type', 'unknown')
        file_types[file_type] = file_types.get(file_type, 0) + 1
        total_size += info.get('file_size', 0)
        
        if not is_valid:
            errors.extend(info.get('validation_errors', []))
    
    summary = {
        'total_count': total_count,
        'valid_count': valid_count,
        'invalid_count': total_count - valid_count,
        'success_rate': valid_count / total_count if total_count > 0 else 0,
        'file_types': file_types,
        'total_size_bytes': total_size,
        'total_size_gb': total_size / (1024**3),
        'validation_errors': errors,
        'all_valid': valid_count == total_count
    }
    
    logger.info(f"Resumen de validación: {valid_count}/{total_count} archivos válidos "
               f"({summary['success_rate']:.1%})")
    
    return summary


# Exportar funciones
__all__ = [
    'validate_data',
    'validate_metadata',
    'validate_frequency_range',
    'validate_time_range',
    'validate_file_for_processing',
    'get_validation_summary',
] 