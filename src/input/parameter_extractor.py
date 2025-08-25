"""Extractor de parámetros de observación para archivos astronómicos.

Este módulo se encarga de:
1. Extraer parámetros de observación de archivos FITS y filterbank
2. Configurar variables globales en config
3. Aplicar configuraciones automáticas de decimación
"""

from pathlib import Path
from typing import Dict, Any
import logging

from .file_detector import detect_file_type, validate_file_compatibility
from .fits_handler import get_obparams
from .filterbank_handler import get_obparams_fil
from .utils import auto_config_downsampling

logger = logging.getLogger(__name__)

def extract_parameters_auto(file_path: Path) -> Dict[str, Any]:
    """
    Extrae parámetros automáticamente según el tipo de archivo detectado.
    
    Args:
        file_path: Path al archivo del cual extraer parámetros
        
    Returns:
        Dict con información de la extracción:
        - success: bool
        - file_type: str
        - parameters_extracted: List[str]
        - errors: List[str]
        
    Raises:
        ValueError: Si el archivo no es compatible o hay errores durante la extracción
    """
    extraction_result = {
        'success': False,
        'file_type': None,
        'parameters_extracted': [],
        'errors': [],
        'file_info': {}
    }
    
    try:
        # Validar compatibilidad del archivo
        validation = validate_file_compatibility(file_path)
        if not validation['is_compatible']:
            extraction_result['errors'].extend(validation['validation_errors'])
            raise ValueError(f"Archivo no compatible: {', '.join(validation['validation_errors'])}")
        
        # Detectar tipo de archivo
        file_type = detect_file_type(file_path)
        extraction_result['file_type'] = file_type
        
        logger.info(f"Extrayendo parámetros de archivo {file_type.upper()}: {file_path.name}")
        
        # Extraer parámetros según el tipo
        if file_type == "fits":
            get_obparams(str(file_path))
            extraction_result['parameters_extracted'] = [
                'TIME_RESO', 'FREQ_RESO', 'FILE_LENG', 'FREQ',
                'NBITS', 'NPOL', 'POL_TYPE', 'TSTART_MJD', 'NSUBOFFS'
            ]
        elif file_type == "filterbank":
            get_obparams_fil(str(file_path))
            extraction_result['parameters_extracted'] = [
                'TIME_RESO', 'FREQ_RESO', 'FILE_LENG', 'FREQ'
            ]
        
        # Aplicar configuraciones automáticas de decimación
        logger.info("Aplicando configuraciones automáticas de decimación...")
        auto_config_downsampling()
        
        # Verificar que los parámetros críticos se extrajeron correctamente
        from ..config import config
        
        critical_params = ['TIME_RESO', 'FREQ_RESO', 'FILE_LENG']
        missing_params = []
        
        for param in critical_params:
            if not hasattr(config, param) or getattr(config, param) is None:
                missing_params.append(param)
        
        if missing_params:
            raise ValueError(f"Parámetros críticos faltantes: {', '.join(missing_params)}")
        
        extraction_result['success'] = True
        logger.info(f"Parámetros extraídos exitosamente de {file_path.name}")
        
        # Log de parámetros extraídos
        logger.info(f"Parámetros extraídos:")
        logger.info(f"  - Resolución temporal: {config.TIME_RESO:.2e} s")
        logger.info(f"  - Canales de frecuencia: {config.FREQ_RESO}")
        logger.info(f"  - Muestras totales: {config.FILE_LENG:,}")
        logger.info(f"  - Rango de frecuencias: {config.FREQ.min():.1f} - {config.FREQ.max():.1f} MHz")
        logger.info(f"  - Decimación frecuencia: {getattr(config, 'DOWN_FREQ_RATE', 'N/A')}x")
        logger.info(f"  - Decimación tiempo: {getattr(config, 'DOWN_TIME_RATE', 'N/A')}x")
        
    except Exception as e:
        extraction_result['errors'].append(str(e))
        logger.error(f"Error extrayendo parámetros de {file_path}: {e}")
        raise
    
    return extraction_result

def get_parameters_function(file_path: Path):
    """
    Retorna la función apropiada para extraer parámetros según el tipo de archivo.
    
    Args:
        file_path: Path al archivo
        
    Returns:
        Función apropiada para extraer parámetros
    """
    file_type = detect_file_type(file_path)
    
    if file_type == "fits":
        return get_obparams
    else:
        return get_obparams_fil

def extract_parameters_for_target(file_list: list[Path]) -> Dict[str, Any]:
    """
    Extrae parámetros del primer archivo de una lista para configurar el pipeline.
    
    Args:
        file_list: Lista de archivos del mismo target FRB
        
    Returns:
        Dict con información de la extracción
        
    Raises:
        ValueError: Si no hay archivos o hay errores durante la extracción
    """
    if not file_list:
        raise ValueError("Lista de archivos vacía")
    
    # Usar el primer archivo para extraer parámetros
    first_file = file_list[0]
    logger.info(f"Extrayendo parámetros desde: {first_file.name}")
    
    try:
        result = extract_parameters_auto(first_file)
        logger.info("Parámetros de observación cargados exitosamente")
        return result
        
    except Exception as e:
        logger.error(f"Error obteniendo parámetros: {e}")
        raise

def validate_extracted_parameters() -> Dict[str, Any]:
    """
    Valida que los parámetros extraídos sean coherentes y válidos.
    
    Returns:
        Dict con resultado de la validación
    """
    from ..config import config
    
    validation_result = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'parameter_summary': {}
    }
    
    try:
        # Verificar parámetros críticos
        critical_params = {
            'TIME_RESO': config.TIME_RESO,
            'FREQ_RESO': config.FREQ_RESO,
            'FILE_LENG': config.FILE_LENG
        }
        
        for param_name, param_value in critical_params.items():
            if param_value is None:
                validation_result['errors'].append(f"Parámetro {param_name} es None")
                validation_result['is_valid'] = False
            elif param_value <= 0:
                validation_result['errors'].append(f"Parámetro {param_name} debe ser > 0, actual: {param_value}")
                validation_result['is_valid'] = False
        
        # Verificar coherencia de frecuencias
        if hasattr(config, 'FREQ') and config.FREQ is not None:
            if len(config.FREQ) != config.FREQ_RESO:
                validation_result['warnings'].append(
                    f"Longitud de array FREQ ({len(config.FREQ)}) no coincide con FREQ_RESO ({config.FREQ_RESO})"
                )
            
            if len(config.FREQ) > 1:
                freq_range = config.FREQ.max() - config.FREQ.min()
                if freq_range <= 0:
                    validation_result['warnings'].append("Rango de frecuencias inválido o muy pequeño")
        
        # Verificar parámetros de decimación
        down_freq = getattr(config, 'DOWN_FREQ_RATE', 1)
        down_time = getattr(config, 'DOWN_TIME_RATE', 1)
        
        if down_freq <= 0 or down_time <= 0:
            validation_result['errors'].append("Factores de decimación deben ser > 0")
            validation_result['is_valid'] = False
        
        # Crear resumen de parámetros
        validation_result['parameter_summary'] = {
            'time_resolution_sec': getattr(config, 'TIME_RESO', 'N/A'),
            'frequency_channels': getattr(config, 'FREQ_RESO', 'N/A'),
            'total_samples': getattr(config, 'FILE_LENG', 'N/A'),
            'frequency_range_mhz': f"{getattr(config, 'FREQ', [0]).min():.1f} - {getattr(config, 'FREQ', [0]).max():.1f}" if hasattr(config, 'FREQ') and config.FREQ is not None else 'N/A',
            'downsampling_freq': down_freq,
            'downsampling_time': down_time
        }
        
    except Exception as e:
        validation_result['errors'].append(f"Error durante validación: {e}")
        validation_result['is_valid'] = False
    
    return validation_result
