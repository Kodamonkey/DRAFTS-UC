# This module selects streaming strategies for input files.

"""Orquestador de streaming para archivos astronómicos.

Este módulo se encarga de:
1. Seleccionar la función de streaming apropiada según el tipo de archivo
2. Proporcionar interfaz unificada para streaming de FITS y filterbank
3. Gestionar metadatos y configuración de streaming
"""

from pathlib import Path
from typing import Tuple, Callable, Dict, Any
import logging

from .file_detector import detect_file_type, validate_file_compatibility
from .fits_handler import stream_fits
from .filterbank_handler import stream_fil

logger = logging.getLogger(__name__)

# This function gets streaming function.
def get_streaming_function(file_path: Path) -> Tuple[Callable, str]:
    """
    Retorna la función de streaming apropiada y el tipo de archivo.
    
    Args:
        file_path: Path al archivo
        
    Returns:
        Tuple[streaming_function, file_type]: Función de streaming y tipo de archivo
        
    Raises:
        ValueError: Si el archivo no es compatible
    """
                                        
    validation = validate_file_compatibility(file_path)
    if not validation['is_compatible']:
        raise ValueError(f"Archivo no compatible: {', '.join(validation['validation_errors'])}")
    
                              
    file_type = detect_file_type(file_path)
    
                                                
    if file_type == "fits":
        return stream_fits, "fits"
    else:
        return stream_fil, "filterbank"

# This function gets streaming info.
def get_streaming_info(file_path: Path, chunk_samples: int, overlap_samples: int = 0) -> Dict[str, Any]:
    """
    Obtiene información sobre la configuración de streaming para un archivo.
    
    Args:
        file_path: Path al archivo
        chunk_samples: Tamaño de chunk configurado
        overlap_samples: Solapamiento configurado
        
    Returns:
        Dict con información de streaming
    """
    try:
                         
        validation = validate_file_compatibility(file_path)
        if not validation['is_compatible']:
            return {
                'is_valid': False,
                'errors': validation['validation_errors'],
                'streaming_config': None
            }
        
                                 
        file_type = detect_file_type(file_path)
        
                                         
        from ..config import config
        
        streaming_config = {
            'file_type': file_type,
            'file_path': str(file_path),
            'file_name': file_path.name,
            'chunk_samples': chunk_samples,
            'overlap_samples': overlap_samples,
            'total_samples': getattr(config, 'FILE_LENG', 'N/A'),
            'time_resolution': getattr(config, 'TIME_RESO', 'N/A'),
            'frequency_channels': getattr(config, 'FREQ_RESO', 'N/A'),
            'estimated_chunks': 'N/A',
            'chunk_duration_sec': 'N/A',
            'overlap_duration_sec': 'N/A'
        }
        
                                                           
        if hasattr(config, 'FILE_LENG') and config.FILE_LENG is not None:
            total_samples = config.FILE_LENG
            streaming_config['total_samples'] = total_samples
            
                                                
            if chunk_samples > 0:
                estimated_chunks = (total_samples + chunk_samples - 1) // chunk_samples
                streaming_config['estimated_chunks'] = estimated_chunks
                
                                             
                if hasattr(config, 'TIME_RESO') and config.TIME_RESO is not None:
                    chunk_duration = chunk_samples * config.TIME_RESO
                    streaming_config['chunk_duration_sec'] = chunk_duration
                    
                                                        
                    overlap_duration = overlap_samples * config.TIME_RESO
                    streaming_config['overlap_duration_sec'] = overlap_duration
        
        return {
            'is_valid': True,
            'errors': [],
            'streaming_config': streaming_config
        }
        
    except Exception as e:
        return {
            'is_valid': False,
            'errors': [str(e)],
            'streaming_config': None
        }

# This function validates streaming parameters.
def validate_streaming_parameters(
    file_path: Path, 
    chunk_samples: int, 
    overlap_samples: int = 0
) -> Dict[str, Any]:
    """
    Valida que los parámetros de streaming sean apropiados para el archivo.
    
    Args:
        file_path: Path al archivo
        chunk_samples: Tamaño de chunk propuesto
        overlap_samples: Solapamiento propuesto
        
    Returns:
        Dict con resultado de la validación
    """
    validation_result = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'recommendations': []
    }
    
    try:
                         
        file_validation = validate_file_compatibility(file_path)
        if not file_validation['is_compatible']:
            validation_result['errors'].extend(file_validation['validation_errors'])
            validation_result['is_valid'] = False
            return validation_result
        
                                        
        if chunk_samples <= 0:
            validation_result['errors'].append("chunk_samples debe ser > 0")
            validation_result['is_valid'] = False
        
        if overlap_samples < 0:
            validation_result['errors'].append("overlap_samples debe ser >= 0")
            validation_result['is_valid'] = False
        
        if overlap_samples >= chunk_samples:
            validation_result['warnings'].append("overlap_samples es >= chunk_samples, esto puede causar problemas")
        
                                                                       
        from ..config import config
        
        if hasattr(config, 'FILE_LENG') and config.FILE_LENG is not None:
            total_samples = config.FILE_LENG
            
                                                                            
            if chunk_samples > total_samples:
                validation_result['warnings'].append(
                    f"chunk_samples ({chunk_samples:,}) es mayor que el archivo completo ({total_samples:,})"
                )
                validation_result['recommendations'].append(
                    f"Considerar usar chunk_samples = {total_samples:,} para archivos pequeños"
                )
            
                                                         
            if overlap_samples > total_samples // 2:
                validation_result['warnings'].append(
                    f"overlap_samples ({overlap_samples:,}) es muy grande comparado con el archivo ({total_samples:,})"
                )
        
                                                      
        file_type = detect_file_type(file_path)
        
        if file_type == "fits":
                                                
            if chunk_samples > 10_000_000:                
                validation_result['warnings'].append(
                    "chunk_samples muy grande para archivos FITS, puede causar problemas de memoria"
                )
        
        elif file_type == "filterbank":
                                                      
            if chunk_samples > 50_000_000:                
                validation_result['warnings'].append(
                    "chunk_samples muy grande para archivos filterbank, puede causar problemas de memoria"
                )
        
                                   
        if chunk_samples < 1_000_000:                 
            validation_result['recommendations'].append(
                "chunk_samples pequeño puede resultar en muchos chunks y overhead de procesamiento"
            )
        
        if chunk_samples > 100_000_000:                   
            validation_result['recommendations'].append(
                "chunk_samples muy grande puede causar problemas de memoria, considerar valores entre 1M-50M"
            )
        
    except Exception as e:
        validation_result['errors'].append(f"Error durante validación: {e}")
        validation_result['is_valid'] = False
    
    return validation_result
