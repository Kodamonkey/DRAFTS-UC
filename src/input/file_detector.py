"""Detector y clasificador de tipos de archivos astronómicos.

Este módulo se encarga de:
1. Detectar automáticamente el tipo de archivo basado en extensión y contenido
2. Validar que el archivo sea compatible con el pipeline
3. Proporcionar información sobre el formato detectado
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Tipos de archivo soportados
SUPPORTED_EXTENSIONS = {'.fits', '.fil'}
SUPPORTED_FORMATS = {'fits', 'filterbank'}

def detect_file_type(file_path: Path) -> str:
    """
    Detecta automáticamente el tipo de archivo basado en su extensión.
    
    Args:
        file_path: Path al archivo a analizar
        
    Returns:
        str: Tipo de archivo detectado ('fits' o 'filterbank')
        
    Raises:
        ValueError: Si el tipo de archivo no es soportado
    """
    suffix = file_path.suffix.lower()
    
    if suffix == ".fits":
        return "fits"
    elif suffix == ".fil":
        return "filterbank"
    else:
        raise ValueError(
            f"Tipo de archivo no soportado: {file_path}\n"
            f"Extensión detectada: {suffix}\n"
            f"Extensiones soportadas: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

def validate_file_compatibility(file_path: Path) -> Dict[str, Any]:
    """
    Valida que el archivo sea compatible con el pipeline.
    
    Args:
        file_path: Path al archivo a validar
        
    Returns:
        Dict con información de validación:
        - is_compatible: bool
        - file_type: str
        - extension: str
        - size_bytes: int
        - validation_errors: List[str]
    """
    validation_result = {
        'is_compatible': False,
        'file_type': None,
        'extension': None,
        'size_bytes': 0,
        'validation_errors': []
    }
    
    try:
        # Verificar que el archivo existe
        if not file_path.exists():
            validation_result['validation_errors'].append(f"Archivo no encontrado: {file_path}")
            return validation_result
        
        # Verificar que es un archivo (no directorio)
        if not file_path.is_file():
            validation_result['validation_errors'].append(f"Path no es un archivo: {file_path}")
            return validation_result
        
        # Obtener extensión y tipo
        extension = file_path.suffix.lower()
        validation_result['extension'] = extension
        
        try:
            file_type = detect_file_type(file_path)
            validation_result['file_type'] = file_type
        except ValueError as e:
            validation_result['validation_errors'].append(str(e))
            return validation_result
        
        # Verificar tamaño del archivo
        try:
            size_bytes = file_path.stat().st_size
            validation_result['size_bytes'] = size_bytes
            
            if size_bytes == 0:
                validation_result['validation_errors'].append("Archivo vacío (0 bytes)")
                return validation_result
                
            # Advertencia para archivos muy grandes (> 10GB)
            if size_bytes > 10 * 1024**3:
                logger.warning(f"Archivo muy grande detectado: {size_bytes / (1024**3):.1f} GB")
                
        except OSError as e:
            validation_result['validation_errors'].append(f"Error accediendo al archivo: {e}")
            return validation_result
        
        # Si llegamos aquí, el archivo es compatible
        validation_result['is_compatible'] = True
        
    except Exception as e:
        validation_result['validation_errors'].append(f"Error inesperado durante validación: {e}")
    
    return validation_result

def get_file_info(file_path: Path) -> Dict[str, Any]:
    """
    Obtiene información completa del archivo.
    
    Args:
        file_path: Path al archivo
        
    Returns:
        Dict con información del archivo
    """
    file_info = {
        'path': str(file_path),
        'name': file_path.name,
        'stem': file_path.stem,
        'suffix': file_path.suffix,
        'parent': str(file_path.parent),
        'exists': file_path.exists(),
        'is_file': file_path.is_file() if file_path.exists() else False,
        'size_bytes': 0,
        'size_mb': 0.0,
        'size_gb': 0.0,
        'file_type': None,
        'is_compatible': False
    }
    
    if file_path.exists() and file_path.is_file():
        try:
            size_bytes = file_path.stat().st_size
            file_info.update({
                'size_bytes': size_bytes,
                'size_mb': size_bytes / (1024**2),
                'size_gb': size_bytes / (1024**3)
            })
            
            # Detectar tipo de archivo
            try:
                file_type = detect_file_type(file_path)
                file_info['file_type'] = file_type
                file_info['is_compatible'] = True
            except ValueError:
                pass
                
        except OSError:
            pass
    
    return file_info

def log_file_detection(file_path: Path) -> None:
    """
    Registra información de detección del archivo.
    
    Args:
        file_path: Path al archivo
    """
    try:
        file_info = get_file_info(file_path)
        validation = validate_file_compatibility(file_path)
        
        logger.info(f"Detectando archivo: {file_path.name}")
        logger.info(f"  - Tipo detectado: {file_info.get('file_type', 'N/A')}")
        logger.info(f"  - Extensión: {file_info.get('suffix', 'N/A')}")
        logger.info(f"  - Tamaño: {file_info.get('size_mb', 0):.1f} MB")
        logger.info(f"  - Compatible: {validation['is_compatible']}")
        
        if not validation['is_compatible']:
            logger.warning(f"  - Errores de validación: {validation['validation_errors']}")
            
    except Exception as e:
        logger.error(f"Error durante detección de archivo {file_path}: {e}")
