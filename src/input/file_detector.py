# This module determines file types and validates compatibility.

"""Detector y clasificador de tipos de archivos astronómicos.

Este módulo se encarga de:
1. Detectar automáticamente el tipo de archivo basado en extensión y contenido
2. Validar que el archivo sea compatible con el pipeline
3. Proporcionar información sobre el formato detectado
"""

from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

                             
SUPPORTED_EXTENSIONS = {'.fits', '.fil'}
SUPPORTED_FORMATS = {'fits', 'filterbank'}

# This function detects file type.
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

# This function validates file compatibility.
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
                                         
        if not file_path.exists():
            validation_result['validation_errors'].append(f"Archivo no encontrado: {file_path}")
            return validation_result
        
                                                     
        if not file_path.is_file():
            validation_result['validation_errors'].append(f"Path no es un archivo: {file_path}")
            return validation_result
        
                                  
        extension = file_path.suffix.lower()
        validation_result['extension'] = extension
        
        try:
            file_type = detect_file_type(file_path)
            validation_result['file_type'] = file_type
        except ValueError as e:
            validation_result['validation_errors'].append(str(e))
            return validation_result
        
                                      
        try:
            size_bytes = file_path.stat().st_size
            validation_result['size_bytes'] = size_bytes
            
            if size_bytes == 0:
                validation_result['validation_errors'].append("Archivo vacío (0 bytes)")
                return validation_result
                
                                                            
            if size_bytes > 10 * 1024**3:
                logger.warning(f"Archivo muy grande detectado: {size_bytes / (1024**3):.1f} GB")
                
        except OSError as e:
            validation_result['validation_errors'].append(f"Error accediendo al archivo: {e}")
            return validation_result
        
                                                    
        validation_result['is_compatible'] = True
        
    except Exception as e:
        validation_result['validation_errors'].append(f"Error inesperado durante validación: {e}")
    
    return validation_result
