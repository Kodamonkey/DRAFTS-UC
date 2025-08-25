"""Buscador y filtrador de archivos astronómicos.

Este módulo se encarga de:
1. Buscar archivos FITS y filterbank en directorios
2. Filtrar archivos por criterios específicos (nombre, tipo, tamaño)
3. Validar que los archivos encontrados sean compatibles
4. Proporcionar información sobre los archivos encontrados
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from .file_detector import validate_file_compatibility
from ..config import config

logger = logging.getLogger(__name__)

def find_data_files(frb_target: str, data_dir: Optional[Path] = None) -> List[Path]:
    """
    Busca archivos FITS o filterbank que coincidan con el target FRB.
    
    Args:
        frb_target: Target FRB a buscar en los nombres de archivo
        data_dir: Directorio de datos (usa config.DATA_DIR si no se especifica)
        
    Returns:
        Lista ordenada de archivos compatibles encontrados
        
    Raises:
        ValueError: Si el directorio no existe o no hay archivos compatibles
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
    
    if not data_dir.exists():
        raise ValueError(f"Directorio de datos no existe: {data_dir}")
    
    if not data_dir.is_dir():
        raise ValueError(f"Path no es un directorio: {data_dir}")
    
    logger.info(f"Buscando archivos para target '{frb_target}' en: {data_dir}")
    
    # Buscar archivos por extensión
    fits_files = list(data_dir.glob("*.fits"))
    fil_files = list(data_dir.glob("*.fil"))
    
    all_files = fits_files + fil_files
    
    if not all_files:
        logger.warning(f"No se encontraron archivos .fits o .fil en: {data_dir}")
        return []
    
    logger.info(f"Archivos encontrados: {len(fits_files)} .fits, {len(fil_files)} .fil")
    
    # Filtrar por target FRB
    matching_files = [f for f in all_files if frb_target.lower() in f.name.lower()]
    
    if not matching_files:
        logger.warning(f"No se encontraron archivos que coincidan con target '{frb_target}'")
        return []
    
    # Ordenar por nombre
    matching_files.sort(key=lambda x: x.name)
    
    logger.info(f"Archivos coincidentes encontrados: {len(matching_files)}")
    for file in matching_files:
        logger.debug(f"  - {file.name}")
    
    return matching_files

def find_files_by_pattern(pattern: str, data_dir: Optional[Path] = None, 
                         file_types: Optional[List[str]] = None) -> List[Path]:
    """
    Busca archivos por patrón de nombre y tipos específicos.
    
    Args:
        pattern: Patrón de búsqueda (ej: "FRB*", "*2020*")
        data_dir: Directorio de datos
        file_types: Tipos de archivo a buscar ['.fits', '.fil']
        
    Returns:
        Lista de archivos encontrados
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
    
    if file_types is None:
        file_types = ['.fits', '.fil']
    
    matching_files = []
    
    for file_type in file_types:
        if not file_type.startswith('.'):
            file_type = '.' + file_type
        
        files = list(data_dir.glob(f"*{file_type}"))
        matching_files.extend([f for f in files if pattern.lower() in f.name.lower()])
    
    return sorted(matching_files, key=lambda x: x.name)

def get_file_summary(file_list: List[Path]) -> Dict[str, Any]:
    """
    Genera un resumen de la lista de archivos encontrados.
    
    Args:
        file_list: Lista de archivos a analizar
        
    Returns:
        Dict con resumen de archivos
    """
    if not file_list:
        return {
            'total_files': 0,
            'file_types': {},
            'total_size_gb': 0.0,
            'compatibility_summary': {}
        }
    
    summary = {
        'total_files': len(file_list),
        'file_types': {},
        'total_size_bytes': 0,
        'compatibility_summary': {
            'compatible': 0,
            'incompatible': 0,
            'errors': []
        }
    }
    
    for file_path in file_list:
        # Contar por tipo
        suffix = file_path.suffix.lower()
        summary['file_types'][suffix] = summary['file_types'].get(suffix, 0) + 1
        
        # Sumar tamaños
        try:
            file_size = file_path.stat().st_size
            summary['total_size_bytes'] += file_size
        except OSError:
            pass
        
        # Verificar compatibilidad
        try:
            validation = validate_file_compatibility(file_path)
            if validation['is_compatible']:
                summary['compatibility_summary']['compatible'] += 1
            else:
                summary['compatibility_summary']['incompatible'] += 1
                summary['compatibility_summary']['errors'].extend(validation['validation_errors'])
        except Exception as e:
            summary['compatibility_summary']['incompatible'] += 1
            summary['compatibility_summary']['errors'].append(str(e))
    
    # Convertir bytes a GB
    summary['total_size_gb'] = summary['total_size_bytes'] / (1024**3)
    
    return summary

def validate_file_list(file_list: List[Path]) -> Dict[str, Any]:
    """
    Valida una lista de archivos y retorna información de compatibilidad.
    
    Args:
        file_list: Lista de archivos a validar
        
    Returns:
        Dict con resultados de validación
    """
    validation_results = {
        'total_files': len(file_list),
        'valid_files': [],
        'invalid_files': [],
        'errors': [],
        'summary': {}
    }
    
    for file_path in file_list:
        try:
            validation = validate_file_compatibility(file_path)
            if validation['is_compatible']:
                validation_results['valid_files'].append({
                    'path': str(file_path),
                    'name': file_path.name,
                    'size_bytes': validation['size_bytes'],
                    'file_type': validation['file_type']
                })
            else:
                validation_results['invalid_files'].append({
                    'path': str(file_path),
                    'name': file_path.name,
                    'errors': validation['validation_errors']
                })
        except Exception as e:
            validation_results['errors'].append({
                'path': str(file_path),
                'name': file_path.name,
                'error': str(e)
            })
    
    # Generar resumen
    validation_results['summary'] = {
        'valid_count': len(validation_results['valid_files']),
        'invalid_count': len(validation_results['invalid_files']),
        'error_count': len(validation_results['errors']),
        'success_rate': len(validation_results['valid_files']) / len(file_list) if file_list else 0
    }
    
    return validation_results
