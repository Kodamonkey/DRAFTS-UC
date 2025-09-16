"""Buscador y filtrador de archivos astronómicos.

Este módulo se encarga de:
1. Buscar archivos FITS y filterbank en directorios
2. Filtrar archivos por criterios específicos (nombre, tipo, tamaño)
3. Validar que los archivos encontrados sean compatibles
4. Proporcionar información sobre los archivos encontrados
"""

from pathlib import Path
from typing import List, Optional
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
