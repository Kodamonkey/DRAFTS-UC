# This module locates input data files for processing.

"""Locate and filter FITS or filterbank files relevant to a target."""

from pathlib import Path
from typing import List, Optional
import logging

from .file_detector import validate_file_compatibility
from ..config import config

logger = logging.getLogger(__name__)

def find_data_files(frb_target: str, data_dir: Optional[Path] = None) -> List[Path]:
    """Return FITS or filterbank files whose names contain ``frb_target``."""
    if data_dir is None:
        data_dir = config.DATA_DIR
    
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    if not data_dir.is_dir():
        raise ValueError(f"Path is not a directory: {data_dir}")

    logger.info(f"Searching files for target '{frb_target}' in: {data_dir}")
    
                                   
    fits_files = list(data_dir.glob("*.fits"))
    fil_files = list(data_dir.glob("*.fil"))
    
    all_files = fits_files + fil_files
    
    if not all_files:
        logger.warning(f"No .fits or .fil files found in: {data_dir}")
        return []

    logger.info(f"Files found: {len(fits_files)} .fits, {len(fil_files)} .fil")
    
                            
    matching_files = [f for f in all_files if frb_target.lower() in f.name.lower()]
    
    if not matching_files:
        logger.warning(f"No files match target '{frb_target}'")
        return []
    
                        
    matching_files.sort(key=lambda x: x.name)
    
    logger.info(f"Matching files found: {len(matching_files)}")
    for file in matching_files:
        logger.debug(f"  - {file.name}")
    
    return matching_files
