# This module extracts observational parameters from input files.

"""Utilities to extract observation parameters from FITS and filterbank files."""

from pathlib import Path
from typing import Dict, Any
import logging

from .file_detector import detect_file_type, validate_file_compatibility
from .fits_handler import get_obparams
from .filterbank_handler import get_obparams_fil
from .utils import auto_config_downsampling

logger = logging.getLogger(__name__)

def extract_parameters_auto(file_path: Path) -> Dict[str, Any]:
    """Extract observation parameters using the appropriate handler for the file type."""
    extraction_result = {
        'success': False,
        'file_type': None,
        'parameters_extracted': [],
        'errors': [],
        'file_info': {}
    }
    
    try:
                                            
        validation = validate_file_compatibility(file_path)
        if not validation['is_compatible']:
            extraction_result['errors'].extend(validation['validation_errors'])
            raise ValueError(f"Incompatible file: {', '.join(validation['validation_errors'])}")
        
                                  
        file_type = detect_file_type(file_path)
        extraction_result['file_type'] = file_type
        
        logger.info(f"Extracting parameters from {file_type.upper()} file: {file_path.name}")
        
                                          
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
        
                                                           
        logger.info("Applying automatic downsampling configuration...")
        auto_config_downsampling()
        
                                                                           
        from ..config import config
        
        critical_params = ['TIME_RESO', 'FREQ_RESO', 'FILE_LENG']
        missing_params = []
        
        for param in critical_params:
            if not hasattr(config, param) or getattr(config, param) is None:
                missing_params.append(param)
        
        if missing_params:
            raise ValueError(f"Missing critical parameters: {', '.join(missing_params)}")

        extraction_result['success'] = True
        logger.info(f"Successfully extracted parameters from {file_path.name}")

        logger.info("Extracted parameters:")
        logger.info(f"  - Time resolution: {config.TIME_RESO:.2e} s")
        logger.info(f"  - Frequency channels: {config.FREQ_RESO}")
        logger.info(f"  - Total samples: {config.FILE_LENG:,}")
        logger.info(f"  - Frequency range: {config.FREQ.min():.1f} - {config.FREQ.max():.1f} MHz")
        logger.info(f"  - Frequency downsampling: {getattr(config, 'DOWN_FREQ_RATE', 'N/A')}x")
        logger.info(f"  - Time downsampling: {getattr(config, 'DOWN_TIME_RATE', 'N/A')}x")

    except Exception as e:
        extraction_result['errors'].append(str(e))
        logger.error(f"Error extracting parameters from {file_path}: {e}")
        raise
    
    return extraction_result



