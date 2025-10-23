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

def get_parameters_function(file_path: Path):
    """Return the helper that extracts parameters for the detected file type."""
    file_type = detect_file_type(file_path)
    
    if file_type == "fits":
        return get_obparams
    else:
        return get_obparams_fil

def extract_parameters_for_target(file_list: list[Path]) -> Dict[str, Any]:
    """Extract parameters from the first file in ``file_list`` for pipeline setup."""
    if not file_list:
        raise ValueError("File list is empty")


    first_file = file_list[0]
    logger.info(f"Extracting parameters from: {first_file.name}")

    try:
        result = extract_parameters_auto(first_file)
        logger.info("Observation parameters loaded successfully")
        return result

    except Exception as e:
        logger.error(f"Error obtaining parameters: {e}")
        raise

def validate_extracted_parameters() -> Dict[str, Any]:
    """Validate that the extracted parameters are consistent and usable."""
    from ..config import config
    
    validation_result = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'parameter_summary': {}
    }
    
    try:
                                       
        critical_params = {
            'TIME_RESO': config.TIME_RESO,
            'FREQ_RESO': config.FREQ_RESO,
            'FILE_LENG': config.FILE_LENG
        }
        
        for param_name, param_value in critical_params.items():
            if param_value is None:
                validation_result['errors'].append(f"Parameter {param_name} is None")
                validation_result['is_valid'] = False
            elif param_value <= 0:
                validation_result['errors'].append(f"Parameter {param_name} must be > 0, current: {param_value}")
                validation_result['is_valid'] = False
        
                                             
        if hasattr(config, 'FREQ') and config.FREQ is not None:
            if len(config.FREQ) != config.FREQ_RESO:
                validation_result['warnings'].append(
                    f"Length of FREQ array ({len(config.FREQ)}) does not match FREQ_RESO ({config.FREQ_RESO})"
                )
            
            if len(config.FREQ) > 1:
                freq_range = config.FREQ.max() - config.FREQ.min()
                if freq_range <= 0:
                    validation_result['warnings'].append("Frequency range is invalid or too small")
        
                                            
        down_freq = getattr(config, 'DOWN_FREQ_RATE', 1)
        down_time = getattr(config, 'DOWN_TIME_RATE', 1)
        
        if down_freq <= 0 or down_time <= 0:
            validation_result['errors'].append("Downsampling factors must be greater than zero")
            validation_result['is_valid'] = False
        
                                     
        validation_result['parameter_summary'] = {
            'time_resolution_sec': getattr(config, 'TIME_RESO', 'N/A'),
            'frequency_channels': getattr(config, 'FREQ_RESO', 'N/A'),
            'total_samples': getattr(config, 'FILE_LENG', 'N/A'),
            'frequency_range_mhz': f"{getattr(config, 'FREQ', [0]).min():.1f} - {getattr(config, 'FREQ', [0]).max():.1f}" if hasattr(config, 'FREQ') and config.FREQ is not None else 'N/A',
            'downsampling_freq': down_freq,
            'downsampling_time': down_time
        }
        
    except Exception as e:
        validation_result['errors'].append(f"Validation error: {e}")
        validation_result['is_valid'] = False
    
    return validation_result
