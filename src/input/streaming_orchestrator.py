# This module selects streaming strategies for input files.

"""Select streaming helpers for FITS or filterbank files."""

from pathlib import Path
from typing import Tuple, Callable, Dict, Any
import logging

from .file_detector import detect_file_type, validate_file_compatibility
from .fits_handler import stream_fits
from .filterbank_handler import stream_fil

logger = logging.getLogger(__name__)

def get_streaming_function(file_path: Path) -> Tuple[Callable, str]:
    """Return the streaming function and detected file type."""

    validation = validate_file_compatibility(file_path)
    if not validation['is_compatible']:
        raise ValueError(f"Incompatible file: {', '.join(validation['validation_errors'])}")


    file_type = detect_file_type(file_path)

    if file_type == "fits":
        return stream_fits, "fits"
    if file_type == "filterbank":
        return stream_fil, "filterbank"

    raise ValueError(f"Unsupported file type: {file_type or 'unknown'}")

def get_streaming_info(file_path: Path, chunk_samples: int, overlap_samples: int = 0) -> Dict[str, Any]:
    """Return metadata about the streaming configuration for ``file_path``."""
    try:

        if chunk_samples <= 0:
            return {
                'is_valid': False,
                'errors': ["chunk_samples must be greater than zero"],
                'streaming_config': None
            }

        if overlap_samples < 0:
            return {
                'is_valid': False,
                'errors': ["overlap_samples must be non-negative"],
                'streaming_config': None
            }

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

def validate_streaming_parameters(
    file_path: Path,
    chunk_samples: int,
    overlap_samples: int = 0
) -> Dict[str, Any]:
    """Validate whether the requested streaming parameters are appropriate."""
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
            validation_result['errors'].append("chunk_samples must be > 0")
            validation_result['is_valid'] = False
        
        if overlap_samples < 0:
            validation_result['errors'].append("overlap_samples must be >= 0")
            validation_result['is_valid'] = False

        if overlap_samples >= chunk_samples:
            validation_result['warnings'].append("overlap_samples is >= chunk_samples; this may cause issues")
        
                                                                       
        from ..config import config
        
        if hasattr(config, 'FILE_LENG') and config.FILE_LENG is not None:
            total_samples = config.FILE_LENG
            
                                                                            
            if chunk_samples > total_samples:
                validation_result['warnings'].append(
                    f"chunk_samples ({chunk_samples:,}) is larger than the entire file ({total_samples:,})"
                )
                validation_result['recommendations'].append(
                    f"Consider using chunk_samples = {total_samples:,} for small files"
                )


            if overlap_samples > total_samples // 2:
                validation_result['warnings'].append(
                    f"overlap_samples ({overlap_samples:,}) is too large compared to the file ({total_samples:,})"
                )
        
                                                      
        file_type = detect_file_type(file_path)
        
        if file_type == "fits":
                                                
            if chunk_samples > 10_000_000:
                validation_result['warnings'].append(
                    "chunk_samples is very large for FITS files and may cause memory issues"
                )
        
        elif file_type == "filterbank":
                                                      
            if chunk_samples > 50_000_000:
                validation_result['warnings'].append(
                    "chunk_samples is very large for filterbank files and may cause memory issues"
                )


        if chunk_samples < 1_000_000:
            validation_result['recommendations'].append(
                "Small chunk_samples may lead to many chunks and processing overhead"
            )

        if chunk_samples > 100_000_000:
            validation_result['recommendations'].append(
                "Large chunk_samples may cause memory issues; consider values between 1M-50M"
            )

    except Exception as e:
        validation_result['errors'].append(f"Validation error: {e}")
        validation_result['is_valid'] = False
    
    return validation_result
