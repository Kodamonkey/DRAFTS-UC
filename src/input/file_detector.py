# This module determines file types and validates compatibility.

"""Detect and validate astronomical file formats used by the pipeline."""

from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

                             
SUPPORTED_EXTENSIONS = {'.fits', '.fil'}
SUPPORTED_FORMATS = {'fits', 'filterbank'}

def detect_file_type(file_path: Path) -> str:
    """Detect the file type from its suffix.

    Raises ``ValueError`` when the extension is unsupported.
    """
    suffix = file_path.suffix.lower()
    
    if suffix == ".fits":
        return "fits"
    elif suffix == ".fil":
        return "filterbank"
    else:
        raise ValueError(
            f"Unsupported file type: {file_path}\n"
            f"Detected extension: {suffix}\n"
            f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

def validate_file_compatibility(file_path: Path) -> Dict[str, Any]:
    """Validate whether a file can be processed by the pipeline."""
    validation_result = {
        'is_compatible': False,
        'file_type': None,
        'extension': None,
        'size_bytes': 0,
        'validation_errors': []
    }
    
    try:
                                         
        if not file_path.exists():
            validation_result['validation_errors'].append(f"File not found: {file_path}")
            return validation_result
        
                                                     
        if not file_path.is_file():
            validation_result['validation_errors'].append(f"Path is not a file: {file_path}")
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
                validation_result['validation_errors'].append("Empty file (0 bytes)")
                return validation_result
                
                                                            
            if size_bytes > 10 * 1024**3:
                logger.warning(f"Very large file detected: {size_bytes / (1024**3):.1f} GB")
                
        except OSError as e:
            validation_result['validation_errors'].append(f"Error accessing file: {e}")
            return validation_result
        
        # Verify magic bytes / file integrity
        integrity_error = _check_magic_bytes(file_path, file_type)
        if integrity_error:
            validation_result['validation_errors'].append(integrity_error)
            return validation_result

        validation_result['is_compatible'] = True

    except Exception as e:
        validation_result['validation_errors'].append(f"Unexpected validation error: {e}")

    return validation_result


def _check_magic_bytes(file_path: Path, file_type: str) -> str | None:
    """Verify that the file starts with the expected magic bytes.

    Returns an error string if invalid, ``None`` if OK.
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(80)

        if file_type == "fits":
            # FITS files must start with "SIMPLE  ="
            if not header.startswith(b"SIMPLE"):
                return (
                    f"File does not look like a valid FITS file (expected "
                    f"'SIMPLE' header, got {header[:20]!r})"
                )
        elif file_type == "filterbank":
            # Filterbank files start with the keyword "HEADER_START"
            if b"HEADER_START" not in header[:32]:
                return (
                    f"File does not look like a valid filterbank file "
                    f"(expected 'HEADER_START' in first 32 bytes)"
                )
    except OSError as e:
        return f"Cannot read file header: {e}"

    return None
