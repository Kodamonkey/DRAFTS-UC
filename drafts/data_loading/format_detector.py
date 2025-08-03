"""
Detector de Formatos - Detección automática de tipos de archivo
=============================================================

Este módulo proporciona funciones para detectar automáticamente el tipo
de archivo astronómico y su formato específico.

Funciones principales:
- detect_file_format: Detectar tipo de archivo automáticamente

Para astrónomos:
- Detecta automáticamente si es FITS, FIL, u otro formato
- Adapta el procesamiento según el formato detectado
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def detect_file_format(file_path: str | Path) -> Dict[str, Any]:
    """
    Detectar automáticamente el tipo y formato de archivo astronómico.
    
    Args:
        file_path: Ruta al archivo
        
    Returns:
        Dict con información del formato detectado:
        - file_type: Tipo principal (FITS, FIL, UNKNOWN)
        - format_subtype: Subtipo específico (PSRFITS, SIGPROC, etc.)
        - confidence: Nivel de confianza (0.0-1.0)
        - details: Detalles adicionales
        
    Ejemplo:
        >>> format_info = detect_file_format("observacion.fits")
        >>> print(f"Tipo detectado: {format_info['file_type']}")
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    logger.info(f"Detectando formato de archivo: {file_path}")
    
    format_info = {
        'file_path': str(file_path),
        'file_type': 'UNKNOWN',
        'format_subtype': 'UNKNOWN',
        'confidence': 0.0,
        'details': {}
    }
    
    # Detectar por extensión
    extension = file_path.suffix.lower()
    
    if extension in ['.fits', '.fit']:
        format_info = _detect_fits_format(file_path)
    elif extension == '.fil':
        format_info = _detect_fil_format(file_path)
    else:
        # Intentar detectar por contenido
        format_info = _detect_by_content(file_path)
    
    logger.info(f"Formato detectado: {format_info['file_type']} "
               f"({format_info['format_subtype']}) - confianza: {format_info['confidence']:.2f}")
    
    return format_info


def _detect_fits_format(file_path: Path) -> Dict[str, Any]:
    """
    Detectar formato específico de archivo FITS.
    
    Args:
        file_path: Ruta al archivo FITS
        
    Returns:
        Dict con información del formato FITS
    """
    format_info = {
        'file_path': str(file_path),
        'file_type': 'FITS',
        'format_subtype': 'UNKNOWN',
        'confidence': 0.8,  # Alta confianza por extensión
        'details': {}
    }
    
    try:
        from astropy.io import fits
        
        with fits.open(file_path, memmap=True) as hdul:
            # Verificar si es PSRFITS
            if "SUBINT" in [hdu.name for hdu in hdul]:
                format_info['format_subtype'] = 'PSRFITS'
                format_info['confidence'] = 0.95
                
                # Extraer detalles adicionales
                if "SUBINT" in hdul:
                    subint_hdr = hdul["SUBINT"].header
                    format_info['details'] = {
                        'nchan': subint_hdr.get('NCHAN', 'UNKNOWN'),
                        'npol': subint_hdr.get('NPOL', 'UNKNOWN'),
                        'nsblk': subint_hdr.get('NSBLK', 'UNKNOWN'),
                        'tbin': subint_hdr.get('TBIN', 'UNKNOWN'),
                    }
            else:
                # FITS estándar
                format_info['format_subtype'] = 'STANDARD_FITS'
                format_info['confidence'] = 0.9
                
                # Extraer detalles del primer HDU
                if len(hdul) > 0:
                    hdr = hdul[0].header
                    format_info['details'] = {
                        'naxis': hdr.get('NAXIS', 'UNKNOWN'),
                        'bitpix': hdr.get('BITPIX', 'UNKNOWN'),
                    }
                    
    except Exception as e:
        logger.warning(f"Error detectando formato FITS: {e}")
        format_info['confidence'] = 0.6  # Reducir confianza
    
    return format_info


def _detect_fil_format(file_path: Path) -> Dict[str, Any]:
    """
    Detectar formato específico de archivo FIL.
    
    Args:
        file_path: Ruta al archivo FIL
        
    Returns:
        Dict con información del formato FIL
    """
    format_info = {
        'file_path': str(file_path),
        'file_type': 'FIL',
        'format_subtype': 'UNKNOWN',
        'confidence': 0.8,  # Alta confianza por extensión
        'details': {}
    }
    
    try:
        from .header_parser import _read_string, _read_header
        
        with open(file_path, "rb") as f:
            # Intentar leer como SIGPROC estándar
            try:
                start = _read_string(f)
                if start == "HEADER_START":
                    format_info['format_subtype'] = 'SIGPROC_STANDARD'
                    format_info['confidence'] = 0.95
                    
                    # Leer header completo para detalles
                    f.seek(0)
                    header, _ = _read_header(f)
                    format_info['details'] = {
                        'nchans': header.get('nchans', 'UNKNOWN'),
                        'nbits': header.get('nbits', 'UNKNOWN'),
                        'tsamp': header.get('tsamp', 'UNKNOWN'),
                        'fch1': header.get('fch1', 'UNKNOWN'),
                        'foff': header.get('foff', 'UNKNOWN'),
                    }
                else:
                    format_info['format_subtype'] = 'NON_STANDARD_FIL'
                    format_info['confidence'] = 0.7
                    
            except Exception:
                format_info['format_subtype'] = 'NON_STANDARD_FIL'
                format_info['confidence'] = 0.7
                
    except Exception as e:
        logger.warning(f"Error detectando formato FIL: {e}")
        format_info['confidence'] = 0.6  # Reducir confianza
    
    return format_info


def _detect_by_content(file_path: Path) -> Dict[str, Any]:
    """
    Detectar formato por contenido del archivo.
    
    Args:
        file_path: Ruta al archivo
        
    Returns:
        Dict con información del formato detectado
    """
    format_info = {
        'file_path': str(file_path),
        'file_type': 'UNKNOWN',
        'format_subtype': 'UNKNOWN',
        'confidence': 0.0,
        'details': {}
    }
    
    try:
        with open(file_path, "rb") as f:
            # Leer primeros bytes para detectar magic numbers
            magic = f.read(8)
            
            # Detectar FITS por magic number
            if magic.startswith(b'SIMPLE') or magic.startswith(b'XTENSION'):
                format_info['file_type'] = 'FITS'
                format_info['format_subtype'] = 'STANDARD_FITS'
                format_info['confidence'] = 0.9
                
            # Detectar otros formatos si es necesario
            elif magic.startswith(b'HEADER_START'):
                format_info['file_type'] = 'FIL'
                format_info['format_subtype'] = 'SIGPROC_STANDARD'
                format_info['confidence'] = 0.9
                
            else:
                # Formato desconocido
                format_info['confidence'] = 0.1
                
    except Exception as e:
        logger.warning(f"Error detectando formato por contenido: {e}")
    
    return format_info


def get_format_handler(format_info: Dict[str, Any]) -> Optional[str]:
    """
    Obtener el nombre del módulo handler apropiado para el formato detectado.
    
    Args:
        format_info: Información del formato detectado
        
    Returns:
        str: Nombre del módulo handler o None si no se encuentra
        
    Ejemplo:
        >>> handler = get_format_handler(format_info)
        >>> if handler:
        >>>     print(f"Usar handler: {handler}")
    """
    file_type = format_info.get('file_type', 'UNKNOWN')
    format_subtype = format_info.get('format_subtype', 'UNKNOWN')
    
    if file_type == 'FITS':
        return 'fits_loader'
    elif file_type == 'FIL':
        return 'fil_loader'
    else:
        return None


def validate_format_compatibility(format_info: Dict[str, Any]) -> bool:
    """
    Validar si el formato detectado es compatible con el pipeline.
    
    Args:
        format_info: Información del formato detectado
        
    Returns:
        bool: True si el formato es compatible
        
    Ejemplo:
        >>> if validate_format_compatibility(format_info):
        >>>     print("Formato compatible")
    """
    file_type = format_info.get('file_type', 'UNKNOWN')
    confidence = format_info.get('confidence', 0.0)
    
    # Verificar que el tipo es conocido
    if file_type not in ['FITS', 'FIL']:
        logger.warning(f"Tipo de archivo no soportado: {file_type}")
        return False
    
    # Verificar nivel de confianza
    if confidence < 0.5:
        logger.warning(f"Confianza muy baja en detección: {confidence:.2f}")
        return False
    
    logger.info(f"Formato compatible: {file_type} (confianza: {confidence:.2f})")
    return True


def get_format_recommendations(format_info: Dict[str, Any]) -> list:
    """
    Obtener recomendaciones para el formato detectado.
    
    Args:
        format_info: Información del formato detectado
        
    Returns:
        List: Lista de recomendaciones
        
    Ejemplo:
        >>> recommendations = get_format_recommendations(format_info)
        >>> for rec in recommendations:
        >>>     print(f"Recomendación: {rec}")
    """
    recommendations = []
    
    file_type = format_info.get('file_type', 'UNKNOWN')
    format_subtype = format_info.get('format_subtype', 'UNKNOWN')
    confidence = format_info.get('confidence', 0.0)
    
    if file_type == 'FITS':
        if format_subtype == 'PSRFITS':
            recommendations.append("Usar procesamiento optimizado para PSRFITS")
        elif format_subtype == 'STANDARD_FITS':
            recommendations.append("Usar procesamiento estándar para FITS")
    elif file_type == 'FIL':
        if format_subtype == 'SIGPROC_STANDARD':
            recommendations.append("Usar procesamiento optimizado para SIGPROC")
        elif format_subtype == 'NON_STANDARD_FIL':
            recommendations.append("Verificar parámetros estimados del header")
    
    if confidence < 0.8:
        recommendations.append("Verificar manualmente el formato del archivo")
    
    return recommendations


# Exportar funciones
__all__ = [
    'detect_file_format',
    'get_format_handler',
    'validate_format_compatibility',
    'get_format_recommendations',
] 