# This module analyzes FITS headers and structures.

                      
"""
FITS Header Analyzer for DRAFTS Pipeline
=====================================

Robust script to analyze and extract detailed information from headers
of FITS/PSRFITS files used in the FRB detection pipeline.

Features:
- Complete analysis of primary headers and extensions
- Detailed information about astronomical metadata
- Data statistics and file structure
- Integrity and compatibility validation
- Integrated logging system
- Robust error handling

Usage:
    python fits_header_analyzer.py [file1.fits] [file2.fits] ...
    python fits_header_analyzer.py --all  # Analyze all files in DATA_DIR
    python fits_header_analyzer.py --targets  # Analyze files in FRB_TARGETS

Author: DRAFTS System
Version: 1.0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
import sys

                     
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u

                  
try:
    import fitsio
    HAS_FITSIO = True
except ImportError:
    HAS_FITSIO = False
    fitsio = None

               
import sys
import logging
from pathlib import Path

                                                        
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

                                 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

                                             
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'

                        
try:
    from config import config
    logger.info("Project configuration loaded successfully")
except ImportError as e:
    logger.warning(f"Could not load project configuration: {e}")
                                      
    class DefaultConfig:
        DATA_DIR = Path("./Data/raw")
    config = DefaultConfig()
    logger.info("Using default configuration")

                                                                               
                              
                                                                               

                                               
ASTRONOMICAL_HEADERS = [
    'OBJECT', 'OBSERVER', 'TELESCOP', 'INSTRUME', 'OBS_MODE',
    'RA', 'DEC', 'GLON', 'GLAT', 'EQUINOX', 'RADECSYS',
    'DATE-OBS', 'TIME-OBS', 'MJD', 'LST', 'AZIMUTH', 'ELEVATIO',
    'FREQ', 'BW', 'CHAN_BW', 'NCHAN', 'CHAN_DM', 'DM',
    'TSAMP', 'NSBLK', 'NBITS', 'NPOL', 'POL_TYPE',
    'OBSFREQ', 'OBSBW', 'OBSNCHAN', 'OBS_MODE'
]

                                     
CALIBRATION_HEADERS = [
    'DAT_SCL', 'DAT_OFFS', 'DAT_WTS', 'CALIBRATED',
    'ZERO_OFF', 'SCALE', 'OFFSET', 'WEIGHTS'
]

                                     
PROCESSING_HEADERS = [
    'HISTORY', 'COMMENT', 'ORIGIN', 'CREATOR', 'PROC_HIST',
    'RFI_MASK', 'BAD_CHAN', 'MASKED', 'CLEANED'
]

                                                                               
                                  
                                                                               

def analyze_primary_header(header: fits.Header) -> Dict[str, Any]:
    """
    Analyzes the primary header of the FITS file.

    Args:
        header: Primary header from astropy.io.fits

    Returns:
        Dict with detailed information from the primary header
    """
    analysis = {
        'basic_info': {},
        'astronomical': {},
        'observational': {},
        'technical': {},
        'calibration': {},
        'processing': {},
        'warnings': [],
        'errors': []
    }

    try:
                                        
        analysis['basic_info'] = {
            'filename': getattr(header, 'filename', 'N/A'),
            'naxis': header.get('NAXIS', 0),
            'naxis1': header.get('NAXIS1', 'N/A'),
            'naxis2': header.get('NAXIS2', 'N/A'),
            'naxis3': header.get('NAXIS3', 'N/A'),
            'naxis4': header.get('NAXIS4', 'N/A'),
            'bitpix': header.get('BITPIX', 'N/A'),
            'extend': header.get('EXTEND', False),
            'groups': header.get('GROUPS', False),
            'pcount': header.get('PCOUNT', 0),
            'gcount': header.get('GCOUNT', 1),
            'bzero': header.get('BZERO', 0.0),
            'bscale': header.get('BSCALE', 1.0)
        }

                                 
        astronomical_info = {}
        for key in ASTRONOMICAL_HEADERS:
            if key in header:
                astronomical_info[key] = header[key]
        analysis['astronomical'] = astronomical_info

                                   
        analysis['observational'] = {
            'object': header.get('OBJECT', 'N/A'),
            'observer': header.get('OBSERVER', 'N/A'),
            'telescope': header.get('TELESCOP', 'N/A'),
            'instrument': header.get('INSTRUME', 'N/A'),
            'obs_mode': header.get('OBS_MODE', 'N/A'),
            'date_obs': header.get('DATE-OBS', 'N/A'),
            'time_obs': header.get('TIME-OBS', 'N/A'),
            'mjd': header.get('MJD', 'N/A'),
            'lst': header.get('LST', 'N/A')
        }

                             
        analysis['technical'] = {
            'nbits': header.get('NBITS', 'N/A'),
            'npol': header.get('NPOL', 'N/A'),
            'nchan': header.get('NCHAN', 'N/A'),
            'nsblk': header.get('NSBLK', 'N/A'),
            'tsamp': header.get('TSAMP', 'N/A'),
            'freq': header.get('FREQ', 'N/A'),
            'bw': header.get('BW', 'N/A'),
            'chan_bw': header.get('CHAN_BW', 'N/A'),
            'obsfreq': header.get('OBSFREQ', 'N/A'),
            'obsbw': header.get('OBSBW', 'N/A'),
            'obsnchan': header.get('OBSNCHAN', 'N/A'),
            'pol_type': header.get('POL_TYPE', 'N/A')
        }

                     
        calibration_info = {}
        for key in CALIBRATION_HEADERS:
            if key in header:
                calibration_info[key] = header[key]
        analysis['calibration'] = calibration_info

                       
        processing_info = {}
        for key in PROCESSING_HEADERS:
            if key in header:
                processing_info[key] = header[key]
        analysis['processing'] = processing_info

                                 
        analysis['warnings'] = _validate_header_integrity(header)
        analysis['errors'] = _check_header_errors(header)

    except Exception as e:
        analysis['errors'].append(f"Error analizando header primario: {e}")

    return analysis

def analyze_extension_headers(hdul: fits.HDUList) -> List[Dict[str, Any]]:
    """
    Analiza los headers de todas las extensiones del archivo FITS.

    Args:
        hdul: HDUList de astropy.io.fits

    Returns:
        Lista de diccionarios con análisis de cada extensión
    """
    extensions_analysis = []

    for i, hdu in enumerate(hdul):
        if i == 0:                                       
            continue

        ext_analysis = {
            'extension_number': i,
            'extension_type': type(hdu).__name__,
            'header_size': len(hdu.header),
            'data_shape': getattr(hdu.data, 'shape', 'N/A') if hdu.data is not None else 'N/A',
            'data_type': str(getattr(hdu.data, 'dtype', 'N/A')) if hdu.data is not None else 'N/A',
            'data_size_mb': _calculate_data_size(hdu),
            'specific_headers': {},
            'warnings': [],
            'errors': []
        }

        try:
                                                         
            if hasattr(hdu, 'header'):
                ext_analysis['specific_headers'] = _extract_extension_specific_headers(hdu.header, hdu)

                                      
            ext_analysis['warnings'] = _validate_extension_integrity(hdu)
            ext_analysis['errors'] = _check_extension_errors(hdu)

        except Exception as e:
            ext_analysis['errors'].append(f"Error analizando extensión {i}: {e}")

        extensions_analysis.append(ext_analysis)

    return extensions_analysis

def analyze_file_structure(filepath: Path) -> Dict[str, Any]:
    """
    Analiza la estructura general del archivo FITS.

    Args:
        filepath: Ruta al archivo FITS

    Returns:
        Dict con información de estructura del archivo
    """
    structure_info = {
        'file_path': str(filepath),
        'file_size_mb': filepath.stat().st_size / (1024 * 1024),
        'file_exists': filepath.exists(),
        'is_fits': False,
        'is_psrfits': False,
        'num_extensions': 0,
        'extensions_info': [],
        'warnings': [],
        'errors': []
    }

    if not filepath.exists():
        structure_info['errors'].append(f"Archivo no encontrado: {filepath}")
        return structure_info

    try:
                                                
        with fits.open(filepath, mode='readonly', memmap=True) as hdul:
            structure_info['is_fits'] = True
            structure_info['num_extensions'] = len(hdul)

                                     
            if len(hdul) > 1:
                primary_header = hdul[0].header
                if 'TELESCOP' in primary_header and 'INSTRUME' in primary_header:
                                                                  
                    has_subint = any('SUBINT' in str(hdu.header) for hdu in hdul[1:])
                    has_pol = any('NPOL' in hdu.header for hdu in hdul if hasattr(hdu, 'header'))
                    if has_subint or has_pol:
                        structure_info['is_psrfits'] = True

                                        
            for i, hdu in enumerate(hdul):
                ext_info = {
                    'number': i,
                    'type': type(hdu).__name__,
                    'name': getattr(hdu, 'name', 'N/A'),
                    'ver': getattr(hdu, 'ver', 'N/A'),
                    'has_data': hdu.data is not None,
                    'data_shape': hdu.data.shape if hdu.data is not None else None,
                    'header_cards': len(hdu.header) if hasattr(hdu, 'header') else 0
                }
                structure_info['extensions_info'].append(ext_info)

    except Exception as e:
        structure_info['errors'].append(f"Error analizando estructura del archivo: {e}")

    return structure_info

def analyze_observational_metadata(header: fits.Header) -> Dict[str, Any]:
    """
    Analiza metadatos astronómicos y observacionales detalladamente.

    Args:
        header: Header de astropy.io.fits

    Returns:
        Dict con metadatos astronómicos detallados
    """
    metadata = {
        'coordinates': {},
        'timing': {},
        'frequency': {},
        'polarization': {},
        'calibration': {},
        'derived_parameters': {},
        'warnings': []
    }

    try:
                                  
        if 'RA' in header and 'DEC' in header:
            try:
                ra = header['RA']
                dec = header['DEC']
                coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
                metadata['coordinates'] = {
                    'ra_deg': float(ra),
                    'dec_deg': float(dec),
                    'ra_hms': coords.ra.to_string(unit=u.hour, sep=':'),
                    'dec_dms': coords.dec.to_string(unit=u.degree, sep=':'),
                    'galactic_l': coords.galactic.l.deg,
                    'galactic_b': coords.galactic.b.deg
                }
            except Exception as e:
                metadata['warnings'].append(f"Error procesando coordenadas: {e}")

                              
        if 'DATE-OBS' in header:
            try:
                obs_time = Time(header['DATE-OBS'])
                metadata['timing'] = {
                    'date_obs': header['DATE-OBS'],
                    'mjd': obs_time.mjd,
                    'jd': obs_time.jd,
                    'datetime': obs_time.datetime.isoformat()
                }
            except Exception as e:
                metadata['warnings'].append(f"Error procesando tiempo de observación: {e}")

                                   
        metadata['frequency'] = {
            'center_freq_mhz': header.get('OBSFREQ', header.get('FREQ', 'N/A')),
            'bandwidth_mhz': header.get('OBSBW', header.get('BW', 'N/A')),
            'num_channels': header.get('OBSNCHAN', header.get('NCHAN', 'N/A')),
            'channel_width_mhz': header.get('CHAN_BW', 'N/A')
        }

                                     
        metadata['polarization'] = {
            'num_pol': header.get('NPOL', 'N/A'),
            'pol_type': header.get('POL_TYPE', 'N/A')
        }

                     
        metadata['calibration'] = {
            'calibrated': header.get('CALIBRATED', False),
            'zero_off': header.get('ZERO_OFF', 'N/A')
        }

                              
        metadata['derived_parameters'] = _calculate_derived_parameters(header)

    except Exception as e:
        metadata['warnings'].append(f"Error en análisis de metadatos: {e}")

    return metadata

                                                                               
                      
                                                                               

def _calculate_data_size(hdu: fits.HDU) -> float:
    """Calcula el tamaño de los datos en MB."""
    if hdu.data is None:
        return 0.0

    try:
                                 
        if hasattr(hdu.data, 'nbytes'):
            size_bytes = hdu.data.nbytes
        else:
                                    
            size_bytes = np.prod(hdu.data.shape) * hdu.data.dtype.itemsize

        return size_bytes / (1024 * 1024)                  
    except:
        return 0.0

def _extract_extension_specific_headers(header: fits.Header, hdu: fits.HDU) -> Dict[str, Any]:
    """Extrae headers específicos según el tipo de extensión."""
    specific_headers = {}

                                              
    if 'EXTNAME' in header and header['EXTNAME'] == 'SUBINT':
        subint_keys = [
            'NAXIS2', 'TFORM1', 'TTYPE1', 'TUNIT1', 'TFORM2', 'TTYPE2', 'TUNIT2',
            'TFORM3', 'TTYPE3', 'TUNIT3', 'TFORM4', 'TTYPE4', 'TUNIT4',
            'TFORM5', 'TTYPE5', 'TUNIT5', 'TFORM6', 'TTYPE6', 'TUNIT6',
            'TFORM7', 'TTYPE7', 'TUNIT7', 'TFORM8', 'TTYPE8', 'TUNIT8',
            'INT_TYPE', 'INT_UNIT', 'SCALE', 'OFFSET', 'WEIGHTS'
        ]
        for key in subint_keys:
            if key in header:
                specific_headers[key] = header[key]

                                        
    elif hasattr(hdu, 'data') and hdu.data is not None:
        specific_headers.update({
            'data_type': str(hdu.data.dtype),
            'data_shape': hdu.data.shape,
            'data_min': float(np.min(hdu.data)) if hdu.data.size > 0 else 'N/A',
            'data_max': float(np.max(hdu.data)) if hdu.data.size > 0 else 'N/A',
            'data_mean': float(np.mean(hdu.data)) if hdu.data.size > 0 else 'N/A',
            'data_std': float(np.std(hdu.data)) if hdu.data.size > 0 else 'N/A'
        })

    return specific_headers

def _validate_header_integrity(header: fits.Header) -> List[str]:
    """Valida la integridad del header y retorna warnings."""
    warnings = []

                                          
    critical_headers = ['NAXIS', 'BITPIX']
    for key in critical_headers:
        if key not in header:
            warnings.append(f"Header crítico faltante: {key}")

                                           
    naxis = header.get('NAXIS', 0)
    for i in range(1, naxis + 1):
        axis_key = f'NAXIS{i}'
        if axis_key not in header:
            warnings.append(f"NAXIS{i} faltante para NAXIS={naxis}")

                                                
    if 'TELESCOP' not in header:
        warnings.append("Header TELESCOP faltante - telescopio no especificado")
    if 'OBS_MODE' not in header:
        warnings.append("Header OBS_MODE faltante - modo de observación no especificado")

    return warnings

def _check_header_errors(header: fits.Header) -> List[str]:
    """Verifica errores críticos en el header."""
    errors = []

                                 
    if header.get('NAXIS', 0) < 0:
        errors.append("NAXIS negativo - archivo corrupto")
    if header.get('BITPIX', 0) not in [-64, -32, 8, 16, 32, 64]:
        errors.append("BITPIX inválido - no es un valor FITS estándar")

    return errors

def _validate_extension_integrity(hdu: fits.HDU) -> List[str]:
    """Valida la integridad de una extensión."""
    warnings = []

    if hasattr(hdu, 'data') and hdu.data is not None:
        if hdu.data.size == 0:
            warnings.append("Extensión contiene datos vacíos")

                                  
        if not np.issubdtype(hdu.data.dtype, np.number):
            warnings.append(f"Tipo de datos no numérico: {hdu.data.dtype}")

    return warnings

def _check_extension_errors(hdu: fits.HDU) -> List[str]:
    """Verifica errores en una extensión."""
    errors = []

    if hasattr(hdu, 'header'):
                                                                
        if 'XTENSION' not in hdu.header:
            errors.append("Header XTENSION faltante en extensión")

    return errors

def _calculate_derived_parameters(header: fits.Header) -> Dict[str, Any]:
    """Calcula parámetros derivados de los headers."""
    derived = {}

    try:
                                     
        if 'TSAMP' in header and 'NSBLK' in header and 'NAXIS2' in header:
            tsamp = float(header['TSAMP'])
            nsblk = int(header['NSBLK'])
            naxis2 = int(header['NAXIS2'])
            total_samples = nsblk * naxis2
            total_time_sec = total_samples * tsamp
            derived['total_observation_time_sec'] = total_time_sec
            derived['total_observation_time_min'] = total_time_sec / 60
            derived['total_observation_time_hours'] = total_time_sec / 3600

                                      
        if 'TSAMP' in header:
            derived['temporal_resolution_ms'] = float(header['TSAMP']) * 1000

                                
        if 'CHAN_BW' in header:
            derived['frequency_resolution_mhz'] = abs(float(header['CHAN_BW']))

                       
        if 'OBSBW' in header and 'NBITS' in header and 'NPOL' in header:
            bw = abs(float(header['OBSBW']))
            nbits = int(header['NBITS'])
            npol = int(header['NPOL'])
            data_rate_mbps = (bw * nbits * npol * 2) / 1e6                    
            derived['data_rate_mbps'] = data_rate_mbps

    except Exception as e:
        derived['calculation_error'] = str(e)

    return derived

                                                                               
                                     
                                                                               

def print_analysis_report(filepath: Path, analysis: Dict[str, Any]) -> None:
    """
    Imprime un reporte completo del análisis del archivo FITS.

    Args:
        filepath: Ruta al archivo analizado
        analysis: Resultados del análisis completo
    """
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}ANÁLISIS DE HEADER FITS{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}Archivo:{Colors.RESET} {filepath.name}")
    print(f"{Colors.BOLD}Ruta:{Colors.RESET} {filepath}")
    print(f"{Colors.BOLD}Tamaño:{Colors.RESET} {analysis['structure']['file_size_mb']:.2f} MB")

                               
    struct = analysis['structure']
    print(f"\n{Colors.BOLD}{Colors.GREEN}ESTRUCTURA DEL ARCHIVO:{Colors.RESET}")
    print(f"  • Es FITS válido: {struct['is_fits']}")
    print(f"  • Es PSRFITS: {struct['is_psrfits']}")
    print(f"  • Número de extensiones: {struct['num_extensions']}")

    if struct['extensions_info']:
        print(f"  • Extensiones:")
        for ext in struct['extensions_info']:
            print(f"    - Ext {ext['number']}: {ext['type']} ({ext['header_cards']} headers)")

                             
    if analysis['primary']['astronomical']:
        print(f"\n{Colors.BOLD}{Colors.GREEN}INFORMACIÓN ASTRONÓMICA:{Colors.RESET}")
        astro = analysis['primary']['astronomical']
        for key, value in astro.items():
            print(f"  • {key}: {value}")

                               
    obs = analysis['primary']['observational']
    print(f"\n{Colors.BOLD}{Colors.GREEN}INFORMACIÓN OBSERVACIONAL:{Colors.RESET}")
    print(f"  • Objeto: {obs['object']}")
    print(f"  • Observador: {obs['observer']}")
    print(f"  • Telescopio: {obs['telescope']}")
    print(f"  • Instrumento: {obs['instrument']}")
    print(f"  • Modo: {obs['obs_mode']}")
    print(f"  • Fecha observación: {obs['date_obs']}")
    print(f"  • Tiempo observación: {obs['time_obs']}")

                         
    tech = analysis['primary']['technical']
    print(f"\n{Colors.BOLD}{Colors.GREEN}INFORMACIÓN TÉCNICA:{Colors.RESET}")
    print(f"  • Bits por muestra: {tech['nbits']}")
    print(f"  • Número de polarizaciones: {tech['npol']}")
    print(f"  • Número de canales: {tech['nchan']}")
    print(f"  • Muestras por bloque: {tech['nsblk']}")
    print(f"  • Tiempo muestreo: {tech['tsamp']}")
    print(f"  • Frecuencia central: {tech['freq']} MHz")
    print(f"  • Ancho de banda: {tech['bw']} MHz")

                          
    if analysis['metadata']['coordinates']:
        coords = analysis['metadata']['coordinates']
        print(f"\n{Colors.BOLD}{Colors.GREEN}COORDENADAS ASTRONÓMICAS:{Colors.RESET}")
        print(f"  • RA: {coords.get('ra_hms', 'N/A')} ({coords.get('ra_deg', 'N/A')}°)")
        print(f"  • DEC: {coords.get('dec_dms', 'N/A')} ({coords.get('dec_deg', 'N/A')}°)")
        print(f"  • Longitud galáctica: {coords.get('galactic_l', 'N/A')}°")
        print(f"  • Latitud galáctica: {coords.get('galactic_b', 'N/A')}°")

                          
    if analysis['metadata']['derived_parameters']:
        derived = analysis['metadata']['derived_parameters']
        print(f"\n{Colors.BOLD}{Colors.GREEN}PARÁMETROS DERIVADOS:{Colors.RESET}")
        for key, value in derived.items():
            if isinstance(value, float):
                print(f"  • {key}: {value:.2f}")
            else:
                print(f"  • {key}: {value}")

                 
    if analysis['extensions']:
        print(f"\n{Colors.BOLD}{Colors.GREEN}ANÁLISIS DE EXTENSIONES:{Colors.RESET}")
        for ext in analysis['extensions']:
            print(f"  • Extensión {ext['extension_number']} ({ext['extension_type']}):")
            print(f"    - Tamaño datos: {ext['data_size_mb']:.2f} MB")
            print(f"    - Forma datos: {ext['data_shape']}")
            print(f"    - Tipo datos: {ext['data_type']}")

                        
    if analysis['primary']['warnings']:
        print(f"\n{Colors.BOLD}{Colors.YELLOW}ADVERTENCIAS:{Colors.RESET}")
        for warning in analysis['primary']['warnings']:
            print(f"  • {warning}")

    if analysis['primary']['errors']:
        print(f"\n{Colors.BOLD}{Colors.RED}ERRORES:{Colors.RESET}")
        for error in analysis['primary']['errors']:
            print(f"  • {error}")

def print_summary_report(files_analyzed: List[Dict[str, Any]]) -> None:
    """
    Imprime un reporte resumen de todos los archivos analizados.

    Args:
        files_analyzed: Lista de análisis de archivos
    """
    if not files_analyzed:
        print(f"\n{Colors.YELLOW}No se analizaron archivos.{Colors.RESET}")
        return

    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}REPORTE RESUMEN{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"Archivos analizados: {len(files_analyzed)}")

    total_size = sum(f['analysis']['structure']['file_size_mb'] for f in files_analyzed)
    print(f"Tamaño total: {total_size:.2f} MB")

                           
    psrfits_count = sum(1 for f in files_analyzed if f['analysis']['structure']['is_psrfits'])
    fits_count = sum(1 for f in files_analyzed if f['analysis']['structure']['is_fits'] and not f['analysis']['structure']['is_psrfits'])

    print(f"Archivos PSRFITS: {psrfits_count}")
    print(f"Archivos FITS estándar: {fits_count}")

                         
    total_errors = sum(len(f['analysis']['primary']['errors']) for f in files_analyzed)
    total_warnings = sum(len(f['analysis']['primary']['warnings']) for f in files_analyzed)

    if total_errors > 0:
        print(f"{Colors.RED}Errores totales: {total_errors}{Colors.RESET}")
    if total_warnings > 0:
        print(f"{Colors.YELLOW}Advertencias totales: {total_warnings}{Colors.RESET}")

                                     
    problematic_files = [f for f in files_analyzed if f['analysis']['primary']['errors']]
    if problematic_files:
        print(f"\n{Colors.RED}Archivos con errores:{Colors.RESET}")
        for f in problematic_files:
            error_count = len(f['analysis']['primary']['errors'])
            print(f"  • {f['filepath'].name}: {error_count} errores")

                                                                               
                       
                                                                               

def analyze_fits_file(filepath: Path) -> Dict[str, Any]:
    """
    Función principal para analizar un archivo FITS completo.

    Args:
        filepath: Ruta al archivo FITS

    Returns:
        Dict con análisis completo del archivo
    """
    logger.info(f"Analizando archivo: {filepath.name}")

    analysis = {
        'structure': {},
        'primary': {},
        'extensions': [],
        'metadata': {},
        'processing_time': 0.0,
        'errors': []
    }

    try:
        import time
        start_time = time.time()

                                
        analysis['structure'] = analyze_file_structure(filepath)

        if not analysis['structure']['is_fits']:
            analysis['errors'].append("No es un archivo FITS válido")
            return analysis

                                        
        with fits.open(filepath, mode='readonly', memmap=True) as hdul:
                             
            analysis['primary'] = analyze_primary_header(hdul[0].header)

                         
            analysis['extensions'] = analyze_extension_headers(hdul)

                                    
            analysis['metadata'] = analyze_observational_metadata(hdul[0].header)

        analysis['processing_time'] = time.time() - start_time
        logger.info(f"Análisis completado en {analysis['processing_time']:.2f} segundos")

    except Exception as e:
        error_msg = f"Error crítico analizando {filepath.name}: {e}"
        logger.error(error_msg)
        analysis['errors'].append(error_msg)

    return analysis

def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Analizador de headers FITS para el pipeline DRAFTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python fits_header_analyzer.py archivo.fits
  python fits_header_analyzer.py archivo1.fits archivo2.fits
  python fits_header_analyzer.py --all
  python fits_header_analyzer.py --targets
  python fits_header_analyzer.py --dir /ruta/a/directorio
        """
    )

    parser.add_argument(
        'files',
        nargs='*',
        help='Archivos FITS a analizar'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Analizar todos los archivos FITS en DATA_DIR'
    )

    parser.add_argument(
        '--targets',
        action='store_true',
        help='Analizar archivos especificados en FRB_TARGETS'
    )

    parser.add_argument(
        '--dir',
        type=str,
        help='Directorio específico para buscar archivos FITS'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Archivo de salida para el reporte (opcional)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Modo silencioso, solo reportes importantes'
    )

    args = parser.parse_args()

                        
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)

                                    
    files_to_analyze = []

    if args.files:
                                                      
        for file_path in args.files:
            path = Path(file_path)
            if path.exists():
                files_to_analyze.append(path)
            else:
                logger.warning(f"Archivo no encontrado: {file_path}")

    elif args.all:
                                        
        data_dir = Path(config.DATA_DIR)
        if data_dir.exists():
            fits_files = list(data_dir.glob("*.fits")) + list(data_dir.glob("*.FITS"))
            files_to_analyze.extend(fits_files)
            logger.info(f"Encontrados {len(fits_files)} archivos FITS en {data_dir}")
        else:
            logger.error(f"Directorio de datos no encontrado: {data_dir}")

    elif args.targets:
                                 
        data_dir = Path(config.DATA_DIR)
        for target in config.FRB_TARGETS:
            target_path = data_dir / f"{target}.fits"
            if target_path.exists():
                files_to_analyze.append(target_path)
            else:
                logger.warning(f"Archivo target no encontrado: {target_path}")

    elif args.dir:
                               
        search_dir = Path(args.dir)
        if search_dir.exists():
            fits_files = list(search_dir.glob("*.fits")) + list(search_dir.glob("*.FITS"))
            files_to_analyze.extend(fits_files)
            logger.info(f"Encontrados {len(fits_files)} archivos FITS en {search_dir}")
        else:
            logger.error(f"Directorio no encontrado: {search_dir}")

    else:
                                                                                
        data_dir = Path(config.DATA_DIR)
        target_files = ["B0355+54_FB_20220918", "FRB20201124_0009"]
        for target in target_files:
            target_path = data_dir / f"{target}.fits"
            if target_path.exists():
                files_to_analyze.append(target_path)
            else:
                logger.warning(f"Archivo target no encontrado: {target_path}")

    if not files_to_analyze:
        logger.error("No se encontraron archivos FITS para analizar")
        return 1

                       
    logger.info(f"Iniciando análisis de {len(files_to_analyze)} archivos FITS")
    files_analyzed = []

    for filepath in files_to_analyze:
        try:
            analysis = analyze_fits_file(filepath)
            files_analyzed.append({
                'filepath': filepath,
                'analysis': analysis
            })

                                        
            if not args.quiet:
                print_analysis_report(filepath, analysis)

        except Exception as e:
            logger.error(f"Error analizando {filepath.name}: {e}")

                     
    print_summary_report(files_analyzed)

                                      
    if args.output:
        try:
            import json
            output_data = {
                'timestamp': str(np.datetime64('now')),
                'files_analyzed': len(files_analyzed),
                'analyses': [
                    {
                        'filepath': str(f['filepath']),
                        'analysis': f['analysis']
                    } for f in files_analyzed
                ]
            }

            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Reporte guardado en: {args.output}")

        except Exception as e:
            logger.error(f"Error guardando reporte: {e}")

    logger.info("Análisis completado")
    return 0

if __name__ == "__main__":
                                  
    warnings.filterwarnings('ignore', category=UserWarning, module='astropy')

    sys.exit(main())
