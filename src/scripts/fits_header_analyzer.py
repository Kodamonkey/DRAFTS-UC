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
        analysis['errors'].append(f"Error analyzing primary header: {e}")

    return analysis

def analyze_extension_headers(hdul: fits.HDUList) -> List[Dict[str, Any]]:
    """Analyze headers of all FITS file extensions.

    Args:
        hdul: ``astropy.io.fits.HDUList`` instance

    Returns:
        List of dictionaries with the analysis of each extension
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
            ext_analysis['errors'].append(f"Error analyzing extension {i}: {e}")

        extensions_analysis.append(ext_analysis)

    return extensions_analysis

def analyze_file_structure(filepath: Path) -> Dict[str, Any]:
    """Analyze the general structure of the FITS file.

    Args:
        filepath: Path to the FITS file

    Returns:
        Dict with structural information about the file
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
        structure_info['errors'].append(f"File not found: {filepath}")
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
        structure_info['errors'].append(f"Error analyzing file structure: {e}")

    return structure_info

def analyze_observational_metadata(header: fits.Header) -> Dict[str, Any]:
    """Analyze astronomical and observational metadata in detail.

    Args:
        header: ``astropy.io.fits`` header

    Returns:
        Dict with detailed astronomical metadata
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
                metadata['warnings'].append(f"Error processing coordinates: {e}")

                              
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
                metadata['warnings'].append(
                    f"Error processing observation time: {e}"
                )

                                   
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
        metadata['warnings'].append(f"Error in metadata analysis: {e}")

    return metadata

                                                                               
                      
                                                                               

def _calculate_data_size(hdu: fits.HDU) -> float:
    """Calculate the size of the data in MB."""
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
    """Extract specific headers according to the extension type."""
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
    """Validate header integrity and return warnings."""
    warnings = []

                                          
    critical_headers = ['NAXIS', 'BITPIX']
    for key in critical_headers:
        if key not in header:
            warnings.append(f"Missing critical header: {key}")

                                           
    naxis = header.get('NAXIS', 0)
    for i in range(1, naxis + 1):
        axis_key = f'NAXIS{i}'
        if axis_key not in header:
            warnings.append(f"Missing NAXIS{i} for NAXIS={naxis}")

                                                
    if 'TELESCOP' not in header:
        warnings.append("Missing TELESCOP header - telescope not specified")
    if 'OBS_MODE' not in header:
        warnings.append("Missing OBS_MODE header - observation mode not specified")

    return warnings

def _check_header_errors(header: fits.Header) -> List[str]:
    """Check for critical errors in the header."""
    errors = []

                                 
    if header.get('NAXIS', 0) < 0:
        errors.append("Negative NAXIS - corrupted file")
    if header.get('BITPIX', 0) not in [-64, -32, 8, 16, 32, 64]:
        errors.append("Invalid BITPIX - not a standard FITS value")

    return errors

def _validate_extension_integrity(hdu: fits.HDU) -> List[str]:
    """Validate the integrity of an extension."""
    warnings = []

    if hasattr(hdu, 'data') and hdu.data is not None:
        if hdu.data.size == 0:
            warnings.append("Extension contains empty data")

                                  
        if not np.issubdtype(hdu.data.dtype, np.number):
            warnings.append(f"Non-numeric data type: {hdu.data.dtype}")

    return warnings

def _check_extension_errors(hdu: fits.HDU) -> List[str]:
    """Check for errors in an extension."""
    errors = []

    if hasattr(hdu, 'header'):
                                                                
        if 'XTENSION' not in hdu.header:
            errors.append("Missing XTENSION header in extension")

    return errors

def _calculate_derived_parameters(header: fits.Header) -> Dict[str, Any]:
    """Calculate derived parameters from the headers."""
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
    """Print a complete report of the FITS file analysis.

    Args:
        filepath: Path to the analyzed file
        analysis: Results from the full analysis
    """
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}FITS HEADER ANALYSIS{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}File:{Colors.RESET} {filepath.name}")
    print(f"{Colors.BOLD}Path:{Colors.RESET} {filepath}")
    print(f"{Colors.BOLD}Size:{Colors.RESET} {analysis['structure']['file_size_mb']:.2f} MB")

                               
    struct = analysis['structure']
    print(f"\n{Colors.BOLD}{Colors.GREEN}FILE STRUCTURE:{Colors.RESET}")
    print(f"  • Is valid FITS: {struct['is_fits']}")
    print(f"  • Is PSRFITS: {struct['is_psrfits']}")
    print(f"  • Number of extensions: {struct['num_extensions']}")

    if struct['extensions_info']:
        print(f"  • Extensiones:")
        for ext in struct['extensions_info']:
            print(f"    - Ext {ext['number']}: {ext['type']} ({ext['header_cards']} headers)")

                             
    if analysis['primary']['astronomical']:
        print(f"\n{Colors.BOLD}{Colors.GREEN}ASTRONOMICAL INFORMATION:{Colors.RESET}")
        astro = analysis['primary']['astronomical']
        for key, value in astro.items():
            print(f"  • {key}: {value}")

                               
    obs = analysis['primary']['observational']
    print(f"\n{Colors.BOLD}{Colors.GREEN}OBSERVATIONAL INFORMATION:{Colors.RESET}")
    print(f"  • Object: {obs['object']}")
    print(f"  • Observer: {obs['observer']}")
    print(f"  • Telescope: {obs['telescope']}")
    print(f"  • Instrument: {obs['instrument']}")
    print(f"  • Mode: {obs['obs_mode']}")
    print(f"  • Observation date: {obs['date_obs']}")
    print(f"  • Observation time: {obs['time_obs']}")

                         
    tech = analysis['primary']['technical']
    print(f"\n{Colors.BOLD}{Colors.GREEN}TECHNICAL INFORMATION:{Colors.RESET}")
    print(f"  • Bits per sample: {tech['nbits']}")
    print(f"  • Number of polarizations: {tech['npol']}")
    print(f"  • Number of channels: {tech['nchan']}")
    print(f"  • Samples per block: {tech['nsblk']}")
    print(f"  • Sampling time: {tech['tsamp']}")
    print(f"  • Center frequency: {tech['freq']} MHz")
    print(f"  • Bandwidth: {tech['bw']} MHz")

                          
    if analysis['metadata']['coordinates']:
        coords = analysis['metadata']['coordinates']
        print(f"\n{Colors.BOLD}{Colors.GREEN}ASTRONOMICAL COORDINATES:{Colors.RESET}")
        print(f"  • RA: {coords.get('ra_hms', 'N/A')} ({coords.get('ra_deg', 'N/A')}°)")
        print(f"  • DEC: {coords.get('dec_dms', 'N/A')} ({coords.get('dec_deg', 'N/A')}°)")
        print(f"  • Galactic longitude: {coords.get('galactic_l', 'N/A')}°")
        print(f"  • Galactic latitude: {coords.get('galactic_b', 'N/A')}°")

                          
    if analysis['metadata']['derived_parameters']:
        derived = analysis['metadata']['derived_parameters']
        print(f"\n{Colors.BOLD}{Colors.GREEN}DERIVED PARAMETERS:{Colors.RESET}")
        for key, value in derived.items():
            if isinstance(value, float):
                print(f"  • {key}: {value:.2f}")
            else:
                print(f"  • {key}: {value}")

                 
    if analysis['extensions']:
        print(f"\n{Colors.BOLD}{Colors.GREEN}EXTENSION ANALYSIS:{Colors.RESET}")
        for ext in analysis['extensions']:
            print(f"  • Extension {ext['extension_number']} ({ext['extension_type']}):")
            print(f"    - Data size: {ext['data_size_mb']:.2f} MB")
            print(f"    - Data shape: {ext['data_shape']}")
            print(f"    - Data type: {ext['data_type']}")

                        
    if analysis['primary']['warnings']:
        print(f"\n{Colors.BOLD}{Colors.YELLOW}WARNINGS:{Colors.RESET}")
        for warning in analysis['primary']['warnings']:
            print(f"  • {warning}")

    if analysis['primary']['errors']:
        print(f"\n{Colors.BOLD}{Colors.RED}ERRORS:{Colors.RESET}")
        for error in analysis['primary']['errors']:
            print(f"  • {error}")

def print_summary_report(files_analyzed: List[Dict[str, Any]]) -> None:
    """Print a summary report for all analyzed files.

    Args:
        files_analyzed: List of file analyses
    """
    if not files_analyzed:
        print(f"\n{Colors.YELLOW}No files were analyzed.{Colors.RESET}")
        return

    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}SUMMARY REPORT{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"Analyzed files: {len(files_analyzed)}")

    total_size = sum(f['analysis']['structure']['file_size_mb'] for f in files_analyzed)
    print(f"Total size: {total_size:.2f} MB")

                           
    psrfits_count = sum(1 for f in files_analyzed if f['analysis']['structure']['is_psrfits'])
    fits_count = sum(1 for f in files_analyzed if f['analysis']['structure']['is_fits'] and not f['analysis']['structure']['is_psrfits'])

    print(f"PSRFITS files: {psrfits_count}")
    print(f"Standard FITS files: {fits_count}")

                         
    total_errors = sum(len(f['analysis']['primary']['errors']) for f in files_analyzed)
    total_warnings = sum(len(f['analysis']['primary']['warnings']) for f in files_analyzed)

    if total_errors > 0:
        print(f"{Colors.RED}Total errors: {total_errors}{Colors.RESET}")
    if total_warnings > 0:
        print(f"{Colors.YELLOW}Total warnings: {total_warnings}{Colors.RESET}")

                                     
    problematic_files = [f for f in files_analyzed if f['analysis']['primary']['errors']]
    if problematic_files:
        print(f"\n{Colors.RED}Files with errors:{Colors.RESET}")
        for f in problematic_files:
            error_count = len(f['analysis']['primary']['errors'])
            print(f"  • {f['filepath'].name}: {error_count} errors")

                                                                               
                       
                                                                               

def analyze_fits_file(filepath: Path) -> Dict[str, Any]:
    """Main function to analyze a complete FITS file.

    Args:
        filepath: Path to the FITS file

    Returns:
        Dict with the complete file analysis
    """
    logger.info(f"Analyzing file: {filepath.name}")

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
            analysis['errors'].append("Not a valid FITS file")
            return analysis

                                        
        with fits.open(filepath, mode='readonly', memmap=True) as hdul:
                             
            analysis['primary'] = analyze_primary_header(hdul[0].header)

                         
            analysis['extensions'] = analyze_extension_headers(hdul)

                                    
            analysis['metadata'] = analyze_observational_metadata(hdul[0].header)

        analysis['processing_time'] = time.time() - start_time
        logger.info(
            f"Analysis completed in {analysis['processing_time']:.2f} seconds"
        )

    except Exception as e:
        error_msg = f"Critical error analyzing {filepath.name}: {e}"
        logger.error(error_msg)
        analysis['errors'].append(error_msg)

    return analysis

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="FITS header analyzer for the DRAFTS pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python fits_header_analyzer.py file.fits
  python fits_header_analyzer.py file1.fits file2.fits
  python fits_header_analyzer.py --all
  python fits_header_analyzer.py --targets
  python fits_header_analyzer.py --dir /path/to/directory
        """
    )

    parser.add_argument(
        'files',
        nargs='*',
        help='FITS files to analyze'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Analyze all FITS files in DATA_DIR'
    )

    parser.add_argument(
        '--targets',
        action='store_true',
        help='Analyze files listed in FRB_TARGETS'
    )

    parser.add_argument(
        '--dir',
        type=str,
        help='Specific directory to search for FITS files'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output file for the report (optional)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Quiet mode, only important reports'
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
                logger.warning(f"File not found: {file_path}")

    elif args.all:
                                        
        data_dir = Path(config.DATA_DIR)
        if data_dir.exists():
            fits_files = list(data_dir.glob("*.fits")) + list(data_dir.glob("*.FITS"))
            files_to_analyze.extend(fits_files)
            logger.info(f"Found {len(fits_files)} FITS files in {data_dir}")
        else:
            logger.error(f"Data directory not found: {data_dir}")

    elif args.targets:
                                 
        data_dir = Path(config.DATA_DIR)
        for target in config.FRB_TARGETS:
            target_path = data_dir / f"{target}.fits"
            if target_path.exists():
                files_to_analyze.append(target_path)
            else:
                logger.warning(f"Target file not found: {target_path}")

    elif args.dir:
                               
        search_dir = Path(args.dir)
        if search_dir.exists():
            fits_files = list(search_dir.glob("*.fits")) + list(search_dir.glob("*.FITS"))
            files_to_analyze.extend(fits_files)
            logger.info(f"Found {len(fits_files)} FITS files in {search_dir}")
        else:
            logger.error(f"Directory not found: {search_dir}")

    else:
                                                                                
        data_dir = Path(config.DATA_DIR)
        target_files = ["B0355+54_FB_20220918", "FRB20201124_0009"]
        for target in target_files:
            target_path = data_dir / f"{target}.fits"
            if target_path.exists():
                files_to_analyze.append(target_path)
            else:
                logger.warning(f"Target file not found: {target_path}")

    if not files_to_analyze:
        logger.error("No FITS files found to analyze")
        return 1

                       
    logger.info(f"Starting analysis of {len(files_to_analyze)} FITS files")
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
            logger.error(f"Error analyzing {filepath.name}: {e}")

                     
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

            logger.info(f"Report saved to: {args.output}")

        except Exception as e:
            logger.error(f"Error saving report: {e}")

    logger.info("Analysis completed")
    return 0

if __name__ == "__main__":
                                  
    warnings.filterwarnings('ignore', category=UserWarning, module='astropy')

    sys.exit(main())
