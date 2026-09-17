# This module handles FITS and PSRFITS data ingestion.

"""FITS and PSRFITS file handling for FRB detection pipeline."""
from __future__ import annotations

import os
import time
from typing import Any, Generator, Optional, Tuple, Dict

import numpy as np
from astropy.io import fits
import logging

                              
try:
    import fitsio
except ImportError:
    fitsio = None

                                                               
try:
    from your.formats import psrfits as your_psrfits
except Exception:
    your_psrfits = None

               
from ..config import config
from ..log_utils import (
    log_stream_fits_block_generation,
    log_stream_fits_parameters,
    log_stream_fits_summary
)
from .utils import safe_float, safe_int, auto_config_downsampling, print_debug_frequencies, save_file_debug_info, normalize_frequency_axis
from .polarization_utils import debiased_linear_polarization
from .psrfits_chunking import (
    BufferLimits,
    ChunkSpan,
    SubintBuffer,
    chunk_metadata,
    compute_buffer_limits,
    expected_start_sample,
    plan_buffered_window,
    plan_sequential_span,
)


logger = logging.getLogger(__name__)


def _read_fits_like_fitsio(file_name: str) -> Tuple[np.ndarray, Any]:
    """Return ``(data, header)`` like ``fitsio.read(..., header=True)``.

    Uses ``fitsio`` when installed (faster on supported platforms). If missing—e.g.
    Python 3.14 on Windows without wheels—uses the first HDU with data via Astropy.
    """
    if fitsio is not None:
        return fitsio.read(file_name, header=True)
    with fits.open(file_name, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            return hdu.data, hdu.header
    raise ValueError(f"No HDU with data in {file_name}")


def _unpack_1bit(data: np.ndarray) -> np.ndarray:
    """Unpack 1-bit samples stored in bytes into ``uint8`` values."""
    b0 = (data >> 0x07) & 0x01
    b1 = (data >> 0x06) & 0x01
    b2 = (data >> 0x05) & 0x01
    b3 = (data >> 0x04) & 0x01
    b4 = (data >> 0x03) & 0x01
    b5 = (data >> 0x02) & 0x01
    b6 = (data >> 0x01) & 0x01
    b7 = data & 0x01
    return np.dstack([b0, b1, b2, b3, b4, b5, b6, b7]).flatten()


def _unpack_2bit(data: np.ndarray) -> np.ndarray:
    """Unpack 2-bit samples stored in bytes into ``uint8`` values."""
    p0 = (data >> 0x06) & 0x03
    p1 = (data >> 0x04) & 0x03
    p2 = (data >> 0x02) & 0x03
    p3 = data & 0x03
    return np.dstack([p0, p1, p2, p3]).flatten()


def _unpack_4bit(data: np.ndarray) -> np.ndarray:
    """Unpack 4-bit samples stored in bytes into ``uint8`` values."""
    p0 = (data >> 0x04) & 0x0F
    p1 = data & 0x0F
    return np.dstack([p0, p1]).flatten()


def _apply_calibration(
    data: np.ndarray,
    dat_wts: np.ndarray | None,
    dat_scl: np.ndarray | None,
    dat_offs: np.ndarray | None,
    zero_off: float,
) -> np.ndarray:
    """Apply PSRFITS calibration to a data block.

    Parameters
    ----------
    data : np.ndarray
        Float32 array with shape ``(nsblk, npol, nchan)``.
    dat_wts : np.ndarray | None
        Channel weights ``(nchan,)``.
    dat_scl : np.ndarray | None
        Scale factors ``(npol * nchan,)``.
    dat_offs : np.ndarray | None
        Offsets ``(npol * nchan,)``.
    """
    if data.dtype != np.float32:
        data = data.astype(np.float32, copy=False)
    if zero_off:
        data -= np.float32(zero_off)
    if dat_scl is not None:
                                  
        npol = data.shape[1]
        nchan = data.shape[2]
        scl = dat_scl.reshape(npol, nchan).astype(np.float32, copy=False)
        data *= scl[np.newaxis, :, :]
    if dat_offs is not None:
        npol = data.shape[1]
        nchan = data.shape[2]
        offs = dat_offs.reshape(npol, nchan).astype(np.float32, copy=False)
        data += offs[np.newaxis, :, :]
    if dat_wts is not None:
        wts = dat_wts.astype(np.float32, copy=False)
        data *= wts[np.newaxis, np.newaxis, :]
    return data


def _select_polarization(
    data: np.ndarray,
    pol_type: str,
    mode: str,
    default_index: int = 0,
) -> np.ndarray:
    """Select or compose the polarisation according to configuration.

    - data: (nsamp, npol, nchan)
    - pol_type: Header ``POL_TYPE`` (e.g., "IQUV", "AABB", etc.)
    - mode: "intensity", "linear", "circular", or "pol{idx}"
    """
    npol = data.shape[1]
    if npol == 1:
        return data[:, 0:1, :]

    mode_l = (mode or "").strip().lower()
    pol_type_u = (pol_type or "").strip().upper()

                                       
    if pol_type_u == "IQUV" and npol >= 4:
        if mode_l in ("intensity", "i", "stokes_i", "intensidad"):
            return data[:, 0:1, :]
        if mode_l in ("linear", "l", "lineal"):
            q = data[:, 1, :]
            u = data[:, 2, :]
            l = debiased_linear_polarization(q, u, enabled=bool(getattr(config, "POLARIZATION_LINEAR_DEBIAS", True)))
            return l[:, np.newaxis, :]
        if mode_l in ("circular", "v", "c"):
            v = np.abs(data[:, 3, :]).astype(data.dtype, copy=False)
            return v[:, np.newaxis, :]
        if mode_l.startswith("pol"):
            try:
                idx = int(mode_l.replace("pol", ""))
            except Exception:
                idx = default_index
            idx = max(0, min(npol - 1, idx))
            return data[:, idx:idx + 1, :]
                                
        return data[:, 0:1, :]

                                 
    if mode_l.startswith("pol"):
        try:
            idx = int(mode_l.replace("pol", ""))
        except Exception:
            idx = default_index
        idx = max(0, min(npol - 1, idx))
        return data[:, idx:idx + 1, :]

                          
    return data[:, default_index:default_index + 1, :]


def _load_fits_non_subint(file_name: str) -> np.ndarray:
    """Load a non-SUBINT FITS file into memory.

    Only used as a last-resort fallback for FITS files that lack a SUBINT table
    (simple IMAGE/BINTABLE formats).  PSRFITS files with SUBINT tables are
    handled by the streaming path in ``stream_fits()``.
    """
    backend = "fitsio" if fitsio is not None else "astropy"
    logger.warning(
        "Loading non-SUBINT FITS via %s (entire file in RAM): %s", backend, file_name
    )

    temp_data, h = _read_fits_like_fitsio(file_name)
    total_samples = safe_int(h.get("NAXIS2", 1)) * safe_int(h.get("NSBLK", 1))
    num_pols = safe_int(h.get("NPOL", 2))
    num_chans = safe_int(h.get("NCHAN", 512))

    if any(x <= 0 for x in [total_samples, num_pols, num_chans]):
        raise ValueError(
            f"Invalid FITS dimensions: samples={total_samples}, "
            f"pols={num_pols}, chans={num_chans}"
        )

    has_data_field = hasattr(temp_data, "dtype") and hasattr(temp_data.dtype, "names") and temp_data.dtype.names and "DATA" in temp_data.dtype.names
    raw = temp_data["DATA"] if has_data_field else temp_data

    try:
        data_array = raw.reshape(total_samples, num_pols, num_chans)
    except Exception as e:
        raise ValueError(
            f"Cannot reshape data to ({total_samples}, {num_pols}, {num_chans}): {e}"
        ) from e

    data_array = data_array[:, 0:1, :]

    if data_array.dtype != np.float32:
        data_array = data_array.astype(np.float32)

    if config.DATA_NEEDS_REVERSAL:
        data_array = np.ascontiguousarray(data_array[:, :, ::-1])

    return data_array


def get_obparams(file_name: str) -> None:
    """Extract observation parameters and populate :mod:`config`."""
    
                                               
    if config.DEBUG_FREQUENCY_ORDER:
        logger.debug(f"[DEBUG HEADER] Starting parameter extraction from: {file_name}")
        logger.debug(f"[DEBUG HEADER] " + "="*60)
    
    with fits.open(file_name, memmap=True) as f:
        freq_axis_inverted = False
        
                                            
        if config.DEBUG_FREQUENCY_ORDER:
            logger.debug(f"[DEBUG HEADER] FITS file structure:")
            for i, hdu in enumerate(f):
                hdu_type = type(hdu).__name__
                if hasattr(hdu, 'header') and hdu.header:
                    if 'EXTNAME' in hdu.header:
                        ext_name = hdu.header['EXTNAME']
                    else:
                        ext_name = 'PRIMARY' if i == 0 else f'HDU_{i}'
                    logger.debug(f"[DEBUG HEADER]   HDU {i}: {hdu_type} - {ext_name}")
                    if hasattr(hdu, 'columns') and hdu.columns:
                        logger.debug(f"[DEBUG HEADER]     Columns: {[col.name for col in hdu.columns]}")
                else:
                    logger.debug(f"[DEBUG HEADER]   HDU {i}: {hdu_type} - No header")

        if "SUBINT" in [hdu.name for hdu in f] and "TBIN" in f["SUBINT"].header:

            if config.DEBUG_FREQUENCY_ORDER:
                logger.debug(f"[DEBUG HEADER] Detected format: PSRFITS (SUBINT)")
            
            hdr = f["SUBINT"].header
            primary = f["PRIMARY"].header if "PRIMARY" in [h.name for h in f] else {}
                                                                                     
            try:
                sub_data = f["SUBINT"].data
            except (TypeError, ValueError, OSError) as e:
                if "buffer is too small" in str(e) or "truncated" in str(e).lower():
                    logger.warning(
                        "Truncated FITS file detected in get_obparams (%s): %s. Skipping.",
                        file_name,
                        e,
                    )
                    raise ValueError(f"Truncated FITS file: {file_name}") from e
                else:
                    raise
                                                                                   
            config.TIME_RESO = safe_float(hdr.get("TBIN"))
                                                     
            try:
                tsubint = safe_float(hdr.get("TSUBINT"))
            except Exception:
                tsubint = safe_float(hdr.get("NSBLK", 1)) * config.TIME_RESO
            try:
                config.TSUBINT = tsubint
            except Exception:
                pass
            config.FREQ_RESO = safe_int(hdr.get("NCHAN", 512))
            config.FILE_LENG = (
                safe_int(hdr.get("NSBLK")) * safe_int(hdr.get("NAXIS2"))
            )
                                                                    
            try:
                config.NBITS = safe_int(hdr.get("NBITS", 8))
            except Exception:
                config.NBITS = 8
            try:
                config.NPOL = safe_int(hdr.get("NPOL", 1))
            except Exception:
                config.NPOL = 1
            try:
                config.POL_TYPE = str(hdr.get("POL_TYPE", "")).upper()
            except Exception:
                config.POL_TYPE = ""
                                                                                               
            try:
                imjd = safe_int(primary.get("STT_IMJD", 0))
                smjd = safe_float(primary.get("STT_SMJD", 0.0))
                offs = safe_float(primary.get("STT_OFFS", 0.0))
                tstart_mjd = float(imjd) + (float(smjd) + float(offs)) / 86400.0
            except Exception:
                tstart_mjd = None
                                                          
                                                          
            try:
                nsuboffs = safe_int(hdr.get("NSUBOFFS", 0))
            except Exception:
                nsuboffs = 0
                                                                            
            try:
                if "OFFS_SUB" in f["SUBINT"].columns.names:
                    offs_sub_first = safe_float(sub_data[0]["OFFS_SUB"])                                            
                                                                  
                    numrows = int((offs_sub_first - 0.5 * tsubint) / tsubint + 1e-7)
                    if numrows >= 0:
                        nsuboffs = numrows
            except Exception:
                pass
            try:
                config.NSUBOFFS = nsuboffs
            except Exception:
                pass
                                                                        
            # Both epochs are rewritten on every file, including when this one
            # carries no start time. Leaving a stale TSTART_MJD_CORR behind makes
            # every candidate of the current file inherit the previous file's
            # observation date -- silently, since mjd_utils prefers _CORR.
            if tstart_mjd is not None:
                config.TSTART_MJD = tstart_mjd
                try:
                    config.TSTART_MJD_CORR = tstart_mjd + (nsuboffs * tsubint) / 86400.0
                except Exception as e:
                    logger.warning(
                        "Could not apply the subint offset to TSTART for %s (%s); "
                        "using the uncorrected start time",
                        file_name, e,
                    )
                    config.TSTART_MJD_CORR = tstart_mjd
            else:
                logger.warning("STT_* not found in %s: absolute MJD unavailable", file_name)
                config.TSTART_MJD = None
                config.TSTART_MJD_CORR = None

            try:
                freq_temp = sub_data["DAT_FREQ"][0].astype(np.float64)
            except Exception as e:
                if config.DEBUG_FREQUENCY_ORDER:
                    logger.debug(f"[DEBUG HEADER] Error converting DAT_FREQ: {e}")
                    logger.debug("[DEBUG HEADER] Using default frequency range")
                nchan = safe_int(hdr.get("NCHAN", 512), 512)
                freq_temp = np.linspace(1000, 1500, nchan)
            
                                                
            if config.DEBUG_FREQUENCY_ORDER:
                logger.debug(f"[DEBUG HEADER] Extracted PSRFITS headers:")
                logger.debug(
                    f"[DEBUG HEADER]   TBIN (temporal resolution): {safe_float(hdr.get('TBIN')):.2e} s"
                )
                logger.debug(f"[DEBUG HEADER]   NCHAN (channels): {hdr['NCHAN']}")
                logger.debug(f"[DEBUG HEADER]   NSBLK (samples per subint): {hdr['NSBLK']}")
                logger.debug(f"[DEBUG HEADER]   NAXIS2 (number of subints): {hdr['NAXIS2']}")
                logger.debug(f"[DEBUG HEADER]   NPOL (polarizations): {hdr.get('NPOL', 'N/A')}")
                logger.debug(f"[DEBUG HEADER]   Total samples: {config.FILE_LENG}")
                if 'OBS_MODE' in hdr:
                    logger.debug(f"[DEBUG HEADER]   Observation mode: {hdr['OBS_MODE']}")
                if 'SRC_NAME' in hdr:
                    logger.debug(f"[DEBUG HEADER]   Fuente: {hdr['SRC_NAME']}")
            
                                                                                          
            if len(freq_temp) > 1:
                df = float(freq_temp[1] - freq_temp[0])
                if df < 0:
                    freq_axis_inverted = True
                    if config.DEBUG_FREQUENCY_ORDER:
                        logger.debug(f"[DEBUG HEADER] DAT_FREQ descendente → invertir banda (estilo PRESTO)")
                else:
                                                                                                                    
                                                                         
                    freq_axis_inverted = False
                    if config.DEBUG_FREQUENCY_ORDER:
                        logger.debug(f"[DEBUG HEADER] DAT_FREQ ascending → invert band (radioastronomy style)")
        else:
                                                     
            if config.DEBUG_FREQUENCY_ORDER:
                logger.debug(f"[DEBUG HEADER] Detected format: standard FITS (not PSRFITS)")
            
            try:
                data_hdu_index = 0
                for i, hdu_item in enumerate(f):
                    if hdu_item.is_image or isinstance(hdu_item, (fits.BinTableHDU, fits.TableHDU)):
                        if 'NAXIS' in hdu_item.header and hdu_item.header['NAXIS'] > 0:
                            if 'CTYPE3' in hdu_item.header and 'FREQ' in hdu_item.header['CTYPE3'].upper():
                                data_hdu_index = i
                                break
                            if 'CTYPE2' in hdu_item.header and 'FREQ' in hdu_item.header['CTYPE2'].upper():
                                data_hdu_index = i
                                break
                            if 'CTYPE1' in hdu_item.header and 'FREQ' in hdu_item.header['CTYPE1'].upper():
                                data_hdu_index = i
                                break
                if data_hdu_index == 0 and len(f) > 1:
                    data_hdu_index = 1
                
                                         
                if config.DEBUG_FREQUENCY_ORDER:
                    logger.debug(f"[DEBUG HEADER] HDU selected for data: {data_hdu_index}")
                
                hdr = f[data_hdu_index].header
                
                                              
                if config.DEBUG_FREQUENCY_ORDER:
                    logger.debug(f"[DEBUG HEADER] Standard FITS headers from HDU {data_hdu_index}:")
                    relevant_keys = ['TBIN', 'NCHAN', 'NAXIS2', 'NSBLK', 'NPOL', 'CRVAL1', 'CRVAL2', 'CRVAL3', 
                                   'CDELT1', 'CDELT2', 'CDELT3', 'CTYPE1', 'CTYPE2', 'CTYPE3']
                    for key in relevant_keys:
                        if key in hdr:
                            logger.debug(f"[DEBUG HEADER]   {key}: {hdr[key]}")
                
                if "DAT_FREQ" in f[data_hdu_index].columns.names:
                    try:
                        freq_temp = f[data_hdu_index].data["DAT_FREQ"][0].astype(np.float64)
                    except Exception as e:
                        if config.DEBUG_FREQUENCY_ORDER:
                            logger.debug(f"[DEBUG HEADER] Error converting DAT_FREQ: {e}")
                            logger.debug("[DEBUG HEADER] Using default frequency range")
                        nchan = safe_int(hdr.get("NCHAN", 512), 512)
                        freq_temp = np.linspace(1000, 1500, nchan)
                    else:
                        if config.DEBUG_FREQUENCY_ORDER:
                            logger.debug(f"[DEBUG HEADER] Frequencies extracted from DAT_FREQ column")
                else:
                    freq_axis_num = ''
                    for i in range(1, hdr.get('NAXIS', 0) + 1):
                        if 'FREQ' in hdr.get(f'CTYPE{i}', '').upper():
                            freq_axis_num = str(i)
                            break
                    
                    if config.DEBUG_FREQUENCY_ORDER:
                        logger.debug(f"[DEBUG HEADER] Searching for frequency axis in WCS headers...")
                        logger.debug(f"[DEBUG HEADER] Frequency axis detected: CTYPE{freq_axis_num}" if freq_axis_num else "[DEBUG HEADER] No frequency axis found")
                    
                    if freq_axis_num:
                        crval = hdr.get(f'CRVAL{freq_axis_num}', 0)
                        cdelt = hdr.get(f'CDELT{freq_axis_num}', 1)
                        crpix = hdr.get(f'CRPIX{freq_axis_num}', 1)
                        naxis = hdr.get(f'NAXIS{freq_axis_num}', hdr.get('NCHAN', 512))
                        try:
                            crval = float(crval)
                            cdelt = float(cdelt)
                            crpix = float(crpix)
                            naxis = int(naxis)
                        except (TypeError, ValueError):
                            crval = float(crval) if isinstance(crval, (int, float)) else 0.0
                            cdelt = float(cdelt) if isinstance(cdelt, (int, float)) else 1.0
                            crpix = float(crpix) if isinstance(crpix, (int, float)) else 1.0
                            naxis = int(naxis) if isinstance(naxis, (int, float)) else hdr.get('NCHAN', 512)
                        freq_temp = crval + (np.arange(naxis) - (crpix - 1)) * cdelt
                        
                        if config.DEBUG_FREQUENCY_ORDER:
                            logger.debug(f"[DEBUG HEADER] WCS frequency parameters:")
                            logger.debug(f"[DEBUG HEADER]   CRVAL{freq_axis_num}: {crval} (reference value)")
                            logger.debug(f"[DEBUG HEADER]   CDELT{freq_axis_num}: {cdelt} (increment per channel)")
                            logger.debug(f"[DEBUG HEADER]   CRPIX{freq_axis_num}: {crpix} (reference pixel)")
                            logger.debug(f"[DEBUG HEADER]   NAXIS{freq_axis_num}: {naxis} (number of channels)")
                        
                        if cdelt < 0:
                            freq_axis_inverted = True
                            if config.DEBUG_FREQUENCY_ORDER:
                                logger.debug(f"[DEBUG HEADER]   [WARNING] Negative CDELT - frequencies inverted!")
                        else:
                                                                                                                            
                            freq_axis_inverted = False
                            if config.DEBUG_FREQUENCY_ORDER:
                                logger.debug(f"[DEBUG HEADER]   [WARNING] Positive CDELT - inverting for radioastronomy standard!")
                    else:
                        if config.DEBUG_FREQUENCY_ORDER:
                            logger.debug(f"[DEBUG HEADER] [WARNING] Using default frequencies: 1000-1500 MHz")
                        freq_temp = np.linspace(1000, 1500, hdr.get('NCHAN', 512))
                
                                                                                
                config.TIME_RESO = safe_float(hdr.get("TBIN"))
                config.FREQ_RESO = safe_int(hdr.get("NCHAN", len(freq_temp)))
                config.FILE_LENG = safe_int(hdr.get("NAXIS2", 0)) * safe_int(hdr.get("NSBLK", 1))
                
                                                     
                if config.DEBUG_FREQUENCY_ORDER:
                    logger.debug(f"[DEBUG HEADER] Final standard FITS parameters:")
                    logger.debug(f"[DEBUG HEADER]   TIME_RESO: {config.TIME_RESO:.2e} s")
                    logger.debug(f"[DEBUG HEADER]   FREQ_RESO: {config.FREQ_RESO}")
                    logger.debug(f"[DEBUG HEADER]   FILE_LENG: {config.FILE_LENG}")
                    
            except Exception as e_std:
                if config.DEBUG_FREQUENCY_ORDER:
                    logger.debug(f"[DEBUG HEADER] [WARNING] Error processing standard FITS: {e_std}")
                    logger.debug(f"[DEBUG HEADER] Using default values...")
                logger.debug(f"Error processing standard FITS: {e_std}")
                config.TIME_RESO = 5.12e-5
                config.FREQ_RESO = 512
                config.FILE_LENG = 100000
                freq_temp = np.linspace(1000, 1500, config.FREQ_RESO)
        normalized_freq, needs_reversal = normalize_frequency_axis(freq_temp)
        if freq_axis_inverted:
                                                                                                    
            config.FREQ = normalized_freq
            config.DATA_NEEDS_REVERSAL = needs_reversal
            try:
                config.NEED_FLIPBAND = True
            except Exception:
                pass
        else:
            config.FREQ = normalized_freq
            config.DATA_NEEDS_REVERSAL = needs_reversal
            try:
                config.NEED_FLIPBAND = False
            except Exception:
                pass

                                 
    if config.DEBUG_FREQUENCY_ORDER:
        print_debug_frequencies("[DEBUG FREQUENCIES]", file_name, freq_axis_inverted)

                                             
    if config.DEBUG_FREQUENCY_ORDER:
        logger.debug(f"[DEBUG FILE] Complete file information: {file_name}")
        logger.debug(f"[DEBUG FILE] " + "="*60)
        logger.debug(f"[DEBUG FILE] DIMENSIONS AND RESOLUTION:")
        logger.debug(f"[DEBUG FILE]   - Temporal resolution: {config.TIME_RESO:.2e} seconds/sample")
        logger.debug(f"[DEBUG FILE]   - Frequency resolution: {config.FREQ_RESO} channels")
        logger.debug(f"[DEBUG FILE]   - File length: {config.FILE_LENG:,} samples")
        
                                 
        duracion_total_seg = config.FILE_LENG * config.TIME_RESO
        duracion_min = duracion_total_seg / 60
        duracion_horas = duracion_min / 60
        logger.debug(f"[DEBUG FILE]   - Total duration: {duracion_total_seg:.2f} sec ({duracion_min:.2f} min, {duracion_horas:.2f} h)")
        
        logger.debug(f"[DEBUG FILE] FREQUENCIES:")
        logger.debug(f"[DEBUG FILE]   - Total range: {config.FREQ.min():.2f} - {config.FREQ.max():.2f} MHz")
        logger.debug(f"[DEBUG FILE]   - Bandwidth: {abs(config.FREQ.max() - config.FREQ.min()):.2f} MHz")
        logger.debug(f"[DEBUG FILE]   - Resolution per channel: {abs(config.FREQ[1] - config.FREQ[0]):.4f} MHz/channel")
        logger.debug(f"[DEBUG FILE]   - Original order: {'DESCENDING' if freq_axis_inverted else 'ASCENDING'}")
        logger.debug(f"[DEBUG FILE]   - Final order (post-correction): {'ASCENDING' if config.FREQ[0] < config.FREQ[-1] else 'DESCENDING'}")
        
        logger.debug(f"[DEBUG FILE] DECIMATION:")
        logger.debug(f"[DEBUG FILE]   - Frequency reduction factor: {config.DOWN_FREQ_RATE}x")
        logger.debug(f"[DEBUG FILE]   - Time reduction factor: {config.DOWN_TIME_RATE}x")
        logger.debug(f"[DEBUG FILE]   - Channels after decimation: {config.FREQ_RESO // config.DOWN_FREQ_RATE}")
        logger.debug(f"[DEBUG FILE]   - Temporal resolution after: {config.TIME_RESO * config.DOWN_TIME_RATE:.2e} sec/sample")
        
                                             
        size_original_gb = (config.FILE_LENG * config.FREQ_RESO * 4) / (1024**3)                       
        size_decimated_gb = size_original_gb / (config.DOWN_FREQ_RATE * config.DOWN_TIME_RATE)
        logger.debug(f"[DEBUG FILE] ESTIMATED SIZE:")
        logger.debug(f"[DEBUG FILE]   - Original data: ~{size_original_gb:.2f} GB")
        logger.debug(f"[DEBUG FILE]   - Data after decimation: ~{size_decimated_gb:.2f} GB")
        
        
        logger.debug(f"[DEBUG FILE] SLICE CONFIGURATION:")
        logger.debug(f"[DEBUG FILE]   - SLICE_DURATION_MS configured: {config.SLICE_DURATION_MS} ms")
        expected_slice_len = round(config.SLICE_DURATION_MS / (config.TIME_RESO * config.DOWN_TIME_RATE * 1000))
        logger.debug(f"[DEBUG FILE]   - SLICE_LEN calculated: {expected_slice_len} samples")
        logger.debug(f"[DEBUG FILE]   - SLICE_LEN limits: [{config.SLICE_LEN_MIN}, {config.SLICE_LEN_MAX}]")
        
        logger.debug(f"[DEBUG FILE] PROCESSING:")
        logger.debug(f"[DEBUG FILE]   - Multi-band enabled: {'YES' if config.USE_MULTI_BAND else 'NO'}")
        logger.debug(f"[DEBUG FILE]   - DM range: {config.DM_min} - {config.DM_max} pc cm⁻³")
        logger.debug(f"[DEBUG FILE]   - Thresholds: DET_PROB={config.DET_PROB}, CLASS_PROB={config.CLASS_PROB}, SNR_THRESH={config.SNR_THRESH}")
        logger.debug(f"[DEBUG FILE] " + "="*60)

                                                                                    
    auto_config_downsampling()

                                              
    if config.DEBUG_FREQUENCY_ORDER:
        logger.debug(f"[DEBUG CONFIG FINAL] Final configuration after get_obparams:")
        logger.debug(f"[DEBUG CONFIG FINAL] " + "="*60)
        logger.debug(f"[DEBUG CONFIG FINAL] DOWN_FREQ_RATE calculated: {config.DOWN_FREQ_RATE}x")
        logger.debug(f"[DEBUG CONFIG FINAL] DOWN_TIME_RATE calculated: {config.DOWN_TIME_RATE}x")
        logger.debug(f"[DEBUG CONFIG FINAL] Data after decimation:")
        logger.debug(f"[DEBUG CONFIG FINAL]   - Channels: {config.FREQ_RESO // config.DOWN_FREQ_RATE}")
        logger.debug(f"[DEBUG CONFIG FINAL]   - Temporal resolution: {config.TIME_RESO * config.DOWN_TIME_RATE:.2e} s/sample")
        logger.debug(f"[DEBUG CONFIG FINAL]   - Total data reduction: {config.DOWN_FREQ_RATE * config.DOWN_TIME_RATE}x")
        logger.debug(f"[DEBUG CONFIG FINAL] Final DATA_NEEDS_REVERSAL: {config.DATA_NEEDS_REVERSAL}")
        logger.debug(f"[DEBUG CONFIG FINAL] Final frequency order: {'ASCENDING' if config.FREQ[0] < config.FREQ[-1] else 'DESCENDING'}")
        logger.debug(f"[DEBUG CONFIG FINAL] " + "="*60)

                                                               
    if config.DEBUG_FREQUENCY_ORDER:
        save_file_debug_info(file_name, {
            "file_type": "fits",
            "file_size_bytes": os.path.getsize(file_name),
            "file_size_gb": os.path.getsize(file_name) / (1024**3),
            "format": "PSRFITS (.fits)",
            "frequency_analysis": {
                "freq_min_mhz": float(config.FREQ.min()),
                "freq_max_mhz": float(config.FREQ.max()),
                "bandwidth_mhz": abs(config.FREQ.max() - config.FREQ.min()),
                "resolution_per_channel_mhz": abs(config.FREQ[1] - config.FREQ[0]) if len(config.FREQ) > 1 else 0,
                "original_order": "DESCENDING" if freq_axis_inverted else "ASCENDING",
                "final_order": "ASCENDING" if config.FREQ[0] < config.FREQ[-1] else "DESCENDING",
                "freq_axis_inverted": freq_axis_inverted,
                "data_needs_reversal": config.DATA_NEEDS_REVERSAL
            },
            "time_analysis": {
                "time_resolution_sec": config.TIME_RESO,
                "total_samples": config.FILE_LENG,
                "total_duration_sec": config.FILE_LENG * config.TIME_RESO,
                "total_duration_min": (config.FILE_LENG * config.TIME_RESO) / 60,
                "total_duration_hours": (config.FILE_LENG * config.TIME_RESO) / 3600
            },
            "decimation": {
                "down_freq_rate": config.DOWN_FREQ_RATE,
                "down_time_rate": config.DOWN_TIME_RATE,
                "channels_after_decimation": config.FREQ_RESO // config.DOWN_FREQ_RATE,
                "time_resolution_after_decimation_sec": config.TIME_RESO * config.DOWN_TIME_RATE,
                "total_reduction_factor": config.DOWN_FREQ_RATE * config.DOWN_TIME_RATE
            },
            "slice_config": {
                "slice_duration_ms_configured": config.SLICE_DURATION_MS,
                "slice_len_calculated": round(config.SLICE_DURATION_MS / (config.TIME_RESO * config.DOWN_TIME_RATE * 1000)),
                "slice_len_limits": [config.SLICE_LEN_MIN, config.SLICE_LEN_MAX]
            },
            "processing_config": {
                "multi_band_enabled": config.USE_MULTI_BAND,
                "dm_range": [config.DM_min, config.DM_max],
                "detection_thresholds": {
                    "det_prob": config.DET_PROB,
                    "class_prob": config.CLASS_PROB,
                    "snr_thresh": config.SNR_THRESH
                }
            },
            "file_temporal_info": {
                "total_duration_sec": config.FILE_LENG * config.TIME_RESO,
                "total_duration_formatted": f"{(config.FILE_LENG * config.TIME_RESO) // 3600:.0f}h {((config.FILE_LENG * config.TIME_RESO) % 3600) // 60:.0f}m {(config.FILE_LENG * config.TIME_RESO) % 60:.1f}s",
                "sample_rate_hz": 1.0 / config.TIME_RESO,
                "effective_sample_rate_after_decimation_hz": 1.0 / (config.TIME_RESO * config.DOWN_TIME_RATE)
            }
        })


def stream_fits_multi_pol(
    file_name: str,
    chunk_samples: int = 2_097_152,
    overlap_samples: int = 0,
) -> Generator[Tuple[np.ndarray, np.ndarray, Dict, str], None, None]:
    """
    Generator for high-frequency pipeline that preserves multi-polarization data.
    
    Args:
        file_name: Path to .fits file
        chunk_samples: Number of samples per block (default: 2M)
        overlap_samples: Number of overlap samples between blocks
    
    Yields:
        Tuple[data_block, raw_block, metadata, pol_type]:
            - data_block: Block with selected polarization (time, 1, chan)
            - raw_block: Block with ALL polarizations (time, npol, chan)
            - metadata: Chunk metadata dictionary
            - pol_type: Polarization type from header (e.g., "IQUV")
    """
    try:
        logger.info("Streaming FITS data (multi-pol mode): chunk_size=%d, overlap=%d", chunk_samples, overlap_samples)
        
        # Try using 'your' library first
        try:
            if your_psrfits is not None:
                pf = your_psrfits.PsrfitsFile([file_name])
                nspec = int(pf.nspectra())
                npol = int(pf.npol)
                nchan = int(pf.nchans)
                tsamp = float(pf.native_tsamp())
                pol_type = getattr(pf, 'pol_type', 'IQUV') if hasattr(pf, 'pol_type') else 'IQUV'
                
                logger.info(
                    "Streaming PSRFITS ('your'): nspec=%d, npol=%d, nchan=%d, pol_type=%s",
                    nspec,
                    npol,
                    nchan,
                    pol_type,
                )
                
                chunk_counter = 0
                emitted = 0
                step = chunk_samples
                
                while emitted < nspec:
                    start = emitted
                    read_start = max(0, start - overlap_samples)
                    read_end = min(nspec, start + step + overlap_samples)
                    count = read_end - read_start
                    
                    # Get RAW data with ALL polarizations
                    arr_raw = pf.get_data(read_start, count, npoln=npol)
                    if arr_raw.ndim != 3:
                        raise ValueError("Unexpected shape in 'your' get_data")
                    
                    # Reverse frequency if needed. Same criterion as every other
                    # reader: DATA_NEEDS_REVERSAL means the file stores channels
                    # descending, and config.FREQ is always ascending. `your`
                    # returns channels in the file's native DAT_FREQ order -- it
                    # normalises nothing (need_flipband is hardcoded False and the
                    # flip is commented out in your/formats/psrfits.py). This used
                    # to test `foff > 0`, the opposite condition (audit P1-02).
                    try:
                        if getattr(config, 'DATA_NEEDS_REVERSAL', False):
                            arr_raw = arr_raw[:, :, ::-1]
                    except Exception:
                        pass
                    
                    if arr_raw.dtype != np.float32:
                        arr_raw = arr_raw.astype(np.float32)
                    
                    # Create the selected polarization block (for compatibility)
                    from .polarization_utils import extract_polarization_from_raw
                    block_selected = extract_polarization_from_raw(
                        arr_raw, pol_type,
                        getattr(config, 'POLARIZATION_MODE', 'intensity'),
                        getattr(config, 'POLARIZATION_INDEX', 0)
                    )
                    
                    chunk_counter += 1
                    valid_start = start
                    valid_end = min(start + step, nspec)
                    start_with_overlap = read_start
                    end_with_overlap = read_end
                    
                    metadata = {
                        "chunk_idx": valid_start // chunk_samples,
                        "start_sample": valid_start,
                        "end_sample": valid_end,
                        "actual_chunk_size": valid_end - valid_start,
                        "block_start_sample": start_with_overlap,
                        "block_end_sample": end_with_overlap,
                        "overlap_left": valid_start - start_with_overlap,
                        "overlap_right": end_with_overlap - valid_end,
                        "total_samples": nspec,
                        "nchans": nchan,
                        "npol": npol,
                        "nifs": 1,
                        "dtype": str(block_selected.dtype),
                        "shape": block_selected.shape,
                        "file_type": "fits",
                        "tbin_sec": tsamp,
                        "t_rel_start_sec": valid_start * tsamp,
                        "t_rel_end_sec": valid_end * tsamp,
                    }
                    
                    yield block_selected, arr_raw, metadata, pol_type
                    emitted += step
                
                return
        
        except Exception as e:
            logger.debug("'your' library streaming failed (%s), falling back", e)
        
        raise RuntimeError(
            "Multi-polarisation streaming requires the 'your_psrfits' library. "
            "Install it or use the standard single-polarisation pipeline via stream_fits()."
        )
        
    except Exception as e:
        logger.error("Error in multi-pol streaming: %s", e)
        raise


def _row_to_block(
    row_data: np.ndarray,
    row: np.ndarray,
    subint,
    nbits: int,
    nsblk: int,
    npol: int,
    nchan: int,
    zero_off: float,
    pol_type: str,
) -> np.ndarray:
    """One SUBINT row to a ``(nsblk, 1, nchan)`` float32 block.

    Unpacks sub-byte samples, applies the PSRFITS calibration columns and picks
    the polarisation. Both astropy readers call this; it was a closure inside
    ``stream_fits`` and its body is unchanged.
    """

    if nbits < 8:
        if nbits == 4:
            unpacked = _unpack_4bit(row_data)
        elif nbits == 2:
            unpacked = _unpack_2bit(row_data)
        elif nbits == 1:
            unpacked = _unpack_1bit(row_data)
        else:
            raise ValueError(f"NBITS={nbits} not supported")
        try:
            tmpb = unpacked.reshape(nsblk, npol, nchan)
        except Exception:
            tmpb = unpacked.reshape(nsblk, nchan, npol).swapaxes(1, 2)
        tmpb = tmpb.astype(np.float32, copy=False)
    else:
        arrb = np.asarray(row_data)
        try:
            tmpb = arrb.reshape(nsblk, npol, nchan)
        except Exception:
            tmpb = arrb.reshape(nsblk, nchan, npol).swapaxes(1, 2)
        if tmpb.dtype != np.float32:
            tmpb = tmpb.astype(np.float32, copy=False)


    dat_wts_b = row["DAT_WTS"].astype(np.float32, copy=False) if "DAT_WTS" in subint.columns.names else None
    dat_scl_b = row["DAT_SCL"].astype(np.float32, copy=False) if "DAT_SCL" in subint.columns.names else None
    dat_offs_b = row["DAT_OFFS"].astype(np.float32, copy=False) if "DAT_OFFS" in subint.columns.names else None
    tmpb = _apply_calibration(tmpb, dat_wts_b, dat_scl_b, dat_offs_b, zero_off)

    sel = _select_polarization(tmpb, pol_type, getattr(config, 'POLARIZATION_MODE', 'intensity'), getattr(config, 'POLARIZATION_INDEX', 0))
    return sel


def _buffer_limits_for(chunk_samples: int, overlap_samples: int, nchan: int) -> BufferLimits:
    """Memory-derived buffer ceiling, and the two log lines that announce it.

    The numbers come from :func:`compute_buffer_limits`; only the reading of the
    machine's free RAM happens here.
    """
    import psutil
    vm = psutil.virtual_memory()
    available_ram_gb = vm.available / (1024**3)
    limits = compute_buffer_limits(chunk_samples, overlap_samples, nchan, available_ram_gb)

    if limits.large_chunk:
        logger.warning(
            f"Large chunk detected ({chunk_samples:,} samples). "
            f"Using conservative buffer limit: {limits.max_buffer_samples:,} samples "
            f"({limits.max_buffer_gb:.2f} GB) to prevent system freeze."
        )
    logger.info(
        f"Buffer limits: max_samples={limits.max_buffer_samples:,} "
        f"({limits.max_buffer_samples * limits.bytes_per_sample / (1024**3):.2f} GB), "
        f"max_blocks={limits.max_buffer_blocks}, chunk_samples={chunk_samples:,}"
    )
    return limits


# --------------------------------------------------------------------------- #
# the `your` library reader
# --------------------------------------------------------------------------- #
class _YourSource:
    """An open PSRFITS handle from the ``your`` library, plus its scalars."""

    def __init__(self, pf, nspec: int, npol: int, nchan: int, tsamp: float) -> None:
        self.pf = pf
        self.nspec = nspec
        self.npol = npol
        self.nchan = nchan
        self.tsamp = tsamp

    def close(self) -> None:
        """``your`` hands back no handle of its own to release."""
        return None


def _open_your_source(file_name: str, chunk_samples: int, overlap_samples: int) -> _YourSource:
    """Open *file_name* with ``your`` and read its scalars. May raise.

    Everything that can tell us this reader will not work happens here, before
    any block exists: ``SpectraInfo`` indexes ``STT_IMJD`` with no default, so a
    file without an epoch fails at construction (divergence D6).
    """
    pf = your_psrfits.PsrfitsFile([file_name])
    nspec = int(pf.nspectra())
    npol = int(pf.npol)
    nchan = int(pf.nchans)
    tsamp = float(pf.native_tsamp())
    logger.info(
        "Streaming PSRFITS ('your'): nspec=%d, npol=%d, nchan=%d, tsamp=%s",
        nspec,
        npol,
        nchan,
        tsamp,
    )
    log_stream_fits_parameters(nspec, chunk_samples, overlap_samples, None, nchan, npol, None)
    return _YourSource(pf, nspec, npol, nchan, tsamp)


def _emit_your_blocks(
    source: _YourSource, chunk_samples: int, overlap_samples: int
) -> Generator[Tuple[np.ndarray, Dict], None, None]:
    """Seek-and-read: the only reader whose valid windows tile the file exactly.

    Divergences preserved deliberately (audit REF-03):
      D1  this tiles; the two buffered astropy readers do not.
      D3  no ``ZERO_OFF`` subtraction -- ``your`` never reads that card and this
          branch applies no calibration of its own.
      D4  ``NSUBOFFS`` is ignored, so a continuation file streams as written.
      D5  ``getattr(pf, 'pol_type', 'IQUV')`` -- ``PsrfitsFile`` has no such
          attribute (it is ``poln_order``), so the default always wins.
      It also publishes no epoch keys at all, unlike the astropy readers.
    """
    pf = source.pf
    nspec, npol, nchan, tsamp = source.nspec, source.npol, source.nchan, source.tsamp

    chunk_counter = 0
    emitted = 0

    step = chunk_samples
    try:
        while emitted < nspec:
            span = plan_sequential_span(emitted, step, overlap_samples, nspec)
            count = span.block_end_sample - span.block_start_sample
            arr = pf.get_data(span.block_start_sample, count, npoln=npol)
            if arr.ndim != 3:
                raise ValueError("Unexpected shape in 'your' get_data")

            block = _select_polarization(arr, getattr(pf, 'pol_type', 'IQUV'), getattr(config, 'POLARIZATION_MODE', 'intensity'), getattr(config, 'POLARIZATION_INDEX', 0))
            # RESOLVED (audit P1-02). This branch used to reverse on
            # foff > 0, the opposite of every other reader path, which
            # reverses on config.DATA_NEEDS_REVERSAL -- set when DAT_FREQ
            # is DESCENDING. Only one criterion can leave the channel axis
            # matching the ascending config.FREQ that dedispersion indexes
            # against, and it is DATA_NEEDS_REVERSAL:
            #
            #   - your/formats/psrfits.py returns channels in the file's
            #     native DAT_FREQ order. It normalises nothing: the flip is
            #     commented out at its lines 447-453 and need_flipband is
            #     hardcoded False at 485. Measured on synthetic PSRFITS in
            #     both orientations; blimpy agrees through a round trip.
            #   - A burst injected at DM 500 into a descending-axis file
            #     recovers at exactly DM 500 under DATA_NEEDS_REVERSAL, and
            #     at DM 748 with S/N 10.4 under foff > 0. The failure mode
            #     is not a missed burst -- it is a burst with a fabricated
            #     DM that still clears the detection threshold.
            #
            # The mismatch warning below is kept as a live detector: it now
            # fires when `your` would disagree with the rule we follow, on
            # a file whose header we have not seen before.
            # Anomaly detector, and it has to compare two things that
            # normally AGREE. The first version compared `foff > 0`
            # against DATA_NEEDS_REVERSAL, which are opposite by
            # construction -- descending DAT_FREQ means foff < 0 and
            # DATA_NEEDS_REVERSAL True -- so it fired once per chunk on
            # every ordinary file: constant noise, not a signal.
            #
            # What is worth reporting is a header that contradicts
            # itself: foff's sign disagreeing with the ordering
            # normalize_frequency_axis derived from DAT_FREQ.
            _foff = getattr(pf, 'foff', 0.0)
            _rest_says_reverse = bool(getattr(config, 'DATA_NEEDS_REVERSAL', False))
            _foff_says_descending = bool(_foff < 0)
            if _foff and _foff_says_descending != _rest_says_reverse:
                logger.warning(
                    "FREQ-ORDER ANOMALY (audit P1-02): foff=%.6f implies %s channels "
                    "but DAT_FREQ was read as %s. The file's own header disagrees with "
                    "itself; this reader follows DAT_FREQ. Check candidates from this "
                    "file against a known pulsar.",
                    _foff,
                    "descending" if _foff_says_descending else "ascending",
                    "descending" if _rest_says_reverse else "ascending",
                )
            try:
                if _rest_says_reverse:
                    block = block[:, :, ::-1]
            except Exception:
                pass
            if block.dtype != np.float32:
                block = block.astype(np.float32)

            chunk_counter += 1
            log_stream_fits_block_generation(
                chunk_counter,
                block.shape,
                str(block.dtype),
                span.start_sample,
                span.end_sample,
                span.block_start_sample,
                span.block_end_sample,
                span.actual_chunk_size,
            )
            metadata = chunk_metadata(
                span,
                chunk_samples=chunk_samples,
                total_samples=nspec,
                nchans=nchan,
                nifs=1,
                block=block,
                tbin_sec=tsamp,
            )
            yield block, metadata
            emitted += step

        log_stream_fits_summary(chunk_counter)
    finally:
        source.close()


# --------------------------------------------------------------------------- #
# the astropy SUBINT readers
# --------------------------------------------------------------------------- #
class _SubintSource:
    """An open SUBINT table and every scalar the emitter needs.

    Two of the fields record which of the two conventions this source was built
    with, because the primary and the fallback readers disagree and the audit
    decided to keep both until each can be fixed on purpose:

    ``tstart_mjd`` comes from ``STT_IMJD/STT_SMJD/STT_OFFS`` in the primary and
    from a non-standard ``TSTART`` card in the fallback (divergence D2), and
    ``position_from_offs_sub`` says whether the emitter places subints by
    ``OFFS_SUB`` -- absolutely, zero-padding continuation files -- or simply at
    ``index * NSBLK`` (divergence D4).
    """

    def __init__(
        self,
        *,
        hdul,
        subint,
        tbl,
        nsubint: int,
        nchan: int,
        npol: int,
        nsblk: int,
        nbits: int,
        zero_off: float,
        tbin: float,
        tsub: float,
        pol_type: str,
        tstart_mjd: float,
        nsuboffs: int,
        has_offs_sub: bool,
        total_samples: int,
        limits: BufferLimits,
        position_from_offs_sub: bool,
    ) -> None:
        self.hdul = hdul
        self.subint = subint
        self.tbl = tbl
        self.nsubint = nsubint
        self.nchan = nchan
        self.npol = npol
        self.nsblk = nsblk
        self.nbits = nbits
        self.zero_off = zero_off
        self.tbin = tbin
        self.tsub = tsub
        self.pol_type = pol_type
        self.tstart_mjd = tstart_mjd
        self.nsuboffs = nsuboffs
        self.has_offs_sub = has_offs_sub
        self.total_samples = total_samples
        self.limits = limits
        self.position_from_offs_sub = position_from_offs_sub

    @property
    def tstart_mjd_corr(self) -> float:
        return self.tstart_mjd + (self.nsuboffs * self.tsub) / 86400.0

    def close(self) -> None:
        try:
            self.hdul.close()
        except Exception:
            pass


def _open_subint_table(file_name: str):
    """``(hdul, subint, tbl)`` for a PSRFITS SUBINT file, or ``(hdul, None, None)``.

    The caller owns ``hdul`` and must close it. A truncated file raises here,
    before anything has been emitted.
    """
    hdul = fits.open(file_name, memmap=True)
    try:
        if not ("SUBINT" in [hdu.name for hdu in hdul] and "DATA" in hdul["SUBINT"].columns.names):
            return hdul, None, None
        subint = hdul["SUBINT"]

        try:
            tbl = subint.data
        except (TypeError, ValueError, OSError) as e:
            if "buffer is too small" in str(e) or "truncated" in str(e).lower():
                logger.warning(
                    "Truncated FITS file detected (%s): %s. Skipping.",
                    file_name,
                    e,
                )
                raise ValueError(f"Truncated FITS file: {file_name}") from e
            else:
                raise
        return hdul, subint, tbl
    except Exception:
        hdul.close()
        raise


def _read_subint_scalars(subint) -> Dict[str, Any]:
    """The header scalars both astropy readers read, read the same way once."""
    hdr = subint.header
    nsblk = safe_int(hdr.get("NSBLK", 1))
    tbin = safe_float(hdr.get("TBIN"))
    return {
        "nsubint": safe_int(hdr.get("NAXIS2", 0)),
        "nchan": safe_int(hdr.get("NCHAN", 0)),
        "npol": safe_int(hdr.get("NPOL", 0)),
        "nsblk": nsblk,
        "nbits": safe_int(hdr.get("NBITS", 8)),
        "zero_off": safe_float(hdr.get("ZERO_OFF", 0.0)),
        "tbin": tbin,

        "tsub": safe_float(hdr.get("TSUBINT", nsblk * tbin)),
        "pol_type": str(hdr.get("POL_TYPE", "")).upper() if hdr.get("POL_TYPE") is not None else "",
    }


def _open_subint_primary(
    file_name: str, chunk_samples: int, overlap_samples: int
) -> Optional[_SubintSource]:
    """Open the file for the primary astropy reader, or ``None`` if not SUBINT.

    Reads the epoch the way PSRFITS stores it -- ``STT_IMJD``, ``STT_SMJD``,
    ``STT_OFFS`` in PRIMARY and ``NSUBOFFS`` in SUBINT, refined by the first
    ``OFFS_SUB``. Raises on anything that makes the reader unusable, which is
    what lets the caller still choose the fallback.
    """
    hdul, subint, tbl = _open_subint_table(file_name)
    if subint is None:
        hdul.close()
        return None
    try:
        scalars = _read_subint_scalars(subint)
        hdr = subint.header
        nsblk, tsub = scalars["nsblk"], scalars["tsub"]
        nchan, npol, nsubint = scalars["nchan"], scalars["npol"], scalars["nsubint"]


        primary = hdul["PRIMARY"].header if "PRIMARY" in [h.name for h in hdul] else {}
        imjd = safe_int(primary.get("STT_IMJD", 0))
        smjd = safe_float(primary.get("STT_SMJD", 0.0))
        offs = safe_float(primary.get("STT_OFFS", 0.0))
        tstart_mjd = float(imjd) + (float(smjd) + float(offs)) / 86400.0

        nsuboffs = safe_int(hdr.get("NSUBOFFS", 0))
        has_offs_sub = "OFFS_SUB" in subint.columns.names
        if has_offs_sub:
            try:
                offs_sub_first = safe_float(tbl[0]["OFFS_SUB"])
                numrows = int((offs_sub_first - 0.5 * tsub) / tsub + 1e-7)
                if numrows >= 0:
                    nsuboffs = numrows
            except Exception:
                pass

        # Assign TSTART_MJD to config for MJD calculations in candidate detection
        if tstart_mjd > 0:
            config.TSTART_MJD = tstart_mjd
            config.TSTART_MJD_CORR = tstart_mjd + (nsuboffs * tsub) / 86400.0
        else:
            logger.warning("TSTART_MJD not calculated correctly from STT_IMJD/STT_SMJD/STT_OFFS")
            config.TSTART_MJD = None
            config.TSTART_MJD_CORR = None

        total_samples = nsubint * nsblk
        logger.info(
            "FITS data detected: samples=%d, polarisations=%d, channels=%d",
            total_samples,
            npol,
            nchan,
        )
        log_stream_fits_parameters(total_samples, chunk_samples, overlap_samples, nsubint, nchan, npol, nsblk)
        logger.info(
            f"[STREAMING] Starting subint processing: {nsubint:,} subints, "
            f"chunk_samples={chunk_samples:,}, overlap={overlap_samples:,} samples"
        )
        limits = _buffer_limits_for(chunk_samples, overlap_samples, nchan)

        return _SubintSource(
            hdul=hdul, subint=subint, tbl=tbl,
            tstart_mjd=tstart_mjd, nsuboffs=nsuboffs, has_offs_sub=has_offs_sub,
            total_samples=total_samples, limits=limits,
            position_from_offs_sub=True,
            **scalars,
        )
    except Exception:
        hdul.close()
        raise


def _open_subint_fallback(
    file_name: str, chunk_samples: int, overlap_samples: int
) -> Optional[_SubintSource]:
    """Open the file for the fallback astropy reader, or ``None`` if not SUBINT.

    PINNED AS-IS, and it is wrong (divergence D2). This copy reads a ``TSTART``
    card from PRIMARY -- not a PSRFITS keyword, and absent from a standard file
    -- plus ``NSUBOFFS`` from PRIMARY rather than from SUBINT, so on an ordinary
    file it reports MJD 0.0 and NULLS ``config.TSTART_MJD``. Every candidate's
    absolute date is computed from that. Fixing it would change the values
    ``tests/test_fits_reader_characterization.py`` pins, so it is left alone
    here and reported instead.
    """
    hdul, subint, tbl = _open_subint_table(file_name)
    if subint is None:
        hdul.close()
        return None
    try:
        scalars = _read_subint_scalars(subint)
        nsblk, tsub = scalars["nsblk"], scalars["tsub"]
        nchan, npol, nsubint, tbin = scalars["nchan"], scalars["npol"], scalars["nsubint"], scalars["tbin"]


        primary = hdul["PRIMARY"].header if "PRIMARY" in [h.name for h in hdul] else {}
        tstart_mjd = safe_float(primary.get("TSTART", 0.0))
        nsuboffs = safe_int(primary.get("NSUBOFFS", 0))

        # Assign TSTART_MJD to config for MJD calculations in candidate detection
        if tstart_mjd > 0:
            config.TSTART_MJD = tstart_mjd
            config.TSTART_MJD_CORR = tstart_mjd + (nsuboffs * tsub) / 86400.0
        else:
            logger.warning("TSTART not found or zero in FITS PRIMARY header")
            config.TSTART_MJD = None
            config.TSTART_MJD_CORR = None

        logger.info(
            "Streaming PSRFITS (astropy fallback): nsubint=%d, nchan=%d, npol=%d, tbin=%s",
            nsubint,
            nchan,
            npol,
            tbin,
        )
        logger.info(
            "NOTE: Using astropy fallback for PSRFITS. This is slower but more compatible. "
            f"For large files ({nsubint:,} subints), this may take several minutes. "
            "Progress will be logged every 5 seconds during streaming."
        )
        total_samples = nsubint * nsblk
        logger.info(
            f"[STREAMING] Starting subint processing (astropy fallback): {nsubint:,} subints, "
            f"chunk_samples={chunk_samples:,}, overlap={overlap_samples:,} samples"
        )
        log_stream_fits_parameters(total_samples, chunk_samples, overlap_samples, None, nchan, npol, None)
        limits = _buffer_limits_for(chunk_samples, overlap_samples, nchan)
        logger.info(
            f"Expected processing time: ~{nsubint * 0.01:.1f}s (estimated ~0.01s per subint). "
            f"Large files may take longer."
        )

        return _SubintSource(
            hdul=hdul, subint=subint, tbl=tbl,
            tstart_mjd=tstart_mjd, nsuboffs=nsuboffs,
            has_offs_sub="OFFS_SUB" in subint.columns.names,
            total_samples=total_samples, limits=limits,
            position_from_offs_sub=False,
            **scalars,
        )
    except Exception:
        hdul.close()
        raise


def _emit_subint_primary_blocks(
    source: _SubintSource, chunk_samples: int, overlap_samples: int
) -> Generator[Tuple[np.ndarray, Dict], None, None]:
    """The buffered astropy reader taken when ``your`` is not installed.

    Divergences preserved deliberately (audit REF-03):
      D1  the first-chunk clamp below pulls ``valid_start`` back to 0 at the
          start of the file without pulling back how far the buffer then
          advances, so with an overlap the valid windows come out
          ``[0,128) [144,272) [272,400) [384,512)``: samples 128..143 are in no
          window and 384..399 are in two. A live defect of the P1-04/05/06
          family, pinned by ``TestValidWindowsTile``.
      D3  ``ZERO_OFF`` IS subtracted here, via ``_apply_calibration``.
      D4  subints are placed by ``OFFS_SUB`` read as an absolute offset, so a
          continuation file is zero-padded at the front and more samples are
          emitted than the file holds.

    Near-identical to :func:`_emit_subint_fallback_blocks`. They are not merged
    because ``tests/test_p1_regressions.py::TestFitsChunkGeometry`` asserts on
    the literal source text of four expressions below and requires exactly two
    copies of each; see the REF-03 report.
    """
    subint, tbl = source.subint, source.tbl
    nsubint, nchan, npol, nsblk = source.nsubint, source.nchan, source.npol, source.nsblk
    nbits, zero_off, tbin, tsub = source.nbits, source.zero_off, source.tbin, source.tsub
    pol_type, nsuboffs, tstart_mjd = source.pol_type, source.nsuboffs, source.tstart_mjd
    total_samples = source.total_samples
    limits = source.limits
    max_buffer_samples = limits.max_buffer_samples
    max_buffer_blocks = limits.max_buffer_blocks
    bytes_per_sample = limits.bytes_per_sample

    buffer = SubintBuffer(nchan)
    emitted = 0
    chunk_counter = 0

    # Progress tracking
    last_progress_log = time.time()
    last_progress_row = 0
    progress_interval = 5.0  # Log every 5 seconds

    # OPTIMIZATION: Check buffer less frequently (every 10 subints instead of every 1)
    buffer_check_interval = 10

    try:
        for isub in range(nsubint):
            row = tbl[isub]
            offs_sub_val = None
            if source.has_offs_sub:
                try:
                    offs_sub_val = safe_float(row["OFFS_SUB"])
                except Exception:
                    offs_sub_val = None

            expected_start = expected_start_sample(
                offs_sub_val, isub, tsub=tsub, tbin=tbin, nsuboffs=nsuboffs, nsblk=nsblk,
            )

            if emitted < expected_start:
                buffer.pad(expected_start - emitted)
                emitted = expected_start

            block = _row_to_block(row["DATA"], row, subint, nbits, nsblk, npol, nchan, zero_off, pol_type)
            buffer.append(block)
            emitted += block.shape[0]

            # OPTIMIZATION: Only check time every N subints (reduces overhead)
            # Progress logging every 5 seconds
            if isub % buffer_check_interval == 0:
                current_time = time.time()
                if current_time - last_progress_log >= progress_interval:
                    progress_pct = (isub + 1) / nsubint * 100
                    rows_processed = isub + 1 - last_progress_row
                    rate = rows_processed / (current_time - last_progress_log)
                    remaining = (nsubint - isub - 1) / max(rate, 0.001)

                    logger.info(
                        f"Streaming progress: {isub+1:,}/{nsubint:,} subints ({progress_pct:.1f}%) | "
                        f"Buffer: {buffer.n_blocks:,} blocks, {buffer.total_samples:,} samples | "
                        f"Rate: {rate:.1f} subints/s | ETA: {remaining:.1f}s"
                    )

                    last_progress_log = current_time
                    last_progress_row = isub + 1

            # CRITICAL: Check buffer more aggressively for large chunks
            check_frequency = 1 if chunk_samples > 1_000_000 else buffer_check_interval

            if isub % check_frequency == 0 or isub == nsubint - 1:
                needs_chunk_emission = (
                    buffer.total_samples >= (chunk_samples + overlap_samples * 2) or
                    buffer.total_samples > max_buffer_samples or
                    buffer.n_blocks > max_buffer_blocks
                )
                if buffer.total_samples > max_buffer_samples * 0.8:
                    logger.warning(
                        f"Buffer approaching limit: {buffer.total_samples:,}/{max_buffer_samples:,} samples "
                        f"({buffer.n_blocks:,} blocks). Will emit chunk soon to prevent OOM."
                    )
            else:
                needs_chunk_emission = False

            # Only concatenate when we actually need to emit a chunk
            if needs_chunk_emission:
                if buffer.n_blocks > 100:
                    logger.warning(
                        f"Concatenating large buffer: {buffer.n_blocks:,} blocks, "
                        f"{buffer.total_samples:,} samples (~{buffer.total_samples * bytes_per_sample / (1024**3):.2f} GB). "
                        f"This may take several seconds. Consider reducing chunk size for better performance."
                    )
                elif buffer.n_blocks > 50:
                    logger.info(
                        f"Preparing chunk emission at subint {isub+1:,}/{nsubint:,}: "
                        f"{buffer.n_blocks:,} blocks, {buffer.total_samples:,} samples"
                    )

                concat_start = time.time()
                out_buf = buffer.concatenate()
                concat_time = time.time() - concat_start
                if concat_time > 2.0:
                    logger.warning(
                        f"Buffer concatenation took {concat_time:.2f}s for {buffer.n_blocks:,} blocks. "
                        f"This is slow and may cause system freeze. Consider reducing chunk size."
                    )

                buffer_too_large = out_buf.shape[0] > max_buffer_samples
                has_complete_chunk = out_buf.shape[0] >= (chunk_samples + overlap_samples * 2)
            else:
                # Don't concatenate yet - continue accumulating
                buffer_too_large = False
                has_complete_chunk = False
                out_buf = None  # Not computed yet

            # Emit chunk if we have a complete chunk OR if buffer is too large
            while needs_chunk_emission and (has_complete_chunk or (buffer_too_large and out_buf is not None and out_buf.shape[0] >= chunk_samples)):
                chunk_counter += 1

                # Ensure out_buf is concatenated before extracting chunk
                if out_buf is None:
                    out_buf = buffer.concatenate()

                window = plan_buffered_window(
                    out_buf.shape[0], chunk_samples, overlap_samples,
                    buffer_too_large=buffer_too_large, large_chunk=limits.large_chunk,
                )
                start_with_overlap = window.start_with_overlap
                end_with_overlap = window.end_with_overlap
                valid_start = window.valid_start
                valid_end = window.valid_end
                actual_chunk_size = window.actual_chunk_size

                if window.emergency:
                    if actual_chunk_size < chunk_samples:
                        logger.warning(
                            f"Buffer too large ({out_buf.shape[0]:,} samples, {out_buf.shape[0] * bytes_per_sample / (1024**3):.2f} GB). "
                            f"Emitting partial chunk of {actual_chunk_size:,} samples "
                            f"(requested: {chunk_samples:,}) to prevent system freeze. "
                            f"Buffer limit: {max_buffer_samples:,} samples."
                        )
                    else:
                        logger.warning(
                            f"Buffer too large ({out_buf.shape[0]:,} samples), "
                            f"emitting emergency chunk of {actual_chunk_size:,} samples "
                            f"(buffer limit: {max_buffer_samples:,})"
                        )

                # At the very beginning of the file there is no
                # preceding data, so nothing can be left overlap:
                # keeping valid_start > 0 here discards the first
                # overlap_samples of every FITS file. stream_fil
                # already clamps this via max(0, valid_start - overlap).
                if emitted - out_buf.shape[0] <= 0 and valid_start > 0:
                    valid_start = 0
                    valid_end = valid_start + actual_chunk_size

                block_out = out_buf[start_with_overlap:end_with_overlap].copy()

                # This path performed no reversal at all, while the
                # duplicated copy of it below (reached only through
                # `except Exception`) did. Any install without the
                # `your` library streamed descending PSRFITS with the
                # channel axis untouched (audit P1-02, third instance).
                if config.DATA_NEEDS_REVERSAL:
                    block_out = block_out[:, :, ::-1]

                start_sample_idx = emitted - out_buf.shape[0] + valid_start
                end_sample_idx = start_sample_idx + actual_chunk_size

                log_stream_fits_block_generation(
                    chunk_counter,
                    block_out.shape,
                    str(block_out.dtype),
                    start_sample_idx,
                    end_sample_idx,
                    start_with_overlap,
                    end_with_overlap,
                    chunk_samples,
                )
                metadata = {
                    "chunk_idx": start_sample_idx // chunk_samples,
                    "start_sample": start_sample_idx,
                    "end_sample": end_sample_idx,
                    "actual_chunk_size": actual_chunk_size,
                    "block_start_sample": emitted - out_buf.shape[0],
                    "block_end_sample": emitted - out_buf.shape[0] + end_with_overlap,
                    # Derived from the real geometry, not assumed:
                    # the first chunk of a file has no left overlap
                    # and the last one has no right overlap. Hardcoding
                    # overlap_samples made _process_block trim data
                    # that was never overlap.
                    "overlap_left": valid_start - start_with_overlap,
                    "overlap_right": max(0, end_with_overlap - valid_end),
                    "total_samples": total_samples,
                    "nchans": nchan,
                    "nifs": 1,
                    "dtype": str(block_out.dtype),
                    "shape": block_out.shape,
                    "file_type": "fits",

                    "tbin_sec": tbin,
                    "t_rel_start_sec": start_sample_idx * tbin,
                    "t_rel_end_sec": end_sample_idx * tbin,

                    "tstart_mjd": tstart_mjd,
                    "tstart_mjd_corr": source.tstart_mjd_corr,
                    "tsubint_sec": tsub,
                }
                yield block_out, metadata

                # Remove emitted chunk from buffer (keep overlap for next chunk).
                # Always advance by exactly the valid span. Dropping
                # end_with_overlap (= actual + 2*overlap) in the
                # emergency path left a 2*overlap hole between
                # consecutive chunks that was never searched.
                samples_to_remove = actual_chunk_size
                buffer.advance(out_buf, samples_to_remove)

                # Update buffer size check for next iteration
                if buffer:
                    out_buf = buffer.concatenate()
                    buffer_too_large = out_buf.shape[0] > max_buffer_samples
                    has_complete_chunk = out_buf.shape[0] >= (chunk_samples + overlap_samples * 2)
                else:
                    out_buf = None
                    buffer_too_large = False
                    has_complete_chunk = False

                # Recalculate needs_chunk_emission for next iteration
                needs_chunk_emission = (
                    buffer.total_samples >= (chunk_samples + overlap_samples * 2) or
                    buffer.total_samples > max_buffer_samples
                )


        # Handle remaining buffer at end of file.
        #
        # This used to read `out_buf = _concatenate_buffer() if buffer_blocks
        # else None` and then dereference out_buf unconditionally, so when the
        # last chunk emptied the buffer exactly -- FILE_LENG % chunk_samples == 0
        # with no overlap, which is divisibility, not a rare case -- it raised
        # AttributeError. stream_fits is a GENERATOR, so the `except Exception`
        # outside this already-yielding loop did not hand the error to the
        # caller: the duplicated reader reopened the file and re-emitted it from
        # sample 0, and the consumer received every sample twice. The reader is
        # now chosen before anything is yielded, so that replay cannot happen at
        # all -- but the None check stays, because the AttributeError was real.
        out_buf = buffer.concatenate() if buffer else None
        if out_buf is not None and out_buf.shape[0] > 0:
            chunk_counter += 1

            valid_start = 0
            valid_end = out_buf.shape[0]
            block_out = out_buf.copy()
            if config.DATA_NEEDS_REVERSAL:
                block_out = block_out[:, :, ::-1]
            log_stream_fits_block_generation(
                chunk_counter,
                block_out.shape,
                str(block_out.dtype),
                emitted - out_buf.shape[0] + valid_start,
                emitted - out_buf.shape[0] + valid_end,
                valid_start,
                valid_end,
                valid_end - valid_start,
            )
            span = ChunkSpan(
                start_sample=emitted - out_buf.shape[0],
                end_sample=emitted,
                block_start_sample=emitted - out_buf.shape[0],
                block_end_sample=emitted,
            )
            metadata = chunk_metadata(
                span,
                chunk_samples=chunk_samples,
                total_samples=total_samples,
                nchans=nchan,
                nifs=1,
                block=block_out,
                tbin_sec=tbin,
                extra={
                    "tstart_mjd": tstart_mjd,
                    "tstart_mjd_corr": source.tstart_mjd_corr,
                    "tsubint_sec": tsub,
                },
            )
            yield block_out, metadata

        log_stream_fits_summary(chunk_counter)
    finally:
        source.close()


def _emit_subint_fallback_blocks(
    source: _SubintSource, chunk_samples: int, overlap_samples: int
) -> Generator[Tuple[np.ndarray, Dict], None, None]:
    """The buffered astropy reader taken when the primary one could not start.

    A near-copy of :func:`_emit_subint_primary_blocks`; the chunk arithmetic is
    identical and the two differ only in where the epoch comes from (D2, decided
    in :func:`_open_subint_fallback`) and in placing subints at ``i * NSBLK``
    instead of by ``OFFS_SUB`` (D4). They are not merged because
    ``tests/test_p1_regressions.py::TestFitsChunkGeometry`` asserts on the
    literal source text of four expressions below and requires exactly two
    copies of each; see the REF-03 report.
    """
    subint, tbl = source.subint, source.tbl
    nsubint, nchan, npol, nsblk = source.nsubint, source.nchan, source.npol, source.nsblk
    nbits, zero_off, tbin, tsub = source.nbits, source.zero_off, source.tbin, source.tsub
    pol_type, tstart_mjd = source.pol_type, source.tstart_mjd
    total_samples = source.total_samples
    limits = source.limits
    max_buffer_samples = limits.max_buffer_samples
    max_buffer_blocks = limits.max_buffer_blocks
    bytes_per_sample = limits.bytes_per_sample

    buffer = SubintBuffer(nchan)
    emitted = 0
    chunk_counter = 0

    # Progress tracking
    last_progress_log = time.time()
    progress_interval = 5.0  # Log progress every 5 seconds
    last_progress_row = 0
    buffer_check_interval_astropy = 10

    try:
        for i, row in enumerate(tbl):

            expected_start = i * nsblk


            if expected_start > emitted:
                buffer.pad(expected_start - emitted)
                emitted = expected_start


            block = _row_to_block(row["DATA"], row, subint, nbits, nsblk, npol, nchan, zero_off, pol_type)
            buffer.append(block)
            emitted += block.shape[0]

            # OPTIMIZATION: Only check time every N subints (reduces overhead)
            # Progress logging every 5 seconds
            if i % buffer_check_interval_astropy == 0:
                current_time = time.time()
                if current_time - last_progress_log >= progress_interval:
                    progress_pct = (i + 1) / nsubint * 100
                    rows_processed = i + 1 - last_progress_row
                    rate = rows_processed / (current_time - last_progress_log)
                    remaining = (nsubint - i - 1) / max(rate, 0.001)

                    logger.info(
                        f"Streaming progress: {i+1:,}/{nsubint:,} subints ({progress_pct:.1f}%) | "
                        f"Buffer: {buffer.n_blocks:,} blocks, {buffer.total_samples:,} samples | "
                        f"Rate: {rate:.1f} subints/s | ETA: {remaining:.1f}s"
                    )

                    last_progress_log = current_time
                    last_progress_row = i + 1

            # CRITICAL: Check buffer more aggressively for large chunks
            check_frequency_astropy = 1 if chunk_samples > 1_000_000 else buffer_check_interval_astropy

            if i % check_frequency_astropy == 0 or i == nsubint - 1:
                needs_chunk_emission = (
                    buffer.total_samples >= (chunk_samples + overlap_samples * 2) or
                    buffer.total_samples > max_buffer_samples or
                    buffer.n_blocks > max_buffer_blocks
                )
                if buffer.total_samples > max_buffer_samples * 0.8:
                    logger.warning(
                        f"Buffer approaching limit: {buffer.total_samples:,}/{max_buffer_samples:,} samples "
                        f"({buffer.n_blocks:,} blocks). Will emit chunk soon to prevent OOM."
                    )
            else:
                needs_chunk_emission = False

            # Only concatenate when we actually need to emit a chunk
            if needs_chunk_emission:
                if buffer.n_blocks > 50:
                    logger.debug(
                        f"Preparing chunk emission at row {i+1:,}/{nsubint:,}: "
                        f"{buffer.n_blocks:,} blocks, {buffer.total_samples:,} samples"
                    )
                out_buf = buffer.concatenate()
                buffer_too_large = out_buf.shape[0] > max_buffer_samples
                has_complete_chunk = out_buf.shape[0] >= (chunk_samples + overlap_samples * 2)

                if buffer_too_large:
                    logger.warning(
                        f"Buffer exceeded limit at row {i+1:,}/{nsubint:,}: "
                        f"{out_buf.shape[0]:,} samples > {max_buffer_samples:,}. "
                        f"Emitting emergency chunk to prevent OOM."
                    )
            else:
                # Don't concatenate yet - continue accumulating
                buffer_too_large = False
                has_complete_chunk = False
                out_buf = None  # Not computed yet

            # Emit chunk if we have a complete chunk OR if buffer is too large
            while needs_chunk_emission and (has_complete_chunk or (buffer_too_large and out_buf is not None and out_buf.shape[0] >= chunk_samples)):
                chunk_counter += 1

                if out_buf is None:
                    out_buf = buffer.concatenate()

                window = plan_buffered_window(
                    out_buf.shape[0], chunk_samples, overlap_samples,
                    buffer_too_large=buffer_too_large, large_chunk=limits.large_chunk,
                )
                start_with_overlap = window.start_with_overlap
                end_with_overlap = window.end_with_overlap
                valid_start = window.valid_start
                valid_end = window.valid_end
                actual_chunk_size = window.actual_chunk_size

                if window.emergency:
                    if actual_chunk_size < chunk_samples:
                        logger.warning(
                            f"Buffer too large ({out_buf.shape[0]:,} samples, {out_buf.shape[0] * bytes_per_sample / (1024**3):.2f} GB). "
                            f"Emitting partial chunk of {actual_chunk_size:,} samples "
                            f"(requested: {chunk_samples:,}) to prevent system freeze. "
                            f"Buffer limit: {max_buffer_samples:,} samples."
                        )
                    else:
                        logger.warning(
                            f"Buffer too large ({out_buf.shape[0]:,} samples), "
                            f"emitting emergency chunk of {actual_chunk_size:,} samples "
                            f"(buffer limit: {max_buffer_samples:,})"
                        )

                # At the very beginning of the file there is no
                # preceding data, so nothing can be left overlap:
                # keeping valid_start > 0 here discards the first
                # overlap_samples of every FITS file.
                if emitted - out_buf.shape[0] <= 0 and valid_start > 0:
                    valid_start = 0
                    valid_end = valid_start + actual_chunk_size

                block_out = out_buf[start_with_overlap:end_with_overlap].copy()

                if config.DATA_NEEDS_REVERSAL:
                    block_out = block_out[:, :, ::-1]

                start_sample_idx = emitted - out_buf.shape[0] + valid_start
                end_sample_idx = start_sample_idx + actual_chunk_size


                log_stream_fits_block_generation(
                    chunk_counter,
                    block_out.shape,
                    str(block_out.dtype),
                    start_sample_idx,
                    end_sample_idx,
                    start_with_overlap,
                    end_with_overlap,
                    actual_chunk_size,
                )
                metadata = {
                    "chunk_idx": start_sample_idx // chunk_samples,
                    "start_sample": start_sample_idx,
                    "end_sample": end_sample_idx,
                    "actual_chunk_size": actual_chunk_size,
                    "block_start_sample": emitted - out_buf.shape[0],
                    "block_end_sample": emitted - out_buf.shape[0] + end_with_overlap,
                    # Derived from the real geometry, not assumed:
                    # the first chunk of a file has no left overlap
                    # and the last one has no right overlap. Hardcoding
                    # overlap_samples made _process_block trim data
                    # that was never overlap.
                    "overlap_left": valid_start - start_with_overlap,
                    "overlap_right": max(0, end_with_overlap - valid_end),
                    "total_samples": total_samples,
                    "nchans": nchan,
                    "nifs": 1,
                    "dtype": str(block_out.dtype),
                    "shape": block_out.shape,
                    "file_type": "fits",

                    "tbin_sec": tbin,
                    "t_rel_start_sec": start_sample_idx * tbin,
                    "t_rel_end_sec": end_sample_idx * tbin,

                    "tstart_mjd": tstart_mjd,
                    "tstart_mjd_corr": source.tstart_mjd_corr,
                    "tsubint_sec": tsub,
                }
                yield block_out, metadata

                # Remove emitted chunk from buffer (keep overlap for next chunk).
                # Always advance by exactly the valid span. Dropping
                # end_with_overlap (= actual + 2*overlap) in the
                # emergency path left a 2*overlap hole between
                # consecutive chunks that was never searched.
                samples_to_remove = actual_chunk_size
                buffer.advance(out_buf, samples_to_remove)

                out_buf = buffer.concatenate() if buffer else np.zeros((0, 1, nchan), dtype=np.float32)

                # Update buffer size check for next iteration
                buffer_too_large = out_buf.shape[0] > max_buffer_samples
                has_complete_chunk = out_buf.shape[0] >= (chunk_samples + overlap_samples * 2)
                needs_chunk_emission = has_complete_chunk or buffer_too_large


        # Handle remaining buffer at end of file. See the same comment in the
        # primary reader: this used to dereference a None out_buf on an exactly
        # divisible file, and the resulting AttributeError made the whole file
        # stream twice.
        out_buf = buffer.concatenate() if buffer else None
        if out_buf is not None and out_buf.shape[0] > 0:
            chunk_counter += 1

            valid_start = 0
            valid_end = out_buf.shape[0]
            block_out = out_buf.copy()


            if config.DATA_NEEDS_REVERSAL:
                block_out = block_out[:, :, ::-1]

            log_stream_fits_block_generation(
                chunk_counter,
                block_out.shape,
                str(block_out.dtype),
                emitted - out_buf.shape[0],
                emitted,
                valid_start,
                valid_end,
                valid_end - valid_start,
            )
            span = ChunkSpan(
                start_sample=emitted - out_buf.shape[0],
                end_sample=emitted,
                block_start_sample=emitted - out_buf.shape[0],
                block_end_sample=emitted,
            )
            metadata = chunk_metadata(
                span,
                chunk_samples=chunk_samples,
                total_samples=total_samples,
                nchans=nchan,
                nifs=1,
                block=block_out,
                tbin_sec=tbin,
                extra={
                    "tstart_mjd": tstart_mjd,
                    "tstart_mjd_corr": source.tstart_mjd_corr,
                    "tsubint_sec": tsub,
                },
            )
            yield block_out, metadata

        log_stream_fits_summary(chunk_counter)
    finally:
        source.close()


# --------------------------------------------------------------------------- #
# the non-SUBINT reader
# --------------------------------------------------------------------------- #
class _NonSubintSource:
    """A whole non-PSRFITS FITS file, already in RAM."""

    def __init__(self, data_array: np.ndarray, nsamples: int, npols: int, nchans: int) -> None:
        self.data_array = data_array
        self.nsamples = nsamples
        self.npols = npols
        self.nchans = nchans


def _open_non_subint_source(
    file_name: str, chunk_samples: int, overlap_samples: int
) -> _NonSubintSource:
    """Last resort for a FITS file with no SUBINT table: load it whole."""
    _, h = _read_fits_like_fitsio(file_name)
    nsamples = safe_int(h.get("NAXIS2", 1)) * safe_int(h.get("NSBLK", 1))
    npols = safe_int(h.get("NPOL", 2))
    nchans = safe_int(h.get("NCHAN", 512))
    logger.info(
        "FITS data detected: samples=%d, polarisations=%d, channels=%d",
        nsamples,
        npols,
        nchans,
    )
    log_stream_fits_parameters(nsamples, chunk_samples, overlap_samples, None, nchans, npols, None)
    data_array = _load_fits_non_subint(file_name)
    return _NonSubintSource(data_array, nsamples, npols, nchans)


def _emit_non_subint_blocks(
    source: _NonSubintSource, chunk_samples: int, overlap_samples: int
) -> Generator[Tuple[np.ndarray, Dict], None, None]:
    """Slice the in-memory array. Moved verbatim; it has no characterization
    test and no counterpart in the fallback reader."""
    data_array = source.data_array
    nsamples, npols, nchans = source.nsamples, source.npols, source.nchans

    chunk_counter = 0
    for chunk_start in range(0, nsamples, chunk_samples):
        chunk_counter += 1
        valid_start = chunk_start
        valid_end = min(chunk_start + chunk_samples, nsamples)
        start_with_overlap = max(0, valid_start - overlap_samples)
        end_with_overlap = min(nsamples, valid_end + overlap_samples)
        block = data_array[start_with_overlap:end_with_overlap].copy()
        log_stream_fits_block_generation(
            chunk_counter,
            block.shape,
            str(block.dtype),
            valid_start,
            valid_end,
            start_with_overlap,
            end_with_overlap,
            valid_end - valid_start,
        )
        metadata = {
            "chunk_idx": valid_start // chunk_samples,
            "start_sample": valid_start,
            "end_sample": valid_end,
            "actual_chunk_size": valid_end - valid_start,
            "block_start_sample": start_with_overlap,
            "block_end_sample": end_with_overlap,
            "overlap_left": valid_start - start_with_overlap,
            "overlap_right": end_with_overlap - valid_end,
            "total_samples": nsamples,
            "nchans": nchans,
            "nifs": npols,
            "dtype": str(block.dtype),
            "shape": block.shape,
            "file_type": "fits",

            "tbin_sec": float(config.TIME_RESO) if hasattr(config, 'TIME_RESO') else None,
            "t_rel_start_sec": (valid_start * float(config.TIME_RESO)) if hasattr(config, 'TIME_RESO') else None,
            "t_rel_end_sec": (valid_end * float(config.TIME_RESO)) if hasattr(config, 'TIME_RESO') else None,
        }
        yield block, metadata
    log_stream_fits_summary(chunk_counter)


# --------------------------------------------------------------------------- #
# choosing a reader -- before anything is yielded
# --------------------------------------------------------------------------- #
def _replay(first, iterator):
    """Yield *first*, then the rest of *iterator*."""
    yield first
    yield from iterator


def _commit_to(reader):
    """Pull the first block, then hand back an iterator that replays it.

    This is the whole point of audit REF-03. ``stream_fits`` is a generator, and
    the old ``except Exception`` sat outside a loop that had ALREADY yielded: if
    the primary reader failed halfway through a file, the fallback reopened it
    and re-emitted from sample 0, so the consumer silently received duplicate
    data with nothing but a generic warning in the log. You cannot retract what
    you have yielded.

    So the choice of reader is made here, before the caller has seen anything. A
    reader that fails while producing its first block can still be replaced; one
    that has handed a block over cannot, and its exception now propagates.
    """
    iterator = iter(reader)
    try:
        first = next(iterator)
    except StopIteration:
        return iter(())
    return _replay(first, iterator)


def _open_primary_reader(file_name: str, chunk_samples: int, overlap_samples: int):
    """The reader we would rather use: ``your`` if installed, else astropy."""
    if your_psrfits is not None:
        source = _open_your_source(file_name, chunk_samples, overlap_samples)
        return _commit_to(_emit_your_blocks(source, chunk_samples, overlap_samples))

    # No `your`: read the SUBINT table with astropy ourselves.
    subint_source = _open_subint_primary(file_name, chunk_samples, overlap_samples)
    if subint_source is not None:
        return _commit_to(_emit_subint_primary_blocks(subint_source, chunk_samples, overlap_samples))

    # No SUBINT table either: a plain FITS file, loaded whole.
    non_subint = _open_non_subint_source(file_name, chunk_samples, overlap_samples)
    return _commit_to(_emit_non_subint_blocks(non_subint, chunk_samples, overlap_samples))


def _open_fallback_reader(file_name: str, chunk_samples: int, overlap_samples: int):
    """The duplicated astropy reader, reached only when the primary cannot start."""
    source = _open_subint_fallback(file_name, chunk_samples, overlap_samples)
    if source is None:
        raise ValueError(
            f"FITS file does not have a valid SUBINT structure: {file_name}"
        )
    return _commit_to(_emit_subint_fallback_blocks(source, chunk_samples, overlap_samples))


def _open_fits_reader(file_name: str, chunk_samples: int, overlap_samples: int):
    """Pick the reader for *file_name*. Nothing has been yielded when this returns."""
    try:
        return _open_primary_reader(file_name, chunk_samples, overlap_samples)
    except Exception as e:
        logger.warning("Error with 'your' PSRFITS implementation (%s); falling back to astropy", e)
    return _open_fallback_reader(file_name, chunk_samples, overlap_samples)


def stream_fits(
    file_name: str,
    chunk_samples: int = 2_097_152,
    overlap_samples: int = 0,
) -> Generator[Tuple[np.ndarray, Dict], None, None]:
    """
    Generator that reads a FITS file in blocks without loading everything into RAM.

    Args:
        file_name: Path to .fits file
        chunk_samples: Number of samples per block (default: 2M)
        overlap_samples: Number of overlap samples between blocks

    Yields:
        Tuple[data_block, metadata]: Data block (time, pol, chan) and metadata
    """
    try:
        logger.info("Streaming FITS data: chunk_size=%d, overlap=%d", chunk_samples, overlap_samples)
        yield from _open_fits_reader(file_name, chunk_samples, overlap_samples)
    except Exception as e:
        logger.error("Error in stream_fits: %s", e)
        raise ValueError(f"Could not read FITS file {file_name}") from e
