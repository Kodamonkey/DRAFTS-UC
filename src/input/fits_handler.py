# This module handles FITS and PSRFITS data ingestion.

"""FITS and PSRFITS file handling for FRB detection pipeline."""
from __future__ import annotations

import os
from typing import Generator, Tuple, Dict

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
from ..logging import (
    log_stream_fits_block_generation,
    log_stream_fits_load_strategy,
    log_stream_fits_parameters,
    log_stream_fits_summary
)
from .utils import safe_float, safe_int, auto_config_downsampling, print_debug_frequencies, save_file_debug_info


logger = logging.getLogger(__name__)


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
            l = np.sqrt(np.maximum(0.0, q * q + u * u)).astype(data.dtype, copy=False)
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


def load_fits_file(file_name: str) -> np.ndarray:
    """Load a FITS file and return the data array in shape (time, pol, channel)."""
    global_vars = config
    data_array = None
    try:
                                                                               
        if your_psrfits is not None:
            try:
                pf = your_psrfits.PsrfitsFile([file_name])
                nspec = int(pf.nspectra())
                npol = int(pf.npol)
                nchan = int(pf.nchans)
                                                      
                data = pf.get_data(0, nspec, npoln=npol)                                        
                if data.ndim != 3:
                    raise ValueError("Forma inesperada en 'your' get_data")
                                                        
                pol_type = getattr(pf, 'pol_type', 'IQUV') if hasattr(pf, 'pol_type') else 'IQUV'
                data = _select_polarization(data, pol_type, getattr(config, 'POLARIZATION_MODE', 'intensity'), getattr(config, 'POLARIZATION_INDEX', 0))
                                                                                              
                try:
                    if getattr(pf, 'foff', 0.0) > 0:
                        data = data[:, :, ::-1]
                except Exception:
                    pass
                if data.dtype != np.float32:
                    data = data.astype(np.float32)
                return data
            except Exception:
                                                        
                pass
        with fits.open(file_name, memmap=True) as hdul:
            if "SUBINT" in [hdu.name for hdu in hdul] and "DATA" in hdul["SUBINT"].columns.names:
                subint = hdul["SUBINT"]
                hdr = subint.header
                                                                                         
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
                nsubint = safe_int(hdr.get("NAXIS2", 0))
                nchan = safe_int(hdr.get("NCHAN", 0))
                npol = safe_int(hdr.get("NPOL", 0))
                nsblk = safe_int(hdr.get("NSBLK", 1))
                nbits = safe_int(hdr.get("NBITS", 8))
                zero_off = safe_float(hdr.get("ZERO_OFF", 0.0))
                                                       
                if any(x <= 0 for x in [nsubint, nchan, npol, nsblk]):
                    raise ValueError(
                        f"Invalid dimensions in FITS header: NAXIS2={nsubint}, NCHAN={nchan}, NPOL={npol}, NSBLK={nsblk} (cannot be <= 0)"
                    )

                                                             
                out = np.empty((nsubint * nsblk, 1, nchan), dtype=np.float32)

                                                                                    
                use_pol_idx = 0
                pol_type = str(hdr.get("POL_TYPE", "")).upper() if hdr.get("POL_TYPE") is not None else ""
                                                                       
                write_ptr = 0
                for isub in range(nsubint):
                    row = tbl[isub]
                    raw = row["DATA"]
                                                
                    if nbits < 8:
                        if nbits == 4:
                            unpacked = _unpack_4bit(raw)
                        elif nbits == 2:
                            unpacked = _unpack_2bit(raw)
                        elif nbits == 1:
                            unpacked = _unpack_1bit(raw)
                        else:
                            raise ValueError(f"NBITS={nbits} not supported")
                                                                              
                                                                     
                        try:
                            tmp = unpacked.reshape(nsblk, npol, nchan)
                        except Exception:
                                                                                    
                            tmp = unpacked.reshape(nsblk, nchan, npol).swapaxes(1, 2)
                        tmp = tmp.astype(np.float32, copy=False)
                    else:
                                                                   
                        arr = np.asarray(raw)
                                                                                        
                        try:
                            tmp = arr.reshape(nsblk, npol, nchan)
                        except Exception:
                            tmp = arr.reshape(nsblk, nchan, npol).swapaxes(1, 2)
                        if tmp.dtype != np.float32:
                            tmp = tmp.astype(np.float32, copy=False)

                                                   
                    dat_wts = None
                    if "DAT_WTS" in subint.columns.names:
                        dat_wts = row["DAT_WTS"].astype(np.float32, copy=False)
                    dat_scl = None
                    if "DAT_SCL" in subint.columns.names:
                        dat_scl = row["DAT_SCL"].astype(np.float32, copy=False)
                    dat_offs = None
                    if "DAT_OFFS" in subint.columns.names:
                        dat_offs = row["DAT_OFFS"].astype(np.float32, copy=False)

                                                                    
                    tmp = _apply_calibration(tmp, dat_wts, dat_scl, dat_offs, zero_off)

                    if npol >= 1:
                                                                 
                        out_block = tmp[:, use_pol_idx:use_pol_idx + 1, :]
                    else:
                        out_block = tmp.reshape(tmp.shape[0], 1, tmp.shape[-1])

                    out[write_ptr:write_ptr + nsblk, :, :] = out_block
                    write_ptr += nsblk

                data_array = out
            else:
                if fitsio is None:
                    raise ImportError("fitsio is not installed. Install with: pip install fitsio")
                temp_data, h = fitsio.read(file_name, header=True)
                if "DATA" in temp_data.dtype.names:
                    total_samples = safe_int(h.get("NAXIS2", 1)) * safe_int(h.get("NSBLK", 1))
                    num_pols = safe_int(h.get("NPOL", 2))
                    num_chans = safe_int(h.get("NCHAN", 512))
                    if any(x <= 0 for x in [total_samples, num_pols, num_chans]):
                                                                  
                        error_details = []
                        if total_samples <= 0:
                            error_details.append(f"total_samples={total_samples} (NAXIS2={h.get('NAXIS2', 1)} × NSBLK={h.get('NSBLK', 1)})")
                        if num_pols <= 0:
                            error_details.append(f"num_pols={num_pols}")
                        if num_chans <= 0:
                            error_details.append(f"num_chans={num_chans}")
                        
                        error_msg = f"Invalid dimensions in FITS header: {', '.join(error_details)}"
                        if total_samples <= 0:
                            error_msg += f"\n  → NSBLK={h.get('NSBLK', 1)} is 0 or negative, making it impossible to calculate the number of temporal samples"
                            error_msg += f"\n  → The pipeline needs data in (time, polarization, channel) format but cannot determine the temporal dimension"
                        if num_pols <= 0:
                            error_msg += f"\n  → NPOL={num_pols} is 0 or negative, making it impossible to process polarizations"
                        if num_chans <= 0:
                            error_msg += f"\n  → NCHAN={num_chans} is 0 or negative, making it impossible to process frequency channels"
                        
                        raise ValueError(error_msg)
                    try:
                        data_array = temp_data["DATA"].reshape(total_samples, num_pols, num_chans)
                                                                    
                        pol_type = str(h.get("POL_TYPE", "")).upper() if h.get("POL_TYPE") is not None else ""
                        if num_pols >= 1:
                            if "IQUV" in pol_type:
                                data_array = data_array[:, 0:1, :]
                            else:
                                data_array = data_array[:, 0:1, :]
                        else:
                            data_array = data_array.reshape(data_array.shape[0], 1, data_array.shape[-1])
                    except Exception as e:
                        raise ValueError(f"Error reshaping data (fitsio): {e}\n  → Data cannot be reorganized into expected format (time={total_samples}, pol={num_pols}, channel={num_chans})")
                else:
                    total_samples = safe_int(h.get("NAXIS2", 1)) * safe_int(h.get("NSBLK", 1))
                    num_pols = safe_int(h.get("NPOL", 2))
                    num_chans = safe_int(h.get("NCHAN", 512))
                    if any(x <= 0 for x in [total_samples, num_pols, num_chans]):
                                                                  
                        error_details = []
                        if total_samples <= 0:
                            error_details.append(f"total_samples={total_samples} (NAXIS2={h.get('NAXIS2', 1)} × NSBLK={h.get('NSBLK', 1)})")
                        if num_pols <= 0:
                            error_details.append(f"num_pols={num_pols}")
                        if num_chans <= 0:
                            error_details.append(f"num_chans={num_chans}")
                        
                        error_msg = f"Invalid dimensions in FITS header: {', '.join(error_details)}"
                        if total_samples <= 0:
                            error_msg += f"\n  → NSBLK={h.get('NSBLK', 1)} is 0 or negative, making it impossible to calculate the number of temporal samples"
                            error_msg += f"\n  → The pipeline needs data in (time, polarization, channel) format but cannot determine the temporal dimension"
                        if num_pols <= 0:
                            error_msg += f"\n  → NPOL={num_pols} is 0 or negative, making it impossible to process polarizations"
                        if num_chans <= 0:
                            error_msg += f"\n  → NCHAN={num_chans} is 0 or negative, making it impossible to process frequency channels"
                        
                        raise ValueError(error_msg)
                    try:
                        data_array = temp_data.reshape(total_samples, num_pols, num_chans)
                                                                                                   
                        data_array = data_array[:, 0:1, :]
                    except Exception as e:
                        raise ValueError(f"Error reshaping data (fitsio): {e}\n  → Data cannot be reorganized into expected format (time={total_samples}, pol={num_pols}, channel={num_chans})")
    except (ValueError, fits.verify.VerifyError) as e:

        if "NSBLK" in str(e) and "0" in str(e):
            raise ValueError(
                f"Corrupted FITS file: {file_name}\n  → The file has NSBLK=0 in the header, indicating it is malformed or corrupted\n  → NSBLK must be > 0 to define the number of temporal samples per block\n  → Recommendation: Verify the file source or obtain a correct version"
            ) from e
        elif "Invalid dimensions" in str(e):
            raise ValueError(
                f"Corrupted FITS file: {file_name}\n  → {str(e)}\n  → The file cannot be processed due to invalid dimensions in the header\n  → Recommendation: Verify the file integrity or use a different file"
            ) from e
        else:
            raise ValueError(
                f"Corrupted FITS file: {file_name}\n  → {str(e)}\n  → The file cannot be read correctly\n  → Recommendation: Ensure the file is not damaged"
            ) from e
    except Exception as e:
        logger.error("Error loading FITS with fitsio/astropy: %s", e)
        try:
                                                         
            with fits.open(file_name, memmap=False) as f:
                data_hdu = None
                for hdu_item in f:
                    try:
                        if (hdu_item.data is not None and 
                            isinstance(hdu_item.data, np.ndarray) and 
                            hdu_item.data.ndim >= 3):
                            data_hdu = hdu_item
                            break
                    except (TypeError, ValueError):
                        continue
                if data_hdu is None and len(f) > 1:
                    data_hdu = f[1]
                elif data_hdu is None:
                    data_hdu = f[0]
                h = data_hdu.header
                try:
                    raw_data = data_hdu.data
                    if raw_data is not None:
                        total_samples = safe_int(h.get("NAXIS2", 1)) * safe_int(h.get("NSBLK", 1))
                        num_pols = safe_int(h.get("NPOL", 2))
                        num_chans = safe_int(h.get("NCHAN", raw_data.shape[-1]))
                        if any(x <= 0 for x in [total_samples, num_pols, num_chans]):
                                                                        
                            error_details = []
                            if total_samples <= 0:
                                error_details.append(f"total_samples={total_samples} (NAXIS2={h.get('NAXIS2', 1)} × NSBLK={h.get('NSBLK', 1)})")
                            if num_pols <= 0:
                                error_details.append(f"num_pols={num_pols}")
                            if num_chans <= 0:
                                error_details.append(f"num_chans={num_chans}")
                            
                            error_msg = f"Invalid dimensions in fallback header: {', '.join(error_details)}"
                            if total_samples <= 0:
                                error_msg += f"\n  → NSBLK={h.get('NSBLK', 1)} is 0 or negative, making it impossible to calculate the number of temporal samples"
                                error_msg += f"\n  → The pipeline needs data in (time, polarization, channel) format but cannot determine the temporal dimension"
                            if num_pols <= 0:
                                error_msg += f"\n  → NPOL={num_pols} is 0 or negative, making it impossible to process polarizations"
                            if num_chans <= 0:
                                error_msg += f"\n  → NCHAN={num_chans} is 0 or negative, making it impossible to process frequency channels"
                            
                            raise ValueError(error_msg)
                        try:
                            data_array = raw_data.reshape(total_samples, num_pols, num_chans)
                            pol_type = str(h.get("POL_TYPE", "")).upper() if h.get("POL_TYPE") is not None else ""
                            if num_pols >= 1:
                                if "IQUV" in pol_type:
                                    data_array = data_array[:, 0:1, :]
                                else:
                                    data_array = data_array[:, 0:1, :]
                            else:
                                data_array = data_array.reshape(data_array.shape[0], 1, data_array.shape[-1])
                        except Exception as e:
                            raise ValueError(f"Error reshaping data (fallback): {e}\n  → Data cannot be reorganized into expected format (time={total_samples}, pol={num_pols}, channel={num_chans})\n  → File may be corrupted or have incompatible format")
                    else:
                        raise ValueError("No valid data in HDU\n  → FITS file does not contain processable data\n  → Verify that the file is not empty or corrupted")
                except (TypeError, ValueError) as e_data:
                    logger.debug(f"Error accessing HDU data: {e_data}")
                    raise ValueError(f"Corrupted FITS file: {file_name}\n  → Error accessing file data: {e_data}\n  → File may be damaged or have unrecognized format")
        except Exception as e_astropy:
            logger.debug(f"Final failure loading with astropy: {e_astropy}")
            raise ValueError(f"Corrupted FITS file: {file_name}\n  → Failure in astropy fallback method: {e_astropy}\n  → File cannot be read by any available method\n  → Recommendation: Verify file integrity or use a different file") from e_astropy
            
    if data_array is None:
        raise ValueError(f"Corrupted FITS file: {file_name}\n  → Could not load valid data from file\n  → File may be empty, corrupted or have incompatible format\n  → Recommendation: Verify file integrity or use a different file")

    if global_vars.DATA_NEEDS_REVERSAL:
        logger.debug(f">> Reversing frequency axis of loaded data for {file_name}")
        data_array = np.ascontiguousarray(data_array[:, :, ::-1])
    
                                              
    if config.DEBUG_FREQUENCY_ORDER:
        logger.debug(f"[DEBUG LOADED DATA] File: {file_name}")
        logger.debug(f"[DEBUG LOADED DATA] Data shape: {data_array.shape}")
        logger.debug(f"[DEBUG LOADED DATA] Dimensions: (time={data_array.shape[0]}, pol={data_array.shape[1]}, freq={data_array.shape[2]})")
        logger.debug(f"[DEBUG LOADED DATA] Data type: {data_array.dtype}")
        logger.debug(f"[DEBUG LOADED DATA] Memory size: {data_array.nbytes / (1024**3):.2f} GB")
        logger.debug(f"[DEBUG LOADED DATA] Reversal applied: {global_vars.DATA_NEEDS_REVERSAL}")
        logger.debug(f"[DEBUG LOADED DATA] Value range: [{data_array.min():.3f}, {data_array.max():.3f}]")
        logger.debug(f"[DEBUG LOADED DATA] Mean value: {data_array.mean():.3f}")
        logger.debug(f"[DEBUG LOADED DATA] Standard deviation: {data_array.std():.3f}")
        logger.debug("[DEBUG LOADED DATA] " + "="*50)
    
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
                                                                        
            if tstart_mjd is not None:
                try:
                    config.TSTART_MJD = tstart_mjd
                    config.TSTART_MJD_CORR = tstart_mjd + (nsuboffs * tsubint) / 86400.0
                except Exception:
                    config.TSTART_MJD = tstart_mjd

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
                                                                                                                    
                                                                         
                    freq_axis_inverted = True
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
                                                                                                                            
                            freq_axis_inverted = True
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
        if freq_axis_inverted:
                                                                                                   
            config.FREQ = freq_temp[::-1]
            config.DATA_NEEDS_REVERSAL = True
            try:
                config.NEED_FLIPBAND = True
            except Exception:
                pass
        else:
            config.FREQ = freq_temp
            config.DATA_NEEDS_REVERSAL = False
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
                    
                    # Reverse frequency if needed
                    try:
                        if getattr(pf, 'foff', 0.0) > 0:
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
        
        # Fallback: load entire file (not streaming, but returns same format)
        logger.warning("Multi-pol streaming not available, loading entire file")
        data_full = load_fits_file(file_name)
        
        # Try to reload with all polarizations
        # This is a simplified fallback - in production you'd need full FITS reading
        metadata = {
            "chunk_idx": 0,
            "start_sample": 0,
            "end_sample": data_full.shape[0],
            "actual_chunk_size": data_full.shape[0],
            "block_start_sample": 0,
            "block_end_sample": data_full.shape[0],
            "overlap_left": 0,
            "overlap_right": 0,
            "total_samples": data_full.shape[0],
            "nchans": data_full.shape[2],
            "npol": 1,
            "file_type": "fits",
        }
        
        # Return data twice (same block) since we don't have multi-pol in fallback
        yield data_full, data_full, metadata, "UNKNOWN"
        
    except Exception as e:
        logger.error("Error in multi-pol streaming: %s", e)
        raise


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
    
                                                                                        
    def _row_to_block(row_data: np.ndarray, row: np.ndarray, subint, nbits: int, nsblk: int, npol: int, nchan: int, zero_off: float, pol_type: str) -> np.ndarray:
                                                        
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
    
    try:
        logger.info("Streaming FITS data: chunk_size=%d, overlap=%d", chunk_samples, overlap_samples)
        
                                                                                       
        try:
            if your_psrfits is not None:
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
                total_samples = nspec
                log_stream_fits_parameters(total_samples, chunk_samples, overlap_samples, None, nchan, npol, None)
                chunk_counter = 0
                emitted = 0
                                                        
                step = chunk_samples
                while emitted < nspec:
                    start = emitted
                                                           
                    read_start = max(0, start - overlap_samples)
                    read_end = min(nspec, start + step + overlap_samples)
                    count = read_end - read_start
                    arr = pf.get_data(read_start, count, npoln=npol)                        
                    if arr.ndim != 3:
                        raise ValueError("Unexpected shape in 'your' get_data")
                                                                                  
                    block = _select_polarization(arr, getattr(pf, 'pol_type', 'IQUV'), getattr(config, 'POLARIZATION_MODE', 'intensity'), getattr(config, 'POLARIZATION_INDEX', 0))
                    try:
                        if getattr(pf, 'foff', 0.0) > 0:
                            block = block[:, :, ::-1]
                    except Exception:
                        pass
                    if block.dtype != np.float32:
                        block = block.astype(np.float32)

                    chunk_counter += 1
                    valid_start = start
                    valid_end = min(start + step, nspec)
                    start_with_overlap = read_start
                    end_with_overlap = read_end
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
                        "total_samples": nspec,
                        "nchans": nchan,
                        "nifs": 1,
                        "dtype": str(block.dtype),
                        "shape": block.shape,
                        "file_type": "fits",
                                                                                   
                        "tbin_sec": tsamp,
                        "t_rel_start_sec": valid_start * tsamp,
                        "t_rel_end_sec": valid_end * tsamp,
                    }
                    yield block, metadata
                    emitted += step

                log_stream_fits_summary(chunk_counter)
                return

                                                                                             
            with fits.open(file_name, memmap=True) as hdul:
                if "SUBINT" in [hdu.name for hdu in hdul] and "DATA" in hdul["SUBINT"].columns.names:
                    subint = hdul["SUBINT"]
                    hdr = subint.header
                                                                                             
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
                    nsubint = safe_int(hdr.get("NAXIS2", 0))
                    nchan = safe_int(hdr.get("NCHAN", 0))
                    npol = safe_int(hdr.get("NPOL", 0))
                    nsblk = safe_int(hdr.get("NSBLK", 1))
                    nbits = safe_int(hdr.get("NBITS", 8))
                    zero_off = safe_float(hdr.get("ZERO_OFF", 0.0))
                    tbin = safe_float(hdr.get("TBIN"))
                                                             
                    tsub = safe_float(hdr.get("TSUBINT", nsblk * tbin))
                                                                             
                    primary = hdul["PRIMARY"].header if "PRIMARY" in [h.name for h in hdul] else {}
                    imjd = safe_int(primary.get("STT_IMJD", 0))
                    smjd = safe_float(primary.get("STT_SMJD", 0.0))
                    offs = safe_float(primary.get("STT_OFFS", 0.0))
                    tstart_mjd = float(imjd) + (float(smjd) + float(offs)) / 86400.0
                                                                       
                    nsuboffs = safe_int(hdr.get("NSUBOFFS", 0))
                    if "OFFS_SUB" in subint.columns.names:
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

                    # OPTIMIZATION: Use list to accumulate blocks, only concatenate when needed
                    # This avoids O(N^2) complexity from repeated np.concatenate calls
                    buffer_blocks = []  # List of (block, emitted_samples) tuples
                    buffer_total_samples = 0  # Track total samples without concatenating
                    emitted = 0
                    chunk_counter = 0
                    
                    # Memory safety: limit buffer growth to prevent OOM
                    # CRITICAL: For very large chunks, we need stricter limits
                    # Calculate based on available memory to prevent system freeze
                    import psutil
                    vm = psutil.virtual_memory()
                    available_ram_gb = vm.available / (1024**3)
                    
                    # Conservative limit: use max 30% of available RAM for buffer
                    # This prevents system freeze even with very large chunks
                    max_buffer_gb = min(available_ram_gb * 0.3, 8.0)  # Cap at 8 GB
                    bytes_per_sample = 4 * nchan  # float32 = 4 bytes
                    max_buffer_bytes = max_buffer_gb * (1024**3)
                    max_buffer_samples = int(max_buffer_bytes / bytes_per_sample)
                    
                    # Also enforce: buffer should not exceed 2x chunk size
                    # But for very large chunks, use memory-based limit instead
                    if chunk_samples > 1_000_000:  # Chunks > 1M samples
                        # For large chunks, use memory-based limit (more conservative)
                        max_buffer_samples = min(max_buffer_samples, chunk_samples + overlap_samples * 2)
                        logger.warning(
                            f"Large chunk detected ({chunk_samples:,} samples). "
                            f"Using conservative buffer limit: {max_buffer_samples:,} samples "
                            f"({max_buffer_gb:.2f} GB) to prevent system freeze."
                        )
                    else:
                        # For normal chunks, allow 2x chunk size
                        max_buffer_samples = max(max_buffer_samples, chunk_samples * 2)
                    
                    # Additional safety: limit number of blocks before forcing concatenation
                    # Too many blocks = slow concatenation = system freeze
                    max_buffer_blocks = 200  # Force concatenation if >200 blocks
                    
                    logger.info(
                        f"Buffer limits: max_samples={max_buffer_samples:,} "
                        f"({max_buffer_samples * bytes_per_sample / (1024**3):.2f} GB), "
                        f"max_blocks={max_buffer_blocks}, chunk_samples={chunk_samples:,}"
                    )
                    
                    def _concatenate_buffer():
                        """Concatenate all blocks in buffer_blocks into a single array.
                        
                        OPTIMIZATION: Uses pre-allocation when possible to reduce memory
                        reallocation overhead during concatenation.
                        """
                        if not buffer_blocks:
                            return np.zeros((0, 1, nchan), dtype=np.float32)
                        if len(buffer_blocks) == 1:
                            return buffer_blocks[0][0]
                        
                        # Log warning if concatenation might be slow
                        if len(buffer_blocks) > 100:
                            logger.warning(
                                f"Concatenating large buffer: {len(buffer_blocks):,} blocks "
                                f"(~{buffer_total_samples:,} samples). This may take a few seconds..."
                            )
                        
                        import time
                        start_time = time.time()
                        
                        # OPTIMIZATION: Pre-allocate result array to avoid repeated reallocation
                        # This is faster than np.concatenate for many blocks
                        result = np.empty((buffer_total_samples, 1, nchan), dtype=np.float32)
                        current_idx = 0
                        for block, _ in buffer_blocks:
                            block_len = block.shape[0]
                            result[current_idx:current_idx + block_len] = block
                            current_idx += block_len
                        
                        elapsed = time.time() - start_time
                        
                        if elapsed > 1.0:
                            logger.warning(
                                f"Buffer concatenation took {elapsed:.2f}s for {len(buffer_blocks):,} blocks "
                                f"({buffer_total_samples:,} samples). Consider reducing chunk size if this is frequent."
                            )
                        
                        return result
                    
                    def _expected_start_sample(offs_sub_seconds: float, sub_index: int) -> int:
                        if offs_sub_seconds is not None:
                            start_sec = float(offs_sub_seconds) - 0.5 * tsub
                            return int(round(start_sec / tbin))
                        return (nsuboffs + sub_index) * nsblk
                    
                    # Progress tracking
                    import time
                    last_progress_log = time.time()
                    last_progress_row = 0
                    progress_interval = 5.0  # Log every 5 seconds
                    
                    pol_type = str(hdr.get("POL_TYPE", "")).upper() if hdr.get("POL_TYPE") is not None else ""
                    
                    # OPTIMIZATION: Pre-verify column existence (avoid repeated checks)
                    has_offs_sub = "OFFS_SUB" in subint.columns.names
                    
                    logger.info(
                        f"[STREAMING] Starting subint processing: {nsubint:,} subints, "
                        f"chunk_samples={chunk_samples:,}, overlap={overlap_samples:,} samples"
                    )
                    
                    # OPTIMIZATION: Check buffer less frequently (every 10 subints instead of every 1)
                    buffer_check_interval = 10
                    
                    for isub in range(nsubint):
                        row = tbl[isub]
                        offs_sub_val = None
                        if has_offs_sub:
                            try:
                                offs_sub_val = safe_float(row["OFFS_SUB"])     
                            except Exception:
                                offs_sub_val = None

                        expected_start = _expected_start_sample(offs_sub_val, isub)
                                                     
                        if emitted < expected_start:
                            gap = expected_start - emitted
                            # OPTIMIZATION: Use np.empty + fill instead of np.zeros (slightly faster)
                            pad = np.empty((gap, 1, nchan), dtype=np.float32)
                            pad.fill(0.0)
                            # Add gap to buffer_blocks instead of concatenating
                            buffer_blocks.append((pad, emitted))
                            buffer_total_samples += gap
                            emitted += gap

                        block = _row_to_block(row["DATA"], row, subint, nbits, nsblk, npol, nchan, zero_off, pol_type)
                        # Add block to buffer_blocks instead of concatenating
                        buffer_blocks.append((block, emitted))
                        buffer_total_samples += block.shape[0]
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
                                    f"Buffer: {len(buffer_blocks):,} blocks, {buffer_total_samples:,} samples | "
                                    f"Rate: {rate:.1f} subints/s | ETA: {remaining:.1f}s"
                                )
                                
                                last_progress_log = current_time
                                last_progress_row = isub + 1
                        
                        # CRITICAL: Check buffer more aggressively for large chunks
                        # For large chunks, check every subint to prevent buffer explosion
                        # For normal chunks, check every N subints to reduce overhead
                        check_frequency = 1 if chunk_samples > 1_000_000 else buffer_check_interval
                        
                        # OPTIMIZATION: Check buffer based on size and block count
                        # Use buffer_total_samples (tracked without concatenation) to estimate
                        # This avoids expensive concatenation until we're ready to emit
                        if isub % check_frequency == 0 or isub == nsubint - 1:
                            # Check multiple conditions:
                            # 1. Have enough for complete chunk
                            # 2. Buffer exceeds memory limit
                            # 3. Too many blocks (will cause slow concatenation)
                            needs_chunk_emission = (
                                buffer_total_samples >= (chunk_samples + overlap_samples * 2) or
                                buffer_total_samples > max_buffer_samples or
                                len(buffer_blocks) > max_buffer_blocks
                            )
                            
                            # Warn if buffer is growing too large
                            if buffer_total_samples > max_buffer_samples * 0.8:
                                logger.warning(
                                    f"Buffer approaching limit: {buffer_total_samples:,}/{max_buffer_samples:,} samples "
                                    f"({len(buffer_blocks):,} blocks). Will emit chunk soon to prevent OOM."
                                )
                        else:
                            needs_chunk_emission = False
                        
                        # Only concatenate when we actually need to emit a chunk
                        if needs_chunk_emission:
                            # CRITICAL: Warn if concatenation will be expensive
                            if len(buffer_blocks) > 100:
                                logger.warning(
                                    f"Concatenating large buffer: {len(buffer_blocks):,} blocks, "
                                    f"{buffer_total_samples:,} samples (~{buffer_total_samples * bytes_per_sample / (1024**3):.2f} GB). "
                                    f"This may take several seconds. Consider reducing chunk size for better performance."
                                )
                            elif len(buffer_blocks) > 50:
                                logger.info(
                                    f"Preparing chunk emission at subint {isub+1:,}/{nsubint:,}: "
                                    f"{len(buffer_blocks):,} blocks, {buffer_total_samples:,} samples"
                                )
                            
                            # Measure concatenation time
                            import time
                            concat_start = time.time()
                            out_buf = _concatenate_buffer()
                            concat_time = time.time() - concat_start
                            
                            if concat_time > 2.0:
                                logger.warning(
                                    f"Buffer concatenation took {concat_time:.2f}s for {len(buffer_blocks):,} blocks. "
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
                            
                            # CRITICAL: If buffer is too large, emit smaller chunks aggressively
                            # For very large chunks, emit partial chunks to prevent system freeze
                            if buffer_too_large:
                                # Calculate safe chunk size: use 50-80% of buffer to leave room
                                # This prevents system freeze by not using all available memory
                                safe_chunk_ratio = 0.7 if chunk_samples > 1_000_000 else 0.9
                                max_safe_chunk = int(out_buf.shape[0] * safe_chunk_ratio)
                                actual_chunk_size = min(chunk_samples, max_safe_chunk, out_buf.shape[0])
                                
                                # Ensure we have at least some overlap for continuity
                                if actual_chunk_size < overlap_samples * 2:
                                    actual_chunk_size = min(overlap_samples * 2, out_buf.shape[0])
                                
                                start_with_overlap = 0
                                end_with_overlap = min(actual_chunk_size + overlap_samples * 2, out_buf.shape[0])
                                valid_start = overlap_samples if end_with_overlap > overlap_samples * 2 else 0
                                valid_end = valid_start + actual_chunk_size
                                
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
                                
                                # Record buffer event for validation metrics
                                try:
                                    from ..core.data_flow_manager import _validation_collector
                                    if _validation_collector is not None:
                                        _validation_collector.record_buffer_event(
                                            buffer_size_samples=out_buf.shape[0],
                                            event_type="emergency_chunk_emission",
                                            chunk_emitted=True
                                        )
                                except Exception:
                                    pass  # Don't fail if collector not available
                            else:
                                # Normal case: emit full chunk with overlap
                                start_with_overlap = 0
                                end_with_overlap = chunk_samples + overlap_samples * 2
                                valid_start = overlap_samples
                                valid_end = valid_start + chunk_samples
                                actual_chunk_size = chunk_samples
                            
                            # Ensure out_buf is concatenated before extracting chunk
                            if out_buf is None:
                                out_buf = _concatenate_buffer()
                            
                            block_out = out_buf[start_with_overlap:end_with_overlap].copy()
                                                 
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
                                "overlap_left": overlap_samples,
                                "overlap_right": overlap_samples,
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
                                "tstart_mjd_corr": tstart_mjd + (nsuboffs * tsub) / 86400.0,
                                "tsubint_sec": tsub,
                            }
                            yield block_out, metadata
                                                                         
                            # Remove emitted chunk from buffer (keep overlap for next chunk)
                            samples_to_remove = actual_chunk_size if not buffer_too_large else end_with_overlap
                            
                            # Update buffer: remove emitted samples, keep overlap
                            # Rebuild buffer_blocks list from remaining data
                            if samples_to_remove >= out_buf.shape[0]:
                                # All buffer was emitted
                                buffer_blocks = []
                                buffer_total_samples = 0
                            else:
                                # Keep tail of buffer (overlap)
                                remaining_buf = out_buf[samples_to_remove:]
                                if remaining_buf.shape[0] > 0:
                                    # Rebuild buffer_blocks with remaining data
                                    buffer_blocks = [(remaining_buf, emitted - remaining_buf.shape[0])]
                                    buffer_total_samples = remaining_buf.shape[0]
                                else:
                                    buffer_blocks = []
                                    buffer_total_samples = 0
                            
                            # Update buffer size check for next iteration
                            if buffer_blocks:
                                out_buf = _concatenate_buffer()
                                buffer_too_large = out_buf.shape[0] > max_buffer_samples
                                has_complete_chunk = out_buf.shape[0] >= (chunk_samples + overlap_samples * 2)
                            else:
                                out_buf = None
                                buffer_too_large = False
                                has_complete_chunk = False
                            
                            # Recalculate needs_chunk_emission for next iteration
                            needs_chunk_emission = (
                                buffer_total_samples >= (chunk_samples + overlap_samples * 2) or
                                buffer_total_samples > max_buffer_samples
                            )
                    
                                            
                    # Handle remaining buffer at end of file
                    if buffer_blocks:
                        out_buf = _concatenate_buffer()
                    if out_buf.shape[0] > 0:
                        chunk_counter += 1
                                                                                     
                        valid_start = 0
                        valid_end = out_buf.shape[0]
                        block_out = out_buf.copy()
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
                        metadata = {
                            "chunk_idx": (emitted - out_buf.shape[0]) // max(1, chunk_samples),
                            "start_sample": emitted - out_buf.shape[0] + valid_start,
                            "end_sample": emitted,
                            "actual_chunk_size": valid_end - valid_start,
                            "block_start_sample": emitted - out_buf.shape[0],
                            "block_end_sample": emitted,
                            "overlap_left": 0,
                            "overlap_right": 0,
                            "total_samples": total_samples,
                            "nchans": nchan,
                            "nifs": 1,
                            "dtype": str(block_out.dtype),
                            "shape": block_out.shape,
                            "file_type": "fits",
                                                                                       
                            "tbin_sec": tbin,
                            "t_rel_start_sec": (emitted - out_buf.shape[0] + valid_start) * tbin,
                            "t_rel_end_sec": emitted * tbin,
                                                                        
                            "tstart_mjd": tstart_mjd,
                            "tstart_mjd_corr": tstart_mjd + (nsuboffs * tsub) / 86400.0,
                            "tsubint_sec": tsub,
                        }
                        yield block_out, metadata

                    log_stream_fits_summary(chunk_counter)
                    return

                                                                     
                if fitsio is None:
                    raise ImportError("fitsio is not installed. Install with: pip install fitsio")
                temp_data, h = fitsio.read(file_name, header=True)
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
                data_array = load_fits_file(file_name)
                use_memmap = False
                                                           
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
        except Exception as e:
            logger.warning("Error with 'your' PSRFITS implementation (%s); falling back to astropy", e)

            with fits.open(file_name, memmap=True) as hdul:
                if "SUBINT" in [hdu.name for hdu in hdul] and "DATA" in hdul["SUBINT"].columns.names:
                    subint = hdul["SUBINT"]
                    hdr = subint.header

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
                    nsubint = safe_int(hdr.get("NAXIS2", 0))
                    nchan = safe_int(hdr.get("NCHAN", 0))
                    npol = safe_int(hdr.get("NPOL", 0))
                    nsblk = safe_int(hdr.get("NSBLK", 1))
                    nbits = safe_int(hdr.get("NBITS", 8))
                    zero_off = safe_float(hdr.get("ZERO_OFF", 0.0))
                    tbin = safe_float(hdr.get("TBIN"))
                                                             
                    tsub = safe_float(hdr.get("TSUBINT", nsblk * tbin))
                                                                             
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
                    
                                                                        
                    pol_type = str(hdr.get("POL_TYPE", "")).upper() if hdr.get("POL_TYPE") is not None else ""
                    
                                                          
                    total_samples = nsubint * nsblk
                    logger.info(
                        f"[STREAMING] Starting subint processing (astropy fallback): {nsubint:,} subints, "
                        f"chunk_samples={chunk_samples:,}, overlap={overlap_samples:,} samples"
                    )
                    log_stream_fits_parameters(total_samples, chunk_samples, overlap_samples, None, nchan, npol, None)
                    
                                                           
                    # OPTIMIZATION: Use list to accumulate blocks, only concatenate when needed
                    # This avoids O(N^2) complexity from repeated np.concatenate calls
                    buffer_blocks = []  # List of (block, emitted_samples) tuples
                    buffer_total_samples = 0  # Track total samples without concatenating
                    emitted = 0
                    chunk_counter = 0
                    
                    # Memory safety: limit buffer growth to prevent OOM
                    # CRITICAL: For very large chunks, we need stricter limits
                    # Calculate based on available memory to prevent system freeze
                    import psutil
                    vm = psutil.virtual_memory()
                    available_ram_gb = vm.available / (1024**3)
                    
                    # Conservative limit: use max 30% of available RAM for buffer
                    # This prevents system freeze even with very large chunks
                    max_buffer_gb = min(available_ram_gb * 0.3, 8.0)  # Cap at 8 GB
                    bytes_per_sample = 4 * nchan  # float32 = 4 bytes
                    max_buffer_bytes = max_buffer_gb * (1024**3)
                    max_buffer_samples = int(max_buffer_bytes / bytes_per_sample)
                    
                    # Also enforce: buffer should not exceed 2x chunk size
                    # But for very large chunks, use memory-based limit instead
                    if chunk_samples > 1_000_000:  # Chunks > 1M samples
                        # For large chunks, use memory-based limit (more conservative)
                        max_buffer_samples = min(max_buffer_samples, chunk_samples + overlap_samples * 2)
                        logger.warning(
                            f"Large chunk detected ({chunk_samples:,} samples). "
                            f"Using conservative buffer limit: {max_buffer_samples:,} samples "
                            f"({max_buffer_gb:.2f} GB) to prevent system freeze."
                        )
                    else:
                        # For normal chunks, allow 2x chunk size
                        max_buffer_samples = max(max_buffer_samples, chunk_samples * 2)
                    
                    # Additional safety: limit number of blocks before forcing concatenation
                    # Too many blocks = slow concatenation = system freeze
                    max_buffer_blocks = 200  # Force concatenation if >200 blocks
                    
                    logger.info(
                        f"Buffer limits: max_samples={max_buffer_samples:,} "
                        f"({max_buffer_samples * bytes_per_sample / (1024**3):.2f} GB), "
                        f"max_blocks={max_buffer_blocks}, chunk_samples={chunk_samples:,}"
                    )
                    
                    def _concatenate_buffer():
                        """Concatenate all blocks in buffer_blocks into a single array.
                        
                        OPTIMIZATION: Uses pre-allocation when possible to reduce memory
                        reallocation overhead during concatenation.
                        """
                        if not buffer_blocks:
                            return np.zeros((0, 1, nchan), dtype=np.float32)
                        if len(buffer_blocks) == 1:
                            return buffer_blocks[0][0]
                        
                        # Log warning if concatenation might be slow
                        if len(buffer_blocks) > 100:
                            logger.warning(
                                f"Concatenating large buffer: {len(buffer_blocks):,} blocks "
                                f"(~{buffer_total_samples:,} samples). This may take a few seconds..."
                            )
                        
                        import time
                        start_time = time.time()
                        
                        # OPTIMIZATION: Pre-allocate result array to avoid repeated reallocation
                        # This is faster than np.concatenate for many blocks
                        result = np.empty((buffer_total_samples, 1, nchan), dtype=np.float32)
                        current_idx = 0
                        for block, _ in buffer_blocks:
                            block_len = block.shape[0]
                            result[current_idx:current_idx + block_len] = block
                            current_idx += block_len
                        
                        elapsed = time.time() - start_time
                        
                        if elapsed > 1.0:
                            logger.warning(
                                f"Buffer concatenation took {elapsed:.2f}s for {len(buffer_blocks):,} blocks "
                                f"({buffer_total_samples:,} samples). Consider reducing chunk size if this is frequent."
                            )
                        
                        return result
                    
                    # Progress tracking
                    import time
                    last_progress_log = time.time()
                    progress_interval = 5.0  # Log progress every 5 seconds
                    last_progress_row = 0
                    
                    logger.info(
                        f"Starting PSRFITS streaming: {nsubint:,} subints, "
                        f"chunk_size={chunk_samples:,}, overlap={overlap_samples:,}"
                    )
                    logger.info(
                        f"Expected processing time: ~{nsubint * 0.01:.1f}s (estimated ~0.01s per subint). "
                        f"Large files may take longer."
                    )
                                               
                    # OPTIMIZATION: Pre-verify column existence
                    has_offs_sub_astropy = "OFFS_SUB" in subint.columns.names
                    buffer_check_interval_astropy = 10
                    
                    for i, row in enumerate(tbl):
                                                             
                        expected_start = i * nsblk
                        
                                                 
                        if expected_start > emitted:
                            gap = expected_start - emitted
                            # OPTIMIZATION: Use np.empty + fill instead of np.zeros
                            pad = np.empty((gap, 1, nchan), dtype=np.float32)
                            pad.fill(0.0)
                            buffer_blocks.append((pad, emitted))
                            buffer_total_samples += gap
                            emitted += gap
                        
                                                           
                        block = _row_to_block(row["DATA"], row, subint, nbits, nsblk, npol, nchan, zero_off, pol_type)
                        buffer_blocks.append((block, emitted))
                        buffer_total_samples += block.shape[0]
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
                                    f"Buffer: {len(buffer_blocks):,} blocks, {buffer_total_samples:,} samples | "
                                    f"Rate: {rate:.1f} subints/s | ETA: {remaining:.1f}s"
                                )
                                
                                last_progress_log = current_time
                                last_progress_row = i + 1
                        
                        # CRITICAL: Check buffer more aggressively for large chunks
                        # For large chunks, check every subint to prevent buffer explosion
                        # For normal chunks, check every N subints to reduce overhead
                        check_frequency_astropy = 1 if chunk_samples > 1_000_000 else buffer_check_interval_astropy
                        
                        # OPTIMIZATION: Check buffer based on size and block count
                        # Use buffer_total_samples (tracked without concatenation) to estimate
                        # This avoids expensive concatenation until we're ready to emit
                        if i % check_frequency_astropy == 0 or i == nsubint - 1:
                            # Check multiple conditions:
                            # 1. Have enough for complete chunk
                            # 2. Buffer exceeds memory limit
                            # 3. Too many blocks (will cause slow concatenation)
                            needs_chunk_emission = (
                                buffer_total_samples >= (chunk_samples + overlap_samples * 2) or
                                buffer_total_samples > max_buffer_samples or
                                len(buffer_blocks) > max_buffer_blocks
                            )
                            
                            # Warn if buffer is growing too large
                            if buffer_total_samples > max_buffer_samples * 0.8:
                                logger.warning(
                                    f"Buffer approaching limit: {buffer_total_samples:,}/{max_buffer_samples:,} samples "
                                    f"({len(buffer_blocks):,} blocks). Will emit chunk soon to prevent OOM."
                                )
                        else:
                            needs_chunk_emission = False
                        
                        # Only concatenate when we actually need to emit a chunk
                        # This reduces concatenation frequency from every 10 subints to only when needed
                        if needs_chunk_emission:
                            # Concatenate buffer only when we're ready to emit
                            if len(buffer_blocks) > 1:
                                # Only log if we have many blocks (indicates we've been accumulating)
                                if len(buffer_blocks) > 50:
                                    logger.debug(
                                        f"Preparing chunk emission at row {i+1:,}/{nsubint:,}: "
                                        f"{len(buffer_blocks):,} blocks, {buffer_total_samples:,} samples"
                                    )
                            out_buf = _concatenate_buffer()
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
                            
                            # Ensure buffer is concatenated for chunk extraction
                            # (should already be concatenated from the if needs_chunk_emission block above)
                            if out_buf is None:
                                out_buf = _concatenate_buffer()
                            
                            # CRITICAL: If buffer is too large, emit smaller chunks aggressively
                            # For very large chunks, emit partial chunks to prevent system freeze
                            if buffer_too_large:
                                # Calculate safe chunk size: use 50-80% of buffer to leave room
                                # This prevents system freeze by not using all available memory
                                safe_chunk_ratio = 0.7 if chunk_samples > 1_000_000 else 0.9
                                max_safe_chunk = int(out_buf.shape[0] * safe_chunk_ratio)
                                actual_chunk_size = min(chunk_samples, max_safe_chunk, out_buf.shape[0])
                                
                                # Ensure we have at least some overlap for continuity
                                if actual_chunk_size < overlap_samples * 2:
                                    actual_chunk_size = min(overlap_samples * 2, out_buf.shape[0])
                                
                                start_with_overlap = 0
                                end_with_overlap = min(actual_chunk_size + overlap_samples * 2, out_buf.shape[0])
                                valid_start = overlap_samples if end_with_overlap > overlap_samples * 2 else 0
                                valid_end = valid_start + actual_chunk_size
                                
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
                                
                                # Record buffer event for validation metrics
                                try:
                                    from ..core.data_flow_manager import _validation_collector
                                    if _validation_collector is not None:
                                        _validation_collector.record_buffer_event(
                                            buffer_size_samples=out_buf.shape[0],
                                            event_type="emergency_chunk_emission",
                                            chunk_emitted=True
                                        )
                                except Exception:
                                    pass  # Don't fail if collector not available
                            else:
                                # Normal case: emit full chunk with overlap
                                start_with_overlap = 0
                                end_with_overlap = chunk_samples + overlap_samples * 2
                                valid_start = overlap_samples
                                valid_end = valid_start + chunk_samples
                                actual_chunk_size = chunk_samples
                            
                            block_out = out_buf[start_with_overlap:end_with_overlap].copy()
                            
                                                                         
                            if config.DATA_NEEDS_REVERSAL:
                                block_out = block_out[:, :, ::-1]
                            
                                                                                                                 
                                                                           
                            start_sample_idx = emitted - out_buf.shape[0]
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
                                "overlap_left": overlap_samples,
                                "overlap_right": overlap_samples,
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
                                "tstart_mjd_corr": tstart_mjd + (nsuboffs * tsub) / 86400.0,
                                "tsubint_sec": tsub,
                            }
                            yield block_out, metadata
                                                                         
                            # Remove emitted chunk from buffer (keep overlap for next chunk)
                            samples_to_remove = actual_chunk_size if not buffer_too_large else end_with_overlap
                            
                            # Update buffer: remove emitted samples, keep overlap
                            # Rebuild buffer_blocks list from remaining data
                            if samples_to_remove >= out_buf.shape[0]:
                                # All buffer was emitted
                                buffer_blocks = []
                                buffer_total_samples = 0
                            else:
                                # Keep tail of buffer (overlap)
                                remaining_buf = out_buf[samples_to_remove:]
                                if remaining_buf.shape[0] > 0:
                                    # Rebuild buffer_blocks with remaining data
                                    buffer_blocks = [(remaining_buf, emitted - remaining_buf.shape[0])]
                                    buffer_total_samples = remaining_buf.shape[0]
                                else:
                                    buffer_blocks = []
                                    buffer_total_samples = 0
                            
                            out_buf = _concatenate_buffer() if buffer_blocks else np.zeros((0, 1, nchan), dtype=np.float32)
                            
                            # Update buffer size check for next iteration
                            buffer_total_samples = out_buf.shape[0]  # Update tracked total
                            buffer_too_large = out_buf.shape[0] > max_buffer_samples
                            has_complete_chunk = out_buf.shape[0] >= (chunk_samples + overlap_samples * 2)
                            needs_chunk_emission = has_complete_chunk or buffer_too_large
                    
                                            
                    # Handle remaining buffer at end of file
                    if buffer_blocks:
                        out_buf = _concatenate_buffer()
                    if out_buf.shape[0] > 0:
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
                        metadata = {
                            "chunk_idx": (emitted - out_buf.shape[0]) // max(1, chunk_samples),
                            "start_sample": emitted - out_buf.shape[0],                                     
                            "end_sample": emitted,
                            "actual_chunk_size": valid_end - valid_start,
                            "block_start_sample": emitted - out_buf.shape[0],
                            "block_end_sample": emitted,
                            "overlap_left": 0,
                            "overlap_right": 0,
                            "total_samples": total_samples,
                            "nchans": nchan,
                            "nifs": 1,
                            "dtype": str(block_out.dtype),
                            "shape": block_out.shape,
                            "file_type": "fits",
                                                                                       
                            "tbin_sec": tbin,
                            "t_rel_start_sec": (emitted - out_buf.shape[0]) * tbin,
                            "t_rel_end_sec": emitted * tbin,
                                                                        
                            "tstart_mjd": tstart_mjd,
                            "tstart_mjd_corr": tstart_mjd + (nsuboffs * tsub) / 86400.0,
                            "tsubint_sec": tsub,
                        }
                        yield block_out, metadata
                    
                    log_stream_fits_summary(chunk_counter)
                    return
                else:
                    raise ValueError(
                        f"FITS file does not have a valid SUBINT structure: {file_name}"
                    )
        
    except Exception as e:
        logger.error("Error in stream_fits: %s", e)
        raise ValueError(f"Could not read FITS file {file_name}") from e
