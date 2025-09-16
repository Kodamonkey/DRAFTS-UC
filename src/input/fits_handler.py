# This module handles FITS and PSRFITS data ingestion.

"""Manejo de archivos FITS y PSRFITS para el pipeline de detección de FRBs."""
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
    """Selecciona/compone la polarización según config.

    - data: (nsamp, npol, nchan)
    - pol_type: POL_TYPE del header (e.g., "IQUV", "AABB", etc.)
    - mode: "intensity", "linear", "circular", o "pol{idx}"
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
                        f"Dimensiones inválidas en header FITS: NAXIS2={nsubint}, NCHAN={nchan}, NPOL={npol}, NSBLK={nsblk} (no pueden ser <= 0)"
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
                            raise ValueError(f"NBITS={nbits} no soportado")
                                                                              
                                                                     
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
                    raise ImportError("fitsio no está instalado. Instale con: pip install fitsio")
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
                        
                        error_msg = f"Dimensiones inválidas en header FITS: {', '.join(error_details)}"
                        if total_samples <= 0:
                            error_msg += f"\n  → NSBLK={h.get('NSBLK', 1)} es 0 o negativo, lo que hace imposible calcular el número de muestras temporales"
                            error_msg += f"\n  → El pipeline necesita datos en formato (tiempo, polarización, canal) pero no puede determinar la dimensión temporal"
                        if num_pols <= 0:
                            error_msg += f"\n  → NPOL={num_pols} es 0 o negativo, lo que hace imposible procesar las polarizaciones"
                        if num_chans <= 0:
                            error_msg += f"\n  → NCHAN={num_chans} es 0 o negativo, lo que hace imposible procesar los canales de frecuencia"
                        
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
                        raise ValueError(f"Error al hacer reshape de los datos (fitsio): {e}\n  → Los datos no pueden reorganizarse en el formato esperado (tiempo={total_samples}, pol={num_pols}, canal={num_chans})")
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
                        
                        error_msg = f"Dimensiones inválidas en header FITS: {', '.join(error_details)}"
                        if total_samples <= 0:
                            error_msg += f"\n  → NSBLK={h.get('NSBLK', 1)} es 0 o negativo, lo que hace imposible calcular el número de muestras temporales"
                            error_msg += f"\n  → El pipeline necesita datos en formato (tiempo, polarización, canal) pero no puede determinar la dimensión temporal"
                        if num_pols <= 0:
                            error_msg += f"\n  → NPOL={num_pols} es 0 o negativo, lo que hace imposible procesar las polarizaciones"
                        if num_chans <= 0:
                            error_msg += f"\n  → NCHAN={num_chans} es 0 o negativo, lo que hace imposible procesar los canales de frecuencia"
                        
                        raise ValueError(error_msg)
                    try:
                        data_array = temp_data.reshape(total_samples, num_pols, num_chans)
                                                                                                   
                        data_array = data_array[:, 0:1, :]
                    except Exception as e:
                        raise ValueError(f"Error al hacer reshape de los datos (fitsio): {e}\n  → Los datos no pueden reorganizarse en el formato esperado (tiempo={total_samples}, pol={num_pols}, canal={num_chans})")
    except (ValueError, fits.verify.VerifyError) as e:
                                                                                       
        if "NSBLK" in str(e) and "0" in str(e):
            raise ValueError(f"Archivo FITS corrupto: {file_name}\n  → El archivo tiene NSBLK=0 en el header, lo que indica que está mal formateado o corrupto\n  → NSBLK debe ser > 0 para definir el número de muestras por bloque temporal\n  → Recomendación: Verificar el origen del archivo o obtener una versión correcta") from e
        elif "Dimensiones inválidas" in str(e):
            raise ValueError(f"Archivo FITS corrupto: {file_name}\n  → {str(e)}\n  → El archivo no puede ser procesado debido a dimensiones inválidas en el header\n  → Recomendación: Verificar la integridad del archivo o usar un archivo diferente") from e
        else:
            raise ValueError(f"Archivo FITS corrupto: {file_name}\n  → {str(e)}\n  → El archivo no puede ser leído correctamente\n  → Recomendación: Verificar que el archivo no esté dañado") from e
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
                            
                            error_msg = f"Dimensiones inválidas en header fallback: {', '.join(error_details)}"
                            if total_samples <= 0:
                                error_msg += f"\n  → NSBLK={h.get('NSBLK', 1)} es 0 o negativo, lo que hace imposible calcular el número de muestras temporales"
                                error_msg += f"\n  → El pipeline necesita datos en formato (tiempo, polarización, canal) pero no puede determinar la dimensión temporal"
                            if num_pols <= 0:
                                error_msg += f"\n  → NPOL={num_pols} es 0 o negativo, lo que hace imposible procesar las polarizaciones"
                            if num_chans <= 0:
                                error_msg += f"\n  → NCHAN={num_chans} es 0 o negativo, lo que hace imposible procesar los canales de frecuencia"
                            
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
                            raise ValueError(f"Error al hacer reshape de los datos (fallback): {e}\n  → Los datos no pueden reorganizarse en el formato esperado (tiempo={total_samples}, pol={num_pols}, canal={num_chans})\n  → El archivo puede estar corrupto o tener un formato no compatible")
                    else:
                        raise ValueError("No hay datos válidos en el HDU\n  → El archivo FITS no contiene datos procesables\n  → Verificar que el archivo no esté vacío o corrupto")
                except (TypeError, ValueError) as e_data:
                    logger.debug(f"Error accediendo a datos del HDU: {e_data}")
                    raise ValueError(f"Archivo FITS corrupto: {file_name}\n  → Error al acceder a los datos del archivo: {e_data}\n  → El archivo puede estar dañado o tener un formato no reconocido")
        except Exception as e_astropy:
            logger.debug(f"Fallo final al cargar con astropy: {e_astropy}")
            raise ValueError(f"Archivo FITS corrupto: {file_name}\n  → Fallo en el método de respaldo con astropy: {e_astropy}\n  → El archivo no puede ser leído por ningún método disponible\n  → Recomendación: Verificar la integridad del archivo o usar un archivo diferente") from e_astropy
            
    if data_array is None:
        raise ValueError(f"Archivo FITS corrupto: {file_name}\n  → No se pudieron cargar datos válidos del archivo\n  → El archivo puede estar vacío, corrupto o tener un formato no compatible\n  → Recomendación: Verificar la integridad del archivo o usar un archivo diferente")

    if global_vars.DATA_NEEDS_REVERSAL:
        logger.debug(f">> Invirtiendo eje de frecuencia de los datos cargados para {file_name}")
        data_array = np.ascontiguousarray(data_array[:, :, ::-1])
    
                                              
    if config.DEBUG_FREQUENCY_ORDER:
        logger.debug(f"[DEBUG DATOS CARGADOS] Archivo: {file_name}")
        logger.debug(f"[DEBUG DATOS CARGADOS] Shape de datos: {data_array.shape}")
        logger.debug(f"[DEBUG DATOS CARGADOS] Dimensiones: (tiempo={data_array.shape[0]}, pol={data_array.shape[1]}, freq={data_array.shape[2]})")
        logger.debug(f"[DEBUG DATOS CARGADOS] Tipo de datos: {data_array.dtype}")
        logger.debug(f"[DEBUG DATOS CARGADOS] Tamaño en memoria: {data_array.nbytes / (1024**3):.2f} GB")
        logger.debug(f"[DEBUG DATOS CARGADOS] Reversión aplicada: {global_vars.DATA_NEEDS_REVERSAL}")
        logger.debug(f"[DEBUG DATOS CARGADOS] Rango de valores: [{data_array.min():.3f}, {data_array.max():.3f}]")
        logger.debug(f"[DEBUG DATOS CARGADOS] Valor medio: {data_array.mean():.3f}")
        logger.debug(f"[DEBUG DATOS CARGADOS] Desviación estándar: {data_array.std():.3f}")
        logger.debug("[DEBUG DATOS CARGADOS] " + "="*50)
    
    return data_array


def get_obparams(file_name: str) -> None:
    """Extract observation parameters and populate :mod:`config`."""
    
                                               
    if config.DEBUG_FREQUENCY_ORDER:
        logger.debug(f"[DEBUG HEADER] Iniciando extracción de parámetros de: {file_name}")
        logger.debug(f"[DEBUG HEADER] " + "="*60)
    
    with fits.open(file_name, memmap=True) as f:
        freq_axis_inverted = False
        
                                            
        if config.DEBUG_FREQUENCY_ORDER:
            logger.debug(f"[DEBUG HEADER] Estructura del archivo FITS:")
            for i, hdu in enumerate(f):
                hdu_type = type(hdu).__name__
                if hasattr(hdu, 'header') and hdu.header:
                    if 'EXTNAME' in hdu.header:
                        ext_name = hdu.header['EXTNAME']
                    else:
                        ext_name = 'PRIMARY' if i == 0 else f'HDU_{i}'
                    logger.debug(f"[DEBUG HEADER]   HDU {i}: {hdu_type} - {ext_name}")
                    if hasattr(hdu, 'columns') and hdu.columns:
                        logger.debug(f"[DEBUG HEADER]     Columnas: {[col.name for col in hdu.columns]}")
                else:
                    logger.debug(f"[DEBUG HEADER]   HDU {i}: {hdu_type} - Sin header")
        
        if "SUBINT" in [hdu.name for hdu in f] and "TBIN" in f["SUBINT"].header:
                                               
            if config.DEBUG_FREQUENCY_ORDER:
                logger.debug(f"[DEBUG HEADER] Formato detectado: PSRFITS (SUBINT)")
            
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
                    logger.debug(f"[DEBUG HEADER] Error convirtiendo DAT_FREQ: {e}")
                    logger.debug("[DEBUG HEADER] Usando rango de frecuencias por defecto")
                nchan = safe_int(hdr.get("NCHAN", 512), 512)
                freq_temp = np.linspace(1000, 1500, nchan)
            
                                                
            if config.DEBUG_FREQUENCY_ORDER:
                logger.debug(f"[DEBUG HEADER] Headers PSRFITS extraídos:")
                logger.debug(
                    f"[DEBUG HEADER]   TBIN (resolución temporal): {safe_float(hdr.get('TBIN')):.2e} s"
                )
                logger.debug(f"[DEBUG HEADER]   NCHAN (canales): {hdr['NCHAN']}")
                logger.debug(f"[DEBUG HEADER]   NSBLK (muestras por subint): {hdr['NSBLK']}")
                logger.debug(f"[DEBUG HEADER]   NAXIS2 (número de subints): {hdr['NAXIS2']}")
                logger.debug(f"[DEBUG HEADER]   NPOL (polarizaciones): {hdr.get('NPOL', 'N/A')}")
                logger.debug(f"[DEBUG HEADER]   Total de muestras: {config.FILE_LENG}")
                if 'OBS_MODE' in hdr:
                    logger.debug(f"[DEBUG HEADER]   Modo de observación: {hdr['OBS_MODE']}")
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
                        logger.debug(f"[DEBUG HEADER] DAT_FREQ ascendente → invertir banda (estilo radioastronomía)")
        else:
                                                     
            if config.DEBUG_FREQUENCY_ORDER:
                logger.debug(f"[DEBUG HEADER] Formato detectado: FITS estándar (no PSRFITS)")
            
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
                    logger.debug(f"[DEBUG HEADER] HDU seleccionado para datos: {data_hdu_index}")
                
                hdr = f[data_hdu_index].header
                
                                              
                if config.DEBUG_FREQUENCY_ORDER:
                    logger.debug(f"[DEBUG HEADER] Headers FITS estándar del HDU {data_hdu_index}:")
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
                            logger.debug(f"[DEBUG HEADER] Error convirtiendo DAT_FREQ: {e}")
                            logger.debug("[DEBUG HEADER] Usando rango de frecuencias por defecto")
                        nchan = safe_int(hdr.get("NCHAN", 512), 512)
                        freq_temp = np.linspace(1000, 1500, nchan)
                    else:
                        if config.DEBUG_FREQUENCY_ORDER:
                            logger.debug(f"[DEBUG HEADER] Frecuencias extraídas de columna DAT_FREQ")
                else:
                    freq_axis_num = ''
                    for i in range(1, hdr.get('NAXIS', 0) + 1):
                        if 'FREQ' in hdr.get(f'CTYPE{i}', '').upper():
                            freq_axis_num = str(i)
                            break
                    
                    if config.DEBUG_FREQUENCY_ORDER:
                        logger.debug(f"[DEBUG HEADER] Buscando eje de frecuencias en headers WCS...")
                        logger.debug(f"[DEBUG HEADER] Eje de frecuencias detectado: CTYPE{freq_axis_num}" if freq_axis_num else "[DEBUG HEADER] No se encontró eje de frecuencias")
                    
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
                            logger.debug(f"[DEBUG HEADER] Parámetros WCS frecuencia:")
                            logger.debug(f"[DEBUG HEADER]   CRVAL{freq_axis_num}: {crval} (valor de referencia)")
                            logger.debug(f"[DEBUG HEADER]   CDELT{freq_axis_num}: {cdelt} (incremento por canal)")
                            logger.debug(f"[DEBUG HEADER]   CRPIX{freq_axis_num}: {crpix} (pixel de referencia)")
                            logger.debug(f"[DEBUG HEADER]   NAXIS{freq_axis_num}: {naxis} (número de canales)")
                        
                        if cdelt < 0:
                            freq_axis_inverted = True
                            if config.DEBUG_FREQUENCY_ORDER:
                                logger.debug(f"[DEBUG HEADER]   [WARNING] CDELT negativo - frecuencias invertidas!")
                        else:
                                                                                                                            
                            freq_axis_inverted = True
                            if config.DEBUG_FREQUENCY_ORDER:
                                logger.debug(f"[DEBUG HEADER]   [WARNING] CDELT positivo - invirtiendo para estándar radioastronomía!")
                    else:
                        if config.DEBUG_FREQUENCY_ORDER:
                            logger.debug(f"[DEBUG HEADER] [WARNING] Usando frecuencias por defecto: 1000-1500 MHz")
                        freq_temp = np.linspace(1000, 1500, hdr.get('NCHAN', 512))
                
                                                                                
                config.TIME_RESO = safe_float(hdr.get("TBIN"))
                config.FREQ_RESO = safe_int(hdr.get("NCHAN", len(freq_temp)))
                config.FILE_LENG = safe_int(hdr.get("NAXIS2", 0)) * safe_int(hdr.get("NSBLK", 1))
                
                                                     
                if config.DEBUG_FREQUENCY_ORDER:
                    logger.debug(f"[DEBUG HEADER] Parámetros finales FITS estándar:")
                    logger.debug(f"[DEBUG HEADER]   TIME_RESO: {config.TIME_RESO:.2e} s")
                    logger.debug(f"[DEBUG HEADER]   FREQ_RESO: {config.FREQ_RESO}")
                    logger.debug(f"[DEBUG HEADER]   FILE_LENG: {config.FILE_LENG}")
                    
            except Exception as e_std:
                if config.DEBUG_FREQUENCY_ORDER:
                    logger.debug(f"[DEBUG HEADER] [WARNING] Error procesando FITS estándar: {e_std}")
                    logger.debug(f"[DEBUG HEADER] Usando valores por defecto...")
                logger.debug(f"Error procesando FITS estándar: {e_std}")
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
        print_debug_frequencies("[DEBUG FRECUENCIAS]", file_name, freq_axis_inverted)

                                             
    if config.DEBUG_FREQUENCY_ORDER:
        logger.debug(f"[DEBUG ARCHIVO] Información completa del archivo: {file_name}")
        logger.debug(f"[DEBUG ARCHIVO] " + "="*60)
        logger.debug(f"[DEBUG ARCHIVO] DIMENSIONES Y RESOLUCIÓN:")
        logger.debug(f"[DEBUG ARCHIVO]   - Resolución temporal: {config.TIME_RESO:.2e} segundos/muestra")
        logger.debug(f"[DEBUG ARCHIVO]   - Resolución de frecuencia: {config.FREQ_RESO} canales")
        logger.debug(f"[DEBUG ARCHIVO]   - Longitud del archivo: {config.FILE_LENG:,} muestras")
        
                                 
        duracion_total_seg = config.FILE_LENG * config.TIME_RESO
        duracion_min = duracion_total_seg / 60
        duracion_horas = duracion_min / 60
        logger.debug(f"[DEBUG ARCHIVO]   - Duración total: {duracion_total_seg:.2f} seg ({duracion_min:.2f} min, {duracion_horas:.2f} h)")
        
        logger.debug(f"[DEBUG ARCHIVO] FRECUENCIAS:")
        logger.debug(f"[DEBUG ARCHIVO]   - Rango total: {config.FREQ.min():.2f} - {config.FREQ.max():.2f} MHz")
        logger.debug(f"[DEBUG ARCHIVO]   - Ancho de banda: {abs(config.FREQ.max() - config.FREQ.min()):.2f} MHz")
        logger.debug(f"[DEBUG ARCHIVO]   - Resolución por canal: {abs(config.FREQ[1] - config.FREQ[0]):.4f} MHz/canal")
        logger.debug(f"[DEBUG ARCHIVO]   - Orden original: {'DESCENDENTE' if freq_axis_inverted else 'ASCENDENTE'}")
        logger.debug(f"[DEBUG ARCHIVO]   - Orden final (post-corrección): {'ASCENDENTE' if config.FREQ[0] < config.FREQ[-1] else 'DESCENDENTE'}")
        
        logger.debug(f"[DEBUG ARCHIVO] DECIMACIÓN:")
        logger.debug(f"[DEBUG ARCHIVO]   - Factor reducción frecuencia: {config.DOWN_FREQ_RATE}x")
        logger.debug(f"[DEBUG ARCHIVO]   - Factor reducción tiempo: {config.DOWN_TIME_RATE}x")
        logger.debug(f"[DEBUG ARCHIVO]   - Canales después de decimación: {config.FREQ_RESO // config.DOWN_FREQ_RATE}")
        logger.debug(f"[DEBUG ARCHIVO]   - Resolución temporal después: {config.TIME_RESO * config.DOWN_TIME_RATE:.2e} seg/muestra")
        
                                             
        size_original_gb = (config.FILE_LENG * config.FREQ_RESO * 4) / (1024**3)                       
        size_decimated_gb = size_original_gb / (config.DOWN_FREQ_RATE * config.DOWN_TIME_RATE)
        logger.debug(f"[DEBUG ARCHIVO] TAMAÑO ESTIMADO:")
        logger.debug(f"[DEBUG ARCHIVO]   - Datos originales: ~{size_original_gb:.2f} GB")
        logger.debug(f"[DEBUG ARCHIVO]   - Datos después decimación: ~{size_decimated_gb:.2f} GB")
        
        
        logger.debug(f"[DEBUG ARCHIVO] CONFIGURACIÓN DE SLICE:")
        logger.debug(f"[DEBUG ARCHIVO]   - SLICE_DURATION_MS configurado: {config.SLICE_DURATION_MS} ms")
        expected_slice_len = round(config.SLICE_DURATION_MS / (config.TIME_RESO * config.DOWN_TIME_RATE * 1000))
        logger.debug(f"[DEBUG ARCHIVO]   - SLICE_LEN calculado: {expected_slice_len} muestras")
        logger.debug(f"[DEBUG ARCHIVO]   - SLICE_LEN límites: [{config.SLICE_LEN_MIN}, {config.SLICE_LEN_MAX}]")
        
        logger.debug(f"[DEBUG ARCHIVO] PROCESAMIENTO:")
        logger.debug(f"[DEBUG ARCHIVO]   - Multi-banda habilitado: {'SÍ' if config.USE_MULTI_BAND else 'NO'}")
        logger.debug(f"[DEBUG ARCHIVO]   - DM rango: {config.DM_min} - {config.DM_max} pc cm⁻³")
        logger.debug(f"[DEBUG ARCHIVO]   - Umbrales: DET_PROB={config.DET_PROB}, CLASS_PROB={config.CLASS_PROB}, SNR_THRESH={config.SNR_THRESH}")
        logger.debug(f"[DEBUG ARCHIVO] " + "="*60)

                                                                                    
    auto_config_downsampling()

                                              
    if config.DEBUG_FREQUENCY_ORDER:
        logger.debug(f"[DEBUG CONFIG FINAL] Configuración final después de get_obparams:")
        logger.debug(f"[DEBUG CONFIG FINAL] " + "="*60)
        logger.debug(f"[DEBUG CONFIG FINAL] DOWN_FREQ_RATE calculado: {config.DOWN_FREQ_RATE}x")
        logger.debug(f"[DEBUG CONFIG FINAL] DOWN_TIME_RATE calculado: {config.DOWN_TIME_RATE}x")
        logger.debug(f"[DEBUG CONFIG FINAL] Datos después de decimación:")
        logger.debug(f"[DEBUG CONFIG FINAL]   - Canales: {config.FREQ_RESO // config.DOWN_FREQ_RATE}")
        logger.debug(f"[DEBUG CONFIG FINAL]   - Resolución temporal: {config.TIME_RESO * config.DOWN_TIME_RATE:.2e} s/muestra")
        logger.debug(f"[DEBUG CONFIG FINAL]   - Reducción total de datos: {config.DOWN_FREQ_RATE * config.DOWN_TIME_RATE}x")
        logger.debug(f"[DEBUG CONFIG FINAL] DATA_NEEDS_REVERSAL final: {config.DATA_NEEDS_REVERSAL}")
        logger.debug(f"[DEBUG CONFIG FINAL] Orden de frecuencias final: {'ASCENDENTE' if config.FREQ[0] < config.FREQ[-1] else 'DESCENDENTE'}")
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
                "original_order": "DESCENDENTE" if freq_axis_inverted else "ASCENDENTE",
                "final_order": "ASCENDENTE" if config.FREQ[0] < config.FREQ[-1] else "DESCENDENTE",
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


def stream_fits(
    file_name: str,
    chunk_samples: int = 2_097_152,
    overlap_samples: int = 0,
) -> Generator[Tuple[np.ndarray, Dict], None, None]:
    """
    Generador que lee un archivo FITS en bloques sin cargar todo en RAM.
    
    Args:
        file_name: Ruta al archivo .fits
        chunk_samples: Número de muestras por bloque (default: 2M)
        overlap_samples: Número de muestras de solapamiento entre bloques
    
    Yields:
        Tuple[data_block, metadata]: Bloque de datos (time, pol, chan) y metadatos
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
                raise ValueError(f"NBITS={nbits} no soportado")
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
                        raise ValueError("Forma inesperada en 'your' get_data")
                                                                                  
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
                                            
                    total_samples = nsubint * nsblk
                    logger.info(
                        "FITS data detected: samples=%d, polarisations=%d, channels=%d",
                        total_samples,
                        npol,
                        nchan,
                    )
                    log_stream_fits_parameters(total_samples, chunk_samples, overlap_samples, nsubint, nchan, npol, nsblk)

                                                        
                    out_buf = np.empty((0, 1, nchan), dtype=np.float32)
                    emitted = 0                     

                                                                                

                                                                                                        
                    def _expected_start_sample(offs_sub_seconds: float, sub_index: int) -> int:
                        if offs_sub_seconds is not None:
                                                                    
                            start_sec = float(offs_sub_seconds) - 0.5 * tsub
                            return int(round(start_sec / tbin))
                                                                       
                        return (nsuboffs + sub_index) * nsblk

                                                
                    chunk_counter = 0
                                         
                    pol_type = str(hdr.get("POL_TYPE", "")).upper() if hdr.get("POL_TYPE") is not None else ""
                    for isub in range(nsubint):
                        row = tbl[isub]
                        offs_sub_val = None
                        if "OFFS_SUB" in subint.columns.names:
                            try:
                                offs_sub_val = safe_float(row["OFFS_SUB"])     
                            except Exception:
                                offs_sub_val = None

                        expected_start = _expected_start_sample(offs_sub_val, isub)
                                                     
                        if emitted < expected_start:
                            gap = expected_start - emitted
                            pad = np.zeros((gap, 1, nchan), dtype=np.float32)
                                                  
                            out_buf = np.concatenate([out_buf, pad], axis=0)
                            emitted += gap

                                                           
                        block = _row_to_block(row["DATA"], row, subint, nbits, nsblk, npol, nchan, zero_off, pol_type)
                        out_buf = np.concatenate([out_buf, block], axis=0)
                        emitted += block.shape[0]

                                                                        
                        while out_buf.shape[0] >= (chunk_samples + overlap_samples * 2):
                            chunk_counter += 1
                            start_with_overlap = 0
                            end_with_overlap = chunk_samples + overlap_samples * 2
                            valid_start = overlap_samples
                            valid_end = valid_start + chunk_samples
                            block_out = out_buf[start_with_overlap:end_with_overlap].copy()
                                                 
                            start_sample_idx = emitted - out_buf.shape[0] + valid_start
                            end_sample_idx = start_sample_idx + chunk_samples
                                         
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
                                "actual_chunk_size": chunk_samples,
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
                                                                         
                            out_buf = out_buf[chunk_samples:]

                                            
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
                    raise ImportError("fitsio no está instalado. Instale con: pip install fitsio")
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
                    
                    logger.info(
                        "Streaming PSRFITS (astropy fallback): nsubint=%d, nchan=%d, npol=%d, tbin=%s",
                        nsubint,
                        nchan,
                        npol,
                        tbin,
                    )
                    
                                                                        
                    pol_type = str(hdr.get("POL_TYPE", "")).upper() if hdr.get("POL_TYPE") is not None else ""
                    
                                                      
                    total_samples = nsubint * nsblk
                    log_stream_fits_parameters(total_samples, chunk_samples, overlap_samples, None, nchan, npol, None)
                    
                                                           
                    out_buf = np.zeros((0, 1, nchan), dtype=np.float32)
                    emitted = 0
                    chunk_counter = 0
                    
                                               
                    for i, row in enumerate(tbl):
                                                             
                        expected_start = i * nsblk
                        
                                                 
                        if expected_start > emitted:
                            gap = expected_start - emitted
                            pad = np.zeros((gap, 1, nchan), dtype=np.float32)
                                                  
                            out_buf = np.concatenate([out_buf, pad], axis=0)
                            emitted += gap
                        
                                                           
                        block = _row_to_block(row["DATA"], row, subint, nbits, nsblk, npol, nchan, zero_off, pol_type)
                        out_buf = np.concatenate([out_buf, block], axis=0)
                        emitted += block.shape[0]
                        
                                                                        
                        while out_buf.shape[0] >= (chunk_samples + overlap_samples * 2):
                            chunk_counter += 1
                            start_with_overlap = 0
                            end_with_overlap = chunk_samples + overlap_samples * 2
                            valid_start = overlap_samples
                            valid_end = valid_start + chunk_samples
                            block_out = out_buf[start_with_overlap:end_with_overlap].copy()
                            
                                                                         
                            if config.DATA_NEEDS_REVERSAL:
                                block_out = block_out[:, :, ::-1]
                            
                                                                                                                 
                                                                           
                            start_sample_idx = emitted - out_buf.shape[0]
                            end_sample_idx = start_sample_idx + chunk_samples
                            
                                         
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
                                "actual_chunk_size": chunk_samples,
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
                                                                         
                            out_buf = out_buf[chunk_samples:]
                    
                                            
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
                    raise ValueError(f"Archivo FITS no tiene estructura SUBINT válida: {file_name}")
        
    except Exception as e:
        logger.error("Error in stream_fits: %s", e)
        raise ValueError(f"No se pudo leer el archivo FITS {file_name}") from e
