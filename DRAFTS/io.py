"""Input/output helpers for PSRFITS and standard FITS files."""
from __future__ import annotations

from typing import List, Any

import numpy as np
from astropy.io import fits

from . import config


def _to_int(value: Any, default: int = 0) -> int:
    """Convert FITS header values to ``int`` safely."""
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def load_fits_file(file_name: str) -> np.ndarray:
    """Load a FITS file and return the data array in shape (time, pol, channel)."""
    global_vars = config
    data_array = None
    try:
        with fits.open(file_name, memmap=True) as hdul:
            if "SUBINT" in [hdu.name for hdu in hdul] and "DATA" in hdul["SUBINT"].columns.names:
                subint = hdul["SUBINT"]
                hdr = subint.header
                data_array = subint.data["DATA"]
                nsubint = _to_int(hdr.get("NAXIS2"))
                nchan = _to_int(hdr.get("NCHAN"))
                npol = _to_int(hdr.get("NPOL"))
                nsblk = _to_int(hdr.get("NSBLK"))
                data_array = data_array.reshape(nsubint, nchan, npol, nsblk).swapaxes(1, 2)
                data_array = data_array.reshape(nsubint * nsblk, npol, nchan)
                data_array = data_array[:, :2, :]
            else:
                import fitsio
                temp_data, h = fitsio.read(file_name, header=True)
                if "DATA" in temp_data.dtype.names:
                    total = _to_int(h.get("NAXIS2", 1)) * _to_int(h.get("NSBLK", 1))
                    num_pols = _to_int(h.get("NPOL", 2))
                    num_chans = _to_int(h.get("NCHAN", temp_data["DATA"].shape[-1]))
                    data_array = temp_data["DATA"].reshape(total, num_pols, num_chans)[:, :2, :]
                else:
                    total_samples = _to_int(h.get("NAXIS2", 1)) * _to_int(h.get("NSBLK", 1))
                    num_pols = _to_int(h.get("NPOL", 2))
                    num_chans = _to_int(h.get("NCHAN", temp_data.shape[-1]))
                    data_array = temp_data.reshape(total_samples, num_pols, num_chans)[:, :2, :]
    except Exception as e:
        print(f"[Error cargando FITS con fitsio/astropy] {e}")
        try:
            # Intentar sin memmap para archivos corruptos
            with fits.open(file_name, memmap=False) as f:
                data_hdu = None
                for hdu_item in f:
                    # Evitar acceder a .data directamente, usar hasattr primero
                    try:
                        if (hdu_item.data is not None and 
                            isinstance(hdu_item.data, np.ndarray) and 
                            hdu_item.data.ndim >= 3):
                            data_hdu = hdu_item
                            break
                    except (TypeError, ValueError):
                        # Si no se puede acceder a los datos, saltar este HDU
                        continue
                        
                if data_hdu is None and len(f) > 1:
                    data_hdu = f[1]
                elif data_hdu is None:
                    data_hdu = f[0]
                    
                h = data_hdu.header
                try:
                    raw_data = data_hdu.data
                    if raw_data is not None:
                        total = _to_int(h.get("NAXIS2", raw_data.shape[0])) * _to_int(h.get("NSBLK", 1))
                        num_pols = _to_int(h.get("NPOL", 2))
                        num_chans = _to_int(h.get("NCHAN", raw_data.shape[-1]))
                        data_array = raw_data.reshape(total, num_pols, num_chans)[:, :2, :]
                    else:
                        raise ValueError("No hay datos válidos en el HDU")
                except (TypeError, ValueError) as e_data:
                    print(f"Error accediendo a datos del HDU: {e_data}")
                    raise ValueError(f"Archivo FITS corrupto: {file_name}")
                    
        except Exception as e_astropy:
            print(f"Fallo final al cargar con astropy: {e_astropy}")
            raise ValueError(f"No se puede leer el archivo FITS corrupto: {file_name}")
            
    if data_array is None:
        raise ValueError(f"No se pudieron cargar los datos de {file_name}")

    if global_vars.DATA_NEEDS_REVERSAL:
        print(f">> Invirtiendo eje de frecuencia de los datos cargados para {file_name}")
        data_array = np.ascontiguousarray(data_array[:, :, ::-1])
    return data_array


def get_obparams(file_name: str) -> None:
    """Extract observation parameters and populate :mod:`config`."""
    with fits.open(file_name, memmap=True) as f:
        freq_axis_inverted = False
        if "SUBINT" in [hdu.name for hdu in f] and "TBIN" in f["SUBINT"].header:
            hdr = f["SUBINT"].header
            sub_data = f["SUBINT"].data
            tbin_val = hdr["TBIN"]
            if isinstance(tbin_val, str):
                try:
                    tbin_val = float(tbin_val)
                except ValueError:
                    tbin_val = 0.0
            config.TIME_RESO = tbin_val
            config.FREQ_RESO = hdr["NCHAN"]
            config.FILE_LENG = hdr["NSBLK"] * hdr["NAXIS2"]
            freq_temp = sub_data["DAT_FREQ"][0].astype(np.float64)
            if "CHAN_BW" in hdr:
                bw = hdr["CHAN_BW"]
                if isinstance(bw, (list, np.ndarray)):
                    bw = bw[0]
                if bw < 0:
                    freq_axis_inverted = True
            elif len(freq_temp) > 1 and freq_temp[0] > freq_temp[-1]:
                freq_axis_inverted = True
        else:
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
                hdr = f[data_hdu_index].header
                if "DAT_FREQ" in f[data_hdu_index].columns.names:
                    freq_temp = f[data_hdu_index].data["DAT_FREQ"][0].astype(np.float64)
                else:
                    freq_axis_num = ''
                    for i in range(1, hdr.get('NAXIS', 0) + 1):
                        if 'FREQ' in hdr.get(f'CTYPE{i}', '').upper():
                            freq_axis_num = str(i)
                            break
                    if freq_axis_num:
                        crval = hdr.get(f'CRVAL{freq_axis_num}', 0)
                        cdelt = hdr.get(f'CDELT{freq_axis_num}', 1)
                        crpix = hdr.get(f'CRPIX{freq_axis_num}', 1)
                        naxis = hdr.get(f'NAXIS{freq_axis_num}', hdr.get('NCHAN', 512))
                        freq_temp = crval + (np.arange(naxis) - (crpix - 1)) * cdelt
                        if cdelt < 0:
                            freq_axis_inverted = True
                    else:
                        freq_temp = np.linspace(1000, 1500, hdr.get('NCHAN', 512))
                tbin_val = hdr["TBIN"]
                if isinstance(tbin_val, str):
                    try:
                        tbin_val = float(tbin_val)
                    except ValueError:
                        tbin_val = 0.0
                config.TIME_RESO = tbin_val
                config.FREQ_RESO = hdr.get("NCHAN", len(freq_temp))
                config.FILE_LENG = hdr.get("NAXIS2", 0) * hdr.get("NSBLK", 1)
            except Exception as e_std:
                print(f"Error procesando FITS estándar: {e_std}")
                config.TIME_RESO = 5.12e-5
                config.FREQ_RESO = 512
                config.FILE_LENG = 100000
                freq_temp = np.linspace(1000, 1500, config.FREQ_RESO)
        if freq_axis_inverted:
            config.FREQ = freq_temp[::-1]
            config.DATA_NEEDS_REVERSAL = True
        else:
            config.FREQ = freq_temp
            config.DATA_NEEDS_REVERSAL = False

    if config.FREQ_RESO >= 512:
        config.DOWN_FREQ_RATE = max(1, int(round(config.FREQ_RESO / 512)))
    else:
        config.DOWN_FREQ_RATE = 1
    if config.TIME_RESO > 1e-9:
        config.DOWN_TIME_RATE = max(1, int((49.152 * 16 / 1e6) / config.TIME_RESO))
    else:
        config.DOWN_TIME_RATE = 15
