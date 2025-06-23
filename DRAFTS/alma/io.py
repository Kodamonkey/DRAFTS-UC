"""Input/output helpers for PSRFITS and standard FITS files for ALMA."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from astropy.io import fits

from . import config


def _get_column_index(header: fits.Header, name: str) -> Optional[int]:
    """Return the index of column ``name`` in ``header`` or ``None``."""
    tfields = header.get("TFIELDS", 0)
    for i in range(1, tfields + 1):
        if header.get(f"TTYPE{i}", "").strip() == name:
            return i
    return None


def _parse_tdim(header: fits.Header, index: int) -> Optional[List[int]]:
    """Parse the ``TDIM`` value for the given column index."""
    tdim = header.get(f"TDIM{index}")
    if tdim:
        tdim = tdim.strip("() ")
        try:
            return [int(v) for v in tdim.split(",")]
        except ValueError:
            return None
    return None


def load_fits_file(file_name: str) -> np.ndarray:
    """Load a FITS file and return the data array in shape (time, pol, channel)."""
    global_vars = config
    data_array = None
    try:
        with fits.open(file_name, memmap=True) as hdul:
            data_hdu = None
            for hdu in hdul:
                if hasattr(hdu, "columns") and "DATA" in hdu.columns.names:
                    data_hdu = hdu
                    break
            if data_hdu is None:
                for hdu in hdul:
                    if (
                        getattr(hdu, "data", None) is not None
                        and isinstance(hdu.data, np.ndarray)
                        and hdu.data.ndim >= 2
                    ):
                        data_hdu = hdu
                        break
            if data_hdu is None:
                raise ValueError("No se encontró un HDU con datos válidos")

            hdr = data_hdu.header
            if hasattr(data_hdu, "columns") and "DATA" in data_hdu.columns.names:
                col_idx = _get_column_index(hdr, "DATA")
                dims = _parse_tdim(hdr, col_idx) if col_idx else None
                nsub = hdr.get("NAXIS2", len(data_hdu.data))
                if dims and len(dims) >= 3:
                    if len(dims) == 4:
                        nbin, nchan, npol, nsblk = dims
                    else:
                        nbin, nchan, npol = dims[:3]
                        nsblk = 1
                else:
                    nbin = hdr.get("NBIN", 1)
                    nchan = hdr.get("NCHAN", hdr.get("OBSNCHAN", 1))
                    npol = hdr.get("NPOL", 1)
                    nsblk = hdr.get("NSBLK", 1)
                data = np.asarray(data_hdu.data["DATA"])
                data_array = data.reshape(nsub, nbin, nchan, npol, nsblk)
            else:
                data = np.asarray(data_hdu.data)
                if data.ndim == 2:
                    nsub, nchan = data.shape
                    npol = 1
                    nbin = nsblk = 1
                    data_array = data.reshape(nsub, nbin, nchan, npol, nsblk)
                elif data.ndim == 3:
                    nsub = data.shape[0]
                    if data.shape[1] <= 4:
                        npol = data.shape[1]
                        nchan = data.shape[2]
                        nbin = nsblk = 1
                        data_array = data.reshape(nsub, nbin, nchan, npol, nsblk)
                    else:
                        nchan = data.shape[1]
                        npol = data.shape[2]
                        nbin = nsblk = 1
                        data_array = data.reshape(nsub, nbin, nchan, npol, nsblk)
                else:
                    raise ValueError("Dimensión de datos no soportada")

            data_array = data_array.transpose(0, 1, 3, 2, 4)
            data_array = data_array.reshape(nsub * nbin * nsblk, npol, nchan)
            data_array = data_array[:, :2, :]
    except Exception as e:
        print(f"[Error cargando FITS] {e}")
        raise

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
        if "SUBINT" in [hdu.name for hdu in f] and "DATA" in f["SUBINT"].columns.names:
            hdr = f["SUBINT"].header
            sub_data = f["SUBINT"].data
            config.TIME_RESO = hdr.get("TBIN", config.TIME_RESO)
            config.FREQ_RESO = hdr.get("NCHAN", len(sub_data["DAT_FREQ"][0]))
            nsblk = hdr.get("NSBLK")
            nbin = hdr.get("NBIN", 1)
            if nsblk is None or not isinstance(nsblk, (int, np.integer)):
                nsblk = 1
            config.FILE_LENG = hdr.get("NAXIS2", 1) * max(nsblk, 1) * max(nbin, 1)
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
                config.TIME_RESO = hdr.get("TBIN", config.TIME_RESO)
                config.FREQ_RESO = hdr.get("NCHAN", len(freq_temp))
                nsblk = hdr.get("NSBLK")
                nbin = hdr.get("NBIN", 1)
                if nsblk is None or not isinstance(nsblk, (int, np.integer)):
                    nsblk = 1
                config.FILE_LENG = hdr.get("NAXIS2", 0) * max(nsblk, 1) * max(nbin, 1)
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
    # For ALMA data we keep the original time resolution.
    config.DOWN_TIME_RATE = 1
