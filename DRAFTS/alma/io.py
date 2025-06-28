"""Input/output helpers for PSRFITS and standard FITS files."""
from __future__ import annotations

from typing import List
import warnings

import numpy as np
from astropy.io import fits

from . import config
from ..aliases import get_header_value, get_column_value


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
                nsubint = int(get_header_value(hdr, "NAXIS2", 1))
                nchan = int(get_header_value(hdr, "NCHAN", 0))
                npol = int(get_header_value(hdr, "NPOL", 2))
                nsblk = int(get_header_value(hdr, "NSBLK", 1))
                data_array = data_array.reshape(nsubint, nchan, npol, nsblk).swapaxes(1, 2)
                data_array = data_array.reshape(nsubint * nsblk, npol, nchan)[:, :2, :]
            else:
                import fitsio
                temp_data, h = fitsio.read(file_name, header=True)
                if "DATA" in temp_data.dtype.names:
                    nsubint = int(get_header_value(h, "NAXIS2", 1))
                    nchan = int(get_header_value(h, "NCHAN", 0))
                    npol = int(get_header_value(h, "NPOL", 2))
                    nsblk = int(get_header_value(h, "NSBLK", 1))
                    data_array = temp_data["DATA"].reshape(nsubint * nsblk, npol, nchan)[:, :2, :]
                else:
                    total_samples = int(get_header_value(h, "NAXIS2", 1)) * int(get_header_value(h, "NSBLK", 1))
                    num_pols = int(get_header_value(h, "NPOL", 2))
                    num_chans = int(get_header_value(h, "NCHAN", 512))
                    data_array = temp_data.reshape(total_samples, num_pols, num_chans)[:, :2, :]
    except Exception as e:
        print(f"[Error cargando FITS con fitsio/astropy] {e}")
        try:
            with fits.open(file_name) as f:
                data_hdu = None
                for hdu_item in f:
                    if hdu_item.data is not None and isinstance(hdu_item.data, np.ndarray) and hdu_item.data.ndim >= 3:
                        data_hdu = hdu_item
                        break
                if data_hdu is None and len(f) > 1:
                    data_hdu = f[1]
                elif data_hdu is None:
                    data_hdu = f[0]
                h = data_hdu.header
                raw_data = data_hdu.data
                data_array = raw_data.reshape(h["NAXIS2"] * h.get("NSBLK", 1), h.get("NPOL", 2), h.get("NCHAN", raw_data.shape[-1]))[:, :2, :]
        except Exception as e_astropy:
            print(f"Fallo final al cargar con astropy: {e_astropy}")
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
        if "SUBINT" in [hdu.name for hdu in f] and get_header_value(f["SUBINT"].header, "TBIN") is not None:
            hdr = f["SUBINT"].header
            sub_data = f["SUBINT"].data
            config.TIME_RESO = float(get_header_value(hdr, "TBIN", 0.0))
            config.FREQ_RESO = int(get_header_value(hdr, "NCHAN", 0))
            config.FILE_LENG = int(get_header_value(hdr, "NSBLK", 1)) * int(get_header_value(hdr, "NAXIS2", 1))
            col = get_column_value(f["SUBINT"], "DAT_FREQ")
            if col is not None:
                freq_temp = col[0].astype(np.float64)
            else:
                edge = get_column_value(f["SUBINT"], "EDGE_CHANNEL")
                if edge is not None:
                    edge = edge[0].astype(np.float64)
                    freq_temp = 0.5 * (edge[:-1] + edge[1:])
                else:
                    crval = get_header_value(hdr, "CRVAL1", get_header_value(hdr, "RESTFRQ", 0))
                    cdelt = get_header_value(hdr, "CDELT1", 1)
                    crpix = get_header_value(hdr, "CRPIX1", 1)
                    nchan = config.FREQ_RESO
                    freq_temp = crval + (np.arange(nchan) - (crpix - 1)) * cdelt
                    if cdelt and cdelt < 0:
                        freq_axis_inverted = True
            bw = get_header_value(hdr, "CHAN_BW")
            if bw is not None and float(bw) < 0:
                freq_axis_inverted = True
            elif len(freq_temp) > 1 and freq_temp[0] > freq_temp[-1]:
                freq_axis_inverted = True
        else:
            try:
                data_hdu_index = 0
                for i, hdu_item in enumerate(f):
                    if hdu_item.is_image or isinstance(hdu_item, (fits.BinTableHDU, fits.TableHDU)):
                        if 'NAXIS' in hdu_item.header and hdu_item.header['NAXIS'] > 0:
                            for ax in ["3", "2", "1"]:
                                if f"CTYPE{ax}" in hdu_item.header and "FREQ" in hdu_item.header[f"CTYPE{ax}"].upper():
                                    data_hdu_index = i
                                    break
                            if data_hdu_index:
                                break
                if data_hdu_index == 0 and len(f) > 1:
                    data_hdu_index = 1
                hdr = f[data_hdu_index].header
                col = get_column_value(f[data_hdu_index], "DAT_FREQ")
                if col is not None:
                    freq_temp = col[0].astype(np.float64)
                else:
                    edge = get_column_value(f[data_hdu_index], "EDGE_CHANNEL")
                    if edge is not None:
                        edge = edge[0].astype(np.float64)
                        freq_temp = 0.5 * (edge[:-1] + edge[1:])
                    else:
                        freq_axis_num = ""
                        for i in range(1, hdr.get("NAXIS", 0) + 1):
                            if "FREQ" in str(hdr.get(f"CTYPE{i}", "")).upper():
                                freq_axis_num = str(i)
                                break
                        crval = get_header_value(hdr, f"CRVAL{freq_axis_num}", get_header_value(hdr, "RESTFRQ", 0))
                        cdelt = get_header_value(hdr, f"CDELT{freq_axis_num}", 1)
                        crpix = get_header_value(hdr, f"CRPIX{freq_axis_num}", 1)
                        naxis = get_header_value(hdr, f"NAXIS{freq_axis_num}", get_header_value(hdr, "NCHAN", 512))
                        freq_temp = crval + (np.arange(naxis) - (crpix - 1)) * cdelt
                        if cdelt and cdelt < 0:
                            freq_axis_inverted = True
                config.TIME_RESO = float(get_header_value(hdr, "TBIN", 0.0))
                config.FREQ_RESO = int(get_header_value(hdr, "NCHAN", len(freq_temp)))
                config.FILE_LENG = int(get_header_value(hdr, "NAXIS2", 0)) * int(get_header_value(hdr, "NSBLK", 1))
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

    # For ALMA data we keep the original time resolution to
    # preserve the microsecond sampling necessary for high
    # frequency observations.
    config.DOWN_TIME_RATE = 1
