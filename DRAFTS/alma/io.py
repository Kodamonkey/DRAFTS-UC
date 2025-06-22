"""Input/output helpers for PSRFITS and standard FITS files."""

from __future__ import annotations

from typing import List

import numpy as np
from astropy.io import fits

from . import config


def _parse_float(value: object, default: float = 0.0) -> float:
    """Safely convert FITS header values to ``float``."""

    try:
        if isinstance(value, bytes):
            value = value.decode()
        if isinstance(value, str):
            value = value.strip()
            if value in {"*", "UNSET", ""}:
                return default
        return float(value)
    except Exception:
        return default


def _parse_int(value: object, default: int = 0) -> int:
    """Safely convert FITS header values to ``int``."""

    try:
        if isinstance(value, bytes):
            value = value.decode()
        if isinstance(value, str):
            value = value.strip()
            if value in {"*", "UNSET", ""}:
                return default
        return int(value)
    except Exception:
        return default


def load_fits_file(file_name: str) -> np.ndarray:
    """Load a FITS file and return the data array in shape (time, pol, channel)."""
    global_vars = config
    data_array = None
    try:
        with fits.open(file_name, memmap=True) as hdul:
            if (
                "SUBINT" in [hdu.name for hdu in hdul]
                and "DATA" in hdul["SUBINT"].columns.names
            ):
                subint = hdul["SUBINT"]
                hdr = subint.header
                data_array = subint.data["DATA"]
                nsubint = _parse_int(hdr.get("NAXIS2", len(data_array)))
                nchan = _parse_int(hdr.get("NCHAN", 0))
                npol = _parse_int(hdr.get("NPOL", 1))
                nsblk = _parse_int(hdr.get("NSBLK", 0))
                if nsblk == 0:
                    nsblk = data_array.shape[1] // max(nchan * npol, 1)
                data_array = data_array.reshape(nsubint, nchan, npol, nsblk).swapaxes(
                    1, 2
                )
                data_array = data_array.reshape(nsubint * nsblk, npol, nchan)
                data_array = data_array[:, :2, :]
            else:
                import fitsio

                temp_data, h = fitsio.read(file_name, header=True)
                if "DATA" in temp_data.dtype.names:
                    nsub = _parse_int(h.get("NAXIS2", len(temp_data)))
                    nsblk = _parse_int(h.get("NSBLK", 0))
                    npol = _parse_int(h.get("NPOL", 1))
                    nchan = _parse_int(h.get("NCHAN", 0))
                    if nsblk == 0 and nchan and npol:
                        nsblk = temp_data["DATA"].shape[1] // (nchan * npol)
                    data_array = temp_data["DATA"].reshape(
                        nsub * nsblk,
                        npol,
                        nchan,
                    )[:, :2, :]
                else:
                    total_samples = _parse_int(h.get("NAXIS2", 1)) * _parse_int(h.get("NSBLK", 1))
                    num_pols = _parse_int(h.get("NPOL", 2))
                    num_chans = _parse_int(h.get("NCHAN", 512))
                    data_array = temp_data.reshape(total_samples, num_pols, num_chans)[
                        :, :2, :
                    ]
    except Exception as e:
        print(f"[Error cargando FITS con fitsio/astropy] {e}")
        try:
            with fits.open(file_name) as f:
                data_hdu = None
                for hdu_item in f:
                    if (
                        hdu_item.data is not None
                        and isinstance(hdu_item.data, np.ndarray)
                        and hdu_item.data.ndim >= 3
                    ):
                        data_hdu = hdu_item
                        break
                if data_hdu is None and len(f) > 1:
                    data_hdu = f[1]
                elif data_hdu is None:
                    data_hdu = f[0]
                h = data_hdu.header
                raw_data = data_hdu.data
                nsub = _parse_int(h.get("NAXIS2", raw_data.shape[0]))
                nsblk = _parse_int(h.get("NSBLK", 0))
                npol = _parse_int(h.get("NPOL", 1))
                nchan = _parse_int(h.get("NCHAN", raw_data.shape[-1]))
                if nsblk == 0 and nchan and npol:
                    nsblk = raw_data.size // (nsub * nchan * npol)
                data_array = raw_data.reshape(nsub * nsblk, npol, nchan)[:, :2, :]
        except Exception as e_astropy:
            print(f"Fallo final al cargar con astropy: {e_astropy}")
            raise
    if data_array is None:
        raise ValueError(f"No se pudieron cargar los datos de {file_name}")

    if global_vars.DATA_NEEDS_REVERSAL:
        print(
            f">> Invirtiendo eje de frecuencia de los datos cargados para {file_name}"
        )
        data_array = np.ascontiguousarray(data_array[:, :, ::-1])
    return data_array


def get_obparams(file_name: str) -> None:
    """Extract observation parameters and populate :mod:`config`."""
    with fits.open(file_name, memmap=True) as f:
        freq_axis_inverted = False
        if "SUBINT" in [hdu.name for hdu in f] and "TBIN" in f["SUBINT"].header:
            hdr = f["SUBINT"].header
            sub_data = f["SUBINT"].data
            config.TIME_RESO = _parse_float(hdr.get("TBIN"))
            config.FREQ_RESO = _parse_int(hdr.get("NCHAN"))
            nsblk = _parse_int(hdr.get("NSBLK"), 1)
            naxis2 = _parse_int(hdr.get("NAXIS2", 0))
            if naxis2 == 0:
                naxis2 = len(sub_data)
            config.FILE_LENG = nsblk * naxis2
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
                    if hdu_item.is_image or isinstance(
                        hdu_item, (fits.BinTableHDU, fits.TableHDU)
                    ):
                        if "NAXIS" in hdu_item.header and hdu_item.header["NAXIS"] > 0:
                            if (
                                "CTYPE3" in hdu_item.header
                                and "FREQ" in hdu_item.header["CTYPE3"].upper()
                            ):
                                data_hdu_index = i
                                break
                            if (
                                "CTYPE2" in hdu_item.header
                                and "FREQ" in hdu_item.header["CTYPE2"].upper()
                            ):
                                data_hdu_index = i
                                break
                            if (
                                "CTYPE1" in hdu_item.header
                                and "FREQ" in hdu_item.header["CTYPE1"].upper()
                            ):
                                data_hdu_index = i
                                break
                if data_hdu_index == 0 and len(f) > 1:
                    data_hdu_index = 1
                hdr = f[data_hdu_index].header
                if "DAT_FREQ" in f[data_hdu_index].columns.names:
                    freq_temp = f[data_hdu_index].data["DAT_FREQ"][0].astype(np.float64)
                else:
                    freq_axis_num = ""
                    for i in range(1, hdr.get("NAXIS", 0) + 1):
                        if "FREQ" in hdr.get(f"CTYPE{i}", "").upper():
                            freq_axis_num = str(i)
                            break
                    if freq_axis_num:
                        crval = hdr.get(f"CRVAL{freq_axis_num}", 0)
                        cdelt = hdr.get(f"CDELT{freq_axis_num}", 1)
                        crpix = hdr.get(f"CRPIX{freq_axis_num}", 1)
                        naxis = hdr.get(f"NAXIS{freq_axis_num}", hdr.get("NCHAN", 512))
                        freq_temp = crval + (np.arange(naxis) - (crpix - 1)) * cdelt
                        if cdelt < 0:
                            freq_axis_inverted = True
                    else:
                        freq_temp = np.linspace(1000, 1500, hdr.get("NCHAN", 512))
                config.TIME_RESO = _parse_float(hdr.get("TBIN"))
                config.FREQ_RESO = _parse_int(hdr.get("NCHAN", len(freq_temp)))
                nsblk = _parse_int(hdr.get("NSBLK"), 1)
                naxis2 = _parse_int(hdr.get("NAXIS2", 0))
                if naxis2 == 0 and "DATA" in f[data_hdu_index].columns.names:
                    naxis2 = len(f[data_hdu_index].data)
                config.FILE_LENG = nsblk * naxis2
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
