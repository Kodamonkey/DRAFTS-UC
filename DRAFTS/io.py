"""Input/output helpers for PSRFITS and standard FITS files."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

import numpy as np
from astropy.io import fits

from . import config
from .config import Config
from .frequency_utils import normalize_frequency_order

logger = logging.getLogger(__name__)


def _find_data_hdu(hdulist: fits.HDUList) -> int:
    """Return the index of the HDU containing spectral data."""
    for i, hdu_item in enumerate(hdulist):
        if hdu_item.is_image or isinstance(hdu_item, (fits.BinTableHDU, fits.TableHDU)):
            if hdu_item.header.get("NAXIS", 0) > 0:
                for ax in ("CTYPE3", "CTYPE2", "CTYPE1"):
                    if "FREQ" in hdu_item.header.get(ax, "").upper():
                        return i
    return 1 if len(hdulist) > 1 else 0


def _extract_standard_fits_frequency(hdu: fits.hdu.base.ExtensionHDU) -> tuple[np.ndarray, bool]:
    """Return frequency array for a standard FITS HDU."""
    hdr = hdu.header
    if "DAT_FREQ" in hdu.columns.names:
        freq = hdu.data["DAT_FREQ"][0].astype(np.float64)
        cdelt = None
    else:
        freq_axis_num = next(
            (str(i) for i in range(1, hdr.get("NAXIS", 0) + 1) if "FREQ" in hdr.get(f"CTYPE{i}", "").upper()),
            "",
        )
        if freq_axis_num:
            crval = hdr.get(f"CRVAL{freq_axis_num}", 0)
            cdelt = hdr.get(f"CDELT{freq_axis_num}", 1)
            crpix = hdr.get(f"CRPIX{freq_axis_num}", 1)
            naxis = hdr.get(f"NAXIS{freq_axis_num}", hdr.get("NCHAN", 512))
            freq = crval + (np.arange(naxis) - (crpix - 1)) * cdelt
        else:
            freq = np.linspace(1000, 1500, hdr.get("NCHAN", 512))
            cdelt = None
    return normalize_frequency_order(freq, cdelt)


def _open_fits_data(file_name: str) -> tuple[np.ndarray, fits.Header]:
    """Return raw data array and header from ``file_name``."""
    with fits.open(file_name, memmap=True) as hdul:
        if "SUBINT" in hdul and "DATA" in hdul["SUBINT"].columns.names:
            subint = hdul["SUBINT"]
            return subint.data["DATA"], subint.header
        data_hdu = hdul[_find_data_hdu(hdul)]
        return data_hdu.data, data_hdu.header


def _reshape_fits_data(data: np.ndarray, hdr: fits.Header) -> np.ndarray:
    """Reshape raw FITS data to ``(time, pol, channel)``."""
    nsubint = hdr.get("NAXIS2", 1)
    nchan = hdr.get("NCHAN", data.shape[-1])
    npol = hdr.get("NPOL", 1)
    nsblk = hdr.get("NSBLK", 1)
    arr = data.reshape(nsubint, nchan, npol, nsblk).swapaxes(1, 2)
    arr = arr.reshape(nsubint * nsblk, npol, nchan)
    return arr[:, :2, :]


def _collect_debug_info(data: np.ndarray, cfg: Config) -> dict:
    """Return debug information for ``data`` and ``cfg``."""
    return {
        "shape": data.shape,
        "dtype": str(data.dtype),
        "freq_min": float(cfg.FREQ.min()),
        "freq_max": float(cfg.FREQ.max()),
        "time_resolution": cfg.TIME_RESO,
    }


def load_fits_file(
    file_name: str,
    cfg: Config | None = None,
    *,
    update_summary_cb: Callable[[Path, str, dict], None] | None = None,
) -> np.ndarray:
    """Return data array from ``file_name`` shaped as ``(time, pol, channel)``."""
    if cfg is None:
        cfg = get_obparams(file_name)

    data, hdr = _open_fits_data(file_name)
    data_array = _reshape_fits_data(data, hdr)

    if cfg.DATA_NEEDS_REVERSAL:
        logger.debug("Reversing frequency axis for %s", file_name)
        data_array = np.ascontiguousarray(data_array[:, :, ::-1])

    if update_summary_cb:
        update_summary_cb(Path(file_name).parent, Path(file_name).stem, _collect_debug_info(data_array, cfg))

    return data_array


def get_obparams(file_name: str) -> Config:
    """Return observation parameters from ``file_name``."""
    with fits.open(file_name, memmap=True) as f:
        if "SUBINT" in f and "TBIN" in f["SUBINT"].header:
            hdr = f["SUBINT"].header
            freq, inverted = normalize_frequency_order(
                f["SUBINT"].data["DAT_FREQ"][0].astype(np.float64), hdr.get("CHAN_BW")
            )
            cfg = Config(
                TIME_RESO=hdr["TBIN"],
                FREQ_RESO=hdr["NCHAN"],
                FILE_LENG=hdr["NSBLK"] * hdr["NAXIS2"],
                FREQ=freq,
                DATA_NEEDS_REVERSAL=inverted,
            )
        else:
            idx = _find_data_hdu(f)
            hdr = f[idx].header
            freq, inverted = _extract_standard_fits_frequency(f[idx])
            cfg = Config(
                TIME_RESO=hdr.get("TBIN", 5.12e-5),
                FREQ_RESO=hdr.get("NCHAN", len(freq)),
                FILE_LENG=hdr.get("NAXIS2", 0) * hdr.get("NSBLK", 1),
                FREQ=freq,
                DATA_NEEDS_REVERSAL=inverted,
            )
    cfg.DOWN_FREQ_RATE = max(1, int(round(cfg.FREQ_RESO / 512))) if cfg.FREQ_RESO >= 512 else 1
    cfg.DOWN_TIME_RATE = max(1, int((49.152 * 16 / 1e6) / cfg.TIME_RESO)) if cfg.TIME_RESO > 1e-9 else 15
    return cfg


def _save_file_debug_info_fits(
    file_name: str,
    debug_info: dict,
    update_cb: Callable[[Path, str, dict], None],
) -> None:
    """Save debug information using ``update_cb``."""
    try:
        results_dir = getattr(config, "RESULTS_DIR", Path("./Results/ObjectDetection"))
        model_dir = results_dir / config.MODEL_NAME
        model_dir.mkdir(parents=True, exist_ok=True)
        update_cb(model_dir, Path(file_name).stem, debug_info)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Error saving debug info for %s: %s", file_name, exc)

def open_data_file(path: str) -> tuple[np.ndarray, Config]:
    """Open FITS or filterbank files using the appropriate loader."""
    ext = Path(path).suffix.lower()
    if ext == ".fits":
        cfg = get_obparams(path)
        return load_fits_file(path, cfg), cfg
    if ext == ".fil":
        from .filterbank_io import get_obparams_fil, load_fil_file
        cfg = get_obparams_fil(path)
        return load_fil_file(path, cfg), cfg
    raise ValueError(f"Unsupported file extension: {ext}")
