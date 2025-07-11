from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from . import config

__all__ = [
    "find_data_files",
    "slice_parameters",
    "load_fil_chunk",
    "write_summary",
]


def find_data_files(frb: str) -> List[Path]:
    """Return FITS or filterbank files matching ``frb`` in ``config.DATA_DIR``."""
    files = list(config.DATA_DIR.glob("*.fits")) + list(config.DATA_DIR.glob("*.fil"))
    return sorted(f for f in files if frb in f.name)


def slice_parameters(width_total: int, slice_len: int) -> tuple[int, int]:
    """Return adjusted ``slice_len`` and number of slices."""
    if width_total == 0:
        return 0, 0
    if width_total < slice_len:
        return width_total, 1
    return slice_len, width_total // slice_len


def load_fil_chunk(file_path: str, start_sample: int, chunk_size: int) -> np.ndarray:
    """Load a specific chunk from a filterbank file."""
    from .filterbank_io import _read_header

    with open(file_path, "rb") as f:
        header, hdr_len = _read_header(f)

    nchans = header["nchans"]
    nbits = header["nbits"]
    nifs = header.get("nifs", 1)
    bytes_per_sample = nifs * nchans * (nbits // 8)

    data_start_offset = hdr_len + start_sample * bytes_per_sample
    bytes_to_read = chunk_size * bytes_per_sample

    dtype = np.uint8
    if nbits == 16:
        dtype = np.int16
    elif nbits == 32:
        dtype = np.float32
    elif nbits == 64:
        dtype = np.float64

    with open(file_path, "rb") as f:
        f.seek(data_start_offset)
        raw_data = f.read(bytes_to_read)

    data_flat = np.frombuffer(raw_data, dtype=dtype)

    if len(data_flat) == chunk_size * nchans * nifs:
        data = data_flat.reshape(chunk_size, nifs, nchans)
    else:
        available = len(data_flat) // (nchans * nifs)
        data = data_flat[: available * nchans * nifs].reshape(available, nifs, nchans)

    if getattr(config, "DATA_NEEDS_REVERSAL", False):
        data = data[:, :, ::-1]
    return data


def write_summary(summary: dict, save_path: Path) -> None:
    """Write global summary information to ``summary.json``."""
    summary_path = save_path / "summary.json"
    with summary_path.open("w") as f_json:
        json.dump(summary, f_json, indent=2)

