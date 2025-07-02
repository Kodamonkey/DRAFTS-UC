"""Helper functions to read SIGPROC filterbank (.fil) files."""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Tuple

import numpy as np

from . import config


def _read_int(f) -> int:
    return struct.unpack("<i", f.read(4))[0]


def _read_double(f) -> float:
    return struct.unpack("<d", f.read(8))[0]


def _read_string(f) -> str:
    length = _read_int(f)
    return f.read(length).decode()


def _read_header(f) -> Tuple[dict, int]:
    start = _read_string(f)
    if start != "HEADER_START":
        raise ValueError("Invalid filterbank file")

    header = {}
    while True:
        key = _read_string(f)
        if key == "HEADER_END":
            break
        if key in {"rawdatafile", "source_name"}:
            header[key] = _read_string(f)
        elif key in {
            "telescope_id",
            "machine_id",
            "data_type",
            "barycentric",
            "pulsarcentric",
            "nbits",
            "nchans",
            "nifs",
            "nbeams",
            "ibeam",
            "nsamples",
        }:
            header[key] = _read_int(f)
        elif key in {
            "az_start",
            "za_start",
            "src_raj",
            "src_dej",
            "tstart",
            "tsamp",
            "fch1",
            "foff",
            "refdm",
        }:
            header[key] = _read_double(f)
        else:
            # Read unknown field as integer by default
            header[key] = _read_int(f)
    return header, f.tell()


def load_fil_file(file_name: str) -> np.ndarray:
    """Load a filterbank file and return data as ``(time, pol, channel)``."""
    with open(file_name, "rb") as f:
        header, hdr_len = _read_header(f)

    nchans = header.get("nchans", 0)
    nifs = header.get("nifs", 1)
    nbits = header.get("nbits", 8)
    nsamples = header.get("nsamples")
    if nsamples is None:
        bytes_per_sample = nifs * nchans * (nbits // 8)
        file_size = os.path.getsize(file_name) - hdr_len
        nsamples = file_size // bytes_per_sample

    dtype = np.uint8
    if nbits == 16:
        dtype = np.int16
    elif nbits == 32:
        dtype = np.float32
    elif nbits == 64:
        dtype = np.float64

    # Memory-map the data to avoid loading the entire file into memory
    data = np.memmap(
        file_name,
        dtype=dtype,
        mode="r",
        offset=hdr_len,
        shape=(nsamples, nifs, nchans),
    )

    if config.DATA_NEEDS_REVERSAL:
        data = np.ascontiguousarray(data[:, :, ::-1])

    return data


def get_obparams_fil(file_name: str) -> None:
    """Populate :mod:`config` using parameters from a filterbank file."""
    with open(file_name, "rb") as f:
        header, hdr_len = _read_header(f)

    nchans = header.get("nchans", 0)
    tsamp = header.get("tsamp", 0.0)
    nifs = header.get("nifs", 1)
    nbits = header.get("nbits", 8)
    nsamples = header.get("nsamples")
    if nsamples is None:
        bytes_per_sample = nifs * nchans * (nbits // 8)
        file_size = os.path.getsize(file_name) - hdr_len
        nsamples = file_size // bytes_per_sample

    fch1 = header.get("fch1", 0.0)
    foff = header.get("foff", 0.0)
    freq_temp = fch1 + np.arange(nchans) * foff
    if foff < 0:
        config.DATA_NEEDS_REVERSAL = True
        freq_temp = freq_temp[::-1]
    else:
        config.DATA_NEEDS_REVERSAL = False

    config.FREQ = freq_temp
    config.FREQ_RESO = nchans
    config.TIME_RESO = tsamp
    config.FILE_LENG = nsamples

    if config.FREQ_RESO >= 512:
        config.DOWN_FREQ_RATE = max(1, int(round(config.FREQ_RESO / 512)))
    else:
        config.DOWN_FREQ_RATE = 1
    if config.TIME_RESO > 1e-9:
        config.DOWN_TIME_RATE = max(1, int((49.152 * 16 / 1e6) / config.TIME_RESO))
    else:
        config.DOWN_TIME_RATE = 15
