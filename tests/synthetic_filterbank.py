"""Write SIGPROC filterbank files with a dispersed pulse injected at a known DM.

Used by the end-to-end tests: it is the only way to exercise the reader, the
chunking geometry and the DM-time cube against a ground truth without shipping
real observations.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from src.analysis.science_metrics import K_DM_MS

_INT_KEYS = {
    "telescope_id", "machine_id", "data_type", "barycentric", "pulsarcentric",
    "nbits", "nchans", "nifs", "nbeams", "ibeam", "nsamples",
}
_DOUBLE_KEYS = {
    "az_start", "za_start", "src_raj", "src_dej", "tstart", "tsamp",
    "fch1", "foff", "refdm",
}


def _pack_string(value: str) -> bytes:
    raw = value.encode("ascii")
    return struct.pack("<i", len(raw)) + raw


def _pack_header(header: dict) -> bytes:
    out = [_pack_string("HEADER_START")]
    for key, value in header.items():
        out.append(_pack_string(key))
        if key in _INT_KEYS:
            out.append(struct.pack("<i", int(value)))
        elif key in _DOUBLE_KEYS:
            out.append(struct.pack("<d", float(value)))
        else:
            raise KeyError(f"unknown SIGPROC key: {key}")
    out.append(_pack_string("HEADER_END"))
    return b"".join(out)


def channel_frequencies(fch1: float, foff: float, nchans: int) -> np.ndarray:
    """Frequency of each stored channel, in the order they are written."""
    return fch1 + foff * np.arange(nchans, dtype=np.float64)


def dispersion_delay_s(dm: float, freq_mhz: np.ndarray, ref_mhz: float) -> np.ndarray:
    """Arrival delay of *freq_mhz* relative to *ref_mhz*, in seconds.

    Uses the single project-wide constant (SPEC-DM-001); ``K_DM_MS`` yields
    seconds for frequencies in MHz.
    """
    return K_DM_MS * float(dm) * (np.asarray(freq_mhz, dtype=np.float64) ** -2 - float(ref_mhz) ** -2)


def write_filterbank(
    path: Path,
    *,
    nsamples: int,
    nchans: int = 128,
    tsamp: float = 0.001,
    fch1: float = 1500.0,
    foff: float = -1.0,
    tstart: float | None = 60000.0,
    dm: float | None = None,
    burst_time_s: float | None = None,
    burst_width_samples: int = 4,
    amplitude: float = 90.0,
    background: float = 80.0,
    noise_sigma: float = 4.0,
    seed: int = 0,
) -> dict:
    """Write an 8-bit filterbank and return the ground truth used to build it.

    A pulse is injected at ``dm``/``burst_time_s`` following the same dispersion
    law the pipeline dedisperses with, so a correct pipeline recovers both.
    ``tstart=None`` omits the key, which is what a header without an epoch looks
    like to the reader.
    """
    rng = np.random.RandomState(seed)
    freqs = channel_frequencies(fch1, foff, nchans)

    data = rng.normal(background, noise_sigma, size=(nsamples, nchans))

    truth: dict = {
        "nsamples": nsamples, "nchans": nchans, "tsamp": tsamp,
        "fch1": fch1, "foff": foff, "tstart": tstart,
        "freqs": freqs, "dm": dm, "burst_time_s": burst_time_s,
    }

    if dm is not None and burst_time_s is not None:
        ref = float(np.max(freqs))
        delays = dispersion_delay_s(dm, freqs, ref)
        centres = (burst_time_s + delays) / tsamp
        half = max(1, burst_width_samples // 2)
        for ch, centre in enumerate(centres):
            lo = int(round(centre)) - half
            hi = int(round(centre)) + half + 1
            lo_c, hi_c = max(0, lo), min(nsamples, hi)
            if hi_c > lo_c:
                data[lo_c:hi_c, ch] += amplitude
        truth["ref_freq_mhz"] = ref
        truth["burst_sample_at_ref"] = int(round(burst_time_s / tsamp))

    block = np.clip(data, 0, 255).astype(np.uint8)

    header = {
        "telescope_id": 6,
        "machine_id": 0,
        "data_type": 1,
        "barycentric": 0,
        "pulsarcentric": 0,
        "nbits": 8,
        "nchans": nchans,
        "nifs": 1,
        "nbeams": 1,
        "ibeam": 1,
        "fch1": fch1,
        "foff": foff,
        "tsamp": tsamp,
    }
    if tstart is not None:
        header["tstart"] = tstart

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(_pack_header(header))
        fh.write(block.tobytes())

    truth["path"] = path
    return truth
