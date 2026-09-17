"""Write PSRFITS SUBINT (SEARCH-mode) files that this repository's readers accept.

The companion of :mod:`tests.synthetic_filterbank`, for the other input format.
Nothing in the suite used to read a ``.fits`` file, so ``stream_fits`` -- four
reader branches inside one function -- had no test at all.

The HDUs are built with ``astropy.io.fits`` directly rather than through
``your.formats.fitswriter.initialize_psrfits``: that helper needs a ``Your``
object wrapped around an existing filterbank, and it always writes a start
epoch, so it cannot produce the no-epoch file audit P1-09 is about. The header
keywords below are the ones ``your.formats.psrfits.SpectraInfo`` reads (it
raises ``KeyError`` on any it is missing), so files written here are readable by
``your.Your`` and by plain ``astropy.io.fits`` alike -- both matter, because
``stream_fits`` has branches for each.

As in ``synthetic_filterbank``, an injected burst is swept with the pipeline's
own ``K_DM_MS`` (SPEC-DM-001), so the ground truth is literally shared between
the two writers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from src.domain.physics import K_DM_MS

__all__ = [
    "K_DM_MS",
    "channel_frequencies",
    "dispersion_delay_s",
    "write_psrfits",
]


def channel_frequencies(fch1: float, foff: float, nchan: int) -> np.ndarray:
    """Frequency of each stored channel, in the order DAT_FREQ lists them."""
    return float(fch1) + float(foff) * np.arange(int(nchan), dtype=np.float64)


def dispersion_delay_s(dm: float, freq_mhz: np.ndarray, ref_mhz: float) -> np.ndarray:
    """Arrival delay of *freq_mhz* relative to *ref_mhz*, in seconds.

    Uses the single project-wide constant (SPEC-DM-001); ``K_DM_MS`` yields
    seconds for frequencies in MHz.
    """
    return K_DM_MS * float(dm) * (
        np.asarray(freq_mhz, dtype=np.float64) ** -2 - float(ref_mhz) ** -2
    )


def _mjd_to_stt(mjd: float) -> tuple[int, int, float]:
    """Split an MJD into the PSRFITS triple ``(STT_IMJD, STT_SMJD, STT_OFFS)``."""
    imjd = int(np.floor(mjd))
    seconds = (float(mjd) - imjd) * 86400.0
    smjd = int(np.floor(seconds))
    return imjd, smjd, seconds - smjd


_DATA_FORMAT = {"uint8": "B", "float32": "E"}


def write_psrfits(
    path: Path | str,
    *,
    nsubint: int = 8,
    nsblk: int = 64,
    nchan: int = 32,
    npol: int = 1,
    tsamp: float = 1.0e-3,
    fch1: float = 1500.0,
    foff: float = -1.0,
    tstart: float | None = 60000.0,
    write_tstart_keyword: bool = False,
    pol_type: str = "AA+BB",
    dtype: str = "uint8",
    nsuboffs: int = 0,
    zero_off: float = 0.0,
    dat_scl: float = 1.0,
    dat_offs: float = 0.0,
    dat_wts: float = 1.0,
    dm: float | None = None,
    burst_time_s: float | None = None,
    burst_width_samples: int = 4,
    amplitude: float = 90.0,
    background: float = 80.0,
    noise_sigma: float = 4.0,
    channel_marker: bool = False,
    seed: int = 0,
    telescope: str = "FAKE",
    src_name: str = "SYNTH",
) -> dict:
    """Write a PSRFITS SEARCH-mode file and return the ground truth used to build it.

    Parameters
    ----------
    nsubint, nsblk
        Rows in the SUBINT table and samples per row. The file holds
        ``nsubint * nsblk`` samples.
    nchan, npol, tsamp
        Channels, polarisations and the sampling interval written to ``TBIN``.
    fch1, foff
        First stored channel and the step between channels. ``foff < 0`` gives a
        DESCENDING ``DAT_FREQ`` (the usual PSRFITS convention, the orientation
        that needs the reader to reverse the channel axis); ``foff > 0`` gives an
        ASCENDING one. Both must come out of the reader matching the ascending
        ``config.FREQ``.
    tstart
        Start MJD, written as ``STT_IMJD``/``STT_SMJD``/``STT_OFFS``. ``None``
        omits all three, which is what a file with no epoch looks like to
        ``get_obparams`` (audit P1-09). Note ``your`` cannot open such a file --
        ``SpectraInfo`` indexes ``STT_IMJD`` without a default.
    write_tstart_keyword
        Also write a non-standard ``TSTART`` card in the PRIMARY header. Only the
        ``except Exception`` copy of the astropy SUBINT reader looks at it; the
        primary copy uses the ``STT_*`` triple.
    pol_type
        ``POL_TYPE`` card. ``your`` needs one of its known orders when
        ``npol > 1`` ("AABB", "IQUV", "AABBCRCI").
    dtype
        ``"uint8"`` (8-bit DATA, what a real backend writes) or ``"float32"``
        (32-bit DATA, useful when a test needs exact per-sample values).
    nsuboffs
        Subint offset. Written to ``NSUBOFFS`` *and* folded into ``OFFS_SUB``,
        so the file is self-consistent: row ``i`` claims to start at sample
        ``(nsuboffs + i) * nsblk`` of the observation.
    zero_off, dat_scl, dat_offs, dat_wts
        Calibration written to the header/columns. The defaults make calibration
        the identity so raw and calibrated values coincide.
    dm, burst_time_s
        Inject a burst swept by ``dm`` arriving at ``burst_time_s`` at the top of
        the band, following the dispersion law the pipeline dedisperses with.
    channel_marker
        Replace the data with a per-channel constant equal to that channel's
        index in ASCENDING frequency order, so a block that has been read
        correctly reads ``block[t, 0, k] == k`` whichever way the file stores its
        channels. Suppresses noise and any burst.
    seed
        Seeds the noise.

    Returns
    -------
    dict
        Ground truth: dimensions, the stored ``freqs`` and the ascending
        ``freqs_ascending``, ``data`` as stored ``(nsamples, npol, nchan)`` and
        ``data_ascending`` -- the same samples with the channel axis in
        ascending-frequency order, which is the orientation every reader is
        supposed to produce.
    """
    if dtype not in _DATA_FORMAT:
        raise ValueError(f"dtype must be one of {sorted(_DATA_FORMAT)}, got {dtype!r}")
    if channel_marker and nchan * max(1, npol) > 256 and dtype == "uint8":
        raise ValueError("channel_marker with uint8 needs nchan * npol <= 256")

    nsamples = int(nsubint) * int(nsblk)
    freqs = channel_frequencies(fch1, foff, nchan)
    descending = bool(nchan > 1 and freqs[0] > freqs[-1])
    # Index of each stored channel once the axis is sorted ascending. For a
    # descending file this is nchan-1-j; for an ascending one it is j.
    ascending_index = np.argsort(np.argsort(freqs))

    rng = np.random.RandomState(seed)
    truth: dict = {
        "nsamples": nsamples, "nsubint": int(nsubint), "nsblk": int(nsblk),
        "nchan": int(nchan), "npol": int(npol), "tsamp": float(tsamp),
        "fch1": float(fch1), "foff": float(foff), "tstart": tstart,
        "freqs": freqs, "freqs_ascending": np.sort(freqs),
        "descending": descending, "dm": dm, "burst_time_s": burst_time_s,
        "pol_type": pol_type, "dtype": dtype, "nsuboffs": int(nsuboffs),
        "zero_off": float(zero_off), "tsubint": float(nsblk) * float(tsamp),
        "channel_marker": bool(channel_marker),
    }

    if channel_marker:
        data = np.zeros((nsamples, npol, nchan), dtype=np.float64)
        for p in range(npol):
            # Polarisations are offset so a test can tell them apart; pol 0 is
            # exactly the ascending channel index.
            data[:, p, :] = ascending_index + p * nchan
    else:
        data = rng.normal(background, noise_sigma, size=(nsamples, npol, nchan))
        if dm is not None and burst_time_s is not None:
            ref = float(np.max(freqs))
            delays = dispersion_delay_s(dm, freqs, ref)
            centres = (float(burst_time_s) + delays) / float(tsamp)
            half = max(1, int(burst_width_samples) // 2)
            for ch, centre in enumerate(centres):
                lo = int(round(centre)) - half
                hi = int(round(centre)) + half + 1
                lo_c, hi_c = max(0, lo), min(nsamples, hi)
                if hi_c > lo_c:
                    data[lo_c:hi_c, :, ch] += amplitude
            truth["ref_freq_mhz"] = ref
            truth["burst_sample_at_ref"] = int(round(float(burst_time_s) / float(tsamp)))

    if dtype == "uint8":
        stored = np.clip(np.round(data), 0, 255).astype(np.uint8)
        nbits = 8
    else:
        stored = data.astype(np.float32)
        nbits = 32

    truth["data"] = stored.astype(np.float32)
    truth["data_ascending"] = truth["data"][:, :, np.argsort(freqs)]

    # ---- PRIMARY header -------------------------------------------------
    bw = float(nchan) * float(foff)
    fcenter = float(fch1) + float(nchan) * float(foff) / 2.0
    p_hdr = fits.Header()
    p_hdr["HDRVER"] = "3.4"
    p_hdr["FITSTYPE"] = "PSRFITS"
    p_hdr["DATE"] = "2020-01-01T00:00:00"
    p_hdr["OBSERVER"] = "TEST"
    p_hdr["PROJID"] = "TEST"
    p_hdr["TELESCOP"] = telescope
    p_hdr["ANT_X"] = -1601185.63
    p_hdr["ANT_Y"] = -5041978.15
    p_hdr["ANT_Z"] = 3554876.43
    p_hdr["FRONTEND"] = "TEST"
    p_hdr["NRCVR"] = 1
    p_hdr["FD_POLN"] = "LIN"
    p_hdr["FD_HAND"] = -1
    p_hdr["FD_SANG"] = 45.0
    p_hdr["FD_XYPH"] = 0.0
    p_hdr["BACKEND"] = "TEST"
    p_hdr["BECONFIG"] = "N/A"
    p_hdr["BE_PHASE"] = -1
    p_hdr["BE_DCC"] = 0
    p_hdr["BE_DELAY"] = 0.0
    p_hdr["TCYCLE"] = 0.0
    p_hdr["OBS_MODE"] = "SEARCH"
    p_hdr["DATE-OBS"] = "2020-01-01T00:00:00"
    p_hdr["OBSFREQ"] = fcenter
    p_hdr["OBSBW"] = bw
    p_hdr["OBSNCHAN"] = int(nchan)
    p_hdr["CHAN_DM"] = 0.0
    p_hdr["SRC_NAME"] = src_name
    p_hdr["COORD_MD"] = "J2000"
    p_hdr["EQUINOX"] = 2000.0
    p_hdr["RA"] = "00:00:00.0000"
    p_hdr["DEC"] = "+00:00:00.000"
    p_hdr["BMAJ"] = 0.0
    p_hdr["BMIN"] = 0.0
    p_hdr["BPA"] = 0.0
    p_hdr["STT_CRD1"] = "00:00:00.0000"
    p_hdr["STT_CRD2"] = "+00:00:00.000"
    p_hdr["TRK_MODE"] = "TRACK"
    p_hdr["STP_CRD1"] = "00:00:00.0000"
    p_hdr["STP_CRD2"] = "+00:00:00.000"
    p_hdr["SCANLEN"] = nsamples * float(tsamp)
    p_hdr["FD_MODE"] = "FA"
    p_hdr["FA_REQ"] = 0.0
    p_hdr["CAL_MODE"] = "OFF"
    p_hdr["CAL_FREQ"] = 0.0
    p_hdr["CAL_DCYC"] = 0.0
    p_hdr["CAL_PHS"] = 0.0
    if tstart is not None:
        imjd, smjd, offs = _mjd_to_stt(float(tstart))
        p_hdr["STT_IMJD"] = imjd
        p_hdr["STT_SMJD"] = smjd
        p_hdr["STT_OFFS"] = offs
        truth["stt"] = (imjd, smjd, offs)
        truth["tstart_mjd"] = float(imjd) + (float(smjd) + float(offs)) / 86400.0
    else:
        truth["stt"] = None
        truth["tstart_mjd"] = None
    p_hdr["STT_LST"] = 0.0
    if write_tstart_keyword and tstart is not None:
        p_hdr["TSTART"] = float(tstart)
    truth["has_tstart_keyword"] = bool(write_tstart_keyword and tstart is not None)

    # ---- SUBINT header --------------------------------------------------
    tsub = float(nsblk) * float(tsamp)
    t_hdr = fits.Header()
    t_hdr["INT_TYPE"] = "TIME"
    t_hdr["INT_UNIT"] = "SEC"
    t_hdr["SCALE"] = "FluxDen"
    t_hdr["NPOL"] = int(npol)
    t_hdr["POL_TYPE"] = pol_type
    t_hdr["TBIN"] = float(tsamp)
    t_hdr["NBIN"] = 1
    t_hdr["NBIN_PRD"] = 0
    t_hdr["PHS_OFFS"] = 0.0
    t_hdr["NBITS"] = nbits
    t_hdr["ZERO_OFF"] = float(zero_off)
    t_hdr["NSUBOFFS"] = int(nsuboffs)
    t_hdr["NCHAN"] = int(nchan)
    t_hdr["CHAN_BW"] = float(foff)
    t_hdr["NCHNOFFS"] = 0
    t_hdr["NSBLK"] = int(nsblk)
    t_hdr["TSUBINT"] = tsub

    # ---- SUBINT columns -------------------------------------------------
    n = int(nsubint)
    offs_sub = (np.arange(n, dtype=np.float64) + 0.5 + int(nsuboffs)) * tsub
    zeros_f4 = np.zeros(n, dtype=np.float32)
    zeros_f8 = np.zeros(n, dtype=np.float64)
    dat_freq = np.vstack([freqs.astype(np.float32)] * n)
    wts = np.full((n, nchan), float(dat_wts), dtype=np.float32)
    scl = np.full((n, npol * nchan), float(dat_scl), dtype=np.float32)
    offsets = np.full((n, npol * nchan), float(dat_offs), dtype=np.float32)
    rows = stored.reshape(n, nsblk, npol, nchan)

    data_format = _DATA_FORMAT[dtype]
    columns = [
        fits.Column(name="TSUBINT", format="1D", unit="s",
                    array=np.full(n, tsub, dtype=np.float64)),
        fits.Column(name="OFFS_SUB", format="1D", unit="s", array=offs_sub),
        fits.Column(name="LST_SUB", format="1D", unit="s", array=zeros_f8),
        fits.Column(name="RA_SUB", format="1D", unit="deg", array=zeros_f8),
        fits.Column(name="DEC_SUB", format="1D", unit="deg", array=zeros_f8),
        fits.Column(name="GLON_SUB", format="1D", unit="deg", array=zeros_f8),
        fits.Column(name="GLAT_SUB", format="1D", unit="deg", array=zeros_f8),
        fits.Column(name="FD_ANG", format="1E", unit="deg", array=zeros_f4),
        fits.Column(name="POS_ANG", format="1E", unit="deg", array=zeros_f4),
        fits.Column(name="PAR_ANG", format="1E", unit="deg", array=zeros_f4),
        fits.Column(name="TEL_AZ", format="1E", unit="deg", array=zeros_f4),
        fits.Column(name="TEL_ZEN", format="1E", unit="deg", array=zeros_f4),
        fits.Column(name="DAT_FREQ", format=f"{nchan}E", unit="MHz", array=dat_freq),
        fits.Column(name="DAT_WTS", format=f"{nchan}E", array=wts),
        fits.Column(name="DAT_OFFS", format=f"{npol * nchan}E", array=offsets),
        fits.Column(name="DAT_SCL", format=f"{npol * nchan}E", array=scl),
        fits.Column(name="DATA",
                    format=f"{nsblk * npol * nchan}{data_format}",
                    dim=f"({nchan}, {npol}, {nsblk})",
                    array=rows),
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdul = fits.HDUList([
        fits.PrimaryHDU(header=p_hdr),
        fits.BinTableHDU(fits.FITS_rec.from_columns(columns), name="SUBINT", header=t_hdr),
    ])
    hdul.writeto(path, overwrite=True)
    hdul.close()

    truth["path"] = path
    truth["offs_sub"] = offs_sub
    return truth
