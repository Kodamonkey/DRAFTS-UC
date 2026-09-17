"""Characterization of the PSRFITS readers, as they behave TODAY.

``src/input/fits_handler.py::stream_fits`` holds four readers inside a single
``try``: the ``your`` library branch, a primary astropy SUBINT branch reached
only when ``your`` is absent, a non-SUBINT branch, and a *duplicated copy* of
the astropy SUBINT branch reached only through ``except Exception``. That
duplication has already produced live defects (audit P1-02). Audit REF-03 splits
the function; this module is the net under that split.

These tests record what the code does, not what it ought to do. Several of the
values pinned below are wrong -- the channel axis is fine now, but the chunk
geometry loses and duplicates samples, and the fallback throws the observation
epoch away. Each such assertion carries a comment saying so and naming the
finding. Do not "fix" them by asserting the right answer: the point is that the
refactor must not change behaviour silently, and a separate change fixes the
behaviour on purpose.

The four branches DISAGREE with each other on identical input. What is pinned:

  D1  chunk geometry: the ``your`` branch tiles the file exactly; both astropy
      copies leave a gap of ``overlap_samples`` after the first chunk and
      duplicate ``overlap_samples`` at the last one (``TestValidWindowsTile``).
  D2  epoch: the primary astropy copy reads STT_IMJD/STT_SMJD/STT_OFFS, the
      fallback copy reads a ``TSTART`` card that standard PSRFITS does not have,
      so the fallback reports MJD 0.0 and NULLS ``config.TSTART_MJD``
      (``TestFallbackAgainstPrimary``).
  D3  ``ZERO_OFF``: the astropy copies subtract it, the ``your`` branch does not
      (``TestBranchesDisagree``).
  D4  ``NSUBOFFS``: the primary astropy copy honours ``OFFS_SUB`` absolutely and
      zero-pads the front of a continuation file; the fallback and the ``your``
      branch ignore it (``TestBranchesDisagree``).
  D5  ``POL_TYPE``: ``stream_fits_multi_pol`` reports the literal ``"IQUV"`` for
      every file (``TestStreamFitsMultiPol``).
  D6  a file with no STT_* keywords cannot be opened by ``your`` at all, so it
      silently falls through to the duplicated copy (``TestBranchesDisagree``).
  D7  when the file length is an exact multiple of ``chunk_samples`` and there
      is no overlap, the primary astropy copy raises AFTER yielding every block,
      and the fallback then replays the whole file from sample 0 -- the consumer
      receives the entire observation twice
      (``TestPrimaryCrashesAndTheFallbackReplaysTheFile``).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.config import config
from src.input import fits_handler
from tests.synthetic_psrfits import dispersion_delay_s, write_psrfits

# Keys ``get_obparams`` creates on the config module that do not exist before it
# runs; the shared autouse fixture in conftest restores them.
METADATA_GEOMETRY_KEYS = (
    "chunk_idx", "start_sample", "end_sample", "actual_chunk_size",
    "block_start_sample", "block_end_sample", "overlap_left", "overlap_right",
)

# Every file below is 16 subints of 32 samples = 512 samples, 16 channels,
# 1 ms per sample. Small enough to run on every commit, large enough that
# chunk_samples=128 needs four chunks.
NSUBINT = 16
NSBLK = 32
NCHAN = 16
TSAMP = 1.0e-3
NSAMPLES = NSUBINT * NSBLK
TSTART = 60000.25          # 0.25 d = 21600 s exactly, so STT_* round-trips exactly


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _geometry(blocks) -> list[tuple]:
    """``(block_len, *METADATA_GEOMETRY_KEYS)`` for each yielded block."""
    return [
        (block.shape[0],) + tuple(meta[k] for k in METADATA_GEOMETRY_KEYS)
        for block, meta in blocks
    ]


def _valid_window(block: np.ndarray, meta: dict) -> np.ndarray:
    """The part of *block* the metadata declares valid, as ``(nsamp, nchan)``."""
    lo = meta["start_sample"] - meta["block_start_sample"]
    return block[lo:lo + meta["actual_chunk_size"], 0, :]


def _use_astropy_primary(monkeypatch) -> None:
    """Take the primary astropy SUBINT branch: it runs only without ``your``."""
    monkeypatch.setattr(fits_handler, "your_psrfits", None)


def _use_astropy_fallback(monkeypatch) -> None:
    """Take the duplicated copy behind ``except Exception``.

    There is no flag for it: the copy runs only when something in the primary
    path raises. ``log_stream_fits_parameters`` is called once by each copy, so
    a one-shot raiser fails the primary and lets the fallback complete. It is
    called before any data is touched, so the fallback starts from scratch.
    """
    monkeypatch.setattr(fits_handler, "your_psrfits", None)
    original = fits_handler.log_stream_fits_parameters
    calls = {"n": 0}

    def _raise_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("forced: exercise the duplicated astropy reader")
        return original(*args, **kwargs)

    monkeypatch.setattr(fits_handler, "log_stream_fits_parameters", _raise_once)


def _write(tmp_path, name: str, **kwargs) -> dict:
    """Write a file with the module-wide defaults, then load its header."""
    params = dict(
        nsubint=NSUBINT, nsblk=NSBLK, nchan=NCHAN, tsamp=TSAMP,
        tstart=TSTART, dtype="float32", seed=3,
    )
    params.update(kwargs)
    truth = write_psrfits(tmp_path / name, **params)
    fits_handler.get_obparams(str(truth["path"]))
    return truth


# ---------------------------------------------------------------------------
# the writer itself
# ---------------------------------------------------------------------------
class TestSyntheticWriterIsAcceptedByBothBackends:
    """``stream_fits`` branches on which library is available; both must work."""

    def test_astropy_reads_the_file(self, tmp_path):
        from astropy.io import fits

        truth = write_psrfits(tmp_path / "a.fits", nsubint=2, nsblk=8, nchan=4)
        with fits.open(truth["path"]) as hdul:
            assert [hdu.name for hdu in hdul] == ["PRIMARY", "SUBINT"]
            subint = hdul["SUBINT"]
            assert subint.header["TBIN"] == pytest.approx(truth["tsamp"])
            assert subint.header["NCHAN"] == 4
            assert subint.header["NSBLK"] == 8
            assert subint.header["NAXIS2"] == 2
            # (nsubint, nsblk, npol, nchan)
            assert subint.data["DATA"].shape == (2, 8, 1, 4)
            assert np.allclose(subint.data["DAT_FREQ"][0], truth["freqs"])

    def test_your_reads_the_file(self, tmp_path):
        your = pytest.importorskip("your")

        truth = write_psrfits(tmp_path / "y.fits", nsubint=2, nsblk=8, nchan=4,
                              fch1=1500.0, foff=-1.0)
        handle = your.Your(str(truth["path"]))
        assert handle.your_header.nspectra == 16
        assert handle.your_header.nchans == 4
        assert handle.your_header.foff == pytest.approx(-1.0)
        assert handle.your_header.fch1 == pytest.approx(1500.0)

    def test_channel_marker_labels_channels_by_ascending_index(self, tmp_path):
        """The marker is defined on frequency, not on storage position."""
        desc = write_psrfits(tmp_path / "d.fits", nsubint=1, nsblk=2, nchan=4,
                             fch1=1500.0, foff=-1.0, channel_marker=True)
        asc = write_psrfits(tmp_path / "a.fits", nsubint=1, nsblk=2, nchan=4,
                            fch1=1497.0, foff=+1.0, channel_marker=True)
        # Stored descending: the first stored channel is the TOP of the band,
        # which is ascending index 3.
        assert list(desc["data"][0, 0, :]) == [3.0, 2.0, 1.0, 0.0]
        assert list(asc["data"][0, 0, :]) == [0.0, 1.0, 2.0, 3.0]
        # Both describe the same sky: identical once sorted by frequency.
        assert np.array_equal(desc["data_ascending"], asc["data_ascending"])


# ---------------------------------------------------------------------------
# get_obparams
# ---------------------------------------------------------------------------
class TestGetObparams:
    def test_descending_axis_file(self, tmp_path):
        truth = _write(tmp_path, "desc.fits", fch1=1500.0, foff=-1.0)

        assert config.FREQ_RESO == NCHAN
        assert config.TIME_RESO == pytest.approx(TSAMP)
        assert config.FILE_LENG == NSAMPLES
        # DAT_FREQ runs 1500 -> 1485; config.FREQ is always ascending and
        # DATA_NEEDS_REVERSAL records that the data must be flipped to match.
        assert config.DATA_NEEDS_REVERSAL is True
        assert config.FREQ[0] == pytest.approx(1485.0)
        assert config.FREQ[-1] == pytest.approx(1500.0)
        assert np.allclose(config.FREQ, truth["freqs_ascending"])
        assert config.TSTART_MJD == pytest.approx(60000.25)
        assert config.TSTART_MJD_CORR == pytest.approx(60000.25)

    def test_ascending_axis_file_gives_the_same_frequency_axis(self, tmp_path):
        truth = _write(tmp_path, "asc.fits", fch1=1485.0, foff=+1.0)

        assert config.DATA_NEEDS_REVERSAL is False
        assert config.FREQ[0] == pytest.approx(1485.0)
        assert config.FREQ[-1] == pytest.approx(1500.0)
        assert np.allclose(config.FREQ, truth["freqs_ascending"])
        assert config.FREQ_RESO == NCHAN
        assert config.FILE_LENG == NSAMPLES

    def test_file_without_an_epoch_does_not_inherit_the_previous_one(self, tmp_path):
        """Audit P1-09: the epoch must not survive from the file read before."""
        _write(tmp_path, "with_epoch.fits", tstart=59500.5)
        assert config.TSTART_MJD == pytest.approx(59500.5)

        _write(tmp_path, "no_epoch.fits", tstart=None)

        # PINNED AS-IS: the epoch does NOT leak -- that is the P1-09 fix and it
        # holds. But the value recorded for a file with no STT_* keywords is the
        # float 0.0, not None: `primary.get("STT_IMJD", 0)` defaults to 0 and the
        # `tstart_mjd is not None` guard below it therefore never fires. MJD 0.0
        # is 17 Nov 1858, which downstream code will happily format as a date.
        assert config.TSTART_MJD == 0.0
        assert config.TSTART_MJD_CORR == 0.0
        assert config.TSTART_MJD is not None

    def test_subint_offset_shifts_the_corrected_epoch(self, tmp_path):
        """A continuation file: NSUBOFFS rows of the observation precede it."""
        truth = _write(tmp_path, "cont.fits", nsuboffs=3)

        # Independent of the reader: three subints of 32 samples at 1 ms each
        # is 0.096 s, and an MJD is a day, so 0.096/86400 of a day.
        expected_corr = 60000.25 + 0.096 / 86400.0
        assert config.TSTART_MJD == pytest.approx(60000.25)
        assert config.TSTART_MJD_CORR == pytest.approx(expected_corr, abs=1e-12)
        assert config.NSUBOFFS == 3
        # FILE_LENG counts the rows present, not the offset ones.
        assert config.FILE_LENG == NSAMPLES == truth["nsamples"]

    def test_nsuboffs_header_card_is_overridden_by_offs_sub(self, tmp_path):
        """PINNED AS-IS. ``OFFS_SUB`` wins over the ``NSUBOFFS`` card.

        The writer keeps the two consistent; this file deliberately does not.
        ``get_obparams`` recomputes the offset from the first ``OFFS_SUB`` and
        silently discards the header value.
        """
        from astropy.io import fits

        truth = _write(tmp_path, "liar.fits")
        with fits.open(truth["path"], mode="update") as hdul:
            hdul["SUBINT"].header["NSUBOFFS"] = 7
        fits_handler.get_obparams(str(truth["path"]))

        assert config.NSUBOFFS == 0
        assert config.TSTART_MJD_CORR == pytest.approx(60000.25)

    def test_header_scalars_reach_config(self, tmp_path):
        _write(tmp_path, "pol.fits", npol=4, pol_type="IQUV")
        assert config.NPOL == 4
        assert config.POL_TYPE == "IQUV"
        assert config.NBITS == 32          # dtype="float32" writes NBITS=32
        assert config.TSUBINT == pytest.approx(NSBLK * TSAMP)


# ---------------------------------------------------------------------------
# the `your` branch -- the one that runs when the library is installed
# ---------------------------------------------------------------------------
class TestYourBranchBlockSequence:
    """Exact sequence of ``(block.shape, metadata)`` from the ``your`` reader."""

    def setup_method(self):
        pytest.importorskip("your")

    @pytest.mark.parametrize(
        ("chunk", "overlap", "expected"),
        [
            # (block_len, chunk_idx, start, end, actual, block_start, block_end,
            #  overlap_left, overlap_right)
            (128, 16, [
                (144, 0, 0, 128, 128, 0, 144, 0, 16),
                (160, 1, 128, 256, 128, 112, 272, 16, 16),
                (160, 2, 256, 384, 128, 240, 400, 16, 16),
                (144, 3, 384, 512, 128, 368, 512, 16, 0),
            ]),
            # chunk_samples is not a multiple of nsblk, and no overlap
            (200, 0, [
                (200, 0, 0, 200, 200, 0, 200, 0, 0),
                (200, 1, 200, 400, 200, 200, 400, 0, 0),
                (112, 2, 400, 512, 112, 400, 512, 0, 0),
            ]),
            # the whole file fits in one chunk
            (1000, 32, [
                (512, 0, 0, 512, 512, 0, 512, 0, 0),
            ]),
        ],
    )
    def test_geometry(self, tmp_path, chunk, overlap, expected):
        truth = _write(tmp_path, "s.fits")
        blocks = list(fits_handler.stream_fits(
            str(truth["path"]), chunk_samples=chunk, overlap_samples=overlap))
        assert _geometry(blocks) == expected

    def test_scalar_metadata(self, tmp_path):
        truth = _write(tmp_path, "s.fits")
        blocks = list(fits_handler.stream_fits(
            str(truth["path"]), chunk_samples=128, overlap_samples=16))

        for block, meta in blocks:
            assert meta["total_samples"] == NSAMPLES
            assert meta["nchans"] == NCHAN
            assert meta["nifs"] == 1
            assert meta["file_type"] == "fits"
            assert meta["tbin_sec"] == pytest.approx(TSAMP)
            assert meta["dtype"] == "float32"
            assert meta["shape"] == block.shape
            assert block.ndim == 3 and block.shape[1] == 1
            # PINNED AS-IS: the `your` branch publishes no epoch at all. The two
            # astropy copies put tstart_mjd / tstart_mjd_corr / tsubint_sec in
            # every metadata dict; this one does not.
            assert "tstart_mjd" not in meta
            assert "tstart_mjd_corr" not in meta
            assert "tsubint_sec" not in meta

        starts = [m["t_rel_start_sec"] for _, m in blocks]
        ends = [m["t_rel_end_sec"] for _, m in blocks]
        assert starts == pytest.approx([0.0, 0.128, 0.256, 0.384])
        assert ends == pytest.approx([0.128, 0.256, 0.384, 0.512])

    def test_reader_does_not_touch_the_epoch_config(self, tmp_path):
        """PINNED AS-IS: unlike both astropy copies, this branch leaves
        ``config.TSTART_MJD`` exactly as ``get_obparams`` left it."""
        truth = _write(tmp_path, "s.fits")
        config.TSTART_MJD = 12345.0
        config.TSTART_MJD_CORR = 12345.0

        list(fits_handler.stream_fits(str(truth["path"]), chunk_samples=256))

        assert config.TSTART_MJD == 12345.0
        assert config.TSTART_MJD_CORR == 12345.0


# ---------------------------------------------------------------------------
# the primary astropy branch -- the one that runs when `your` is absent
# ---------------------------------------------------------------------------
class TestAstropyPrimaryBlockSequence:
    @pytest.mark.parametrize(
        ("chunk", "overlap", "expected"),
        [
            # NOTE how this differs from the `your` table above for the SAME
            # file and the SAME arguments: chunk 1 starts at 144, not 128.
            (128, 16, [
                (160, 0, 0, 128, 128, 0, 160, 0, 32),
                (160, 1, 144, 272, 128, 128, 288, 16, 16),
                (160, 2, 272, 400, 128, 256, 416, 16, 16),
                (128, 3, 384, 512, 128, 384, 512, 0, 0),
            ]),
            (200, 0, [
                (200, 0, 0, 200, 200, 0, 200, 0, 0),
                (200, 1, 200, 400, 200, 200, 400, 0, 0),
                (112, 2, 400, 512, 112, 400, 512, 0, 0),
            ]),
            (1000, 32, [
                (512, 0, 0, 512, 512, 0, 512, 0, 0),
            ]),
        ],
    )
    def test_geometry(self, tmp_path, monkeypatch, chunk, overlap, expected):
        truth = _write(tmp_path, "s.fits")
        _use_astropy_primary(monkeypatch)
        blocks = list(fits_handler.stream_fits(
            str(truth["path"]), chunk_samples=chunk, overlap_samples=overlap))
        assert _geometry(blocks) == expected

    def test_scalar_metadata(self, tmp_path, monkeypatch):
        truth = _write(tmp_path, "s.fits")
        _use_astropy_primary(monkeypatch)
        blocks = list(fits_handler.stream_fits(
            str(truth["path"]), chunk_samples=128, overlap_samples=16))

        for block, meta in blocks:
            assert meta["total_samples"] == NSAMPLES
            assert meta["nchans"] == NCHAN
            assert meta["nifs"] == 1
            assert meta["file_type"] == "fits"
            assert meta["tbin_sec"] == pytest.approx(TSAMP)
            assert meta["dtype"] == "float32"
            assert meta["shape"] == block.shape
            # Only the astropy copies carry these three.
            assert meta["tstart_mjd"] == pytest.approx(60000.25)
            assert meta["tstart_mjd_corr"] == pytest.approx(60000.25)
            assert meta["tsubint_sec"] == pytest.approx(NSBLK * TSAMP)

        starts = [m["t_rel_start_sec"] for _, m in blocks]
        # PINNED AS-IS: 0.144, not 0.128 -- see TestValidWindowsTile.
        assert starts == pytest.approx([0.0, 0.144, 0.272, 0.384])

    def test_each_block_really_contains_the_samples_it_claims(self, tmp_path, monkeypatch):
        """Whatever the geometry says, the bytes under it must be right."""
        truth = _write(tmp_path, "s.fits")
        _use_astropy_primary(monkeypatch)
        full = truth["data_ascending"][:, 0, :]

        for block, meta in fits_handler.stream_fits(
                str(truth["path"]), chunk_samples=128, overlap_samples=16):
            window = _valid_window(block, meta)
            assert np.array_equal(
                window, full[meta["start_sample"]:meta["end_sample"]])


# ---------------------------------------------------------------------------
# D1 -- do the valid windows cover the file exactly once?
# ---------------------------------------------------------------------------
class TestValidWindowsTile:
    """The property every chunk-geometry defect (P1-04/05/06) broke."""

    def test_your_branch_tiles_the_file_exactly(self, tmp_path):
        pytest.importorskip("your")
        truth = _write(tmp_path, "s.fits")
        full = truth["data_ascending"][:, 0, :]

        windows, spans = [], []
        for block, meta in fits_handler.stream_fits(
                str(truth["path"]), chunk_samples=128, overlap_samples=16):
            windows.append(_valid_window(block, meta))
            spans.append((meta["start_sample"], meta["end_sample"]))

        assert spans == [(0, 128), (128, 256), (256, 384), (384, 512)]
        rebuilt = np.concatenate(windows, axis=0)
        assert rebuilt.shape == full.shape
        assert np.array_equal(rebuilt, full)

    @pytest.mark.parametrize("take_branch", [_use_astropy_primary, _use_astropy_fallback])
    def test_astropy_copies_lose_and_duplicate_samples(self, tmp_path, monkeypatch, take_branch):
        """PINNED AS-IS. This is a live defect, of the P1-04/05/06 family.

        With ``overlap_samples=16`` the valid windows come out

            [0, 128)  [144, 272)  [272, 400)  [384, 512)

        so samples 128..143 are in NO valid window -- nothing ever searches
        them -- and samples 384..399 are in two. The cause is the clamp that
        pulls the first chunk's ``valid_start`` back to 0 at the start of the
        file without also pulling back how much the buffer then advances.

        Both astropy copies do it identically, so the refactor may merge them;
        it must not change the numbers while doing so.
        """
        truth = _write(tmp_path, "s.fits")
        take_branch(monkeypatch)

        spans = [(m["start_sample"], m["end_sample"])
                 for _, m in fits_handler.stream_fits(
                     str(truth["path"]), chunk_samples=128, overlap_samples=16)]

        assert spans == [(0, 128), (144, 272), (272, 400), (384, 512)]

        covered = np.zeros(NSAMPLES, dtype=int)
        for start, end in spans:
            covered[start:end] += 1
        assert list(np.flatnonzero(covered == 0)) == list(range(128, 144))
        assert list(np.flatnonzero(covered > 1)) == list(range(384, 400))

    @pytest.mark.parametrize("take_branch", [_use_astropy_primary, _use_astropy_fallback])
    def test_astropy_copies_tile_exactly_without_overlap(self, tmp_path, monkeypatch, take_branch):
        """Without overlap the same code does reconstruct the file exactly."""
        truth = _write(tmp_path, "s.fits")
        take_branch(monkeypatch)
        full = truth["data_ascending"][:, 0, :]

        windows, spans = [], []
        for block, meta in fits_handler.stream_fits(
                str(truth["path"]), chunk_samples=200, overlap_samples=0):
            windows.append(_valid_window(block, meta))
            spans.append((meta["start_sample"], meta["end_sample"]))

        assert spans == [(0, 200), (200, 400), (400, 512)]
        assert np.array_equal(np.concatenate(windows, axis=0), full)


# ---------------------------------------------------------------------------
# channel ordering
# ---------------------------------------------------------------------------
class TestChannelOrdering:
    """A descending file and an ascending file must come out identical.

    Each channel carries a constant equal to its index in ASCENDING frequency
    order, so a correctly oriented block reads ``block[t, 0, k] == k``
    regardless of how the file stores its channels.
    """

    @staticmethod
    def _read_one(path, take_branch, monkeypatch):
        if take_branch is not None:
            take_branch(monkeypatch)
        # 200 is deliberately NOT a divisor of 512: see
        # TestPrimaryCrashesAndTheFallbackReplaysTheFile.
        blocks = list(fits_handler.stream_fits(
            str(path), chunk_samples=200, overlap_samples=0))
        return np.concatenate([_valid_window(b, m) for b, m in blocks], axis=0)

    @pytest.mark.parametrize(
        "take_branch", [None, _use_astropy_primary, _use_astropy_fallback])
    def test_both_orientations_match_ascending_config_freq(self, tmp_path, monkeypatch, take_branch):
        if take_branch is None:
            pytest.importorskip("your")

        desc = _write(tmp_path, "desc.fits", fch1=1500.0, foff=-1.0,
                      channel_marker=True, dtype="uint8")
        assert config.DATA_NEEDS_REVERSAL is True
        desc_data = self._read_one(desc["path"], take_branch, monkeypatch)

        asc = _write(tmp_path, "asc.fits", fch1=1485.0, foff=+1.0,
                     channel_marker=True, dtype="uint8")
        assert config.DATA_NEEDS_REVERSAL is False
        asc_data = self._read_one(asc["path"], take_branch, monkeypatch)

        expected_row = np.arange(NCHAN, dtype=np.float32)
        assert np.array_equal(desc_data[0], expected_row)
        assert np.array_equal(asc_data[0], expected_row)
        assert np.array_equal(desc_data, asc_data)
        assert np.array_equal(desc_data, desc["data_ascending"][:, 0, :])
        # config.FREQ ascends, and channel k of the block is config.FREQ[k].
        assert config.FREQ[0] < config.FREQ[-1]
        assert len(config.FREQ) == desc_data.shape[1]

    def test_a_dispersed_burst_sweeps_the_right_way(self, tmp_path):
        """Orientation checked on real dispersed data, not just a marker.

        The burst was written with the pipeline's own K_DM_MS. In the block the
        reader returns, channel index rises with frequency, so the arrival
        sample must fall as the index rises.
        """
        pytest.importorskip("your")
        truth = _write(tmp_path, "burst.fits", fch1=1500.0, foff=-1.0,
                       dtype="uint8", dm=200.0, burst_time_s=0.05,
                       amplitude=100.0, background=40.0, noise_sigma=0.0)

        blocks = list(fits_handler.stream_fits(
            str(truth["path"]), chunk_samples=200, overlap_samples=0))
        data = np.concatenate([_valid_window(b, m) for b, m in blocks], axis=0)

        arrival = np.argmax(data, axis=0)
        assert np.all(np.diff(arrival) <= 0), "burst must arrive earlier higher up"

        freqs_ascending = truth["freqs_ascending"]
        delays = dispersion_delay_s(200.0, freqs_ascending, float(freqs_ascending.max()))
        centre = np.round((0.05 + delays) / TSAMP).astype(int)
        # The injected pulse is a flat plateau spanning centre-2 .. centre+2
        # (burst_width_samples=4), and argmax lands on its first sample.
        assert np.array_equal(arrival, centre - 2)

    def test_the_freq_order_warning_is_silent_on_a_normal_file(self, tmp_path, caplog):
        """FIXED. It used to fire once per chunk on every ordinary PSRFITS.

        The first version compared ``your``'s ``foff > 0`` rule against
        ``DATA_NEEDS_REVERSAL``, and those are opposite by construction -- a
        descending file has ``foff < 0`` AND needs reversing -- so the two never
        agreed and the "mismatch" was unconditional. It warned on every normal
        file and stayed silent on the one case worth reporting.

        It now compares the sign of ``foff`` against the ordering derived from
        ``DAT_FREQ``, which agree in any well-formed header, so it only speaks
        when the file contradicts itself.
        """
        pytest.importorskip("your")
        truth = _write(tmp_path, "desc.fits", fch1=1500.0, foff=-1.0)
        assert config.DATA_NEEDS_REVERSAL is True

        with caplog.at_level("WARNING", logger="src.input.fits_handler"):
            list(fits_handler.stream_fits(str(truth["path"]),
                                          chunk_samples=200, overlap_samples=0))

        noise = [r for r in caplog.records
                 if "FREQ-ORDER" in r.message]
        assert noise == [], "the detector must be silent on an ordinary file"


# ---------------------------------------------------------------------------
# D2 -- the duplicated copy behind `except Exception`
# ---------------------------------------------------------------------------
class TestFallbackAgainstPrimary:
    """The fallback is a copy of the primary. Where do the copies differ?"""

    @staticmethod
    def _run(path, take_branch, monkeypatch, **kwargs):
        take_branch(monkeypatch)
        out = [(block.copy(), dict(meta))
               for block, meta in fits_handler.stream_fits(str(path), **kwargs)]
        epoch = (getattr(config, "TSTART_MJD", "<unset>"),
                 getattr(config, "TSTART_MJD_CORR", "<unset>"))
        return out, epoch

    def test_the_fallback_really_ran(self, tmp_path, monkeypatch, caplog):
        truth = _write(tmp_path, "s.fits")
        _use_astropy_fallback(monkeypatch)
        with caplog.at_level("WARNING", logger="src.input.fits_handler"):
            list(fits_handler.stream_fits(str(truth["path"]), chunk_samples=200))
        assert any("falling back to astropy" in r.message for r in caplog.records)

    @pytest.mark.parametrize(("chunk", "overlap"), [(128, 16), (200, 0), (1000, 32)])
    def test_blocks_and_geometry_are_identical(self, tmp_path, monkeypatch, chunk, overlap):
        truth = _write(tmp_path, "s.fits")
        kw = dict(chunk_samples=chunk, overlap_samples=overlap)

        with pytest.MonkeyPatch.context() as mp:
            primary, _ = self._run(truth["path"], _use_astropy_primary, mp, **kw)
        with pytest.MonkeyPatch.context() as mp:
            fallback, _ = self._run(truth["path"], _use_astropy_fallback, mp, **kw)

        assert _geometry(fallback) == _geometry(primary)
        for (pb, _), (fb, _) in zip(primary, fallback):
            assert np.array_equal(fb, pb)

    def test_the_fallback_throws_the_epoch_away(self, tmp_path, monkeypatch):
        """PINNED AS-IS, and it is wrong.

        The primary copy builds the epoch from STT_IMJD/STT_SMJD/STT_OFFS, which
        is how PSRFITS stores it. The duplicated copy instead reads a ``TSTART``
        card from the PRIMARY header -- not a PSRFITS keyword, and absent from a
        standard file -- plus ``NSUBOFFS`` from the PRIMARY header rather than
        from SUBINT. On a perfectly ordinary file it therefore reports MJD 0.0
        and NULLS ``config.TSTART_MJD``, which every candidate's absolute date
        is computed from. Same file, same bytes, two different answers.
        """
        truth = _write(tmp_path, "s.fits")
        kw = dict(chunk_samples=200, overlap_samples=0)

        with pytest.MonkeyPatch.context() as mp:
            primary, primary_epoch = self._run(truth["path"], _use_astropy_primary, mp, **kw)
        with pytest.MonkeyPatch.context() as mp:
            fallback, fallback_epoch = self._run(truth["path"], _use_astropy_fallback, mp, **kw)

        assert primary_epoch == (pytest.approx(60000.25), pytest.approx(60000.25))
        assert fallback_epoch == (None, None)

        assert [m["tstart_mjd"] for _, m in primary] == [pytest.approx(60000.25)] * 3
        assert [m["tstart_mjd"] for _, m in fallback] == [0.0, 0.0, 0.0]
        assert [m["tstart_mjd_corr"] for _, m in fallback] == [0.0, 0.0, 0.0]

        # Every other metadata key agrees.
        for (_, pm), (_, fm) in zip(primary, fallback):
            differing = {k for k in pm if pm[k] != fm[k]}
            assert differing == {"tstart_mjd", "tstart_mjd_corr"}

    def test_the_fallback_recovers_the_epoch_from_a_nonstandard_tstart_card(self, tmp_path, monkeypatch):
        """Confirms the mechanism: write the card the fallback looks for."""
        truth = _write(tmp_path, "t.fits", write_tstart_keyword=True)
        assert truth["has_tstart_keyword"] is True

        _, epoch = self._run(truth["path"], _use_astropy_fallback, monkeypatch,
                             chunk_samples=200, overlap_samples=0)
        assert epoch == (pytest.approx(60000.25), pytest.approx(60000.25))


# ---------------------------------------------------------------------------
# D7 -- the primary copy raises after yielding, and the fallback replays
# ---------------------------------------------------------------------------
class TestTheFileIsYieldedExactlyOnce:
    """FIXED. This used to hand the caller every sample twice.

    In the primary astropy copy, the emission loop sets ``out_buf = None`` when
    a chunk consumes the whole buffer. The end-of-file tail handler then does
    ``if buffer_blocks: out_buf = _concatenate_buffer()`` and immediately
    ``if out_buf.shape[0] > 0``. When the last chunk emptied the buffer exactly
    -- which is precisely what happens when the file length is a multiple of
    ``chunk_samples`` and ``overlap_samples`` is 0 -- ``buffer_blocks`` is empty,
    ``out_buf`` is still None, and the reader raises ``AttributeError``.

    ``stream_fits`` is a generator, and its ``except Exception`` sits OUTSIDE
    the loop that has already yielded. So the exception does not reach the
    caller: the duplicated astropy copy starts over at sample 0 and yields the
    entire file a second time. The caller sees one generator that hands it every
    sample twice, with no error and only a generic "falling back to astropy"
    warning in the log.

    The tail handler now reassigns ``out_buf`` unconditionally and checks it for
    None before dereferencing it, in BOTH copies -- the duplicate turned out to
    carry the same code, contrary to the first reading of it.
    """

    @pytest.mark.parametrize(
        ("chunk", "expected_spans"),
        [
            (256, [(0, 256), (256, 512)]),
            (512, [(0, 512)]),
        ],
    )
    def test_the_exactly_divisible_case_yields_each_sample_once(
        self, tmp_path, monkeypatch, chunk, expected_spans
    ):
        """Both parameters are file lengths divisible by the chunk size.

        That is the condition that emptied the buffer exactly and left
        ``out_buf`` None, so these are the two cases that used to come back
        doubled: 256 gave [(0,256),(256,512),(0,256),(256,512)] and 512 gave the
        whole file twice.
        """
        truth = _write(tmp_path, "s.fits")
        _use_astropy_primary(monkeypatch)

        blocks = list(fits_handler.stream_fits(
            str(truth["path"]), chunk_samples=chunk, overlap_samples=0))
        spans = [(m["start_sample"], m["end_sample"]) for _, m in blocks]

        assert spans == expected_spans
        assert sum(e - s for s, e in spans) == NSAMPLES

        # The primary copy ran to completion, so the epoch is the real one --
        # under the old behaviour the second pass came from the duplicate and
        # reported 0.0 (see D2).
        for _, meta in blocks:
            assert meta["tstart_mjd"] == pytest.approx(60000.25)

    def test_a_leftover_partial_chunk_avoids_it(self, tmp_path, monkeypatch):
        """512 is not a multiple of 200, so the tail handler has a buffer."""
        truth = _write(tmp_path, "s.fits")
        _use_astropy_primary(monkeypatch)

        spans = [(m["start_sample"], m["end_sample"])
                 for _, m in fits_handler.stream_fits(
                     str(truth["path"]), chunk_samples=200, overlap_samples=0)]
        assert spans == [(0, 200), (200, 400), (400, 512)]

    def test_overlap_also_avoids_it(self, tmp_path, monkeypatch):
        """With overlap the buffer keeps a tail, so out_buf is never None."""
        truth = _write(tmp_path, "s.fits")
        _use_astropy_primary(monkeypatch)

        spans = [(m["start_sample"], m["end_sample"])
                 for _, m in fits_handler.stream_fits(
                     str(truth["path"]), chunk_samples=256, overlap_samples=16)]
        assert spans == [(0, 256), (256, 512)]

    def test_the_your_branch_is_unaffected(self, tmp_path):
        pytest.importorskip("your")
        truth = _write(tmp_path, "s.fits")

        spans = [(m["start_sample"], m["end_sample"])
                 for _, m in fits_handler.stream_fits(
                     str(truth["path"]), chunk_samples=256, overlap_samples=0)]
        assert spans == [(0, 256), (256, 512)]


# ---------------------------------------------------------------------------
# D3 / D4 / D6 -- inputs on which the branches answer differently
# ---------------------------------------------------------------------------
class TestBranchesDisagree:
    def test_zero_off_is_subtracted_by_astropy_and_ignored_by_your(self, tmp_path, monkeypatch):
        """PINNED AS-IS. ``ZERO_OFF`` is a real PSRFITS calibration card.

        ``_apply_calibration`` subtracts it; ``your.formats.psrfits`` never looks
        at it, and the ``your`` branch applies no calibration of its own. The
        same file therefore yields values differing by exactly ZERO_OFF
        depending on which library happens to be installed.
        """
        pytest.importorskip("your")
        truth = _write(tmp_path, "z.fits", nsubint=2, nsblk=NSBLK, nchan=8,
                       channel_marker=True, dtype="uint8", zero_off=10.0)
        kw = dict(chunk_samples=100, overlap_samples=0)   # 64 samples: one chunk

        your_block, _ = next(iter(fits_handler.stream_fits(str(truth["path"]), **kw)))
        _use_astropy_primary(monkeypatch)
        astropy_block, _ = next(iter(fits_handler.stream_fits(str(truth["path"]), **kw)))

        marker = np.arange(8, dtype=np.float32)
        assert np.array_equal(your_block[0, 0, :], marker)
        assert np.array_equal(astropy_block[0, 0, :], marker - 10.0)

    def test_nsuboffs_makes_the_three_branches_produce_three_geometries(self, tmp_path):
        """PINNED AS-IS, and the primary astropy copy is badly wrong.

        ``NSUBOFFS=3`` marks an ordinary continuation file: 3 subints of the
        observation precede it, so ``OFFS_SUB`` starts at 3.5 subints. Only the
        primary astropy copy reads ``OFFS_SUB`` as an ABSOLUTE position, so it
        zero-pads 96 samples onto the front of a 512-sample file and then emits
        608 samples worth of chunks, while still reporting
        ``total_samples == 512``. Its valid windows run past the end of the file
        and overlap each other. The ``your`` branch and the fallback copy both
        ignore ``NSUBOFFS`` and emit the file as written.
        """
        pytest.importorskip("your")
        truth = _write(tmp_path, "cont.fits", nsuboffs=3)
        kw = dict(chunk_samples=200, overlap_samples=0)

        def spans():
            return [(m["start_sample"], m["end_sample"], m["total_samples"])
                    for _, m in fits_handler.stream_fits(str(truth["path"]), **kw)]

        your_spans = spans()
        with pytest.MonkeyPatch.context() as mp:
            _use_astropy_primary(mp)
            primary_spans = spans()
        with pytest.MonkeyPatch.context() as mp:
            _use_astropy_fallback(mp)
            fallback_spans = spans()

        assert your_spans == [(0, 200, 512), (200, 400, 512), (400, 512, 512)]
        assert fallback_spans == your_spans
        # 96 zero samples prepended, so 608 samples are emitted for a file whose
        # own metadata says it has 512, and the last window runs past the end.
        assert primary_spans == [(0, 200, 512), (200, 400, 512),
                                 (400, 600, 512), (600, 608, 512)]
        assert primary_spans[-1][1] > 512
        assert sum(e - s for s, e, _ in primary_spans) == 608

    def test_a_file_without_stt_keywords_falls_through_to_the_duplicate(self, tmp_path, caplog):
        """PINNED AS-IS. ``your`` indexes ``STT_IMJD`` with no default.

        ``SpectraInfo.__init__`` raises ``KeyError``, the generator's
        ``except Exception`` swallows it and the duplicated astropy copy runs
        instead -- with no warning that says the epoch is gone, only a generic
        "falling back to astropy". So on such a file the reader that runs in
        production is the one that is never otherwise exercised.
        """
        pytest.importorskip("your")
        truth = _write(tmp_path, "noep.fits", tstart=None)

        with caplog.at_level("WARNING", logger="src.input.fits_handler"):
            blocks = list(fits_handler.stream_fits(
                str(truth["path"]), chunk_samples=200, overlap_samples=0))

        assert any("falling back to astropy" in r.message for r in caplog.records)
        # tstart_mjd is an astropy-copy-only key: its presence proves the branch.
        assert all("tstart_mjd" in m for _, m in blocks)
        assert [m["tstart_mjd"] for _, m in blocks] == [0.0, 0.0, 0.0]
        assert getattr(config, "TSTART_MJD", "<unset>") is None


# ---------------------------------------------------------------------------
# stream_fits_multi_pol
# ---------------------------------------------------------------------------
class TestStreamFitsMultiPol:
    def setup_method(self):
        pytest.importorskip("your")

    def test_geometry_and_shapes(self, tmp_path):
        truth = _write(tmp_path, "iquv.fits", npol=4, pol_type="IQUV",
                       channel_marker=True, dtype="uint8")
        out = list(fits_handler.stream_fits_multi_pol(
            str(truth["path"]), chunk_samples=128, overlap_samples=16))

        geometry = [
            (selected.shape[0], raw.shape[1]) + tuple(meta[k] for k in METADATA_GEOMETRY_KEYS)
            for selected, raw, meta, _ in out
        ]
        assert geometry == [
            (144, 4, 0, 0, 128, 128, 0, 144, 0, 16),
            (160, 4, 1, 128, 256, 128, 112, 272, 16, 16),
            (160, 4, 2, 256, 384, 128, 240, 400, 16, 16),
            (144, 4, 3, 384, 512, 128, 368, 512, 16, 0),
        ]
        for selected, raw, meta, _ in out:
            assert selected.shape[1] == 1
            assert raw.shape[0] == selected.shape[0]
            assert meta["npol"] == 4        # multi-pol only: absent in stream_fits
            assert meta["nifs"] == 1
            assert meta["nchans"] == NCHAN
            assert meta["total_samples"] == NSAMPLES
            assert meta["file_type"] == "fits"
            assert meta["tbin_sec"] == pytest.approx(TSAMP)
            assert "tstart_mjd" not in meta

        starts = [m["t_rel_start_sec"] for _, _, m, _ in out]
        assert starts == pytest.approx([0.0, 0.128, 0.256, 0.384])

    def test_selected_polarisation_is_the_first_and_channels_are_ascending(self, tmp_path):
        truth = _write(tmp_path, "iquv.fits", npol=4, pol_type="IQUV",
                       channel_marker=True, dtype="uint8", fch1=1500.0, foff=-1.0)
        assert config.DATA_NEEDS_REVERSAL is True

        selected, raw, meta, _ = next(iter(fits_handler.stream_fits_multi_pol(
            str(truth["path"]), chunk_samples=256, overlap_samples=0)))

        # channel_marker writes pol p as (ascending index + p * nchan).
        assert np.array_equal(raw[0, 0, :], np.arange(NCHAN, dtype=np.float32))
        assert np.array_equal(raw[0, 1, :], np.arange(NCHAN, dtype=np.float32) + NCHAN)
        assert np.array_equal(selected[0, 0, :], raw[0, 0, :])

    def test_reported_pol_type_is_hardcoded(self, tmp_path):
        """PINNED AS-IS, and it is wrong.

        Both ``stream_fits_multi_pol`` and the ``your`` branch of
        ``stream_fits`` do ``getattr(pf, 'pol_type', 'IQUV')``. ``PsrfitsFile``
        has no ``pol_type`` attribute -- the real one is ``poln_order`` -- so the
        default is returned for every file ever read, and the reported
        polarisation order is the literal string "IQUV" whatever the header
        says. ``get_obparams`` reads the header correctly, so the two disagree
        about the same file.
        """
        truth = _write(tmp_path, "crci.fits", npol=4, pol_type="AABBCRCI",
                       channel_marker=True, dtype="uint8")
        assert config.POL_TYPE == "AABBCRCI"

        _, _, _, reported = next(iter(fits_handler.stream_fits_multi_pol(
            str(truth["path"]), chunk_samples=256, overlap_samples=0)))
        assert reported == "IQUV"

        from your.formats import psrfits as your_psrfits
        handle = your_psrfits.PsrfitsFile([str(truth["path"])])
        assert not hasattr(handle, "pol_type")
        assert handle.poln_order == "AABBCRCI"

    def test_two_polarisation_file_cannot_be_streamed(self, tmp_path):
        """PINNED AS-IS. ``your``'s ``read_subint`` sums AA and BB down to one
        polarisation and then reshapes to ``npoln=2``, which cannot work. The
        error is swallowed and re-raised as the generic "requires the
        'your_psrfits' library" message, which is misleading -- the library is
        installed."""
        truth = _write(tmp_path, "aabb.fits", npol=2, pol_type="AABB",
                       channel_marker=True, dtype="uint8")

        with pytest.raises(RuntimeError, match="requires the 'your_psrfits' library"):
            list(fits_handler.stream_fits_multi_pol(
                str(truth["path"]), chunk_samples=256, overlap_samples=0))

    def test_raises_without_the_library(self, tmp_path, monkeypatch):
        truth = _write(tmp_path, "iquv.fits", npol=4, pol_type="IQUV")
        monkeypatch.setattr(fits_handler, "your_psrfits", None)

        with pytest.raises(RuntimeError, match="requires the 'your_psrfits' library"):
            list(fits_handler.stream_fits_multi_pol(str(truth["path"])))
