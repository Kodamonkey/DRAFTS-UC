"""Regression tests for the P1 correctness defects from the 2026-09 audit.

P1-01  trim_valid_window() ignored the right overlap, so the tail of every chunk
       was processed twice: once as its own valid region and again as the next
       chunk's, duplicating candidates and scoring them on a partial channel sum.
P1-03  DM chunking re-derived a grid per chunk, changing the DM step, duplicating
       the boundary rows and never searching DM_max.
P1-04  The astropy fallback reported start_sample at the block's start instead of
       the valid region's, shifting every absolute time by the full overlap.
P1-05  The emergency emission path dropped actual+2*overlap samples per chunk,
       leaving a 2*overlap hole that was never searched.
P1-06  The first chunk of a FITS file declared a left overlap it did not have, so
       the opening overlap_samples of every file were discarded.
P1-09  get_obparams() left the previous file's TSTART_MJD_CORR in place when the
       current file had no epoch, stamping its candidates with another
       observation's date.
P1-11  The SPEC-HF-002 branch returned valid bounds in a different frame from the
       normal branch, shifting the polarisation waveforms against intensity.
P1-12  snr_pre_dedisp was measured on the DM-time cube, so compute_snr_profile
       was filtering along the DM axis and not returning an SNR at all.
"""
from __future__ import annotations

import ast
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.core import detection_engine as de
from src.core.data_flow_manager import (
    _build_dm_time_cube_chunked,
    trim_valid_window,
)
from src.core.pipeline_parameters import calculate_dm_height
from src.detection.model_interface import CNN_IMG_SIZE
from src.input.filterbank_handler import get_obparams_fil, stream_fil
from src.output.candidate_manager import CandidateWriter, ensure_csv_header
from src.preprocessing.dedispersion import d_dm_time_g
from tests.synthetic_filterbank import write_filterbank

FITS_HANDLER = PROJECT_ROOT / "src/input/fits_handler.py"
HF_PIPELINE = PROJECT_ROOT / "src/core/high_freq_pipeline.py"


def _function_ast(module_path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {module_path.name}")


# --------------------------------------------------------------------------- #
# P1-01
# --------------------------------------------------------------------------- #

class TestTrimValidWindow:
    def test_both_overlaps_are_discarded(self):
        n, left, right = 100, 7, 11
        block = np.arange(n, dtype=np.float32)[:, None] * np.ones((1, 4), np.float32)
        cube = np.tile(np.arange(n, dtype=np.float32), (3, 5, 1))

        block_valid, dm_time, start, end = trim_valid_window(block, cube, left, right)

        assert (start, end) == (left, n - right)
        assert block_valid.shape[0] == n - left - right
        assert dm_time.shape[-1] == n - left - right
        # The window must be the interior, not the head of the block.
        assert block_valid[0, 0] == pytest.approx(float(left))
        assert block_valid[-1, 0] == pytest.approx(float(n - right - 1))

    def test_consecutive_chunks_tile_without_overlap(self, tmp_path):
        """With the real reader's geometry, consecutive valid windows must abut:
        no sample processed twice, none skipped."""
        fil = tmp_path / "tile.fil"
        write_filterbank(fil, nsamples=9000, tsamp=0.001)
        get_obparams_fil(str(fil))

        chunk, overlap = 2000, 300
        covered: list[tuple[int, int]] = []
        for block, meta in stream_fil(str(fil), chunk, overlap_samples=overlap):
            left = meta["overlap_left"]
            right = meta["overlap_right"]
            _, _, start, end = trim_valid_window(
                block, np.zeros((3, 2, block.shape[0]), np.float32), left, right
            )
            abs_start = meta["block_start_sample"] + start
            abs_end = meta["block_start_sample"] + end
            covered.append((abs_start, abs_end))

        for (_, prev_end), (next_start, _) in zip(covered, covered[1:]):
            assert next_start == prev_end, (
                f"valid windows do not tile: {prev_end} -> {next_start}"
            )
        assert covered[0][0] == 0, "the start of the file must be covered"
        assert covered[-1][1] == 9000, "the end of the file must be covered"

    def test_degenerate_overlap_falls_back_to_the_whole_block(self):
        block = np.zeros((10, 2), np.float32)
        cube = np.zeros((3, 2, 10), np.float32)
        _, _, start, end = trim_valid_window(block, cube, 8, 8)
        assert (start, end) == (0, 10)


# --------------------------------------------------------------------------- #
# P1-03
# --------------------------------------------------------------------------- #

class TestDmChunkingParity:
    def _setup(self):
        config.DM_min, config.DM_max = 0.0, 1000.0
        config.DM_GRID_MODE = "legacy_uniform"
        config.TIME_RESO, config.DOWN_TIME_RATE = 0.001, 1
        config.DOWN_FREQ_RATE = 1
        config.FREQ = np.linspace(1200.0, 1500.0, 32)
        config.FREQ_RESO = 32

    def test_chunked_cube_matches_the_direct_one(self):
        """The previous test only ever exercised a single DM chunk, which is the
        one case where the per-chunk grid happened to be right."""
        self._setup()
        height = calculate_dm_height()
        block = np.random.RandomState(0).rand(400, 32).astype(np.float32)

        direct = d_dm_time_g(block, height=height, width=400)
        chunked = np.asarray(
            _build_dm_time_cube_chunked(
                block, height=height, dm_min=0.0, dm_max=1000.0, threshold_gb=0.0005,
            )
        )

        assert chunked.shape == direct.shape
        assert np.array_equal(chunked, direct), (
            f"max |diff| = {np.max(np.abs(chunked - direct))}"
        )

    def test_dm_values_must_match_the_declared_height(self):
        self._setup()
        block = np.random.RandomState(0).rand(64, 32).astype(np.float32)
        with pytest.raises(ValueError):
            d_dm_time_g(block, height=10, width=64, dm_values=np.linspace(0, 100, 7))


# --------------------------------------------------------------------------- #
# P1-09
# --------------------------------------------------------------------------- #

class TestEpochIsNotInheritedBetweenFiles:
    def test_file_without_tstart_clears_the_previous_epoch(self, tmp_path):
        with_epoch = tmp_path / "a.fil"
        without_epoch = tmp_path / "b.fil"
        write_filterbank(with_epoch, nsamples=500, tstart=60000.0)
        write_filterbank(without_epoch, nsamples=500, tstart=None)

        get_obparams_fil(str(with_epoch))
        assert config.TSTART_MJD == pytest.approx(60000.0)
        assert config.TSTART_MJD_CORR == pytest.approx(60000.0)

        get_obparams_fil(str(without_epoch))
        assert config.TSTART_MJD is None
        assert config.TSTART_MJD_CORR is None, (
            "the previous file's epoch survived; every candidate of this file "
            "would be stamped with another observation's date"
        )

    def test_fits_reader_also_clears_both_epochs(self):
        """The PSRFITS reader had no else branch at all. Building a synthetic
        PSRFITS is out of proportion here, so the contract is asserted on the
        source: both attributes must be cleared together."""
        src = FITS_HANDLER.read_text(encoding="utf-8")
        marker = "logger.warning(\"STT_* not found in %s: absolute MJD unavailable\", file_name)"
        assert marker in src, "get_obparams no longer reports a missing epoch"
        after = src[src.index(marker):src.index(marker) + 400]
        assert "config.TSTART_MJD = None" in after
        assert "config.TSTART_MJD_CORR = None" in after


# --------------------------------------------------------------------------- #
# P1-11
# --------------------------------------------------------------------------- #

class TestHighFreqValidBoundsFrame:
    def test_unresolved_dm_branch_reports_bounds_in_the_untrimmed_frame(self):
        """Both branches must hand back bounds that index the untrimmed block,
        because block_raw_ds is sliced with them while still untrimmed. Returning
        0-based bounds shifted every polarisation waveform by overlap_left_ds."""
        src = HF_PIPELINE.read_text(encoding="utf-8")
        assert "valid_start_ds = overlap_left_ds" in src, (
            "the SPEC-HF-002 branch no longer aligns with trim_valid_window"
        )
        assert "valid_start_ds, valid_end_ds = 0, n_valid" not in src, (
            "the 0-based bounds are back: intensity and polarisation will be "
            "misaligned by the left overlap"
        )


# --------------------------------------------------------------------------- #
# P1-12
# --------------------------------------------------------------------------- #

class TestSnrPreDedispersionComesFromTheWaterfall:
    def test_snr_pre_dedisp_matches_the_waterfall_at_the_candidate(self, tmp_path, monkeypatch):
        config.DM_min, config.DM_max = 480.0, 640.0
        config.TIME_RESO, config.DOWN_TIME_RATE = 0.000256, 4
        config.DOWN_FREQ_RATE = 1
        config.FREQ_RESO = 32
        config.FREQ = np.linspace(1200.0, 1500.0, 32)
        config.CLASS_PROB = 0.5
        config.SAVE_ONLY_BURST = False
        config.SNR_THRESH = 5.0
        config.TRIAL_CORRECTION = "none"

        slice_len = 512
        rng = np.random.RandomState(0)
        band_img = rng.rand(147, slice_len).astype(np.float32)
        data = rng.rand(slice_len, 32).astype(np.float32)
        # A waterfall with one unmistakable spike.
        waterfall = rng.normal(1.0, 0.01, size=(slice_len, 32)).astype(np.float32)
        spike_at = 300
        waterfall[spike_at - 1:spike_at + 2, :] += 25.0

        box_centre = 300.0 * CNN_IMG_SIZE / slice_len
        monkeypatch.setattr(
            de, "detect",
            lambda m, t: ([0.9], [[box_centre - 2, 250.0, box_centre + 2, 260.0]]),
        )
        monkeypatch.setattr(
            de, "preprocess_img",
            lambda img: np.zeros((3, CNN_IMG_SIZE, CNN_IMG_SIZE), np.float32),
        )
        monkeypatch.setattr(
            de, "postprocess_img",
            lambda t: np.zeros((CNN_IMG_SIZE, CNN_IMG_SIZE, 3), np.uint8),
        )

        csv_file = tmp_path / "c.csv"
        ensure_csv_header(csv_file)
        de.detect_and_classify_candidates_in_band(
            det_model=None, cls_model=None, band_img=band_img, slice_len=slice_len,
            j=0, fits_path=Path("synthetic.fits"), save_dir=tmp_path, data=data,
            freq_down=np.linspace(1200.0, 1500.0, 32), csv_file=csv_file,
            time_reso_ds=config.TIME_RESO * config.DOWN_TIME_RATE, snr_list=[],
            config=config, band_idx=0, chunk_idx=0, slice_start_idx=0,
            waterfall_block=waterfall, patches_dir=tmp_path / "p",
        )
        CandidateWriter.flush_all()

        with csv_file.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        snr_pre = float(rows[0]["snr_pre_dedisp"])

        from src.analysis.snr_utils import compute_snr_profile
        profile, _, _ = compute_snr_profile(waterfall)
        expected = float(profile[int(np.clip(300, 0, profile.size - 1))])

        assert snr_pre == pytest.approx(expected, rel=1e-6), (
            "snr_pre_dedisp is not the waterfall SNR at the candidate"
        )
        # It must be a real detection, not the noise floor of a DM-axis filter.
        assert snr_pre > 5.0


# --------------------------------------------------------------------------- #
# P1-04 / P1-05 / P1-06
# --------------------------------------------------------------------------- #

class TestFitsChunkGeometry:
    """Both astropy emission paths in stream_fits must agree on the geometry.

    Driving these needs a real PSRFITS; the three defects are single expressions
    inside a 600-line generator, so the contract is asserted on the source. The
    behavioural counterpart lives in test_e2e_pipeline, which exercises the same
    invariants over the filterbank reader.
    """

    def test_start_sample_is_the_valid_regions_start(self):
        src = FITS_HANDLER.read_text(encoding="utf-8")
        bare = src.count("start_sample_idx = emitted - out_buf.shape[0]\n")
        withval = src.count("start_sample_idx = emitted - out_buf.shape[0] + valid_start")
        assert bare == 0, (
            "a chunk still reports the block's start as start_sample; every "
            "absolute time in it is early by the full overlap"
        )
        assert withval == 2, f"expected both emission paths to be fixed, found {withval}"

    def test_buffer_advances_by_the_valid_span_only(self):
        src = FITS_HANDLER.read_text(encoding="utf-8")
        assert "samples_to_remove = actual_chunk_size if not buffer_too_large" not in src, (
            "the emergency path drops 2*overlap extra samples between chunks"
        )
        assert src.count("samples_to_remove = actual_chunk_size") == 2

    def test_first_chunk_declares_no_left_overlap(self):
        src = FITS_HANDLER.read_text(encoding="utf-8")
        assert src.count("if emitted - out_buf.shape[0] <= 0 and valid_start > 0:") == 2, (
            "the opening overlap_samples of each file are still discarded"
        )

    def test_metadata_overlaps_are_derived_not_assumed(self):
        src = FITS_HANDLER.read_text(encoding="utf-8")
        # The defect was reporting a constant overlap regardless of where the
        # chunk actually sits in the file.
        assert '"overlap_left": overlap_samples,' not in src
        assert '"overlap_right": overlap_samples,' not in src
        # Every metadata dict either derives the overlap from the geometry or
        # states 0 explicitly (the tail emissions).
        derived = src.count('"overlap_left": valid_start - start_with_overlap,')
        explicit_zero = src.count('"overlap_left": 0,')
        assert derived + explicit_zero == src.count('"overlap_left"'), (
            "some metadata dict reports an overlap that is neither derived nor zero"
        )
        # The two buffered emission paths are the ones the audit flagged.
        assert src.count('"overlap_right": max(0, end_with_overlap - valid_end),') == 2
