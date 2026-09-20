"""Drive ``snr_detect_and_classify_candidates_in_band`` for real.

Why this module exists
----------------------
The high-frequency band function is where a candidate is actually made: it
finds the SNR peaks, measures a DM for each, runs the two classification
phases, applies the decision table, counts the verdicts and writes the CSV row.
Nothing executed it. ``tests/test_hf_e2e.py`` stubs
``process_slice_with_multiple_bands_high_freq`` out, and the unit tests around
it (``test_hf_phases.py``) call its extracted helpers in isolation. So the body
that stitches those helpers together rested on the helpers' own tests and on
reading the code -- mutating lines inside it left the suite green.

This module closes that. Every test here calls the production function with the
arguments its one production caller passes, in the shapes that caller passes
them (see ``process_slice_with_multiple_bands_high_freq``):

    band_img          slice_cube[band_idx]          (dm_rows, slice_len)
    waterfall_block   block[start:end]              (slice_len, n_chan)
    data_block        block                         (chunk_samples, n_chan)
    waterfall_raw     block_raw[start:end]          (slice_len, npol, n_chan)
    data_block_raw    block_raw                     (chunk_samples, npol, n_chan)
    dm_time_fullband  slice_cube[0]                 (dm_rows, slice_len)

No classifier is needed
-----------------------
``classify_patch`` falls back to a deterministic SNR sigmoid when the model is
``None``, so Phases 3a and 3b can be switched *on* with ``cls_model=None`` and
still produce real verdicts, real rows and a real decision. That is what makes
the write path reachable here at all. Where a test needs a *particular* verdict
per candidate -- a burst and a non-burst in the same call -- it installs
``_ScriptedModel`` instead, which is the one seam these tests cut: everything
inside the band function stays the production code.

Why the geometry is what it is
------------------------------
The waterfall carries two Gaussian bursts, at slice samples 128 and 384. On
that data ``_find_snr_peaks`` returns exactly those two for any threshold
between about 14 and 30 (measured: 12 gives eight peaks, 14 through 30 give
two), so ``SNR_THRESH = 20`` sits in the middle of a wide plateau rather than
on a boundary a float32 difference could cross. Two candidates, not one,
because a single one cannot show a burst and a non-burst in the same call, and
so cannot show what ``SAVE_ONLY_BURST`` filters or what the STRICT arm of the
decision table does.

The DM-time cube puts its ridge on row 32 under the first burst and row 96
under the second, so the two candidates must come back with *different* DMs.
A mutation that measured the DM at the wrong column would give them the same
one.
"""
from __future__ import annotations

import csv as _csv
import math
from pathlib import Path

import numpy as np
import pytest

from src.config import config
from src.core import candidate_finalization
from src.core import high_freq_pipeline as hfp
from src.core.contracts import PipelineConfigSnapshot
from src.output.candidate_manager import CandidateWriter, ensure_csv_header

# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
N_CHAN = 32
CHUNK_SAMPLES = 2048
SLICE_START = 512
SLICE_END = 1024
SLICE_LEN = SLICE_END - SLICE_START

DM_ROWS = 128
DM_MIN = 0.0
DM_MAX = 200.0

#: Slice samples the two injected bursts sit on, and the DM-cube row that peaks
#: under each. They differ so the two candidates cannot share a DM by accident.
BURST_A, BURST_B = 128, 384
DM_ROW_A, DM_ROW_B = 32, 96

#: Middle of the plateau on which exactly the two bursts are peaks. See the
#: module docstring.
SNR_THRESH = 20.0

TIME_RESO = 1.0e-3
TSTART_MJD = 60000.0
ABS_START = 7.5
CHUNK_IDX = 2
SLICE_IDX = 3

#: ``DMGrid.dm_for_row`` on a 128-row image over 0-200 pc cm^-3 is linear in the
#: row, so the DM a candidate must report is fixed by the ridge row alone.
DM_A = DM_MAX * DM_ROW_A / (DM_ROWS - 1)
DM_B = DM_MAX * DM_ROW_B / (DM_ROWS - 1)

#: The CSV rounds the DM to two decimals and the time columns to six, so
#: comparisons against the computed value are made to the column's precision
#: rather than to the float's.
DM_COLUMN_ABS = 0.005
SEC_COLUMN_ABS = 5e-7

#: Distinguishes "the caller said nothing" from "the caller passed an array",
#: which ``is None`` cannot do here because ``None`` is a meaningful value for
#: the raw block: it is the single-polarisation case.
_DEFAULT = object()


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def _gaussian(centre: int, amp: float, width: float = 3.0) -> np.ndarray:
    t = np.arange(CHUNK_SAMPLES, dtype=np.float32)
    return (amp * np.exp(-0.5 * ((t - centre) / width) ** 2)).astype(np.float32)


def _intensity_chunk(*, bursts: bool = True) -> np.ndarray:
    """A (time, chan) Stokes-I chunk, with or without the two bursts."""
    rng = np.random.default_rng(17)
    block = rng.normal(10.0, 1.0, size=(CHUNK_SAMPLES, N_CHAN)).astype(np.float32)
    if bursts:
        for offset in (BURST_A, BURST_B):
            block += _gaussian(SLICE_START + offset, 40.0)[:, None]
    return block


def _raw_chunk(block: np.ndarray, *, linear_burst: bool = True) -> np.ndarray:
    """A (time, 4, chan) IQUV chunk whose Stokes I is *block*.

    Q and U carry the burst when *linear_burst*, so the Linear product has an
    SNR peak where Intensity does; without it the Linear profile is noise,
    which is what Phase 2 is supposed to reject.
    """
    rng = np.random.default_rng(29)
    raw = rng.normal(1.0, 0.2, size=(CHUNK_SAMPLES, 4, N_CHAN)).astype(np.float32)
    raw[:, 0, :] = block
    if linear_burst:
        for offset in (BURST_A, BURST_B):
            burst = _gaussian(SLICE_START + offset, 12.0)[:, None]
            raw[:, 1, :] += burst
            raw[:, 2, :] += burst
    return raw


def _dm_cube(*, flat: bool = False) -> np.ndarray:
    """A (band, dm_row, time) cube with a ridge under each burst."""
    if flat:
        # Every DM trial identical: the column has no variation to read a DM
        # from, which is the high-frequency case ``resolve_candidate_dm`` exists
        # for.
        return np.full((3, DM_ROWS, CHUNK_SAMPLES), 2.0, dtype=np.float32)
    rng = np.random.default_rng(5)
    cube = rng.normal(0.0, 0.05, size=(3, DM_ROWS, CHUNK_SAMPLES)).astype(np.float32)
    for offset, row in ((BURST_A, DM_ROW_A), (BURST_B, DM_ROW_B)):
        centre = SLICE_START + offset
        cube[:, row, centre - 40:centre + 40] += 6.0
    return cube


def _snapshot(**over) -> PipelineConfigSnapshot:
    """A snapshot with every phase off, so each test switches on what it needs."""
    from types import SimpleNamespace

    values = {
        "DM_min": DM_MIN,
        "DM_max": DM_MAX,
        "SNR_THRESH": SNR_THRESH,
        "CLASS_PROB": 0.5,
        "SAVE_ONLY_BURST": False,
        "ENABLE_INTENSITY_CLASSIFICATION": False,
        "ENABLE_LINEAR_CLASSIFICATION": False,
        "ENABLE_LINEAR_VALIDATION": False,
    }
    values.update(over)
    return PipelineConfigSnapshot.from_config(SimpleNamespace(**values))


class _ScriptedModel:
    """A stand-in for the ResNet that hands out verdicts in call order.

    The fallback classifier gives every candidate in this data a similar
    probability, which is fine for "all bursts" and "no bursts" but cannot
    produce one of each. This can. It also checks that the ``cls_model``
    argument reaches ``classify_patch``: if the band function stopped
    forwarding it, ``_install`` 's assertion fires.

    Call order inside the loop is Phase 3a then Phase 3b, per candidate, and
    the candidates run global-peak-first.
    """

    def __init__(self, probs):
        self.probs = list(probs)
        self.calls = 0

    def next_prob(self) -> float:
        prob = self.probs[self.calls % len(self.probs)]
        self.calls += 1
        return float(prob)


def _install_scripted(monkeypatch, model: _ScriptedModel) -> None:
    def _fake_classify(passed_model, patch):
        assert passed_model is model, "cls_model did not reach classify_patch"
        proc = np.zeros((4, 4), dtype=np.float32) if patch is None else np.asarray(
            patch, dtype=np.float32
        )
        return model.next_prob(), proc

    # Phase 3a reaches classify_patch through candidate_finalization; Phase 3b
    # calls the name high_freq_pipeline imported. Both are seams of the same
    # function and both must be replaced or the two phases disagree.
    monkeypatch.setattr(candidate_finalization, "classify_patch", _fake_classify)
    monkeypatch.setattr(hfp, "classify_patch", _fake_classify)


# --------------------------------------------------------------------------- #
# the harness
# --------------------------------------------------------------------------- #
@pytest.fixture
def band(tmp_path, monkeypatch):
    """Return ``call(**kwargs) -> (result, rows)`` for the band function."""
    for key, value in {
        "FREQ": np.linspace(1200.0, 1500.0, N_CHAN).astype(np.float64),
        "FREQ_RESO": N_CHAN,
        "TIME_RESO": TIME_RESO,
        "DOWN_TIME_RATE": 1,
        "DOWN_FREQ_RATE": 1,
        "DM_min": DM_MIN,
        "DM_max": DM_MAX,
        # Read by ``classify_patch``'s model-free fallback, not by the band
        # function, which takes its own threshold from the snapshot.
        "SNR_THRESH": 5.0,
        "RESULTS_DIR": tmp_path,
        "TSTART_MJD": TSTART_MJD,
        "HIGH_FREQ_DM_POLICY": "unresolved",
        # Absent source configuration: the barycentric columns are then reported
        # unavailable instead of guessed, and nothing reaches for the network.
        "SOURCE_RA": None,
        "SOURCE_DEC": None,
        "REF_FREQ_MHZ": None,
        "OBSERVATORY": None,
        "EPHEMERIS": None,
    }.items():
        monkeypatch.setattr(config, key, value, raising=False)

    counter = {"n": 0}

    def call(
        *,
        snapshot=None,
        block=None,
        raw=_DEFAULT,
        cube=None,
        band_idx: int = 0,
        cls_model=None,
        metrics_tracker=None,
        snr_list=None,
        slice_len=SLICE_LEN,
        slice_samples=SLICE_LEN,
    ):
        block = _intensity_chunk() if block is None else block
        if raw is _DEFAULT:
            raw = _raw_chunk(block)
        cube = _dm_cube() if cube is None else cube
        slice_cube = cube[:, :, SLICE_START:SLICE_END]

        counter["n"] += 1
        csv_file = tmp_path / f"band{counter['n']}.candidates.csv"
        ensure_csv_header(csv_file)

        result = hfp.snr_detect_and_classify_candidates_in_band(
            cls_model,
            slice_cube[band_idx],
            block[SLICE_START:SLICE_END],
            slice_len,
            SLICE_IDX,
            Path("harness.fits"),
            tmp_path,
            block,
            config.FREQ.astype(np.float32),
            csv_file,
            TIME_RESO,
            [] if snr_list is None else snr_list,
            ABS_START,
            tmp_path,
            CHUNK_IDX,
            band_idx,
            SLICE_START,
            waterfall_block_raw=None if raw is None else raw[SLICE_START:SLICE_END],
            data_block_raw=raw,
            pol_type="IQUV",
            slice_samples=slice_samples,
            dm_time_fullband=slice_cube[0],
            metrics_tracker=metrics_tracker,
            snapshot=_snapshot() if snapshot is None else snapshot,
        )
        CandidateWriter.flush_all()
        with csv_file.open(newline="", encoding="utf-8") as handle:
            rows = list(_csv.DictReader(handle))
        return result, rows

    return call


# --------------------------------------------------------------------------- #
# Phase 1
# --------------------------------------------------------------------------- #
class TestPhase1Detection:
    def test_it_finds_the_two_injected_bursts_and_nothing_else(self, band):
        result, _ = band()
        assert result["cand_counter"] == 2

    def test_the_global_peak_is_the_first_candidate(self, band):
        """The band function reorders the peak list to put the global maximum
        first, because the first candidate is the one the composite plot and
        the driver's "best" patch are taken from. Detection order alone would
        put sample 128 first; the global peak is 384."""
        result, _ = band()
        times = result["candidate_times_abs"]
        assert times[0] == pytest.approx(ABS_START + BURST_B * TIME_RESO)
        assert times[1] == pytest.approx(ABS_START + BURST_A * TIME_RESO)

    def test_a_quiet_waterfall_returns_the_empty_result(self, band):
        result, rows = band(block=_intensity_chunk(bursts=False))
        assert rows == []
        assert result["cand_counter"] == 0
        assert result["first_dm"] is None
        assert result["img_tensor"] is None

    def test_the_empty_result_carries_every_key_the_caller_indexes(self, band):
        """``process_slice_with_multiple_bands_high_freq`` reads these with
        ``[]``, not ``.get()``, so a key missing from the no-candidate result is
        a KeyError on the common path."""
        result, _ = band(block=_intensity_chunk(bursts=False))
        for key in ("top_conf", "top_boxes", "class_probs_list", "first_patch",
                    "first_start", "first_dm", "img_tensor", "cand_counter",
                    "n_bursts", "n_no_bursts", "prob_max", "patch_path",
                    "best_is_burst", "total_candidates", "candidate_times_abs"):
            assert key in result, key

    def test_the_snr_of_each_peak_is_appended_to_the_shared_list(self, band):
        """``snr_list`` is the accumulator the driver passes down a whole file;
        the band function appends to it rather than returning it."""
        shared = [1.0]
        result, _ = band(snr_list=shared)
        assert len(shared) == 1 + result["cand_counter"]
        assert all(value >= SNR_THRESH for value in shared[1:])


# --------------------------------------------------------------------------- #
# the snapshot, not the global, decides
# --------------------------------------------------------------------------- #
class TestSnapshotIsAuthoritative:
    """REF-10: the band function took twenty-one values off the mutable global.
    It now takes them from ``PipelineConfigSnapshot``. These two tests are what
    tells the difference -- with both values equal, as they are in production,
    nothing can."""

    def test_the_threshold_comes_from_the_snapshot_not_from_config(self, band, monkeypatch):
        monkeypatch.setattr(config, "SNR_THRESH", 1.0e6, raising=False)
        result, _ = band(snapshot=_snapshot(SNR_THRESH=SNR_THRESH))
        assert result["cand_counter"] == 2

    def test_a_snapshot_threshold_above_the_data_finds_nothing(self, band, monkeypatch):
        monkeypatch.setattr(config, "SNR_THRESH", 1.0, raising=False)
        result, rows = band(snapshot=_snapshot(SNR_THRESH=1.0e6))
        assert result["cand_counter"] == 0
        assert rows == []

    def test_the_trials_count_uses_the_snapshot_dm_range(self, band, monkeypatch):
        """``score_candidate`` is handed ``snap.dm_min``/``snap.dm_max``. The
        global DM range is left at 0-200 here, so a value of 101 * 512 can only
        have come from the snapshot."""
        monkeypatch.setattr(config, "DM_min", 0.0, raising=False)
        monkeypatch.setattr(config, "DM_max", 200.0, raising=False)
        _, rows = band(snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True,
                                          DM_max=100.0))
        assert int(rows[0]["n_trials"]) == 101 * SLICE_LEN


# --------------------------------------------------------------------------- #
# P1-10: a phase that did not run has no verdict
# --------------------------------------------------------------------------- #
class TestNoClassificationRuns:
    """With both classification phases off nothing can say whether a candidate
    is a burst, and the band function must say exactly that: count it, file it
    in neither column, and write no row. Audit P1-10 is what happens when it
    instead substitutes a value -- every candidate came back a burst."""

    def test_candidates_are_counted_but_not_classified(self, band):
        result, _ = band(snapshot=_snapshot())
        assert result["cand_counter"] == 2
        assert result["n_bursts"] == 0
        assert result["n_no_bursts"] == 0

    def test_nothing_is_written_to_the_csv(self, band):
        _, rows = band(snapshot=_snapshot())
        assert rows == []

    def test_the_probability_lists_are_nan_not_zero(self, band):
        """These two lists feed the composite plot. NaN draws nothing; 0.0 would
        draw a confident non-detection the pipeline never made."""
        result, _ = band(snapshot=_snapshot())
        assert all(math.isnan(p) for p in result["class_probs_list"])
        assert all(math.isnan(p) for p in result["class_probs_linear_list"])

    def test_the_best_candidate_has_no_verdict(self, band):
        result, _ = band(snapshot=_snapshot())
        assert result["best_is_burst"] is None

    def test_the_confidence_is_the_snr_not_a_class_probability(self, band):
        """``prob_max`` is ``min(0.99, snr/10)``, which exists whether or not a
        classifier ran. It is the one number this branch can still report."""
        result, _ = band(snapshot=_snapshot())
        assert result["prob_max"] == pytest.approx(0.99)
        assert result["top_conf"] == [pytest.approx(0.99)] * 2


# --------------------------------------------------------------------------- #
# Phase 3a
# --------------------------------------------------------------------------- #
class TestIntensityClassification:
    def test_enabling_phase_3a_writes_the_rows(self, band):
        result, rows = band(snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True))
        assert result["cand_counter"] == 2
        assert len(rows) == 2

    def test_the_intensity_columns_are_filled_and_the_linear_ones_are_not(self, band):
        _, rows = band(snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True))
        for row in rows:
            assert 0.0 <= float(row["class_prob_intensity"]) <= 1.0
            assert row["is_burst_intensity"] in {"burst", "no_burst"}
            assert row["class_prob_linear"] == ""
            assert row["is_burst_linear"] == ""

    def test_a_threshold_above_every_probability_makes_them_all_no_burst(self, band):
        result, _ = band(snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True,
                                            CLASS_PROB=1.01))
        assert result["n_bursts"] == 0
        assert result["n_no_bursts"] == 2
        assert result["best_is_burst"] is False

    def test_a_threshold_below_every_probability_makes_them_all_bursts(self, band):
        result, _ = band(snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True,
                                            CLASS_PROB=0.0))
        assert result["n_bursts"] == 2
        assert result["n_no_bursts"] == 0
        assert result["best_is_burst"] is True

    def test_the_verdicts_always_account_for_every_candidate(self, band):
        for class_prob in (0.0, 1.01):
            result, rows = band(snapshot=_snapshot(
                ENABLE_INTENSITY_CLASSIFICATION=True, CLASS_PROB=class_prob))
            assert result["n_bursts"] + result["n_no_bursts"] == result["cand_counter"]
            assert len(rows) == result["cand_counter"]


# --------------------------------------------------------------------------- #
# the decision table, driven through the real function
# --------------------------------------------------------------------------- #
class TestDecisionTable:
    """``decide_candidate`` has its own unit tests. What those cannot show is
    that the band function feeds it the right availability flags and then acts
    on what it returns -- which row gets written, which counter moves."""

    def test_one_burst_and_one_non_burst_in_the_same_call(self, band, monkeypatch):
        model = _ScriptedModel([0.9, 0.1])  # candidate 0 burst, candidate 1 not
        _install_scripted(monkeypatch, model)
        result, rows = band(cls_model=model,
                            snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True,
                                               CLASS_PROB=0.5))
        assert model.calls == 2
        assert (result["n_bursts"], result["n_no_bursts"]) == (1, 1)
        assert [row["is_burst"] for row in rows] == ["burst", "no_burst"]

    def test_save_only_burst_keeps_the_burst_and_drops_the_other(self, band, monkeypatch):
        model = _ScriptedModel([0.9, 0.1])
        _install_scripted(monkeypatch, model)
        result, rows = band(cls_model=model,
                            snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True,
                                               CLASS_PROB=0.5, SAVE_ONLY_BURST=True))
        # Both are still counted; only one is written. The counters describe
        # what was found, the file describes what was kept.
        assert result["cand_counter"] == 2
        assert (result["n_bursts"], result["n_no_bursts"]) == (1, 1)
        assert [row["is_burst"] for row in rows] == ["burst"]

    def test_permissive_keeps_a_candidate_either_phase_calls_a_burst(self, band, monkeypatch):
        # Order is 3a, 3b per candidate: candidate 0 is I-burst/L-no,
        # candidate 1 is I-no/L-burst. Neither agrees with itself.
        model = _ScriptedModel([0.9, 0.1, 0.1, 0.9])
        _install_scripted(monkeypatch, model)
        result, rows = band(cls_model=model,
                            snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True,
                                               ENABLE_LINEAR_CLASSIFICATION=True,
                                               CLASS_PROB=0.5, CLASS_PROB_LINEAR=0.5,
                                               SAVE_ONLY_BURST=False))
        assert model.calls == 4
        assert (result["n_bursts"], result["n_no_bursts"]) == (2, 0)
        assert [row["is_burst"] for row in rows] == ["burst", "burst"]

    def test_strict_requires_both_phases_to_agree(self, band, monkeypatch):
        model = _ScriptedModel([0.9, 0.1, 0.1, 0.9])
        _install_scripted(monkeypatch, model)
        result, rows = band(cls_model=model,
                            snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True,
                                               ENABLE_LINEAR_CLASSIFICATION=True,
                                               CLASS_PROB=0.5, CLASS_PROB_LINEAR=0.5,
                                               SAVE_ONLY_BURST=True))
        assert (result["n_bursts"], result["n_no_bursts"]) == (0, 2)
        assert rows == []

    def test_strict_keeps_the_candidate_both_phases_agree_on(self, band, monkeypatch):
        model = _ScriptedModel([0.9, 0.9, 0.1, 0.1])
        _install_scripted(monkeypatch, model)
        result, rows = band(cls_model=model,
                            snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True,
                                               ENABLE_LINEAR_CLASSIFICATION=True,
                                               CLASS_PROB=0.5, CLASS_PROB_LINEAR=0.5,
                                               SAVE_ONLY_BURST=True))
        assert (result["n_bursts"], result["n_no_bursts"]) == (1, 1)
        assert [row["is_burst"] for row in rows] == ["burst"]

    def test_the_two_thresholds_are_separate(self, band, monkeypatch):
        """``CLASS_PROB`` and ``CLASS_PROB_LINEAR`` are different settings. With
        the same probability in both phases and the thresholds either side of
        it, the two verdicts must differ."""
        model = _ScriptedModel([0.6])
        _install_scripted(monkeypatch, model)
        _, rows = band(cls_model=model,
                       snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True,
                                          ENABLE_LINEAR_CLASSIFICATION=True,
                                          CLASS_PROB=0.5, CLASS_PROB_LINEAR=0.7))
        assert rows[0]["is_burst_intensity"] == "burst"
        assert rows[0]["is_burst_linear"] == "no_burst"

    def test_linear_classification_without_multipol_data_produces_no_verdict(self, band):
        """Phase 3b enabled but the file has one polarisation: that is an absent
        verdict, not a negative one, and the Intensity phase decides alone."""
        result, rows = band(raw=None,
                            snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True,
                                               ENABLE_LINEAR_CLASSIFICATION=True,
                                               CLASS_PROB=0.0))
        assert result["n_bursts"] == 2
        assert all(row["is_burst_linear"] == "" for row in rows)
        assert all(math.isnan(p) for p in result["class_probs_linear_list"])

    def test_no_phase_at_all_saves_nothing_even_permissively(self, band):
        """The last row of the table: no verdict means no row, whatever
        ``SAVE_ONLY_BURST`` says."""
        for save_only_burst in (False, True):
            result, rows = band(snapshot=_snapshot(SAVE_ONLY_BURST=save_only_burst))
            assert result["cand_counter"] == 2
            assert rows == []


# --------------------------------------------------------------------------- #
# Phase 2
# --------------------------------------------------------------------------- #
class TestLinearValidation:
    def test_peaks_below_the_linear_threshold_are_rejected(self, band):
        result, rows = band(snapshot=_snapshot(ENABLE_LINEAR_VALIDATION=True,
                                               ENABLE_INTENSITY_CLASSIFICATION=True,
                                               SNR_THRESH_LINEAR=1.0e6))
        assert result["cand_counter"] == 0
        assert rows == []

    def test_peaks_above_it_go_on_to_classification(self, band):
        result, rows = band(snapshot=_snapshot(ENABLE_LINEAR_VALIDATION=True,
                                               ENABLE_INTENSITY_CLASSIFICATION=True,
                                               SNR_THRESH_LINEAR=1.0,
                                               CLASS_PROB=0.0))
        assert result["cand_counter"] == 2
        assert len(rows) == 2

    def test_a_linear_channel_without_the_burst_fails_validation(self, band):
        """The rejection has to come from the data, not only from an absurd
        threshold: with Q and U carrying no burst the Linear SNR at the peak is
        noise."""
        block = _intensity_chunk()
        result, _ = band(block=block, raw=_raw_chunk(block, linear_burst=False),
                         snapshot=_snapshot(ENABLE_LINEAR_VALIDATION=True,
                                            ENABLE_INTENSITY_CLASSIFICATION=True,
                                            SNR_THRESH_LINEAR=SNR_THRESH))
        assert result["cand_counter"] == 0

    def test_validation_is_skipped_when_the_file_has_one_polarisation(self, band):
        """No Linear data is not a rejection. Phase 2 stands down and every
        Intensity peak goes through."""
        result, _ = band(raw=None,
                         snapshot=_snapshot(ENABLE_LINEAR_VALIDATION=True,
                                            ENABLE_INTENSITY_CLASSIFICATION=True,
                                            SNR_THRESH_LINEAR=1.0e6))
        assert result["cand_counter"] == 2


# --------------------------------------------------------------------------- #
# the DM the candidate reports
# --------------------------------------------------------------------------- #
class TestDMMeasurement:
    def test_each_candidate_takes_its_dm_from_its_own_column(self, band):
        _, rows = band(snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True))
        # Global peak first: candidate 0 is the burst at sample 384.
        assert float(rows[0]["dm_pc_cm-3"]) == pytest.approx(DM_B, abs=DM_COLUMN_ABS)
        assert float(rows[1]["dm_pc_cm-3"]) == pytest.approx(DM_A, abs=DM_COLUMN_ABS)

    def test_a_measured_dm_says_so_and_carries_an_uncertainty(self, band):
        _, rows = band(snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True))
        assert rows[0]["dm_status"] == "measured"
        assert float(rows[0]["dm_uncertainty"]) == pytest.approx(0.5)

    def test_a_flat_cube_reports_an_unresolved_dm_not_a_number(self, band):
        """At these frequencies the sweep can be under one sample and every DM
        trial gives the same column. The default policy refuses to name a DM."""
        _, rows = band(cube=_dm_cube(flat=True),
                       snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True))
        assert rows[0]["dm_status"] == "unresolved_high_freq"
        assert math.isnan(float(rows[0]["dm_pc_cm-3"]))
        assert rows[0]["dm_uncertainty"] == ""

    def test_the_catalog_prior_policy_names_the_snapshot_midpoint(self, band, monkeypatch):
        monkeypatch.setattr(config, "HIGH_FREQ_DM_POLICY", "catalog_prior", raising=False)
        _, rows = band(cube=_dm_cube(flat=True),
                       snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True,
                                          DM_min=100.0, DM_max=300.0))
        assert rows[0]["dm_status"] == "catalog_prior"
        assert float(rows[0]["dm_pc_cm-3"]) == pytest.approx(200.0)
        assert float(rows[0]["dm_uncertainty"]) == pytest.approx(100.0)

    def test_the_measured_dm_still_comes_from_the_global_dm_range(self, band, monkeypatch):
        """A remaining REF-10 read, recorded rather than asserted to be right:
        ``_dm_from_image_at_time`` builds its grid from ``config.DM_min`` and
        ``config.DM_max``, while the flat-cube branch beside it uses the
        snapshot's. In production the two are the same object, so nothing shows
        it. This test fails the day that read moves -- update it then."""
        monkeypatch.setattr(config, "DM_max", 400.0, raising=False)
        _, rows = band(snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True,
                                          DM_max=DM_MAX))
        assert float(rows[0]["dm_pc_cm-3"]) == pytest.approx(2 * DM_B, abs=DM_COLUMN_ABS)


# --------------------------------------------------------------------------- #
# the row itself
# --------------------------------------------------------------------------- #
class TestCandidateRow:
    @pytest.fixture
    def first_row(self, band):
        _, rows = band(snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True))
        return rows[0]

    def test_it_identifies_where_the_candidate_came_from(self, first_row):
        assert first_row["file"] == "harness.fits"
        assert int(first_row["chunk_id"]) == CHUNK_IDX
        assert int(first_row["slice_id"]) == SLICE_IDX
        assert int(first_row["band_id"]) == 0

    def test_the_sample_index_is_the_peak_inside_the_slice(self, first_row):
        assert int(first_row["t_sample"]) == BURST_B

    def test_both_time_columns_are_written_and_they_are_different_measurements(self, first_row):
        """``t_sec_dm_time`` is where the DM-time plot puts the candidate,
        ``t_sec_waterfall`` where the waterfall SNR peaks. Collapsing them hides
        a disagreement that means something."""
        dm_time = float(first_row["t_sec_dm_time"])
        waterfall = float(first_row["t_sec_waterfall"])
        assert dm_time == pytest.approx(
            ABS_START + BURST_B * SLICE_LEN / (SLICE_LEN - 1) * TIME_RESO,
            abs=SEC_COLUMN_ABS,
        )
        assert waterfall == pytest.approx(ABS_START + BURST_B * TIME_RESO, abs=2e-3)

    def test_the_box_spans_the_whole_dm_axis(self, first_row):
        """A box drawn around a guessed DM makes the DM read back out of it a
        function of the guess. The box is the full height on purpose."""
        assert int(first_row["y1"]) == 0
        assert int(first_row["y2"]) == round((DM_ROWS - 1) * 512 / DM_ROWS)
        assert int(first_row["x2"]) - int(first_row["x1"]) == 2 * max(4, SLICE_LEN // 64)

    def test_the_topocentric_mjd_follows_tstart_and_the_detection_time(self, first_row):
        expected = TSTART_MJD + float(first_row["t_sec_dm_time"]) / 86400.0
        assert float(first_row["mjd_utc"]) == pytest.approx(expected, abs=1e-10)

    def test_absent_source_configuration_leaves_the_barycentric_columns_empty(self, first_row):
        """P1-07: without a source and a site there is no barycentric time, and
        the status column says so instead of a topocentric value in a
        barycentric column."""
        assert first_row["mjd_bary_status"] == "unavailable:no-source-config"
        for column in ("mjd_bary_utc", "mjd_bary_tdb",
                       "mjd_bary_utc_inf", "mjd_bary_tdb_inf"):
            assert first_row[column] == ""

    def test_a_configured_source_produces_real_barycentric_columns(self, band, monkeypatch):
        pytest.importorskip("astropy")
        from tests.observatory import effelsberg

        # A pinned EarthLocation rather than a site name: resolving the name is
        # a network call. See tests/observatory.py.
        for key, value in {"SOURCE_RA": "05:31:58.7", "SOURCE_DEC": "+33:08:52.5",
                           "REF_FREQ_MHZ": 1350.0, "OBSERVATORY": effelsberg(),
                           "EPHEMERIS": "builtin"}.items():
            monkeypatch.setattr(config, key, value, raising=False)
        _, rows = band(snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True))
        row = rows[0]
        assert row["mjd_bary_status"] == "ok"
        assert float(row["mjd_bary_utc"]) != float(row["mjd_utc"])
        # The infinite-frequency column is the barycentric one minus the
        # dispersion delay, so it must be earlier.
        assert float(row["mjd_bary_utc_inf"]) < float(row["mjd_bary_utc"])

    def test_the_two_snr_columns_report_before_and_after_dedispersion(self, first_row):
        assert float(first_row["snr_pre_dedisp"]) >= SNR_THRESH
        assert float(first_row["snr_post_dedisp"]) > 0.0
        assert first_row["snr_patch_dedispersed"] == first_row["snr_post_dedisp"]

    def test_the_linear_snr_columns_are_filled_when_the_file_is_multipol(self, band):
        _, rows = band(snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True))
        assert float(rows[0]["snr_waterfall_linear"]) > 0.0

    def test_they_are_empty_when_it_is_not(self, band):
        _, rows = band(raw=None, snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True))
        assert rows[0]["snr_waterfall_linear"] == ""
        assert rows[0]["snr_patch_dedispersed_linear"] == ""


# --------------------------------------------------------------------------- #
# slice geometry
# --------------------------------------------------------------------------- #
class TestSliceGeometry:
    """``slice_samples`` is the slice's real sample count and ``slice_len`` the
    nominal one. They differ for the last slice of a chunk, and scaling the band
    image's columns by the nominal length would place every candidate in that
    slice late by the difference.

    The one production caller passes the same number for both, so nothing there
    can tell whether ``slice_samples`` is used at all -- these tests can, and
    that is why they pass them apart.
    """

    def test_the_real_sample_count_places_the_candidate_not_the_nominal_one(self, band):
        _, rows = band(slice_len=SLICE_LEN, slice_samples=SLICE_LEN // 2,
                       snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True))
        assert int(rows[0]["t_sample"]) == int(BURST_B / (SLICE_LEN - 1) * (SLICE_LEN // 2))

    def test_without_a_real_count_it_falls_back_to_the_nominal_one(self, band):
        _, rows = band(slice_len=SLICE_LEN // 2, slice_samples=None,
                       snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True))
        assert int(rows[0]["t_sample"]) == int(BURST_B / (SLICE_LEN - 1) * (SLICE_LEN // 2))

    def test_the_box_width_follows_the_nominal_slice_length(self, band):
        _, rows = band(slice_len=SLICE_LEN // 2,
                       snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True))
        assert int(rows[0]["x2"]) - int(rows[0]["x1"]) == 2 * max(4, (SLICE_LEN // 2) // 64)


# --------------------------------------------------------------------------- #
# what the function hands back to its caller
# --------------------------------------------------------------------------- #
class TestReturnedResult:
    def test_the_multipol_waterfalls_are_returned_for_plotting(self, band):
        result, _ = band(snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True))
        assert result["waterfall_intensity"].shape == (SLICE_LEN, N_CHAN)
        assert result["waterfall_linear"].shape == (SLICE_LEN, N_CHAN)
        assert result["waterfall_circular"].shape == (SLICE_LEN, N_CHAN)

    def test_a_single_polarisation_file_returns_none_for_the_other_two(self, band):
        result, _ = band(raw=None, snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True))
        assert result["waterfall_intensity"].shape == (SLICE_LEN, N_CHAN)
        assert result["waterfall_linear"] is None
        assert result["waterfall_circular"] is None

    def test_every_per_candidate_list_has_one_entry_per_candidate(self, band):
        result, _ = band(snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True,
                                            ENABLE_LINEAR_CLASSIFICATION=True))
        n = result["cand_counter"]
        for key in ("top_conf", "top_boxes", "class_probs_list",
                    "class_probs_linear_list", "snr_waterfall_linear_list",
                    "snr_patch_linear_list", "snr_waterfall_intensity_list",
                    "snr_patch_intensity_list", "candidate_times_abs"):
            assert len(result[key]) == n, key

    def test_total_candidates_and_cand_counter_are_the_same_number(self, band):
        result, _ = band(snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True))
        assert result["total_candidates"] == result["cand_counter"]

    def test_the_best_candidate_is_the_first_burst_found(self, band, monkeypatch):
        """``first_*`` is what the composite plot is drawn from. A burst
        outranks a non-burst even when the non-burst was seen first."""
        model = _ScriptedModel([0.1, 0.9])  # candidate 0 no, candidate 1 burst
        _install_scripted(monkeypatch, model)
        result, _ = band(cls_model=model,
                         snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True,
                                            CLASS_PROB=0.5))
        assert result["best_is_burst"] is True
        assert result["first_start"] == pytest.approx(ABS_START + BURST_A * TIME_RESO)
        assert result["first_dm"] == pytest.approx(DM_A, abs=1e-6)

    def test_the_band_image_is_returned_as_the_tensor_the_network_expects(self, band):
        """``preprocess_img`` resizes the (dm_rows, slice_len) band image onto
        the 3x512x512 input the classifier was trained on; the box coordinates
        in the row are in that same frame."""
        result, _ = band(snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True))
        assert result["img_tensor"].shape == (3, 512, 512)

    def test_the_patch_path_is_named_after_the_slice_and_band(self, band):
        result, _ = band(band_idx=2, snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True))
        assert result["patch_path"].name == f"patch_slice{SLICE_IDX}_band2.png"


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
class TestPhaseMetrics:
    def test_each_phase_records_what_entered_and_what_passed(self, band):
        from src.output.phase_metrics import PhaseMetricsTracker

        tracker = PhaseMetricsTracker()
        result, _ = band(metrics_tracker=tracker,
                         snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True,
                                            CLASS_PROB=1.01))
        assert tracker.phase_1.detected == 2
        assert tracker.phase_2.detected == 2 and tracker.phase_2.failed == 0
        assert tracker.phase_3a_intensity.detected == 2
        assert tracker.phase_3a_intensity.no_burst == 2
        assert tracker.total_candidates == result["cand_counter"]
        assert tracker.total_no_burst == result["n_no_bursts"]

    def test_a_disabled_phase_records_nothing(self, band):
        from src.output.phase_metrics import PhaseMetricsTracker

        tracker = PhaseMetricsTracker()
        band(metrics_tracker=tracker,
             snapshot=_snapshot(ENABLE_INTENSITY_CLASSIFICATION=True))
        assert tracker.phase_3b_linear.detected == 0

    def test_a_rejected_slice_still_records_phase_two(self, band):
        from src.output.phase_metrics import PhaseMetricsTracker

        tracker = PhaseMetricsTracker()
        band(metrics_tracker=tracker,
             snapshot=_snapshot(ENABLE_LINEAR_VALIDATION=True,
                                SNR_THRESH_LINEAR=1.0e6))
        assert tracker.phase_1.detected == 2
        assert tracker.phase_2.failed == 2
