"""Regression tests for P1-02: which criterion reverses the channel axis.

Three PSRFITS reader paths disagreed. Two of them reversed when ``foff > 0``
(channels stored ascending); everything else in the repository reverses on
``config.DATA_NEEDS_REVERSAL``, which is set when the file stores channels
DESCENDING. Those are opposite conditions. A third path -- the primary astropy
SUBINT reader, taken whenever the `your` library is absent -- did not reverse
at all.

The convention, from ``src/input/utils.py::normalize_frequency_axis``:
``config.FREQ`` is ALWAYS ascending, and ``DATA_NEEDS_REVERSAL`` says the file's
channel axis runs the other way and must be flipped to match it. Dedispersion
indexes the data's channel axis against ascending ``config.FREQ``
(``src/preprocessing/dedispersion.py``), so that is the axis the data has to be
in.

The first test reproduces the measurement that settled it. The failure mode is
the dangerous kind: not a missed burst, but a burst recovered at a fabricated DM
that still clears the detection threshold.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from src.analysis.science_metrics import K_DM_MS
from src.input.utils import normalize_frequency_axis

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FITS_HANDLER = PROJECT_ROOT / "src" / "input" / "fits_handler.py"

N_CHAN = 64
F_HIGH = 1500.0
F_LOW = 1200.0
TSAMP = 1.0e-4          # 0.1 ms
N_SAMP = 4000
TRUE_DM = 500.0


def _descending_freqs() -> np.ndarray:
    """A normal PSRFITS axis: DAT_FREQ[0] is the top of the band."""
    return np.linspace(F_HIGH, F_LOW, N_CHAN)


def _inject_dispersed_burst(freqs: np.ndarray, dm: float) -> np.ndarray:
    """(nsamp, nchan) with one narrow pulse swept by *dm*, in *freqs* order."""
    data = np.zeros((N_SAMP, freqs.size), dtype=np.float32)
    t0 = 0.05  # seconds at the reference (highest) frequency
    f_ref = freqs.max()
    for ch, f in enumerate(freqs):
        delay = K_DM_MS * dm * (f ** -2 - f_ref ** -2)
        idx = int(round((t0 + delay) / TSAMP))
        if 0 <= idx < N_SAMP:
            data[idx, ch] = 1.0
    return data


def _best_dm(data_ascending: np.ndarray, freq_ascending: np.ndarray,
             dm_grid: np.ndarray) -> tuple[float, float]:
    """Brute-force dedisperse and return (peak DM, peak-to-noise ratio).

    Deliberately independent of the pipeline's own kernels: the point is the
    channel ordering, not the dedisperser.
    """
    f_ref = freq_ascending.max()
    peaks = np.empty(dm_grid.size)
    for k, dm in enumerate(dm_grid):
        delays = K_DM_MS * dm * (freq_ascending ** -2 - f_ref ** -2)
        shifts = np.round(delays / TSAMP).astype(int)
        acc = np.zeros(N_SAMP, dtype=np.float64)
        for ch, shift in enumerate(shifts):
            acc += np.roll(data_ascending[:, ch], -shift)
        peaks[k] = acc.max()
    best = int(np.argmax(peaks))
    median = float(np.median(peaks))
    spread = float(np.std(peaks)) or 1.0
    return float(dm_grid[best]), (peaks[best] - median) / spread


class TestTheReversalCriterionRecoversTheInjectedDM:
    def test_descending_file_recovers_the_true_dm_only_when_reversed(self):
        """The measurement that settled P1-02.

        A burst injected at DM 500 into a descending-axis file comes back at
        exactly DM 500 under the DATA_NEEDS_REVERSAL rule. Under the `foff > 0`
        rule the same data yields a different, wrong DM.
        """
        file_freqs = _descending_freqs()
        raw = _inject_dispersed_burst(file_freqs, TRUE_DM)

        freq_ascending, needs_reversal = normalize_frequency_axis(file_freqs)
        assert needs_reversal is True, "a descending DAT_FREQ must set the flag"

        foff = float(file_freqs[1] - file_freqs[0])
        assert foff < 0

        dm_grid = np.arange(400.0, 601.0, 1.0)

        # The rule the readers now follow.
        correct = raw[:, ::-1] if needs_reversal else raw
        dm_correct, snr_correct = _best_dm(correct, freq_ascending, dm_grid)

        # The rule two branches used to follow: reverse when foff > 0.
        wrong = raw[:, ::-1] if foff > 0 else raw
        dm_wrong, snr_wrong = _best_dm(wrong, freq_ascending, dm_grid)

        assert dm_correct == pytest.approx(TRUE_DM, abs=1.0)
        assert dm_wrong != pytest.approx(TRUE_DM, abs=5.0)
        assert snr_correct > snr_wrong

    def test_ascending_file_needs_no_reversal(self):
        """The mirror case, so the test cannot pass by always reversing."""
        file_freqs = _descending_freqs()[::-1]
        raw = _inject_dispersed_burst(file_freqs, TRUE_DM)

        freq_ascending, needs_reversal = normalize_frequency_axis(file_freqs)
        assert needs_reversal is False

        dm_grid = np.arange(400.0, 601.0, 1.0)
        dm_correct, _ = _best_dm(raw, freq_ascending, dm_grid)
        dm_wrong, _ = _best_dm(raw[:, ::-1], freq_ascending, dm_grid)

        assert dm_correct == pytest.approx(TRUE_DM, abs=1.0)
        assert dm_wrong != pytest.approx(TRUE_DM, abs=5.0)


class TestEveryReaderPathUsesTheSameCriterion:
    def test_no_reversal_is_guarded_by_foff(self):
        """Two branches reversed on `foff > 0`; that is the inverted condition."""
        tree = ast.parse(FITS_HANDLER.read_text(encoding="utf-8"))
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            test_src = ast.unparse(node.test)
            if "foff" not in test_src:
                continue
            body_src = "\n".join(ast.unparse(stmt) for stmt in node.body)
            if "::-1" in body_src or "[:, :, ::-1]" in body_src:
                offenders.append(f"line {node.lineno}: if {test_src}")
        assert not offenders, (
            "channel reversal guarded by foff again: " + "; ".join(offenders)
        )

    def test_the_your_library_is_not_assumed_to_normalise_channel_order(self):
        """It does not: need_flipband is hardcoded False and the flip is
        commented out in your/formats/psrfits.py. If a future version starts
        normalising, this assumption has to be revisited -- hence the test."""
        psrfits = pytest.importorskip("your.formats.psrfits")
        source = Path(psrfits.__file__).read_text(encoding="utf-8")
        assert "self.need_flipband = False" in source, (
            "the `your` version installed may now flip the band itself; "
            "re-check the reversal criterion in fits_handler"
        )

    def test_the_primary_astropy_path_reverses_too(self):
        """It did not, while the duplicated copy of it below did.

        The duplicate is reached only through `except Exception`, so any install
        without `your` streamed descending PSRFITS with the channel axis
        untouched -- the same defect, third instance.
        """
        # Scoped to the whole module, not to stream_fits. REF-03 moved the
        # emission out into _emit_subint_*_blocks, and while this test still
        # walked the stream_fits FunctionDef both counts were zero, so it passed
        # while checking nothing -- which is worse than not having it.
        tree = ast.parse(FITS_HANDLER.read_text(encoding="utf-8"))

        # The astropy SUBINT readers -- primary and duplicated fallback -- each
        # build block_out from out_buf.
        built = [
            node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "block_out" for t in node.targets)
            and "out_buf" in ast.unparse(node.value)
        ]
        reversed_ = [
            node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and "DATA_NEEDS_REVERSAL" in ast.unparse(node.test)
            and "block_out[:, :, ::-1]" in "\n".join(ast.unparse(s) for s in node.body)
        ]
        assert built, "no block_out emission site found; the readers moved again"
        assert len(reversed_) == len(built), (
            f"{len(built)} block_out emission sites (lines {sorted(built)}) but "
            f"{len(reversed_)} reversals (lines {sorted(reversed_)}): "
            "an astropy emission path is missing the flip"
        )

    def test_the_your_branch_follows_the_shared_criterion(self):
        source = FITS_HANDLER.read_text(encoding="utf-8")
        # Single-polarisation branch: reverses on the flag, not on foff.
        assert "if _rest_says_reverse:" in source
        assert "_rest_says_reverse = bool(getattr(config, 'DATA_NEEDS_REVERSAL', False))" in source
        # Multi-polarisation branch.
        assert "if getattr(config, 'DATA_NEEDS_REVERSAL', False):\n                            arr_raw = arr_raw[:, :, ::-1]" in source

    def test_the_non_subint_loader_reverses_before_returning(self):
        """That emission site has no flip of its own; its loader does it."""
        tree = ast.parse(FITS_HANDLER.read_text(encoding="utf-8"))
        loader = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_load_fits_non_subint"
        )
        body = ast.unparse(loader)
        assert "DATA_NEEDS_REVERSAL" in body and "::-1" in body
