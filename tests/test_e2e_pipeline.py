"""End-to-end tests: a burst injected at a known DM and time must come back.

These run the real ``run_pipeline`` over a synthetic filterbank. The only thing
substituted is the detection network, and it is replaced by a detector that
genuinely locates the brightest point of the DM-time image it is handed and
reports it in the 512x512 frame CenterNet works in. Everything else -- header
parsing, streaming, chunk geometry, overlap trimming, downsampling, the DM-time
cube, the box-to-DM mapping, candidate finalisation and CSV persistence -- is
production code, with ground truth on both ends.
"""
from __future__ import annotations

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
from src.core.data_flow_manager import build_dm_time_cube, downsample_chunk
from src.core.pipeline_parameters import calculate_dm_height, calculate_dm_values
from src.detection.model_interface import CNN_IMG_SIZE
from src.input.filterbank_handler import get_obparams_fil, stream_fil
from tests.synthetic_filterbank import write_filterbank

DM_TRUE = 300.0
BURST_T = 1.5
TSAMP = 0.001

# A DM error of this size smears the burst by about one sample across the band
# (K_DM * dDM * (f_lo^-2 - f_hi^-2) with f = 1373..1500 MHz), and the injected
# pulse is five samples wide, so the recovered peak is expected within a few
# units of the truth rather than exactly on it.
DM_TOLERANCE = 10.0


def _configure(tmp_path: Path, *, dm_min=280.0, dm_max=320.0, chunk_samples=4000):
    # The DM window is deliberately narrow: the cube is the whole cost of these
    # tests and its height is the number of trials. It still brackets the truth
    # with room on both sides, and 41 rows keeps the cube shape well away from
    # the 512 of the CNN frame, which is what the mapping must not confuse.
    config.DATA_DIR = tmp_path / "raw"
    config.RESULTS_DIR = tmp_path / "out"
    config.FRB_TARGETS = ["synthetic"]
    config.DM_min, config.DM_max = dm_min, dm_max
    config.DM_GRID_MODE = "legacy_uniform"
    config.DOWN_TIME_RATE = 1
    config.DOWN_FREQ_RATE = 1
    config.SLICE_DURATION_MS = 1000.0
    config.MAX_CHUNK_SAMPLES = chunk_samples
    config.SAVE_ONLY_BURST = False
    config.FORCE_PLOTS = False
    config.USE_MULTI_BAND = False
    config.AUTO_HIGH_FREQ_PIPELINE = False
    config.DET_PROB = 0.1
    config.CLASS_PROB = 0.5
    config.SNR_THRESH = 5.0
    config.TRIAL_CORRECTION = "none"


def _install_peak_detector(monkeypatch):
    """Replace the network with a detector that finds the real peak.

    It reports the box in the CNN frame, exactly as centernet_utils does after
    scaling by input_shape, so the pipeline's box-to-DM mapping is exercised for
    real.
    """
    seen: dict = {}
    real_preprocess = de.preprocess_img

    def spy_preprocess(img):
        seen["band_img"] = np.asarray(img)
        return real_preprocess(img)

    def peak_detect(model, img_tensor):
        img = seen.get("band_img")
        if img is None or img.size == 0:
            return [], []
        row, col = np.unravel_index(int(np.argmax(img)), img.shape)
        y = row * CNN_IMG_SIZE / float(img.shape[0])
        x = col * CNN_IMG_SIZE / float(img.shape[1])
        half = 4.0
        return [0.95], [[x - half, y - half, x + half, y + half]]

    monkeypatch.setattr(de, "preprocess_img", spy_preprocess)
    monkeypatch.setattr(de, "detect", peak_detect)
    monkeypatch.setattr(
        de, "postprocess_img",
        lambda t: np.zeros((CNN_IMG_SIZE, CNN_IMG_SIZE, 3), dtype=np.uint8),
    )
    return seen


def _read_candidates(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in results_dir.rglob("*.candidates.csv"):
        with path.open(newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


class TestDedispersionRecoversTruth:
    """The DM-time cube alone must peak at the injected DM and time."""

    def test_cube_peaks_at_the_injected_dm(self, tmp_path):
        fil = tmp_path / "synthetic_single.fil"
        write_filterbank(fil, nsamples=8000, dm=DM_TRUE, burst_time_s=BURST_T, tsamp=TSAMP)
        get_obparams_fil(str(fil))
        config.DM_min, config.DM_max = 200.0, 400.0
        config.DM_GRID_MODE = "legacy_uniform"
        config.DOWN_TIME_RATE = 1
        config.DOWN_FREQ_RATE = 1

        block, _ = next(iter(stream_fil(str(fil), 8000, overlap_samples=0)))
        block_ds, _ = downsample_chunk(block)
        cube = build_dm_time_cube(
            block_ds, height=calculate_dm_height(),
            dm_min=config.DM_min, dm_max=config.DM_max,
        )

        plane = np.asarray(cube[0])
        dm_values = calculate_dm_values()
        row = int(np.argmax(plane.max(axis=1)))
        col = int(np.argmax(plane[row]))

        assert float(dm_values[row]) == pytest.approx(DM_TRUE, abs=DM_TOLERANCE)
        assert col * TSAMP == pytest.approx(BURST_T, abs=0.05)


class TestEndToEndPipeline:
    """run_pipeline over a synthetic file must put the burst in the CSV."""

    def _run(self, tmp_path, monkeypatch, *, nsamples, burst_time, chunk_samples,
             render_plots=False):
        from src.core import pipeline as pipeline_mod
        from src.core.pipeline import run_pipeline

        # The networks are the one part these tests deliberately do not exercise
        # -- detection is replaced below -- so the loaders are stubbed too. This
        # keeps the end-to-end run independent of torch being installed.
        monkeypatch.setattr(pipeline_mod, "_load_detection_model", lambda: None)
        monkeypatch.setattr(pipeline_mod, "_load_class_model", lambda: None)

        if not render_plots:
            # Rendering is ~95% of the wall time of a run (measured: ~1.8 s per
            # figure, four per slice) and is not what these tests assert.
            # test_plots_are_produced covers that path on the smallest input.
            monkeypatch.setattr(de, "save_all_plots", lambda *a, **k: None)

        _configure(tmp_path, chunk_samples=chunk_samples)
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        fil = config.DATA_DIR / "synthetic_burst.fil"
        truth = write_filterbank(
            fil, nsamples=nsamples, dm=DM_TRUE, burst_time_s=burst_time, tsamp=TSAMP,
        )
        _install_peak_detector(monkeypatch)
        run_pipeline()
        return truth, _read_candidates(config.RESULTS_DIR)

    def test_single_chunk_recovers_dm_and_time(self, tmp_path, monkeypatch):
        _, rows = self._run(tmp_path, monkeypatch, nsamples=4000,
                            burst_time=BURST_T, chunk_samples=8000)
        assert rows, "the pipeline produced no candidates at all"

        best = max(rows, key=lambda r: float(r["snr_patch_dedispersed"] or 0.0))
        assert float(best["dm_pc_cm-3"]) == pytest.approx(DM_TRUE, abs=DM_TOLERANCE)
        assert float(best["t_sec_dm_time"]) == pytest.approx(BURST_T, abs=0.1)

    def test_burst_in_a_later_chunk_keeps_its_absolute_time(self, tmp_path, monkeypatch):
        """Chunk geometry: a burst well past the first chunk must still be
        reported at its absolute time in the file, not relative to its chunk."""
        burst_at = 5.5
        _, rows = self._run(tmp_path, monkeypatch, nsamples=9000,
                            burst_time=burst_at, chunk_samples=3000)
        assert rows, "the pipeline produced no candidates at all"

        times = [float(r["t_sec_dm_time"]) for r in rows]
        assert min(abs(t - burst_at) for t in times) < 0.2, (
            f"no candidate near t={burst_at}s; got {sorted(times)[:10]}"
        )

    def test_every_counted_candidate_is_on_disk(self, tmp_path, monkeypatch):
        """The CSV must hold every row the run reported, with no duplicates from
        the chunk overlap being processed twice."""
        _, rows = self._run(tmp_path, monkeypatch, nsamples=9000,
                            burst_time=5.5, chunk_samples=3000)
        assert rows

        keys = [(r["chunk_id"], r["slice_id"], r["band_id"], r["t_sample"]) for r in rows]
        assert len(keys) == len(set(keys)), "duplicate candidates in the CSV"

    def test_plots_are_produced_for_a_candidate(self, tmp_path, monkeypatch):
        """The rendering path must run end to end too, not just the numbers."""
        _, rows = self._run(tmp_path, monkeypatch, nsamples=2000,
                            burst_time=1.0, chunk_samples=8000, render_plots=True)
        assert rows
        pngs = list(config.RESULTS_DIR.rglob("*.png"))
        assert pngs, "a candidate was reported but no figure was written"

    def test_results_are_reproducible(self, tmp_path, monkeypatch):
        """Two identical runs over the same input must agree."""
        _, rows_a = self._run(tmp_path / "a", monkeypatch, nsamples=4000,
                              burst_time=BURST_T, chunk_samples=8000)
        _, rows_b = self._run(tmp_path / "b", monkeypatch, nsamples=4000,
                              burst_time=BURST_T, chunk_samples=8000)
        assert len(rows_a) == len(rows_b)
        dm_a = sorted(float(r["dm_pc_cm-3"]) for r in rows_a)
        dm_b = sorted(float(r["dm_pc_cm-3"]) for r in rows_b)
        assert dm_a == pytest.approx(dm_b)
