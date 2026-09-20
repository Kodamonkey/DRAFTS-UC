"""The contracts ``run_pipeline`` and the low-frequency slice processor adopt.

REF-10's last stretch. Two properties matter here and neither is visible in
production, because in production the snapshot and the global always hold the
same values -- which is exactly why they need a test that pulls them apart.

**The run-scoped snapshot must not be reused per file.** ``run_pipeline`` builds
one snapshot before the loop, for the startup banner, and a *second* set of
contracts inside the loop, after ``_prepare_file_parameters`` has rewritten
config from each file's header. Collapsing the two would report the first file's
geometry for every file after it, which is the whole of SPEC-IO-001. A single
file cannot show the difference; two files of different lengths can.

**The slice processor must read what it is given.** It has always *accepted* a
configuration object and always been handed the mutable global, which is not the
same as being given a snapshot.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import config
from src.core.contracts import PipelineConfigSnapshot

TSAMP = 1.0e-3

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _source(relative: str) -> str:
    """Read a source file the way every platform must read it.

    ``Path.read_text()`` with no encoding uses the *locale* encoding, which is
    cp1252 on a Windows runner. ``src/core/pipeline.py`` contains ``pc cm⁻³``
    in a log line, and cp1252 cannot decode those bytes: the CI job for Windows
    failed with ``UnicodeDecodeError: 'charmap' codec can't decode byte 0x81``.
    The project is developed on Windows and the sources are UTF-8, so every
    read of one has to say so.

    The path is resolved against the repository rather than the working
    directory for the same class of reason: a relative path makes the test
    depend on where pytest was invoked from.
    """
    return (PROJECT_ROOT / relative).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# the per-file contracts
# --------------------------------------------------------------------------- #
class TestTheContractsAreBuiltPerFile:
    """Two filterbanks of different lengths in one run. Each file's logged
    geometry must be its own."""

    LENGTHS = (4000, 7000)

    @pytest.fixture
    def two_file_run(self, tmp_path, monkeypatch):
        pytest.importorskip("astropy")
        from src.core import detection_engine as de
        from src.core import pipeline as pipeline_mod
        from src.log_utils import logging_config as lc
        from tests.synthetic_filterbank import write_filterbank
        from tests.test_e2e_pipeline import _configure, _install_peak_detector

        _configure(tmp_path, chunk_samples=4000)
        monkeypatch.setattr(pipeline_mod, "_load_detection_model", lambda: None)
        monkeypatch.setattr(pipeline_mod, "_load_class_model", lambda: None)
        monkeypatch.setattr(de, "save_all_plots", lambda *a, **k: None)
        _install_peak_detector(monkeypatch)

        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        for nsamples in self.LENGTHS:
            write_filterbank(
                config.DATA_DIR / f"synthetic_{nsamples}.fil",
                nsamples=nsamples, dm=300.0, burst_time_s=1.0, tsamp=TSAMP,
            )

        seen: list[tuple[str, dict]] = []
        real_setup = lc.setup_logging

        def _capturing(*args, **kwargs):
            inst = real_setup(*args, **kwargs)
            real_file_start = inst.file_processing_start

            def _file_start(name, info, *a, **k):
                seen.append((name, dict(info)))
                return real_file_start(name, info, *a, **k)

            inst.file_processing_start = _file_start
            return inst

        monkeypatch.setattr(lc, "setup_logging", _capturing)

        from src.core.pipeline import run_pipeline
        run_pipeline()
        return {name: info for name, info in seen}

    def test_both_files_were_processed(self, two_file_run):
        assert len(two_file_run) == len(self.LENGTHS), two_file_run.keys()

    def test_each_file_reports_its_own_length(self, two_file_run):
        """The assertion that carries this module. A contract taken once per
        run, before the loop, would give both files the same ``samples``."""
        for nsamples in self.LENGTHS:
            info = two_file_run[f"synthetic_{nsamples}.fil"]
            assert info["samples"] == nsamples, (
                f"synthetic_{nsamples}.fil reported {info['samples']} samples; the "
                "observation contract was not rebuilt after this file's header"
            )

    def test_each_file_reports_its_own_duration(self, two_file_run):
        for nsamples in self.LENGTHS:
            info = two_file_run[f"synthetic_{nsamples}.fil"]
            assert info["duration_min"] == pytest.approx(nsamples * TSAMP / 60)

    def test_the_reported_lengths_actually_differ(self, two_file_run):
        """Guards the guard: if both files were written the same length, every
        assertion above would pass while proving nothing."""
        lengths = {info["samples"] for info in two_file_run.values()}
        assert len(lengths) == len(self.LENGTHS), lengths


# --------------------------------------------------------------------------- #
# the slice processor reads its snapshot
# --------------------------------------------------------------------------- #
class TestTheSliceProcessorReadsItsSnapshot:
    """``SAVE_ONLY_BURST`` decides the counts the slice processor reports. With
    the global saying one thing and the injected snapshot the other, only the
    snapshot may decide."""

    def _run_slice(self, tmp_path, monkeypatch, *, global_flag, snapshot_flag):
        """One slice carrying a single NO-BURST candidate.

        The two arms of the tail return different tuples only when a candidate
        exists and is not a burst: with the flag on the tail reports
        ``(n_bursts, n_bursts, 0, ...)`` and with it off
        ``(cand_counter, n_bursts, n_no_bursts, ...)``. On an empty slice both
        are ``(0, 0, 0, 0.0)`` and the test would prove nothing -- which is why
        a detection is stubbed in rather than left out.
        """
        from src.core import detection_engine as de
        from src.detection.model_interface import CNN_IMG_SIZE
        from src.output.candidate_manager import CandidateWriter, ensure_csv_header

        monkeypatch.setattr(config, "SAVE_ONLY_BURST", global_flag, raising=False)
        monkeypatch.setattr(config, "CLASS_PROB", 0.5, raising=False)
        monkeypatch.setattr(config, "SNR_THRESH", 5.0, raising=False)
        monkeypatch.setattr(config, "TIME_RESO", TSAMP, raising=False)
        monkeypatch.setattr(config, "DOWN_TIME_RATE", 1, raising=False)
        monkeypatch.setattr(config, "DM_min", 0.0, raising=False)
        monkeypatch.setattr(config, "DM_max", 100.0, raising=False)
        monkeypatch.setattr(config, "FORCE_PLOTS", False, raising=False)

        if snapshot_flag is None:
            snapshot = None
        else:
            base = PipelineConfigSnapshot.from_config(config)
            snapshot = PipelineConfigSnapshot(
                **{**base.__dict__, "save_only_burst": snapshot_flag}
            )

        # One box in the middle of the CNN frame. Flat data means the patch SNR
        # is nil, so the model-free classifier returns a low probability and the
        # candidate comes back NO-BURST -- which is what makes the two arms differ.
        half = CNN_IMG_SIZE / 2
        monkeypatch.setattr(
            de, "detect",
            lambda model, img: ([0.9], [(half - 8, half - 8, half + 8, half + 8)]),
        )
        monkeypatch.setattr(de, "save_all_plots", lambda *a, **k: None)

        csv_file = tmp_path / "cands.csv"
        ensure_csv_header(csv_file)
        dm_time = np.zeros((1, 16, 64), dtype=np.float32)
        block = np.zeros((64, 8), dtype=np.float32)
        try:
            return de.process_slice_with_multiple_bands(
                0, dm_time, block, 64, None, None, Path("x.fil"), tmp_path,
                np.linspace(1200.0, 1500.0, 8).astype(np.float32),
                csv_file, TSAMP, [(0, "", "Full Band")], [], config,
                absolute_start_time=0.0, force_plots=False,
                slice_start_idx=0, slice_end_idx=64,
                snapshot=snapshot,
            )
        finally:
            CandidateWriter.flush_all()

    def test_the_candidate_used_here_is_a_non_burst(self, tmp_path, monkeypatch):
        """Guards the guard. If the stubbed candidate came back a BURST, both
        arms would report the same numbers and the two tests below would pass
        without discriminating anything."""
        counts = self._run_slice(tmp_path, monkeypatch,
                                 global_flag=False, snapshot_flag=False)
        cand_counter, n_bursts, n_no_bursts, _ = counts
        assert (cand_counter, n_bursts, n_no_bursts) == (1, 0, 1), counts

    def test_the_snapshot_decides_and_the_global_does_not(self, tmp_path, monkeypatch):
        """Global says False, snapshot says True: the strict tail must win."""
        counts = self._run_slice(tmp_path, monkeypatch,
                                 global_flag=False, snapshot_flag=True)
        assert counts[:3] == (0, 0, 0), (
            "the slice processor reported the permissive counts, so it read "
            "config.SAVE_ONLY_BURST instead of the snapshot it was given"
        )

    def test_and_the_other_way_round(self, tmp_path, monkeypatch):
        """Global says True, snapshot says False: the permissive tail must win."""
        counts = self._run_slice(tmp_path, monkeypatch,
                                 global_flag=True, snapshot_flag=False)
        assert counts[:3] == (1, 0, 1), counts

    @pytest.mark.parametrize("global_flag", [True, False])
    def test_without_a_snapshot_it_falls_back_to_the_config_it_is_given(
        self, tmp_path, monkeypatch, global_flag
    ):
        """The parameter is optional so no caller had to change. Omitting it
        must behave exactly as before: build the snapshot from the ``config``
        argument, which is the global the caller passes."""
        counts = self._run_slice(tmp_path, monkeypatch,
                                 global_flag=global_flag, snapshot_flag=None)
        expected = (0, 0, 0) if global_flag else (1, 0, 1)
        assert counts[:3] == expected, counts

    def test_it_no_longer_reads_the_global_at_all(self):
        """Counted with the AST rather than by eye: the function went from six
        reads over four keys to none."""
        import ast

        tree = ast.parse(_source("src/core/detection_engine.py"))
        node = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and n.name == "process_slice_with_multiple_bands"
        )
        reads = [
            sub for sub in ast.walk(node)
            if isinstance(sub, ast.Attribute)
            and isinstance(sub.value, ast.Name)
            and sub.value.id == "config"
        ]
        assert reads == [], [f"config.{r.attr} at line {r.lineno}" for r in reads]


# --------------------------------------------------------------------------- #
# what run_pipeline still reads, and why that is the right answer
# --------------------------------------------------------------------------- #
class TestWhatRunPipelineStillReads:
    """The migration deliberately stopped short of the whole function.

    ``PipelineConfigSnapshot`` models the *search* configuration. Paths,
    targets, the log level, the thread count, the device and the decimation
    rates are process settings; putting them on it to make a count go down
    would be miscategorising them. This test records where the line was drawn so
    that moving it is a decision rather than a drift.
    """

    EXPECTED = {
        "AUTO_HIGH_FREQ_PIPELINE", "CPU_THREADS", "DATA_DIR", "DEVICE",
        "DOWN_FREQ_RATE", "DOWN_TIME_RATE", "FRB_TARGETS", "LOG_COLORS",
        "LOG_LEVEL", "POLARIZATION_MODE", "RESULTS_DIR", "SLICE_DURATION_MS",
        "TRIAL_CORRECTION", "USE_MULTI_BAND", "inject_config",
    }

    def _keys(self) -> set[str]:
        import ast

        tree = ast.parse(_source("src/core/pipeline.py"))
        node = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "run_pipeline"
        )
        keys = set()
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "config"):
                keys.add(sub.attr)
            if (isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id == "getattr"
                    and sub.args
                    and isinstance(sub.args[0], ast.Name)
                    and sub.args[0].id == "config"
                    and isinstance(sub.args[1], ast.Constant)):
                keys.add(sub.args[1].value)
        return keys

    def test_no_search_configuration_is_read_from_the_global_any_more(self):
        """The keys the snapshot covers must be gone. If one comes back, a
        banner can disagree with what the pipeline will actually do."""
        search_keys = {
            "DM_min", "DM_max", "DM_GRID_MODE", "SNR_THRESH", "CLASS_PROB",
            "SAVE_ONLY_BURST", "PREWHITEN_BEFORE_DM", "BOWTIE_COLLAPSE_RATIO",
            "SNR_THRESH_LINEAR", "CLASS_PROB_LINEAR", "ENABLE_LINEAR_VALIDATION",
            "ENABLE_INTENSITY_CLASSIFICATION", "ENABLE_LINEAR_CLASSIFICATION",
        }
        assert self._keys() & search_keys == set()

    def test_the_remaining_reads_are_the_ones_that_were_left_on_purpose(self):
        assert self._keys() == self.EXPECTED
