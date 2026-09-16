"""Regression tests for the Phase 2 reliability defects from the 2026-09 audit.

P1-13  Re-running over the same input appended to the previous run's CSV, so one
       file held two runs' detections with nothing to tell them apart.
P1-14  Every output path came from the file stem alone, so two inputs with the
       same basename in different directories merged into one dataset.
P1-15  The five per-file error handlers reported n_candidates: 0 even when rows
       had already been written, leading straight to discarding real science.
P1-16  The checkpoint recorded no schema, no configuration and no input identity,
       so resuming after a parameter change spliced two searches into one CSV.
P1-17  The HF pipeline had no checkpoint at all: an interrupted run restarted
       from zero and, the CSV being append-mode, duplicated everything.
P1-18  Memmap-backed DM cubes left their multi-GB temporary files behind.
P1-19  No operation was ever retried, so a transient network error lost a chunk.
P2-01  A truncated .fil header span forever instead of failing.
P2-02  An unvalidated length prefix read the whole file into memory.
Plus: a corrupt SIGPROC header used to fall through to invented observation
parameters, which is worse than refusing outright.
"""
from __future__ import annotations

import ast
import csv
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.core.checkpoint import (
    compute_run_fingerprint,
    load_checkpoint,
    save_checkpoint,
)
from src.core.data_flow_manager import (
    _LIVE_DM_CUBE_FILES,
    _allocate_dm_cube_buffer,
    release_dm_cube_buffer,
)
from src.core.retry import with_retry
from src.input.filterbank_handler import _read_header
from src.output.candidate_manager import (
    CANDIDATE_HEADER,
    CandidateWriter,
    ensure_csv_header,
    rotate_previous_candidates,
)
from tests.synthetic_filterbank import write_filterbank


def _pack_string(value: str) -> bytes:
    raw = value.encode("ascii")
    return struct.pack("<i", len(raw)) + raw


# --------------------------------------------------------------------------- #
# P2-01 / P2-02 and the invented-parameters fallback
# --------------------------------------------------------------------------- #

class TestMalformedFilterbankHeaders:
    def test_truncated_header_terminates(self, tmp_path):
        """The property is termination, not the exception type.

        The original `continue` never advanced the file position, so at EOF the
        loop span forever: a core at 100% and, at DEBUG, a log growing until the
        disk filled. Asserting "it raises" would not catch a return to that,
        because the field cap added alongside would still stop it eventually --
        so this measures that the call actually returns.
        """
        import threading

        path = tmp_path / "truncated.fil"
        path.write_bytes(
            _pack_string("HEADER_START")
            + _pack_string("nchans") + struct.pack("<i", 128)
            + _pack_string("tsa")  # cut off mid-field
        )

        finished = threading.Event()

        def parse():
            try:
                with path.open("rb") as fh:
                    _read_header(fh)
            except Exception:
                pass
            finally:
                finished.set()

        worker = threading.Thread(target=parse, daemon=True)
        worker.start()
        assert finished.wait(timeout=10), (
            "parsing a truncated header did not return: it is spinning"
        )

    def test_negative_length_prefix_is_rejected_before_reading(self, tmp_path):
        """The property is a bounded read.

        f.read(-1) consumes the rest of the file; these are terabyte files, so
        the length has to be rejected before the read, not after.
        """
        from src.input.filterbank_handler import _read_string

        path = tmp_path / "negative.fil"
        body = b"X" * 100_000
        path.write_bytes(struct.pack("<i", -1) + body)

        with path.open("rb") as fh:
            with pytest.raises(ValueError, match="implausible SIGPROC string length"):
                _read_string(fh)
            # Nothing beyond the 4-byte prefix may have been consumed.
            assert fh.tell() == 4, "the payload was read before the length was checked"

    def test_absurd_length_prefix_is_rejected_before_reading(self, tmp_path):
        from src.input.filterbank_handler import _read_string

        path = tmp_path / "huge.fil"
        path.write_bytes(struct.pack("<i", 2**30) + b"X" * 1000)
        with path.open("rb") as fh:
            with pytest.raises(ValueError, match="implausible SIGPROC string length"):
                _read_string(fh)
            assert fh.tell() == 4

    def test_a_plausible_length_still_reads(self, tmp_path):
        from src.input.filterbank_handler import _read_string

        path = tmp_path / "ok.fil"
        path.write_bytes(_pack_string("tsamp"))
        with path.open("rb") as fh:
            assert _read_string(fh) == "tsamp"

    def test_corrupt_sigproc_is_refused_not_guessed(self, tmp_path):
        """A file that declared itself SIGPROC and then failed to parse must not
        be processed against invented nchans/tsamp/fch1/foff: every DM and time
        derived from it would be meaningless, with nothing signalling it."""
        path = tmp_path / "corrupt.fil"
        path.write_bytes(_pack_string("HEADER_START") + struct.pack("<i", -1) + b"X" * 500)
        with path.open("rb") as fh:
            with pytest.raises(Exception):
                _read_header(fh)

    def test_non_sigproc_file_still_falls_back(self, tmp_path):
        """Files that never claimed to be SIGPROC keep the estimated-parameter
        path: that capability is intentional, only the silent part was not."""
        path = tmp_path / "raw.fil"
        path.write_bytes(b"\x00" * 8192)
        with path.open("rb") as fh:
            header, offset = _read_header(fh)
        assert header["nchans"] == 512
        assert offset == 512

    def test_a_valid_header_still_reads(self, tmp_path):
        path = tmp_path / "good.fil"
        write_filterbank(path, nsamples=100, nchans=64, tsamp=0.002, fch1=1400.0, foff=-0.5)
        with path.open("rb") as fh:
            header, _ = _read_header(fh)
        assert header["nchans"] == 64
        assert header["tsamp"] == pytest.approx(0.002)
        assert header["foff"] == pytest.approx(-0.5)


# --------------------------------------------------------------------------- #
# P1-18
# --------------------------------------------------------------------------- #

class TestDmCubeBufferIsReleased:
    def test_memmap_backing_file_is_deleted(self, monkeypatch):
        monkeypatch.setattr(config, "DM_CUBE_MEMMAP_THRESHOLD_GB", 0.0, raising=False)
        arr = _allocate_dm_cube_buffer((3, 4, 8), size_gb=1.0)
        path = Path(arr._mmap_path)
        assert path.exists()
        assert str(path) in _LIVE_DM_CUBE_FILES

        release_dm_cube_buffer(arr)

        assert not path.exists(), "the temporary DM cube survived; a long run fills the disk"
        assert str(path) not in _LIVE_DM_CUBE_FILES

    def test_release_is_a_noop_for_in_memory_cubes(self, monkeypatch):
        monkeypatch.setattr(config, "DM_CUBE_MEMMAP_THRESHOLD_GB", 999.0, raising=False)
        arr = _allocate_dm_cube_buffer((3, 4, 8), size_gb=0.001)
        assert not hasattr(arr, "_mmap_path")
        release_dm_cube_buffer(arr)  # must not raise

    def test_both_pipelines_release_the_cube(self):
        for name in ("pipeline.py", "high_freq_pipeline.py"):
            src = (PROJECT_ROOT / "src/core" / name).read_text(encoding="utf-8")
            assert "release_dm_cube_buffer(dm_time_full)" in src, (
                f"{name} drops the full cube without releasing its backing file"
            )


# --------------------------------------------------------------------------- #
# P1-16 / P1-17
# --------------------------------------------------------------------------- #

class TestCheckpointValidation:
    def _fingerprint(self, path):
        return compute_run_fingerprint(path, config)

    def test_checkpoint_from_another_configuration_is_refused(self, tmp_path):
        data = tmp_path / "obs.fil"
        write_filterbank(data, nsamples=100)
        config.DM_max = 512.0
        fingerprint_a = self._fingerprint(data)
        save_checkpoint(tmp_path, "obs", 5, 10, fingerprint=fingerprint_a)
        assert load_checkpoint(tmp_path, "obs", fingerprint_a) == 5

        config.DM_max = 1024.0
        fingerprint_b = self._fingerprint(data)
        assert fingerprint_b != fingerprint_a
        assert load_checkpoint(tmp_path, "obs", fingerprint_b) == -1, (
            "a checkpoint from a different DM range was accepted; the CSV would "
            "hold two different searches with no record of the seam"
        )

    def test_checkpoint_from_another_input_file_is_refused(self, tmp_path):
        first = tmp_path / "a.fil"
        second = tmp_path / "b.fil"
        write_filterbank(first, nsamples=100)
        write_filterbank(second, nsamples=200)

        fingerprint_first = self._fingerprint(first)
        save_checkpoint(tmp_path, "shared", 3, 10, fingerprint=fingerprint_first)
        assert load_checkpoint(tmp_path, "shared", self._fingerprint(second)) == -1

    def test_older_schema_is_discarded(self, tmp_path):
        cp_dir = tmp_path / "obs"
        cp_dir.mkdir()
        (cp_dir / ".drafts_checkpoint.json").write_text(
            json.dumps({"file_stem": "obs", "last_completed_chunk": 7}), encoding="utf-8"
        )
        assert load_checkpoint(tmp_path, "obs") == -1

    def test_fingerprint_is_stable_for_an_unchanged_run(self, tmp_path):
        data = tmp_path / "obs.fil"
        write_filterbank(data, nsamples=100)
        assert self._fingerprint(data) == self._fingerprint(data)

    def test_high_freq_pipeline_checkpoints(self):
        """The HF pipeline had no checkpoint at all, so an interrupted run
        restarted from zero and duplicated every candidate already written."""
        src = (PROJECT_ROOT / "src/core/high_freq_pipeline.py").read_text(encoding="utf-8")
        for expected in (
            "load_checkpoint(save_dir, fits_path.stem, run_fingerprint)",
            "should_skip_chunk(chunk_seq, resume_after)",
            "save_checkpoint(",
            "clear_checkpoint(save_dir, fits_path.stem)",
        ):
            assert expected in src, f"HF pipeline is missing: {expected}"

    def test_high_freq_only_checkpoints_successful_chunks(self):
        src = (PROJECT_ROOT / "src/core/high_freq_pipeline.py").read_text(encoding="utf-8")
        assert "chunk_succeeded = False" in src and "chunk_succeeded = True" in src
        assert "if chunk_succeeded:" in src


# --------------------------------------------------------------------------- #
# P1-13
# --------------------------------------------------------------------------- #

class TestRerunDoesNotAppendToThePreviousRun:
    def test_previous_candidates_are_rotated_aside(self, tmp_path):
        csv_file = tmp_path / "obs.candidates.csv"
        ensure_csv_header(csv_file)
        writer = CandidateWriter.get(csv_file)
        for _ in range(2):
            writer.write(["x"] * len(CANDIDATE_HEADER))
        CandidateWriter.flush_all()

        moved = rotate_previous_candidates(csv_file)

        assert moved is not None and moved.exists()
        with moved.open(newline="") as fh:
            assert len(list(csv.reader(fh))) == 3  # header + 2 rows preserved
        with csv_file.open(newline="") as fh:
            assert len(list(csv.reader(fh))) == 1  # fresh file, header only

    def test_header_only_file_is_left_alone(self, tmp_path):
        csv_file = tmp_path / "obs.candidates.csv"
        ensure_csv_header(csv_file)
        assert rotate_previous_candidates(csv_file) is None
        assert csv_file.exists()

    def test_missing_file_is_left_alone(self, tmp_path):
        assert rotate_previous_candidates(tmp_path / "nope.csv") is None

    def test_pipelines_rotate_only_when_not_resuming(self):
        """Resuming must keep appending; only a fresh run starts a new CSV."""
        for name in ("pipeline.py", "high_freq_pipeline.py"):
            src = (PROJECT_ROOT / "src/core" / name).read_text(encoding="utf-8")
            index = src.index("rotate_previous_candidates(csv_file)")
            preceding = src[max(0, index - 300):index]
            assert "if resume_after < 0:" in preceding, (
                f"{name} rotates the CSV without checking for a resume"
            )


# --------------------------------------------------------------------------- #
# P1-14
# --------------------------------------------------------------------------- #

class TestCollidingFileStemsAreRefused:
    def test_two_inputs_with_the_same_basename_stop_the_run(self, tmp_path, monkeypatch):
        from src.core import pipeline as pipeline_mod

        first = tmp_path / "night1" / "scan01.fil"
        second = tmp_path / "night2" / "scan01.fil"
        for path in (first, second):
            write_filterbank(path, nsamples=100)

        monkeypatch.setattr(pipeline_mod, "find_data_files", lambda target: [first, second])
        monkeypatch.setattr(config, "FRB_TARGETS", ["scan"], raising=False)
        monkeypatch.setattr(pipeline_mod, "_load_detection_model", lambda: None)
        monkeypatch.setattr(pipeline_mod, "_load_class_model", lambda: None)

        with pytest.raises(ValueError, match="share the basename"):
            pipeline_mod.run_pipeline()

    def test_distinct_basenames_are_accepted(self, tmp_path, monkeypatch):
        from src.core import pipeline as pipeline_mod

        first = tmp_path / "night1" / "scan01.fil"
        second = tmp_path / "night2" / "scan02.fil"
        for path in (first, second):
            write_filterbank(path, nsamples=100)

        seen: list[Path] = []
        monkeypatch.setattr(pipeline_mod, "find_data_files", lambda target: [first, second])
        monkeypatch.setattr(config, "FRB_TARGETS", ["scan"], raising=False)
        monkeypatch.setattr(pipeline_mod, "_load_detection_model", lambda: None)
        monkeypatch.setattr(pipeline_mod, "_load_class_model", lambda: None)

        def stop_after_discovery(path, override):
            seen.append(path)
            raise RuntimeError("stop here")

        monkeypatch.setattr(pipeline_mod, "_prepare_file_parameters", stop_after_discovery)
        config.RESULTS_DIR = tmp_path / "out"
        pipeline_mod.run_pipeline()
        assert seen, "discovery refused a set of files with distinct basenames"


# --------------------------------------------------------------------------- #
# P1-15
# --------------------------------------------------------------------------- #

class TestErrorResultsReportWhatWasWritten:
    def test_error_result_carries_the_real_counts(self):
        from src.core.pipeline import DetectionStats, _error_result

        stats = DetectionStats()
        stats.update(candidates=7, bursts=3, no_bursts=4, prob_max=0.8)
        stats.snr_values.extend([5.0, 7.0])
        config.SAVE_ONLY_BURST = False

        result = _error_result("ERROR_MEMORY", OSError("disk"), 0.0, stats,
                               chunks_processed=9, failed_chunks=1)

        assert result["n_candidates"] == 7, (
            "reporting zero while the CSV holds rows leads to discarding them"
        )
        assert result["n_bursts"] == 3
        assert result["max_prob"] == pytest.approx(0.8)
        assert result["mean_snr"] == pytest.approx(6.0)
        assert result["status"] == "ERROR_MEMORY"
        assert result["chunks_processed"] == 9 and result["failed_chunks"] == 1

    def test_no_handler_reports_hardcoded_zeros(self):
        src = (PROJECT_ROOT / "src/core/pipeline.py").read_text(encoding="utf-8")
        assert '"n_candidates": 0, "n_bursts": 0, "n_no_bursts": 0,' not in src

    def test_save_only_burst_is_respected(self):
        from src.core.pipeline import DetectionStats, _error_result

        stats = DetectionStats()
        stats.update(candidates=7, bursts=3, no_bursts=4, prob_max=0.8)
        config.SAVE_ONLY_BURST = True
        result = _error_result("ERROR_CHUNKED", OSError("x"), 0.0, stats)
        assert (result["n_candidates"], result["n_bursts"], result["n_no_bursts"]) == (3, 3, 0)
        config.SAVE_ONLY_BURST = False


# --------------------------------------------------------------------------- #
# P1-19
# --------------------------------------------------------------------------- #

class TestTransientFailuresAreRetried:
    def test_a_transient_error_is_retried_and_succeeds(self):
        attempts = {"n": 0}

        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise OSError("network hiccup")
            return "ok"

        assert with_retry(flaky, description="test", base_delay=0.001) == "ok"
        assert attempts["n"] == 3

    def test_a_persistent_error_still_propagates(self):
        def always_fails():
            raise OSError("gone")

        with pytest.raises(OSError):
            with_retry(always_fails, description="test", attempts=2, base_delay=0.001)

    def test_programming_errors_are_not_retried(self):
        attempts = {"n": 0}

        def bug():
            attempts["n"] += 1
            raise ValueError("this is a bug, repeating it helps nobody")

        with pytest.raises(ValueError):
            with_retry(bug, description="test", base_delay=0.001)
        assert attempts["n"] == 1

    def test_candidate_flush_survives_a_transient_error(self, tmp_path, monkeypatch):
        csv_file = tmp_path / "obs.csv"
        ensure_csv_header(csv_file)
        writer = CandidateWriter.get(csv_file)
        writer.write(["x"] * len(CANDIDATE_HEADER))

        calls = {"n": 0}
        real_write = writer._write_buffer

        def flaky_write():
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("share went away")
            real_write()

        monkeypatch.setattr(writer, "_write_buffer", flaky_write)
        monkeypatch.setattr("src.core.retry.time.sleep", lambda s: None)
        writer.flush()

        with csv_file.open(newline="") as fh:
            assert len(list(csv.reader(fh))) == 2, "the row was lost to a transient error"
        CandidateWriter.flush_all()
