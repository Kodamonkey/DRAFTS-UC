"""Regression tests for the four P0 defects found in the 2026-09 audit.

Each defect survived because no test exercised the production path that carried
it. These tests drive the real code: the only things faked are the neural
network models (which are not what broke) and, where the logic lives inside a
200-line loop that cannot be instantiated in a unit test, the structure of the
call site is asserted against the AST.

P0-1  CenterNet boxes were mapped using the DM-cube dimensions instead of the
      512x512 frame the boxes are actually expressed in, so every LF candidate
      got a wrong DM and a wrong arrival time.
P0-2  The HF pipeline returned before the buffered CSV rows were flushed, so up
      to 49 candidates per file (usually all of them) were silently dropped.
P0-3  The resume predicate skipped one chunk past the last completed one.
P0-4  The checkpoint was written even for a chunk that had raised, so a resume
      skipped the failed chunk for good.
"""
from __future__ import annotations

import ast
import csv
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.core import detection_engine as de
from src.core.checkpoint import (
    clear_checkpoint,
    load_checkpoint,
    save_checkpoint,
    should_skip_chunk,
)
from src.detection.model_interface import CNN_IMG_SIZE
from src.output.candidate_manager import (
    CANDIDATE_HEADER,
    CandidateWriter,
    ensure_csv_header,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _function_ast(module_path: Path, func_name: str) -> ast.FunctionDef:
    """Return the AST of *func_name* defined at module level in *module_path*."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node
    raise AssertionError(f"{func_name} not found in {module_path.name}")


def _calls_named(node: ast.AST, dotted: str) -> list[ast.Call]:
    """Every Call node under *node* whose callee renders as *dotted*."""
    found = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            try:
                if ast.unparse(sub.func) == dotted:
                    found.append(sub)
            except Exception:  # pragma: no cover - unparse is available on 3.9+
                continue
    return found


# --------------------------------------------------------------------------- #
# P0-1
# --------------------------------------------------------------------------- #

class TestCenterNetBoxFrame:
    """Boxes live in the CNN frame; the DM they map to must not depend on the
    shape of the cube they were detected in."""

    BOX = [10.0, 256.0, 40.0, 300.0]  # x1, y1, x2, y2 in the 512x512 CNN frame
    SLICE_LEN = 512

    def _run_band(self, tmp_path, monkeypatch, cube_height, cube_width=None,
                  box=None, slice_len=None):
        """Drive the real detection path with a fake detector.

        Returns ``(rows, captured_kwargs)`` where *rows* are the candidate rows
        actually written to the CSV.
        """
        cube_width = cube_width or self.SLICE_LEN
        slice_len = slice_len or self.SLICE_LEN
        box = box or self.BOX

        config.DM_min, config.DM_max = 480.0, 640.0
        config.TIME_RESO, config.DOWN_TIME_RATE = 0.000256, 4
        config.DOWN_FREQ_RATE = 1
        config.FREQ_RESO = 32
        config.FREQ = np.linspace(1200.0, 1500.0, 32)
        config.CLASS_PROB = 0.5
        config.SAVE_ONLY_BURST = False
        config.SNR_THRESH = 5.0
        config.TRIAL_CORRECTION = "none"

        rng = np.random.RandomState(0)
        band_img = rng.rand(cube_height, cube_width).astype(np.float32)
        data = rng.rand(slice_len, 32).astype(np.float32)
        freq_down = np.linspace(1200.0, 1500.0, 32)

        captured: dict = {}
        real_extract = de.extract_candidate_dm

        def spy_extract(px, py, sl, **kw):
            captured.update(kw)
            return real_extract(px, py, sl, **kw)

        # Fake only the network boundary: the mapping under test runs for real.
        monkeypatch.setattr(de, "extract_candidate_dm", spy_extract)
        monkeypatch.setattr(de, "detect", lambda model, t: ([0.9], [list(box)]))
        monkeypatch.setattr(
            de, "preprocess_img",
            lambda img: np.zeros((3, CNN_IMG_SIZE, CNN_IMG_SIZE), dtype=np.float32),
        )
        monkeypatch.setattr(
            de, "postprocess_img",
            lambda t: np.zeros((CNN_IMG_SIZE, CNN_IMG_SIZE, 3), dtype=np.uint8),
        )

        csv_file = tmp_path / f"cand_{cube_height}.csv"
        ensure_csv_header(csv_file)
        de.detect_and_classify_candidates_in_band(
            det_model=None,
            cls_model=None,
            band_img=band_img,
            slice_len=slice_len,
            j=0,
            fits_path=Path("synthetic.fits"),
            save_dir=tmp_path,
            data=data,
            freq_down=freq_down,
            csv_file=csv_file,
            time_reso_ds=config.TIME_RESO * config.DOWN_TIME_RATE,
            snr_list=[],
            config=config,
            band_idx=0,
            chunk_idx=0,
            slice_start_idx=0,
            patches_dir=tmp_path / "patches",
        )
        CandidateWriter.flush_all()

        with csv_file.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        return rows, captured

    def test_extract_is_given_the_cnn_frame_not_the_cube_shape(self, tmp_path, monkeypatch):
        """The regression itself: a cube of 147 rows must not be reported as the
        frame the box coordinates live in."""
        _, captured = self._run_band(tmp_path, monkeypatch, cube_height=147)
        assert captured["img_height"] == CNN_IMG_SIZE
        assert captured["img_width"] == CNN_IMG_SIZE

    def test_dm_does_not_depend_on_cube_height(self, tmp_path, monkeypatch):
        """Same box, two very different cube heights, same DM.

        Before the fix this produced 640.0 (saturated at DM_max) for the short
        cube and roughly half the range for the tall one.
        """
        rows_short, _ = self._run_band(tmp_path, monkeypatch, cube_height=147)
        rows_tall, _ = self._run_band(tmp_path, monkeypatch, cube_height=1001)

        assert len(rows_short) == 1 and len(rows_tall) == 1
        dm_short = float(rows_short[0]["dm_pc_cm-3"])
        dm_tall = float(rows_tall[0]["dm_pc_cm-3"])
        assert dm_short == pytest.approx(dm_tall, abs=1e-6)

    def test_dm_is_not_pinned_to_the_range_edge(self, tmp_path, monkeypatch):
        """A box in the middle of the CNN frame must land in the middle of the
        DM range, not saturate at DM_max."""
        rows, _ = self._run_band(tmp_path, monkeypatch, cube_height=147)
        dm = float(rows[0]["dm_pc_cm-3"])
        assert config.DM_min < dm < config.DM_max
        # Box spans rows 256..300 of 512 -> just past the middle of 480..640.
        assert dm == pytest.approx(480.0 + (278.0 / 511.0) * 160.0, abs=1.0)

    def test_time_sample_scales_from_the_cnn_frame(self, tmp_path, monkeypatch):
        """A box at the right edge of the CNN frame must map to the end of the
        slice, even when the slice is longer than 512 samples."""
        slice_len = 977
        box = [500.0, 256.0, 511.0, 300.0]
        rows, _ = self._run_band(
            tmp_path, monkeypatch, cube_height=147, cube_width=slice_len,
            box=box, slice_len=slice_len,
        )
        t_sample = int(rows[0]["t_sample"])
        centre_px = (box[0] + box[2]) / 2.0
        assert t_sample == int(centre_px * slice_len / CNN_IMG_SIZE)
        # The pre-fix behaviour was scale_time == 1.0, i.e. t_sample == centre_px.
        assert t_sample > centre_px


# --------------------------------------------------------------------------- #
# P0-2
# --------------------------------------------------------------------------- #

class TestCandidateWriterDurability:
    """Buffered rows must reach disk on every exit path."""

    def _row(self) -> list[str]:
        return ["x"] * len(CANDIDATE_HEADER)

    def test_rows_below_flush_interval_reach_disk_on_flush_all(self, tmp_path):
        csv_file = tmp_path / "c.csv"
        ensure_csv_header(csv_file)
        writer = CandidateWriter.get(csv_file)
        for _ in range(3):  # fewer than flush_interval (50)
            writer.write(self._row())

        with csv_file.open(newline="") as fh:
            assert len(list(csv.reader(fh))) == 1, "rows should still be buffered"

        CandidateWriter.flush_all()

        with csv_file.open(newline="") as fh:
            assert len(list(csv.reader(fh))) == 1 + 3

    def test_flush_buffers_persists_without_closing_the_writer(self, tmp_path):
        """The durability barrier used before each checkpoint must not tear the
        writer down, otherwise every chunk would reopen the file."""
        csv_file = tmp_path / "c.csv"
        ensure_csv_header(csv_file)
        CandidateWriter.get(csv_file).write(self._row())

        CandidateWriter.flush_buffers()

        with csv_file.open(newline="") as fh:
            assert len(list(csv.reader(fh))) == 2
        assert csv_file.resolve() in CandidateWriter._instances
        CandidateWriter.flush_all()

    def test_rows_survive_interpreter_exit_without_an_explicit_flush(self, tmp_path):
        """atexit safety net: a process that writes and exits without calling
        flush_all must still leave the rows on disk."""
        csv_file = tmp_path / "c.csv"
        script = textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(PROJECT_ROOT)!r})
            from src.output.candidate_manager import (
                CANDIDATE_HEADER, CandidateWriter, ensure_csv_header,
            )
            from pathlib import Path
            path = Path({str(csv_file)!r})
            ensure_csv_header(path)
            w = CandidateWriter.get(path)
            for _ in range(3):
                w.write(["x"] * len(CANDIDATE_HEADER))
            # deliberately no flush_all(): atexit has to do it
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr

        with csv_file.open(newline="") as fh:
            assert len(list(csv.reader(fh))) == 1 + 3, (
                "rows buffered at interpreter exit were lost"
            )

    def test_file_processor_flushes_on_every_exit_path(self):
        """``_process_file_chunked`` must flush in a ``finally``: it owns the
        five error returns and the early return that hands control to the HF
        pipeline."""
        func = _function_ast(PROJECT_ROOT / "src/core/pipeline.py", "_process_file_chunked")
        tries = [n for n in ast.walk(func) if isinstance(n, ast.Try) and n.finalbody]
        flushed = any(
            _calls_named(ast.Module(body=t.finalbody, type_ignores=[]),
                         "CandidateWriter.flush_all")
            for t in tries
        )
        assert flushed, "no finally: block flushes the candidate writer"

    def test_high_freq_pipeline_flushes_before_returning(self):
        """The HF path must also be correct on its own, not only via its caller."""
        func = _function_ast(
            PROJECT_ROOT / "src/core/high_freq_pipeline.py",
            "_process_file_chunked_high_freq",
        )
        assert _calls_named(func, "CandidateWriter.flush_all"), (
            "HF pipeline returns without flushing buffered candidates"
        )


# --------------------------------------------------------------------------- #
# P0-3
# --------------------------------------------------------------------------- #

class TestResumeSkipsExactlyCompletedChunks:
    """``resume_after`` is the last COMPLETED chunk, so the next one to process
    is ``resume_after + 1``."""

    def test_next_chunk_after_the_checkpoint_is_processed(self):
        assert should_skip_chunk(5, resume_after=5) is True
        assert should_skip_chunk(6, resume_after=5) is False, (
            "chunk 6 was never processed and must not be skipped"
        )

    def test_no_checkpoint_skips_nothing(self):
        assert [i for i in range(1, 6) if should_skip_chunk(i, -1)] == []

    def test_skipped_set_is_exactly_the_completed_prefix(self):
        skipped = [i for i in range(1, 9) if should_skip_chunk(i, resume_after=5)]
        assert skipped == [1, 2, 3, 4, 5]

    def test_checkpoint_round_trip(self, tmp_path):
        """``save_checkpoint`` stores the same 1-based index ``load_checkpoint``
        returns; the resume predicate is built on that agreement."""
        stem = "obs01"
        assert load_checkpoint(tmp_path, stem) == -1

        save_checkpoint(tmp_path, stem, chunk_idx=5, total_chunks=8)
        assert load_checkpoint(tmp_path, stem) == 5
        assert should_skip_chunk(6, load_checkpoint(tmp_path, stem)) is False

        clear_checkpoint(tmp_path, stem)
        assert load_checkpoint(tmp_path, stem) == -1

    def test_corrupt_checkpoint_restarts_from_scratch(self, tmp_path):
        stem = "obs02"
        save_checkpoint(tmp_path, stem, chunk_idx=3, total_chunks=8)
        cp = tmp_path / stem / ".drafts_checkpoint.json"
        cp.write_text("{not json", encoding="utf-8")
        assert load_checkpoint(tmp_path, stem) == -1

    def test_pipeline_uses_the_shared_predicate(self):
        """Guard against the condition being re-inlined with the old ``+ 1``."""
        source = (PROJECT_ROOT / "src/core/pipeline.py").read_text(encoding="utf-8")
        assert "should_skip_chunk(chunk_idx, resume_after)" in source
        assert "chunk_idx <= resume_after + 1" not in source


# --------------------------------------------------------------------------- #
# P0-4
# --------------------------------------------------------------------------- #

class TestCheckpointOnlyAfterSuccessfulChunk:
    """A chunk that raised must stay un-checkpointed so a resume retries it.

    The guard lives inside a 200-line streaming loop that cannot be
    instantiated without real models and a real file, so its structure is
    asserted against the AST -- the same approach ``test_dedispersion_parity``
    already uses for the CUDA kernel.
    """

    @staticmethod
    def _loop_body() -> ast.FunctionDef:
        return _function_ast(PROJECT_ROOT / "src/core/pipeline.py", "_process_file_chunked")

    def test_save_checkpoint_is_guarded_by_chunk_success(self):
        func = self._loop_body()
        calls = _calls_named(func, "save_checkpoint")
        assert calls, "save_checkpoint disappeared from the chunk loop"

        guarded = []
        for node in ast.walk(func):
            if isinstance(node, ast.If) and ast.unparse(node.test) == "chunk_succeeded":
                guarded.extend(_calls_named(ast.Module(body=node.body, type_ignores=[]),
                                            "save_checkpoint"))
        assert len(guarded) == len(calls), (
            "every save_checkpoint in the chunk loop must sit under "
            "`if chunk_succeeded:`; a failed chunk that gets checkpointed is "
            "skipped for good on resume"
        )

    def test_success_flag_is_set_only_after_the_work_completed(self):
        """``chunk_succeeded = True`` must be the last statement of the try body,
        so any raise inside leaves it False."""
        func = self._loop_body()
        for node in ast.walk(func):
            if not isinstance(node, ast.Try):
                continue
            assigns = [
                s for s in node.body
                if isinstance(s, ast.Assign)
                and any(getattr(t, "id", None) == "chunk_succeeded" for t in s.targets)
            ]
            if assigns:
                assert assigns[-1] is node.body[-1], (
                    "chunk_succeeded must be set as the final statement of the try"
                )
                return
        pytest.fail("no try block sets chunk_succeeded")

    def test_rows_are_flushed_before_the_checkpoint_lands(self):
        """The checkpoint claims the chunk's rows are durable, so they must be
        written before it is recorded."""
        source = (PROJECT_ROOT / "src/core/pipeline.py").read_text(encoding="utf-8")
        flush_at = source.index("CandidateWriter.flush_buffers()")
        save_at = source.index(
            "save_checkpoint(save_dir, fits_path.stem, chunk_idx, chunk_count)"
        )
        assert flush_at < save_at, (
            "candidate rows must be flushed before save_checkpoint records progress"
        )
