"""Checkpoint, resume and CSV rotation, asserted on both drivers' behaviour.

This replaces the last group of tests that pinned the drivers by reading their
source. Six of them (audit REF-01's "known limit"):

    test_p0_regressions.py::TestCheckpointOnlyAfterSuccessfulChunk
        ::test_save_checkpoint_is_guarded_by_chunk_success        [AST]
        ::test_success_flag_is_set_only_after_the_work_completed  [AST]
    test_p0_regressions.py::...::test_rows_are_flushed_before_the_checkpoint_lands
                                                                  [string index]
    test_p2_reliability.py::...::test_high_freq_pipeline_checkpoints
    test_p2_reliability.py::...::test_high_freq_only_checkpoints_successful_chunks
    test_p2_reliability.py::...::test_pipelines_rotate_only_when_not_resuming

They were written that way because, as one of their docstrings puts it, "the
guard lives inside a 200-line streaming loop that cannot be instantiated
without real models and a real file". That stopped being true: the filterbank
and PSRFITS writers in this suite drive both drivers end to end, and
``test_e2e_pipeline`` and ``test_hf_e2e`` already do.

Why it matters beyond tidiness. Those tests pin the SHAPE of code that exists
twice -- the checkpoint/resume/rotate preamble is ~30 duplicated lines across
``pipeline.py`` and ``high_freq_pipeline.py``, and ``file_driver.py`` says so
where it declines to share them. Asserting on shape means the duplication
cannot be removed without the tests failing, which is exactly what REF-01 ran
into. What follows asserts the contract instead:

    * a chunk that fails does not advance the checkpoint, so a resume retries it
    * a resume does not reprocess what was already done, and does not duplicate
      its rows
    * the rows a checkpoint vouches for are on disk BEFORE it is written
    * a fresh run rotates the previous CSV aside; a resuming run appends

One more thing the source-text form could not do: it asserted these of the
low-frequency driver by AST and of the high-frequency driver by substring, in
two different files, with no shared statement of what the contract IS. Here
both drivers are the same parametrised tests, so they cannot drift apart.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.config import config
from src.core.checkpoint import load_checkpoint
from src.output.candidate_manager import CANDIDATE_HEADER, CandidateWriter

# --------------------------------------------------------------------------- #
# the two drivers, behind one interface
# --------------------------------------------------------------------------- #

ROWS_PER_SLICE = 3


def _csv_path(save_dir: Path, stem: str) -> Path:
    return save_dir / "Summary" / stem / f"{stem}.candidates.csv"


def _rotated_files(save_dir: Path, stem: str) -> list[Path]:
    """Copies ``rotate_previous_candidates`` moved aside.

    It builds the name from ``Path.stem``/``Path.suffix`` of
    ``<stem>.candidates.csv``, so the stamp lands before the extension:
    ``<stem>.candidates.<UTC stamp>.csv``.
    """
    live = _csv_path(save_dir, stem)
    return [p for p in live.parent.glob(f"{stem}.candidates.*.csv") if p != live]


def _rows(save_dir: Path, stem: str) -> list[dict]:
    CandidateWriter.flush_all()
    path = _csv_path(save_dir, stem)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _setup_lf(tmp_path, monkeypatch):
    """Low-frequency driver over a synthetic filterbank."""
    from src.core import detection_engine as de
    from src.core import pipeline as pipeline_mod
    from tests.synthetic_filterbank import write_filterbank
    from tests.test_e2e_pipeline import DM_TRUE, TSAMP, _configure, _install_peak_detector

    monkeypatch.setattr(pipeline_mod, "_load_detection_model", lambda: None)
    monkeypatch.setattr(pipeline_mod, "_load_class_model", lambda: None)
    monkeypatch.setattr(de, "save_all_plots", lambda *a, **k: None)

    _configure(tmp_path, chunk_samples=2000)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = config.DATA_DIR / "synthetic_burst.fil"
    write_filterbank(data, nsamples=8000, dm=DM_TRUE, burst_time_s=1.5, tsamp=TSAMP)
    _install_peak_detector(monkeypatch)
    return data, config.RESULTS_DIR


def _setup_hf(tmp_path, monkeypatch):
    """High-frequency driver over a synthetic multi-pol PSRFITS."""
    from tests.test_hf_e2e import _configure_hf, _stub_slices, _write_hf_file

    fits_path = _write_hf_file(tmp_path)
    save_dir = _configure_hf(monkeypatch, tmp_path)
    _stub_slices(monkeypatch, {"rows": 0})
    return fits_path, save_dir


def _run_lf(data: Path, save_dir: Path) -> dict | None:
    from src.core.pipeline import run_pipeline

    run_pipeline()
    return None


def _run_hf(fits_path: Path, save_dir: Path) -> dict:
    from src.core.pipeline import _process_file_chunked

    return _process_file_chunked(None, None, fits_path, save_dir, 1024)


DRIVERS = [
    pytest.param("lf", id="low_frequency"),
    pytest.param("hf", id="high_frequency"),
]


@pytest.fixture
def driver(request, tmp_path, monkeypatch):
    """Whichever driver the test was parametrised with, set up and runnable."""
    pytest.importorskip("astropy")
    which = request.param
    if which == "lf":
        data, save_dir = _setup_lf(tmp_path, monkeypatch)
        run = lambda: _run_lf(data, save_dir)  # noqa: E731
    else:
        data, save_dir = _setup_hf(tmp_path, monkeypatch)
        run = lambda: _run_hf(data, save_dir)  # noqa: E731
    return {
        "which": which, "path": data, "stem": data.stem,
        "save_dir": save_dir, "run": run, "monkeypatch": monkeypatch,
    }


def _break_chunk(driver: dict, chunk_number: int, error=RuntimeError) -> dict:
    """Make the *chunk_number*-th chunk (1-based) fail inside the chunk body.

    The two drivers do their per-chunk work through different functions, so the
    injection point differs; what is asserted afterwards does not.
    """
    seen = {"n": 0, "armed": True, "raised": 0}
    mp = driver["monkeypatch"]

    if driver["which"] == "lf":
        from src.core import pipeline as pipeline_mod

        real = pipeline_mod._process_block

        def _maybe_fail(*args, **kwargs):
            seen["n"] += 1
            if seen["armed"] and seen["n"] == chunk_number:
                seen["raised"] += 1
                raise error("simulated chunk failure")
            return real(*args, **kwargs)

        mp.setattr(pipeline_mod, "_process_block", _maybe_fail)
    else:
        from src.core import high_freq_pipeline as hfp

        real_slice = hfp.process_slice_with_multiple_bands_high_freq
        chunks: set = set()

        def _maybe_fail(**kwargs):
            chunks.add(kwargs["chunk_idx"])
            seen["n"] = len(chunks)
            if seen["armed"] and len(chunks) == chunk_number:
                seen["raised"] += 1
                raise error("simulated chunk failure")
            return real_slice(**kwargs)

        mp.setattr(hfp, "process_slice_with_multiple_bands_high_freq", _maybe_fail)

    return seen


def _spy_on_save_checkpoint(driver: dict) -> list:
    """Record ``(chunk_idx, rows on disk at that moment)`` for every checkpoint.

    Watching the calls rather than the file on disk matters: a run that finishes
    clears its checkpoint, so by the time a test looks there may be nothing
    left to inspect.
    """
    import importlib

    from src.core import checkpoint as checkpoint_mod

    observations: list[tuple[int, int]] = []
    real_save = checkpoint_mod.save_checkpoint

    def _spy(results_dir, file_stem, chunk_idx, total_chunks, *a, **kw):
        path = _csv_path(Path(results_dir), file_stem)
        on_disk = 0
        if path.exists():
            with path.open(newline="", encoding="utf-8") as fh:
                on_disk = len(list(csv.DictReader(fh)))
        observations.append((int(chunk_idx), on_disk))
        return real_save(results_dir, file_stem, chunk_idx, total_chunks, *a, **kw)

    # Patch every module that holds a binding. Since REF-01 shared the block,
    # the live call site is ``file_driver.checkpoint_completed_chunk``, so
    # ``file_driver`` is the one that matters; the drivers are patched too in
    # case a future change calls save_checkpoint directly again.
    driver["monkeypatch"].setattr(checkpoint_mod, "save_checkpoint", _spy)
    for module_name in ("src.core.file_driver", "src.core.pipeline",
                        "src.core.high_freq_pipeline"):
        module = importlib.import_module(module_name)
        if hasattr(module, "save_checkpoint"):
            driver["monkeypatch"].setattr(module, "save_checkpoint", _spy)
    return observations


# --------------------------------------------------------------------------- #
# P0-4: a chunk that failed must not be checkpointed
# --------------------------------------------------------------------------- #

class TestOnlySuccessfulChunksAdvanceTheCheckpoint:
    """P0-4, as a contract rather than as a shape.

    The AST tests asserted that every ``save_checkpoint`` sits under
    ``if chunk_succeeded:`` and that ``chunk_succeeded = True`` is the last
    statement of the try. Both spell one thing: a chunk that raised is never
    recorded as done. That is asserted here instead, so it survives the flag
    being renamed, the guard restructured, or the whole block moved into
    ``file_driver`` and shared -- which is what REF-01 wanted and these tests
    forbade.
    """

    @pytest.mark.parametrize("driver", DRIVERS, indirect=True)
    def test_a_clean_run_clears_its_checkpoint(self, driver):
        """The control. A finished file leaves nothing to resume from."""
        driver["run"]()

        assert _rows(driver["save_dir"], driver["stem"]), "the run wrote nothing"
        assert load_checkpoint(driver["save_dir"], driver["stem"]) == -1, (
            "a file that completed still has a checkpoint; the next run would "
            "skip chunks it never processed"
        )

    @pytest.mark.parametrize("driver", DRIVERS, indirect=True)
    def test_the_failed_chunk_is_never_checkpointed(self, driver):
        """The assertion itself: that chunk index is not among those recorded.

        Checked on the calls, not on the file: a run that survives its failed
        chunk finishes and clears the checkpoint, so looking afterwards would
        find nothing either way and the test would pass vacuously.
        """
        observations = _spy_on_save_checkpoint(driver)
        seen = _break_chunk(driver, chunk_number=2)

        driver["run"]()

        assert seen["raised"] == 1, "the injected failure never fired"
        assert observations, "no chunk was checkpointed at all"
        recorded = [idx for idx, _ in observations]
        assert 2 not in recorded, (
            f"the failed chunk was checkpointed anyway: {recorded}. On a resume "
            "it would be skipped for good."
        )
        assert 1 in recorded, (
            f"the chunk BEFORE the failure was not checkpointed either: "
            f"{recorded}; a resume would redo work that succeeded"
        )


class TestTheRowsAreOnDiskBeforeTheCheckpointClaimsThem:
    """Replaces a test that compared two ``str.index`` offsets in the source.

    A checkpoint says "everything up to chunk N is written". If the rows are
    still in the writer's buffer when it is recorded, a crash between the two
    loses rows the checkpoint vouches for, and the resume never revisits that
    chunk. Rather than asserting that the flush call appears earlier in the
    file than the checkpoint call, this reads the CSV from inside
    ``save_checkpoint`` and checks the rows are really there.
    """

    @pytest.mark.parametrize("driver", DRIVERS, indirect=True)
    def test_every_checkpoint_is_backed_by_rows_already_written(self, driver):
        observations = _spy_on_save_checkpoint(driver)

        driver["run"]()

        assert observations, "no checkpoint was ever written"
        for chunk_idx, on_disk in observations:
            assert on_disk > 0, (
                f"the checkpoint for chunk {chunk_idx} was written while the "
                "CSV held no rows: the buffer had not been flushed, so a crash "
                "there loses rows the checkpoint says are safe"
            )
        counts = [n for _, n in observations]
        assert counts == sorted(counts), (
            f"a later checkpoint vouched for fewer rows than an earlier one: "
            f"{observations}"
        )


class TestResumingDoesTheRestAndNotTheSame:
    """What the checkpoint is FOR, which no source-text test could reach.

    They asserted the machinery is present -- that the four checkpoint calls
    appear in the file, spelled a certain way. None could say whether resuming
    works.

    A resume needs a run that ABORTED, not merely one with a failed chunk: both
    drivers record a failed chunk, carry on, finish the file and clear the
    checkpoint (see TestAFailedChunkDoesNotStopTheFile). MemoryError is the
    exception both re-raise out of the chunk loop, so it is what aborts here.
    """

    @staticmethod
    def _keys(rows):
        return [
            (r["file"], r["chunk_id"], r["slice_id"], r["band_id"], r["t_sample"])
            for r in rows
        ]

    @pytest.mark.parametrize("driver", DRIVERS, indirect=True)
    def test_a_resume_finishes_the_file_without_duplicating_rows(self, driver):
        seen = _break_chunk(driver, chunk_number=2, error=MemoryError)

        try:
            driver["run"]()
        except MemoryError:
            pass                      # LF lets it out; HF returns a result

        after_abort = _rows(driver["save_dir"], driver["stem"])
        assert seen["raised"] == 1, "the injected failure never fired"
        assert after_abort, "the aborted run wrote nothing to resume from"
        resume_after = load_checkpoint(driver["save_dir"], driver["stem"])
        assert resume_after >= 1, (
            f"the aborted run left checkpoint {resume_after}; a resume would "
            "redo the whole file and duplicate every row already written"
        )

        seen["armed"] = False         # this pass is the resume
        driver["run"]()
        after_resume = _rows(driver["save_dir"], driver["stem"])

        assert len(after_resume) > len(after_abort), (
            "the resume added no rows, so the chunk that aborted the first run "
            "was never reprocessed"
        )
        keys = self._keys(after_resume)
        duplicates = {k for k in keys if keys.count(k) > 1}
        assert not duplicates, (
            f"{len(duplicates)} candidate key(s) written twice across the two "
            f"runs, e.g. {sorted(duplicates)[:2]}. The resume redid work the "
            "checkpoint said was already done."
        )
        assert load_checkpoint(driver["save_dir"], driver["stem"]) == -1, (
            "the completed resume left its checkpoint behind"
        )


class TestAFailedChunkDoesNotStopTheFile:
    """PINNED AS-IS, and worth knowing about.

    A chunk that raises anything other than MemoryError is recorded, the loop
    carries on, the file finishes with a ``*_PARTIAL`` status and its
    checkpoint is CLEARED. So that chunk is never retried: P0-4's guarantee --
    that a failed chunk stays un-checkpointed so a resume retries it -- only
    has an effect when the run aborts outright.

    That may well be intended; a file whose status says PARTIAL has recorded
    its loss. It is pinned here because no test said it either way, and the
    source-text tests it replaces could not have.
    """

    @pytest.mark.parametrize("driver", DRIVERS, indirect=True)
    def test_the_run_completes_and_keeps_no_checkpoint(self, driver):
        seen = _break_chunk(driver, chunk_number=2)

        driver["run"]()

        assert seen["raised"] == 1
        assert _rows(driver["save_dir"], driver["stem"]), "nothing was written"
        assert load_checkpoint(driver["save_dir"], driver["stem"]) == -1, (
            "a file with a failed chunk kept its checkpoint; if that is the "
            "intended behaviour now, this test records the old one"
        )


class TestAFreshRunRotatesAndAResumeAppends:
    """P1-13, replacing a test that searched 300 characters of source.

    The old one located ``rotate_previous_candidates(csv_file)`` in each
    driver's text and asserted ``if resume_after < 0:`` appears somewhere in the
    300 characters before it. That is three things at once -- a call, a guard,
    and their proximity -- none of which is the contract. The contract is: a
    fresh run must not append to the previous run's candidates, and a resume
    must not start a new file.

    The writer opens in append mode, so without the rotation two runs over the
    same input interleave in one CSV with nothing to tell them apart, while the
    plots -- which are overwritten -- show only the newer run.
    """

    @pytest.mark.parametrize("driver", DRIVERS, indirect=True)
    def test_a_second_fresh_run_moves_the_first_one_aside(self, driver):
        driver["run"]()
        first = _rows(driver["save_dir"], driver["stem"])
        assert first, "the first run wrote nothing"
        assert load_checkpoint(driver["save_dir"], driver["stem"]) == -1, (
            "the first run left a checkpoint, so the second is a resume and "
            "this test would be asserting the wrong half"
        )

        driver["run"]()
        second = _rows(driver["save_dir"], driver["stem"])

        assert len(second) == len(first), (
            f"the second run's CSV holds {len(second)} rows against the first "
            f"run's {len(first)}: the runs were interleaved instead of the "
            "previous file being moved aside"
        )
        rotated = sorted(_rotated_files(driver["save_dir"], driver["stem"]))
        assert rotated, (
            "no rotated file: the previous run's candidates were lost rather "
            "than kept beside the new ones"
        )
        with rotated[-1].open(newline="", encoding="utf-8") as fh:
            assert len(list(csv.DictReader(fh))) == len(first), (
                "the rotated file does not hold the first run's rows"
            )

    @pytest.mark.parametrize("driver", DRIVERS, indirect=True)
    def test_a_resume_appends_instead_of_rotating(self, driver):
        """The other half, and the reason the guard exists: rotating on a
        resume would throw away everything the interrupted run had written."""
        seen = _break_chunk(driver, chunk_number=2, error=MemoryError)
        try:
            driver["run"]()
        except MemoryError:
            pass
        after_abort = _rows(driver["save_dir"], driver["stem"])
        assert seen["raised"] == 1 and after_abort

        seen["armed"] = False
        driver["run"]()
        after_resume = _rows(driver["save_dir"], driver["stem"])

        assert len(after_resume) > len(after_abort), (
            "the resume started a new CSV instead of appending; the interrupted "
            "run's candidates are gone"
        )
        rotated = _rotated_files(driver["save_dir"], driver["stem"])
        assert not rotated, f"a resume rotated the CSV aside: {rotated}"
