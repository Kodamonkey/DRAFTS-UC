"""End-to-end tests for the high-frequency driver, over a real PSRFITS.

Nothing in this suite drove the high-frequency path from end to end. The audit
found four defects in it; all four survived because the only tests that touched
the driver read its source text or called one helper in isolation. This module
runs the production entry point -- ``pipeline._process_file_chunked`` -- over a
file written by ``tests/synthetic_psrfits.py``, and lets the real dispatch
decide the path.

What is real here and what is not
---------------------------------
Real: the file, the header load, ``select_pipeline_path``, the chunk plan, the
multi-polarisation reader, the chunk loop, the candidate writer, the checkpoint,
and the result dictionary the caller receives.

Stubbed: ``process_slice_with_multiple_bands_high_freq``, the per-slice
detection. It needs a classifier and its arithmetic is covered elsewhere
(``test_hf_phases.py``, ``test_scientific_physics.py``). The stub writes real
rows through the real ``append_candidate``, which is the part these tests are
about: what the driver does with the counts when something goes wrong.

Why the band is 350 GHz
-----------------------
``select_pipeline_path`` sends a file to the high-frequency driver when the
dispersion sweep collapses below the time resolution. A 349.7-350.0 GHz band at
DM 500 gives a sweep of ~0 ms against a 1 ms sample, so the real criterion picks
the real driver -- no flag is forced, which is what makes this a dispatch test
as well as a driver test.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.config import config
from src.output.candidate_manager import CANDIDATE_HEADER

# 350 GHz, so the bow-tie criterion collapses and the HF driver is chosen.
FCH1_MHZ = 350_000.0
FOFF_MHZ = -10.0
NSUBINT = 64
NSBLK = 64
NCHAN = 32
TSAMP = 1.0e-3
CHUNK_SAMPLES = 1024

#: Rows the stubbed slice processor writes per slice.
ROWS_PER_SLICE = 5


def _write_hf_file(tmp_path: Path) -> Path:
    """A multi-polarisation PSRFITS the high-frequency driver accepts."""
    from src.input.fits_handler import get_obparams
    from tests.synthetic_psrfits import write_psrfits

    path = tmp_path / "hf_synth.fits"
    write_psrfits(
        path, nsubint=NSUBINT, nsblk=NSBLK, nchan=NCHAN, npol=4,
        pol_type="IQUV", tsamp=TSAMP, fch1=FCH1_MHZ, foff=FOFF_MHZ,
        dm=300.0, burst_time_s=0.1,
    )
    get_obparams(str(path))
    return path


def _configure_hf(monkeypatch, tmp_path: Path) -> Path:
    save_dir = tmp_path / "out"
    save_dir.mkdir(parents=True, exist_ok=True)
    for key, value in {
        "DM_min": 0.0, "DM_max": 500.0,
        "DOWN_TIME_RATE": 1, "DOWN_FREQ_RATE": 1,
        "SLICE_DURATION_MS": 64.0,
        "AUTO_HIGH_FREQ_PIPELINE": True,
        "SAVE_ONLY_BURST": False,
        "FORCE_PLOTS": False,
        "USE_MULTI_BAND": False,
        "RESULTS_DIR": save_dir,
    }.items():
        monkeypatch.setattr(config, key, value, raising=False)
    return save_dir


def _stub_slices(monkeypatch, counter: dict) -> None:
    """Write real candidate rows without needing a classifier."""
    from src.core import high_freq_pipeline as hfp
    from src.output.candidate_manager import append_candidate

    def _fake_slice(**kwargs):
        for _ in range(ROWS_PER_SLICE):
            row = [""] * len(CANDIDATE_HEADER)
            row[0] = str(kwargs["fits_path"].name)
            row[1] = kwargs["chunk_idx"]
            row[2] = kwargs["j"]
            row[3] = 0
            row[4] = 0.9
            append_candidate(kwargs["csv_file"], row)
            counter["rows"] += 1
        return ROWS_PER_SLICE, ROWS_PER_SLICE, 0, 0.9

    monkeypatch.setattr(
        hfp, "process_slice_with_multiple_bands_high_freq", _fake_slice
    )


def _fail_after_chunk(monkeypatch, n_chunks: int, error: Exception) -> None:
    """Let the real reader yield *n_chunks*, then raise mid-stream."""
    from src.input import fits_handler

    real_stream = fits_handler.stream_fits_multi_pol

    def _failing(file_name, chunk_samples, overlap_samples=0):
        for i, item in enumerate(
            real_stream(file_name, chunk_samples, overlap_samples=overlap_samples), 1
        ):
            if i > n_chunks:
                raise error
            yield item
        raise error

    monkeypatch.setattr(fits_handler, "stream_fits_multi_pol", _failing)


def _rows_on_disk(save_dir: Path, stem: str) -> list[dict]:
    from src.output.candidate_manager import CandidateWriter

    CandidateWriter.flush_all()
    csv_path = save_dir / "Summary" / stem / f"{stem}.candidates.csv"
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _run(fits_path: Path, save_dir: Path) -> dict:
    from src.core.pipeline import _process_file_chunked

    return _process_file_chunked(None, None, fits_path, save_dir, CHUNK_SAMPLES)


@pytest.fixture
def hf_run(tmp_path, monkeypatch):
    pytest.importorskip("astropy")
    fits_path = _write_hf_file(tmp_path)
    save_dir = _configure_hf(monkeypatch, tmp_path)
    counter = {"rows": 0}
    _stub_slices(monkeypatch, counter)
    return fits_path, save_dir, counter, monkeypatch


class TestTheHighFrequencyDriverIsReached:
    """Before anything else: the dispatch must actually send this file to HF."""

    def test_the_bowtie_criterion_selects_the_high_frequency_path(self, hf_run):
        from src.core.file_driver import select_pipeline_path

        use_hf, reason = select_pipeline_path()
        assert use_hf, f"the HF driver was not selected; reason was {reason!r}"
        assert "bow-tie" in reason or "collapsed" in reason

    def test_a_clean_run_reports_what_it_wrote(self, hf_run):
        """The success path, which is also the control for the tests below."""
        fits_path, save_dir, counter, monkeypatch = hf_run

        result = _run(fits_path, save_dir)

        assert result["status"].startswith("SUCCESS"), result
        rows = _rows_on_disk(save_dir, fits_path.stem)
        assert rows, "the run wrote no candidate rows, so it proves nothing"
        assert counter["rows"] == len(rows)
        assert result["n_candidates"] == len(rows), (
            f"reported {result['n_candidates']} candidates, {len(rows)} on disk"
        )


class TestAFailedRunStillReportsWhatItWrote:
    """The defect the audit calls the serious one.

    The driver re-raises instead of returning a result, so the counts it kept --
    which are function locals -- die with the frame. Its caller converts the
    exception into a per-file result built from a fresh, empty ``DetectionStats``.
    A run that wrote hundreds of rows and then failed reported
    ``n_candidates: 0`` with those rows already on disk, which reads as "this
    file had no detections" and is how real candidates get discarded.

    ``_error_result``'s own docstring says the counts "are what the run actually
    produced and wrote before failing, not zeros". That is true of the
    low-frequency driver and was never true of this one.
    """

    def test_the_reported_count_matches_the_rows_on_disk(self, hf_run):
        fits_path, save_dir, counter, monkeypatch = hf_run
        _fail_after_chunk(monkeypatch, n_chunks=1,
                          error=IOError("simulated read failure"))

        result = _run(fits_path, save_dir)

        rows = _rows_on_disk(save_dir, fits_path.stem)
        assert rows, "the failing run wrote nothing, so the test is vacuous"
        assert result["status"].startswith("ERROR"), result
        assert result["n_candidates"] == len(rows), (
            f"the run wrote {len(rows)} rows to disk and reported "
            f"{result['n_candidates']}; a caller reading this result concludes "
            "the file had no detections"
        )

    def test_the_rows_it_counted_are_all_flushed(self, hf_run):
        """The second defect: the driver flushes on its success path only.

        The writer buffers rows, so a failure between the last flush and the
        raise drops rows the driver had already counted. The caller has a
        ``finally: flush_all()`` stop-gap; this asserts the driver is correct on
        its own, because the counts it returns have to match what is readable.
        """
        from src.output.candidate_manager import CandidateWriter

        fits_path, save_dir, counter, monkeypatch = hf_run
        _fail_after_chunk(monkeypatch, n_chunks=1,
                          error=IOError("simulated read failure"))

        result = _run(fits_path, save_dir)

        csv_path = save_dir / "Summary" / fits_path.stem / f"{fits_path.stem}.candidates.csv"
        assert csv_path.exists()
        with csv_path.open(newline="", encoding="utf-8") as fh:
            on_disk_before_flush = len(list(csv.DictReader(fh)))
        CandidateWriter.flush_all()
        with csv_path.open(newline="", encoding="utf-8") as fh:
            on_disk_after_flush = len(list(csv.DictReader(fh)))

        assert on_disk_before_flush == on_disk_after_flush, (
            f"{on_disk_after_flush - on_disk_before_flush} rows were still "
            "buffered when the driver gave up"
        )
        assert result["n_candidates"] == on_disk_after_flush

    def test_the_error_is_still_reported(self, hf_run):
        """Returning a result must not turn a failure into a success."""
        fits_path, save_dir, _, monkeypatch = hf_run
        _fail_after_chunk(monkeypatch, n_chunks=1,
                          error=IOError("simulated read failure"))

        result = _run(fits_path, save_dir)

        assert result["status"].startswith("ERROR")
        assert "simulated read failure" in result.get("error_details", "")


class TestAFailedChunkStillReleasesItsArrays:
    """The third HF defect: cleanup sat inside the per-chunk ``try``.

    ``del block_ds, dm_time, block_raw_ds`` and ``optimize_memory`` were the
    last statements of the try body, just before ``chunk_succeeded = True``. A
    chunk that raised anywhere in the slice loop skipped all of it, so its
    arrays stayed referenced until the next iteration rebound the names -- and
    the chunk most likely to raise is the one that ran out of memory, which is
    exactly when holding a cube and two blocks longer than necessary is worst.

    The obvious fix is wrong and was not made. Mirroring the low-frequency
    driver -- moving the ``del`` after the handler -- works there because LF's
    ``block`` is bound by the ``for`` statement, before its try. Every name here
    is bound INSIDE the try, so on the failure path the move raises NameError
    while handling the original error, losing it.
    """

    def _fail_one_chunk(self, monkeypatch, counter, fail_on_chunk: int):
        """Real rows for every chunk, an exception raised inside chunk N."""
        from src.core import high_freq_pipeline as hfp
        from src.output.candidate_manager import append_candidate

        def _slice(**kwargs):
            counter.setdefault("chunks", set()).add(kwargs["chunk_idx"])
            if kwargs["chunk_idx"] == fail_on_chunk:
                raise RuntimeError("simulated slice failure")
            for _ in range(ROWS_PER_SLICE):
                row = [""] * len(CANDIDATE_HEADER)
                row[0] = str(kwargs["fits_path"].name)
                row[1] = kwargs["chunk_idx"]
                row[2] = kwargs["j"]
                row[3] = 0
                row[4] = 0.9
                append_candidate(kwargs["csv_file"], row)
                counter["rows"] += 1
            return ROWS_PER_SLICE, ROWS_PER_SLICE, 0, 0.9

        monkeypatch.setattr(
            hfp, "process_slice_with_multiple_bands_high_freq", _slice
        )

    def _spy_on_optimize_memory(self, monkeypatch):
        from src.core import file_driver, high_freq_pipeline as hfp

        calls = []
        real = file_driver.optimize_memory

        def _spy(*args, **kwargs):
            calls.append(kwargs.get("aggressive"))
            return real(*args, **kwargs)

        # Bound into the driver's namespace at import, so patch it there.
        monkeypatch.setattr(hfp, "optimize_memory", _spy)
        return calls

    def test_memory_is_reclaimed_for_a_chunk_that_failed(self, hf_run):
        fits_path, save_dir, counter, monkeypatch = hf_run
        self._fail_one_chunk(monkeypatch, counter, fail_on_chunk=1)
        calls = self._spy_on_optimize_memory(monkeypatch)

        result = _run(fits_path, save_dir)

        assert result["failed_chunks"] >= 1, (
            f"no chunk was recorded as failed, so this proves nothing: {result}"
        )
        n_chunks = len(counter["chunks"])
        assert n_chunks > 1, "only one chunk ran; there is no failed one to check"
        assert len(calls) == n_chunks, (
            f"optimize_memory ran {len(calls)} times for {n_chunks} chunks "
            f"(one of which failed); the failed chunk skipped its cleanup"
        )

    def test_the_downsampled_block_is_not_kept_alive_by_the_failure(
        self, hf_run, caplog
    ):
        """The assertion that actually looks at the arrays.

        A weakref to the block the failing chunk downsampled must be dead once
        that chunk is over. If cleanup is skipped, the name still refers to it.

        ``caplog.clear()`` is load-bearing and took a while to find.
        ``record_chunk_failure`` calls ``logger.exception``, which attaches
        ``exc_info`` -- and therefore the traceback, the frame and every local
        in it -- to the LogRecord. pytest's logging plugin keeps those records
        for the duration of the test, so the block stayed reachable through the
        captured record and this test failed against correct code. Verified by
        running it under ``-p no:logging``, where it passes either way.

        Production does not have that problem: ``log_utils.logging_config``
        installs only a StreamHandler and a RotatingFileHandler, and neither
        retains records.
        """
        import gc
        import weakref

        from src.core import data_flow_manager as dfm

        fits_path, save_dir, counter, monkeypatch = hf_run
        self._fail_one_chunk(monkeypatch, counter, fail_on_chunk=1)

        refs: list = []
        real_downsample = dfm.downsample_chunk

        def _tracking_downsample(block):
            out, dt = real_downsample(block)
            refs.append(weakref.ref(out))
            return out, dt

        # The driver imports this from data_flow_manager at call time.
        monkeypatch.setattr(dfm, "downsample_chunk", _tracking_downsample)

        _run(fits_path, save_dir)

        # Drop every captured LogRecord before looking: see the docstring.
        # pytest attaches more than one record-retaining handler to the root
        # logger, so caplog.clear() alone leaves the traceback reachable.
        import logging

        caplog.clear()
        for handler in list(logging.getLogger().handlers):
            records = getattr(handler, "records", None)
            if isinstance(records, list):
                records.clear()
        gc.collect()

        assert refs, "downsample_chunk was never called"
        alive = [i for i, r in enumerate(refs) if r() is not None]
        assert not alive, (
            f"{len(alive)} downsampled block(s) still referenced after the run: "
            f"chunk indices {alive}. A failed chunk skipped its cleanup."
        )

    def test_a_chunk_that_fails_before_its_arrays_exist_is_still_just_one_chunk(
        self, hf_run
    ):
        """The case that decides the SHAPE of the fix, not just its presence.

        The failure above happens in the slice loop, by which point all three
        names are bound -- so freeing them after the handler, as the
        low-frequency driver does, would work there. It is a failure EARLIER in
        the chunk body that separates the two: with the names bound only inside
        the try, an unguarded ``del`` raises NameError while the real error is
        being handled, and that NameError escapes the chunk loop. One bad chunk
        becomes a dead file.

        Pre-binding the names to None is what makes the ``finally`` safe, and
        this is the test that says so. Verified against the alternative: the
        naive after-the-handler variant passes the test above and fails this
        one.
        """
        from src.core import data_flow_manager as dfm

        fits_path, save_dir, counter, monkeypatch = hf_run

        real_downsample = dfm.downsample_chunk
        seen = {"n": 0}

        def _fail_first_downsample(block):
            seen["n"] += 1
            if seen["n"] == 2:            # chunk index 1, before anything binds
                raise RuntimeError("simulated failure before block_ds exists")
            return real_downsample(block)

        monkeypatch.setattr(dfm, "downsample_chunk", _fail_first_downsample)

        result = _run(fits_path, save_dir)

        assert seen["n"] > 2, "the run stopped at the failing chunk"
        assert result["failed_chunks"] == 1, (
            f"expected exactly one failed chunk, got {result['failed_chunks']}; "
            f"status={result['status']}"
        )
        assert result["status"].startswith("SUCCESS"), (
            f"one chunk failing early killed the whole file: {result['status']} "
            f"({result.get('error_details', '')})"
        )
        rows = _rows_on_disk(save_dir, fits_path.stem)
        assert rows, "the surviving chunks wrote nothing"


class TestTheChunkLimitDivergenceBetweenTheDrivers:
    """Audit HF defect 4, pinned as it is rather than fixed.

    ``plan_chunking`` takes ``max_chunk_limit``: the LF driver passes
    ``config.MAX_CHUNK_SAMPLES``, the HF driver passes ``None``. Changing that
    moves chunk boundaries on the path that processes real observations, and
    there is no golden baseline for HF output, so it is a decision for the
    project rather than a cleanup. This records exactly what the difference is
    so it cannot drift while that decision is outstanding -- the same bargain
    ``test_fits_reader_characterization.py`` makes for D1-D7.

    The scope is much narrower than "HF ignores MAX_CHUNK_SAMPLES", which is
    how the audit's summary reads. Measured:

      * The cap's main application is in ``slice_len_calculator``, which both
        drivers go through, so the memory-safe chunk size IS capped for HF.
      * In ``plan_chunking`` the cap is only ever consulted when the file is
        SHORTER than the requested chunk. For any file longer than its chunk --
        every large observation -- both drivers take the same ``else`` branch
        and neither applies it.
      * So the two differ on exactly one shape of input: a file shorter than
        the requested chunk size but longer than the cap.
    """

    CAP = 1_000_000

    def test_they_agree_on_every_large_file(self):
        """The case that matters operationally: both drivers, same geometry."""
        from src.core.file_driver import plan_chunking

        for total, chunk in [
            (50_000_000, 2_000_000),
            (2_000_000_000, 4_000_000),
            (10_000_000, 1_000_000),
        ]:
            lf = plan_chunking(total, chunk, self.CAP)
            hf = plan_chunking(total, chunk, None)
            assert lf == hf, (
                f"total={total} chunk={chunk}: LF {lf} vs HF {hf}. The drivers "
                "were believed to agree on every file longer than its chunk."
            )

    def test_they_agree_when_the_file_is_under_both(self):
        from src.core.file_driver import plan_chunking

        lf = plan_chunking(500_000, 2_000_000, self.CAP)
        hf = plan_chunking(500_000, 2_000_000, None)
        assert lf == hf == (500_000, 1)

    def test_the_one_input_shape_where_they_differ(self):
        """PINNED AS-IS. A file shorter than its chunk but longer than the cap.

        LF splits it; HF runs it whole. If this ever starts agreeing, the
        divergence was closed -- which is a real decision and wants a commit
        message saying so, not a silently updated test.
        """
        from src.core.file_driver import plan_chunking

        total, chunk = 1_500_000, 2_000_000
        assert plan_chunking(total, chunk, self.CAP) == (1_000_000, 2)
        assert plan_chunking(total, chunk, None) == (1_500_000, 1)

    def test_the_hf_driver_really_is_the_one_passing_none(self, hf_run):
        """Ties the arithmetic above to the driver that uses it.

        Observed by watching the real run call ``plan_chunking``, not by reading
        the source for ``max_chunk_limit=None`` -- counting source text is what
        audit REF-03 spent two refactors trapped behind.
        """
        from src.core import file_driver

        fits_path, save_dir, counter, monkeypatch = hf_run
        seen = []
        real = file_driver.plan_chunking

        def _spy(total_samples, chunk_samples, max_chunk_limit=None):
            seen.append(max_chunk_limit)
            return real(total_samples, chunk_samples, max_chunk_limit)

        monkeypatch.setattr(file_driver, "plan_chunking", _spy)
        _run(fits_path, save_dir)

        assert seen, "plan_chunking was never called"
        assert seen == [None] * len(seen), (
            f"the HF driver passed {seen} as max_chunk_limit; if the cap is now "
            "applied, this class describes behaviour that no longer exists"
        )
