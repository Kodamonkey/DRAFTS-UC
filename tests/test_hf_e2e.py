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
