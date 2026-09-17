"""A characterisation test that pins the candidate CSV, cell by cell.

This is the safety net the risky refactors need. REF-01 unifies the duplicated
low- and high-frequency file drivers, REF-03 splits the two reader
implementations that share one 1046-line try block, and REF-05 decomposes an
884-line function. None of those is safe without something that says "the output
is byte-for-byte what it was before you started".

What this test can and cannot promise
-------------------------------------
It pins the output **within one environment**, which is exactly what a refactor
needs: run before, run after, compare. It deliberately does NOT claim the bytes
are portable across machines, because they are not, and pretending otherwise
would produce a test that fails for reasons having nothing to do with the code:

* ``np.convolve`` in the SNR profile accumulates in float32 through whatever
  BLAS kernel the CPU has, so the last digits follow the CPU generation.
* The dedisperser has four kernels -- Numba CPU, pure NumPy, Torch GPU,
  Numba CUDA -- that sum channels in different orders, and ``fastmath=True``
  permits reassociation even within one of them.
* The chunk geometry is derived from ``psutil.virtual_memory().available``, so
  a busy machine can produce different chunk and slice indices.
* The barycentric columns need astropy plus its ephemeris and IERS tables.

So there are two layers. ``test_two_runs_are_byte_identical`` needs no stored
file and holds everywhere. The comparison against the stored baseline pins every
column exactly except a short, named list compared numerically instead -- and
each name on that list has a reason next to it.

Regenerating the baseline
-------------------------
``DRAFTS_UPDATE_GOLDEN=1 python -m pytest tests/test_golden_csv.py``

Do that when a change is *meant* to alter the output, and say so in the commit
message along with what moved. A silently regenerated baseline is worse than no
baseline at all.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from src.config import config
from src.core import detection_engine as de
from src.output.candidate_manager import CANDIDATE_HEADER, CandidateWriter
from tests.synthetic_filterbank import write_filterbank
from tests.test_e2e_pipeline import DM_TRUE, TSAMP, _configure, _install_peak_detector

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
GOLDEN_LF = GOLDEN_DIR / "lf_candidates.csv"

NSAMPLES = 8000
BURST_T = 1.5
CHUNK_SAMPLES = 4000

#: Columns whose exact digits depend on the machine rather than on the code.
#: Compared numerically with a tolerance instead of byte-for-byte. Keep this
#: list short and keep the reasons attached: every name here is a column the
#: golden file has stopped protecting.
FLOAT_TOLERANT = {
    # float32 accumulation in np.convolve, through the platform BLAS.
    "snr_waterfall": 1e-2,
    "snr_patch_dedispersed": 1e-2,
    "snr_waterfall_linear": 1e-2,
    "snr_patch_dedispersed_linear": 1e-2,
    "snr_pre_dedisp": 1e-2,
    "snr_post_dedisp": 1e-2,
    "post_trials_sigma": 1e-2,
    "width_ms": 1e-3,
    "best_width_ms": 1e-3,
    "physical_score": 1e-4,
    "rank_score": 1e-4,
    "linear_fraction": 1e-4,
    # Derived from the SNR profile above.
    "t_sec_waterfall": 1e-6,
    # The classifier stub is a logistic of the SNR peak, so it inherits it.
    "class_prob_intensity": 1e-3,
    "class_prob_linear": 1e-3,
}

#: Columns that require astropy, its site registry and its ephemeris. Compared
#: only when the baseline and the current run agree that the correction ran.
BARYCENTRIC = [
    "mjd_bary_utc", "mjd_bary_tdb", "mjd_bary_utc_inf", "mjd_bary_tdb_inf",
]


def _pin_every_config_key_the_csv_depends_on(tmp_path: Path) -> None:
    """_configure covers most of it; these are the ones it leaves to config.yaml.

    Without them the golden breaks whenever someone edits config.yaml, which
    would teach everyone to regenerate it without looking.
    """
    _configure(tmp_path, chunk_samples=CHUNK_SAMPLES)
    config.PREWHITEN_BEFORE_DM = False
    config.TEMPORAL_DOWNSAMPLING_MODE = "sum"
    config.DETECTION_WIDTHS_MS = []
    config.TRIAL_CORRECTION = "none"
    config.HIGH_FREQ_DM_POLICY = "unresolved"
    config.DM_GRID_MODE = "legacy_uniform"
    config.MAX_DM_SMEARING_MS = "auto"
    config.SOURCE_RA = "05:31:58.70"
    config.SOURCE_DEC = "33:08:52.5"
    config.REF_FREQ_MHZ = 1400.0
    config.OBSERVATORY = "Effelsberg"
    # The bundled ephemeris: no download, no network, same answer everywhere.
    config.EPHEMERIS = "builtin"


def _run_once(tmp_path: Path, monkeypatch) -> str:
    """Run the real pipeline over the synthetic file; return the CSV text."""
    from src.core import pipeline as pipeline_mod
    from src.core.pipeline import run_pipeline

    monkeypatch.setattr(pipeline_mod, "_load_detection_model", lambda: None)
    monkeypatch.setattr(pipeline_mod, "_load_class_model", lambda: None)
    monkeypatch.setattr(de, "save_all_plots", lambda *a, **k: None)

    _pin_every_config_key_the_csv_depends_on(tmp_path)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    fil = config.DATA_DIR / "synthetic_burst.fil"
    write_filterbank(
        fil, nsamples=NSAMPLES, dm=DM_TRUE, burst_time_s=BURST_T, tsamp=TSAMP,
    )
    _install_peak_detector(monkeypatch)

    run_pipeline()
    # The writer buffers 50 rows; without this the file can be short.
    CandidateWriter.flush_all()

    csv_path = config.RESULTS_DIR / "Summary" / "synthetic_burst" / "synthetic_burst.candidates.csv"
    assert csv_path.exists(), f"no candidate CSV at {csv_path}"
    return csv_path.read_text(encoding="utf-8")


def _rows(text: str) -> list[dict]:
    return list(csv.DictReader(text.splitlines()))


class TestTheOutputIsReproducible:
    def test_two_runs_are_byte_identical(self, tmp_path, monkeypatch):
        """No stored file involved, so this one holds on any machine.

        If it ever fails, something in the pipeline has become
        non-deterministic -- an unseeded RNG, a set or dict iteration reaching
        the output, a wall-clock value in a column -- and the golden comparison
        below is meaningless until that is found.
        """
        first = _run_once(tmp_path / "a", monkeypatch)
        second = _run_once(tmp_path / "b", monkeypatch)
        assert first == second

    def test_the_run_produces_rows(self, tmp_path, monkeypatch):
        text = _run_once(tmp_path, monkeypatch)
        rows = _rows(text)
        assert rows, "the golden comparison is vacuous without rows"
        assert len(rows) >= 1


class TestTheOutputMatchesTheStoredBaseline:
    @pytest.fixture
    def current(self, tmp_path, monkeypatch) -> str:
        text = _run_once(tmp_path, monkeypatch)
        if os.environ.get("DRAFTS_UPDATE_GOLDEN"):
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            GOLDEN_LF.write_text(text, encoding="utf-8", newline="")
            pytest.skip(f"baseline regenerated at {GOLDEN_LF}")
        if not GOLDEN_LF.exists():
            pytest.skip(
                f"no baseline at {GOLDEN_LF}; create it with DRAFTS_UPDATE_GOLDEN=1"
            )
        return text

    def test_the_header_is_unchanged(self, current):
        """The schema is a contract with every downstream consumer."""
        golden = GOLDEN_LF.read_text(encoding="utf-8")
        assert _rows(current) is not None
        current_header = current.splitlines()[0].split(",")
        golden_header = golden.splitlines()[0].split(",")
        assert current_header == golden_header
        assert current_header == CANDIDATE_HEADER, (
            "the file on disk and CANDIDATE_HEADER have diverged"
        )

    def test_the_row_count_is_unchanged(self, current):
        """A refactor that finds one more or one fewer candidate is not a
        refactor. This is the assertion REF-01 most needs."""
        assert len(_rows(current)) == len(_rows(GOLDEN_LF.read_text(encoding="utf-8")))

    def test_every_cell_matches(self, current):
        golden_rows = _rows(GOLDEN_LF.read_text(encoding="utf-8"))
        current_rows = _rows(current)

        bary_comparable = all(
            row.get("mjd_bary_status") == cur.get("mjd_bary_status") == "ok"
            for row, cur in zip(golden_rows, current_rows)
        )

        mismatches: list[str] = []
        for i, (want, got) in enumerate(zip(golden_rows, current_rows)):
            for column in CANDIDATE_HEADER:
                a, b = want.get(column, ""), got.get(column, "")
                if column in BARYCENTRIC and not bary_comparable:
                    continue
                if a == b:
                    continue
                tol = FLOAT_TOLERANT.get(column)
                if tol is not None and a and b:
                    try:
                        if abs(float(a) - float(b)) <= tol:
                            continue
                    except ValueError:
                        pass
                mismatches.append(f"row {i}, {column}: baseline {a!r} != now {b!r}")

        assert not mismatches, (
            "the candidate CSV changed:\n  " + "\n  ".join(mismatches[:20])
            + "\n\nIf the change was intended, regenerate with "
              "DRAFTS_UPDATE_GOLDEN=1 and say in the commit message what moved."
        )

    def test_the_tolerant_columns_are_still_close(self, current):
        """A tolerance that silently absorbs a real change is worse than none.

        This asserts the tolerant columns are within their tolerance, which is
        the part the byte comparison above skips for them.
        """
        golden_rows = _rows(GOLDEN_LF.read_text(encoding="utf-8"))
        current_rows = _rows(current)
        for i, (want, got) in enumerate(zip(golden_rows, current_rows)):
            for column, tol in FLOAT_TOLERANT.items():
                a, b = want.get(column, ""), got.get(column, "")
                if not a or not b:
                    continue
                assert float(a) == pytest.approx(float(b), abs=tol), (
                    f"row {i}, {column} moved beyond its tolerance"
                )
