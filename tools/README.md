# `tools/` — standalone scripts

Everything here is run **by hand**, from the repository root. None of it is
imported by the pipeline, none of it is an entry point, and none of it ships in
the Docker images — `main.py` is the only way to run a search.

This directory is audit item 42. The scripts used to live in `src/scripts/`
(22 files) and `src/tests/` (3), which put non-library code inside the package
directory and misled on both counts: `src/tests/` contained no tests, and
`src/scripts/test_slicing_alignment.py` was collected by pytest as if it
contained some — it errored on two missing fixtures, so `pytest src/` was broken
and nobody noticed, because nothing runs `pytest src/`. That file is now
`diagnostics/check_slicing_alignment.py`: same script, name that does not lie.

## Layout

| Directory | What is in it |
|---|---|
| `validation/` | Compares pipeline output against published values for known sources. `_case_common.py` and `_matching_common.py` are shared by their siblings, so this stays one directory. |
| `diagnostics/` | One-off inspection: FITS headers, MJD arithmetic, chunk/slice alignment, segment plots. |
| `thesis/` | Generates the tables and reports for the thesis. Re-run only if the thesis is being regenerated. |
| `analysis/` | Memory and chunking studies. See `analysis/README.md`. |
| `legacy/` | `time_series_with_polarization_dos.py`. The audit recommends deleting it; it is parked here rather than deleted because that is the project owner's call, not a refactor's. |

## Running them

From the repository root, so that `import src.<package>` resolves:

```bash
python tools/validation/analyze_case_b0355.py --summary
python tools/diagnostics/fits_header_analyzer.py <file.fits>
```

Scripts that import from the project add the repository root to `sys.path`
themselves, derived from `__file__`. They do not assume the working directory,
but they do assume the repository layout — `tools/<group>/<script>.py`, two
levels below the root. Moving a script one level up or down without adjusting
its path derivation will break it silently, by resolving the root to the wrong
directory rather than by raising.

## What they are NOT covered by

- **No tests.** These scripts have none, and the pytest suite does not collect
  them. `pyproject.toml` pins `testpaths = ["tests"]`.
- **Lint only.** CI runs `ruff check --select F821,F811,F601` over `tools`, the
  same narrow set of real errors it runs over `src`. That catches an undefined
  name; it does not catch a wrong number.
- **Four hardcoded `K_DM` literals** remain in `diagnostics/MJD.py`,
  `analysis/simulate_5_5tb_processing.py` and
  `analysis/analisis_chunk_reduccion.py`, instead of importing `K_DM_MS` from
  `src/domain/physics.py` (SPEC-DM-001). They predate the move.
  `tests/test_spec_constants.py` names those three files explicitly, so the
  guard still fails on a **new** one anywhere else under `tools/`.
