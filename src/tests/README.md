# Legacy analysis scripts

Pytest suite lives in `/tests` at project root (`pyproject.toml` → `testpaths = ["tests"]`).
Do not recreate a junction from `tests/` → `src/tests/` (breaks collection).

This folder keeps offline analysis/simulation scripts only (`analisis_chunk_reduccion.py`, `simulate_5_5tb_processing.py`).
