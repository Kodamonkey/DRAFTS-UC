# SPEC: Física del pipeline FRB (SDD)

Contratos físicos verificables. Cada cambio crítico referencia un SPEC-ID.
Tests: `tests/test_scientific_physics.py`, `tests/test_dedispersion_parity.py`,
`tests/test_downsampler.py`, `tests/test_audit_fixes.py`, `tests/test_contracts.py`.

## SPEC-DM-001 — Constante de dispersión única
Fuente única `K_DM_MS = 4.148808e3` en `src/analysis/science_metrics.py`.
Re-exportada en `src/domain/physics.py`; todos los módulos del pipeline importan desde allí.
Delay (s) = `K_DM_MS * DM * (nu^-2)` con `nu` en MHz (PRESTO `delay_from_dm`).
Prohibido redefinir literales `4.148808e3`, `4.1488e3`, `4.15e3` fuera de `science_metrics.py`.
Enforcement: `tests/test_spec_constants.py::test_no_hardcoded_kdm_literals` escanea `src/`.
Verifica: `test_dedispersion_parity` (parity CPU/GPU), `test_spec_constants`.

## SPEC-DM-002 — Delay dispersivo monotónico
Para DM>0, menor frecuencia ⇒ mayor delay.
`dispersion_delay_ms(dm, f_lo, f_hi) >= 0` y crece al bajar `f_lo`.
Verifica: `test_audit_fixes.test_delay_is_monotonic_in_frequency`.

## SPEC-DM-003 — Dedispersión maximiza SNR en DM correcto
Señal sintética dispersada a `dm_true` ⇒ el cubo DM-tiempo presenta máximo de SNR
en la fila DM ≈ `dm_true` (±2 pc cm⁻³).
Verifica: `test_scientific_physics.TestDedispersionPhysics`.

## SPEC-DM-004 — Grid DM smear-limited
Paso DM depende de `freq_low`, `freq_high`, `TIME_RESO`, `DOWN_TIME_RATE`,
`MAX_DM_SMEARING_MS`. Modo `legacy_uniform` = paso 1 (reproducibilidad).
`calculate_dm_values` nunca produce DM > `DM_max`.
Verifica: `test_scientific_physics.TestAdaptiveDMGrid`.

## SPEC-DM-005 — Mapeo caja CNN → DM sin off-by-one
`extract_candidate_dm(px, py, slice_len)` mapea fila de imagen a DM con
`dm = DM_min + (py / (H-1)) * (DM_max - DM_min)`; el resultado nunca excede `DM_max`.
Alineado con `_dm_from_image_at_time` (HF). Imagen no fijada a 512 (`img_height` param).
Verifica: `test_audit_fixes.test_extract_candidate_dm_*`.

## SPEC-FREQ-001 — Eje de frecuencia ascendente, una sola inversión
`normalize_frequency_axis` devuelve eje ascendente y un flag `needs_reversal`.
Invertir datos SÓLO si el orden original es descendente. Nunca doble inversión.
Kernel GPU: la frecuencia de referencia se computa como `float(freq.max())**-2` en el
host y se pasa como escalar `f_ref_inv2`; no se usa `freq[-1]` (asunción posicional).
Verifica: `test_scientific_physics.TestFrequencyAxisPhysics`,
          `test_dedispersion_parity.TestDedispersionParity.test_gpu_freq_ref_uses_max`.

## SPEC-HF-001 — Decisión LF/HF por colapso bow-tie
HF se activa cuando `Δt_disp / Δt_res < collapse_ratio` (no por frecuencia central fija).
`Δt_disp = dispersion_delay_ms(DM_max, f_lo, f_hi)`, `Δt_res = TIME_RESO*DOWN_TIME_RATE*1000`.
Verifica: `test_scientific_physics.TestBowtieCollapse`.

## SPEC-POL-001 — Polarización lineal con debias
`L = sqrt(Q^2 + U^2)` con remoción opcional de sesgo de ruido de 1er orden.
El debias reduce la mediana del ruido y preserva pulsos reales.
Verifica: `test_scientific_physics.TestPolarizationPhysics`.

## SPEC-PRE-001 — Prewhitening explícito y opcional
`PREWHITEN_BEFORE_DM` se expone en `config.yaml` (`preprocessing.prewhiten_before_dm`).
Default científico `false` (no alterar la física del cubo DM sin intención).
Verifica: `test_audit_fixes.test_prewhiten_*`.

## SPEC-IO-001 — Metadata por archivo
`extract_parameters_auto` se ejecuta por cada archivo procesado, no una vez por target.
Archivos heterogéneos no heredan `TIME_RESO/FREQ/FILE_LENG` de otro archivo.
Verifica: `test_audit_fixes.test_parameters_extracted_per_file`.

## SPEC-IO-002 — Errores por chunk no se tragan
Si fallan chunks, el archivo reporta `PARTIAL` (no `SUCCESS`) y registra el conteo.
Verifica: `test_audit_fixes.test_chunk_failure_marks_partial`.

## SPEC-MEM-001 — Pico de memoria acotado en cubo DM
Durante DM chunking, pico de cómputo ≤ tamaño de una ventana DM (`3 × dm_chunk_height × width × 4` bytes).
Cubos grandes (≥ `DM_CUBE_MEMMAP_THRESHOLD_GB`, default 4 GB) usan `np.memmap` para el buffer resultado.
Paridad numérica con alloc directa debe mantenerse.
Verifica: `test_cube_windowed_parity`, `test_memmap_cube_parity`.

## SPEC-CAND-001 — Salida de candidatos unificada
Ambas rutas de detección (CenterNet LF y SNR-peak HF) usan `finalize_patch()`
para producir (proc_patch, class_prob, snr_val, peak_idx_patch, width_ms, start_sample).
El CSV de candidatos debe ser byte-idéntico antes/después de la refactorización.
Verifica: `tests/test_candidate_finalizer.py`.

## SPEC-PURE-001 — Dedispersión sin mutar config global
`d_dm_time_g` no muta `config.FREQ` ni `config.FREQ_RESO`; usa cómputo local puro.
Verifica: `test_audit_fixes.test_dedispersion_does_not_mutate_config`.
