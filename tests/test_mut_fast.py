"""Fast, import-light unit tests used as the mutation-testing oracle (Etapa 6).

These avoid importing numba/torch-heavy modules so cosmic-ray can run one pytest
process per mutant cheaply. They pin down the behaviour of the architecture- and
physics-critical pure functions so surviving mutants reveal weak coverage.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.analysis.science_metrics import (
    K_DM_MS,
    dispersion_delay_ms,
    dm_step_for_smearing,
    estimate_dm_uncertainty,
    physical_consistency_score,
    post_trials_sigma,
)
from src.core.pipeline_parameters import (
    calculate_absolute_slice_time,
    calculate_dm_values,
    calculate_frequency_downsampled,
    calculate_overlap_decimated,
    calculate_time_slice,
    calculate_width_total,
    should_use_hf_pipeline,
)
from src.preprocessing.dm_candidate_extractor import extract_candidate_dm
from src.input.utils import normalize_frequency_axis, safe_float, safe_int


class TestScienceMetrics(unittest.TestCase):
    def test_k_dm_constant(self):
        self.assertAlmostEqual(K_DM_MS, 4.148808e3, places=2)

    def test_delay_positive_and_monotonic(self):
        self.assertGreater(dispersion_delay_ms(100.0, 400.0, 800.0), 0.0)
        self.assertGreater(
            dispersion_delay_ms(100.0, 400.0, 800.0),
            dispersion_delay_ms(100.0, 700.0, 800.0),
        )

    def test_delay_scales_with_dm(self):
        d1 = dispersion_delay_ms(100.0, 400.0, 800.0)
        d2 = dispersion_delay_ms(200.0, 400.0, 800.0)
        self.assertAlmostEqual(d2 / d1, 2.0, places=5)

    def test_delay_zero_for_bad_freq(self):
        self.assertEqual(dispersion_delay_ms(100.0, 0.0, 800.0), 0.0)
        self.assertEqual(dispersion_delay_ms(100.0, -1.0, 800.0), 0.0)

    def test_dm_step_for_smearing_positive(self):
        step = dm_step_for_smearing(1.0, 400.0, 800.0)
        self.assertGreater(step, 0.0)
        # Larger allowed smearing => larger DM step.
        self.assertGreater(dm_step_for_smearing(2.0, 400.0, 800.0), step)

    def test_dm_step_degenerate_band(self):
        self.assertEqual(dm_step_for_smearing(1.0, 400.0, 400.0), 1.0)

    def test_estimate_dm_uncertainty(self):
        vals = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.assertEqual(estimate_dm_uncertainty(vals, 2), 1.0)
        self.assertEqual(estimate_dm_uncertainty(vals, 0), 0.5)
        self.assertEqual(estimate_dm_uncertainty(vals, 4), 0.5)
        self.assertEqual(estimate_dm_uncertainty([1.0], 0), 0.0)

    def test_post_trials_sigma_penalizes(self):
        self.assertLess(post_trials_sigma(8.0, 10_000), 8.0)
        self.assertEqual(post_trials_sigma(8.0, 1), 8.0)
        self.assertEqual(post_trials_sigma(8.0, 10_000, mode="none"), 8.0)

    def test_post_trials_sigma_nonnegative(self):
        self.assertGreaterEqual(post_trials_sigma(0.5, 10_000), 0.0)

    def test_physical_consistency_score_bounds(self):
        s = physical_consistency_score(8.0, 1.0, 4.0, "measured", 0.5)
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)
        # Unresolved status penalises the score.
        s_full = physical_consistency_score(8.0, None, None, "measured")
        s_unres = physical_consistency_score(8.0, None, None, "unresolved_high_freq")
        self.assertLess(s_unres, s_full)

    def test_consistency_base_division(self):
        # base = clamp(snr/8). Exact values pin the /8.0 divisor.
        self.assertAlmostEqual(physical_consistency_score(4.0, None, None, "ok"), 0.5, places=9)
        self.assertAlmostEqual(physical_consistency_score(8.0, None, None, "ok"), 1.0, places=9)
        self.assertAlmostEqual(physical_consistency_score(0.0, None, None, "ok"), 0.0, places=9)
        self.assertAlmostEqual(physical_consistency_score(16.0, None, None, "ok"), 1.0, places=9)

    def test_consistency_gain_factor(self):
        # gain = post/pre; factor = clamp(gain, 0.5, 1.25)/1.25 applied to base.
        self.assertAlmostEqual(physical_consistency_score(8.0, 4.0, 4.0, "ok"), 0.8, places=6)
        self.assertAlmostEqual(physical_consistency_score(8.0, 4.0, 2.0, "ok"), 0.4, places=6)
        self.assertAlmostEqual(physical_consistency_score(8.0, 4.0, 8.0, "ok"), 1.0, places=6)
        # pre <= 0 disables the gain correction entirely.
        self.assertAlmostEqual(physical_consistency_score(8.0, 0.0, 8.0, "ok"), 1.0, places=6)

    def test_consistency_unresolved_penalty(self):
        # Exactly 0.85 multiplier, no gain branch (pre/post None).
        self.assertAlmostEqual(
            physical_consistency_score(8.0, None, None, "unresolved_high_freq"), 0.85, places=6
        )

    def test_consistency_linear_fraction_factor(self):
        # factor = clamp(0.9 + 0.2*min(|lf|,1), 0.8, 1.1). base = 0.5 (snr=4).
        self.assertAlmostEqual(physical_consistency_score(4.0, None, None, "ok", 1.0), 0.55, places=6)
        self.assertAlmostEqual(physical_consistency_score(4.0, None, None, "ok", 0.5), 0.5, places=6)
        self.assertAlmostEqual(physical_consistency_score(4.0, None, None, "ok", 0.0), 0.45, places=6)

    def test_consistency_base_division(self):
        # base = clamp(snr/8). Exact values pin the /8.0 divisor.
        self.assertAlmostEqual(physical_consistency_score(4.0, None, None, "ok"), 0.5, places=9)
        self.assertAlmostEqual(physical_consistency_score(8.0, None, None, "ok"), 1.0, places=9)
        self.assertAlmostEqual(physical_consistency_score(0.0, None, None, "ok"), 0.0, places=9)
        self.assertAlmostEqual(physical_consistency_score(16.0, None, None, "ok"), 1.0, places=9)

    def test_consistency_gain_factor(self):
        # gain = post/pre; factor = clamp(gain, 0.5, 1.25)/1.25 applied to base.
        self.assertAlmostEqual(physical_consistency_score(8.0, 4.0, 4.0, "ok"), 0.8, places=6)
        self.assertAlmostEqual(physical_consistency_score(8.0, 4.0, 2.0, "ok"), 0.4, places=6)
        self.assertAlmostEqual(physical_consistency_score(8.0, 4.0, 8.0, "ok"), 1.0, places=6)
        # pre <= 0 disables the gain correction entirely.
        self.assertAlmostEqual(physical_consistency_score(8.0, 0.0, 8.0, "ok"), 1.0, places=6)

    def test_consistency_unresolved_penalty(self):
        # Exactly 0.85 multiplier, no gain branch (pre/post None).
        self.assertAlmostEqual(
            physical_consistency_score(8.0, None, None, "unresolved_high_freq"), 0.85, places=6
        )

    def test_consistency_linear_fraction_factor(self):
        # factor = clamp(0.9 + 0.2*min(|lf|,1), 0.8, 1.1). base = 0.5 (snr=4).
        self.assertAlmostEqual(physical_consistency_score(4.0, None, None, "ok", 1.0), 0.55, places=6)
        self.assertAlmostEqual(physical_consistency_score(4.0, None, None, "ok", 0.5), 0.5, places=6)
        self.assertAlmostEqual(physical_consistency_score(4.0, None, None, "ok", 0.0), 0.45, places=6)


class TestPipelineParameters(unittest.TestCase):
    def test_legacy_uniform_grid(self):
        old = (config.DM_min, config.DM_max, config.DM_GRID_MODE)
        try:
            config.DM_min, config.DM_max, config.DM_GRID_MODE = 0, 100, "legacy_uniform"
            vals = calculate_dm_values()
            self.assertEqual(vals.size, 101)
            self.assertAlmostEqual(float(vals[0]), 0.0)
            self.assertAlmostEqual(float(vals[-1]), 100.0)
            self.assertLessEqual(float(vals.max()), 100.0)
        finally:
            (config.DM_min, config.DM_max, config.DM_GRID_MODE) = old

    def test_empty_grid_when_inverted(self):
        old = (config.DM_min, config.DM_max, config.DM_GRID_MODE)
        try:
            config.DM_min, config.DM_max, config.DM_GRID_MODE = 100, 0, "legacy_uniform"
            self.assertEqual(calculate_dm_values().size, 0)
        finally:
            (config.DM_min, config.DM_max, config.DM_GRID_MODE) = old

    def test_smear_limited_never_exceeds_dm_max(self):
        old = (config.DM_min, config.DM_max, config.DM_GRID_MODE, config.FREQ,
               config.TIME_RESO, config.DOWN_TIME_RATE, config.MAX_DM_SMEARING_MS)
        try:
            config.DM_min, config.DM_max = 0, 100
            config.DM_GRID_MODE = "smear_limited"
            config.FREQ = np.linspace(400.0, 800.0, 64)
            config.TIME_RESO = 1e-3
            config.DOWN_TIME_RATE = 1
            config.MAX_DM_SMEARING_MS = 1.0
            vals = calculate_dm_values()
            self.assertGreater(vals.size, 1)
            self.assertLessEqual(float(vals.max()), 100.0 + 1e-6)
        finally:
            (config.DM_min, config.DM_max, config.DM_GRID_MODE, config.FREQ,
             config.TIME_RESO, config.DOWN_TIME_RATE, config.MAX_DM_SMEARING_MS) = old

    def test_smear_limited_step_matches_spec(self):
        old = (config.DM_min, config.DM_max, config.DM_GRID_MODE, config.FREQ,
               config.TIME_RESO, config.DOWN_TIME_RATE, config.MAX_DM_SMEARING_MS)
        try:
            config.FREQ = np.array([400.0, 800.0])
            config.TIME_RESO, config.DOWN_TIME_RATE = 1e-3, 1
            config.MAX_DM_SMEARING_MS = 1.0
            config.DM_GRID_MODE = "smear_limited"
            step = dm_step_for_smearing(1.0, 400.0, 800.0)
            # DM_max chosen as an exact multiple of step so no extra append happens.
            config.DM_min, config.DM_max = 0.0, float(step) * 5.0
            vals = calculate_dm_values()
            self.assertEqual(vals.size, 6)
            np.testing.assert_allclose(np.diff(vals), step, rtol=1e-4)
            self.assertAlmostEqual(float(vals[0]), 0.0, places=4)
            self.assertLessEqual(float(vals[-1]), config.DM_max + 1e-4)
        finally:
            (config.DM_min, config.DM_max, config.DM_GRID_MODE, config.FREQ,
             config.TIME_RESO, config.DOWN_TIME_RATE, config.MAX_DM_SMEARING_MS) = old

    def test_smear_limited_auto_scales_with_down_rate(self):
        old = (config.DM_min, config.DM_max, config.DM_GRID_MODE, config.FREQ,
               config.TIME_RESO, config.DOWN_TIME_RATE, config.MAX_DM_SMEARING_MS)
        try:
            config.FREQ = np.array([400.0, 800.0])
            config.TIME_RESO = 1e-3
            config.MAX_DM_SMEARING_MS = "auto"
            config.DM_GRID_MODE = "smear_limited"
            config.DM_min, config.DM_max = 0.0, 200.0
            config.DOWN_TIME_RATE = 1
            n_fine = calculate_dm_values().size
            config.DOWN_TIME_RATE = 4
            n_coarse = calculate_dm_values().size
            # Higher decimation => larger allowed smear => bigger step => fewer trials.
            self.assertLess(n_coarse, n_fine)
        finally:
            (config.DM_min, config.DM_max, config.DM_GRID_MODE, config.FREQ,
             config.TIME_RESO, config.DOWN_TIME_RATE, config.MAX_DM_SMEARING_MS) = old

    def test_smear_limited_appends_dm_max(self):
        old = (config.DM_min, config.DM_max, config.DM_GRID_MODE, config.FREQ,
               config.TIME_RESO, config.DOWN_TIME_RATE, config.MAX_DM_SMEARING_MS)
        try:
            config.FREQ = np.array([400.0, 800.0])
            config.TIME_RESO, config.DOWN_TIME_RATE = 1e-3, 1
            config.MAX_DM_SMEARING_MS = 1.0
            config.DM_GRID_MODE = "smear_limited"
            step = dm_step_for_smearing(1.0, 400.0, 800.0)
            # Not a multiple of step => last grid point must be appended at exactly DM_max.
            config.DM_min, config.DM_max = 0.0, float(step) * 4.5
            vals = calculate_dm_values()
            self.assertAlmostEqual(float(vals[-1]), float(config.DM_max), places=4)
        finally:
            (config.DM_min, config.DM_max, config.DM_GRID_MODE, config.FREQ,
             config.TIME_RESO, config.DOWN_TIME_RATE, config.MAX_DM_SMEARING_MS) = old

    def test_absolute_slice_time(self):
        self.assertAlmostEqual(calculate_absolute_slice_time(10.0, 5, 0.1), 10.5, places=9)
        self.assertAlmostEqual(calculate_absolute_slice_time(0.0, 0, 0.1), 0.0, places=9)
        self.assertAlmostEqual(calculate_absolute_slice_time(2.0, 3, 0.0), 2.0, places=9)

    def test_smear_limited_step_matches_spec(self):
        old = (config.DM_min, config.DM_max, config.DM_GRID_MODE, config.FREQ,
               config.TIME_RESO, config.DOWN_TIME_RATE, config.MAX_DM_SMEARING_MS)
        try:
            config.FREQ = np.array([400.0, 800.0])
            config.TIME_RESO, config.DOWN_TIME_RATE = 1e-3, 1
            config.MAX_DM_SMEARING_MS = 1.0
            config.DM_GRID_MODE = "smear_limited"
            step = dm_step_for_smearing(1.0, 400.0, 800.0)
            # DM_max chosen as an exact multiple of step so no extra append happens.
            config.DM_min, config.DM_max = 0.0, float(step) * 5.0
            vals = calculate_dm_values()
            self.assertEqual(vals.size, 6)
            np.testing.assert_allclose(np.diff(vals), step, rtol=1e-4)
            self.assertAlmostEqual(float(vals[0]), 0.0, places=4)
            self.assertLessEqual(float(vals[-1]), config.DM_max + 1e-4)
        finally:
            (config.DM_min, config.DM_max, config.DM_GRID_MODE, config.FREQ,
             config.TIME_RESO, config.DOWN_TIME_RATE, config.MAX_DM_SMEARING_MS) = old

    def test_smear_limited_auto_scales_with_down_rate(self):
        old = (config.DM_min, config.DM_max, config.DM_GRID_MODE, config.FREQ,
               config.TIME_RESO, config.DOWN_TIME_RATE, config.MAX_DM_SMEARING_MS)
        try:
            config.FREQ = np.array([400.0, 800.0])
            config.TIME_RESO = 1e-3
            config.MAX_DM_SMEARING_MS = "auto"
            config.DM_GRID_MODE = "smear_limited"
            config.DM_min, config.DM_max = 0.0, 200.0
            config.DOWN_TIME_RATE = 1
            n_fine = calculate_dm_values().size
            config.DOWN_TIME_RATE = 4
            n_coarse = calculate_dm_values().size
            # Higher decimation => larger allowed smear => bigger step => fewer trials.
            self.assertLess(n_coarse, n_fine)
        finally:
            (config.DM_min, config.DM_max, config.DM_GRID_MODE, config.FREQ,
             config.TIME_RESO, config.DOWN_TIME_RATE, config.MAX_DM_SMEARING_MS) = old

    def test_smear_limited_appends_dm_max(self):
        old = (config.DM_min, config.DM_max, config.DM_GRID_MODE, config.FREQ,
               config.TIME_RESO, config.DOWN_TIME_RATE, config.MAX_DM_SMEARING_MS)
        try:
            config.FREQ = np.array([400.0, 800.0])
            config.TIME_RESO, config.DOWN_TIME_RATE = 1e-3, 1
            config.MAX_DM_SMEARING_MS = 1.0
            config.DM_GRID_MODE = "smear_limited"
            step = dm_step_for_smearing(1.0, 400.0, 800.0)
            # Not a multiple of step => last grid point must be appended at exactly DM_max.
            config.DM_min, config.DM_max = 0.0, float(step) * 4.5
            vals = calculate_dm_values()
            self.assertAlmostEqual(float(vals[-1]), float(config.DM_max), places=4)
        finally:
            (config.DM_min, config.DM_max, config.DM_GRID_MODE, config.FREQ,
             config.TIME_RESO, config.DOWN_TIME_RATE, config.MAX_DM_SMEARING_MS) = old

    def test_absolute_slice_time(self):
        self.assertAlmostEqual(calculate_absolute_slice_time(10.0, 5, 0.1), 10.5, places=9)
        self.assertAlmostEqual(calculate_absolute_slice_time(0.0, 0, 0.1), 0.0, places=9)
        self.assertAlmostEqual(calculate_absolute_slice_time(2.0, 3, 0.0), 2.0, places=9)

    def test_should_use_hf_resolved_vs_collapsed(self):
        use_hf_lf, reason_lf = should_use_hf_pipeline(400.0, 800.0, 500.0, 5.12e-5, 1)
        self.assertFalse(use_hf_lf)
        self.assertIn("resolved", reason_lf)
        use_hf_hf, reason_hf = should_use_hf_pipeline(84000.0, 88000.0, 1000.0, 5.12e-5, 1)
        self.assertTrue(use_hf_hf)
        self.assertIn("collapsed", reason_hf)

    def test_should_use_hf_zero_time_reso(self):
        use_hf, _ = should_use_hf_pipeline(400.0, 800.0, 500.0, 0.0, 1)
        self.assertFalse(use_hf)

    def test_width_total(self):
        old = config.DOWN_TIME_RATE
        try:
            config.DOWN_TIME_RATE = 1
            self.assertEqual(calculate_width_total(1000), 1000)
            config.DOWN_TIME_RATE = 4
            self.assertEqual(calculate_width_total(1000), 250)
            self.assertEqual(calculate_width_total(0), 0)
        finally:
            config.DOWN_TIME_RATE = old

    def test_time_slice(self):
        self.assertEqual(calculate_time_slice(1000, 256), 4)
        self.assertEqual(calculate_time_slice(512, 512), 1)

    def test_overlap_decimated_rounds_up(self):
        old = config.DOWN_TIME_RATE
        try:
            config.DOWN_TIME_RATE = 4
            left, right = calculate_overlap_decimated(5, 8)
            self.assertEqual(left, 2)   # ceil(5/4)
            self.assertEqual(right, 2)  # ceil(8/4)
        finally:
            config.DOWN_TIME_RATE = old

    def test_frequency_downsampled(self):
        old = (config.FREQ, config.FREQ_RESO, config.DOWN_FREQ_RATE)
        try:
            config.FREQ = np.array([400.0, 500.0, 600.0, 700.0])
            config.FREQ_RESO = 4
            config.DOWN_FREQ_RATE = 2
            ds = calculate_frequency_downsampled()
            np.testing.assert_allclose(ds, [450.0, 650.0])
        finally:
            (config.FREQ, config.FREQ_RESO, config.DOWN_FREQ_RATE) = old


class TestExtractCandidateDM(unittest.TestCase):
    def setUp(self):
        self._old = (config.DM_min, config.DM_max, config.TIME_RESO, config.DOWN_TIME_RATE)
        config.DM_min, config.DM_max = 0, 100
        config.TIME_RESO, config.DOWN_TIME_RATE = 1e-3, 1

    def tearDown(self):
        (config.DM_min, config.DM_max, config.TIME_RESO, config.DOWN_TIME_RATE) = self._old

    # --- linear (no dm_values) branch: exact arithmetic so mutants die ---
    def test_bottom_row(self):
        dm, _, _ = extract_candidate_dm(0.0, 0.0, 512, img_height=11)
        self.assertAlmostEqual(dm, 0.0, places=9)

    def test_exact_midpoint(self):
        # H=11 => denom=10; py=5 => frac=0.5 => dm = 50 exactly.
        dm, _, _ = extract_candidate_dm(0.0, 5.0, 512, img_height=11)
        self.assertAlmostEqual(dm, 50.0, places=9)

    def test_exact_quarter(self):
        # py=2 => frac=0.2 => dm=20 exactly (kills mul/div/sub mutants on the slope).
        dm, _, _ = extract_candidate_dm(0.0, 2.0, 512, img_height=11)
        self.assertAlmostEqual(dm, 20.0, places=9)

    def test_full_row(self):
        # py=denom => frac=1 => dm=DM_max exactly.
        dm, _, _ = extract_candidate_dm(0.0, 10.0, 512, img_height=11)
        self.assertAlmostEqual(dm, 100.0, places=9)

    def test_denom_uses_height_minus_one(self):
        # If denom were img_height (no -1), dm would be 45.45.. not 50.
        dm, _, _ = extract_candidate_dm(0.0, 5.0, 512, img_height=11)
        self.assertNotAlmostEqual(dm, 5.0 / 11.0 * 100.0, places=3)

    def test_clamp_high(self):
        # py beyond grid => dm would overshoot, must clamp to DM_max.
        dm, _, _ = extract_candidate_dm(0.0, 50.0, 512, img_height=11)
        self.assertAlmostEqual(dm, 100.0, places=9)

    def test_clamp_low(self):
        # negative py => dm would go negative, must clamp to DM_min.
        dm, _, _ = extract_candidate_dm(0.0, -50.0, 512, img_height=11)
        self.assertAlmostEqual(dm, 0.0, places=9)

    # --- dm_values (real grid) branch ---
    def test_dm_values_exact_rows(self):
        grid = [0.0, 25.0, 50.0, 75.0, 100.0]
        for py, expected in [(0.0, 0.0), (1.0, 25.0), (2.0, 50.0), (3.0, 75.0), (4.0, 100.0)]:
            dm, _, _ = extract_candidate_dm(0.0, py, slice_len=512, img_height=5, dm_values=grid)
            self.assertAlmostEqual(dm, expected, places=9)

    def test_dm_values_frac_clamped_high(self):
        grid = [0.0, 25.0, 50.0, 75.0, 100.0]
        # py > denom => frac clamped to 1 => last grid entry.
        dm, _, _ = extract_candidate_dm(0.0, 99.0, slice_len=512, img_height=5, dm_values=grid)
        self.assertAlmostEqual(dm, 100.0, places=9)

    def test_dm_values_frac_clamped_low(self):
        grid = [0.0, 25.0, 50.0, 75.0, 100.0]
        dm, _, _ = extract_candidate_dm(0.0, -10.0, slice_len=512, img_height=5, dm_values=grid)
        self.assertAlmostEqual(dm, 0.0, places=9)

    def test_dm_values_empty_falls_back_to_dm_min(self):
        dm, _, _ = extract_candidate_dm(0.0, 3.0, slice_len=512, img_height=5, dm_values=[])
        self.assertAlmostEqual(dm, 0.0, places=9)

    # --- time mapping: exact so arithmetic mutants die ---
    def test_time_mapping(self):
        # slice_len=1024, img_width=512 => scale=2; px=128 => sample_off=256.
        _, t_sec, t_sample = extract_candidate_dm(
            128.0, 0.0, slice_len=1024, img_width=512, img_height=11
        )
        self.assertEqual(t_sample, 256)
        self.assertAlmostEqual(t_sec, 256 * 1e-3 * 1, places=9)

    def test_time_mapping_scales_with_down_rate(self):
        config.DOWN_TIME_RATE = 2
        _, t_sec, t_sample = extract_candidate_dm(
            128.0, 0.0, slice_len=1024, img_width=512, img_height=11
        )
        self.assertEqual(t_sample, 256)
        self.assertAlmostEqual(t_sec, 256 * 1e-3 * 2, places=9)

    def test_time_mapping_truncates_sample(self):
        # px=100, scale=2.5 => sample_off=250.0; use odd width to force fractional.
        _, _, t_sample = extract_candidate_dm(
            100.0, 0.0, slice_len=1280, img_width=512, img_height=11
        )
        self.assertEqual(t_sample, 250)


class TestUtils(unittest.TestCase):
    def test_normalize_ascending(self):
        arr, rev = normalize_frequency_axis(np.array([1.0, 2.0, 3.0]))
        self.assertFalse(rev)
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_normalize_descending(self):
        arr, rev = normalize_frequency_axis(np.array([3.0, 2.0, 1.0]))
        self.assertTrue(rev)
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_normalize_single_element(self):
        arr, rev = normalize_frequency_axis(np.array([5.0]))
        self.assertFalse(rev)

    def test_safe_float(self):
        self.assertEqual(safe_float("3.5"), 3.5)
        self.assertEqual(safe_float(None, default=1.0), 1.0)
        self.assertEqual(safe_float("12*"), 12.0)

    def test_safe_int(self):
        self.assertEqual(safe_int("7"), 7)
        self.assertEqual(safe_int("bad", default=3), 3)
        self.assertEqual(safe_int("9.0"), 9)


if __name__ == "__main__":
    unittest.main()
