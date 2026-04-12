"""Scientific regression tests for FRB search physics."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.science_metrics import dm_step_for_smearing, post_trials_sigma
from src.analysis.snr_utils import compute_snr_profile
from src.config import config
from src.core.pipeline_parameters import calculate_dm_values, should_use_hf_pipeline
from src.input.polarization_utils import debiased_linear_polarization
from src.input.utils import normalize_frequency_axis
from src.preprocessing.data_downsampler import downsample_data
from src.preprocessing.dedispersion import d_dm_time_g


class TestFrequencyAxisPhysics(unittest.TestCase):
    def test_frequency_axis_keeps_ascending_without_reversal(self):
        freq, reverse = normalize_frequency_axis(np.array([1000.0, 1100.0, 1200.0]))
        self.assertFalse(reverse)
        np.testing.assert_allclose(freq, [1000.0, 1100.0, 1200.0])

    def test_frequency_axis_reverses_descending(self):
        freq, reverse = normalize_frequency_axis(np.array([1200.0, 1100.0, 1000.0]))
        self.assertTrue(reverse)
        np.testing.assert_allclose(freq, [1000.0, 1100.0, 1200.0])


class TestAdaptiveDMGrid(unittest.TestCase):
    def test_smear_limited_grid_uses_physical_spacing(self):
        old = (config.DM_GRID_MODE, config.MAX_DM_SMEARING_MS, config.FREQ, config.DM_min, config.DM_max, config.TIME_RESO, config.DOWN_TIME_RATE)
        try:
            config.DM_GRID_MODE = "smear_limited"
            config.MAX_DM_SMEARING_MS = 1.0
            config.FREQ = np.linspace(400.0, 800.0, 64)
            config.DM_min = 0
            config.DM_max = 100
            config.TIME_RESO = 0.001
            config.DOWN_TIME_RATE = 1
            vals = calculate_dm_values()
            expected = dm_step_for_smearing(1.0, 400.0, 800.0)
            self.assertGreater(vals.size, 1)
            self.assertAlmostEqual(float(vals[1] - vals[0]), expected, delta=expected * 0.05)
        finally:
            (config.DM_GRID_MODE, config.MAX_DM_SMEARING_MS, config.FREQ, config.DM_min, config.DM_max, config.TIME_RESO, config.DOWN_TIME_RATE) = old


class TestDedispersionPhysics(unittest.TestCase):
    def test_correct_dm_maximizes_dedispersed_snr(self):
        old = (config.FREQ, config.FREQ_RESO, config.DOWN_FREQ_RATE, config.TIME_RESO, config.DOWN_TIME_RATE, config.DM_GRID_MODE, config.PREWHITEN_BEFORE_DM)
        try:
            n_time = 512
            n_chan = 32
            dm_true = 200.0
            config.FREQ = np.linspace(400.0, 800.0, n_chan).astype(np.float32)
            config.FREQ_RESO = n_chan
            config.DOWN_FREQ_RATE = 1
            config.TIME_RESO = 0.0005
            config.DOWN_TIME_RATE = 1
            config.DM_GRID_MODE = "legacy_uniform"
            config.PREWHITEN_BEFORE_DM = False

            data = np.zeros((n_time, n_chan), dtype=np.float32)
            t0 = 120
            delays = (4.148808e3 * dm_true * (config.FREQ ** -2 - config.FREQ.max() ** -2) / (config.TIME_RESO * 1000.0)).round().astype(int)
            for ch, d in enumerate(delays):
                data[t0 + d, ch] = 10.0

            cube = d_dm_time_g(data, height=301, width=300, dm_min=0, dm_max=300)
            profile_by_dm = cube[0].max(axis=1)
            dm_hat = int(np.argmax(profile_by_dm))
            self.assertLessEqual(abs(dm_hat - dm_true), 2.0)
            self.assertGreater(profile_by_dm[dm_hat], profile_by_dm[0] * 1.5)
        finally:
            (config.FREQ, config.FREQ_RESO, config.DOWN_FREQ_RATE, config.TIME_RESO, config.DOWN_TIME_RATE, config.DM_GRID_MODE, config.PREWHITEN_BEFORE_DM) = old


class TestSNRAndDownsamplingPhysics(unittest.TestCase):
    def test_matched_filter_width_tracks_physical_width(self):
        old = (config.TIME_RESO, config.DOWN_TIME_RATE, config.DETECTION_WIDTHS_MS)
        try:
            config.TIME_RESO = 0.001
            config.DOWN_TIME_RATE = 1
            config.DETECTION_WIDTHS_MS = [1.0, 3.0, 10.0, 50.0, 150.0]
            wf = np.random.RandomState(4).normal(0, 0.05, (512, 16)).astype(np.float32)
            wf[240:250] += 2.0
            snr, _, widths = compute_snr_profile(wf)
            best_ms = widths[int(np.argmax(snr))] * config.TIME_RESO * 1000.0
            self.assertIn(best_ms, {10.0, 50.0})
        finally:
            (config.TIME_RESO, config.DOWN_TIME_RATE, config.DETECTION_WIDTHS_MS) = old

    def test_phase_preserving_downsample_keeps_offset_pulse(self):
        old = (config.DOWN_TIME_RATE, config.DOWN_FREQ_RATE, config.TEMPORAL_DOWNSAMPLING_MODE)
        try:
            data = np.zeros((64, 1, 8), dtype=np.float32)
            data[7, 0, :] = 10.0
            config.DOWN_TIME_RATE = 8
            config.DOWN_FREQ_RATE = 1
            config.TEMPORAL_DOWNSAMPLING_MODE = "sum"
            summed = downsample_data(data)
            config.TEMPORAL_DOWNSAMPLING_MODE = "phase_preserving"
            preserved = downsample_data(data)
            self.assertGreaterEqual(float(preserved.max()), float(summed.max()))
        finally:
            (config.DOWN_TIME_RATE, config.DOWN_FREQ_RATE, config.TEMPORAL_DOWNSAMPLING_MODE) = old


class TestPolarizationPhysics(unittest.TestCase):
    def test_debiased_linear_noise_is_suppressed(self):
        rng = np.random.RandomState(7)
        q = rng.normal(0, 1, (512, 32)).astype(np.float32)
        u = rng.normal(0, 1, (512, 32)).astype(np.float32)
        raw_l = np.sqrt(q * q + u * u)
        debiased = debiased_linear_polarization(q, u, enabled=True)
        self.assertLess(float(np.median(debiased)), float(np.median(raw_l)))

    def test_debiased_linear_real_pulse_survives(self):
        q = np.zeros((128, 8), dtype=np.float32)
        u = np.zeros((128, 8), dtype=np.float32)
        q[60:64] = 5.0
        l = debiased_linear_polarization(q, u, enabled=True)
        self.assertGreater(float(l[61].mean()), 4.0)


class TestBowtieCollapse(unittest.TestCase):
    """Verify the physics-based bow-tie collapse criterion."""

    def test_lband_wideband_stays_lf(self):
        """L-band 400-800 MHz, DM=500: huge dispersive sweep → LF."""
        use_hf, reason = should_use_hf_pipeline(
            freq_low_mhz=400.0, freq_high_mhz=800.0,
            dm_max=500.0, time_reso_s=5.12e-5, down_time_rate=1,
        )
        self.assertFalse(use_hf)
        self.assertIn("resolved", reason)

    def test_alma_band3_triggers_hf(self):
        """ALMA Band 3 ~86 GHz, DM=1000: sweep ~0.046 ms → HF."""
        use_hf, reason = should_use_hf_pipeline(
            freq_low_mhz=84000.0, freq_high_mhz=88000.0,
            dm_max=1000.0, time_reso_s=5.12e-5, down_time_rate=1,
        )
        self.assertTrue(use_hf)
        self.assertIn("collapsed", reason)

    def test_narrow_band_triggers_hf(self):
        """Very narrow band at any frequency: sweep → 0 → HF."""
        use_hf, _ = should_use_hf_pipeline(
            freq_low_mhz=1400.0, freq_high_mhz=1400.5,
            dm_max=100.0, time_reso_s=5.12e-5, down_time_rate=1,
        )
        self.assertTrue(use_hf)

    def test_dm_max_zero_triggers_hf(self):
        """DM_max=0: no dispersion to resolve → HF."""
        use_hf, _ = should_use_hf_pipeline(
            freq_low_mhz=400.0, freq_high_mhz=800.0,
            dm_max=0.0, time_reso_s=5.12e-5, down_time_rate=1,
        )
        self.assertTrue(use_hf)

    def test_zero_time_reso_falls_back(self):
        """TIME_RESO=0: cannot compute ratio → safe fallback to LF."""
        use_hf, _ = should_use_hf_pipeline(
            freq_low_mhz=400.0, freq_high_mhz=800.0,
            dm_max=500.0, time_reso_s=0.0, down_time_rate=1,
        )
        self.assertFalse(use_hf)

    def test_downsampling_pushes_toward_hf(self):
        """Heavy downsampling increases Δt_res → may cause collapse."""
        # Without downsampling: resolved
        use_hf_no_ds, _ = should_use_hf_pipeline(
            freq_low_mhz=8000.0, freq_high_mhz=8500.0,
            dm_max=200.0, time_reso_s=5.12e-5, down_time_rate=1,
        )
        # With heavy downsampling: may collapse
        use_hf_ds, _ = should_use_hf_pipeline(
            freq_low_mhz=8000.0, freq_high_mhz=8500.0,
            dm_max=200.0, time_reso_s=5.12e-5, down_time_rate=100,
        )
        # Downsampling should push toward HF (or at least not away)
        if not use_hf_no_ds:
            self.assertTrue(use_hf_ds)

    def test_custom_collapse_ratio(self):
        """A higher collapse_ratio makes HF triggering more aggressive."""
        # With default ratio=2.0
        use_hf_default, _ = should_use_hf_pipeline(
            freq_low_mhz=1000.0, freq_high_mhz=1500.0,
            dm_max=10.0, time_reso_s=5.12e-5, down_time_rate=12,
        )
        # With very high ratio=1000 — almost anything collapses
        use_hf_aggressive, _ = should_use_hf_pipeline(
            freq_low_mhz=1000.0, freq_high_mhz=1500.0,
            dm_max=10.0, time_reso_s=5.12e-5, down_time_rate=12,
            collapse_ratio=1000.0,
        )
        self.assertTrue(use_hf_aggressive)


class TestTrialCorrection(unittest.TestCase):
    def test_post_trials_sigma_penalizes_many_trials(self):
        self.assertLess(post_trials_sigma(8.0, 10_000), 8.0)


if __name__ == "__main__":
    unittest.main()
