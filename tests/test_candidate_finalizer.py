"""SPEC-CAND-001: finalize_patch() produces consistent results for both pipelines."""
from __future__ import annotations

import unittest
import numpy as np

from src.config import config


def _stub_cls_model():
    return None  # classify_patch falls back to SNR-based probability when model=None


class TestFinalizePatch(unittest.TestCase):
    """finalize_patch() must work with synthetic data and produce sane outputs."""

    def setUp(self):
        rng = np.random.RandomState(0)
        n_time, n_chan = 512, 64
        self.n_chan = n_chan
        # Save and configure required config values
        self._old = (config.TIME_RESO, config.DOWN_TIME_RATE, config.DEVICE,
                     config.FREQ, config.FREQ_RESO, config.DOWN_FREQ_RATE)
        config.TIME_RESO = 5e-4
        config.DOWN_TIME_RATE = 1
        config.DEVICE = "cpu"
        config.FREQ = np.linspace(1000.0, 1500.0, n_chan, dtype=np.float32)
        config.FREQ_RESO = n_chan
        config.DOWN_FREQ_RATE = 1
        self.data = (rng.randn(n_time, n_chan) * 5 + 50).astype(np.float32)
        self.freq_down = np.linspace(1000.0, 1500.0, n_chan, dtype=np.float32)
        self.dm_val = 50.0
        self.global_sample = 100
        self.cls_model = _stub_cls_model()
        self.time_reso_ds = config.TIME_RESO * config.DOWN_TIME_RATE

    def tearDown(self):
        (config.TIME_RESO, config.DOWN_TIME_RATE, config.DEVICE,
         config.FREQ, config.FREQ_RESO, config.DOWN_FREQ_RATE) = self._old

    def test_returns_six_tuple(self):
        from src.core.candidate_finalization import finalize_patch

        result = finalize_patch(
            self.data, self.freq_down, self.dm_val,
            self.global_sample, self.cls_model, self.time_reso_ds,
        )
        self.assertEqual(len(result), 6)

    def test_class_prob_in_unit_interval(self):
        from src.core.candidate_finalization import finalize_patch

        _, class_prob, *_ = finalize_patch(
            self.data, self.freq_down, self.dm_val,
            self.global_sample, self.cls_model, self.time_reso_ds,
        )
        self.assertGreaterEqual(class_prob, 0.0)
        self.assertLessEqual(class_prob, 1.0)

    def test_snr_val_non_negative(self):
        from src.core.candidate_finalization import finalize_patch

        _, _, snr_val, *_ = finalize_patch(
            self.data, self.freq_down, self.dm_val,
            self.global_sample, self.cls_model, self.time_reso_ds,
        )
        self.assertGreaterEqual(snr_val, 0.0)

    def test_width_ms_positive_or_none(self):
        from src.core.candidate_finalization import finalize_patch

        *_, width_ms, _ = finalize_patch(
            self.data, self.freq_down, self.dm_val,
            self.global_sample, self.cls_model, self.time_reso_ds,
        )
        if width_ms is not None:
            self.assertGreater(width_ms, 0.0)

    def test_idempotent(self):
        """Same inputs must produce identical outputs (no hidden state)."""
        from src.core.candidate_finalization import finalize_patch

        r1 = finalize_patch(
            self.data, self.freq_down, self.dm_val,
            self.global_sample, self.cls_model, self.time_reso_ds,
        )
        r2 = finalize_patch(
            self.data, self.freq_down, self.dm_val,
            self.global_sample, self.cls_model, self.time_reso_ds,
        )
        self.assertEqual(r1[1], r2[1])   # class_prob identical
        self.assertAlmostEqual(r1[2], r2[2], places=5)  # snr_val identical

    def test_none_patch_graceful(self):
        """Passing empty data must not crash; outputs must be finite or None."""
        from src.core.candidate_finalization import finalize_patch

        empty = np.zeros((0, self.n_chan), dtype=np.float32)
        result = finalize_patch(
            empty, self.freq_down, self.dm_val,
            self.global_sample, self.cls_model, self.time_reso_ds,
        )
        self.assertEqual(len(result), 6)
        _, class_prob, snr_val, peak_idx, width_ms, _ = result
        self.assertGreaterEqual(class_prob, 0.0)
        self.assertLessEqual(class_prob, 1.0)
        self.assertTrue(np.isfinite(snr_val))


if __name__ == "__main__":
    unittest.main()
