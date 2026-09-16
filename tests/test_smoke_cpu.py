"""CPU-only smoke test: core dedispersion+SNR path without heavy models/GPU (Etapa 6).

No CenterNet/ResNet .pth files are loaded and no GPU is required. This exercises
the scientific hot path end-to-end on tiny synthetic data so CI catches breakage
in the dedispersion -> DM cube -> SNR -> candidate-DM chain quickly.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.analysis.snr_utils import compute_snr_profile, find_snr_peak
from src.core.data_flow_manager import build_dm_time_cube
from src.preprocessing.dedispersion import dedisperse_patch
from src.preprocessing.dm_candidate_extractor import extract_candidate_dm


class TestCPUSmoke(unittest.TestCase):
    def setUp(self):
        self._old = (config.FREQ, config.FREQ_RESO, config.DOWN_FREQ_RATE,
                     config.TIME_RESO, config.DOWN_TIME_RATE, config.DEVICE,
                     config.DM_min, config.DM_max, config.DM_GRID_MODE,
                     config.PREWHITEN_BEFORE_DM)
        self.n_time = 512
        self.n_chan = 32
        self.dm_true = 6.0
        config.FREQ = np.linspace(400.0, 800.0, self.n_chan).astype(np.float32)
        config.FREQ_RESO = self.n_chan
        config.DOWN_FREQ_RATE = 1
        config.TIME_RESO = 5e-4
        config.DOWN_TIME_RATE = 1
        config.DEVICE = "cpu"
        config.DM_min = 0
        config.DM_max = 20
        config.DM_GRID_MODE = "legacy_uniform"
        config.PREWHITEN_BEFORE_DM = False

    def tearDown(self):
        (config.FREQ, config.FREQ_RESO, config.DOWN_FREQ_RATE,
         config.TIME_RESO, config.DOWN_TIME_RATE, config.DEVICE,
         config.DM_min, config.DM_max, config.DM_GRID_MODE,
         config.PREWHITEN_BEFORE_DM) = self._old

    def _make_dispersed(self) -> np.ndarray:
        from src.analysis.science_metrics import K_DM_MS
        data = np.zeros((self.n_time, self.n_chan), dtype=np.float32)
        t0 = 150
        delays = (K_DM_MS * self.dm_true
                  * (config.FREQ ** -2 - config.FREQ.max() ** -2)
                  / config.TIME_RESO).round().astype(int)
        for ch, d in enumerate(delays):
            if 0 <= t0 + d < self.n_time:
                data[t0 + d, ch] = 12.0
        return data

    def test_cpu_pipeline_chain_runs(self):
        data = self._make_dispersed()

        # 1) DM-time cube on CPU
        cube = build_dm_time_cube(data, height=21, dm_min=0, dm_max=20)
        self.assertEqual(cube.shape, (3, 21, self.n_time))

        # 2) DM with maximum response should be near the true DM
        profile_by_dm = cube[0].max(axis=1)
        dm_hat = int(np.argmax(profile_by_dm))
        self.assertLessEqual(abs(dm_hat - self.dm_true), 2.0)

        # 3) SNR profile + peak
        snr, _, _ = compute_snr_profile(data)
        peak_snr, _, peak_idx = find_snr_peak(snr)
        self.assertGreater(float(peak_snr), 0.0)

        # 4) Dedisperse a patch at the recovered DM (no model needed)
        patch, _ = dedisperse_patch(data, config.FREQ, float(dm_hat), int(peak_idx))
        self.assertEqual(patch.shape[1], self.n_chan)

        # 5) Candidate DM mapping stays within range
        dm_val, _, _ = extract_candidate_dm(256.0, 511.0, self.n_time)
        self.assertLessEqual(dm_val, float(config.DM_max) + 1e-6)

    def test_device_is_cpu(self):
        self.assertEqual(str(config.DEVICE), "cpu")


if __name__ == "__main__":
    unittest.main()
