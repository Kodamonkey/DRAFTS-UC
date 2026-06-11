"""TDD regression tests for the FRB audit bugfixes.

Each test references a SPEC-ID in SPECS-physics.md and reproduces a concrete bug
found during the architecture/physics/performance audit.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.analysis.science_metrics import dispersion_delay_ms
from src.preprocessing.dm_candidate_extractor import extract_candidate_dm
from src.preprocessing.dedispersion import d_dm_time_g


class TestDispersionMonotonic(unittest.TestCase):
    """SPEC-DM-002: lower frequency => larger delay."""

    def test_delay_is_monotonic_in_frequency(self):
        d_low = dispersion_delay_ms(100.0, 400.0, 800.0)
        d_high = dispersion_delay_ms(100.0, 700.0, 800.0)
        self.assertGreater(d_low, 0.0)
        self.assertGreater(d_low, d_high)


class TestExtractCandidateDM(unittest.TestCase):
    """SPEC-DM-005: CNN box -> DM mapping has no off-by-one and respects DM_max."""

    def setUp(self):
        self._old = (config.DM_min, config.DM_max, config.TIME_RESO, config.DOWN_TIME_RATE)
        config.DM_min = 0
        config.DM_max = 100
        config.TIME_RESO = 1e-3
        config.DOWN_TIME_RATE = 1

    def tearDown(self):
        (config.DM_min, config.DM_max, config.TIME_RESO, config.DOWN_TIME_RATE) = self._old

    def test_dm_at_bottom_row_is_dm_min(self):
        dm, _, _ = extract_candidate_dm(0.0, 0.0, 512)
        self.assertAlmostEqual(dm, float(config.DM_min), places=4)

    def test_dm_at_top_row_does_not_exceed_dm_max(self):
        # py at the very top of a 512px image must map to DM_max, never DM_max+1.
        dm, _, _ = extract_candidate_dm(0.0, 511.0, 512)
        self.assertLessEqual(dm, float(config.DM_max) + 1e-6)
        self.assertAlmostEqual(dm, float(config.DM_max), delta=0.5)

    def test_dm_mapping_is_linear_midpoint(self):
        dm, _, _ = extract_candidate_dm(0.0, 255.5, 512)
        mid = (config.DM_min + config.DM_max) / 2.0
        self.assertAlmostEqual(dm, mid, delta=0.5)


class TestPrewhitenConfig(unittest.TestCase):
    """SPEC-PRE-001: prewhitening is exposed in YAML and defaults to False."""

    def test_prewhiten_default_false(self):
        self.assertFalse(bool(config.PREWHITEN_BEFORE_DM))


class TestDedispersionPurity(unittest.TestCase):
    """SPEC-PURE-001: d_dm_time_g must not mutate global config state."""

    def test_dedispersion_does_not_mutate_config(self):
        old = (config.FREQ, config.FREQ_RESO, config.DOWN_FREQ_RATE,
               config.TIME_RESO, config.DOWN_TIME_RATE, config.DEVICE,
               config.PREWHITEN_BEFORE_DM)
        try:
            # FREQ_RESO (30) deliberately not divisible by DOWN_FREQ_RATE (4).
            config.FREQ = np.linspace(400.0, 800.0, 30).astype(np.float32)
            config.FREQ_RESO = 30
            config.DOWN_FREQ_RATE = 4
            config.TIME_RESO = 5e-4
            config.DOWN_TIME_RATE = 1
            config.DEVICE = "cpu"
            config.PREWHITEN_BEFORE_DM = False

            freq_snapshot = config.FREQ.copy()
            n_chan_ds = 30 // 4  # 7 decimated channels
            data = np.zeros((64, n_chan_ds), dtype=np.float32)
            data[20, :] = 5.0

            _ = d_dm_time_g(data, height=8, width=64, dm_min=0, dm_max=10)

            self.assertEqual(config.FREQ_RESO, 30, "FREQ_RESO was mutated by dedispersion")
            np.testing.assert_array_equal(config.FREQ, freq_snapshot)
        finally:
            (config.FREQ, config.FREQ_RESO, config.DOWN_FREQ_RATE,
             config.TIME_RESO, config.DOWN_TIME_RATE, config.DEVICE,
             config.PREWHITEN_BEFORE_DM) = old


class TestChunkFailurePolicy(unittest.TestCase):
    """SPEC-IO-002: swallowed chunk errors must surface as PARTIAL, not SUCCESS."""

    def test_no_failures_keeps_base_status(self):
        from src.core.pipeline import finalize_file_status
        self.assertEqual(finalize_file_status("SUCCESS_CHUNKED", 0, 5), "SUCCESS_CHUNKED")

    def test_some_failures_marks_partial(self):
        from src.core.pipeline import finalize_file_status
        self.assertEqual(finalize_file_status("SUCCESS_CHUNKED", 2, 3), "SUCCESS_CHUNKED_PARTIAL")

    def test_all_failed_is_error(self):
        from src.core.pipeline import finalize_file_status
        self.assertEqual(finalize_file_status("SUCCESS_CHUNKED", 3, 0), "ERROR_ALL_CHUNKS_FAILED")


class TestPerFileParameters(unittest.TestCase):
    """SPEC-IO-001: parameters are extracted for the file actually being processed."""

    def test_parameters_extracted_per_file(self):
        import src.core.pipeline as pl

        calls = []

        def fake_extract(path):
            calls.append(Path(path))
            return {"success": True, "parameters_extracted": ["TIME_RESO"], "errors": []}

        orig = pl.extract_parameters_auto
        pl.extract_parameters_auto = fake_extract
        try:
            res, cs = pl._prepare_file_parameters(Path("only_this.fil"), manual_chunk_override=12345)
            self.assertTrue(res["success"])
            self.assertEqual(cs, 12345)
            self.assertEqual(calls, [Path("only_this.fil")])
        finally:
            pl.extract_parameters_auto = orig


if __name__ == "__main__":
    unittest.main()
