"""Tests for pipeline data contracts (SDD Etapa 3)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.core.contracts import (
    ChunkPlan,
    CandidateRecord,
    DMGrid,
    ObservationMetadata,
    PipelineConfigSnapshot,
)
from src.preprocessing.dm_candidate_extractor import extract_candidate_dm


class TestObservationMetadata(unittest.TestCase):
    def test_from_config_roundtrip(self):
        old = (config.TIME_RESO, config.FREQ_RESO, config.FILE_LENG, config.FREQ,
               config.DOWN_TIME_RATE, config.DOWN_FREQ_RATE)
        try:
            config.TIME_RESO = 1e-3
            config.FREQ_RESO = 4
            config.FILE_LENG = 1000
            config.FREQ = np.array([400.0, 500.0, 600.0, 700.0])
            config.DOWN_TIME_RATE = 4
            config.DOWN_FREQ_RATE = 2
            meta = ObservationMetadata.from_config(config)
            self.assertEqual(meta.freq_low, 400.0)
            self.assertEqual(meta.freq_high, 700.0)
            self.assertAlmostEqual(meta.effective_time_reso, 4e-3)
            self.assertAlmostEqual(meta.duration_s, 1.0)
        finally:
            (config.TIME_RESO, config.FREQ_RESO, config.FILE_LENG, config.FREQ,
             config.DOWN_TIME_RATE, config.DOWN_FREQ_RATE) = old

    def test_is_immutable(self):
        meta = ObservationMetadata(time_reso=1e-3, freq_reso=2, file_leng=10, freq_mhz=(1.0, 2.0))
        with self.assertRaises(Exception):
            meta.time_reso = 2e-3  # type: ignore[misc]


class TestChunkPlan(unittest.TestCase):
    def test_valid_length_and_block_bounds(self):
        plan = ChunkPlan(chunk_idx=1, start_sample=100, end_sample=200, overlap_left=10, overlap_right=20)
        self.assertEqual(plan.valid_length, 100)
        self.assertEqual(plan.block_start, 90)
        self.assertEqual(plan.block_end, 220)

    def test_block_start_clamped_at_zero(self):
        plan = ChunkPlan(chunk_idx=0, start_sample=5, end_sample=50, overlap_left=10)
        self.assertEqual(plan.block_start, 0)

    def test_invalid_bounds_raise(self):
        with self.assertRaises(ValueError):
            ChunkPlan(chunk_idx=0, start_sample=200, end_sample=100)


class TestDMGrid(unittest.TestCase):
    def test_row_mapping_never_exceeds_dm_max(self):
        grid = DMGrid.from_values(np.linspace(0, 100, 101))
        self.assertEqual(grid.dm_min, 0.0)
        self.assertEqual(grid.dm_max, 100.0)
        self.assertLessEqual(grid.dm_for_row(511, height=512), grid.dm_max)
        self.assertAlmostEqual(grid.dm_for_row(0, height=512), 0.0)

    def test_row_mapping_monotonic(self):
        grid = DMGrid.from_values(np.linspace(10, 50, 41))
        prev = -np.inf
        for row in range(0, 512, 32):
            dm = grid.dm_for_row(row, height=512)
            self.assertGreaterEqual(dm, prev)
            prev = dm

    def test_from_config(self):
        old = (config.DM_min, config.DM_max, config.DM_GRID_MODE)
        try:
            config.DM_min = 0
            config.DM_max = 50
            config.DM_GRID_MODE = "legacy_uniform"
            grid = DMGrid.from_config(config)
            self.assertEqual(grid.dm_min, 0.0)
            self.assertEqual(grid.dm_max, 50.0)
            self.assertEqual(grid.size, 51)
        finally:
            (config.DM_min, config.DM_max, config.DM_GRID_MODE) = old

    def test_dm_grid_aligned_with_extractor(self):
        grid = DMGrid.from_values(np.linspace(0, 100, 101))
        old = (config.DM_min, config.DM_max)
        try:
            config.DM_min = 0.0
            config.DM_max = 100.0
            for row in (0, 128, 255, 511):
                dm_grid = grid.dm_for_row(row, height=512)
                dm_ext, _, _ = extract_candidate_dm(
                    0.0, float(row), 512, img_height=512, dm_values=grid.values
                )
                self.assertAlmostEqual(dm_grid, dm_ext, places=4)
        finally:
            config.DM_min, config.DM_max = old


class TestDmFromImageVsDMGrid(unittest.TestCase):
    """SPEC-DM-005: _dm_from_image_at_time must match DMGrid.dm_for_row (Etapa 4)."""

    def _make_image_with_peak_at_row(self, height: int, width: int, peak_row: int) -> np.ndarray:
        img = np.zeros((height, width), dtype=np.float32)
        img[peak_row, width // 2] = 1000.0
        return img

    def test_mapping_matches_dmgrid(self):
        """_dm_from_image_at_time and DMGrid.dm_for_row must give the same DM."""
        from src.core.high_freq_pipeline import _dm_from_image_at_time
        from src.config import config

        old_min, old_max = config.DM_min, config.DM_max
        try:
            config.DM_min = 10.0
            config.DM_max = 200.0
            height, width = 64, 100

            for peak_row in (0, height // 2, height - 1):
                img = self._make_image_with_peak_at_row(height, width, peak_row)
                dm_from_fn = _dm_from_image_at_time(img, width // 2)
                grid = DMGrid.from_values(np.linspace(config.DM_min, config.DM_max, height))
                dm_from_grid = grid.dm_for_row(peak_row, height=height)
                self.assertAlmostEqual(
                    dm_from_fn, dm_from_grid, places=4,
                    msg=f"Mismatch at peak_row={peak_row}: fn={dm_from_fn}, grid={dm_from_grid}"
                )
        finally:
            config.DM_min, config.DM_max = old_min, old_max


class TestSnapshotAndRecord(unittest.TestCase):
    def test_config_snapshot_reads_prewhiten_false(self):
        snap = PipelineConfigSnapshot.from_config(config)
        self.assertFalse(snap.prewhiten_before_dm)

    def test_candidate_record_defaults(self):
        rec = CandidateRecord(file_name="x.fil", dm=12.3, time_sec=1.5, snr=7.0)
        self.assertEqual(rec.box, (0, 0, 0, 0))
        self.assertFalse(rec.is_burst)
        self.assertIsInstance(rec.extra, dict)


if __name__ == "__main__":
    unittest.main()
