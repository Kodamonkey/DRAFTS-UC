"""Parity: windowed DM-cube assembly must match direct dedispersion (SPEC-MEM-001)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.core.data_flow_manager import (
    _allocate_dm_cube_buffer,
    _build_dm_time_cube_chunked,
    build_dm_time_cube,
)
from src.preprocessing.dedispersion import d_dm_time_g


class TestWindowedCubeParity(unittest.TestCase):
    def setUp(self):
        self._old = (
            config.FREQ,
            config.FREQ_RESO,
            config.TIME_RESO,
            config.DOWN_TIME_RATE,
            config.DOWN_FREQ_RATE,
            config.DM_min,
            config.DM_max,
            config.DM_GRID_MODE,
            config.PREWHITEN_BEFORE_DM,
            config.DM_CHUNKING_THRESHOLD_GB,
            config.MAX_DM_CUBE_SIZE_GB,
        )
        n_time, n_chan = 256, 16
        config.FREQ = np.linspace(400.0, 800.0, n_chan).astype(np.float32)
        config.FREQ_RESO = n_chan
        config.TIME_RESO = 5e-4
        config.DOWN_TIME_RATE = 1
        config.DOWN_FREQ_RATE = 1
        config.DM_min = 0.0
        config.DM_max = 10.0
        config.DM_GRID_MODE = "legacy_uniform"
        config.PREWHITEN_BEFORE_DM = False
        config.DM_CHUNKING_THRESHOLD_GB = 16.0
        config.MAX_DM_CUBE_SIZE_GB = 64.0
        self.data = np.random.RandomState(0).normal(0, 1, (n_time, n_chan)).astype(np.float32)
        self.height = 11
        self.dm_min = 0.0
        self.dm_max = 10.0

    def tearDown(self):
        (
            config.FREQ,
            config.FREQ_RESO,
            config.TIME_RESO,
            config.DOWN_TIME_RATE,
            config.DOWN_FREQ_RATE,
            config.DM_min,
            config.DM_max,
            config.DM_GRID_MODE,
            config.PREWHITEN_BEFORE_DM,
            config.DM_CHUNKING_THRESHOLD_GB,
            config.MAX_DM_CUBE_SIZE_GB,
        ) = self._old

    def test_windowed_matches_direct(self):
        direct = d_dm_time_g(
            self.data,
            height=self.height,
            width=self.data.shape[0],
            dm_min=self.dm_min,
            dm_max=self.dm_max,
        )
        cube_gb = (3 * self.height * self.data.shape[0] * 4) / (1024**3)
        windowed = _build_dm_time_cube_chunked(
            self.data,
            self.height,
            self.dm_min,
            self.dm_max,
            max(cube_gb * 1.01, 0.001),
        )
        np.testing.assert_allclose(windowed, direct, rtol=1e-5, atol=1e-4)

    def test_memmap_buffer_matches_zeros(self):
        shape = (3, self.height, self.data.shape[0])
        size_gb = (shape[0] * shape[1] * shape[2] * 4) / (1024**3)
        with patch.object(config, "DM_CUBE_MEMMAP_THRESHOLD_GB", 0.0):
            mmap_buf = _allocate_dm_cube_buffer(shape, size_gb)
        zeros_buf = _allocate_dm_cube_buffer(shape, size_gb * 100)
        mmap_buf[:] = 1.0
        zeros_buf[:] = 1.0
        np.testing.assert_array_equal(mmap_buf, zeros_buf)
        if hasattr(mmap_buf, "_mmap_path"):
            mmap_path = Path(mmap_buf._mmap_path)  # type: ignore[attr-defined]
            del mmap_buf
            import gc
            gc.collect()
            mmap_path.unlink(missing_ok=True)

    def test_build_dm_time_cube_uses_windowed_path(self):
        with patch(
            "src.core.data_flow_manager._build_dm_time_cube_chunked",
            wraps=_build_dm_time_cube_chunked,
        ) as mocked:
            result = build_dm_time_cube(
                self.data,
                height=self.height,
                dm_min=self.dm_min,
                dm_max=self.dm_max,
            )
            mocked.assert_called_once()
            self.assertEqual(result.shape, (3, self.height, self.data.shape[0]))


if __name__ == "__main__":
    unittest.main()
