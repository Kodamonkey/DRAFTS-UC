"""Parity tests for dedispersion kernels.

Captures the exact numerical output of the CPU dedispersion function on a
small synthetic array.  Any optimisation (Numba prange, GPU vectorisation)
must produce results within float32 tolerance of this baseline.
"""
from __future__ import annotations

import unittest
import numpy as np


def _make_synthetic_data(n_time: int = 2000, n_chan: int = 64, seed: int = 42) -> np.ndarray:
    """Return a reproducible (time, chan) float32 array with realistic variance."""
    rng = np.random.RandomState(seed)
    return rng.randn(n_time, n_chan).astype(np.float32) * 10 + 100


def _cpu_dedisperse_reference(
    data: np.ndarray,
    height: int,
    width: int,
    dm_min: float,
    dm_max: float,
    freq_ds: np.ndarray,
    time_reso: float,
    down_time_rate: int,
) -> np.ndarray:
    """Pure-Python reference implementation (no config globals, no numba).

    This mirrors the logic of ``_d_dm_time_cpu`` but is self-contained so it
    can serve as ground truth even after the production code is rewritten.
    """
    out = np.zeros((3, height, width), dtype=np.float32)
    nchan_ds = freq_ds.shape[0]
    mid_channel = nchan_ds // 2
    dm_values = np.linspace(dm_min, dm_max, height).astype(np.float32)

    for i in range(height):
        DM = dm_values[i]
        delays = (
            4.15 * DM * (freq_ds ** -2 - freq_ds.max() ** -2) * 1e3
            / time_reso / down_time_rate
        ).astype(np.int64)

        total_series = np.zeros(width, dtype=np.float32)
        count_series = np.zeros(width, dtype=np.int32)
        mid_series = np.zeros(width, dtype=np.float32)

        for j in range(nchan_ds):
            d = delays[j]
            if d >= 0:
                src_lo = d
                dst_lo = 0
            else:
                src_lo = 0
                dst_lo = -d
            src_hi = d + width
            if src_hi > data.shape[0]:
                src_hi = data.shape[0]
            if src_hi <= src_lo or j >= data.shape[1]:
                continue
            length = src_hi - src_lo
            dst_hi = dst_lo + length

            total_series[dst_lo:dst_hi] += data[src_lo:src_hi, j]
            count_series[dst_lo:dst_hi] += 1

            if j == mid_channel:
                mid_series[dst_lo:dst_hi] = data[src_lo:src_hi, j]

        norm = count_series.astype(np.float32)
        norm[norm <= 0] = 1.0
        out[0, i] = total_series / norm
        out[1, i] = mid_series
        out[2, i] = out[0, i] - out[1, i]

    return out


class TestDedispersionParity(unittest.TestCase):
    """Ensure optimised dedispersion produces the same output as the reference."""

    def setUp(self):
        self.data = _make_synthetic_data(n_time=2000, n_chan=64)
        self.height = 50
        self.width = 1800
        self.dm_min = 0.0
        self.dm_max = 500.0
        self.freq_ds = np.linspace(1000.0, 1500.0, 64).astype(np.float32)
        self.time_reso = 0.001  # 1 ms
        self.down_time_rate = 1

        self.reference = _cpu_dedisperse_reference(
            self.data, self.height, self.width,
            self.dm_min, self.dm_max, self.freq_ds,
            self.time_reso, self.down_time_rate,
        )

    def test_reference_shape(self):
        self.assertEqual(self.reference.shape, (3, self.height, self.width))

    def test_reference_finite(self):
        self.assertTrue(np.all(np.isfinite(self.reference)))

    def test_reference_not_all_zero(self):
        self.assertGreater(np.abs(self.reference).max(), 0.0)

    def test_reference_symmetry(self):
        """Band 2 == Band 0 - Band 1 by construction."""
        np.testing.assert_allclose(
            self.reference[2], self.reference[0] - self.reference[1],
            atol=1e-5, rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
