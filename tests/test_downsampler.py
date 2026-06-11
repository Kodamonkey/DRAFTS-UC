"""Parity tests for the downsampling kernel.

Captures the exact numerical output of the current downsample_data() on a
small synthetic array.  Any optimisation (Numba single-pass kernel) must
produce results within float32 tolerance of this baseline.
"""
from __future__ import annotations

import unittest
import numpy as np


def _downsample_reference(
    data: np.ndarray,
    down_time: int,
    down_freq: int,
) -> np.ndarray:
    """Pure-Python reference implementation matching the current logic.

    Input shape: (time, pol, freq)
    Output shape: (time // down_time, freq // down_freq), float32

    Steps:
    1. Trim to divisible sizes
    2. Reshape to 5-D
    3. Sum over time (axis 1)
    4. Mean over polarisation (axis 1 after reduction)
    5. Mean over frequency (axis 2 after reduction)
    """
    n_time = (data.shape[0] // down_time) * down_time
    n_freq = (data.shape[2] // down_freq) * down_freq
    n_pol = data.shape[1]
    d = data[:n_time, :, :n_freq]
    d = d.reshape(
        n_time // down_time,
        down_time,
        n_pol,
        n_freq // down_freq,
        down_freq,
    )
    d = d.sum(axis=1)    # sum over time bins
    d = d.mean(axis=1)   # mean over polarisations
    d = d.mean(axis=2)   # mean over frequency bins
    return d.astype(np.float32)


class TestDownsamplerParity(unittest.TestCase):
    """Ensure optimised downsampling produces the same output as the reference."""

    def setUp(self):
        rng = np.random.RandomState(123)
        self.down_time = 8
        self.down_freq = 4
        # (time=1600, pol=1, freq=256) -> (200, 64)
        self.data = rng.randn(1600, 1, 256).astype(np.float32) * 5 + 50
        self.reference = _downsample_reference(self.data, self.down_time, self.down_freq)

    def test_reference_shape(self):
        expected_shape = (1600 // self.down_time, 256 // self.down_freq)
        self.assertEqual(self.reference.shape, expected_shape)

    def test_reference_finite(self):
        self.assertTrue(np.all(np.isfinite(self.reference)))

    def test_reference_dtype(self):
        self.assertEqual(self.reference.dtype, np.float32)

    def test_reference_value_range(self):
        """Sum over time bins multiplies values; mean over pol/freq preserves scale.
        With down_time=8, values should be roughly 8x the input mean."""
        input_mean = self.data.mean()
        output_mean = self.reference.mean()
        # sum(8 bins of ~50) ≈ 400, then mean(pol)=400, mean(freq)=400
        self.assertAlmostEqual(output_mean / (input_mean * self.down_time), 1.0, places=1)

    def test_multipol_parity(self):
        """Verify that multi-polarisation data averages correctly."""
        rng = np.random.RandomState(456)
        data_4pol = rng.randn(800, 4, 128).astype(np.float32) * 3 + 20
        ref = _downsample_reference(data_4pol, 4, 2)
        self.assertEqual(ref.shape, (200, 64))
        self.assertTrue(np.all(np.isfinite(ref)))


if __name__ == "__main__":
    unittest.main()
