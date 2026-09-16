"""Parity tests for dedispersion kernels.

Holds the production CPU dedispersion core against a self-contained reference
implementation.  Any optimisation (Numba prange, GPU vectorisation) must
reproduce the reference bit-for-bit on the fixture below.

Two things make this test meaningful rather than decorative:

1. It imports and calls ``_d_dm_time_cpu_core`` — the code that actually runs
   in the pipeline.  A reference checked only against itself proves nothing.
2. The fixture is chosen so that no per-channel delay lands near a ``.5``
   rounding boundary (see ``TestDelayRoundingMargin``).  Sub-sample float
   noise therefore cannot flip a delay by one sample and turn this into a
   flaky test.

Delays are rounded to nearest, not truncated.  Truncation biases every
channel's delay downward by ~0.5 samples on average, which smears the
dedispersed profile; ``TestDelayRounding`` pins that distinction down.
"""
from __future__ import annotations

import unittest
import numpy as np
from src.domain.physics import K_DM_MS
from src.preprocessing.dedispersion import _d_dm_time_cpu_core

# Fixture parameters.  Chosen for a wide margin to the nearest .5 rounding
# boundary relative to float32 error at the peak delay — see
# TestDelayRoundingMargin, which fails loudly if these are changed into an
# unstable regime.
N_TIME = 2000
N_CHAN = 32
HEIGHT = 24
WIDTH = 1950  # > N_TIME - peak_delay, so the src_hi clamp is exercised
DM_MIN = 0.0
DM_MAX = 500.0
FREQ_LO = 1000.0
FREQ_HI = 1500.0
TIME_RESO = 0.01
DOWN_TIME_RATE = 1


def _make_synthetic_data(n_time: int = N_TIME, n_chan: int = N_CHAN, seed: int = 42) -> np.ndarray:
    """Return a reproducible (time, chan) float32 array with realistic variance."""
    rng = np.random.RandomState(seed)
    return rng.randn(n_time, n_chan).astype(np.float32) * 10 + 100


def _channel_delays(
    dm: float,
    freq_ds: np.ndarray,
    time_reso: float,
    down_time_rate: int,
    round_delays: bool = True,
) -> np.ndarray:
    """Per-channel sample delay for one DM, in float64.

    ``round_delays=False`` reproduces the historical truncating behaviour and
    exists only so a test can show the two disagree.
    """
    freq64 = freq_ds.astype(np.float64)
    f_ref_inv2 = np.float64(freq64.max()) ** -2
    raw = (
        np.float64(K_DM_MS) * np.float64(dm) * (freq64 ** -2 - f_ref_inv2)
        / (np.float64(time_reso) * down_time_rate)
    )
    return (np.rint(raw) if round_delays else raw).astype(np.int64)


def _cpu_dedisperse_reference(
    data: np.ndarray,
    height: int,
    width: int,
    dm_min: float,
    dm_max: float,
    freq_ds: np.ndarray,
    time_reso: float,
    down_time_rate: int,
    round_delays: bool = True,
) -> np.ndarray:
    """Pure-NumPy reference implementation (no config globals, no numba).

    Mirrors the contract of ``_d_dm_time_cpu_core`` but is self-contained, so
    it stays valid ground truth after the production code is rewritten.
    """
    out = np.zeros((3, height, width), dtype=np.float32)
    nchan_ds = freq_ds.shape[0]
    mid_channel = nchan_ds // 2
    dm_values = np.linspace(dm_min, dm_max, height).astype(np.float32)

    for i in range(height):
        delays = _channel_delays(
            dm_values[i], freq_ds, time_reso, down_time_rate, round_delays
        )

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


class _Fixture(unittest.TestCase):
    """Shared fixture: synthetic data plus its reference dedispersion."""

    def setUp(self):
        self.data = _make_synthetic_data()
        self.height = HEIGHT
        self.width = WIDTH
        self.dm_min = DM_MIN
        self.dm_max = DM_MAX
        self.freq_ds = np.linspace(FREQ_LO, FREQ_HI, N_CHAN).astype(np.float32)
        self.time_reso = TIME_RESO
        self.down_time_rate = DOWN_TIME_RATE
        self.dm_values = np.linspace(DM_MIN, DM_MAX, HEIGHT).astype(np.float32)

        self.reference = _cpu_dedisperse_reference(
            self.data, self.height, self.width,
            self.dm_min, self.dm_max, self.freq_ds,
            self.time_reso, self.down_time_rate,
        )

    def _run_production(self) -> np.ndarray:
        return _d_dm_time_cpu_core(
            self.data, self.height, self.width,
            self.dm_min, self.dm_max, self.freq_ds,
            self.time_reso, self.down_time_rate, self.dm_values,
        )


class TestDedispersionParity(_Fixture):
    """The production CPU core must reproduce the reference."""

    def test_production_matches_reference(self):
        """Whichever ``_d_dm_time_cpu_core`` is active (Numba or NumPy fallback)
        must match the reference.  This is the assertion the module exists for.
        """
        produced = self._run_production()
        np.testing.assert_allclose(
            produced, self.reference, rtol=1e-6, atol=1e-5,
            err_msg="production dedispersion diverged from the reference kernel",
        )

    def test_production_shape_and_dtype(self):
        produced = self._run_production()
        self.assertEqual(produced.shape, (3, self.height, self.width))
        self.assertEqual(produced.dtype, np.float32)

    def test_production_finite(self):
        self.assertTrue(np.all(np.isfinite(self._run_production())))

    def test_production_not_all_zero(self):
        self.assertGreater(np.abs(self._run_production()).max(), 0.0)

    def test_production_symmetry(self):
        """Band 2 == Band 0 - Band 1 by construction."""
        produced = self._run_production()
        np.testing.assert_allclose(
            produced[2], produced[0] - produced[1], atol=1e-5, rtol=1e-5,
        )

    def test_reference_symmetry(self):
        """Same invariant on the reference, so a broken reference is caught."""
        np.testing.assert_allclose(
            self.reference[2], self.reference[0] - self.reference[1],
            atol=1e-5, rtol=1e-5,
        )


class TestDelayRounding(_Fixture):
    """Delays round to nearest; truncation is a different, wrong answer."""

    def test_truncation_reference_differs(self):
        """Guards against production silently reverting to ``int()`` truncation.

        If this ever stops failing to differ, the parity assertion above has
        lost its power to tell the two apart.
        """
        truncated = _cpu_dedisperse_reference(
            self.data, self.height, self.width,
            self.dm_min, self.dm_max, self.freq_ds,
            self.time_reso, self.down_time_rate,
            round_delays=False,
        )
        self.assertFalse(
            np.allclose(truncated, self.reference, rtol=1e-6, atol=1e-5),
            "truncated and rounded delays produced identical output: "
            "the fixture no longer discriminates between them",
        )

    def test_production_is_not_truncating(self):
        truncated = _cpu_dedisperse_reference(
            self.data, self.height, self.width,
            self.dm_min, self.dm_max, self.freq_ds,
            self.time_reso, self.down_time_rate,
            round_delays=False,
        )
        produced = self._run_production()
        self.assertFalse(
            np.allclose(produced, truncated, rtol=1e-6, atol=1e-5),
            "production output matches the truncating reference: "
            "delay rounding appears to have regressed",
        )


class TestDelayRoundingMargin(_Fixture):
    """The fixture must keep delays away from ``.5`` rounding boundaries.

    A delay sitting on a boundary can be rounded either way depending on
    float32 vs float64 accumulation order — and the Numba kernel runs with
    ``fastmath=True``, which permits reassociation.  Without margin the parity
    test would be flaky rather than wrong.  This test fails first, with a clear
    reason, if the fixture is edited into an unstable regime.
    """

    # float32 relative epsilon, with headroom for fastmath reassociation.
    FLOAT32_EPS = 1.2e-7
    MIN_SAFETY_RATIO = 20.0

    def test_delays_are_clear_of_rounding_boundaries(self):
        freq64 = self.freq_ds.astype(np.float64)
        f_ref_inv2 = np.float64(freq64.max()) ** -2

        worst_margin = 1.0
        peak_delay = 0.0
        for dm in self.dm_values:
            raw = (
                np.float64(K_DM_MS) * np.float64(dm) * (freq64 ** -2 - f_ref_inv2)
                / (np.float64(self.time_reso) * self.down_time_rate)
            )
            peak_delay = max(peak_delay, float(raw.max()))
            worst_margin = min(
                worst_margin, float(np.abs(raw - np.floor(raw) - 0.5).min())
            )

        float32_error = peak_delay * self.FLOAT32_EPS
        self.assertGreater(
            worst_margin, self.MIN_SAFETY_RATIO * float32_error,
            f"fixture delays come within {worst_margin:.3e} of a .5 boundary "
            f"while float32 error at the peak delay ({peak_delay:.1f} samples) "
            f"is {float32_error:.3e}; parity would be luck, not proof",
        )

    def test_fixture_exercises_the_source_clamp(self):
        """Peak delay + WIDTH must overrun N_TIME, or the edge-handling branch
        in both implementations goes untested."""
        delays = _channel_delays(
            self.dm_values[-1], self.freq_ds, self.time_reso, self.down_time_rate
        )
        self.assertGreater(int(delays.max()) + self.width, self.data.shape[0])


class TestGpuFreqRefContract(unittest.TestCase):
    """SPEC-FREQ-001: the GPU path must derive its reference frequency from
    ``freq.max()``, never from the positional ``freq[-1]``.

    The kernel itself needs CUDA hardware to run, so these assertions inspect
    the module's AST instead.  That is weaker than executing the kernel, but it
    does catch the actual regression — reverting to positional indexing — which
    a numpy identity check on an ascending array cannot.
    """

    @classmethod
    def setUpClass(cls):
        import ast
        import inspect
        import src.preprocessing.dedispersion as dedisp

        cls.ast = ast
        cls.source = inspect.getsource(dedisp)
        cls.tree = ast.parse(cls.source)

    def _find_function(self, name):
        for node in self.ast.walk(self.tree):
            if isinstance(node, self.ast.FunctionDef) and node.name == name:
                return node
        self.fail(f"function {name!r} not found in dedispersion module")

    def test_kernel_takes_reference_frequency_as_argument(self):
        """``f_ref_inv2`` is computed on the host and passed in, so the kernel
        cannot re-derive it positionally."""
        kernel = self._find_function("_de_disp_gpu")
        params = [a.arg for a in kernel.args.args]
        self.assertIn(
            "f_ref_inv2", params,
            "GPU kernel no longer receives the host-computed reference frequency",
        )

    def test_kernel_does_not_index_frequency_positionally(self):
        """No ``freq[-1]`` anywhere inside the kernel body."""
        kernel = self._find_function("_de_disp_gpu")
        for node in self.ast.walk(kernel):
            if not isinstance(node, self.ast.Subscript):
                continue
            value = node.value
            if not (isinstance(value, self.ast.Name) and value.id.startswith("freq")):
                continue
            index = node.slice
            if isinstance(index, self.ast.UnaryOp) and isinstance(index.op, self.ast.USub):
                self.fail(
                    "GPU kernel indexes the frequency array from the end; "
                    "SPEC-FREQ-001 requires the host-computed freq.max()"
                )

    def test_host_derives_reference_from_max(self):
        """The call site assigns ``f_ref_inv2`` from a ``.max()`` call."""
        host = self._find_function("d_dm_time_g")
        assignments = [
            node for node in self.ast.walk(host)
            if isinstance(node, self.ast.Assign)
            and any(
                isinstance(t, self.ast.Name) and t.id == "f_ref_inv2"
                for t in node.targets
            )
        ]
        self.assertTrue(
            assignments, "d_dm_time_g no longer computes f_ref_inv2 on the host"
        )
        for assign in assignments:
            calls_max = any(
                isinstance(node, self.ast.Attribute) and node.attr == "max"
                for node in self.ast.walk(assign.value)
            )
            self.assertTrue(
                calls_max,
                "f_ref_inv2 is not derived from .max(); SPEC-FREQ-001 regression",
            )


if __name__ == "__main__":
    unittest.main()
