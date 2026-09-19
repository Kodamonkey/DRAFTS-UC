"""Array copies the pipeline makes: the ones removed, and the one that must stay.

Audit PERF-03 and PERF-05. Between them they accounted for roughly five
cube-sized copies. Three are gone; one is load-bearing and is pinned here so it
does not get "optimised" later, because the failure mode is not a wrong number.

PERF-05 -- src/input/filterbank_handler.py
    ``stream_fil`` read each chunk out of a read-only memmap with ``.copy()``
    and then widened it to float32, so an 8-bit chunk cost five times its own
    size in resident memory before anything had looked at it. Both are gone:
    the block is a view, and ``downsample_data`` casts as it accumulates.
    ``TestTheRawBlockDownsamplesIdentically`` asserts that across dtypes and
    both channel orders.

PERF-03 -- src/core/data_flow_manager.py
    ``_build_dm_time_cube_chunked`` allocated a full result cube and memcpy'd
    into it even when there was exactly one DM window -- which is the ordinary
    case, since the non-chunking caller sets the threshold so the whole cube is
    one window. Peak memory for the step was 3.0x the cube; it is 2.0x now.

The one that stays -- ``trim_valid_window``
    Its ``.copy()`` looks like the same kind of waste and is not. When the cube
    is memmap-backed, the caller releases the mapping and unlinks the file
    immediately afterwards. A view would then point at unmapped memory and
    reading it segfaults the interpreter. ``TestTheTrimmedWindowOutlivesTheCube``
    pins that, in a subprocess, because a regression here does not fail an
    assertion -- it takes the test runner down with it.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.config import config
from src.core.data_flow_manager import (
    _allocate_dm_cube_buffer,
    _build_dm_time_cube_chunked,
    release_dm_cube_buffer,
    trim_valid_window,
)
from src.preprocessing.data_downsampler import downsample_data


# --------------------------------------------------------------------------- #
# PERF-05: the raw block is enough
# --------------------------------------------------------------------------- #

class TestTheRawBlockDownsamplesIdentically:
    """The widening to float32 in stream_fil was redundant, not load-bearing.

    ``downsample_data`` casts element-wise while it accumulates, so it produces
    the same float32 output from a uint8 block as from a float32 one. If that
    ever stops being true, removing the copy silently changes every number the
    pipeline produces -- so it is asserted, per dtype and per channel order,
    rather than assumed.
    """

    @pytest.fixture(autouse=True)
    def _pin(self, monkeypatch):
        monkeypatch.setattr(config, "DOWN_TIME_RATE", 4, raising=False)
        monkeypatch.setattr(config, "DOWN_FREQ_RATE", 2, raising=False)
        monkeypatch.setattr(
            config, "TEMPORAL_DOWNSAMPLING_MODE", "sum", raising=False
        )

    @staticmethod
    def _block(dtype) -> np.ndarray:
        rng = np.random.default_rng(11)
        if np.issubdtype(np.dtype(dtype), np.integer):
            info = np.iinfo(dtype)
            hi = min(int(info.max), 250)
            return rng.integers(0, hi, size=(64, 1, 16), dtype=dtype)
        return rng.normal(100.0, 10.0, size=(64, 1, 16)).astype(dtype)

    @pytest.mark.parametrize("dtype", ["uint8", "int16", "float32"])
    @pytest.mark.parametrize("reversed_channels", [False, True])
    def test_the_view_matches_the_copied_and_widened_block(
        self, dtype, reversed_channels
    ):
        raw = self._block(dtype)
        if reversed_channels:
            raw = np.ascontiguousarray(raw[:, :, ::-1])

        # What stream_fil used to hand downstream.
        widened = raw.copy().astype(np.float32)

        from_raw = downsample_data(raw)
        from_widened = downsample_data(widened)

        assert from_raw.shape == from_widened.shape
        assert np.array_equal(from_raw, from_widened), (
            f"{dtype}: downsampling the raw block differs from downsampling the "
            f"widened copy; max|delta|="
            f"{np.abs(from_raw.astype(np.float64) - from_widened.astype(np.float64)).max()}"
        )

    @pytest.mark.parametrize("mode", ["phase_preserving", "snr_preserving"])
    def test_the_other_downsampling_modes_agree_too(self, monkeypatch, mode):
        """These take a different branch of downsample_data entirely."""
        monkeypatch.setattr(
            config, "TEMPORAL_DOWNSAMPLING_MODE", mode, raising=False
        )
        raw = self._block("uint8")
        widened = raw.copy().astype(np.float32)
        assert np.allclose(
            downsample_data(raw), downsample_data(widened), rtol=0, atol=0
        )

    def _stream_first_block(self, tmp_path, monkeypatch, *, reversal: bool):
        from src.input.filterbank_handler import get_obparams_fil, stream_fil
        from tests.synthetic_filterbank import write_filterbank

        fil = tmp_path / "obs.fil"
        write_filterbank(fil, nsamples=512)
        get_obparams_fil(str(fil))
        # get_obparams_fil sets this from the header; pin it so each case tests
        # the branch it means to.
        monkeypatch.setattr(config, "DATA_NEEDS_REVERSAL", reversal, raising=False)

        blocks = list(stream_fil(str(fil), chunk_samples=128))
        assert blocks, "the reader yielded nothing"
        return blocks[0][0]

    def test_an_unreversed_block_is_a_read_only_view(self, tmp_path, monkeypatch):
        """The property that makes dropping the copy safe.

        Read-only means a consumer that tried to write would raise rather than
        corrupt the input file. That is the whole safety argument, so it is
        asserted rather than assumed.
        """
        block = self._stream_first_block(tmp_path, monkeypatch, reversal=False)

        assert not block.flags.writeable, (
            "the streamed block is writeable, so it is not the read-only memmap "
            "view this optimisation assumes"
        )
        assert not block.flags.owndata
        with pytest.raises(ValueError):
            block[0, 0, 0] = 1

    def test_a_reversed_block_is_a_fresh_array(self, tmp_path, monkeypatch):
        """The other branch still copies, and has to.

        ``np.ascontiguousarray(block[:, :, ::-1])`` cannot alias the file: the
        channel order differs. PERF-05 removed the copies that were redundant,
        not this one.
        """
        block = self._stream_first_block(tmp_path, monkeypatch, reversal=True)

        assert block.flags.owndata
        assert block.flags.c_contiguous


# --------------------------------------------------------------------------- #
# PERF-03: one window needs no result array
# --------------------------------------------------------------------------- #

class TestTheSingleWindowCubeIsNotCopied:
    @pytest.fixture(autouse=True)
    def _pin(self, monkeypatch):
        n_chan = 32
        monkeypatch.setattr(
            config, "FREQ", np.linspace(400.0, 800.0, n_chan).astype(np.float32),
            raising=False,
        )
        for key, value in {
            "FREQ_RESO": n_chan, "TIME_RESO": 5e-4,
            "DOWN_TIME_RATE": 1, "DOWN_FREQ_RATE": 1,
            "DM_min": 0.0, "DM_max": 64.0,
            "DM_GRID_MODE": "legacy_uniform",
            "PREWHITEN_BEFORE_DM": False,
            "MAX_DM_SMEARING_MS": "auto",
        }.items():
            monkeypatch.setattr(config, key, value, raising=False)

    @staticmethod
    def _block() -> np.ndarray:
        rng = np.random.default_rng(7)
        return rng.normal(10.0, 1.0, size=(512, 32)).astype(np.float32)

    def _cube(self, height: int, threshold_mult: float) -> np.ndarray:
        block = self._block()
        cube_gb = 3 * height * block.shape[0] * 4 / (1024 ** 3)
        return _build_dm_time_cube_chunked(
            block, height, 0.0, 64.0, max(cube_gb * threshold_mult, 1e-9)
        )

    def test_the_single_window_path_allocates_no_result_array(self, monkeypatch):
        """The assertion PERF-03 is actually about: the second cube is never
        allocated at all.

        Checking the returned array's ``base`` does not work -- the Numba CPU
        route hands back an array whose base is its own allocator's MemInfo --
        so the allocator is watched directly instead.
        """
        from src.core import data_flow_manager as dfm

        calls = []
        real = dfm._allocate_dm_cube_buffer
        monkeypatch.setattr(
            dfm, "_allocate_dm_cube_buffer",
            lambda shape, size_gb: (calls.append(shape), real(shape, size_gb))[1],
        )

        cube = self._cube(height=128, threshold_mult=1.01)

        assert cube.shape == (3, 128, 512)
        assert cube.dtype == np.float32
        assert calls == [], (
            f"a full result cube was still allocated for a single DM window: "
            f"{calls}"
        )

    def test_the_multi_window_path_does_allocate_one(self, monkeypatch):
        """The control: more than one window genuinely needs somewhere to
        assemble, so the short-circuit must not fire there."""
        from src.core import data_flow_manager as dfm

        calls = []
        real = dfm._allocate_dm_cube_buffer
        monkeypatch.setattr(
            dfm, "_allocate_dm_cube_buffer",
            lambda shape, size_gb: (calls.append(shape), real(shape, size_gb))[1],
        )

        self._cube(height=300, threshold_mult=0.30)

        assert calls == [(3, 300, 512)], calls

    def test_the_multi_window_path_still_assembles_the_whole_cube(self):
        """The short-circuit must not touch the path it does not apply to."""
        single = self._cube(height=300, threshold_mult=10.0)
        chunked = self._cube(height=300, threshold_mult=0.30)
        assert single.shape == chunked.shape == (3, 300, 512)
        assert np.array_equal(single, chunked), (
            "assembling the cube from several DM windows no longer matches "
            "building it in one"
        )

    def test_every_dm_row_is_populated(self):
        """A short-circuit that returned a partly-filled cube would still have
        the right shape, so check the content is really there."""
        cube = self._cube(height=128, threshold_mult=1.01)
        assert np.isfinite(cube).all()
        row_energy = np.abs(cube[0]).sum(axis=1)
        assert (row_energy > 0).all(), "some DM rows are entirely zero"


class TestTheFloat32GuaranteeSurvivesCopyFalse:
    """Backs the one PERF-03 change that could not be executed here.

    ``_d_dm_time_torch_gpu`` ends in ``.numpy().astype(np.float32, copy=False)``.
    That route needs a CUDA device and this machine has none, so what the change
    rests on is numpy's astype contract rather than a run of that function.
    Asserting the contract is the honest substitute, and it is what would break
    if a numpy upgrade ever changed it.
    """

    def test_it_is_a_no_op_when_the_dtype_already_matches(self):
        already = np.ones((4, 4), dtype=np.float32)
        assert already.astype(np.float32, copy=False) is already
        assert np.shares_memory(already.astype(np.float32, copy=False), already)

    def test_the_default_would_have_copied(self):
        """Which is what made the original line cost a full cube."""
        already = np.ones((4, 4), dtype=np.float32)
        assert not np.shares_memory(already.astype(np.float32), already)

    def test_it_still_converts_when_the_dtype_differs(self):
        other = np.ones((4, 4), dtype=np.float64)
        out = other.astype(np.float32, copy=False)
        assert out.dtype == np.float32
        assert not np.shares_memory(out, other)


# --------------------------------------------------------------------------- #
# the copy that must stay
# --------------------------------------------------------------------------- #

class TestTheTrimmedWindowOutlivesTheCube:
    """``trim_valid_window`` must return an array that owns its data.

    This is the one copy of the five that is not waste. The caller releases the
    DM cube right after trimming, and when the cube is memmap-backed that closes
    the mapping and unlinks the file.
    """

    def test_the_result_does_not_alias_the_cube(self):
        """The cheap half, in-process: no shared memory, so no dangling view."""
        cube = np.arange(3 * 8 * 64, dtype=np.float32).reshape(3, 8, 64)
        block_ds = np.zeros((64, 4), dtype=np.float32)

        _, dm_time, _, _ = trim_valid_window(block_ds, cube, 8, 8)

        assert not np.shares_memory(dm_time, cube), (
            "trim_valid_window returned a view into the DM cube; the caller "
            "releases that cube immediately afterwards"
        )
        assert dm_time.base is None

    def test_it_survives_releasing_a_memmap_backed_cube(self, tmp_path):
        """The real half. Run out-of-process: the regression is a SIGSEGV, not
        an AssertionError, and it would take the whole test session with it."""
        script = textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(PROJECT_ROOT)!r})
            import numpy as np
            from src.config import config
            from src.core.data_flow_manager import (
                _allocate_dm_cube_buffer, release_dm_cube_buffer, trim_valid_window,
            )

            config.DM_CUBE_MEMMAP_THRESHOLD_GB = 0.0   # force the memmap path
            cube = _allocate_dm_cube_buffer((3, 32, 256), size_gb=0.001)
            assert hasattr(cube, "_mmap_path"), "not memmap-backed; test is vacuous"
            cube[:] = 3.0
            block_ds = np.zeros((256, 4), dtype=np.float32)

            _, dm_time, _, _ = trim_valid_window(block_ds, cube, 16, 16)

            release_dm_cube_buffer(cube)
            del cube
            total = float(dm_time.sum())          # segfaults if dm_time is a view
            assert total == 3.0 * dm_time.size, total
            print("SURVIVED")
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=300,
        )
        assert proc.returncode == 0, (
            f"reading the trimmed window after the cube was released failed "
            f"with exit {proc.returncode} "
            f"({'SIGSEGV' if proc.returncode in (-11, 139) else 'error'}). "
            f"trim_valid_window must copy.\nstdout: {proc.stdout}\n"
            f"stderr: {proc.stderr[-2000:]}"
        )
        assert "SURVIVED" in proc.stdout
