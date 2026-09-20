"""The torch dedispersion kernel: vectorised, and checked without a GPU.

Audit PERF-02. The kernel had a vectorised outer loop wrapped around a Python
inner loop -- one tensor-indexing call per channel per DM chunk, O(H/D * C)
kernel launches -- and it built three full ``(D, B, W)`` tensors to feed it.
It is now a single ``torch.gather`` per channel batch.

Why these tests exist in this shape
-----------------------------------
This function produces the science and could not be executed by the test suite
at all: it hardcoded ``torch.device('cuda')`` and called
``torch.cuda.mem_get_info``, so on a machine without CUDA -- which is every
machine this suite has ever run on, including CI -- it was unreachable. The
comment on its last line said as much: "NOT EXECUTED HERE ... the claim rests
on numpy's documented astype semantics, and not on having run this function."

Taking the device as a parameter is what makes the rest of this module
possible. The kernel now runs on CPU, where it can be compared against
``_d_dm_time_cpu`` element for element.

What is NOT verified here, and cannot be: that any of this is faster on a GPU.
Measured on CPU with two threads it is about twice as quick, and the kernel
launches it removes are a GPU cost that does not exist on CPU at all, so the
CPU figure is a lower bound on the GPU one rather than an estimate of it.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.config import config
from src.preprocessing.dedispersion import (
    _d_dm_time_cpu,
    _d_dm_time_torch,
    _torch_dm_chunk_size,
)

#: Geometries, chosen so the loops actually iterate: channel counts either side
#: of the 32-channel batch, DM heights either side of the 16-row chunk floor,
#: and one case where nothing divides evenly.
GEOMETRIES = [
    (512, 32, 40, 512),
    (300, 17, 25, 300),
    (1024, 64, 64, 1024),
    (256, 8, 16, 256),
]


@pytest.fixture(autouse=True)
def _pinned_resolution(monkeypatch):
    monkeypatch.setattr(config, "TIME_RESO", 1.0e-3, raising=False)
    monkeypatch.setattr(config, "DOWN_TIME_RATE", 1, raising=False)


def _inputs(n_time, n_chan, seed=7):
    rng = np.random.default_rng(seed)
    data = rng.normal(10.0, 1.0, size=(n_time, n_chan)).astype(np.float32)
    freq = np.linspace(1200.0, 1500.0, n_chan).astype(np.float32)
    return data, freq


class TestItAgreesWithTheCPUKernel:
    """The two kernels are independent implementations of one definition. On
    the same input they must produce the same cube."""

    @pytest.mark.parametrize("n_time,n_chan,height,width", GEOMETRIES)
    def test_the_cube_matches_element_for_element(self, n_time, n_chan, height, width):
        data, freq = _inputs(n_time, n_chan)
        reference = _d_dm_time_cpu(data, height, width, 0.0, 300.0, freq)
        got = _d_dm_time_torch(data, height, width, 0.0, 300.0, freq, device="cpu")
        assert got.shape == reference.shape == (3, height, width)
        # float32 accumulation in a different order; the tolerance is rounding,
        # not slack -- the measured difference is around 1e-6 on values of
        # order 10.
        assert got == pytest.approx(reference, rel=1e-5, abs=1e-4)

    def test_all_three_planes_are_checked_and_none_is_empty(self):
        """A kernel that returned zeros for the mid and difference planes would
        pass a comparison that only looked at plane 0."""
        data, freq = _inputs(512, 32)
        got = _d_dm_time_torch(data, 40, 512, 0.0, 300.0, freq, device="cpu")
        for plane in range(3):
            assert np.ptp(got[plane]) > 0.0, f"plane {plane} is constant"

    def test_an_explicit_dm_grid_is_honoured(self):
        """The non-uniform grid path (DM_GRID_MODE smear_limited) takes the
        other branch of the dm_values argument."""
        data, freq = _inputs(512, 32)
        dm_values = np.array([0.0, 17.5, 120.0, 300.0], dtype=np.float32)
        reference = _d_dm_time_cpu(data, 4, 512, 0.0, 300.0, freq, dm_values=dm_values)
        got = _d_dm_time_torch(data, 4, 512, 0.0, 300.0, freq, dm_values, device="cpu")
        assert got == pytest.approx(reference, rel=1e-5, abs=1e-4)


class TestTheChannelLoopIsGone:
    """The defect was the launch count, so the test counts launches."""

    def _count_gathers(self, monkeypatch, *, n_chan, height, width=512, n_time=512):
        calls = {"n": 0}
        real_gather = torch.gather

        def _counting(*args, **kwargs):
            calls["n"] += 1
            return real_gather(*args, **kwargs)

        monkeypatch.setattr(torch, "gather", _counting)
        data, freq = _inputs(n_time, n_chan)
        _d_dm_time_torch(data, height, width, 0.0, 300.0, freq, device="cpu")
        return calls["n"]

    def test_one_gather_per_channel_batch_not_one_per_channel(self, monkeypatch):
        """64 channels in batches of 32 is two gathers per DM chunk. The old
        shape did 64 indexing operations per DM chunk instead."""
        n_chan, height, width = 64, 16, 512
        gathers = self._count_gathers(monkeypatch, n_chan=n_chan, height=height,
                                      width=width)
        chunk = _torch_dm_chunk_size(channels_in_batch=min(n_chan, 32),
                                     width=width, device=torch.device("cpu"),
                                     height=height)
        n_chunks = (height + chunk - 1) // chunk
        assert gathers == n_chunks * 2

    def test_doubling_the_channels_does_not_double_the_launches(self, monkeypatch):
        """The property that distinguishes a vectorised kernel from a loop: the
        launch count follows the number of BATCHES, and a batch holds 32
        channels however many there are."""
        few = self._count_gathers(monkeypatch, n_chan=32, height=16)
        many = self._count_gathers(monkeypatch, n_chan=64, height=16)
        assert many == 2 * few, (few, many)
        assert few <= 4, (
            f"{few} launches for a single 32-channel batch; the per-channel "
            "loop is back"
        )


class TestTheMemoryHeuristic:
    """``bytes_per_dm`` counted every channel in the file when only one batch is
    resident, so it overestimated by C / chan_batch and the computed chunk
    collapsed onto the hard floor of 16. A heuristic that always returns its
    floor is not a heuristic."""

    def test_it_counts_the_resident_channels_not_all_of_them(self):
        """Same width, same free memory, channel counts 32x apart: the answer
        must not move, because the same 32 channels are resident either way."""
        cpu = torch.device("cpu")
        small = _torch_dm_chunk_size(channels_in_batch=32, width=2048,
                                     device=cpu, height=512)
        # What the old formula would have been handed for a 1024-channel file.
        big_if_it_counted_everything = _torch_dm_chunk_size(
            channels_in_batch=1024, width=2048, device=cpu, height=512
        )
        assert small > big_if_it_counted_everything, (
            "counting all channels no longer changes the answer, so this test "
            "is not measuring the thing it was written for"
        )
        assert small > 16, (
            f"the chunk size came back at {small}, on or below the floor: the "
            "heuristic is not deciding anything"
        )

    def test_it_never_returns_less_than_the_floor(self):
        cpu = torch.device("cpu")
        absurd = _torch_dm_chunk_size(channels_in_batch=1 << 20, width=1 << 20,
                                      device=cpu, height=512)
        assert absurd == 16

    def test_it_is_capped(self):
        cpu = torch.device("cpu")
        assert _torch_dm_chunk_size(channels_in_batch=1, width=1,
                                    device=cpu, height=512) == 256

    def test_it_does_not_ask_a_cpu_device_for_vram(self, monkeypatch):
        """``torch.cuda.mem_get_info`` raises without a CUDA device. The old
        kernel called it unconditionally, which is one reason it could not run
        here at all."""
        def _explode(*a, **k):
            raise AssertionError("mem_get_info called for a CPU device")

        if hasattr(torch.cuda, "mem_get_info"):
            monkeypatch.setattr(torch.cuda, "mem_get_info", _explode)
        assert _torch_dm_chunk_size(channels_in_batch=32, width=512,
                                    device=torch.device("cpu"), height=64) > 0


class TestItRunsWithoutACudaDevice:
    """The property the whole module rests on. Before this change the function
    was unreachable from any test."""

    def test_the_default_device_is_still_cuda(self):
        """The production wrapper must keep asking for the GPU; only the kernel
        became device-agnostic."""
        import inspect

        from src.preprocessing.dedispersion import _d_dm_time_torch_gpu

        source = inspect.getsource(_d_dm_time_torch_gpu)
        assert 'torch.device("cuda")' in source or "torch.device('cuda')" in source

    def test_the_kernel_accepts_a_device_and_uses_it(self):
        data, freq = _inputs(256, 16)
        got = _d_dm_time_torch(data, 16, 256, 0.0, 200.0, freq,
                               device=torch.device("cpu"))
        assert got.dtype == np.float32
        assert np.isfinite(got).all()
