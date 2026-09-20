"""Batched inference, AMP and the backend switches.

Audit PERF-04. Four things were wrong and three of them are here:

  * inference ran with a batch of one;
  * the probability came back with ``.item()`` **per candidate**, and ``.item()``
    is a device-to-host copy that synchronises, so every candidate stalled until
    its own forward finished;
  * there was no AMP, no ``channels_last``, no ``torch.compile`` and no
    ``cudnn.benchmark`` anywhere in the repository, while
    ``advanced-config/performance.yaml`` declared ``enable_mixed_precision``
    and ``batch_size`` that nothing read.

(The fourth, ``gc.collect()`` per slice, was closed earlier in ``9eb891a``.)

What these tests can and cannot show
------------------------------------
They can show that the batch is a batch: that N patches produce one forward
call rather than N, that the results are the same ones, and that the YAML keys
reach the code. They cannot show that it is faster on a GPU --
``torch.cuda.is_available()`` is False here -- so nothing below claims a speed.
The AMP and cudnn paths are asserted to be *correctly gated*, which is the part
that would otherwise silently do nothing or silently break a CPU run.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.config import config
from src.detection import model_interface as mi


class _CountingModel:
    """Records the shape of every batch it is asked to classify."""

    def __init__(self, prob_for=None):
        self.batches: list[tuple[int, ...]] = []
        self._prob_for = prob_for

    def __call__(self, tensor):
        self.batches.append(tuple(tensor.shape))
        n = tensor.shape[0]
        if self._prob_for is None:
            # Logits whose softmax gives a distinct, recoverable probability
            # per row, so the mapping from input order to output order is
            # observable rather than assumed.
            probs = [0.1 + 0.8 * i / max(1, n - 1) for i in range(n)]
        else:
            probs = [self._prob_for(i) for i in range(n)]
        out = torch.zeros((n, 2), dtype=torch.float32)
        for i, p in enumerate(probs):
            p = min(max(p, 1e-6), 1 - 1e-6)
            out[i, 1] = float(np.log(p / (1 - p)))
        return out

    @property
    def calls(self) -> int:
        return len(self.batches)


def _patches(n, size=32, seed=3):
    rng = np.random.default_rng(seed)
    return [rng.normal(5.0, 1.0, size=(size, size)).astype(np.float32) for _ in range(n)]


@pytest.fixture(autouse=True)
def _cpu_device(monkeypatch):
    monkeypatch.setattr(config, "DEVICE", "cpu", raising=False)
    monkeypatch.setattr(config, "SNR_THRESH", 5.0, raising=False)


class TestTheBatchIsABatch:
    def test_eight_patches_at_batch_four_is_two_forwards(self, monkeypatch):
        monkeypatch.setattr(config, "INFERENCE_BATCH_SIZE", 4, raising=False)
        model = _CountingModel()
        results = mi.classify_patches(model, _patches(8))
        assert model.calls == 2, model.batches
        assert [shape[0] for shape in model.batches] == [4, 4]
        assert len(results) == 8

    def test_a_remainder_is_not_dropped(self, monkeypatch):
        monkeypatch.setattr(config, "INFERENCE_BATCH_SIZE", 4, raising=False)
        model = _CountingModel()
        results = mi.classify_patches(model, _patches(9))
        assert [shape[0] for shape in model.batches] == [4, 4, 1]
        assert len(results) == 9

    def test_batch_one_is_one_forward_per_patch(self, monkeypatch):
        """The old behaviour, still reachable: an operator who sets
        ``batch_size: 1`` in performance.yaml gets it."""
        monkeypatch.setattr(config, "INFERENCE_BATCH_SIZE", 1, raising=False)
        model = _CountingModel()
        mi.classify_patches(model, _patches(5))
        assert model.calls == 5

    def test_the_results_keep_the_order_they_were_given_in(self, monkeypatch):
        """A batch is only useful if row i of the output is patch i. Reversing
        the pairing would be invisible to a test that only checked the count."""
        monkeypatch.setattr(config, "INFERENCE_BATCH_SIZE", 8, raising=False)
        model = _CountingModel(prob_for=lambda i: 0.05 + 0.1 * i)
        results = mi.classify_patches(model, _patches(8))
        probs = [p for p, _ in results]
        assert probs == pytest.approx([0.05 + 0.1 * i for i in range(8)], abs=1e-5)

    def test_it_matches_one_at_a_time(self, monkeypatch):
        """The property that makes the change safe: batching must not move a
        probability."""
        patches = _patches(6)
        monkeypatch.setattr(config, "INFERENCE_BATCH_SIZE", 1, raising=False)
        one_at_a_time = [p for p, _ in mi.classify_patches(_CountingModel(prob_for=lambda i: 0.3), patches)]
        monkeypatch.setattr(config, "INFERENCE_BATCH_SIZE", 6, raising=False)
        batched = [p for p, _ in mi.classify_patches(_CountingModel(prob_for=lambda i: 0.3), patches)]
        assert batched == pytest.approx(one_at_a_time, abs=1e-6)

    def test_an_empty_list_does_not_call_the_model(self):
        model = _CountingModel()
        assert mi.classify_patches(model, []) == []
        assert model.calls == 0

    def test_ragged_shapes_fall_back_instead_of_being_reshaped(self, monkeypatch):
        """Patches of different sizes cannot be stacked. Silently padding or
        truncating them would classify something that is not the candidate."""
        monkeypatch.setattr(config, "INFERENCE_BATCH_SIZE", 8, raising=False)
        model = _CountingModel()
        ragged = _patches(2, size=32) + _patches(1, size=16)
        results = mi.classify_patches(model, ragged)
        assert len(results) == 3
        assert model.calls == 3, "a ragged list must not be stacked"

    def test_a_failing_model_does_not_lose_the_candidates(self, monkeypatch):
        monkeypatch.setattr(config, "INFERENCE_BATCH_SIZE", 4, raising=False)

        class _Boom:
            def __call__(self, tensor):
                raise RuntimeError("out of memory")

        results = mi.classify_patches(_Boom(), _patches(5))
        assert len(results) == 5
        assert all(prob == 0.0 for prob, _ in results)


class TestTheSinglePatchEntryPointStillWorks:
    """``classify_patch`` is called from two places and its contract is
    unchanged; it is one patch through the batched path now, so there is one
    implementation of the autocast handling and the softmax indexing."""

    def test_it_returns_the_same_pair(self, monkeypatch):
        monkeypatch.setattr(config, "INFERENCE_BATCH_SIZE", 4, raising=False)
        model = _CountingModel(prob_for=lambda i: 0.42)
        patch = _patches(1)[0]
        prob, proc = mi.classify_patch(model, patch)
        assert prob == pytest.approx(0.42, abs=1e-5)
        assert proc.shape == patch.shape

    def test_without_a_model_it_uses_the_snr_fallback(self):
        """Several test modules classify with ``cls_model=None``; the fallback
        must survive the batching."""
        patch = _patches(1)[0]
        prob, proc = mi.classify_patch(None, patch)
        assert 0.0 <= prob <= 1.0
        assert proc.shape == patch.shape

    def test_the_fallback_batches_too(self):
        probs = [p for p, _ in mi.classify_patches(None, _patches(4))]
        assert len(probs) == 4
        assert all(0.0 <= p <= 1.0 for p in probs)


class TestTheBackendSwitches:
    def test_the_yaml_keys_reach_the_code(self):
        """They were loaded into a dictionary and read by nothing."""
        assert isinstance(config.INFERENCE_BATCH_SIZE, int)
        assert config.INFERENCE_BATCH_SIZE >= 1
        assert isinstance(config.ENABLE_MIXED_PRECISION, bool)
        assert isinstance(config.CUDNN_BENCHMARK, bool)

    def test_configure_reports_what_it_decided(self, monkeypatch):
        monkeypatch.setattr(config, "INFERENCE_BATCH_SIZE", 7, raising=False)
        decided = mi.configure_inference_backend()
        assert decided["batch_size"] == 7
        assert set(decided) == {
            "cudnn_benchmark", "mixed_precision", "batch_size", "device"
        }

    @pytest.mark.skipif(torch.cuda.is_available(), reason="needs a CPU-only host")
    def test_on_a_cpu_host_nothing_cuda_is_claimed(self, monkeypatch):
        """cudnn.benchmark and AMP are CUDA features. Reporting them enabled on
        a CPU run would be reporting something that is not going to happen --
        and calling into them would raise."""
        monkeypatch.setattr(config, "CUDNN_BENCHMARK", True, raising=False)
        monkeypatch.setattr(config, "ENABLE_MIXED_PRECISION", True, raising=False)
        decided = mi.configure_inference_backend()
        assert decided["cudnn_benchmark"] is False
        assert decided["mixed_precision"] is False

    @pytest.mark.skipif(torch.cuda.is_available(), reason="needs a CPU-only host")
    def test_autocast_is_not_entered_on_cpu(self, monkeypatch):
        monkeypatch.setattr(config, "ENABLE_MIXED_PRECISION", True, raising=False)
        assert mi._autocast_context() is None

    def test_mixed_precision_off_means_no_autocast_anywhere(self, monkeypatch):
        monkeypatch.setattr(config, "ENABLE_MIXED_PRECISION", False, raising=False)
        assert mi._autocast_context() is None

    def test_a_cpu_run_classifies_normally_with_mixed_precision_requested(
        self, monkeypatch
    ):
        """The gate has to be a no-op, not an error: a config file written for
        a GPU host must still run on a laptop."""
        monkeypatch.setattr(config, "ENABLE_MIXED_PRECISION", True, raising=False)
        monkeypatch.setattr(config, "INFERENCE_BATCH_SIZE", 4, raising=False)
        results = mi.classify_patches(_CountingModel(prob_for=lambda i: 0.6), _patches(4))
        assert [p for p, _ in results] == pytest.approx([0.6] * 4, abs=1e-5)


class TestTheBandFunctionUsesTheBatch:
    """The adoption. An API nothing calls is the state audit item 27 was about."""

    def test_phase_3a_classifies_a_whole_slice_in_one_go(self, monkeypatch):
        """Two candidates, batch size 4: one forward, not two."""
        from src.core import candidate_finalization

        seen = {"batches": []}
        real = candidate_finalization.classify_patches

        def _spy(model, patches):
            seen["batches"].append(len(patches))
            return real(model, patches)

        monkeypatch.setattr(candidate_finalization, "classify_patches", _spy)

        import tests.test_hf_band_function as band_module

        # Reuse the band harness's inputs rather than rebuilding them.
        block = band_module._intensity_chunk()
        cube = band_module._dm_cube()
        slice_cube = cube[:, :, band_module.SLICE_START:band_module.SLICE_END]

        for key, value in {
            "FREQ": np.linspace(1200.0, 1500.0, band_module.N_CHAN).astype(np.float64),
            "FREQ_RESO": band_module.N_CHAN, "TIME_RESO": band_module.TIME_RESO,
            "DOWN_TIME_RATE": 1, "DOWN_FREQ_RATE": 1,
            "DM_min": band_module.DM_MIN, "DM_max": band_module.DM_MAX,
            "SNR_THRESH": 5.0, "TSTART_MJD": band_module.TSTART_MJD,
            "HIGH_FREQ_DM_POLICY": "unresolved",
            "SOURCE_RA": None, "SOURCE_DEC": None, "REF_FREQ_MHZ": None,
            "OBSERVATORY": None, "EPHEMERIS": None,
            "INFERENCE_BATCH_SIZE": 4,
        }.items():
            monkeypatch.setattr(config, key, value, raising=False)

        from pathlib import Path
        import tempfile

        from src.core import high_freq_pipeline as hfp
        from src.output.candidate_manager import CandidateWriter, ensure_csv_header

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_file = tmp_path / "c.csv"
            ensure_csv_header(csv_file)
            result = hfp.snr_detect_and_classify_candidates_in_band(
                None, slice_cube[0], block[band_module.SLICE_START:band_module.SLICE_END],
                band_module.SLICE_LEN, 0, Path("x.fits"), tmp_path, block,
                config.FREQ.astype(np.float32), csv_file, band_module.TIME_RESO, [],
                0.0, tmp_path, 0, 0, band_module.SLICE_START,
                slice_samples=band_module.SLICE_LEN,
                dm_time_fullband=slice_cube[0],
                snapshot=band_module._snapshot(ENABLE_INTENSITY_CLASSIFICATION=True),
            )
            CandidateWriter.flush_all()

        assert result["cand_counter"] == 2
        assert seen["batches"] == [2], (
            f"Phase 3a made {len(seen['batches'])} classification calls for "
            f"{result['cand_counter']} candidates; the batching is not adopted"
        )
