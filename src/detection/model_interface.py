# This module bridges the detection and classification models.

"""Model interface for FRB detection and classification - handles neural network inference."""
                          
import logging               
import numpy as np               
from ..config import config

               
try:
    from ..models.ObjectDet.centernet_utils import get_res
except ImportError:
    get_res = None

                              
try:
    import torch
except ImportError:
    torch = None


logger = logging.getLogger(__name__)

# Side of the square frame the detection network works in. preprocess_img()
# resizes every band to this size and centernet_utils.get_res() scales the
# predicted boxes by it, so every box returned by detect() is expressed in
# this frame -- never in the frame of the DM-time cube it came from.
CNN_IMG_SIZE = 512


def detect(model, img_tensor: np.ndarray):
    """Run the detection model and return confidences and boxes."""
    if get_res is None:
        logger.error(
            "get_res is not available. Verify that models.ObjectDet.centernet_utils is installed."
        )
        return [], []
    
    try:
        with torch.no_grad():
            hm, wh, offset = model(
                torch.from_numpy(img_tensor)
                .to(config.DEVICE)
                .float()
                .unsqueeze(0)
            )
        top_conf, top_boxes = get_res(hm, wh, offset, confidence=config.DET_PROB)
                                                    
        if top_boxes is None:
            return [], []
        if torch.is_tensor(top_conf):
            top_conf = top_conf.detach().cpu().numpy()
        if torch.is_tensor(top_boxes):
            top_boxes = top_boxes.detach().cpu().numpy()
        if isinstance(top_conf, np.ndarray):
            top_conf = top_conf.tolist()
        if isinstance(top_boxes, np.ndarray):
            top_boxes = top_boxes.tolist()
        return top_conf, top_boxes
    except Exception as e:
        logger.error(f"Error in detect: {e}")
        return [], []

def prep_patch(patch: np.ndarray) -> np.ndarray:
    """Normalize patch for classification."""
    patch = patch.copy()
    patch += 1
    patch /= np.mean(patch, axis=0)
    vmin, vmax = np.nanpercentile(patch, [5, 95])
    patch = np.clip(patch, vmin, vmax)
    patch = (patch - patch.min()) / (patch.max() - patch.min())
    return patch

def configure_inference_backend() -> dict:
    """Apply the inference backend settings, once per run.

    Audit PERF-04: "No hay AMP, ``channels_last``, ``torch.compile`` ni
    ``cudnn.benchmark`` en el repositorio, y ``advanced-config/performance.yaml``
    declara ``enable_mixed_precision`` y ``batch_size`` que nadie lee." Both
    keys are read now (``user_config``), and this is where the two that are
    backend switches get applied.

    Returns what it decided, so a caller can log it and a test can assert on it
    without inspecting global torch state.

    ``torch.compile`` is deliberately NOT applied. It trades a multi-second
    warm-up for per-call speed, which pays off across a long run and costs on a
    short one, and its failure mode is a graph break that silently falls back
    to eager -- so it needs a measurement to justify it, on a GPU, which this
    environment does not have. Adding it blind is how a "performance" change
    becomes a slowdown nobody attributes.
    """

    decided = {
        "cudnn_benchmark": False,
        "mixed_precision": False,
        "batch_size": int(getattr(config, "INFERENCE_BATCH_SIZE", 1)),
        "device": str(getattr(config, "DEVICE", "cpu")),
    }
    if torch is None:
        return decided

    want_benchmark = bool(getattr(config, "CUDNN_BENCHMARK", True))
    on_cuda = bool(getattr(torch, "cuda", None)) and torch.cuda.is_available()
    if on_cuda:
        try:
            torch.backends.cudnn.benchmark = want_benchmark
            decided["cudnn_benchmark"] = want_benchmark
        except Exception as exc:  # pragma: no cover - depends on the build
            logger.debug("Could not set cudnn.benchmark: %s", exc)
        # Mixed precision is a CUDA feature. Reporting it enabled on a CPU run
        # would be reporting something that is not going to happen.
        decided["mixed_precision"] = bool(
            getattr(config, "ENABLE_MIXED_PRECISION", False)
        )
    return decided


def _autocast_context():
    """``torch.autocast`` when mixed precision is on and we are on CUDA."""
    if torch is None:
        return None
    if not bool(getattr(config, "ENABLE_MIXED_PRECISION", False)):
        return None
    if not (getattr(torch, "cuda", None) and torch.cuda.is_available()):
        return None
    try:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    except Exception as exc:  # pragma: no cover - depends on the build
        logger.debug("autocast unavailable: %s", exc)
        return None


def _snr_fallback_probability(proc: np.ndarray) -> float:
    """The probability used when there is no model (or no torch).

    A deterministic function of the patch's own SNR. It is what makes the
    classification phases runnable without weights, which several test modules
    depend on.
    """
    from ..analysis.snr_utils import compute_snr_profile

    snr_profile, _, _ = compute_snr_profile(proc)
    snr_peak = (
        float(np.max(snr_profile))
        if snr_profile is not None and snr_profile.size > 0
        else 0.0
    )
    thresh = float(getattr(config, "SNR_THRESH", 3.0))
    return float(1.0 / (1.0 + np.exp(-((snr_peak - thresh) / 2.0))))


def classify_patches(model, patches):
    """Classify several patches in one forward pass per batch.

    Audit PERF-04 lists two costs this addresses. Inference ran with a batch of
    one, which on a GPU leaves most of the device idle per call; and the
    probability was read back with ``.item()`` **per candidate**, and ``.item()``
    is a device-to-host copy that synchronises, so every candidate stalled the
    pipeline until its own forward had finished. Here the whole batch comes
    back in one ``tolist()``: one synchronisation per batch instead of one per
    candidate.

    Returns ``[(probability, processed_patch), ...]`` in the order given, which
    is the same pair ``classify_patch`` returns, so a caller can adopt this
    without changing how it reads the result.

    The batch size is ``config.INFERENCE_BATCH_SIZE``
    (``performance.yaml: gpu.batch_size``). Patches are processed in order and
    an oversized list is split, so the caller does not have to know the limit.
    """

    processed = [prep_patch(p) for p in patches]
    if not processed:
        return []

    if model is None or torch is None:
        return [(_snr_fallback_probability(p), p) for p in processed]

    # A batch has to be one tensor, so the patches must agree on shape. They do
    # in this pipeline (dedisperse_patch pads to a fixed patch_len), but a
    # ragged list must not be silently truncated or reshaped: fall back to one
    # at a time, which is correct if slower.
    shapes = {p.shape for p in processed}
    batch_size = max(1, int(getattr(config, "INFERENCE_BATCH_SIZE", 1)))
    if len(shapes) > 1:
        logger.debug("Patches have %d distinct shapes; not batching", len(shapes))
        batch_size = 1

    results: list[tuple[float, np.ndarray]] = []
    for start in range(0, len(processed), batch_size):
        group = processed[start:start + batch_size]
        try:
            stacked = np.stack(group)[:, None, :, :]
            tensor = torch.from_numpy(stacked).float().to(config.DEVICE)
            autocast = _autocast_context()
            with torch.no_grad():
                if autocast is None:
                    out = model(tensor)
                else:
                    with autocast:
                        out = model(tensor)
                # One host transfer for the group. float() first: a float16
                # autocast output would otherwise round the probabilities on
                # the way out.
                probs = out.float().softmax(dim=1)[:, 1].detach().cpu().tolist()
            results.extend((float(p), proc) for p, proc in zip(probs, group))
        except Exception as e:
            logger.error(f"Error in classify_patches (fallback): {e}")
            results.extend((0.0, proc) for proc in group)
    return results


def classify_patch(model, patch: np.ndarray):
    """Return probability from binary model for patch along with the processed patch.

    One patch through the batched path, so there is a single implementation of
    the autocast handling, the fallback and the softmax indexing. Callers that
    have several patches in hand should use ``classify_patches`` instead: this
    one cannot avoid a batch of one.
    """
    try:
        return classify_patches(model, [patch])[0]
    except Exception as e:
        logger.error(f"Error in classify_patch (fallback): {e}")
        return 0.0, prep_patch(patch)
