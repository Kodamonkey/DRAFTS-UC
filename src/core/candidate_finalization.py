"""Shared candidate finalization logic (SPEC-CAND-001).

Both detection pathways (CenterNet LF and SNR-peak HF) use finalize_patch()
to produce a consistent (proc_patch, class_prob, snr_val, peak_idx_patch,
width_ms, start_sample) tuple.  The Candidate assembly with pipeline-specific
fields remains in each pipeline.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from ..analysis.snr_utils import compute_snr_profile
from ..detection.model_interface import classify_patch, classify_patches
from ..preprocessing.dedispersion import dedisperse_patch

logger = logging.getLogger(__name__)


def _dedisperse_and_measure(
    data: np.ndarray,
    freq_down: np.ndarray,
    dm_val: float,
    global_sample: int,
) -> Tuple[Optional[np.ndarray], int, float, Optional[int], np.ndarray]:
    """Everything before the classifier: dedisperse, then measure the patch.

    Split out of ``finalize_patch`` so that several candidates can be prepared
    and then classified together (audit PERF-04). Returns
    ``(patch, start_sample, snr_val, peak_idx_patch, best_w_vec)``.
    """
    patch, start_sample = dedisperse_patch(data, freq_down, dm_val, global_sample)

    snr_val: float = 0.0
    peak_idx_patch: Optional[int] = None
    best_w_vec: np.ndarray = np.array([])

    if patch is not None and patch.size > 0:
        try:
            snr_profile_pre, _, best_w_vec = compute_snr_profile(patch)
            if snr_profile_pre is not None and snr_profile_pre.size > 0:
                peak_idx_patch = int(np.argmax(snr_profile_pre))
                snr_val = float(np.max(snr_profile_pre))
        except Exception as exc:
            logger.debug("SNR computation failed in finalize_patch: %s", exc)

    return patch, start_sample, snr_val, peak_idx_patch, best_w_vec


def _width_ms(peak_idx_patch, best_w_vec, time_reso_ds) -> Optional[float]:
    try:
        if peak_idx_patch is not None and best_w_vec.size > 0:
            return float(best_w_vec[int(peak_idx_patch)] * time_reso_ds * 1000.0)
    except Exception:
        pass
    return None


def finalize_patches(
    requests,
    cls_model,
    time_reso_ds: float,
):
    """``finalize_patch`` for several candidates, with ONE batched inference.

    *requests* is an iterable of ``(data, freq_down, dm_val, global_sample)``.
    Returns a list of the same six-tuples ``finalize_patch`` returns, in order.

    The dedispersion and the SNR measurement stay per candidate -- they are
    numpy work on different windows and there is nothing to batch. What batches
    is the classifier call, which is the part that was running at batch one and
    synchronising the device once per candidate (audit PERF-04).
    """
    prepared = [
        _dedisperse_and_measure(data, freq_down, dm_val, global_sample)
        for data, freq_down, dm_val, global_sample in requests
    ]
    if not prepared:
        return []

    classified = classify_patches(cls_model, [p[0] for p in prepared])
    return [
        (
            proc_patch,
            float(class_prob),
            snr_val,
            peak_idx_patch,
            _width_ms(peak_idx_patch, best_w_vec, time_reso_ds),
            start_sample,
        )
        for (_, start_sample, snr_val, peak_idx_patch, best_w_vec), (class_prob, proc_patch)
        in zip(prepared, classified)
    ]


def finalize_patch(
    data: np.ndarray,
    freq_down: np.ndarray,
    dm_val: float,
    global_sample: int,
    cls_model,
    time_reso_ds: float,
) -> Tuple[Optional[np.ndarray], float, float, Optional[int], Optional[float], Optional[int]]:
    """Dedisperse → SNR → classify → width.

    Returns
    -------
    proc_patch      : preprocessed patch array (or None)
    class_prob      : classifier probability in [0, 1]
    snr_val         : peak SNR on the dedispersed patch (0.0 if unavailable)
    peak_idx_patch  : sample index of SNR peak within the patch (None if unavailable)
    width_ms        : estimated pulse width in ms (None if unavailable)
    start_sample    : first sample of the dedispersed patch (None if patch is None)
    """
    patch, start_sample, snr_val, peak_idx_patch, best_w_vec = _dedisperse_and_measure(
        data, freq_down, dm_val, global_sample
    )
    class_prob, proc_patch = classify_patch(cls_model, patch)
    width_ms = _width_ms(peak_idx_patch, best_w_vec, time_reso_ds)
    return proc_patch, float(class_prob), snr_val, peak_idx_patch, width_ms, start_sample
