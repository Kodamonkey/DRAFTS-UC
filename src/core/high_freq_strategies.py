"""High-frequency FRB detection strategies inspired by Yong-Kun Zhang.

This module implements two complementary approaches tailored for the
millimetre regime where dispersion sweeps collapse and traditional
time--DM heuristics lose discriminative power:

* Strategy 1 expands the DM search space and detects "bow-tie" patterns
  on the time--DM plane with adaptive contrast metrics.
* Strategy 2 performs a permissive zero-DM search followed by strict
  DM-aware validation that enforces astrophysical consistency tests.

Both strategies emit :class:`HighFreqCandidate` instances which are
later deduplicated and ranked before being persisted by the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import logging
import math

import numpy as np

from ..analysis.snr_utils import compute_snr_profile
from ..config import config
from ..detection.model_interface import classify_patch
from ..preprocessing.dedispersion import d_dm_time_g, dedisperse_block, dedisperse_patch


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExpandedDMRange:
    """Descriptor for an expanded DM search space."""

    min_dm: float
    max_dm: float
    step: float
    n_trials: int
    dm_values: np.ndarray

    def clamp_index(self, idx: int) -> int:
        return int(max(0, min(self.n_trials - 1, idx)))

    def index_to_dm(self, idx: int) -> float:
        idx = self.clamp_index(idx)
        return float(self.dm_values[idx])

    def dm_to_index(self, dm: float) -> int:
        if self.n_trials <= 1 or self.max_dm <= self.min_dm:
            return 0
        ratio = (float(dm) - self.min_dm) / (self.max_dm - self.min_dm)
        idx = int(round(ratio * (self.n_trials - 1)))
        return self.clamp_index(idx)


@dataclass
class HighFreqCandidate:
    """Container for candidates produced by the high-frequency pipeline."""

    chunk_idx: int
    slice_idx: int
    band_idx: int
    time_idx: int
    dm_idx: int
    dm: float
    method: str
    absolute_time: float
    global_sample: int
    snr: float
    class_prob: float
    confidence: float
    width_samples: int | None = None
    validation_results: dict[str, str] = field(default_factory=dict)
    final_dm: float | None = None
    initial_dm: float | None = None
    contrast: float | None = None
    candidate_box: tuple[int, int, int, int] | None = None
    processed_patch: np.ndarray | None = None

    def effective_dm(self) -> float:
        return float(self.final_dm if self.final_dm is not None else self.dm)

    def is_burst(self) -> bool:
        return bool(self.class_prob >= float(config.CLASS_PROB))


def calculate_expanded_dm_range(
    traditional_max_dm: float,
    multiplier: float,
    step_coarse: float,
    *,
    min_dm: float = 0.0,
    min_step: float = 1.0,
) -> ExpandedDMRange:
    """Return an expanded DM range compensating the suppressed dispersion."""

    base_max = max(float(traditional_max_dm), 1.0)
    multiplier = max(float(multiplier), 1.0)
    max_dm = base_max * multiplier
    step = max(float(step_coarse), float(min_step), 1.0)

    n_trials = int(max(1, math.floor((max_dm - min_dm) / step) + 1))
    adjusted_max = float(min_dm + step * (n_trials - 1))
    dm_values = np.linspace(min_dm, adjusted_max, n_trials, dtype=np.float32)

    logger.debug(
        "Expanded DM range: min=%.2f max=%.2f step=%.2f trials=%d",
        min_dm,
        adjusted_max,
        step,
        n_trials,
    )

    return ExpandedDMRange(
        min_dm=float(min_dm),
        max_dm=adjusted_max,
        step=float(step),
        n_trials=n_trials,
        dm_values=dm_values,
    )


def generate_time_dm_plane(block: np.ndarray, dm_range: ExpandedDMRange) -> np.ndarray:
    """Generate the time--DM cube for the expanded DM range."""

    height = int(dm_range.n_trials)
    width = int(block.shape[0])
    return d_dm_time_g(
        block,
        height=height,
        width=width,
        dm_min=dm_range.min_dm,
        dm_max=dm_range.max_dm,
    )


def _compute_bow_tie_contrast(
    dm_profile: np.ndarray,
    peak_idx: int,
    wing_width: int,
) -> float:
    left_lo = max(0, peak_idx - wing_width)
    left_hi = max(left_lo, peak_idx)
    right_lo = min(dm_profile.size, peak_idx + 1)
    right_hi = min(dm_profile.size, peak_idx + wing_width + 1)

    wings: List[float] = []
    if left_hi > left_lo:
        wings.append(float(np.mean(np.abs(dm_profile[left_lo:left_hi]))))
    if right_hi > right_lo:
        wings.append(float(np.mean(np.abs(dm_profile[right_lo:right_hi]))))
    wing_level = float(np.mean(wings)) if wings else 0.0
    if wing_level <= 0.0:
        return 0.0
    return float(dm_profile[peak_idx]) / wing_level


def run_bow_tie_strategy(
    band_img: np.ndarray,
    dm_range: ExpandedDMRange,
    *,
    block: np.ndarray,
    freq_down: np.ndarray,
    start_idx: int,
    absolute_start_time: float,
    cls_model,
    chunk_idx: int,
    slice_idx: int,
    band_idx: int,
    dt_ds: float,
) -> list[HighFreqCandidate]:
    """Detect bow-tie candidates on a DM-expanded time--DM plane."""

    if band_img is None or band_img.size == 0:
        return []

    wing_width = int(getattr(config, "HF_BOW_TIE_WING_WIDTH", 20))
    min_idx = int(getattr(config, "HF_BOW_TIE_MIN_DM_INDEX", 10))
    min_snr = float(getattr(config, "HF_BOW_TIE_MIN_SNR", 5.0))
    threshold = float(getattr(config, "HF_BOW_TIE_THRESHOLD", 2.0))

    height, width = band_img.shape
    scale_x = 512.0 / max(1, width)
    scale_y = 512.0 / max(1, height)
    half_w = max(4, width // 64)
    half_h = max(3, height // 32)

    candidates: list[HighFreqCandidate] = []

    for t_idx in range(width):
        dm_profile = band_img[:, t_idx]
        if dm_profile.size == 0:
            continue
        peak_idx = int(np.argmax(dm_profile))
        if peak_idx < min_idx:
            continue
        contrast = _compute_bow_tie_contrast(dm_profile, peak_idx, wing_width)
        if contrast < threshold:
            continue

        dm_val = dm_range.index_to_dm(peak_idx)
        global_sample = int(start_idx + t_idx)

        patch, _ = dedisperse_patch(block, freq_down, dm_val, global_sample)
        snr_profile, _, width_vec = compute_snr_profile(patch)
        if snr_profile.size == 0:
            continue
        peak_pos = int(np.argmax(snr_profile))
        snr_val = float(snr_profile[peak_pos])
        if snr_val < min_snr:
            continue

        width_samples = int(width_vec[peak_pos]) if width_vec.size > peak_pos else None
        class_prob, proc_patch = classify_patch(cls_model, patch)

        conf = min(0.99, max(0.05, contrast / max(threshold, 1e-3)))
        x1 = int(round(max(0, t_idx - half_w) * scale_x))
        x2 = int(round(min(width - 1, t_idx + half_w) * scale_x))
        y1 = int(round(max(0, peak_idx - half_h) * scale_y))
        y2 = int(round(min(height - 1, peak_idx + half_h) * scale_y))
        candidate = HighFreqCandidate(
            chunk_idx=chunk_idx,
            slice_idx=slice_idx,
            band_idx=band_idx,
            time_idx=t_idx,
            dm_idx=peak_idx,
            dm=float(dm_val),
            method="bow_tie_expanded",
            absolute_time=float(absolute_start_time + t_idx * dt_ds),
            global_sample=global_sample,
            snr=snr_val,
            class_prob=float(class_prob),
            confidence=float(conf),
            width_samples=width_samples,
            final_dm=float(dm_val),
            initial_dm=float(dm_val),
            contrast=float(contrast),
            candidate_box=(x1, y1, x2, y2),
            processed_patch=proc_patch,
        )
        candidates.append(candidate)

    return candidates


def zero_dm_classifier(
    block: np.ndarray,
    freq_down: np.ndarray,
    start_idx: int,
    end_idx: int,
    *,
    cls_model,
    absolute_start_time: float,
    chunk_idx: int,
    slice_idx: int,
    dt_ds: float,
    dm_range: ExpandedDMRange,
) -> list[HighFreqCandidate]:
    """Permissive zero-DM classification stage."""

    trials: Sequence[float] = getattr(config, "HF_ZERO_DM_TRIALS", [0.0, 1.0, 2.0, 5.0])
    sensitivity = float(getattr(config, "HF_ZERO_DM_SENSITIVITY", 0.4))
    min_snr = float(getattr(config, "HF_ZERO_DM_MIN_SNR", 3.0))
    max_candidates = int(getattr(config, "HF_ZERO_DM_MAX_CANDIDATES", 1000))

    width = max(0, int(end_idx - start_idx))
    scale_x = 512.0 / max(1, width)
    half_w = max(4, width // 64)
    height = dm_range.n_trials
    scale_y = 512.0 / max(1, height)
    half_h = max(3, height // 32)

    provisional: list[HighFreqCandidate] = []

    for dm_trial in trials:
        dedisp_block = dedisperse_block(block, freq_down, float(dm_trial), start_idx, width)
        if dedisp_block is None or dedisp_block.size == 0:
            continue
        snr_profile, _, width_vec = compute_snr_profile(dedisp_block)
        if snr_profile.size == 0:
            continue

        order = np.argsort(snr_profile)[::-1]
        for idx in order:
            snr_val = float(snr_profile[idx])
            if snr_val < min_snr:
                break
            global_sample = int(start_idx + idx)
            patch, _ = dedisperse_patch(block, freq_down, float(dm_trial), global_sample)
            class_prob, proc_patch = classify_patch(cls_model, patch)
            if class_prob < sensitivity:
                continue
            width_samples = int(width_vec[idx]) if width_vec.size > idx else None
            dm_idx = dm_range.dm_to_index(dm_trial)
            y1 = int(round(max(0, dm_idx - half_h) * scale_y))
            y2 = int(round(min(height - 1, dm_idx + half_h) * scale_y))
            x1 = int(round(max(0, idx - half_w) * scale_x))
            x2 = int(round(min(width - 1, idx + half_w) * scale_x))
            candidate = HighFreqCandidate(
                chunk_idx=chunk_idx,
                slice_idx=slice_idx,
                band_idx=0,
                time_idx=int(idx),
                dm_idx=dm_idx,
                dm=float(dm_trial),
                method="zero_dm_seed",
                absolute_time=float(absolute_start_time + idx * dt_ds),
                global_sample=global_sample,
                snr=snr_val,
                class_prob=float(class_prob),
                confidence=float(class_prob),
                width_samples=width_samples,
                initial_dm=float(dm_trial),
                candidate_box=(x1, y1, x2, y2),
                processed_patch=proc_patch,
            )
            provisional.append(candidate)

    provisional.sort(key=lambda c: (c.class_prob, c.snr), reverse=True)
    if len(provisional) > max_candidates:
        provisional = provisional[:max_candidates]
    return provisional


def _iter_dm_grid(dm_min: float, dm_max: float, step: float) -> Iterable[float]:
    if step <= 0:
        step = 1.0
    dm = float(dm_min)
    while dm <= dm_max + 1e-6:
        yield dm
        dm += step


def find_best_dm(
    candidate: HighFreqCandidate,
    block: np.ndarray,
    freq_down: np.ndarray,
    dt_ds: float,
) -> tuple[float, float, int | None]:
    """Return the DM that maximises S/N for the candidate."""

    dm_min = float(getattr(config, "HF_VALIDATION_DM_MIN", 0.0))
    dm_max = float(getattr(config, "HF_VALIDATION_DM_MAX", config.DM_max))
    dm_step = float(getattr(config, "HF_VALIDATION_DM_STEP", 5.0))
    patch_len = int(getattr(config, "HF_VALIDATION_PATCH_LEN", 256))

    best_dm = float(candidate.dm)
    best_snr = float(candidate.snr)
    best_width: int | None = candidate.width_samples

    for dm_val in _iter_dm_grid(dm_min, dm_max, dm_step):
        patch, _ = dedisperse_patch(block, freq_down, dm_val, candidate.global_sample, patch_len)
        if patch is None or patch.size == 0:
            continue
        snr_profile, _, width_vec = compute_snr_profile(patch)
        if snr_profile.size == 0:
            continue
        idx = int(np.argmax(snr_profile))
        snr_val = float(snr_profile[idx])
        if snr_val > best_snr:
            best_snr = snr_val
            best_dm = float(dm_val)
            best_width = int(width_vec[idx]) if width_vec.size > idx else best_width

    return best_dm, best_snr, best_width


def _split_frequency_subbands(
    block: np.ndarray,
    freq_down: np.ndarray,
    n_subbands: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_subbands = max(1, min(int(n_subbands), freq_down.size))
    subbands: list[tuple[np.ndarray, np.ndarray]] = []
    for sub_idx in range(n_subbands):
        start = int(np.floor(sub_idx * freq_down.size / n_subbands))
        end = int(np.floor((sub_idx + 1) * freq_down.size / n_subbands))
        if end <= start:
            continue
        subbands.append((block[:, start:end], freq_down[start:end]))
    return subbands


def validate_subband_consistency(
    candidate: HighFreqCandidate,
    block: np.ndarray,
    freq_down: np.ndarray,
) -> bool:
    n_subbands = int(getattr(config, "HF_SUBBAND_COUNT", 4))
    threshold = float(getattr(config, "HF_SUBBAND_SNR_THRESHOLD", 5.0))
    min_ratio = float(getattr(config, "HF_SUBBAND_CONSISTENCY_THRESHOLD", 0.75))

    subbands = _split_frequency_subbands(block, freq_down, n_subbands)
    detections = 0
    for sub_block, sub_freq in subbands:
        patch, _ = dedisperse_patch(sub_block, sub_freq, candidate.effective_dm(), candidate.global_sample)
        if patch is None or patch.size == 0:
            continue
        snr_profile, _, _ = compute_snr_profile(patch)
        if snr_profile.size == 0:
            continue
        snr_val = float(np.max(snr_profile))
        if snr_val >= threshold:
            detections += 1
    if not subbands:
        return False
    ratio = detections / len(subbands)
    return ratio >= min_ratio


def validate_temporal_consistency(
    candidate: HighFreqCandidate,
    block: np.ndarray,
    freq_down: np.ndarray,
    dt_ds: float,
) -> bool:
    window_sec = float(getattr(config, "HF_TEMPORAL_CHUNK_SEC", 30.0))
    window_samples = max(1, int(window_sec / dt_ds))
    start = max(0, candidate.global_sample - window_samples // 2)
    end = min(block.shape[0], start + window_samples)
    if end <= start:
        return False
    dedisp_block = dedisperse_block(block, freq_down, candidate.effective_dm(), start, end - start)
    if dedisp_block is None or dedisp_block.size == 0:
        return False
    snr_profile, _, _ = compute_snr_profile(dedisp_block)
    if snr_profile.size == 0:
        return False
    rel_idx = candidate.global_sample - start
    if rel_idx < 0 or rel_idx >= snr_profile.size:
        return False
    snr_val = float(snr_profile[rel_idx])
    threshold = float(getattr(config, "HF_TEMPORAL_SNR_THRESHOLD", 5.0))
    return snr_val >= threshold


def dm_aware_validation(
    candidates: Sequence[HighFreqCandidate],
    block: np.ndarray,
    freq_down: np.ndarray,
    dt_ds: float,
    dm_range: ExpandedDMRange,
) -> list[HighFreqCandidate]:
    """Apply DM-aware validation to permissive zero-DM candidates."""

    validated: list[HighFreqCandidate] = []
    min_significant_dm = float(getattr(config, "HF_MIN_SIGNIFICANT_DM", 15.0))

    for candidate in candidates:
        best_dm, best_snr, best_width = find_best_dm(candidate, block, freq_down, dt_ds)
        candidate.final_dm = best_dm
        candidate.snr = best_snr
        candidate.width_samples = best_width
        candidate.dm_idx = dm_range.dm_to_index(best_dm)
        candidate.method = "zero_dm_validated"

        validation = {}
        if best_dm <= min_significant_dm:
            validation["dm_test"] = "FAILED"
            candidate.validation_results = validation
            continue
        validation["dm_test"] = "PASSED"

        if validate_subband_consistency(candidate, block, freq_down):
            validation["subband_test"] = "PASSED"
        else:
            validation["subband_test"] = "FAILED"

        if validation["subband_test"] == "FAILED":
            candidate.validation_results = validation
            continue

        if validate_temporal_consistency(candidate, block, freq_down, dt_ds):
            validation["temporal_test"] = "PASSED"
        else:
            validation["temporal_test"] = "FAILED"

        candidate.validation_results = validation
        if all(val == "PASSED" for val in validation.values()):
            candidate.dm = best_dm
            candidate.confidence = max(candidate.confidence, candidate.class_prob)
            validated.append(candidate)

    return validated


def deduplicate_and_rank_candidates(
    candidates: Sequence[HighFreqCandidate],
) -> list[HighFreqCandidate]:
    if not candidates:
        return []
    time_tol = float(getattr(config, "HF_DEDUP_TIME_TOL_SEC", 1.0))
    dm_tol = float(getattr(config, "HF_DEDUP_DM_TOL", 50.0))

    groups: list[list[HighFreqCandidate]] = []
    for cand in candidates:
        placed = False
        for group in groups:
            ref = group[0]
            if abs(cand.absolute_time - ref.absolute_time) <= time_tol and abs(cand.effective_dm() - ref.effective_dm()) <= dm_tol:
                group.append(cand)
                placed = True
                break
        if not placed:
            groups.append([cand])

    ranked: list[HighFreqCandidate] = []
    for group in groups:
        def score(entry: HighFreqCandidate) -> tuple:
            passed = sum(1 for v in entry.validation_results.values() if v == "PASSED")
            return (entry.snr, passed, entry.class_prob, entry.confidence)

        best = max(group, key=score)
        ranked.append(best)

    ranked.sort(key=lambda c: (c.snr, c.class_prob, c.confidence), reverse=True)
    return ranked


def monitor_pipeline_health(processing_results: dict, config_module) -> list[dict]:
    """Evaluate monitoring metrics for the integrated high-frequency pipeline."""

    alerts: list[dict] = []
    stats = processing_results.get("processing_stats", {})

    strat1 = float(stats.get("strategy_1_count", 0))
    strat2_initial = float(stats.get("strategy_2_initial_count", 0))
    strat2_valid = float(stats.get("strategy_2_validated_count", 0))
    total_time = float(stats.get("total_processing_time", 0.0))
    max_time = float(getattr(config_module, "MAX_PROCESSING_TIME_SEC", getattr(config_module, "max_processing_time_sec", 0)))

    ratio_threshold = float(getattr(config_module, "HF_MONITOR_BOW_TIE_RATIO", 0.1))
    if strat2_initial > 0:
        bow_tie_ratio = strat1 / max(strat2_initial, 1.0)
        if bow_tie_ratio < ratio_threshold:
            alerts.append({
                "level": "WARNING",
                "message": "Bow-tie recovery rate below expected threshold",
                "recommendation": "Increase HF_DM_EXPANSION_FACTOR or adjust contrast threshold",
            })

    if strat2_initial > 0:
        false_positive_rate = 1.0 - (strat2_valid / strat2_initial)
        if false_positive_rate > 0.9:
            alerts.append({
                "level": "ERROR",
                "message": f"Zero-DM false positive rate {false_positive_rate:.0%} exceeds limit",
                "recommendation": "Tighten HF_ZERO_DM_SENSITIVITY or validation thresholds",
            })

    if max_time and total_time > max_time:
        alerts.append({
            "level": "WARNING",
            "message": f"Processing time {total_time:.1f}s exceeds configured maximum",
            "recommendation": "Reduce DM sampling resolution or optimise dedispersion",
        })

    return alerts

