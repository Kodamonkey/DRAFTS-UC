from __future__ import annotations

# Standard library imports
from dataclasses import dataclass
from pathlib import Path
import gc
import logging
import time

# Third-party imports
import numpy as np

# Local imports
from ..config import config
from ..analysis.snr_utils import compute_snr_profile, find_snr_peak
from ..analysis.science_metrics import physical_consistency_score, post_trials_sigma
from ..detection.model_interface import classify_patch
from ..domain.physics import K_DM_MS
from ..log_utils.logging_config import get_global_logger
from ..output.candidate_manager import (
    Candidate,
    CandidateWriter,
    append_candidate,
    rotate_previous_candidates,
)
from ..output.phase_metrics import PhaseMetricsTracker
from ..preprocessing.dedispersion import dedisperse_block, dedisperse_patch
from ..visualization.visualization_unified import preprocess_img, postprocess_img
from .candidate_finalization import finalize_patch as _finalize_patch
from .contracts import DMGrid, ObservationMetadata, PipelineConfigSnapshot
from .file_driver import (
    ChunkLoopState,
    begin_resumable_run,
    checkpoint_completed_chunk,
    DetectionStats,
    begin_chunk,
    check_file_length,
    compute_overlap_raw,
    error_result,
    export_validation_metrics,
    finalize_file_status,
    finish_chunk,
    log_chunk_arrival_latency,
    optimize_memory,
    prepare_chunked_run,
    record_chunk_failure,
    record_oom,
    start_arrival_clock,
    status_for_error,
)
from .mjd_utils import calculate_candidate_mjd

logger = logging.getLogger(__name__)




def _find_snr_peaks(snr_profile: np.ndarray, threshold: float, min_distance: int = 16) -> list[int]:
    """Return time indices where the SNR exceeds the threshold and is a local maximum."""
    if snr_profile is None or snr_profile.size == 0:
        return []
    peaks: list[int] = []
    n = snr_profile.size
    for i in range(1, n - 1):
        if snr_profile[i] >= threshold and snr_profile[i] >= snr_profile[i - 1] and snr_profile[i] >= snr_profile[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_distance:
                peaks.append(i)
    return peaks


def _fmt_prob(prob: float | None) -> str:
    """Format a classification probability for a log line.

    A phase that was disabled, or that found no patch to classify, has no
    probability at all. ``"n/a"`` says so; formatting ``None`` with ``%.2f``
    would raise, and substituting a number would be the very thing P1-10 was.
    """
    return "n/a" if prob is None else f"{prob:.2f}"


def decide_candidate(
    *,
    has_intensity_result: bool,
    has_linear_result: bool,
    is_burst_intensity: bool | None,
    class_prob_intensity: float | None,
    is_burst_linear: bool | None,
    class_prob_linear: float | None,
    enable_linear_class: bool,
    save_only_burst: bool,
) -> tuple[bool, bool | None, str]:
    """Decide whether to keep a high-frequency candidate, and what to call it.

    Returns ``(should_save, is_burst, save_reason)``.

    ``is_burst`` is the verdict that reaches the CSV, and it is produced here,
    by the same branches that produce ``should_save``. They were two separate
    decisions until audit finding P1-10: ``should_save`` consulted whichever
    phase had actually run, while ``is_burst`` was an unconditional alias of the
    Intensity verdict. With Phase 3a disabled that alias was a hardcoded
    ``True``, so every saved candidate was labelled a burst no matter what the
    Linear classifier had concluded, and the burst counter equalled the
    candidate counter.

    ``is_burst`` is ``None`` when no phase produced a verdict -- an absent
    classification, not a negative one.

    The function is pure so the four availability combinations can be tested
    without driving the whole pipeline.
    """
    if has_intensity_result and has_linear_result:
        if save_only_burst:
            # STRICT: a burst in Intensity AND in Linear.
            is_burst = bool(is_burst_intensity and is_burst_linear)
            if is_burst:
                reason = f"BURST in I+L (p_I={class_prob_intensity:.2f}, p_L={class_prob_linear:.2f})"
            else:
                reason = (
                    f"Filtered: I={'BURST' if is_burst_intensity else 'NO'}({class_prob_intensity:.2f}), "
                    f"L={'BURST' if is_burst_linear else 'NO'}({class_prob_linear:.2f})"
                )
        else:
            # PERMISSIVE: a burst in either.
            is_burst = bool(is_burst_intensity or is_burst_linear)
            if is_burst:
                reason = f"BURST: I={is_burst_intensity}({class_prob_intensity:.2f}), L={is_burst_linear}({class_prob_linear:.2f})"
            else:
                reason = f"NO-BURST in both: I={class_prob_intensity:.2f}, L={class_prob_linear:.2f}"
        return is_burst, is_burst, reason

    if has_intensity_result:
        is_burst = bool(is_burst_intensity)
        reason = f"{'BURST' if is_burst else 'NO-BURST'} in I({class_prob_intensity:.2f})"
        if enable_linear_class:
            reason += " [Linear N/A - no multi-pol data]"
        return (not save_only_burst or is_burst), is_burst, reason

    if has_linear_result:
        is_burst = bool(is_burst_linear)
        reason = f"{'BURST' if is_burst else 'NO-BURST'} in L({class_prob_linear:.2f})"
        reason += " [Intensity disabled]"
        return (not save_only_burst or is_burst), is_burst, reason

    # Neither phase produced a verdict. Reachable: Phase 3a disabled and
    # Phase 3b enabled but the file carries no multi-polarisation data.
    return False, None, "ERROR: No classification available"


def dm_for_dedispersion(dm_val: float | None) -> float:
    """The DM to dedisperse at, given what the band was able to measure.

    A DM the band could not resolve is ``NaN`` (``resolve_candidate_dm`` under
    the default policy, which is the normal outcome at these frequencies: the
    sweep is under one sample and SPEC-HF-002 skips the cube entirely). ``NaN``
    is not a dispersion measure and must not be handed to a dedisperser:
    ``dedisperse_block`` and ``dedisperse_patch`` both build their per-channel
    delays with ``(K_DM_MS * dm * ...).round().astype(np.int64)``, and
    ``NaN.round().astype(np.int64)`` is undefined -- it is where the
    ``invalid value encountered in cast`` warning in every high-frequency run
    comes from, and the delays it produces index the block arbitrarily.

    Zero is the honest substitute: no measured dispersion, so no correction
    applied. The waterfall is shown as it was recorded rather than shifted by a
    number nobody measured.

    This exists as a function because the per-candidate patch and the
    whole-block plot each used to decide it for themselves, and only one of them
    checked. They then dedispersed the same candidate at different DMs -- the
    patch at 0.0, the figure beside it at NaN.
    """

    if dm_val is None:
        return 0.0
    value = float(dm_val)
    return value if np.isfinite(value) else 0.0


def _dm_from_image_at_time(dm_time_band_img: np.ndarray, time_idx: int) -> float:
    """
    Map a time index to the DM row with the highest intensity.
    
    The DM-time cube is constructed with uniform DM distribution:
    - Row 0 corresponds to DM_min
    - Row (height-1) corresponds to DM_max
    - Row i corresponds to: DM_min + (i / (height-1)) * (DM_max - DM_min)
    
    This function finds the row with maximum intensity at the given time,
    then maps that row index to the actual DM value.
    """
    h, w = dm_time_band_img.shape[:2]
    t = int(max(0, min(w - 1, time_idx)))
    
    # Get the intensity column at time t (all DM values at this time)
    intensity_column = dm_time_band_img[:, t]
    
    # Find the row (DM) with maximum intensity
    row_idx = int(np.argmax(intensity_column))
    max_intensity = float(intensity_column[row_idx])
    
    # Diagnostic: Check if intensity is uniform or if max is always at row 0
    intensity_min = float(np.min(intensity_column))
    intensity_max = float(np.max(intensity_column))
    intensity_mean = float(np.mean(intensity_column))
    intensity_std = float(np.std(intensity_column))
    
    logger.info(
        f"[DM_CALC_DIAG] Column at t={t}: row_idx={row_idx}, "
        f"intensity stats: min={intensity_min:.3f}, max={intensity_max:.3f}, "
        f"mean={intensity_mean:.3f}, std={intensity_std:.3f}, "
        f"max_at_row0={row_idx == 0}"
    )
    
    # Map row index to DM via DMGrid contract (SPEC-DM-005)
    dm_min = float(config.DM_min)
    dm_max = float(config.DM_max)
    grid = DMGrid.from_values(np.linspace(dm_min, dm_max, h))
    dm_val = grid.dm_for_row(row_idx, height=h)

    logger.info(
        f"[DM_CALC] time_idx={time_idx}, row_idx={row_idx}/{h-1}, "
        f"intensity={max_intensity:.3f}, DM={dm_val:.2f} "
        f"(range: {dm_min:.2f}-{dm_max:.2f}, DM_mid={(dm_min+dm_max)/2:.2f})"
    )

    return float(dm_val)


# --------------------------------------------------------------------------- #
# the per-candidate steps, extracted from the band loop (audit REF-05)
#
# ``snr_detect_and_classify_candidates_in_band`` was 866 lines carrying about
# forty live locals, and it had reached the point of asking ``'x' in locals()``
# to find out what its own earlier branches had done. Each function below is one
# step of the per-candidate work, taking what it needs as keyword-only arguments
# and reading no configuration the caller could have passed -- the shape
# ``decide_candidate`` above already uses, for the same reason: the four
# combinations of "which phase ran" can then be exercised without a model, a
# file and a DM-time cube.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class IntensityVerdict:
    """What Phase 3a concluded about one candidate, or that it concluded nothing."""

    proc_patch: np.ndarray | None
    class_prob: float | None
    is_burst: bool | None
    snr_val: float
    width_ms: float | None


@dataclass(frozen=True)
class LinearVerdict:
    """What Phase 3b concluded about one candidate, or that it concluded nothing."""

    proc_patch: np.ndarray | None
    class_prob: float | None
    is_burst: bool | None
    snr_val: float | None


@dataclass(frozen=True)
class CandidateScore:
    """The trials correction and physical plausibility of one candidate."""

    n_trials: int
    post_sigma: float
    physical_score: float
    rank_score: float


def _empty_band_result(metrics_tracker: PhaseMetricsTracker | None) -> dict:
    """The band result for a slice that produced no candidate.

    Both early exits -- no SNR peak in Intensity at all, and no peak surviving
    the Phase 2 Linear check -- returned this dict, spelled out twice. Every key
    here is one that ``process_slice_with_multiple_bands_high_freq`` reads with
    ``[]`` rather than ``.get()``, so a key dropped from one copy turns a quiet
    "nothing found" into a KeyError on the path that finds nothing, which is the
    common one.
    """

    return {
        "top_conf": [],
        "top_boxes": [],
        "class_probs_list": [],
        "first_patch": None,
        "first_start": None,
        "first_dm": None,
        "img_tensor": None,
        "cand_counter": 0,
        "n_bursts": 0,
        "n_no_bursts": 0,
        "prob_max": 0.0,
        "patch_path": None,
        "best_is_burst": False,
        "total_candidates": 0,
        "candidate_times_abs": [],
        "phase_metrics": metrics_tracker if metrics_tracker else None,
    }


def locate_candidate_box(
    *,
    peak_idx: int,
    img_w: int,
    img_h: int,
    half_w: int,
    scale_x: float,
    scale_y: float,
    effective_len: int,
    time_reso_ds: float,
) -> tuple[int, tuple[int, int, int, int], int, float]:
    """Place one SNR peak in the DM-time image and in time.

    Returns ``(cx, box, t_sample_real, t_sec_real)``: the peak's column in the
    band image, its box in the 512x512 frame the classifier sees, and the sample
    and second the box centre corresponds to.

    The box spans the **full** DM axis, row 0 to ``img_h - 1``, and that is the
    regression this shape prevents. A box drawn around an assumed DM makes the
    DM read back out of it a function of the box, so the reported DM tracks the
    guess rather than the data and clusters at whatever edge the box happened to
    have. Detection here comes from boxcar matching, which locates a peak in
    time only, so the DM axis is left whole and the DM is measured separately by
    ``resolve_candidate_dm``.

    ``effective_len`` is the slice's real sample count, which is not always
    ``slice_len``: the last slice of a chunk is short, and scaling its columns by
    the nominal length would place every candidate in it late by the difference.
    """

    cx = int(max(0, min(img_w - 1, peak_idx)))

    x1_raw = max(0, cx - half_w)
    x2_raw = min(img_w - 1, cx + half_w)
    y1_raw = 0  # Start at DM_min (row 0)
    y2_raw = img_h - 1  # End at DM_max (row img_h-1)

    # Time from the centre of the box.
    center_x = (x1_raw + x2_raw) / 2.0
    sample_off = (center_x / max(img_w - 1, 1)) * effective_len
    t_sample_real = int(sample_off)
    t_sec_real = float(sample_off) * time_reso_ds

    # Transform the box to 512x512 coordinates to match img_rgb.
    box = (
        int(round(x1_raw * scale_x)),
        int(round(y1_raw * scale_y)),
        int(round(x2_raw * scale_x)),
        int(round(y2_raw * scale_y)),
    )
    return cx, box, t_sample_real, t_sec_real


def resolve_candidate_dm(
    *,
    dm_img: np.ndarray,
    cx: int,
    dm_min: float,
    dm_max: float,
    dm_policy: str,
) -> tuple[float, str, float | None]:
    """Return ``(dm_val, dm_status, dm_uncertainty)`` for the column at *cx*.

    At the frequencies this pipeline exists for, the dispersive sweep across the
    band can be smaller than one sample, and then every DM trial produces the
    same column: the cube is flat and the row of maximum intensity means nothing.
    That is what the ``col_std < 1e-6`` test detects, and why the answer is a
    status as well as a number.

    ``dm_status`` is the regression this shape prevents. Returning the midpoint
    of the DM range as though it had been measured -- which is what
    ``HIGH_FREQ_DM_POLICY`` still does when set to something other than
    ``unresolved``/``estimate_if_resolved`` -- puts a fabricated DM in the CSV
    that nothing downstream can distinguish from a real one. Under the default
    policy the value is NaN and the status says the band could not resolve it.

    The DM row lookup runs first and unconditionally, as it always did, because
    its two log lines are how a flat cube is recognised in a run log.
    """

    dm_val_approx = _dm_from_image_at_time(dm_img, cx)

    intensity_column = dm_img[:, cx] if cx < dm_img.shape[1] else dm_img[:, 0]
    col_std = float(np.std(intensity_column))

    if col_std >= 1e-6:
        logger.info(
            f"[DM_CALC] Calculated DM: {dm_val_approx:.2f} "
            f"(from _dm_from_image_at_time, std={col_std:.6f})"
        )
        return dm_val_approx, "measured", 0.5

    if str(dm_policy).lower() in {"unresolved", "estimate_if_resolved"}:
        dm_val: float = float("nan")
        dm_status = "unresolved_high_freq"
        dm_uncertainty: float | None = None
    else:
        dm_val = (dm_min + dm_max) / 2.0
        dm_status = "catalog_prior"
        dm_uncertainty = (dm_max - dm_min) / 2.0

    logger.info(
        f"[DM_CALC] No DM variation detected (std={col_std:.6f}) - typical at very high frequencies. "
        f"dm_status={dm_status} (range: {dm_min:.2f}-{dm_max:.2f})"
    )
    return dm_val, dm_status, dm_uncertainty


def classify_intensity_patch(
    *,
    enabled: bool,
    cls_model,
    data_block: np.ndarray,
    freq_down: np.ndarray,
    dm_for_dedisp: float,
    global_sample: int,
    time_reso_ds: float,
    snr_peak: float,
    class_prob_threshold: float,
) -> IntensityVerdict:
    """Phase 3a: dedisperse the Intensity patch at this DM and classify it.

    A disabled phase returns ``class_prob`` and ``is_burst`` as ``None``. That is
    the shape audit P1-10 is about: the disabled branch used to hand back
    in-range stand-ins -- ``0.0``/``False`` in one place and ``1.0``/``True`` in
    another -- which neither the CSV nor ``decide_candidate`` could tell apart
    from a real classification, so with Phase 3a switched off every saved
    candidate was labelled a burst and the burst count equalled the candidate
    count.

    ``snr_val`` is always a number, because it is what the candidate's
    ``snr_post_dedisp`` column holds: the dedispersed-patch SNR when there is
    one, and the waterfall peak *snr_peak* when the patch yielded nothing.

    ``time_reso_ds`` and ``class_prob_threshold`` are arguments rather than
    config reads so that the caller keeps deciding when configuration is
    consulted. The Phase 3a block used to recompute the time resolution from
    config per candidate under the name ``time_reso_ds``, which was also the
    enclosing function's parameter -- so it rebound it for every line that
    followed. The two values are equal today; nothing made them stay equal.
    """

    if not enabled:
        return IntensityVerdict(
            proc_patch=None,
            class_prob=None,
            is_burst=None,
            snr_val=float(snr_peak),
            width_ms=None,
        )

    # ``_finalize_patch`` also reports the in-patch peak index and the patch's
    # first sample; the band loop has never used either.
    proc_patch, class_prob, snr_from_patch, _peak_idx_patch, width_ms, _start_sample = _finalize_patch(
        data_block, freq_down, dm_for_dedisp, global_sample, cls_model, time_reso_ds
    )
    return IntensityVerdict(
        proc_patch=proc_patch,
        class_prob=class_prob,
        is_burst=class_prob >= float(class_prob_threshold),
        snr_val=snr_from_patch if snr_from_patch > 0.0 else float(snr_peak),
        width_ms=width_ms,
    )


def classify_linear_patch(
    *,
    enabled: bool,
    cls_model,
    data_block_linear: np.ndarray | None,
    freq_down: np.ndarray,
    dm_for_dedisp: float,
    global_sample: int,
    class_prob_threshold: float,
) -> LinearVerdict:
    """Phase 3b: dedisperse the Linear-polarisation patch and classify it.

    Three distinct situations produce the same absent verdict -- the phase is
    disabled, the file carries no multi-polarisation data, or the dedispersed
    patch came back empty -- and they must, because none of them is a negative
    classification. Their difference is a log line at the call site, not a value
    here.

    The Linear threshold is separate from the Intensity one (``CLASS_PROB_LINEAR``
    against ``CLASS_PROB``) and arrives as an argument for the same reason as in
    ``classify_intensity_patch``: so that a caller can ask what this phase would
    conclude at a given threshold without reaching into configuration.
    """

    if not enabled or data_block_linear is None:
        return LinearVerdict(proc_patch=None, class_prob=None, is_burst=None, snr_val=None)

    patch_linear, _ = dedisperse_patch(data_block_linear, freq_down, dm_for_dedisp, global_sample)
    if patch_linear is None or patch_linear.size == 0:
        return LinearVerdict(proc_patch=None, class_prob=None, is_burst=None, snr_val=None)

    snr_val: float | None = None
    snr_profile_linear_patch, _, _ = compute_snr_profile(patch_linear)
    if snr_profile_linear_patch.size > 0:
        snr_val = float(np.max(snr_profile_linear_patch))

    class_prob, proc_patch = classify_patch(cls_model, patch_linear)
    return LinearVerdict(
        proc_patch=proc_patch,
        class_prob=class_prob,
        is_burst=class_prob >= class_prob_threshold,
        snr_val=snr_val,
    )


def score_candidate(
    *,
    snr_pre_dedisp: float,
    snr_post_dedisp: float,
    dm_status: str,
    linear_fraction: float | None,
    class_prob_intensity: float | None,
    class_prob_linear: float | None,
    n_snr_samples: int,
    dm_min: float,
    dm_max: float,
    trial_correction: str,
) -> CandidateScore:
    """Convert one candidate's measurements into its trials-corrected ranking.

    ``rank_score`` is the product of a morphology term and a physics term, and
    the morphology term is the *highest* probability among the phases that
    actually classified. Substituting a default for a phase that did not run
    would silently reorder the candidate list -- a 0.0 would bury every
    single-phase candidate, a 1.0 would float them all to the top -- so an absent
    probability is dropped from the maximum rather than replaced, and a candidate
    with no classification at all scores 0.0 and ranks last.

    ``dm_status`` reaches the physics term unchanged: a DM the band could not
    resolve must not be scored as though it had been measured.
    """

    n_trials = max(1, int((dm_max - dm_min + 1) * max(1, n_snr_samples)))
    post_sigma = post_trials_sigma(float(snr_post_dedisp), n_trials, trial_correction)
    phys_score = physical_consistency_score(
        post_sigma,
        snr_pre_dedisp,
        snr_post_dedisp,
        dm_status,
        linear_fraction,
    )

    available_probs = [p for p in (class_prob_intensity, class_prob_linear) if p is not None]
    morphology_prob = max(float(p) for p in available_probs) if available_probs else 0.0

    return CandidateScore(
        n_trials=n_trials,
        post_sigma=post_sigma,
        physical_score=phys_score,
        rank_score=morphology_prob * phys_score,
    )


def build_candidate_record(
    *,
    fits_name: str,
    chunk_idx: int | None,
    slice_idx: int,
    band_idx: int,
    conf: float,
    dm_val: float,
    dm_status: str,
    dm_uncertainty: float | None,
    detection_time_dm_time: float,
    peak_time_waterfall: float | None,
    t_sample_real: int,
    box: tuple[int, int, int, int],
    snr_waterfall: float | None,
    snr_waterfall_linear: float | None,
    snr_pre_dedisp: float,
    linear_fraction: float | None,
    intensity: IntensityVerdict,
    linear: LinearVerdict,
    score: CandidateScore,
    is_burst: bool | None,
    patch_file: str,
    mjd_data: dict,
) -> Candidate:
    """Assemble the CSV row for one high-frequency candidate.

    Every classification column is ``None`` or a value, never a stand-in, and
    that is the regression this function's shape prevents (audit P1-10). The
    three verdict-bearing arguments arrive as records rather than as a dozen
    loose floats and bools precisely so that "Phase 3b did not run" cannot be
    spelled one way here and another way in the decision that produced
    *is_burst*.

    The two time columns are two different measurements of the same event and
    both are kept: ``detection_time_dm_time`` is where the DM-time plot puts it,
    ``peak_time_waterfall`` is where the waterfall SNR peaks. They disagree by
    the width of the pulse, and collapsing them to one column is how a
    discrepancy that means something becomes a discrepancy that looks like a bug.
    """

    return Candidate(
        fits_name,
        chunk_idx if chunk_idx is not None else 0,
        slice_idx,
        band_idx,
        float(conf),
        float(dm_val),  # DM calculated with extract_candidate_dm (same as plot)
        float(detection_time_dm_time),  # Time from DM-time plot (same as plot label)
        peak_time_waterfall,  # Time from waterfall SNR peak (different method)
        int(t_sample_real),  # Sample index
        tuple(map(int, box)),
        snr_waterfall,  # SNR from waterfall raw (peak_snr_wf) - Intensity
        float(intensity.snr_val),  # SNR from dedispersed patch - Intensity
        snr_waterfall_linear=snr_waterfall_linear,
        snr_patch_dedispersed_linear=linear.snr_val,
        width_ms=intensity.width_ms,
        dm_uncertainty=dm_uncertainty,
        dm_status=dm_status,
        best_width_ms=intensity.width_ms,
        n_trials=score.n_trials,
        post_trials_sigma=score.post_sigma,
        snr_pre_dedisp=snr_pre_dedisp,
        snr_post_dedisp=float(intensity.snr_val),
        linear_fraction=linear_fraction,
        physical_score=score.physical_score,
        rank_score=score.rank_score,
        class_prob_intensity=None if intensity.class_prob is None else float(intensity.class_prob),
        is_burst_intensity=None if intensity.is_burst is None else bool(intensity.is_burst),
        class_prob_linear=None if linear.class_prob is None else float(linear.class_prob),
        is_burst_linear=None if linear.is_burst is None else bool(linear.is_burst),
        is_burst=None if is_burst is None else bool(is_burst),  # Final verdict, from the decision table
        patch_file=patch_file,
        mjd_utc=mjd_data.get('mjd_utc'),
        mjd_bary_utc=mjd_data.get('mjd_bary_utc'),
        mjd_bary_tdb=mjd_data.get('mjd_bary_tdb'),
        mjd_bary_utc_inf=mjd_data.get('mjd_bary_utc_inf'),
        mjd_bary_tdb_inf=mjd_data.get('mjd_bary_tdb_inf'),
        mjd_bary_status=mjd_data.get('mjd_bary_status'),
    )


def count_candidate(*, is_burst: bool | None) -> tuple[int, int, int]:
    """Return the ``(candidates, bursts, no_bursts)`` deltas for one candidate.

    ``is_burst`` is ``None`` when no classification phase produced a verdict, and
    an absent verdict is neither a burst nor a non-burst: the candidate is
    counted once and in neither column, so ``bursts + no_bursts`` can be less
    than ``candidates`` and that difference is the number of unclassified rows.

    Written the obvious way -- ``if is_burst: ... else: ...`` -- every absent
    verdict is filed under NO-BURST, which is the reporting half of audit P1-10:
    a run with Phase 3a and Phase 3b both off would report a confident column of
    non-detections it never made.
    """

    return 1, 1 if is_burst is True else 0, 1 if is_burst is False else 0


def snr_detect_and_classify_candidates_in_band(
    cls_model,
    band_img: np.ndarray,  # DM x time image used for visualisation
    waterfall_block: np.ndarray,  # time x frequency slice block (Intensity)
    slice_len: int,
    j: int,
    fits_path: Path,
    save_dir: Path,
    data_block: np.ndarray,  # decimated chunk block (Intensity)
    freq_down: np.ndarray,
    csv_file: Path,
    time_reso_ds: float,
    snr_list: list,
    absolute_start_time: float | None,
    patches_dir: Path | None,
    chunk_idx: int | None,
    band_idx: int,
    slice_start_idx: int,  # actual slice start in decimated samples
    waterfall_block_raw: np.ndarray | None = None,  # RAW multi-pol data (time, npol, chan) - SLICE
    data_block_raw: np.ndarray | None = None,  # RAW multi-pol data (time, npol, chan) - FULL CHUNK
    pol_type: str = "IQUV",
    slice_samples: int | None = None,  # actual slice samples (may differ from slice_len)
    dm_time_fullband: np.ndarray | None = None,  # DM-time cube band 0 (fullband average) for DM calculation
    metrics_tracker: PhaseMetricsTracker | None = None,  # Optional metrics tracker
    snapshot: PipelineConfigSnapshot | None = None,
) -> dict:
    """Detect candidates from SNR peaks with multi-polarization validation.
    
    3-Phase Detection Strategy:
    1. Phase 1: Detect SNR peaks in Intensity (Stokes I) - MANDATORY first step
    2. Phase 2: Re-evaluate same time samples in Linear Polarization - conditional
    3. Phase 3: ResNet classification - ONLY if peaks detected in BOTH polarizations
    """
    try:
        global_logger = get_global_logger()
    except Exception:
        global_logger = None

    # =========================================================================
    # PIPELINE CONFIGURATION SUMMARY
    # =========================================================================
    logger.info("=" * 80)
    logger.info("HIGH-FREQUENCY PIPELINE - CONFIGURATION SUMMARY")
    logger.info("=" * 80)
    logger.info("Phase 1 (Matched Filtering - Intensity): ALWAYS ENABLED")
    # REF-10. This read the mutable global twenty-one times across ten keys.
    # ``snapshot`` is optional so the one production caller passes the one it
    # already built, and a test can inject its own; without it the behaviour is
    # exactly what it was, because from_config resolves the same fallbacks the
    # reads below used to spell out inline.
    snap = PipelineConfigSnapshot.from_config(config) if snapshot is None else snapshot

    logger.info("  └─ SNR threshold (Intensity): %.1f sigma", snap.snr_thresh)
    
    # Read config values directly (not using getattr with defaults to catch errors)
    enable_phase2 = snap.enable_linear_validation
    # Log the actual value read from config for debugging
    logger.info("Config check: ENABLE_LINEAR_VALIDATION = %s (type: %s, hasattr: %s)", 
               enable_phase2, type(enable_phase2).__name__, hasattr(config, 'ENABLE_LINEAR_VALIDATION'))
    snr_threshold_linear = snap.snr_thresh_linear
    logger.info("Phase 2 (SNR Validation - Linear): %s", "ENABLED" if enable_phase2 else "DISABLED")
    if enable_phase2:
        logger.info("  └─ SNR threshold (Linear): %.1f sigma", snr_threshold_linear)
    
    # Read config values directly (not using getattr with defaults to catch errors)
    enable_intensity_class = snap.enable_intensity_classification
    enable_linear_class = snap.enable_linear_classification
    class_prob_linear_thresh = snap.class_prob_linear
    
    # Log all config values for debugging
    logger.info("Config check: ENABLE_INTENSITY_CLASSIFICATION = %s (hasattr: %s)", 
               enable_intensity_class, hasattr(config, 'ENABLE_INTENSITY_CLASSIFICATION'))
    logger.info("Config check: ENABLE_LINEAR_CLASSIFICATION = %s (hasattr: %s)", 
               enable_linear_class, hasattr(config, 'ENABLE_LINEAR_CLASSIFICATION'))
    logger.info("Config check: CLASS_PROB_LINEAR = %.2f", class_prob_linear_thresh)
    logger.info("Config check: SNR_THRESH_LINEAR = %.1f", snr_threshold_linear)
    
    logger.info("Phase 3a (Classification - Intensity): %s", "ENABLED" if enable_intensity_class else "DISABLED")
    if enable_intensity_class:
        logger.info("  └─ Classification threshold (Intensity): %.2f", snap.class_prob)
    
    logger.info("Phase 3b (Classification - Linear): %s", "ENABLED" if enable_linear_class else "DISABLED")
    if enable_linear_class:
        logger.info("  └─ Classification threshold (Linear): %.2f", class_prob_linear_thresh)
    
    logger.info("Decision Mode: %s", "STRICT (require ALL enabled phases)" if snap.save_only_burst else "PERMISSIVE (require ANY enabled phase)")
    logger.info("=" * 80)

    # =========================================================================
    # PHASE 1: SNR PEAK DETECTION IN INTENSITY (Stokes I) - MANDATORY
    # =========================================================================
    logger.info("Phase 1: SNR peak detection in Intensity (waterfall_block shape: %s)", waterfall_block.shape)
    
    # Compute the SNR profile on the waterfall (time × frequency) - INTENSITY
    snr_profile_intensity, _, _ = compute_snr_profile(waterfall_block)
    logger.debug("Calculated snr_profile_intensity: size=%d, shape=%s", 
                len(snr_profile_intensity) if snr_profile_intensity is not None else 0,
                snr_profile_intensity.shape if snr_profile_intensity is not None else None)
    peaks_intensity = _find_snr_peaks(snr_profile_intensity, snap.snr_thresh)
    
    # Ensure the first candidate corresponds to the main SNR peak used downstream.
    peak_snr_global, _, peak_idx_global = find_snr_peak(snr_profile_intensity)
    if peak_snr_global >= snap.snr_thresh:
        # Insert at the front if absent, otherwise move it to the front.
        peaks_intensity = [peak_idx_global] + [p for p in peaks_intensity if p != peak_idx_global]
    else:
        # If the global peak is below threshold, keep a sorted list (possibly empty).
        peaks_intensity = sorted(peaks_intensity, key=lambda p: snr_profile_intensity[p], reverse=True)

    if global_logger:
        band_names = ["Full Band", "Low Band", "High Band"]
        band_name = band_names[band_idx] if band_idx < len(band_names) else f"Band {band_idx}"
        global_logger.band_candidates(f"{band_name} (Intensity)", len(peaks_intensity))
    
    # Record Phase 1 metrics
    if metrics_tracker is not None:
        metrics_tracker.record_phase_1(len(peaks_intensity))
    
    # If no peaks detected in Intensity, return empty result immediately
    if len(peaks_intensity) == 0:
        logger.info("Phase 1: No peaks detected in Intensity - skipping phases 2 & 3")
        return _empty_band_result(metrics_tracker)

    # =========================================================================
    # PHASE 2: RE-EVALUATE IN LINEAR POLARIZATION - CONDITIONAL
    # =========================================================================
    from ..input.polarization_utils import extract_polarization_from_raw, has_full_polarization_data
    
    peaks_final = []  # Peaks that pass validation
    waterfall_block_linear = None  # Will be needed for Phase 3b
    # enable_phase2 already defined in configuration summary above
    
    # Extract Linear Polarization if multi-pol data available (needed for Phase 2 and 3b)
    has_multipol = waterfall_block_raw is not None and has_full_polarization_data(waterfall_block_raw, pol_type)
    
    # Extract both slice and full chunk in Linear polarization
    waterfall_block_linear = None  # Slice for Phase 2 (SNR validation)
    data_block_linear = None       # Full chunk for Phase 3b (dedispersion)
    snr_profile_linear = None      # SNR profile in Linear (calculated once, reused)
    
    if has_multipol:
        # Extract Linear from slice (for Phase 2 SNR validation)
        waterfall_block_linear = extract_polarization_from_raw(
            waterfall_block_raw, pol_type, "linear", default_index=0
        )
        # Remove polarization dimension for SNR computation
        waterfall_block_linear = waterfall_block_linear[:, 0, :]
        
        # Extract Linear from full chunk (for Phase 3b dedispersion)
        if data_block_raw is not None:
            data_block_linear_full = extract_polarization_from_raw(
                data_block_raw, pol_type, "linear", default_index=0
            )
            data_block_linear = data_block_linear_full[:, 0, :]
        else:
            logger.warning("data_block_raw not available for Phase 3b classification")
    
        # Compute SNR profile in Linear Polarization ONCE (reused in Phase 2 and for labels)
        try:
            # Verify waterfall blocks have compatible shapes
            if waterfall_block.shape[0] != waterfall_block_linear.shape[0]:
                logger.warning("Waterfall block size mismatch: Intensity=%d, Linear=%d. This may cause SNR profile size mismatch.",
                             waterfall_block.shape[0], waterfall_block_linear.shape[0])
            
            snr_profile_linear, _, _ = compute_snr_profile(waterfall_block_linear)
            logger.info("Calculated snr_profile_linear: size=%d (waterfall_block_linear shape=%s, waterfall_block shape=%s)", 
                        len(snr_profile_linear) if snr_profile_linear is not None else 0,
                        waterfall_block_linear.shape, waterfall_block.shape)
            
            # Verify that both profiles have the same size
            if snr_profile_intensity is not None and snr_profile_linear is not None:
                if len(snr_profile_intensity) != len(snr_profile_linear):
                    logger.warning("SNR profile size mismatch: Intensity=%d, Linear=%d. This will cause issues accessing Linear SNR by index.",
                                 len(snr_profile_intensity), len(snr_profile_linear))
                else:
                    logger.info("SNR profiles match: both have size=%d", len(snr_profile_intensity))
        except Exception as e:
            logger.warning(f"Could not calculate Linear SNR profile: {e}")
            snr_profile_linear = None
    
    # Phase 2: SNR validation in Linear (conditional)
    if enable_phase2 and has_multipol and snr_profile_linear is not None:
        # Use independent SNR threshold for Linear polarization
        snr_threshold_linear = snap.snr_thresh_linear
        logger.info("Phase 2: ENABLED - Re-evaluating %d peaks in Linear Polarization (threshold=%.1f)", 
                   len(peaks_intensity), snr_threshold_linear)
        
        # Check each Intensity peak in Linear Polarization
        for peak_idx in peaks_intensity:
            snr_in_linear = float(snr_profile_linear[peak_idx])
            snr_in_intensity = float(snr_profile_intensity[peak_idx])
            
            # Peak must be above threshold in Linear polarization
            if snr_in_linear >= snr_threshold_linear:
                peaks_final.append(peak_idx)
                logger.debug(
                    "Peak at t_idx=%d passed: SNR_I=%.2f, SNR_L=%.2f (threshold_L=%.1f)",
                    peak_idx, snr_in_intensity, snr_in_linear, snr_threshold_linear
                )
            else:
                logger.debug(
                    "Peak at t_idx=%d REJECTED: SNR_I=%.2f, SNR_L=%.2f < %.1f (Linear below threshold)",
                    peak_idx, snr_in_intensity, snr_in_linear, snr_threshold_linear
                )
        
        logger.info("Phase 2: %d/%d peaks passed Linear Polarization check (SNR_L >= %.1f)", 
                   len(peaks_final), len(peaks_intensity), snr_threshold_linear)
        
        # Record Phase 2 metrics
        if metrics_tracker is not None:
            # For Phase 2, we don't track BURST/NO_BURST (that's classification)
            # Phase 2 only validates SNR, so all passed are just "passed", no burst classification yet
            metrics_tracker.record_phase_2(
                num_entered=len(peaks_intensity),
                num_passed=len(peaks_final),
                num_burst=0,  # Phase 2 doesn't classify, just validates SNR
                num_no_burst=0
            )
        
        if len(peaks_final) == 0:
            logger.info("Phase 2: No peaks passed Linear check - skipping Phase 3")
            return _empty_band_result(metrics_tracker)
    else:
        # Phase 2 disabled or no multi-pol data available
        if not enable_phase2 and has_multipol:
            logger.info("Phase 2: DISABLED - Skipping Linear Polarization SNR validation")
            logger.info("  → Linear will still be used in Phase 3b for classification")
        elif not has_multipol:
            logger.warning("Phase 2: SKIPPED - No multi-pol data available")
        peaks_final = peaks_intensity
        
        # Record Phase 2 metrics (all passed since Phase 2 is disabled)
        if metrics_tracker is not None:
            metrics_tracker.record_phase_2(
                num_entered=len(peaks_intensity),
                num_passed=len(peaks_final),  # All pass when Phase 2 is disabled
                num_burst=0,
                num_no_burst=0
            )
    
    # Use the validated peaks for classification
    peaks = peaks_final
    
    logger.info("Phase 3: Proceeding to ResNet classification with %d validated peaks", len(peaks))

    # Geometry parameters for synthetic boxes in the original band_img space.
    img_h, img_w = band_img.shape[:2]
    # Box width: small, centered on the temporal peak detected by boxcar matching
    half_w = max(4, slice_len // 64)
    # Box height: FULL DM range (DM_min to DM_max) to ensure accurate DM calculation
    # This allows the DM to be calculated from the entire DM range, not limited to a small box
    half_h = img_h // 2  # Full height: from 0 to img_h-1
    # Scale factors that match the 512x512 network input.
    target_w = 512
    target_h = 512
    scale_x = float(target_w) / float(max(1, img_w))
    scale_y = float(target_h) / float(max(1, img_h))

    # Outputs accumulated for compatibility with the main pipeline.
    top_conf: list[float] = []
    top_boxes: list[tuple[int, int, int, int]] = []
    class_probs_list: list[float] = []
    class_probs_linear_list: list[float] = []  # NEW: Linear classification probs
    snr_waterfall_linear_list: list[float | None] = []  # NEW: SNR from Linear waterfall
    snr_patch_linear_list: list[float | None] = []  # NEW: SNR from dedispersed Linear patch
    snr_waterfall_intensity_list: list[float | None] = []  # NEW: SNR from Intensity waterfall
    snr_patch_intensity_list: list[float | None] = []  # NEW: SNR from dedispersed Intensity patch
    candidate_times_abs: list[float] = []
    cand_counter = 0
    n_bursts = 0
    n_no_bursts = 0
    prob_max = 0.0
    
    # Track Phase 3 metrics (per candidate)
    phase_3a_burst = 0
    phase_3a_no_burst = 0
    phase_3a_passed = 0
    phase_3b_burst = 0
    phase_3b_no_burst = 0
    phase_3b_passed = 0

    # Track the best candidate for the composite view.
    best_patch = None
    best_start = None
    best_dm = None
    best_is_burst = False

    # Patch directory for saving cut-outs.
    if patches_dir is not None:
        patch_path = patches_dir / f"patch_slice{j}_band{band_idx}.png"
    else:
        patch_path = (save_dir / "Patches" / fits_path.stem / f"patch_slice{j}_band{band_idx}.png")

    # =========================================================================
    # PHASE 3: RESNET CLASSIFICATION - FOR VALIDATED PEAKS
    # =========================================================================
    # Classification control flags already defined in configuration summary above
    # enable_intensity_class, enable_linear_class, class_prob_linear_thresh

    logger.info("Phase 3: Proceeding with %d validated peaks", len(peaks))
    if not has_multipol and enable_linear_class:
        logger.warning("  - Linear classification requested but no multi-pol data - will be skipped for this slice")
    
    # Calculate waterfall SNR values (same as in plot_composite.py)
    # This ensures consistency between CSV and plots
    snr_waterfall = None
    snr_waterfall_linear = None  # NEW: SNR from Linear waterfall
    peak_time_waterfall = None
    if waterfall_block is not None and waterfall_block.size > 0:
        try:
            snr_wf, _, _ = compute_snr_profile(waterfall_block)
            if snr_wf.size > 0:
                peak_snr_wf, _, peak_idx_wf = find_snr_peak(snr_wf)
                snr_waterfall = float(peak_snr_wf)
                # Calculate absolute time for waterfall peak (same method as plot)
                if absolute_start_time is not None:
                    slice_start_abs = absolute_start_time
                else:
                    slice_start_abs = j * slice_len * time_reso_ds
                # CRITICAL: Use len(snr_wf) for both slice_end_abs and time_axis_wf
                # to ensure they match exactly and avoid broadcasting errors
                snr_samples = len(snr_wf)
                slice_end_abs = slice_start_abs + snr_samples * time_reso_ds
                time_axis_wf = np.linspace(slice_start_abs, slice_end_abs, snr_samples)
                peak_time_waterfall = float(time_axis_wf[peak_idx_wf])
        except Exception as e:
            logger.debug(f"Could not calculate waterfall SNR: {e}")
            snr_waterfall = None
            peak_time_waterfall = None
    
    # Calculate SNR from Linear waterfall if available
    if waterfall_block_linear is not None and waterfall_block_linear.size > 0:
        try:
            snr_wf_linear, _, _ = compute_snr_profile(waterfall_block_linear)
            if snr_wf_linear.size > 0:
                peak_snr_wf_linear, _, _ = find_snr_peak(snr_wf_linear)
                snr_waterfall_linear = float(peak_snr_wf_linear)
        except Exception as e:
            logger.debug(f"Could not calculate Linear waterfall SNR: {e}")
            snr_waterfall_linear = None
    
    for peak_idx in peaks:
        # Create a box centred on the temporal peak (detected by boxcar matching).
        # The box spans the FULL DM range (DM_min to DM_max) so the DM measured
        # below is a property of the data and not of the box; see the function.
        cx, box, t_sample_real, t_sec_real = locate_candidate_box(
            peak_idx=peak_idx,
            img_w=img_w,
            img_h=img_h,
            half_w=half_w,
            scale_x=scale_x,
            scale_y=scale_y,
            effective_len=slice_samples if slice_samples is not None else slice_len,
            time_reso_ds=time_reso_ds,
        )

        # CRITICAL: Calculate DM from the peak position in the DM-time cube
        # Since we don't know the exact DM from boxcar matching (only temporal position),
        # we find the DM with maximum intensity at this temporal position
        # IMPORTANT: Use band 0 (fullband average) which has the dedispersed signal
        # The DM-time cube has 3 bands: [0]=average, [1]=mid_channel, [2]=difference
        # We need to use dm_time_fullband (band 0) for DM calculation, not band_img which
        # might be band 1 or 2 and may not have proper DM variation
        if dm_time_fullband is not None:
            logger.info(f"[DM_CALC] Using dm_time_fullband (band 0) for DM calculation, shape={dm_time_fullband.shape}")
            dm_img_for_calc = dm_time_fullband
        else:
            logger.warning(f"[DM_CALC] dm_time_fullband not provided, falling back to band_img (may not have DM variation)")
            dm_img_for_calc = band_img
        
        logger.info(f"[DM_CALC] Calculating DM for peak_idx={peak_idx}, cx={cx}, dm_img shape={dm_img_for_calc.shape}")

        # In very high frequencies (e.g. ALMA) dispersion across the band is
        # negligible, so the DM-time cube has no variation to read a DM from;
        # resolve_candidate_dm says so in dm_status instead of inventing one.
        dm_val, dm_status, dm_uncertainty = resolve_candidate_dm(
            dm_img=dm_img_for_calc,
            cx=cx,
            dm_min=snap.dm_min,
            dm_max=snap.dm_max,
            dm_policy=str(getattr(config, "HIGH_FREQ_DM_POLICY", "unresolved")),
        )

        # Confidence derived from the SNR value in Intensity (clipped to a sensible range).
        snr_peak = float(snr_profile_intensity[peak_idx])
        conf = float(min(0.99, max(0.05, snr_peak / 10.0)))

        global_sample = int(slice_start_idx) + int(peak_idx)
        dm_for_dedisp = dm_for_dedispersion(dm_val)
        
        # =====================================================================
        # PHASE 3a: ResNet Classification on INTENSITY (conditional)
        # =====================================================================
        # None means "Phase 3a produced no verdict for this candidate", which is
        # what the CSV and the decision table must both see (audit P1-10); see
        # classify_intensity_patch.
        intensity = classify_intensity_patch(
            enabled=enable_intensity_class,
            cls_model=cls_model,
            data_block=data_block,
            freq_down=freq_down,
            dm_for_dedisp=dm_for_dedisp,
            global_sample=global_sample,
            time_reso_ds=time_reso_ds,
            snr_peak=snr_peak,
            class_prob_threshold=snap.class_prob,
        )
        class_prob_intensity = intensity.class_prob
        is_burst_intensity = intensity.is_burst
        snr_val_intensity = intensity.snr_val
        proc_patch_intensity = intensity.proc_patch

        if enable_intensity_class:
            logger.debug(
                "Phase 3a: Intensity classification - DM=%.2f t_idx=%d class_prob=%.3f is_burst=%s",
                dm_val, peak_idx, class_prob_intensity, is_burst_intensity,
            )
        else:
            logger.debug("Phase 3a: DISABLED - Skipping Intensity classification for peak_idx=%d", peak_idx)

        # =====================================================================
        # PHASE 3b: ResNet Classification on LINEAR POLARIZATION (conditional)
        # =====================================================================
        # Same contract as Phase 3a: None until Phase 3b actually classifies.
        linear = classify_linear_patch(
            enabled=enable_linear_class,
            cls_model=cls_model,
            data_block_linear=data_block_linear,
            freq_down=freq_down,
            dm_for_dedisp=dm_for_dedisp,
            global_sample=global_sample,
            class_prob_threshold=class_prob_linear_thresh,
        )
        class_prob_linear = linear.class_prob
        is_burst_linear = linear.is_burst
        snr_val_linear = linear.snr_val
        proc_patch_linear = linear.proc_patch

        if not enable_linear_class:
            logger.debug("Phase 3b: DISABLED - Skipping Linear classification for peak_idx=%d", peak_idx)
        elif class_prob_linear is not None:
            logger.debug(
                "Phase 3b: Linear classification - DM=%.2f t_idx=%d SNR_L=%.2f class_prob=%.3f is_burst=%s (threshold=%.2f)",
                dm_val, peak_idx, snr_val_linear if snr_val_linear is not None else 0.0,
                class_prob_linear, is_burst_linear, class_prob_linear_thresh
            )

        # =====================================================================
        # DECISION LOGIC: Determine if candidate should be saved
        # =====================================================================
        # New granular control based on which classification phases are enabled:
        # - Both enabled: Logical AND (STRICT) or OR (PERMISSIVE) based on save_only_burst
        # - Only Intensity enabled: Decision based solely on Intensity
        # - Only Linear enabled: Decision based solely on Linear
        
        has_intensity_result = enable_intensity_class and is_burst_intensity is not None
        has_linear_result = (
            enable_linear_class
            and data_block_linear is not None
            and is_burst_linear is not None
        )
        should_save, is_burst, save_reason = decide_candidate(
            has_intensity_result=has_intensity_result,
            has_linear_result=has_linear_result,
            is_burst_intensity=is_burst_intensity,
            class_prob_intensity=class_prob_intensity,
            is_burst_linear=is_burst_linear,
            class_prob_linear=class_prob_linear,
            enable_linear_class=enable_linear_class,
            save_only_burst=snap.save_only_burst,
        )
        if not has_intensity_result and not has_linear_result:
            logger.error("CRITICAL: No classification results available for peak_idx=%d", peak_idx)
        
        # Track Phase 3a metrics per candidate
        if has_intensity_result:
            phase_3a_passed += 1
            if is_burst_intensity:
                phase_3a_burst += 1
            else:
                phase_3a_no_burst += 1
        # else: Phase 3a disabled: candidate didn't go through classification
        # We don't count it as passed/failed in Phase 3a
        
        # Track Phase 3b metrics per candidate
        if has_linear_result:
            phase_3b_passed += 1
            if is_burst_linear:
                phase_3b_burst += 1
            else:
                phase_3b_no_burst += 1
        # else: Phase 3b disabled or no data - don't count
        
        # Calculate detection time from DM-time plot (same as plot_composite.py line 264-269)
        # This is the time shown in the plot label
        if absolute_start_time is not None:
            detection_time_dm_time = absolute_start_time + t_sec_real
        else:
            detection_time_dm_time = j * slice_len * time_reso_ds + t_sec_real
        
        # Force the candidate time to align with the waterfall SNR peak (for backward compatibility)
        absolute_candidate_time = (absolute_start_time or 0.0) + (peak_idx * time_reso_ds)

        # Get SNR from Linear waterfall at this peak (if available) - MUST be before using it
        # Reuse snr_profile_linear calculated earlier (more efficient and reliable)
        snr_linear_at_peak = None
        if snr_profile_linear is not None and snr_profile_intensity is not None:
            # Ensure peak_idx is valid for both profiles
            max_valid_idx = min(len(snr_profile_intensity), len(snr_profile_linear)) - 1
            if 0 <= peak_idx <= max_valid_idx:
                try:
                    snr_val = snr_profile_linear[peak_idx]
                    # Handle NaN and None values
                    if snr_val is not None and not (isinstance(snr_val, float) and np.isnan(snr_val)):
                        snr_linear_at_peak = float(snr_val)
                        logger.info("SNR Linear at peak_idx=%d: %.2f (I=%.2f)", 
                                   peak_idx, snr_linear_at_peak, float(snr_profile_intensity[peak_idx]))
                    else:
                        logger.warning("SNR Linear at peak_idx=%d is NaN/None", peak_idx)
                except (IndexError, ValueError, TypeError) as e:
                    logger.warning("Could not get SNR Linear at peak_idx=%d: %s", peak_idx, e)
                    snr_linear_at_peak = None
            else:
                logger.warning("peak_idx=%d out of valid range [0, %d] for SNR profiles (I size=%d, L size=%d)", 
                             peak_idx, max_valid_idx, len(snr_profile_intensity), len(snr_profile_linear))
        elif snr_profile_linear is None:
            logger.warning("snr_profile_linear is None - cannot get SNR Linear for peak_idx=%d (has_multipol=%s)", 
                        peak_idx, has_multipol)
        elif snr_profile_intensity is None:
            logger.warning("snr_profile_intensity is None - unexpected state")

        # Record outputs.
        snr_list.append(snr_peak)
        top_conf.append(conf)
        top_boxes.append(box)
        # NaN rather than None: these two lists feed the composite plot, and a
        # phase that did not run has no probability to draw.
        class_probs_list.append(float('nan') if class_prob_intensity is None else class_prob_intensity)
        class_probs_linear_list.append(float('nan') if class_prob_linear is None else class_prob_linear)
        candidate_times_abs.append(float(absolute_candidate_time))
        
        # Store SNR in Linear for plotting (NEW)
        snr_waterfall_linear_list.append(snr_linear_at_peak)
        snr_patch_linear_list.append(snr_val_linear)
        
        # Store SNR in Intensity for plotting (NEW - to match Linear)
        # Get SNR from waterfall at candidate peak (similar to Linear)
        # Reuse snr_profile_intensity calculated earlier (more efficient and reliable)
        snr_intensity_at_peak = None
        if snr_profile_intensity is not None and snr_profile_intensity.size > 0:
            # Ensure peak_idx is valid for the SNR profile
            if 0 <= peak_idx < len(snr_profile_intensity):
                try:
                    snr_val = snr_profile_intensity[peak_idx]
                    # Handle NaN and None values
                    if snr_val is not None and not (isinstance(snr_val, float) and np.isnan(snr_val)):
                        snr_intensity_at_peak = float(snr_val)
                        logger.info("SNR Intensity at peak_idx=%d: %.2f (L=%.2f)", 
                                   peak_idx, snr_intensity_at_peak, 
                                   float(snr_profile_linear[peak_idx]) if snr_profile_linear is not None and peak_idx < len(snr_profile_linear) else 0.0)
                    else:
                        logger.warning("SNR Intensity at peak_idx=%d is NaN/None", peak_idx)
                except (IndexError, ValueError, TypeError) as e:
                    logger.warning("Could not get SNR Intensity at peak_idx=%d: %s", peak_idx, e)
            else:
                logger.warning("peak_idx=%d out of valid range [0, %d) for SNR profile Intensity", 
                             peak_idx, len(snr_profile_intensity))
        else:
            logger.warning("snr_profile_intensity is None or empty - cannot get SNR Intensity for peak_idx=%d", peak_idx)
        snr_waterfall_intensity_list.append(snr_intensity_at_peak)
        # ``snr_val_intensity if 'snr_val_intensity' in locals() else None`` --
        # the function asking itself which of its own branches had run. Phase 3a
        # always answers, with the dedispersed-patch SNR or with the waterfall
        # peak it falls back to, so the test was always true and the None arm
        # unreachable.
        snr_patch_intensity_list.append(snr_val_intensity)
        
        # Log for debugging
        if snr_linear_at_peak is not None:
            logger.info("Stored SNR Linear: waterfall=%.2f, patch=%s for peak_idx=%d", 
                        snr_linear_at_peak, snr_val_linear, peak_idx)
        else:
            logger.warning("SNR Linear is None for peak_idx=%d (has_multipol=%s, snr_profile_linear available=%s)", 
                        peak_idx, has_multipol, snr_profile_linear is not None)

        linear_fraction = None
        if waterfall_block_linear is not None and waterfall_block is not None:
            try:
                lo = max(0, peak_idx - 2)
                hi = min(waterfall_block.shape[0], peak_idx + 3)
                i_level = float(np.nanmedian(np.abs(waterfall_block[lo:hi])))
                l_level = float(np.nanmedian(np.abs(waterfall_block_linear[lo:hi])))
                if i_level > 1e-6:
                    linear_fraction = l_level / i_level
            except Exception:
                linear_fraction = None

        # Keep track of the best candidate.
        # Use proc_patch_intensity if available, otherwise use proc_patch_linear
        patch_to_use = proc_patch_intensity if proc_patch_intensity is not None else proc_patch_linear
        
        if best_patch is None or (is_burst and not best_is_burst):
            best_patch = patch_to_use
            best_start = absolute_candidate_time
            best_dm = dm_val
            best_is_burst = is_burst

        # width_ms already computed by _finalize_patch (Phase 3a)
        score = score_candidate(
            snr_pre_dedisp=snr_peak,
            snr_post_dedisp=snr_val_intensity,
            dm_status=dm_status,
            linear_fraction=linear_fraction,
            class_prob_intensity=class_prob_intensity,
            class_prob_linear=class_prob_linear,
            n_snr_samples=len(snr_profile_intensity),
            dm_min=snap.dm_min,
            dm_max=snap.dm_max,
            trial_correction=getattr(config, "TRIAL_CORRECTION", "gaussian_extreme"),
        )

        # Calculate MJD values for the candidate (using DM-time detection time, same as plot)
        mjd_data = calculate_candidate_mjd(
            t_sec=float(detection_time_dm_time),
            compute_bary=True,
            dm=float(dm_val) if np.isfinite(float(dm_val)) else None,
        )

        cand = build_candidate_record(
            fits_name=fits_path.name,
            chunk_idx=chunk_idx,
            slice_idx=j,
            band_idx=band_idx,
            conf=conf,
            dm_val=dm_val,
            dm_status=dm_status,
            dm_uncertainty=dm_uncertainty,
            detection_time_dm_time=detection_time_dm_time,
            peak_time_waterfall=peak_time_waterfall,
            t_sample_real=t_sample_real,
            box=box,
            snr_waterfall=snr_waterfall,
            snr_waterfall_linear=snr_linear_at_peak,
            snr_pre_dedisp=snr_peak,
            linear_fraction=linear_fraction,
            intensity=intensity,
            linear=linear,
            score=score,
            is_burst=is_burst,
            patch_file=patch_path.name,
            mjd_data=mjd_data,
        )
        cand_delta, burst_delta, no_burst_delta = count_candidate(is_burst=is_burst)
        cand_counter += cand_delta
        n_bursts += burst_delta
        n_no_bursts += no_burst_delta
        prob_max = max(prob_max, float(conf))

        # Save candidate based on dual-polarization filtering logic
        if should_save:
            append_candidate(csv_file, cand.to_row())
            try:
                gl = get_global_logger()
                gl.candidate_detected(dm_val, absolute_candidate_time, conf, _fmt_prob(class_prob_intensity), is_burst, snr_peak, snr_val_intensity)
            except Exception:
                pass
            
            logger.info(
                "SAVED: DM=%.2f t=%.3fs I_class=%s L_class=%s → %s",
                dm_val, absolute_candidate_time,
                _fmt_prob(class_prob_intensity), _fmt_prob(class_prob_linear), save_reason
            )
        else:
            logger.debug(
                "FILTERED: DM=%.2f t=%.3fs I_class=%s L_class=%s → %s",
                dm_val, absolute_candidate_time,
                _fmt_prob(class_prob_intensity), _fmt_prob(class_prob_linear),
                save_reason
            )

    # Record Phase 3 metrics after processing all candidates
    if metrics_tracker is not None:
        # Phase 3a metrics
        if enable_intensity_class:
            metrics_tracker.record_phase_3a(
                num_entered=len(peaks),  # All peaks that reached Phase 3
                num_passed=phase_3a_passed,
                num_burst=phase_3a_burst,
                num_no_burst=phase_3a_no_burst
            )
        
        # Phase 3b metrics
        if enable_linear_class and data_block_linear is not None:
            metrics_tracker.record_phase_3b(
                num_entered=len(peaks),  # All peaks that reached Phase 3
                num_passed=phase_3b_passed,
                num_burst=phase_3b_burst,
                num_no_burst=phase_3b_no_burst
            )
        
        # Record final classification counts
        metrics_tracker.record_final_classification(
            num_total=cand_counter,
            num_burst=n_bursts,
            num_no_burst=n_no_bursts
        )
    
    # Generate an RGB image using the same colour pipeline as the standard flow.
    img_tensor = preprocess_img(band_img)
    # Applied at the plot site instead; see the note in detection_engine.
    
    # =========================================================================
    # PREPARE MULTI-POLARIZATION WATERFALLS FOR PLOTTING
    # =========================================================================
    waterfall_intensity = waterfall_block  # Already have this (from main block)
    waterfall_linear = None
    waterfall_circular = None
    
    if waterfall_block_raw is not None and has_full_polarization_data(waterfall_block_raw, pol_type):
        try:
            logger.debug("Extracting multi-pol waterfalls from RAW block: shape=%s, pol_type=%s",
                        waterfall_block_raw.shape, pol_type)
            
            # Extract Linear Polarization waterfall
            wf_linear_raw = extract_polarization_from_raw(
                waterfall_block_raw, pol_type, "linear", default_index=0
            )
            waterfall_linear = wf_linear_raw[:, 0, :]  # Remove pol dimension
            logger.debug("Linear waterfall extracted: shape=%s", waterfall_linear.shape)
            
            # Extract Circular Polarization waterfall
            wf_circular_raw = extract_polarization_from_raw(
                waterfall_block_raw, pol_type, "circular", default_index=0
            )
            waterfall_circular = wf_circular_raw[:, 0, :]  # Remove pol dimension
            logger.debug("Circular waterfall extracted: shape=%s", waterfall_circular.shape)
            
            logger.info("Multi-pol waterfalls prepared: I=%s, L=%s, V=%s",
                        waterfall_intensity.shape, waterfall_linear.shape, waterfall_circular.shape)
        except Exception as e:
            logger.warning("Failed to extract multi-pol waterfalls: %s", e)
    else:
        if waterfall_block_raw is None:
            logger.debug("waterfall_block_raw is None, cannot extract multi-pol data")
        else:
            logger.debug("has_full_polarization_data() returned False: shape=%s, pol_type=%s",
                        waterfall_block_raw.shape, pol_type)

    return {
        "top_conf": top_conf,
        "top_boxes": top_boxes,
        "class_probs_list": class_probs_list,
        "class_probs_linear_list": class_probs_linear_list,  # NEW: Pass Linear probs
        "snr_waterfall_linear_list": snr_waterfall_linear_list,  # NEW: SNR from Linear waterfall
        "snr_patch_linear_list": snr_patch_linear_list,  # NEW: SNR from dedispersed Linear patch
        "snr_waterfall_intensity_list": snr_waterfall_intensity_list,  # NEW: SNR from Intensity waterfall
        "snr_patch_intensity_list": snr_patch_intensity_list,  # NEW: SNR from dedispersed Intensity patch
        "first_patch": best_patch,
        "first_start": best_start,
        "first_dm": best_dm,
        "img_tensor": img_tensor,  # postprocess_img() is applied at the plot site
        "cand_counter": cand_counter,
        "n_bursts": n_bursts,
        "n_no_bursts": n_no_bursts,
        "prob_max": prob_max,
        "patch_path": patch_path,
        "best_is_burst": best_is_burst,
        "total_candidates": cand_counter,
        "candidate_times_abs": candidate_times_abs,
        # Multi-polarization waterfalls for plotting
        "waterfall_intensity": waterfall_intensity,
        "waterfall_linear": waterfall_linear,
        "waterfall_circular": waterfall_circular,
        # Phase metrics tracker
        "phase_metrics": metrics_tracker if metrics_tracker else None,
    }


def process_slice_with_multiple_bands_high_freq(
    j: int,
    dm_time: np.ndarray,
    block: np.ndarray,
    slice_len: int,
    cls_model,
    fits_path: Path,
    save_dir: Path,
    freq_down: np.ndarray,
    csv_file: Path,
    time_reso_ds: float,
    band_configs: list[tuple[int, str, str]],
    snr_list: list,
    absolute_start_time: float | None,
    composite_dir: Path | None,
    detections_dir: Path | None,
    patches_dir: Path | None,
    chunk_idx: int | None,
    slice_start_idx: int,
    slice_end_idx: int,
    block_raw: np.ndarray | None = None,
    pol_type: str = "IQUV",
    metrics_tracker: PhaseMetricsTracker | None = None,
    snapshot: PipelineConfigSnapshot | None = None,
) -> tuple[int, int, int, float]:
    """Process a slice using SNR peaks with multi-polarization detection.
    
    Detection flow:
    1. Phase 1: SNR peak detection in Intensity (Stokes I) - MANDATORY
    2. Phase 2: Re-evaluate same peaks in Linear Polarization - conditional
    3. Phase 3: ResNet classification - ONLY if detected in BOTH polarizations
    """
    try:
        global_logger = get_global_logger()
    except Exception:
        global_logger = None

    start_idx = int(slice_start_idx)
    end_idx = int(slice_end_idx)
    slice_cube = dm_time[:, :, start_idx:end_idx]
    waterfall_block = block[start_idx:end_idx]
    if slice_cube.size == 0 or waterfall_block.size == 0:
        return 0, 0, 0, 0.0

    fits_stem = fits_path.stem
    if composite_dir is not None:
        comp_path = composite_dir / f"{fits_stem}_slice{j:03d}.png"
    else:
        comp_path = save_dir / "Composite" / f"{fits_stem}_slice{j:03d}.png"

    cand_counter = 0
    n_bursts = 0
    n_no_bursts = 0
    prob_max = 0.0
    slice_has_candidates = False

    # Extract the RAW slice for multi-polarization extraction
    waterfall_block_raw = None
    if block_raw is not None:
        # block_raw is 3D: (time, npol, chan) for multi-pol data
        if block_raw.ndim == 3 and block_raw.shape[1] >= 4:
            waterfall_block_raw = block_raw[start_idx:end_idx]
            logger.debug("Extracted RAW slice for multi-pol: shape=%s", waterfall_block_raw.shape)
        else:
            logger.debug("block_raw available but not multi-pol (shape=%s)", block_raw.shape)
    
    # Process all configured bands.
    # IMPORTANT: Use band 0 (fullband average) for DM calculation, as it contains the dedispersed signal
    dm_time_fullband = slice_cube[0]  # Band 0 is the fullband average with dedispersion
    
    # Diagnostic: Check if the DM-time cube has variation in DM
    if dm_time_fullband.size > 0:
        # Check a few columns to see if there's DM variation
        sample_cols = [dm_time_fullband.shape[1] // 4, dm_time_fullband.shape[1] // 2, 3 * dm_time_fullband.shape[1] // 4]
        sample_cols = [c for c in sample_cols if c < dm_time_fullband.shape[1]]
        for col_idx in sample_cols[:1]:  # Just check first sample column
            col_data = dm_time_fullband[:, col_idx]
            col_min = float(np.min(col_data))
            col_max = float(np.max(col_data))
            col_std = float(np.std(col_data))
            col_mean = float(np.mean(col_data))
            logger.info(
                f"[DM_CUBE_DIAG] slice_cube[0] column {col_idx}: "
                f"min={col_min:.3f}, max={col_max:.3f}, mean={col_mean:.3f}, std={col_std:.3f}, "
                f"has_variation={col_std > 1e-6}"
            )
    
    for band_idx, band_suffix, band_name in band_configs:
        band_img = slice_cube[band_idx]
        result = snr_detect_and_classify_candidates_in_band(
            cls_model,
            band_img,
            waterfall_block,
            end_idx - start_idx,
            j,
            fits_path,
            save_dir,
            block,
            freq_down,
            csv_file,
            time_reso_ds,
            snr_list,
            absolute_start_time,
            patches_dir,
            chunk_idx,
            band_idx,
            start_idx,
            waterfall_block_raw=waterfall_block_raw,
            data_block_raw=block_raw,  # Pass full chunk RAW data
            pol_type=pol_type,
            slice_samples=end_idx - start_idx,  # Actual slice samples
            dm_time_fullband=dm_time_fullband,  # Pass band 0 for DM calculation
            metrics_tracker=metrics_tracker,  # Pass metrics tracker
            snapshot=snapshot,  # REF-10: built once by the driver, per file
        )
        
        # Merge phase metrics from result if available
        if metrics_tracker is not None and result.get("phase_metrics") is not None:
            # The metrics_tracker is already updated in the function, but we merge anyway for safety
            pass
        cand_counter += result["cand_counter"]
        n_bursts += result["n_bursts"]
        n_no_bursts += result["n_no_bursts"]
        prob_max = max(prob_max, result["prob_max"])
        if len(result["top_conf"]) > 0:
            slice_has_candidates = True

        # Decide whether plots should be generated.
        should_generate_plots = (slice_has_candidates or config.FORCE_PLOTS)
        if config.SAVE_ONLY_BURST:
            should_generate_plots = (n_bursts > 0) or config.FORCE_PLOTS

        if should_generate_plots:
            # ``first_dm`` is NaN whenever the band could not resolve a DM,
            # which at these frequencies is every candidate. Guarding only
            # against None let that NaN through to all three dedispersions
            # below; the same rule the per-candidate patch uses is applied here
            # so the figure and the patch show the same candidate.
            dm_to_use = dm_for_dedispersion(result["first_dm"])

            # Dedisperse ALL three polarizations if available
            dedisp_block_intensity = dedisperse_block(block, freq_down, dm_to_use, start_idx, end_idx - start_idx)
            dedisp_block_linear = None
            dedisp_block_circular = None
            
            if result.get("waterfall_linear") is not None:
                # Dedisperse linear polarization waterfall
                dedisp_block_linear = dedisperse_block(
                    result["waterfall_linear"], freq_down, dm_to_use, 0, result["waterfall_linear"].shape[0]
                )
                logger.debug("Dedispersed Linear: shape=%s", dedisp_block_linear.shape)
            else:
                logger.debug("No Linear waterfall in result, skipping dedispersion")
            
            if result.get("waterfall_circular") is not None:
                # Dedisperse circular polarization waterfall
                dedisp_block_circular = dedisperse_block(
                    result["waterfall_circular"], freq_down, dm_to_use, 0, result["waterfall_circular"].shape[0]
                )
                logger.debug("Dedispersed Circular: shape=%s", dedisp_block_circular.shape)
            else:
                logger.debug("No Circular waterfall in result, skipping dedispersion")
            
            try:
                global_logger.generating_plots()
            except Exception:
                pass
            
            logger.info("Generating plots with multi-pol data: Linear=%s, Circular=%s",
                       dedisp_block_linear is not None, dedisp_block_circular is not None)
            
            from ..visualization.visualization_unified import save_all_plots
            save_all_plots(
                waterfall_block,
                dedisp_block_intensity,
                postprocess_img(result["img_tensor"]) if result.get("img_tensor") is not None else None,
                result["first_patch"],
                result["first_start"],
                result["first_dm"],
                result["top_conf"],
                result["top_boxes"],
                result["class_probs_list"],
                comp_path,
                j,
                block.shape[0] // slice_len + (1 if block.shape[0] % slice_len != 0 else 0),
                band_name,
                band_suffix,
                fits_stem,
                end_idx - start_idx,
                normalize=True,
                thresh_snr=config.SNR_THRESH,
                band_idx=band_idx,
                absolute_start_time=absolute_start_time,
                chunk_idx=chunk_idx,
                candidate_times_abs=result.get("candidate_times_abs"),  # NEW: Pass candidate times for polarization plots
                # Multi-polarization dedispersed waterfalls for HF pipeline
                dedisp_block_linear=dedisp_block_linear,
                dedisp_block_circular=dedisp_block_circular,
                class_probs_linear_list=result.get("class_probs_linear_list"),  # NEW: Linear classification
                snr_waterfall_linear_list=result.get("snr_waterfall_linear_list"),  # NEW: SNR from Linear waterfall
                snr_patch_linear_list=result.get("snr_patch_linear_list"),  # NEW: SNR from dedispersed Linear patch
                snr_waterfall_intensity_list=result.get("snr_waterfall_intensity_list"),  # NEW: SNR from Intensity waterfall
                snr_patch_intensity_list=result.get("snr_patch_intensity_list"),  # NEW: SNR from dedispersed Intensity patch
            )

    # Effective metrics after applying the SAVE_ONLY_BURST flag.
    if config.SAVE_ONLY_BURST:
        return n_bursts, n_bursts, 0, prob_max
    return cand_counter, n_bursts, n_no_bursts, prob_max


def _process_file_chunked_high_freq(
    cls_model,
    fits_path: Path,
    save_dir: Path,
    chunk_samples: int,
) -> dict:
    """High-frequency pipeline with multi-polarization detection.

    This pipeline implements a 3-phase detection strategy:
    1. SNR peak detection in Intensity (Stokes I) - MANDATORY
    2. Re-evaluation with Linear Polarization - ONLY if detected in Intensity
    3. ResNet classification - ONLY if detected in BOTH polarizations

    There is no ``streaming_func`` parameter any more. There was one, the caller
    resolved a reader with ``get_streaming_function`` and passed it, and this
    function handed it to ``log_streaming_parameters`` and then iterated
    ``stream_fits_multi_pol`` regardless -- so the log named a reader that was
    not running, and the only honest way to read the log was to know it was
    wrong. The reader is now named where it is used, once (audit REF-02).
    """
    from .data_flow_manager import (
        build_dm_time_cube,
        create_chunk_directories,
        downsample_chunk,
        get_chunk_processing_parameters,
        plan_slices,
        release_dm_cube_buffer,
        trim_valid_window,
    )
    from ..log_utils import log_streaming_parameters
    from ..input.fits_handler import stream_fits_multi_pol

    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be greater than zero")

    # ===== VALIDATION METRICS COLLECTOR =====
    from ..output.validation_metrics import ValidationMetricsCollector
    collector = ValidationMetricsCollector(fits_path.name)
    collector.record_data_characteristics()

    # ===== PHASE METRICS TRACKER =====
    phase_metrics_tracker = PhaseMetricsTracker()

    # Streaming parameters reused from the main pipeline. ``max_chunk_limit`` is
    # None on purpose: this driver has never applied config.MAX_CHUNK_SAMPLES,
    # and giving it one here would be an untested change to the path that
    # processes the observations this project exists for.
    plan = prepare_chunked_run(
        fits_path,
        save_dir,
        chunk_samples,
        collector,
        log_prefix="[HF Pipeline] ",
        max_chunk_limit=None,
    )
    chunk_samples = plan.chunk_samples
    total_samples = plan.total_samples
    effective_chunk_samples = plan.effective_chunk_samples
    chunk_count = plan.chunk_count
    csv_file = plan.csv_file

    # REF-10. The low-frequency driver reads its search configuration from
    # contracts taken once; this one read the mutable global seventeen times.
    # Built here, before the first read, and valid for the whole call: nothing
    # in src/core writes to config, which tests/test_layering.py pins.
    obs_meta = ObservationMetadata.from_config(config)
    pipe_snap = PipelineConfigSnapshot.from_config(config)

    t_start = time.time()
    cand_counter_total = 0
    n_bursts_total = 0
    n_no_bursts_total = 0
    prob_max_total = 0.0
    snr_list_total: list[float] = []
    state = ChunkLoopState()

    try:
        check_file_length(total_samples)

        overlap_raw = compute_overlap_raw()

        logger.info("High-frequency pipeline: Multi-polarization detection enabled")
        logger.info("Detection flow: Intensity → Linear → ResNet (if both pass)")
        
        # Log streaming parameters with the adjusted chunk size (after adaptive
        # budgeting). The reader named here is the one the loop below iterates,
        # which is the whole point of naming it here rather than taking it as an
        # argument that was never used.
        log_streaming_parameters(
            effective_chunk_samples, overlap_raw, total_samples,
            effective_chunk_samples, stream_fits_multi_pol, "fits/multi-pol",
        )

        # Use multi-polarization streaming for HF pipeline
        start_arrival_clock(state)

        # Checkpoint/resume: an interrupted HF run used to restart from chunk 0
        # and, because the CSV is opened in append mode, duplicate every
        # candidate it had already written. Shared with the LF driver (REF-01).
        from ..core.checkpoint import clear_checkpoint, should_skip_chunk

        resume = begin_resumable_run(fits_path, save_dir, csv_file)
        run_fingerprint, resume_after = resume.fingerprint, resume.resume_after

        for chunk_seq, (block, block_raw, metadata, pol_type) in enumerate(
            stream_fits_multi_pol(
                str(fits_path), effective_chunk_samples, overlap_samples=overlap_raw
            ), 1
        ):
            if should_skip_chunk(chunk_seq, resume_after):
                logger.debug('Skipping chunk %d (already completed)', chunk_seq)
                continue
            begin_chunk(state, block, metadata)

            # Gated on chunks actually processed, which is what this driver has
            # always used; the LF driver gates on the stream sequence number.
            log_chunk_arrival_latency(
                state, metadata['chunk_idx'], is_first=state.actual_chunk_count <= 1
            )

            logger.info(
                "Processing chunk %03d • samples %s→%s",
                metadata['chunk_idx'],
                f"{metadata['start_sample']:,}",
                f"{metadata['end_sample']:,}",
            )
            
            chunk_succeeded = False
            # Bound BEFORE the try so the finally below can always release them.
            # This is the whole reason the obvious fix for this defect does not
            # work: the low-frequency driver can free its block after the
            # handler because `block` is bound by its `for` statement, outside
            # the try. Every name here is bound inside it, so on a failure path
            # some or none of them exist, and an unguarded `del` would raise
            # NameError while the original error was being handled -- losing the
            # error and replacing it with a worse one.
            block_ds = dm_time = block_raw_ds = None
            try:
                # DIAGNOSTIC: Log block_raw shape before downsampling
                if block_raw is not None:
                    logger.info("Received block_raw from stream: shape=%s, ndim=%d, dtype=%s", 
                               block_raw.shape, block_raw.ndim, block_raw.dtype)
                else:
                    logger.warning("Received block_raw=None from stream")

                # Downsample and extract the valid window.
                block_ds, dt_ds = downsample_chunk(block)
                chunk_params = get_chunk_processing_parameters(metadata)
                freq_down = chunk_params['freq_down']
                slice_len = chunk_params['slice_len']
                time_slice = chunk_params['time_slice']
                overlap_left_ds = chunk_params['overlap_left_ds']
                overlap_right_ds = chunk_params['overlap_right_ds']

                # SPEC-HF-002: Skip cube if DM smearing < 1 sample (unresolved band)
                height = chunk_params['height']
                freq_low = float(freq_down.min())
                freq_high = float(freq_down.max())
                dm_range = pipe_snap.dm_max - pipe_snap.dm_min
                dm_delay_s = K_DM_MS * dm_range * (freq_low ** -2 - freq_high ** -2)
                dm_smear_samples = dm_delay_s / obs_meta.effective_time_reso
                if dm_smear_samples < 1.0:
                    logger.info(
                        "SPEC-HF-002: DM unresolved (%.4f samples at %.0f-%.0f MHz) — "
                        "skipping DM-time cube build.",
                        dm_smear_samples, freq_low, freq_high,
                    )
                    # valid_start_ds/valid_end_ds must index the UNTRIMMED block,
                    # exactly like trim_valid_window() reports them in the branch
                    # below: block_raw_ds is trimmed with these same bounds and is
                    # still untrimmed at that point. Returning 0-based bounds here
                    # shifted every polarisation waveform by overlap_left_ds
                    # relative to intensity, so Phase 2 compared SNR_I and SNR_L at
                    # different instants.
                    n_valid = max(0, block_ds.shape[0] - overlap_left_ds - overlap_right_ds)
                    dm_time = np.zeros((3, height, n_valid), dtype=np.float32)
                    valid_start_ds = overlap_left_ds
                    valid_end_ds = block_ds.shape[0] - overlap_right_ds
                    block_ds = block_ds[valid_start_ds:valid_end_ds]
                else:
                    dm_time_full = build_dm_time_cube(
                        block_ds, height=height, dm_min=pipe_snap.dm_min,
                        dm_max=pipe_snap.dm_max, collector=collector,
                    )
                    block_ds, dm_time, valid_start_ds, valid_end_ds = trim_valid_window(block_ds, dm_time_full, overlap_left_ds, overlap_right_ds)
                    release_dm_cube_buffer(dm_time_full)
                    del dm_time_full
                    gc.collect()

                # Record chunk processing for validation metrics
                collector.record_chunk_processing(
                    chunk_idx=metadata['chunk_idx'],
                    overlap_left=overlap_left_ds,
                    overlap_right=overlap_right_ds,
                    valid_start=valid_start_ds,
                    valid_end=valid_end_ds,
                    chunk_samples=block_ds.shape[0]
                )

                # Also downsample the RAW multi-pol block for polarization extraction
                block_raw_ds = None
                if block_raw is not None:
                    # Check if block_raw is 3D multi-pol data
                    if block_raw.ndim == 3 and block_raw.shape[1] >= 4:
                        try:
                            # Downsample multi-pol block preserving all polarizations
                            logger.debug("Downsampling multi-pol block: input shape=%s", block_raw.shape)
                            
                            # Manual downsampling that preserves polarization dimension
                            n_time = (block_raw.shape[0] // obs_meta.down_time_rate) * obs_meta.down_time_rate
                            n_pol = block_raw.shape[1]
                            n_freq = (block_raw.shape[2] // obs_meta.down_freq_rate) * obs_meta.down_freq_rate
                            
                            # Trim to divisible sizes
                            block_trimmed = block_raw[:n_time, :, :n_freq]
                            
                            # Reshape to separate downsample axes
                            block_reshaped = block_trimmed.reshape(
                                n_time // obs_meta.down_time_rate,
                                obs_meta.down_time_rate,
                                n_pol,
                                n_freq // obs_meta.down_freq_rate,
                                obs_meta.down_freq_rate,
                            )
                            # Shape: (n_time_ds, DOWN_TIME_RATE, n_pol, n_freq_ds, DOWN_FREQ_RATE)
                            
                            # Sum over time axis (PRESTO style)
                            block_ds_time = block_reshaped.sum(axis=1)
                            # Shape: (n_time_ds, n_pol, n_freq_ds, DOWN_FREQ_RATE)
                            
                            # Average over frequency axis
                            block_raw_ds = block_ds_time.mean(axis=3)
                            # Shape: (n_time_ds, n_pol, n_freq_ds) ✅
                            
                            block_raw_ds = block_raw_ds.astype(np.float32)
                            
                            logger.debug("Downsampled multi-pol block: output shape=%s", block_raw_ds.shape)
                            
                            # Apply same trimming as the main block
                            if valid_start_ds >= 0 and valid_end_ds <= block_raw_ds.shape[0]:
                                block_raw_ds = block_raw_ds[valid_start_ds:valid_end_ds]
                            logger.info("Multi-pol RAW block downsampled: shape=%s (time, npol=%d, chan), pol_type=%s", 
                                       block_raw_ds.shape, block_raw_ds.shape[1], pol_type)
                        except Exception as e:
                            logger.error("Failed to downsample multi-pol block: %s", e, exc_info=True)
                            block_raw_ds = None
                    else:
                        logger.warning("block_raw is not 3D multi-pol data: ndim=%d, shape=%s", 
                                     block_raw.ndim, block_raw.shape)
                else:
                    logger.warning("No multi-pol data available (block_raw is None)")

                # Plan slices.
                slices_to_process = plan_slices(block_ds, slice_len, metadata['chunk_idx'])
                composite_dir, detections_dir, patches_dir, summary_dir = create_chunk_directories(save_dir, fits_path, metadata['chunk_idx'])

                # Match the classic pipeline's chunk start time computation.
                chunk_start_time_sec = metadata["start_sample"] * obs_meta.time_reso

                for j, start_idx, end_idx in slices_to_process:
                    cands, bursts, nobursts, pmax = process_slice_with_multiple_bands_high_freq(
                        j=j,
                        dm_time=dm_time,
                        block=block_ds,
                        slice_len=slice_len,
                        cls_model=cls_model,
                        fits_path=fits_path,
                        save_dir=save_dir,
                        freq_down=freq_down,
                        csv_file=csv_file,
                        time_reso_ds=dt_ds,
                        snapshot=pipe_snap,
                        band_configs=config.get_band_configs(),
                        snr_list=snr_list_total,
                        absolute_start_time=chunk_start_time_sec + start_idx * dt_ds,
                        composite_dir=composite_dir,
                        detections_dir=detections_dir,
                        patches_dir=patches_dir,
                        chunk_idx=metadata['chunk_idx'],
                        slice_start_idx=start_idx,
                        slice_end_idx=end_idx,
                        block_raw=block_raw_ds,
                        pol_type=pol_type,
                        metrics_tracker=phase_metrics_tracker,
                    )
                    cand_counter_total += cands
                    n_bursts_total += bursts
                    n_no_bursts_total += nobursts
                    prob_max_total = max(prob_max_total, pmax)
                
                # This driver's stream does not announce a chunk count, so the
                # ETA is estimated from the file length. Both this and the
                # timing stay inside the ``try``, where this driver has always
                # had them: a chunk that raised records no time and no ETA.
                # Estimate total chunks (may not be exact, but gives an idea)
                estimated_total_chunks = max(
                    state.actual_chunk_count, int(total_samples / effective_chunk_samples) + 1
                )
                finish_chunk(
                    state,
                    metadata['chunk_idx'],
                    remaining_chunks=max(0, estimated_total_chunks - state.actual_chunk_count),
                    report_eta=state.actual_chunk_count > 1,
                )

                chunk_succeeded = True
            except MemoryError as mem_error:
                record_oom(collector, metadata['chunk_idx'], mem_error)
                raise
            except Exception as chunk_error:
                record_chunk_failure(state, metadata['chunk_idx'], chunk_error)
            finally:
                # CRITICAL: free the chunk-level arrays whatever happened.
                # These were the last statements of the try body, so a chunk
                # that raised anywhere in the slice loop skipped them and held a
                # cube and two blocks until the next iteration rebound the
                # names. The chunk most likely to raise is the one that ran out
                # of memory, which is precisely when that costs most -- and on
                # the MemoryError path this now runs before the exception
                # propagates out of the loop.
                del block_ds, dm_time, block_raw_ds
                optimize_memory(aggressive=(state.actual_chunk_count % 5 == 0))

            # Only a chunk that completed may advance the checkpoint, and its
            # rows have to be on disk before it does. Shared helper (REF-01).
            if chunk_succeeded:
                checkpoint_completed_chunk(
                    save_dir, fits_path.stem, chunk_seq, chunk_count,
                    run_fingerprint,
                )

        from ..log_utils import log_processing_summary
        log_processing_summary(state.actual_chunk_count, chunk_count, cand_counter_total, n_bursts_total)

        export_validation_metrics(collector, save_dir, fits_path)

        runtime = time.time() - t_start
    except Exception as e:
        logger.exception(f"Error in high-frequency pipeline: {e}")
        # Return a result instead of re-raising. The counters above are
        # function locals, so a raise destroyed them with the frame and the
        # caller -- which cannot see them -- rebuilt the result from an empty
        # DetectionStats. A run that wrote rows to disk and then failed reported
        # n_candidates: 0, which reads as "this file had no detections".
        #
        # The flush comes first, for the same reason it does on the success
        # path: the numbers reported here have to match what is readable. The
        # caller also flushes in a finally, but that is its stop-gap, not this
        # driver's correctness.
        CandidateWriter.flush_all()
        # The checkpoint is deliberately NOT cleared here: a file that failed
        # part-way has to stay resumable, and clear_checkpoint is what says
        # "this file is finished".
        stats = DetectionStats(
            n_candidates=cand_counter_total,
            n_bursts=n_bursts_total,
            n_no_bursts=n_no_bursts_total,
            max_prob=prob_max_total,
            snr_values=list(snr_list_total),
        )
        status, message = status_for_error(e)
        logger.error(message, fits_path.name, e)
        return error_result(
            status, e, t_start, stats,
            chunks_processed=state.actual_chunk_count,
            failed_chunks=state.failed_chunk_count,
        )

    if pipe_snap.save_only_burst:
        effective_cand_counter_total = n_bursts_total
        effective_n_bursts_total = n_bursts_total
        effective_n_no_bursts_total = 0
    else:
        effective_cand_counter_total = cand_counter_total
        effective_n_bursts_total = n_bursts_total
        effective_n_no_bursts_total = n_no_bursts_total

    # The file finished: the checkpoint has served its purpose.
    try:
        clear_checkpoint(save_dir, fits_path.stem)
    except Exception as e:
        logger.debug('Could not clear HF checkpoint: %s', e)

    # Flush buffered candidate rows before reporting counts. The caller also
    # flushes in a finally, but this keeps the HF path correct on its own: the
    # numbers returned below must match what is on disk.
    CandidateWriter.flush_all()

    successful_chunks = state.actual_chunk_count - state.failed_chunk_count
    status = finalize_file_status("SUCCESS_CHUNKED_HIGH_FREQ", state.failed_chunk_count, successful_chunks)
    if state.failed_chunk_count > 0:
        logger.warning(
            "HF file %s completed with %d/%d chunks failed -> status=%s",
            fits_path.name, state.failed_chunk_count, state.actual_chunk_count, status,
        )

    return {
        "n_candidates": effective_cand_counter_total,
        "n_bursts": effective_n_bursts_total,
        "n_no_bursts": effective_n_no_bursts_total,
        "runtime_s": runtime,
        "max_prob": prob_max_total,
        "mean_snr": float(np.mean(snr_list_total)) if snr_list_total else 0.0,
        "status": status,
        "failed_chunks": state.failed_chunk_count,
        "phase_metrics": phase_metrics_tracker,
    }


