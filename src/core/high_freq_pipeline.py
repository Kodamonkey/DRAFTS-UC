from __future__ import annotations

# Standard library imports
from dataclasses import dataclass
from pathlib import Path
import logging
import time

# Third-party imports
import numpy as np

# Local imports
from ..config import config
from ..analysis.snr_utils import compute_snr_profile, find_snr_peak
from ..logging.logging_config import Colors, get_global_logger
from ..output.candidate_manager import Candidate, append_candidate
from ..output.phase_metrics import PhaseMetricsTracker
from ..preprocessing.dedispersion import dedisperse_block, dedisperse_patch
from ..preprocessing.dm_candidate_extractor import extract_candidate_dm
from ..visualization.visualization_unified import preprocess_img, postprocess_img
from .mjd_utils import calculate_candidate_mjd
from ..analysis.snr_utils import compute_snr_profile, find_snr_peak

logger = logging.getLogger(__name__)


@dataclass
class PeakCandidateBox:
    x1: int
    y1: int
    x2: int
    y2: int


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
    
    # Map the row index to actual DM value
    # The DM-time cube uses np.linspace(dm_min, dm_max, height) for uniform distribution
    dm_min = float(config.DM_min)
    dm_max = float(config.DM_max)
    dm_range = dm_max - dm_min
    
    # Formula: DM = DM_min + (row_index / (total_rows - 1)) * DM_range
    # This matches how np.linspace distributes values
    if h > 1:
        dm_val = dm_min + (row_idx / (h - 1)) * dm_range
    else:
        dm_val = dm_min  # Edge case: only one row
    
    # Log for verification (INFO level so it appears in logs)
    logger.info(
        f"[DM_CALC] time_idx={time_idx}, row_idx={row_idx}/{h-1}, "
        f"intensity={max_intensity:.3f}, DM={dm_val:.2f} "
        f"(range: {dm_min:.2f}-{dm_max:.2f}, DM_mid={(dm_min+dm_max)/2:.2f})"
    )
    
    return float(dm_val)


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
    logger.info("  └─ SNR threshold (Intensity): %.1f sigma", float(config.SNR_THRESH))
    
    # Read config values directly (not using getattr with defaults to catch errors)
    enable_phase2 = config.ENABLE_LINEAR_VALIDATION if hasattr(config, 'ENABLE_LINEAR_VALIDATION') else False
    # Log the actual value read from config for debugging
    logger.info("Config check: ENABLE_LINEAR_VALIDATION = %s (type: %s, hasattr: %s)", 
               enable_phase2, type(enable_phase2).__name__, hasattr(config, 'ENABLE_LINEAR_VALIDATION'))
    snr_threshold_linear = getattr(config, 'SNR_THRESH_LINEAR', config.SNR_THRESH)
    logger.info("Phase 2 (SNR Validation - Linear): %s", "ENABLED" if enable_phase2 else "DISABLED")
    if enable_phase2:
        logger.info("  └─ SNR threshold (Linear): %.1f sigma", snr_threshold_linear)
    
    # Read config values directly (not using getattr with defaults to catch errors)
    enable_intensity_class = config.ENABLE_INTENSITY_CLASSIFICATION if hasattr(config, 'ENABLE_INTENSITY_CLASSIFICATION') else True
    enable_linear_class = config.ENABLE_LINEAR_CLASSIFICATION if hasattr(config, 'ENABLE_LINEAR_CLASSIFICATION') else True
    class_prob_linear_thresh = getattr(config, 'CLASS_PROB_LINEAR', config.CLASS_PROB)
    
    # Log all config values for debugging
    logger.info("Config check: ENABLE_INTENSITY_CLASSIFICATION = %s (hasattr: %s)", 
               enable_intensity_class, hasattr(config, 'ENABLE_INTENSITY_CLASSIFICATION'))
    logger.info("Config check: ENABLE_LINEAR_CLASSIFICATION = %s (hasattr: %s)", 
               enable_linear_class, hasattr(config, 'ENABLE_LINEAR_CLASSIFICATION'))
    logger.info("Config check: CLASS_PROB_LINEAR = %.2f", class_prob_linear_thresh)
    logger.info("Config check: SNR_THRESH_LINEAR = %.1f", snr_threshold_linear)
    
    logger.info("Phase 3a (Classification - Intensity): %s", "ENABLED" if enable_intensity_class else "DISABLED")
    if enable_intensity_class:
        logger.info("  └─ Classification threshold (Intensity): %.2f", float(config.CLASS_PROB))
    
    logger.info("Phase 3b (Classification - Linear): %s", "ENABLED" if enable_linear_class else "DISABLED")
    if enable_linear_class:
        logger.info("  └─ Classification threshold (Linear): %.2f", class_prob_linear_thresh)
    
    logger.info("Decision Mode: %s", "STRICT (require ALL enabled phases)" if config.SAVE_ONLY_BURST else "PERMISSIVE (require ANY enabled phase)")
    logger.info("=" * 80)

    # =========================================================================
    # PHASE 1: SNR PEAK DETECTION IN INTENSITY (Stokes I) - MANDATORY
    # =========================================================================
    logger.info("Phase 1: SNR peak detection in Intensity (waterfall_block shape: %s)", waterfall_block.shape)
    
    # Compute the SNR profile on the waterfall (time × frequency) - INTENSITY
    # Use off_regions=None for consistency (same as Linear calculation)
    snr_profile_intensity, _, _ = compute_snr_profile(waterfall_block, off_regions=None)
    logger.debug("Calculated snr_profile_intensity: size=%d, shape=%s", 
                len(snr_profile_intensity) if snr_profile_intensity is not None else 0,
                snr_profile_intensity.shape if snr_profile_intensity is not None else None)
    peaks_intensity = _find_snr_peaks(snr_profile_intensity, float(config.SNR_THRESH))
    
    # Ensure the first candidate corresponds to the main SNR peak used downstream.
    peak_snr_global, _, peak_idx_global = find_snr_peak(snr_profile_intensity)
    if peak_snr_global >= float(config.SNR_THRESH):
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
        return {
            "top_conf": [],
            "top_boxes": [],
            "class_probs_list": [],
            "first_patch": None,
            "first_start": None,
            "first_dm": None,
            "img_rgb": None,
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
        # Use off_regions=None for consistency with Intensity calculation
        try:
            # Verify waterfall blocks have compatible shapes
            if waterfall_block.shape[0] != waterfall_block_linear.shape[0]:
                logger.warning("Waterfall block size mismatch: Intensity=%d, Linear=%d. This may cause SNR profile size mismatch.",
                             waterfall_block.shape[0], waterfall_block_linear.shape[0])
            
            snr_profile_linear, _, _ = compute_snr_profile(waterfall_block_linear, off_regions=None)
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
        snr_threshold_linear = getattr(config, 'SNR_THRESH_LINEAR', config.SNR_THRESH)
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
            return {
                "top_conf": [],
                "top_boxes": [],
                "class_probs_list": [],
                "first_patch": None,
                "first_start": None,
                "first_dm": None,
                "img_rgb": None,
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
    
    # Import classify_patch here so it's available for both Phase 3a and 3b
    from ..detection.model_interface import classify_patch
    
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
            snr_wf, _, _ = compute_snr_profile(waterfall_block, off_regions)
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
            snr_wf_linear, _, _ = compute_snr_profile(waterfall_block_linear, off_regions)
            if snr_wf_linear.size > 0:
                peak_snr_wf_linear, _, _ = find_snr_peak(snr_wf_linear)
                snr_waterfall_linear = float(peak_snr_wf_linear)
        except Exception as e:
            logger.debug(f"Could not calculate Linear waterfall SNR: {e}")
            snr_waterfall_linear = None
    
    for peak_idx in peaks:
        # Create a box centred on the temporal peak (detected by boxcar matching)
        # The box spans the FULL DM range (DM_min to DM_max) for accurate DM calculation
        cx = int(max(0, min(img_w - 1, peak_idx)))
        
        # Box spans full DM range: from row 0 to row img_h-1
        # This ensures DM is calculated from the entire DM range, not limited to a small box
        x1_raw = max(0, cx - half_w)
        x2_raw = min(img_w - 1, cx + half_w)
        y1_raw = 0  # Start at DM_min (row 0)
        y2_raw = img_h - 1  # End at DM_max (row img_h-1)
        
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
        dm_val_approx = _dm_from_image_at_time(dm_img_for_calc, cx)
        
        # Check if DM calculation is valid (has variation in DM dimension)
        # In very high frequencies (e.g., ALMA), dispersion is negligible, so DM-time cube
        # may not have variation. In this case, we use the middle DM as approximation.
        intensity_column = dm_img_for_calc[:, cx] if cx < dm_img_for_calc.shape[1] else dm_img_for_calc[:, 0]
        col_std = float(np.std(intensity_column))
        
        if col_std < 1e-6:
            # No variation in DM dimension - this is expected at very high frequencies
            # Use middle DM as approximation since dispersion is negligible
            dm_min = float(config.DM_min)
            dm_max = float(config.DM_max)
            dm_val = (dm_min + dm_max) / 2.0
            logger.info(
                f"[DM_CALC] No DM variation detected (std={col_std:.6f}) - typical at very high frequencies. "
                f"Using middle DM as approximation: {dm_val:.2f} (range: {dm_min:.2f}-{dm_max:.2f})"
            )
        else:
            # Normal case: use the DM with maximum intensity at the peak time
            dm_val = dm_val_approx
            logger.info(f"[DM_CALC] Calculated DM: {dm_val:.2f} (from _dm_from_image_at_time, std={col_std:.6f})")
        
        # Calculate time from the center of the box (temporal position)
        center_x = (x1_raw + x2_raw) / 2.0
        effective_len_det = slice_samples if slice_samples is not None else slice_len
        sample_off = (center_x / max(img_w - 1, 1)) * effective_len_det
        t_sample_real = int(sample_off)
        t_sec_real = float(sample_off) * config.TIME_RESO * config.DOWN_TIME_RATE
        
        # Transform the box to 512x512 coordinates to match img_rgb.
        x1 = int(round(x1_raw * scale_x))
        x2 = int(round(x2_raw * scale_x))
        y1 = int(round(y1_raw * scale_y))
        y2 = int(round(y2_raw * scale_y))
        box = (x1, y1, x2, y2)

        # Confidence derived from the SNR value in Intensity (clipped to a sensible range).
        snr_peak = float(snr_profile_intensity[peak_idx])
        conf = float(min(0.99, max(0.05, snr_peak / 10.0)))

        global_sample = int(slice_start_idx) + int(peak_idx)
        
        # =====================================================================
        # PHASE 3a: ResNet Classification on INTENSITY (conditional)
        # =====================================================================
        class_prob_intensity = 0.0
        is_burst_intensity = False
        snr_val_intensity = snr_peak
        
        if enable_intensity_class:
            patch_intensity, start_sample = dedisperse_patch(data_block, freq_down, dm_val, global_sample)

            peak_idx_patch = None
            if patch_intensity is not None and patch_intensity.size > 0:
                snr_profile_pre, _, best_w_vec = compute_snr_profile(patch_intensity)
                if snr_profile_pre.size > 0:
                    peak_idx_patch = int(np.argmax(snr_profile_pre))
                    snr_val_intensity = float(np.max(snr_profile_pre))

            # Classify patch - EXACTLY same logic as classic pipeline (line 145 in detection_engine.py)
            class_prob_intensity, proc_patch_intensity = classify_patch(cls_model, patch_intensity)
            is_burst_intensity = class_prob_intensity >= float(config.CLASS_PROB)
            
            # Log classification result with patch info for debugging
            patch_info = f"shape={patch_intensity.shape if patch_intensity is not None else 'None'}, size={patch_intensity.size if patch_intensity is not None else 0}"
            logger.debug(
                "Phase 3a: Intensity classification - DM=%.2f t_idx=%d class_prob=%.3f is_burst=%s patch_info=%s",
                dm_val, peak_idx, class_prob_intensity, is_burst_intensity, patch_info
            )
            
            # Warn if probability is 0.0 but we have a valid patch (this shouldn't happen for valid candidates)
            if class_prob_intensity == 0.0 and patch_intensity is not None and patch_intensity.size > 0:
                logger.warning(
                    "WARNING: class_prob_intensity is 0.0 for valid patch at DM=%.2f peak_idx=%d. "
                    "This may indicate a classification error.",
                    dm_val, peak_idx
                )
        else:
            # Phase 3a disabled: Set default values (will rely on Phase 3b)
            logger.debug("Phase 3a: DISABLED - Skipping Intensity classification for peak_idx=%d", peak_idx)
            class_prob_intensity = 1.0  # Neutral value (will depend on Linear)
            is_burst_intensity = True  # Pass through to Linear decision
            patch_intensity = None
            proc_patch_intensity = None  # Initialize to None when Phase 3a is disabled
            start_sample = None
            peak_idx_patch = None
        
        # =====================================================================
        # PHASE 3b: ResNet Classification on LINEAR POLARIZATION (conditional)
        # =====================================================================
        class_prob_linear = 0.0
        is_burst_linear = False
        snr_val_linear = None  # NEW: SNR from dedispersed Linear patch
        
        if enable_linear_class and data_block_linear is not None:
            # Dedisperse Linear polarization patch at same DM and time
            patch_linear, _ = dedisperse_patch(data_block_linear, freq_down, dm_val, global_sample)
            
            if patch_linear is not None and patch_linear.size > 0:
                # Calculate SNR from dedispersed Linear patch (similar to Intensity)
                snr_profile_linear_patch, _, _ = compute_snr_profile(patch_linear)
                if snr_profile_linear_patch.size > 0:
                    snr_val_linear = float(np.max(snr_profile_linear_patch))
                
                class_prob_linear, proc_patch_linear = classify_patch(cls_model, patch_linear)
                # Use independent threshold for Linear classification
                is_burst_linear = class_prob_linear >= class_prob_linear_thresh
                
                logger.debug(
                    "Phase 3b: Linear classification - DM=%.2f t_idx=%d SNR_L=%.2f class_prob=%.3f is_burst=%s (threshold=%.2f)",
                    dm_val, peak_idx, snr_val_linear if snr_val_linear is not None else 0.0, 
                    class_prob_linear, is_burst_linear, class_prob_linear_thresh
                )
        elif not enable_linear_class:
            # Phase 3b disabled: Set default values (will rely on Phase 3a)
            logger.debug("Phase 3b: DISABLED - Skipping Linear classification for peak_idx=%d", peak_idx)
            class_prob_linear = 1.0  # Neutral value (will depend on Intensity)
            is_burst_linear = True  # Pass through to Intensity decision
            snr_val_linear = None
            proc_patch_linear = None  # Initialize to None when Phase 3b is disabled
        else:
            # Phase 3b disabled due to no multi-pol data
            proc_patch_linear = None
        
        # =====================================================================
        # DECISION LOGIC: Determine if candidate should be saved
        # =====================================================================
        # New granular control based on which classification phases are enabled:
        # - Both enabled: Logical AND (STRICT) or OR (PERMISSIVE) based on save_only_burst
        # - Only Intensity enabled: Decision based solely on Intensity
        # - Only Linear enabled: Decision based solely on Linear
        
        should_save = False
        save_reason = ""
        
        # Determine which polarizations are actually available for decision
        has_intensity_result = enable_intensity_class
        has_linear_result = enable_linear_class and (data_block_linear is not None)
        
        if has_intensity_result and has_linear_result:
            # BOTH classifications available - apply dual logic
            if config.SAVE_ONLY_BURST:
                # STRICT mode: Both must be BURST
                should_save = is_burst_intensity and is_burst_linear
                if should_save:
                    save_reason = f"BURST in I+L (p_I={class_prob_intensity:.2f}, p_L={class_prob_linear:.2f})"
                else:
                    save_reason = f"Filtered: I={'BURST' if is_burst_intensity else 'NO'}({class_prob_intensity:.2f}), L={'BURST' if is_burst_linear else 'NO'}({class_prob_linear:.2f})"
            else:
                # PERMISSIVE mode: Either can be BURST (logical OR)
                should_save = is_burst_intensity or is_burst_linear
                if should_save:
                    save_reason = f"BURST: I={is_burst_intensity}({class_prob_intensity:.2f}), L={is_burst_linear}({class_prob_linear:.2f})"
                else:
                    save_reason = f"NO-BURST in both: I={class_prob_intensity:.2f}, L={class_prob_linear:.2f}"
        
        elif has_intensity_result and not has_linear_result:
            # Only INTENSITY classification available
            should_save = not config.SAVE_ONLY_BURST or is_burst_intensity
            save_reason = f"BURST in I({class_prob_intensity:.2f})" if is_burst_intensity else f"NO-BURST in I({class_prob_intensity:.2f})"
            if enable_linear_class:
                save_reason += " [Linear N/A - no multi-pol data]"
        
        elif not has_intensity_result and has_linear_result:
            # Only LINEAR classification available
            should_save = not config.SAVE_ONLY_BURST or is_burst_linear
            save_reason = f"BURST in L({class_prob_linear:.2f})" if is_burst_linear else f"NO-BURST in L({class_prob_linear:.2f})"
            save_reason += " [Intensity disabled]"
        
        else:
            # Neither classification available (should not happen due to validation)
            logger.error("CRITICAL: No classification results available for peak_idx=%d", peak_idx)
            should_save = False
            save_reason = "ERROR: No classification available"
        
        # Track Phase 3a metrics per candidate
        if enable_intensity_class:
            phase_3a_passed += 1
            if is_burst_intensity:
                phase_3a_burst += 1
            else:
                phase_3a_no_burst += 1
        # else: Phase 3a disabled: candidate didn't go through classification
        # We don't count it as passed/failed in Phase 3a
        
        # Track Phase 3b metrics per candidate
        if enable_linear_class and data_block_linear is not None:
            phase_3b_passed += 1
            if is_burst_linear:
                phase_3b_burst += 1
            else:
                phase_3b_no_burst += 1
        # else: Phase 3b disabled or no data - don't count
        
        # Use Intensity classification as primary for is_burst flag
        is_burst = is_burst_intensity
        
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
        class_probs_list.append(class_prob_intensity)
        class_probs_linear_list.append(class_prob_linear)  # NEW: Store Linear prob
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
        snr_patch_intensity_list.append(snr_val_intensity if 'snr_val_intensity' in locals() else None)
        
        # Log for debugging
        if snr_linear_at_peak is not None:
            logger.info("Stored SNR Linear: waterfall=%.2f, patch=%s for peak_idx=%d", 
                        snr_linear_at_peak, snr_val_linear, peak_idx)
        else:
            logger.warning("SNR Linear is None for peak_idx=%d (has_multipol=%s, snr_profile_linear available=%s)", 
                        peak_idx, has_multipol, snr_profile_linear is not None)

        # Keep track of the best candidate.
        # Use proc_patch_intensity if available, otherwise use proc_patch_linear
        # Initialize proc_patch_linear if not already defined (for cases where Phase 3b is skipped)
        if 'proc_patch_linear' not in locals():
            proc_patch_linear = None
        patch_to_use = proc_patch_intensity if proc_patch_intensity is not None else proc_patch_linear
        
        if best_patch is None or (is_burst and not best_is_burst):
            best_patch = patch_to_use
            best_start = absolute_candidate_time
            best_dm = dm_val
            best_is_burst = is_burst

        # Build the CSV row and estimate width_ms using the optimal width at the peak.
        width_ms = None
        try:
            if peak_idx_patch is not None and 'best_w_vec' in locals() and best_w_vec.size > 0:
                width_ms = float(best_w_vec[int(peak_idx_patch)] * time_reso_ds * 1000.0)
        except Exception:
            width_ms = None

        # Calculate MJD values for the candidate (using DM-time detection time, same as plot)
        mjd_data = calculate_candidate_mjd(
            t_sec=float(detection_time_dm_time),
            compute_bary=True,
            dm=float(dm_val),
        )

        cand = Candidate(
            fits_path.name,
            chunk_idx if chunk_idx is not None else 0,
            j,
            band_idx,
            float(conf),
            float(dm_val),  # DM calculated with extract_candidate_dm (same as plot)
            float(detection_time_dm_time),  # Time from DM-time plot (same as plot label)
            peak_time_waterfall,  # Time from waterfall SNR peak (different method)
            int(t_sample_real),  # Sample index
            tuple(map(int, box)),
            snr_waterfall,  # SNR from waterfall raw (peak_snr_wf) - Intensity
            float(snr_val_intensity),  # SNR from dedispersed patch - Intensity
            snr_waterfall_linear=snr_linear_at_peak,  # NEW: SNR from Linear waterfall at peak
            snr_patch_dedispersed_linear=snr_val_linear,  # NEW: SNR from dedispersed Linear patch
            width_ms=width_ms,
            class_prob_intensity=float(class_prob_intensity),  # Classification probability in Intensity (I)
            is_burst_intensity=bool(is_burst_intensity),  # BURST classification in Intensity (I)
            class_prob_linear=float(class_prob_linear),  # Classification probability in Linear (L) - HF only
            is_burst_linear=bool(is_burst_linear) if data_block_linear is not None else None,  # BURST classification in Linear (L) - HF only
            is_burst=bool(is_burst),  # Final classification (I+L when SAVE_ONLY_BURST=True)
            patch_file=patch_path.name,
            mjd_utc=mjd_data.get('mjd_utc'),
            mjd_bary_utc=mjd_data.get('mjd_bary_utc'),
            mjd_bary_tdb=mjd_data.get('mjd_bary_tdb'),
            mjd_bary_utc_inf=mjd_data.get('mjd_bary_utc_inf'),
            mjd_bary_tdb_inf=mjd_data.get('mjd_bary_tdb_inf'),
        )
        cand_counter += 1
        if is_burst:
            n_bursts += 1
        else:
            n_no_bursts += 1
        prob_max = max(prob_max, float(conf))

        # Save candidate based on dual-polarization filtering logic
        if should_save:
            append_candidate(csv_file, cand.to_row())
            try:
                gl = get_global_logger()
                gl.candidate_detected(dm_val, absolute_candidate_time, conf, class_prob_intensity, is_burst, snr_peak, snr_val_intensity)
            except Exception:
                pass
            
            logger.info(
                "SAVED: DM=%.2f t=%.3fs I_class=%.2f L_class=%.2f → %s",
                dm_val, absolute_candidate_time, class_prob_intensity, class_prob_linear, save_reason
            )
        else:
            logger.debug(
                "FILTERED: DM=%.2f t=%.3fs I_class=%.2f(%.0f%%) L_class=%.2f(%.0f%%) → %s",
                dm_val, absolute_candidate_time, 
                class_prob_intensity, class_prob_intensity*100,
                class_prob_linear, class_prob_linear*100,
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
    img_rgb = postprocess_img(img_tensor)
    
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
        "img_rgb": img_rgb,
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
            dm_to_use = result["first_dm"] if result["first_dm"] is not None else 0.0
            
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
                result["img_rgb"],
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
                off_regions=None,
                thresh_snr=config.SNR_THRESH,
                band_idx=band_idx,
                patch_path=result["patch_path"],
                absolute_start_time=absolute_start_time,
                chunk_idx=chunk_idx,
                force_plots=config.FORCE_PLOTS,
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
    streaming_func,
) -> dict:
    """High-frequency pipeline with multi-polarization detection.
    
    This pipeline implements a 3-phase detection strategy:
    1. SNR peak detection in Intensity (Stokes I) - MANDATORY
    2. Re-evaluation with Linear Polarization - ONLY if detected in Intensity
    3. ResNet classification - ONLY if detected in BOTH polarizations
    """
    from .data_flow_manager import (
        build_dm_time_cube,
        create_chunk_directories,
        downsample_chunk,
        get_chunk_processing_parameters,
        plan_slices,
        trim_valid_window,
    )
    from .pipeline_parameters import calculate_frequency_downsampled
    from ..output.candidate_manager import ensure_csv_header
    from ..logging import log_block_processing, log_streaming_parameters
    from ..input.fits_handler import stream_fits_multi_pol
    from ..input.polarization_utils import extract_polarization_from_raw, has_full_polarization_data
    import time

    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be greater than zero")

    # ===== VALIDATION METRICS COLLECTOR =====
    from ..output.validation_metrics import ValidationMetricsCollector
    from ..core.data_flow_manager import set_validation_collector
    collector = ValidationMetricsCollector(fits_path.name)
    collector.record_data_characteristics()
    set_validation_collector(collector)  # Set global collector for memory validations
    
    # ===== PHASE METRICS TRACKER =====
    phase_metrics_tracker = PhaseMetricsTracker()

    # Streaming parameters reused from the main pipeline.
    total_samples = config.FILE_LENG
    
    # ===== ADAPTIVE MEMORY BUDGETING: Calculate memory-safe chunk size =====
    # This ensures we never exceed available RAM, even with large DM ranges
    from ..preprocessing.slice_len_calculator import calculate_memory_safe_chunk_size
    
    try:
        safe_chunk_samples, budget_diagnostics = calculate_memory_safe_chunk_size()
        
        # Record budget diagnostics
        collector.record_memory_budget(budget_diagnostics)
        collector.record_dm_cube(budget_diagnostics)
        collector.record_chunk_calculation(budget_diagnostics)
        
        # Calculate physical lower bound
        min_required_raw = budget_diagnostics.get('required_min_size', 0) * max(1, config.DOWN_TIME_RATE)
        
        # Logic to determine final chunk_samples:
        if chunk_samples < min_required_raw:
            logger.warning(
                f"[HF Pipeline] Requested chunk size ({chunk_samples:,}) is too small for physical constraints "
                f"(overlap + slice_len requires {min_required_raw:,} raw samples). "
                f"Upgrading to memory-safe calculated size: {safe_chunk_samples:,}."
            )
            chunk_samples = safe_chunk_samples
        elif chunk_samples > safe_chunk_samples:
            logger.info(
                f"[HF Pipeline] Adaptive budgeting: Reducing chunk size from {chunk_samples:,} to {safe_chunk_samples:,} samples "
                f"to fit in available memory ({budget_diagnostics['usable_bytes_gb']:.2f} GB usable). "
                f"Scenario: {budget_diagnostics['scenario']}."
            )
            if budget_diagnostics['will_use_dm_chunking']:
                logger.info(
                    f"Expected DM-time cube size: {budget_diagnostics['expected_cube_gb']:.2f} GB. "
                    f"DM chunking will activate automatically."
                )
            chunk_samples = safe_chunk_samples
        else:
            logger.debug(
                f"[HF Pipeline] Requested chunk size ({chunk_samples:,}) is within safe limits "
                f"(min={min_required_raw:,}, safe={safe_chunk_samples:,}). Proceeding with requested size."
            )
    except Exception as e:
        logger.warning(
            f"[HF Pipeline] Failed to calculate memory-safe chunk size: {e}. "
            f"Using requested chunk_samples={chunk_samples:,} (may cause OOM with large DM ranges)."
        )

    if total_samples <= chunk_samples:
        logger.info(
            "Small file detected (%s samples); running in a single optimised chunk",
            f"{total_samples:,}",
        )
        effective_chunk_samples = total_samples
        chunk_count = 1
        logger.info(
            "Using single chunk optimisation • chunk_samples=%s (entire file)",
            f"{effective_chunk_samples:,}",
        )
    else:
        effective_chunk_samples = chunk_samples
        chunk_count = (total_samples + chunk_samples - 1) // chunk_samples
        logger.info("Standard chunking • estimated chunks=%d", chunk_count)

    total_duration_sec = total_samples * config.TIME_RESO
    chunk_duration_sec = effective_chunk_samples * config.TIME_RESO

    logger.info(
        "File summary • chunks=%d • samples=%s • duration=%.2fs (%.1f min) • chunk_size=%s (%.2fs)",
        chunk_count,
        f"{total_samples:,}",
        total_duration_sec,
        total_duration_sec / 60,
        f"{effective_chunk_samples:,}",
        chunk_duration_sec,
    )
    logger.info("Starting streaming processing...")
    
    # Create Summary directory structure: Summary/(file_name)/
    summary_dir = save_dir / "Summary" / fits_path.stem
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_file = summary_dir / f"{fits_path.stem}.candidates.csv"
    ensure_csv_header(csv_file)

    t_start = time.time()
    cand_counter_total = 0
    n_bursts_total = 0
    n_no_bursts_total = 0
    prob_max_total = 0.0
    snr_list_total: list[float] = []
    actual_chunk_count = 0

    try:
        if total_samples <= 0:
            raise ValueError(f"Invalid file length: {total_samples} samples")
        if total_samples > 1_000_000_000:
            logger.warning(
                "Large file detected (%s samples); processing may take longer",
                f"{total_samples:,}",
            )
        
        try:
            freq_ds = calculate_frequency_downsampled()
        except ValueError as exc:
            logger.warning(
                "Failed to compute frequency downsampling (%s); using original axis.",
                exc,
            )
            if config.FREQ is None or len(config.FREQ) == 0:
                raise
            freq_ds = config.FREQ
        nu_min = float(freq_ds.min())
        nu_max = float(freq_ds.max())
        dt_max_sec = 4.1488e3 * config.DM_max * (nu_min**-2 - nu_max**-2)

        if config.TIME_RESO <= 0:
            logger.warning(
                "Invalid TIME_RESO (%s); using default overlap window",
                config.TIME_RESO,
            )
            overlap_raw = 1024
        else:
            overlap_raw = max(0, int(np.ceil(dt_max_sec / config.TIME_RESO)))
        
        logger.info("High-frequency pipeline: Multi-polarization detection enabled")
        logger.info("Detection flow: Intensity → Linear → ResNet (if both pass)")
        
        # Log streaming parameters with the adjusted chunk size (after adaptive budgeting)
        log_streaming_parameters(effective_chunk_samples, overlap_raw, total_samples, effective_chunk_samples, streaming_func, "fits/fil")

        # Use multi-polarization streaming for HF pipeline
        stream_start_time = time.time()
        chunk_processing_times = []
        last_chunk_arrival_time = stream_start_time
        
        for block, block_raw, metadata, pol_type in stream_fits_multi_pol(str(fits_path), effective_chunk_samples, overlap_samples=overlap_raw):
            chunk_start_time = time.time()
            actual_chunk_count += 1
            log_block_processing(actual_chunk_count, block.shape, str(block.dtype), metadata)
            
            # Log time since last chunk arrived from file
            chunk_arrival_time = time.time()
            if actual_chunk_count > 1:
                time_since_last = chunk_arrival_time - last_chunk_arrival_time
                if time_since_last > 10:
                    logger.warning(
                        f"Chunk {metadata['chunk_idx']} took {time_since_last:.1f}s to arrive from file. "
                        f"This may indicate slow I/O or large buffer concatenation. "
                        f"Consider reducing chunk size if this is frequent."
                    )
                elif time_since_last > 5:
                    logger.info(
                        f"Chunk {metadata['chunk_idx']} arrived after {time_since_last:.1f}s. "
                        f"File I/O is proceeding normally."
                    )
            last_chunk_arrival_time = chunk_arrival_time
            
            logger.info(
                "Processing chunk %03d • samples %s→%s",
                metadata['chunk_idx'],
                f"{metadata['start_sample']:,}",
                f"{metadata['end_sample']:,}",
            )
            
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

                # Memory validation is now done inside build_dm_time_cube (PRESTO-style)
                height = chunk_params['height']
                dm_time_full = build_dm_time_cube(block_ds, height=height, dm_min=config.DM_min, dm_max=config.DM_max)
                block_ds, dm_time, valid_start_ds, valid_end_ds = trim_valid_window(block_ds, dm_time_full, overlap_left_ds, overlap_right_ds)
                
                # Record chunk processing for validation metrics
                collector.record_chunk_processing(
                    chunk_idx=metadata['chunk_idx'],
                    overlap_left=overlap_left_ds,
                    overlap_right=overlap_right_ds,
                    valid_start=valid_start_ds,
                    valid_end=valid_end_ds,
                    chunk_samples=block_ds.shape[0]
                )
                
                # CRITICAL: Free the full cube immediately after trimming
                del dm_time_full
                import gc
                gc.collect()

                # Also downsample the RAW multi-pol block for polarization extraction
                block_raw_ds = None
                if block_raw is not None:
                    # Check if block_raw is 3D multi-pol data
                    if block_raw.ndim == 3 and block_raw.shape[1] >= 4:
                        try:
                            # Downsample multi-pol block preserving all polarizations
                            logger.debug("Downsampling multi-pol block: input shape=%s", block_raw.shape)
                            
                            # Manual downsampling that preserves polarization dimension
                            n_time = (block_raw.shape[0] // config.DOWN_TIME_RATE) * config.DOWN_TIME_RATE
                            n_pol = block_raw.shape[1]
                            n_freq = (block_raw.shape[2] // config.DOWN_FREQ_RATE) * config.DOWN_FREQ_RATE
                            
                            # Trim to divisible sizes
                            block_trimmed = block_raw[:n_time, :, :n_freq]
                            
                            # Reshape to separate downsample axes
                            block_reshaped = block_trimmed.reshape(
                                n_time // config.DOWN_TIME_RATE,
                                config.DOWN_TIME_RATE,
                                n_pol,
                                n_freq // config.DOWN_FREQ_RATE,
                                config.DOWN_FREQ_RATE,
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
                chunk_start_time_sec = metadata["start_sample"] * config.TIME_RESO

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
                
                # Track chunk processing time
                chunk_processing_time = time.time() - chunk_start_time
                chunk_processing_times.append(chunk_processing_time)
                
                if chunk_processing_time > 30:
                    logger.warning(
                        f"Chunk {metadata['chunk_idx']} processing took {chunk_processing_time:.1f}s. "
                        f"This is unusually long. Check system resources."
                    )
                
                # Estimate remaining time
                if actual_chunk_count > 1:
                    avg_time = sum(chunk_processing_times) / len(chunk_processing_times)
                    # Estimate total chunks (may not be exact, but gives an idea)
                    estimated_total_chunks = max(actual_chunk_count, int(total_samples / effective_chunk_samples) + 1)
                    remaining_chunks = max(0, estimated_total_chunks - actual_chunk_count)
                    eta_seconds = remaining_chunks * avg_time
                    if eta_seconds > 60:
                        logger.info(
                            f"Chunk {metadata['chunk_idx']} processed in {chunk_processing_time:.1f}s. "
                            f"Average: {avg_time:.1f}s/chunk. ETA: {eta_seconds/60:.1f} min"
                        )
                    else:
                        logger.debug(
                            f"Chunk {metadata['chunk_idx']} processed in {chunk_processing_time:.1f}s. "
                            f"Average: {avg_time:.1f}s/chunk. ETA: {eta_seconds:.1f}s"
                        )
                
                # CRITICAL: Free chunk-level arrays after processing all slices
                del block_ds, dm_time, block_raw_ds
                from ..core.pipeline import _optimize_memory
                _optimize_memory(aggressive=(actual_chunk_count % 5 == 0))
            except MemoryError as mem_error:
                collector.record_oom_error()
                logger.exception(f"Out of memory processing chunk {metadata['chunk_idx']:03d}: {mem_error}")
                raise
            except Exception as chunk_error:
                logger.exception(f"Error processing chunk {metadata['chunk_idx']:03d}: {chunk_error}")

        from ..logging import log_processing_summary
        log_processing_summary(actual_chunk_count, chunk_count, cand_counter_total, n_bursts_total)

        # Export validation metrics
        try:
            validation_dir = save_dir / "Validation" / fits_path.stem
            collector.export_to_json(validation_dir)
            logger.info(f"Validation metrics exported to: {validation_dir}")
        except Exception as e:
            logger.warning(f"Failed to export validation metrics: {e}")

        runtime = time.time() - t_start
    except Exception as e:
        logger.exception(f"Error in high-frequency pipeline: {e}")
        raise
    
    if config.SAVE_ONLY_BURST:
        effective_cand_counter_total = n_bursts_total
        effective_n_bursts_total = n_bursts_total
        effective_n_no_bursts_total = 0
    else:
        effective_cand_counter_total = cand_counter_total
        effective_n_bursts_total = n_bursts_total
        effective_n_no_bursts_total = n_no_bursts_total

    return {
        "n_candidates": effective_cand_counter_total,
        "n_bursts": effective_n_bursts_total,
        "n_no_bursts": effective_n_no_bursts_total,
        "runtime_s": runtime,
        "max_prob": prob_max_total,
        "mean_snr": float(np.mean(snr_list_total)) if snr_list_total else 0.0,
        "status": "SUCCESS_CHUNKED_HIGH_FREQ",
        "phase_metrics": phase_metrics_tracker,
    }


