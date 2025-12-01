from __future__ import annotations

# Standard library imports
from dataclasses import dataclass
from pathlib import Path
import logging

# Third-party imports
import numpy as np

# Local imports
from ..config import config
from ..analysis.snr_utils import compute_snr_profile, find_snr_peak
from ..logging.logging_config import Colors, get_global_logger
from ..output.candidate_manager import Candidate, append_candidate
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
    """Map a time index to the DM row with the highest intensity."""
    h, w = dm_time_band_img.shape[:2]
    t = int(max(0, min(w - 1, time_idx)))
    row_idx = int(np.argmax(dm_time_band_img[:, t]))
    # Map the row index linearly between DM_min and DM_max.
    dm_min = float(config.DM_min)
    dm_max = float(config.DM_max)
    dm_val = dm_min + (row_idx / max(h - 1, 1)) * (dm_max - dm_min)
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
    # PHASE 1: SNR PEAK DETECTION IN INTENSITY (Stokes I) - MANDATORY
    # =========================================================================
    logger.info("Phase 1: SNR peak detection in Intensity (waterfall_block shape: %s)", waterfall_block.shape)
    
    # Compute the SNR profile on the waterfall (time × frequency) - INTENSITY
    snr_profile_intensity, _, _ = compute_snr_profile(waterfall_block)
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
        }
    
    # =========================================================================
    # PHASE 2: RE-EVALUATE IN LINEAR POLARIZATION - CONDITIONAL
    # =========================================================================
    from ..input.polarization_utils import extract_polarization_from_raw, has_full_polarization_data
    
    peaks_final = []  # Peaks that pass validation
    waterfall_block_linear = None  # Will be needed for Phase 3b
    enable_phase2 = getattr(config, 'ENABLE_LINEAR_VALIDATION', True)
    
    # Extract Linear Polarization if multi-pol data available (needed for Phase 2 and 3b)
    has_multipol = waterfall_block_raw is not None and has_full_polarization_data(waterfall_block_raw, pol_type)
    
    # Extract both slice and full chunk in Linear polarization
    waterfall_block_linear = None  # Slice for Phase 2 (SNR validation)
    data_block_linear = None       # Full chunk for Phase 3b (dedispersion)
    
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
    
    # Phase 2: SNR validation in Linear (conditional)
    if enable_phase2 and has_multipol:
        logger.info("Phase 2: ENABLED - Re-evaluating %d peaks in Linear Polarization", len(peaks_intensity))
        
        # Compute SNR profile in Linear Polarization
        snr_profile_linear, _, _ = compute_snr_profile(waterfall_block_linear)
        
        # Check each Intensity peak in Linear Polarization
        for peak_idx in peaks_intensity:
            snr_in_linear = float(snr_profile_linear[peak_idx])
            snr_in_intensity = float(snr_profile_intensity[peak_idx])
            
            # Peak must be above threshold in BOTH polarizations
            if snr_in_linear >= float(config.SNR_THRESH):
                peaks_final.append(peak_idx)
                logger.debug(
                    "Peak at t_idx=%d passed: SNR_I=%.2f, SNR_L=%.2f",
                    peak_idx, snr_in_intensity, snr_in_linear
                )
            else:
                logger.debug(
                    "Peak at t_idx=%d REJECTED: SNR_I=%.2f, SNR_L=%.2f (Linear below threshold)",
                    peak_idx, snr_in_intensity, snr_in_linear
                )
        
        logger.info("Phase 2: %d/%d peaks passed Linear Polarization check", 
                   len(peaks_final), len(peaks_intensity))
        
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
            }
    else:
        # Phase 2 disabled or no multi-pol data available
        if not enable_phase2 and has_multipol:
            logger.info("Phase 2: DISABLED - Skipping Linear Polarization SNR validation")
            logger.info("  → Linear will still be used in Phase 3b for classification")
        elif not has_multipol:
            logger.warning("Phase 2: SKIPPED - No multi-pol data available")
        peaks_final = peaks_intensity
    
    # Use the validated peaks for classification
    peaks = peaks_final
    
    logger.info("Phase 3: Proceeding to ResNet classification with %d validated peaks", len(peaks))

    # Geometry parameters for synthetic boxes in the original band_img space.
    img_h, img_w = band_img.shape[:2]
    half_w = max(4, slice_len // 64)
    half_h = max(3, img_h // 32)
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
    candidate_times_abs: list[float] = []
    cand_counter = 0
    n_bursts = 0
    n_no_bursts = 0
    prob_max = 0.0

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
    logger.info("Phase 3: ResNet classification on Intensity and Linear Polarization patches")
    
    # Calculate waterfall SNR values (same as in plot_composite.py)
    # This ensures consistency between CSV and plots
    snr_waterfall = None
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
                real_samples = slice_samples if slice_samples is not None else slice_len
                slice_end_abs = slice_start_abs + real_samples * time_reso_ds
                time_axis_wf = np.linspace(slice_start_abs, slice_end_abs, len(snr_wf))
                peak_time_waterfall = float(time_axis_wf[peak_idx_wf])
        except Exception as e:
            logger.debug(f"Could not calculate waterfall SNR: {e}")
            snr_waterfall = None
            peak_time_waterfall = None
    
    for peak_idx in peaks:
        # Create a box centred on the peak.
        cx = int(max(0, min(img_w - 1, peak_idx)))
        # First get approximate DM to calculate box position
        dm_val_approx = _dm_from_image_at_time(band_img, cx)
        cy_row = int(round((dm_val_approx - float(config.DM_min)) / max(float(config.DM_max - config.DM_min), 1e-6) * (img_h - 1)))
        x1_raw = max(0, cx - half_w)
        x2_raw = min(img_w - 1, cx + half_w)
        y1_raw = max(0, cy_row - half_h)
        y2_raw = min(img_h - 1, cy_row + half_h)
        
        # CRITICAL: Use extract_candidate_dm with box center (same as plot_composite.py line 260)
        # This ensures DM in CSV matches exactly what's shown in plot label
        center_x = (x1_raw + x2_raw) / 2.0
        center_y = (y1_raw + y2_raw) / 2.0
        effective_len_det = slice_samples if slice_samples is not None else slice_len
        dm_val, t_sec_real, t_sample_real = extract_candidate_dm(center_x, center_y, effective_len_det)
        
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
        # PHASE 3a: ResNet Classification on INTENSITY
        # =====================================================================
        patch_intensity, start_sample = dedisperse_patch(data_block, freq_down, dm_val, global_sample)

        snr_val_intensity = 0.0
        peak_idx_patch = None
        if patch_intensity is not None and patch_intensity.size > 0:
            snr_profile_pre, _, best_w_vec = compute_snr_profile(patch_intensity)
            if snr_profile_pre.size > 0:
                peak_idx_patch = int(np.argmax(snr_profile_pre))
                snr_val_intensity = float(np.max(snr_profile_pre))
        else:
            snr_val_intensity = snr_peak

        from ..detection.model_interface import classify_patch
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
        
        # =====================================================================
        # PHASE 3b: ResNet Classification on LINEAR POLARIZATION
        # =====================================================================
        class_prob_linear = 0.0
        is_burst_linear = False
        
        if data_block_linear is not None:
            # Dedisperse Linear polarization patch at same DM and time
            patch_linear, _ = dedisperse_patch(data_block_linear, freq_down, dm_val, global_sample)
            
            if patch_linear is not None and patch_linear.size > 0:
                class_prob_linear, proc_patch_linear = classify_patch(cls_model, patch_linear)
                is_burst_linear = class_prob_linear >= float(config.CLASS_PROB)
                
                logger.debug(
                    "Phase 3b: Linear classification - DM=%.2f t_idx=%d class_prob=%.3f is_burst=%s",
                    dm_val, peak_idx, class_prob_linear, is_burst_linear
                )
        
        # =====================================================================
        # DECISION LOGIC: Determine if candidate should be saved
        # =====================================================================
        # save_only_burst = true:  Save ONLY if BOTH Intensity AND Linear classify as BURST
        # save_only_burst = false: Save if Intensity classifies as BURST (ignore Linear)
        
        should_save = False
        save_reason = ""
        
        if waterfall_block_linear is not None:
            # Multi-pol data available - apply dual validation
            if config.SAVE_ONLY_BURST:
                # Strict mode: Both must be BURST
                should_save = is_burst_intensity and is_burst_linear
                if should_save:
                    save_reason = "BURST in BOTH polarizations"
                else:
                    save_reason = f"Filtered: Intensity={'BURST' if is_burst_intensity else 'NO-BURST'}, Linear={'BURST' if is_burst_linear else 'NO-BURST'}"
            else:
                # Permissive mode: Only Intensity needs to be BURST
                should_save = is_burst_intensity
                if should_save:
                    save_reason = f"BURST in Intensity (Linear={'BURST' if is_burst_linear else 'NO-BURST'})"
                else:
                    save_reason = "NO-BURST in Intensity"
        else:
            # No multi-pol data - use only Intensity
            should_save = not config.SAVE_ONLY_BURST or is_burst_intensity
            save_reason = "BURST in Intensity" if is_burst_intensity else "NO-BURST in Intensity"
        
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

        # Record outputs.
        snr_list.append(snr_peak)
        top_conf.append(conf)
        top_boxes.append(box)
        class_probs_list.append(class_prob_intensity)
        class_probs_linear_list.append(class_prob_linear)  # NEW: Store Linear prob
        candidate_times_abs.append(float(absolute_candidate_time))

        # Keep track of the best candidate.
        if best_patch is None or (is_burst and not best_is_burst):
            best_patch = proc_patch_intensity
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
            snr_waterfall,  # SNR from waterfall raw (peak_snr_wf)
            float(snr_val_intensity),  # SNR from dedispersed patch
            width_ms,
            float(class_prob_intensity),  # Classification probability in Intensity (I)
            bool(is_burst_intensity),  # BURST classification in Intensity (I)
            float(class_prob_linear),  # Classification probability in Linear (L) - HF only
            bool(is_burst_linear) if data_block_linear is not None else None,  # BURST classification in Linear (L) - HF only
            bool(is_burst),  # Final classification (I+L when SAVE_ONLY_BURST=True)
            patch_path.name,
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
        )
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
        for block, block_raw, metadata, pol_type in stream_fits_multi_pol(str(fits_path), effective_chunk_samples, overlap_samples=overlap_raw):
            actual_chunk_count += 1
            log_block_processing(actual_chunk_count, block.shape, str(block.dtype), metadata)
            
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
                chunk_params = get_chunk_processing_parameters(metadata, collector=collector)
                freq_down = chunk_params['freq_down']
                slice_len = chunk_params['slice_len']
                time_slice = chunk_params['time_slice']
                overlap_left_ds = chunk_params['overlap_left_ds']
                overlap_right_ds = chunk_params['overlap_right_ds']

                # Memory validation is now done inside build_dm_time_cube (PRESTO-style)
                height = chunk_params['height']
                dm_time_full = build_dm_time_cube(block_ds, height=height, dm_min=config.DM_min, dm_max=config.DM_max, collector=collector)
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
                    )
                    cand_counter_total += cands
                    n_bursts_total += bursts
                    n_no_bursts_total += nobursts
                    prob_max_total = max(prob_max_total, pmax)
                
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
    }


