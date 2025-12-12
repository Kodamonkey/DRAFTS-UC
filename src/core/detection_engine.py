# This module runs detection, classification, and visualization logic.

"""Detection engine that scores candidates, classifies them, and generates plots."""
from __future__ import annotations

                          
import logging

                     
import numpy as np

# Local imports
from ..analysis.snr_utils import compute_snr_profile, find_snr_peak
from ..detection.model_interface import classify_patch, detect
from ..logging.logging_config import Colors, get_global_logger
from ..output.candidate_manager import Candidate, append_candidate
from ..preprocessing.dm_candidate_extractor import extract_candidate_dm
from ..preprocessing.dedispersion import dedisperse_block, dedisperse_patch
from .mjd_utils import calculate_candidate_mjd

from ..visualization.visualization_unified import (
    postprocess_img,
    preprocess_img,
    save_all_plots
)

              
logger = logging.getLogger(__name__)

def detect_and_classify_candidates_in_band(
    det_model,
    cls_model,
    band_img,
    slice_len,
    j,
    fits_path,
    save_dir,
    data,
    freq_down,
    csv_file,
    time_reso_ds,
    snr_list,
    config,
    absolute_start_time=None,
    patches_dir=None,
    chunk_idx=None,                
    band_idx=None,                  
    slice_start_idx: int | None = None,
    waterfall_block=None,  # time x frequency slice block for waterfall SNR calculation
    slice_samples: int | None = None,  # actual slice samples (may differ from slice_len)
    off_regions=None,  # off-pulse regions for SNR calculation
):
    """Run detection and classification for a specific frequency band.

    The function detects candidates, classifies them, computes SNR metrics,
    selects the best example for visualisation, and persists metadata for
    later aggregation.
    """
                                                         
    try:
        global_logger = get_global_logger()
    except ImportError:
        global_logger = None
    
                                                 
    band_names = ["Full Band", "Low Band", "High Band"]
    band_name = band_names[band_idx] if band_idx is not None and band_idx < len(band_names) else f"Band {band_idx}"
    if global_logger:
        global_logger.processing_band(band_name, j)
    
    img_tensor = preprocess_img(band_img)
    top_conf, top_boxes = detect(det_model, img_tensor)
    img_rgb = postprocess_img(img_tensor)
    if top_boxes is None:
        top_conf = []
        top_boxes = []
    
                                           
    if global_logger:
        global_logger.band_candidates(band_name, len(top_conf))
    
    first_patch = None
    first_start = None
    first_dm = None
    if patches_dir is not None:
        patch_path = patches_dir / f"patch_slice{j}_band{band_img.shape[0] if hasattr(band_img, 'shape') else 0}.png"
    else:
        patch_dir = save_dir / "Patches" / fits_path.stem
        patch_path = patch_dir / f"patch_slice{j}_band{band_img.shape[0] if hasattr(band_img, 'shape') else 0}.png"
    class_probs_list = []
    candidate_times_abs: list[float] = []
    snr_waterfall_intensity_list: list[float | None] = []  # NEW: SNR from Intensity waterfall per candidate
    snr_patch_intensity_list: list[float | None] = []  # NEW: SNR from dedispersed Intensity patch per candidate
    cand_counter = 0
    n_bursts = 0
    n_no_bursts = 0
    prob_max = 0.0
    
                                                                     
    best_patch = None
    best_start = None
    best_dm = None
    best_is_burst = False
    first_patch = None
    first_start = None
    first_dm = None
    
    # Calculate waterfall SNR profile once (will be reused for each candidate)
    # This ensures consistency between CSV and plots
    snr_wf_profile = None
    snr_waterfall_global = None  # Global peak (for backward compatibility)
    peak_time_waterfall = None
    if waterfall_block is not None and waterfall_block.size > 0:
        try:
            snr_wf, _, _ = compute_snr_profile(waterfall_block, off_regions)
            snr_wf_profile = snr_wf  # Store profile for per-candidate SNR calculation
            if snr_wf.size > 0:
                peak_snr_wf, _, peak_idx_wf = find_snr_peak(snr_wf)
                snr_waterfall_global = float(peak_snr_wf)  # Global peak (for backward compatibility)
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
                if peak_idx_wf < len(time_axis_wf):
                    peak_time_waterfall = float(time_axis_wf[peak_idx_wf])
                else:
                    # Fallback: calculate directly from peak index
                    peak_time_waterfall = slice_start_abs + peak_idx_wf * time_reso_ds
        except Exception as e:
            logger.debug(f"Could not calculate waterfall SNR: {e}")
            snr_wf_profile = None
            snr_waterfall_global = None
            peak_time_waterfall = None
                                                          
    all_candidates = []
    
    for conf, box in zip(top_conf, top_boxes):
        dm_val, t_sec, t_sample = extract_candidate_dm(
            (box[0] + box[2]) / 2,
            (box[1] + box[3]) / 2,
            slice_len,
        )
        
                                                
                                                          
        x1, y1, x2, y2 = map(int, box)
        candidate_region = band_img[y1:y2, x1:x2]
        if candidate_region.size > 0:
            # Compute SNR for consistency with the composite visualisation.
            snr_profile, _, _ = compute_snr_profile(candidate_region)
            snr_val_raw = np.max(snr_profile)  # Use the peak SNR value.
        else:
            snr_val_raw = 0.0
        
        snr_list.append(snr_val_raw)                                     
                                                                                                     
        if slice_start_idx is not None:
            global_sample = int(slice_start_idx) + int(t_sample)
        else:
                                                                                        
            global_sample = j * slice_len + int(t_sample)
        patch, start_sample = dedisperse_patch(
            data, freq_down, dm_val, global_sample
        )
        
                                                                   
        snr_val = 0.0                     
        peak_idx_patch = None
        if patch is not None and patch.size > 0:
            # Measure SNR on the dedispersed patch using the unified routine.
            snr_profile_pre, _, best_w_vec = compute_snr_profile(patch)
            peak_idx_patch = int(np.argmax(snr_profile_pre)) if snr_profile_pre.size > 0 else None
            snr_val = float(np.max(snr_profile_pre)) if snr_profile_pre.size > 0 else 0.0
        else:
                                                            
            snr_val = snr_val_raw
        class_prob, proc_patch = classify_patch(cls_model, patch)
        class_probs_list.append(class_prob)
        is_burst = class_prob >= config.CLASS_PROB
        
                                                      
                                                         
        if first_patch is None:
            first_patch = proc_patch
                                                                                                 
            offset_within_slice = (start_sample - (slice_start_idx if slice_start_idx is not None else 0))
            first_start = (absolute_start_time if absolute_start_time is not None else 0.0) 
            first_start += offset_within_slice * config.TIME_RESO * config.DOWN_TIME_RATE
            
            first_dm = dm_val
        
                                                                     
        candidate_info = {
            'patch': proc_patch,
            'start': first_start,                             
            'dm': dm_val,
            'is_burst': is_burst,
            'confidence': conf,
            'class_prob': class_prob
        }
        all_candidates.append(candidate_info)
        
                                                                          
        if best_patch is None:
                                                              
            best_patch = proc_patch
            best_start = first_start
            best_dm = dm_val
            best_is_burst = is_burst
        elif is_burst and not best_is_burst:
                                                                               
            best_patch = proc_patch
            best_start = first_start
            best_dm = dm_val
            best_is_burst = is_burst
                                                                                           
        
                                                                     
        dt_ds = config.TIME_RESO * config.DOWN_TIME_RATE
        # Calculate detection time from DM-time plot (same as plot_composite.py line 264-269)
        # This is the time shown in the plot label
        if absolute_start_time is not None:
            detection_time_dm_time = absolute_start_time + t_sec
        else:
            detection_time_dm_time = j * slice_len * time_reso_ds + t_sec
        
        if peak_idx_patch is not None:
                                                                    
            slice_offset_samples = (start_sample - (slice_start_idx if slice_start_idx is not None else 0))
            patch_start_abs = (absolute_start_time if absolute_start_time is not None else 0.0) + slice_offset_samples * dt_ds
            absolute_candidate_time = patch_start_abs + peak_idx_patch * dt_ds
        else:
            # Use DM-time detection time as fallback
            absolute_candidate_time = detection_time_dm_time

        
        candidate_times_abs.append(float(detection_time_dm_time))
        
        # Calculate SNR from waterfall at candidate position (similar to HF pipeline)
        # Map candidate time to index in waterfall SNR profile
        snr_intensity_at_peak = None
        if snr_wf_profile is not None and snr_wf_profile.size > 0:
            try:
                # Use detection_time_dm_time (already calculated above, same as shown in plot label)
                candidate_time_abs = detection_time_dm_time
                
                # Calculate time axis for SNR profile (same as in waterfall calculation above)
                if absolute_start_time is not None:
                    slice_start_abs = absolute_start_time
                else:
                    slice_start_abs = j * slice_len * time_reso_ds
                snr_samples = len(snr_wf_profile)
                slice_end_abs = slice_start_abs + snr_samples * time_reso_ds
                time_axis_wf = np.linspace(slice_start_abs, slice_end_abs, snr_samples)
                
                # Find closest index in SNR profile to candidate time
                candidate_idx_in_wf = np.argmin(np.abs(time_axis_wf - candidate_time_abs))
                if 0 <= candidate_idx_in_wf < len(snr_wf_profile):
                    snr_intensity_at_peak = float(snr_wf_profile[candidate_idx_in_wf])
                    logger.info(f"Candidate at t={candidate_time_abs:.6f}s -> SNR profile idx={candidate_idx_in_wf}/{len(snr_wf_profile)}, SNR={snr_intensity_at_peak:.2f} (time_axis range: {time_axis_wf[0]:.6f} to {time_axis_wf[-1]:.6f})")
                else:
                    logger.warning(f"Candidate idx {candidate_idx_in_wf} out of range [0, {len(snr_wf_profile)})")
            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"Could not get SNR at candidate position: {e}")
        else:
            logger.warning(f"snr_wf_profile is None or empty (size={snr_wf_profile.size if snr_wf_profile is not None else 0})")
        snr_waterfall_intensity_list.append(snr_intensity_at_peak)
        snr_patch_intensity_list.append(float(snr_val) if snr_val is not None else None)
        
        # Assemble a candidate record with optional width estimates.
        width_ms = None
        try:
            if peak_idx_patch is not None and 'best_w_vec' in locals() and best_w_vec.size > 0:
                dt_ds = config.TIME_RESO * config.DOWN_TIME_RATE
                width_ms = float(best_w_vec[int(peak_idx_patch)] * dt_ds * 1000.0)
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
            band_idx if band_idx is not None else 0,                  
            float(conf),
            dm_val,  # DM calculated with extract_candidate_dm (same as plot)
            float(detection_time_dm_time),  # Time from DM-time plot (same as plot label)
            peak_time_waterfall,  # Time from waterfall SNR peak (different method)
            t_sample,
            tuple(map(int, box)),
            snr_intensity_at_peak if snr_intensity_at_peak is not None else snr_waterfall_global,  # SNR from waterfall at candidate position (or global peak as fallback)
            float(snr_val),  # SNR from dedispersed patch
            width_ms,
            class_prob,  # Classification probability in Intensity (I) - classic pipeline
            is_burst,  # BURST classification in Intensity (I) - classic pipeline
            None,  # class_prob_linear - not available in classic pipeline
            None,  # is_burst_linear - not available in classic pipeline
            is_burst,  # Final classification (same as Intensity for classic)
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
        
        # Save candidate based on filtering mode
        # SAVE_ONLY_BURST = True  → Only save if classified as BURST
        # SAVE_ONLY_BURST = False → Save all CenterNet detections (BURST + NON-BURST)
        if not config.SAVE_ONLY_BURST or is_burst:
            append_candidate(csv_file, cand.to_row())
            
            # Log the saved candidate
            try:
                global_logger = get_global_logger()
                global_logger.candidate_detected(dm_val, absolute_candidate_time, conf, class_prob, is_burst, snr_val_raw, snr_val)
            except ImportError:
                logger.info(
                    f"Candidate DM {dm_val:.2f} t={absolute_candidate_time:.3f}s conf={conf:.2f} class={class_prob:.2f} → {'BURST' if is_burst else 'no burst'}"
                )
                logger.info(
                    f"SNR Raw: {snr_val_raw:.2f}σ, SNR Patch Dedispersed: {snr_val:.2f}σ (saved to CSV)"
                )
        else:
            # Candidate detected by CenterNet but filtered out because not classified as BURST
            logger.debug(
                f"NON-BURST filtered (SAVE_ONLY_BURST=True): DM {dm_val:.2f} t={absolute_candidate_time:.3f}s "
                f"CenterNet_conf={conf:.2f} ResNet_class={class_prob:.2f} → NOT SAVED"
            )
                                                      
                                                                 
                                               
    final_patch = best_patch if best_patch is not None else first_patch
    final_start = best_start if best_start is not None else first_start
    final_dm = best_dm if best_dm is not None else first_dm
    
                                                      
    if len(all_candidates) > 1:
        burst_count = sum(1 for c in all_candidates if c['is_burst'])
        if global_logger:
            global_logger.logger.info(
                f"{Colors.OKCYAN}Slice {j} - {band_name}: {len(all_candidates)} candidates "
                f"({burst_count} BURST, {len(all_candidates) - burst_count} NO BURST). "
                f"Selected: {'BURST' if best_is_burst else 'NO BURST'} (DM={final_dm:.2f}){Colors.ENDC}"
            )
    
    return {
        "top_conf": top_conf,
        "top_boxes": top_boxes,
        "class_probs_list": class_probs_list,
        "snr_waterfall_intensity_list": snr_waterfall_intensity_list,  # NEW: SNR from Intensity waterfall per candidate
        "snr_patch_intensity_list": snr_patch_intensity_list,  # NEW: SNR from dedispersed Intensity patch per candidate
        "first_patch": final_patch,                                
        "first_start": final_start,                                
        "first_dm": final_dm,                                      
        "img_rgb": img_rgb,
        "cand_counter": cand_counter,
        "n_bursts": n_bursts,
        "n_no_bursts": n_no_bursts,
        "prob_max": prob_max,
        "patch_path": patch_path,
        "best_is_burst": best_is_burst,                                    
        "total_candidates": len(all_candidates),                                    
        "candidate_times_abs": candidate_times_abs,
    }

def process_slice_with_multiple_bands(
    j,
    dm_time,
    block,
    slice_len,
    det_model,
    cls_model,
    fits_path,
    save_dir,
    freq_down,
    csv_file,
    time_reso_ds,
    band_configs,
    snr_list,
    config,
    absolute_start_time=None,
    composite_dir=None,
    detections_dir=None,
    patches_dir=None,
    chunk_idx=None,                 
    force_plots: bool = False,
    slice_start_idx: int | None = None,
    slice_end_idx: int | None = None,
):
    """Process a slice across all configured frequency bands and persist outputs."""

    try:
        global_logger = get_global_logger()
    except ImportError:
        global_logger = None
    
                                                  
    if global_logger:
        chunk_info = f" (chunk {chunk_idx:03d})" if chunk_idx is not None else ""
        global_logger.logger.info(f"{Colors.PROCESSING} Processing slice {j:03d}{chunk_info}{Colors.ENDC}")
    
                                                                                              
    if slice_start_idx is not None and slice_end_idx is not None:
        start_idx = int(slice_start_idx)
        end_idx = int(slice_end_idx)
    else:
        start_idx = slice_len * j
        end_idx = slice_len * (j + 1)

                                                                 
    try:
        real_samples = end_idx - start_idx
        if global_logger:
            chunk_info = f" (chunk {chunk_idx:03d})" if chunk_idx is not None else ""
            global_logger.logger.info(
                f"{Colors.PROCESSING} Slice {j:03d}{chunk_info}: {real_samples} real samples "
                f"(downsampled) [{start_idx}→{end_idx}){Colors.ENDC}"
            )
        else:
            logger.info(
                f"Slice {j:03d}: {end_idx - start_idx} real samples (downsampled) "
                f"[{start_idx}→{end_idx})"
            )
    except Exception:

        logger.info(
            f"Slice {j:03d}: {end_idx - start_idx} real samples (downsampled) "
            f"[{start_idx}→{end_idx})"
        )

    slice_cube = dm_time[:, :, start_idx:end_idx]
    waterfall_block = block[start_idx:end_idx]
    if slice_cube.size == 0 or waterfall_block.size == 0:
        logger.warning(f"Slice {j}: slice_cube or waterfall_block empty, skipping...")
        return 0, 0, 0, 0.0
    
                                                             
    if absolute_start_time is None:
                                                 
        absolute_start_time = start_idx * config.TIME_RESO * config.DOWN_TIME_RATE
    
                                                    
    if global_logger:
        global_logger.logger.debug(f"{Colors.OKCYAN} Creating dedispersed waterfall for slice {j}{Colors.ENDC}")
    

    slice_has_candidates = False                                      
    cand_counter = 0                         
    n_bursts = 0                                       
    n_no_bursts = 0                                          
    prob_max = 0.0                                   
    
    fits_stem = fits_path.stem                     
                                         
    if composite_dir is not None:
        comp_path = composite_dir / f"{fits_stem}_slice{j:03d}.png"                         
    else:
        comp_path = save_dir / "Composite" / f"{fits_stem}_slice{j:03d}.png"

    if patches_dir is not None:
        patch_path = patches_dir / f"{fits_stem}_slice{j:03d}.png"
    else:
        patch_path = save_dir / "Patches" / f"{fits_stem}_slice{j:03d}.png"

    if detections_dir is not None:
        out_img_path = detections_dir / f"{fits_stem}_slice{j:03d}.png"
    else:
        out_img_path = save_dir / "Detections" / f"{fits_stem}_slice{j:03d}.png"

                                                                                       
    time_slice = block.shape[0] // slice_len
    if block.shape[0] % slice_len != 0:
        time_slice += 1

    for band_idx, band_suffix, band_name in band_configs:
        band_img = slice_cube[band_idx]
        band_result = detect_and_classify_candidates_in_band(
            det_model,
            cls_model,
            band_img,
            end_idx - start_idx,                                  
            j,
            fits_path,
            save_dir,
            block,
            freq_down,
            csv_file,
            time_reso_ds,
            snr_list,
            config,
            absolute_start_time=absolute_start_time,                             
            patches_dir=patches_dir,                                   
            chunk_idx=chunk_idx,                
            band_idx=band_idx,                  
            slice_start_idx=start_idx,
            waterfall_block=waterfall_block,  # Pass waterfall block for SNR calculation
            slice_samples=end_idx - start_idx,  # Actual slice samples
            off_regions=None,  # Can be passed if available
        )
        cand_counter += band_result["cand_counter"]
        n_bursts += band_result["n_bursts"]
        n_no_bursts += band_result["n_no_bursts"]
        prob_max = max(prob_max, band_result["prob_max"])
        if len(band_result["top_conf"]) > 0:
            slice_has_candidates = True

        dedisp_block = None

                                                                                                          
        should_generate_plots = False
        if config.SAVE_ONLY_BURST:
                                                                                       
            should_generate_plots = (n_bursts > 0) or force_plots
        else:
                                                                                  
            should_generate_plots = slice_has_candidates or force_plots
        
        if should_generate_plots:
            if slice_has_candidates and global_logger:
                global_logger.slice_completed(j, cand_counter, n_bursts, n_no_bursts)

            dm_to_use = band_result["first_dm"] if band_result["first_dm"] is not None else 0.0
       
            start = start_idx
            block_len = end_idx - start_idx
            dedisp_block = dedisperse_block(block, freq_down, dm_to_use, start, block_len)
            
            if global_logger:
                global_logger.creating_waterfall("dedispersed", j, dm_to_use)
                global_logger.generating_plots()

            save_all_plots(
                waterfall_block,
                dedisp_block,
                band_result["img_rgb"],
                band_result["first_patch"],
                band_result["first_start"],
                band_result["first_dm"],
                band_result["top_conf"],
                band_result["top_boxes"],
                band_result["class_probs_list"],
                comp_path,
                j,
                time_slice,
                band_name,
                band_suffix,
                fits_stem,
                end_idx - start_idx,
                normalize=True,
                off_regions=None,
                thresh_snr=config.SNR_THRESH,
                band_idx=band_idx,
                patch_path=band_result["patch_path"],
                absolute_start_time=absolute_start_time,
                chunk_idx=chunk_idx, 
                force_plots=force_plots,
                snr_waterfall_intensity_list=band_result.get("snr_waterfall_intensity_list"),  # NEW: SNR from Intensity waterfall
                snr_patch_intensity_list=band_result.get("snr_patch_intensity_list"),  # NEW: SNR from dedispersed Intensity patch
            )
        else:
            if global_logger:
                if config.SAVE_ONLY_BURST and n_no_bursts > 0:
                    global_logger.logger.debug(f"{Colors.OKCYAN} Slice {j}: Only non-burst candidates detected (SAVE_ONLY_BURST=True, no plots){Colors.ENDC}")
                else:
                    global_logger.logger.debug(f"{Colors.OKCYAN} Slice {j}: No candidates detected{Colors.ENDC}")
    
                                                                                                  
    if config.SAVE_ONLY_BURST:
                                                                          
        effective_cand_counter = n_bursts
        effective_n_bursts = n_bursts
        effective_n_no_bursts = 0                                                   
    else:
                                                            
        effective_cand_counter = cand_counter
        effective_n_bursts = n_bursts
        effective_n_no_bursts = n_no_bursts
    
    return effective_cand_counter, effective_n_bursts, effective_n_no_bursts, prob_max 
