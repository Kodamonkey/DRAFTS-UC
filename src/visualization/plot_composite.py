# This module creates composite diagnostic visualizations.

"""Composite plot generation module for FRB pipeline."""
from __future__ import annotations

                          
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

                     
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

               
from ..analysis.snr_utils import compute_snr_profile, find_snr_peak
from ..config import config
from ..preprocessing.dm_candidate_extractor import extract_candidate_dm
from .visualization_ranges import get_dynamic_dm_range_for_candidate

              
logger = logging.getLogger(__name__)


def _calculate_dynamic_dm_range(
    top_boxes: Iterable | None,
    slice_len: int,
    fallback_dm_min: int = None,
    fallback_dm_max: int = None,
    confidence_scores: Iterable | None = None
) -> Tuple[float, float]:
    """Unified delegate: use visualization_ranges for dynamic DM range."""
    if (not getattr(config, 'DM_DYNAMIC_RANGE_ENABLE', True)
        or top_boxes is None
        or len(top_boxes) == 0):
        dm_min = fallback_dm_min if fallback_dm_min is not None else config.DM_min
        dm_max = fallback_dm_max if fallback_dm_max is not None else config.DM_max
        return float(dm_min), float(dm_max)

    dm_candidates: List[float] = []
    for box in top_boxes:
        x1, y1, x2, y2 = map(int, box)
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        dm_val, _, _ = extract_candidate_dm(center_x, center_y, slice_len)
        dm_candidates.append(dm_val)
    if not dm_candidates:
        dm_min = fallback_dm_min if fallback_dm_min is not None else config.DM_min
        dm_max = fallback_dm_max if fallback_dm_max is not None else config.DM_max
        return float(dm_min), float(dm_max)

    if confidence_scores is not None and len(confidence_scores) > 0:
        best_idx = int(np.argmax(confidence_scores))
        dm_optimal = float(dm_candidates[best_idx])
        confidence = float(confidence_scores[best_idx])
    else:
        dm_optimal = float(np.median(dm_candidates))
        confidence = 0.8

    try:
        return get_dynamic_dm_range_for_candidate(
            dm_optimal=dm_optimal,
            config_module=config,
            visualization_type=getattr(config, 'DM_RANGE_DEFAULT_VISUALIZATION', 'detailed'),
            confidence=confidence,
            range_factor=getattr(config, 'DM_RANGE_FACTOR', 0.2),
            min_range_width=getattr(config, 'DM_RANGE_MIN_WIDTH', 50.0),
            max_range_width=getattr(config, 'DM_RANGE_MAX_WIDTH', 200.0),
        )
    except Exception as e:
        print(f"[WARNING] Error calculating dynamic DM range: {e}")
        dm_min = fallback_dm_min if fallback_dm_min is not None else config.DM_min
        dm_max = fallback_dm_max if fallback_dm_max is not None else config.DM_max
        return float(dm_min), float(dm_max)


def get_band_frequency_range(band_idx: int) -> Tuple[float, float]:
    """Get the frequency range (min, max) for a specific band."""
    freq_ds = np.mean(
        config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE),
        axis=1,
    )
    
    if band_idx == 0:             
        return freq_ds.min(), freq_ds.max()
    elif band_idx == 1:             
        mid_channel = len(freq_ds) // 2
        return freq_ds.min(), freq_ds[mid_channel]
    elif band_idx == 2:             
        mid_channel = len(freq_ds) // 2  
        return freq_ds[mid_channel], freq_ds.max()
    else:
        logger.warning(f"Invalid band index {band_idx}, using Full Band range")
        return freq_ds.min(), freq_ds.max()


def get_band_name_with_freq_range(band_idx: int, band_name: str) -> str:
    """Get band name with frequency range information."""
    freq_min, freq_max = get_band_frequency_range(band_idx)
    return f"{band_name} ({freq_min:.0f}-{freq_max:.0f} MHz)"


def create_composite_plot(
    waterfall_block: np.ndarray,
    dedispersed_block: np.ndarray,
    img_rgb: np.ndarray,
    patch_img: np.ndarray,
    patch_start: float,
    dm_val: float,
    top_conf: Iterable,
    top_boxes: Iterable | None,
    class_probs: Iterable | None,
    slice_idx: int,
    time_slice: int,
    band_name: str,
    band_suffix: str,
    fits_stem: str,
    slice_len: int,
    normalize: bool = False,
    off_regions: Optional[List[Tuple[int, int]]] = None,
    thresh_snr: Optional[float] = None,
    band_idx: int = 0,
    absolute_start_time: Optional[float] = None, 
    chunk_idx: Optional[int] = None,  
    slice_samples: Optional[int] = None,  
    candidate_times_abs: Optional[Iterable[float]] = None,
    dedisp_block_linear: Optional[np.ndarray] = None,
    dedisp_block_circular: Optional[np.ndarray] = None,
    class_probs_linear: Optional[Iterable[float]] = None,  # NEW: Linear classification probs
) -> plt.Figure:
    """Create composite figure with detections and waterfalls with SNR analysis.
    
    For high-frequency pipeline with multi-polarization data, the bottom panels show:
    - Left: Dedispersed waterfall in Intensity (Stokes I)
    - Middle: Dedispersed waterfall in Linear Polarization
    - Right: Dedispersed waterfall in Circular Polarization
    
    For standard pipeline, shows the classic layout:
    - Left: Dispersed waterfall
    - Middle: Dedispersed waterfall  
    - Right: Candidate patch
    """
    
                                          
    band_name_with_freq = get_band_name_with_freq_range(band_idx, band_name)
    
    freq_ds = np.mean(
        config.FREQ.reshape(
            config.FREQ_RESO // config.DOWN_FREQ_RATE,
            config.DOWN_FREQ_RATE,
        ),
        axis=1,
    )
    time_reso_ds = config.TIME_RESO * config.DOWN_TIME_RATE

    # Determine if we're in multi-polarization mode (HF pipeline)
    multi_pol_mode = (dedisp_block_linear is not None and dedisp_block_circular is not None)
    
    if multi_pol_mode:
        logger.info("Composite plot: Multi-polarization mode (HF pipeline) - Using 3 dedispersed waterfalls (I, L, V)")
        logger.debug("Multi-pol blocks: Linear shape=%s, Circular shape=%s", 
                    dedisp_block_linear.shape if dedisp_block_linear is not None else None,
                    dedisp_block_circular.shape if dedisp_block_circular is not None else None)
    else:
        logger.info("Composite plot: Standard mode - Using dispersed/dedispersed/patch layout")
        if dedisp_block_linear is None:
            logger.debug("dedisp_block_linear is None")
        if dedisp_block_circular is None:
            logger.debug("dedisp_block_circular is None")
    
                                       
    if waterfall_block is not None and waterfall_block.size > 0:
        wf_block = waterfall_block.copy()
    else:
        wf_block = None
    
                                         
    if dedispersed_block is not None and dedispersed_block.size > 0:
        dw_block = dedispersed_block.copy()
    else:
        dw_block = None
    
    # Prepare multi-pol blocks for HF pipeline
    dw_linear = None
    dw_circular = None
    if multi_pol_mode:
        if dedisp_block_linear is not None and dedisp_block_linear.size > 0:
            dw_linear = dedisp_block_linear.copy()
        if dedisp_block_circular is not None and dedisp_block_circular.size > 0:
            dw_circular = dedisp_block_circular.copy()
    
    if normalize:
        blocks_to_norm = [wf_block, dw_block]
        if multi_pol_mode:
            blocks_to_norm.extend([dw_linear, dw_circular])
        
        for block in blocks_to_norm:
            if block is not None:
                block += 1
                block /= np.mean(block, axis=0)
                vmin, vmax = np.nanpercentile(block, [5, 95])
                block[:] = np.clip(block, vmin, vmax)
                block -= block.min()
                block /= block.max() - block.min()

                                    
    if absolute_start_time is not None:
        slice_start_abs = absolute_start_time
    else:
        slice_start_abs = slice_idx * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
    
    real_samples = slice_samples if slice_samples is not None else slice_len
    slice_end_abs = slice_start_abs + real_samples * config.TIME_RESO * config.DOWN_TIME_RATE

                                
    fig = plt.figure(figsize=(14, 12))
    gs_main = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.3, figure=fig)
    
                            
    ax_det = fig.add_subplot(gs_main[0, 0])
    ax_det.imshow(img_rgb, origin="lower", aspect="auto")
    ax_det.set_title("Detection Results", fontsize=10, fontweight="bold")
    ax_det.set_xlabel("Time (s)", fontsize=9)
    ax_det.set_ylabel("Dispersion Measure (pc cm⁻³)", fontsize=9)

                                              
    n_time_ticks_det = 8
    time_positions_det = np.linspace(0, img_rgb.shape[1] - 1, n_time_ticks_det)
    n_px = img_rgb.shape[1]
    denom = float(max(n_px - 1, 1))
    time_values_det = slice_start_abs + (time_positions_det / denom) * (slice_end_abs - slice_start_abs)
    ax_det.set_xticks(time_positions_det)
    ax_det.set_xticklabels([f"{t:.6f}" for t in time_values_det])
    ax_det.set_xlabel("Time (s)", fontsize=10, fontweight="bold")

                                           
    n_dm_ticks = 8
    dm_positions = np.linspace(0, img_rgb.shape[0] - 1, n_dm_ticks)
    dm_plot_min, dm_plot_max = _calculate_dynamic_dm_range(
        top_boxes=top_boxes,
        slice_len=slice_len,
        fallback_dm_min=config.DM_min,
        fallback_dm_max=config.DM_max,
        confidence_scores=top_conf if top_conf is not None else None
    )
    
    dm_values = dm_plot_min + (dm_positions / img_rgb.shape[0]) * (dm_plot_max - dm_plot_min)
    ax_det.set_yticks(dm_positions)
    ax_det.set_yticklabels([f"{dm:.0f}" for dm in dm_values])
    ax_det.set_ylabel("Dispersion Measure (pc cm⁻³)", fontsize=10, fontweight="bold")

                                                   
    if top_boxes is not None:
        for idx, (conf, box) in enumerate(zip(top_conf, top_boxes)):
            x1, y1, x2, y2 = map(int, box)
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            effective_len_det = slice_samples if slice_samples is not None else slice_len
            dm_val_cand, t_sec_real, t_sample_real = extract_candidate_dm(center_x, center_y, effective_len_det)
            
                                               
            if candidate_times_abs is not None and idx < len(candidate_times_abs):
                detection_time = float(candidate_times_abs[idx])
            else:
                if absolute_start_time is not None:
                    detection_time = absolute_start_time + t_sec_real
                else:
                    detection_time = slice_idx * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE + t_sec_real
            
            # Dual-polarization classification color logic (HF pipeline)
            if class_probs is not None and idx < len(class_probs):
                class_prob_I = class_probs[idx]
                is_burst_I = class_prob_I >= config.CLASS_PROB
                
                # Check if we have Linear classification (HF pipeline)
                is_burst_L = False
                class_prob_L = 0.0
                has_linear_classification = (class_probs_linear is not None and 
                                            idx < len(class_probs_linear) and 
                                            class_probs_linear[idx] > 0.0)
                
                if has_linear_classification:
                    class_prob_L = class_probs_linear[idx]
                    is_burst_L = class_prob_L >= config.CLASS_PROB
                    
                    # HF Mode: Verde solo si AMBAS clasifican BURST
                    color = "lime" if (is_burst_I and is_burst_L) else "orange"
                    
                    if is_burst_I and is_burst_L:
                        burst_status = "BURST (I+L)"
                    elif is_burst_I and not is_burst_L:
                        burst_status = "I:BURST L:NO"
                    else:
                        burst_status = "NO BURST"
                    
                    label = (
                        f"#{idx+1}\n"
                        f"DM: {dm_val_cand:.1f}\n"
                        f"Time: {detection_time:.3f}s\n"
                        f"I: {class_prob_I:.2f}\n"
                        f"L: {class_prob_L:.2f}\n"
                        f"{burst_status}"
                    )
                else:
                    # Standard mode: Solo Intensity
                    color = "lime" if is_burst_I else "orange"
                    burst_status = "BURST" if is_burst_I else "NO BURST"
                    
                    label = (
                        f"#{idx+1}\n"
                        f"DM: {dm_val_cand:.1f}\n"
                        f"Time: {detection_time:.3f}s\n"
                        f"Det: {conf:.2f}\n"
                        f"Cls: {class_prob_I:.2f}\n"
                        f"{burst_status}"
                    )
            else:
                color = "lime"
                label = f"#{idx+1}\nDM: {dm_val_cand:.1f}\nTime: {detection_time:.3f}s\nDet: {conf:.2f}"
            
                                      
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, 
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax_det.add_patch(rect)
            
            ax_det.annotate(
                label,
                xy=(center_x, center_y),
                xytext=(center_x, y2 + 10),
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                fontsize=7,
                ha="center",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            )
    
                               
    dm_range_info = f"{dm_plot_min:.0f}\u2013{dm_plot_max:.0f}"
    if getattr(config, 'DM_DYNAMIC_RANGE_ENABLE', True) and top_boxes is not None and len(top_boxes) > 0:
        dm_range_info += " (auto)"
    else:
        dm_range_info += " (full)"
    
    exact_slice_ms_det = (slice_samples * (config.TIME_RESO * config.DOWN_TIME_RATE)) * 1000.0
    title_det = (
        f"Detection Map - {fits_stem} ({band_name_with_freq})\n"
        f"Slice {slice_idx:03d} of {time_slice} | Duration: {exact_slice_ms_det:.6f} ms | "
        f"DM Range: {dm_range_info} pc cm⁻³"
    )
    ax_det.set_title(title_det, fontsize=11, fontweight="bold")

                                         
    gs_bottom_row = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_main[1, 0], width_ratios=[1, 1, 1], wspace=0.3
    )

    # =============================================================================
    # BOTTOM PANELS LAYOUT
    # =============================================================================
    # Multi-pol mode (HF pipeline): Show 3 dedispersed waterfalls (I, L, V)
    # Standard mode: Show dispersed, dedispersed, patch
    # =============================================================================
    
    if multi_pol_mode:
        # Use multi-polarization panel layout
        from .plot_multi_pol_panels import create_multi_pol_panels
        create_multi_pol_panels(
            fig=fig,
            gs_bottom_row=gs_bottom_row,
            dedisp_intensity=dw_block,
            dedisp_linear=dw_linear,
            dedisp_circular=dw_circular,
            dm_val=dm_val,
            slice_start_abs=slice_start_abs,
            slice_end_abs=slice_end_abs,
            freq_ds=freq_ds,
            time_reso_ds=time_reso_ds,
            thresh_snr=thresh_snr,
            off_regions=off_regions,
        )
        # Skip the standard 3-panel code below
        _skip_standard_panels = True
    else:
        _skip_standard_panels = False
        # LEFT PANEL: Dispersed waterfall (classic mode)
        gs_waterfall_nested = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs_bottom_row[0, 0], height_ratios=[1, 4], hspace=0.05
        )
        ax_prof_wf = fig.add_subplot(gs_waterfall_nested[0, 0])
    
    if not _skip_standard_panels and wf_block is not None and wf_block.size > 0:
        snr_wf, sigma_wf, best_w_wf = compute_snr_profile(wf_block, off_regions)
        peak_snr_wf, peak_time_wf, peak_idx_wf = find_snr_peak(snr_wf)
        
        time_axis_wf = np.linspace(slice_start_abs, slice_end_abs, len(snr_wf))
        peak_time_wf_abs = float(time_axis_wf[peak_idx_wf]) if len(snr_wf) > 0 else None
        ax_prof_wf.plot(time_axis_wf, snr_wf, color="royalblue", alpha=0.8, lw=1.5, label='SNR Profile')
        
        if thresh_snr is not None and config.SNR_SHOW_PEAK_LINES:
            above_thresh_wf = snr_wf >= thresh_snr
            if np.any(above_thresh_wf):
                ax_prof_wf.plot(time_axis_wf[above_thresh_wf], snr_wf[above_thresh_wf], 
                              color=config.SNR_HIGHLIGHT_COLOR, lw=2, alpha=0.9)
            ax_prof_wf.axhline(y=thresh_snr, color=config.SNR_HIGHLIGHT_COLOR, 
                             linestyle='--', alpha=0.7, linewidth=1)
        
        ax_prof_wf.plot(time_axis_wf[peak_idx_wf], peak_snr_wf, 'ro', markersize=5)
        ax_prof_wf.text(time_axis_wf[peak_idx_wf], peak_snr_wf + 0.1 * (ax_prof_wf.get_ylim()[1] - ax_prof_wf.get_ylim()[0]), 
                       f'{peak_snr_wf:.1f}σ', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax_prof_wf.set_xlim(time_axis_wf[0], time_axis_wf[-1])
        ax_prof_wf.set_ylabel('SNR (σ)', fontsize=8, fontweight='bold')
        ax_prof_wf.grid(True, alpha=0.3)
        ax_prof_wf.set_xticks([])
        
        width_ms_wf = float(best_w_wf[int(peak_idx_wf)]) * time_reso_ds * 1000.0 if len(best_w_wf) == len(snr_wf) else None
        if peak_time_wf_abs is not None:
            if width_ms_wf is not None:
                ax_prof_wf.set_title(
                    f"Raw Waterfall SNR\nPeak={peak_snr_wf:.1f}σ (w≈{width_ms_wf:.3f} ms) -> {peak_time_wf_abs:.6f}s",
                    fontsize=9, fontweight="bold",
                )
            else:
                ax_prof_wf.set_title(
                    f"Raw Waterfall SNR\nPeak={peak_snr_wf:.1f}σ -> {peak_time_wf_abs:.6f}s",
                    fontsize=9, fontweight="bold",
                )
        else:
            if width_ms_wf is not None:
                ax_prof_wf.set_title(f"Raw Waterfall SNR\nPeak={peak_snr_wf:.1f}σ (w≈{width_ms_wf:.3f} ms)", fontsize=9, fontweight="bold")
            else:
                ax_prof_wf.set_title(f"Raw Waterfall SNR\nPeak={peak_snr_wf:.1f}σ", fontsize=9, fontweight="bold")
    elif not _skip_standard_panels:
        ax_prof_wf.text(0.5, 0.5, 'No waterfall data\navailable', 
                       transform=ax_prof_wf.transAxes, 
                       ha='center', va='center', fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        ax_prof_wf.set_ylabel('SNR (σ)', fontsize=8, fontweight='bold')
        ax_prof_wf.grid(True, alpha=0.3)
        ax_prof_wf.set_xticks([])
        ax_prof_wf.set_title("No Raw Waterfall Data", fontsize=9, fontweight="bold")

    if not _skip_standard_panels:                     
        ax_wf = fig.add_subplot(gs_waterfall_nested[1, 0])
    
    if not _skip_standard_panels and wf_block is not None and wf_block.size > 0:
        im_wf = ax_wf.imshow(
            wf_block.T,
            origin="lower",
            cmap="mako",
            aspect="auto",
            vmin=np.nanpercentile(wf_block, 1),
            vmax=np.nanpercentile(wf_block, 99),
            extent=[slice_start_abs, slice_end_abs, freq_ds.min(), freq_ds.max()],
        )
        ax_wf.set_xlim(slice_start_abs, slice_end_abs)
        ax_wf.set_ylim(freq_ds.min(), freq_ds.max())

        n_freq_ticks = 6
        freq_tick_positions = np.linspace(freq_ds.min(), freq_ds.max(), n_freq_ticks)
        ax_wf.set_yticks(freq_tick_positions)

        n_time_ticks = 5
        time_tick_positions = np.linspace(slice_start_abs, slice_end_abs, n_time_ticks)
        ax_wf.set_xticks(time_tick_positions)
        ax_wf.set_xticklabels([f"{t:.6f}" for t in time_tick_positions])
        ax_wf.set_xlabel("Time (s)", fontsize=9)
        ax_wf.set_ylabel("Frequency (MHz)", fontsize=9)
        
        if 'peak_snr_wf' in locals() and config.SNR_SHOW_PEAK_LINES:
            ax_wf.axvline(x=time_axis_wf[peak_idx_wf], color=config.SNR_HIGHLIGHT_COLOR, 
                         linestyle='-', alpha=0.8, linewidth=2)
    elif not _skip_standard_panels:
        ax_wf.text(0.5, 0.5, 'No waterfall data available', 
                  transform=ax_wf.transAxes, 
                  ha='center', va='center', fontsize=12, 
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        ax_wf.set_xticks([])
        ax_wf.set_yticks([])
        ax_wf.set_xlabel("Time (s)", fontsize=9)
        ax_wf.set_ylabel("Frequency (MHz)", fontsize=9)

    if not _skip_standard_panels:                                         
        gs_dedisp_nested = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs_bottom_row[0, 1], height_ratios=[1, 4], hspace=0.05
        )
        ax_prof_dw = fig.add_subplot(gs_dedisp_nested[0, 0])
    
                                   
    if not _skip_standard_panels and top_boxes is not None and len(top_boxes) > 0:
        best_candidate_idx = np.argmax(top_conf)
        best_box = top_boxes[best_candidate_idx]
        center_x, center_y = (best_box[0] + best_box[2]) / 2, (best_box[1] + best_box[3]) / 2
        dm_val_consistent, _, _ = extract_candidate_dm(center_x, center_y, slice_len)
        
        x1, y1, x2, y2 = map(int, best_box)
        candidate_region = waterfall_block[:, y1:y2] if waterfall_block is not None else None
        if candidate_region is not None and candidate_region.size > 0:
            snr_profile_candidate, _, _ = compute_snr_profile(candidate_region)
            snr_val_candidate = np.max(snr_profile_candidate)
        else:
            snr_val_candidate = 0.0
    elif not _skip_standard_panels:
        dm_val_consistent = dm_val
        snr_val_candidate = 0.0
    
    if not _skip_standard_panels and dw_block is not None and dw_block.size > 0:
        snr_dw, sigma_dw, best_w_dw = compute_snr_profile(dw_block, off_regions)
        peak_snr_dw, peak_time_dw, peak_idx_dw = find_snr_peak(snr_dw)
        width_ms_dw = float(best_w_dw[int(peak_idx_dw)]) * time_reso_ds * 1000.0 if len(best_w_dw) == len(snr_dw) else None
        
        time_axis_dw = np.linspace(slice_start_abs, slice_end_abs, len(snr_dw))
        peak_time_dw_abs = float(time_axis_dw[peak_idx_dw]) if len(snr_dw) > 0 else None
        ax_prof_dw.plot(time_axis_dw, snr_dw, color="green", alpha=0.8, lw=1.5, label='Dedispersed SNR')
        
        if thresh_snr is not None and config.SNR_SHOW_PEAK_LINES:
            above_thresh_dw = snr_dw >= thresh_snr
            if np.any(above_thresh_dw):
                ax_prof_dw.plot(time_axis_dw[above_thresh_dw], snr_dw[above_thresh_dw], 
                               color=config.SNR_HIGHLIGHT_COLOR, lw=2.5, alpha=0.9)
            ax_prof_dw.axhline(y=thresh_snr, color=config.SNR_HIGHLIGHT_COLOR, 
                              linestyle='--', alpha=0.7, linewidth=1)
        
        ax_prof_dw.plot(time_axis_dw[peak_idx_dw], peak_snr_dw, 'ro', markersize=5)
        ax_prof_dw.text(time_axis_dw[peak_idx_dw], peak_snr_dw + 0.1 * (ax_prof_dw.get_ylim()[1] - ax_prof_dw.get_ylim()[0]), 
                       f'{peak_snr_dw:.1f}σ', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax_prof_dw.set_xlim(slice_start_abs, slice_end_abs)
        ax_prof_dw.set_ylabel('SNR (σ)', fontsize=8, fontweight='bold')
        ax_prof_dw.grid(True, alpha=0.3)
        ax_prof_dw.set_xticks([])
        
        if snr_val_candidate > 0:
            if peak_time_dw_abs is not None:
                if width_ms_dw is not None:
                    title_text = (
                        f"Dedispersed SNR DM={dm_val_consistent:.2f} pc cm⁻³\n"
                        f"Peak={peak_snr_dw:.1f}σ (w≈{width_ms_dw:.3f} ms) -> {peak_time_dw_abs:.6f}s (block) / {snr_val_candidate:.1f}σ (candidate)"
                    )
                else:
                    title_text = (
                        f"Dedispersed SNR DM={dm_val_consistent:.2f} pc cm⁻³\n"
                        f"Peak={peak_snr_dw:.1f}σ -> {peak_time_dw_abs:.6f}s (block) / {snr_val_candidate:.1f}σ (candidate)"
                    )
            else:
                if width_ms_dw is not None:
                    title_text = (
                        f"Dedispersed SNR DM={dm_val_consistent:.2f} pc cm⁻³\n"
                        f"Peak={peak_snr_dw:.1f}σ (w≈{width_ms_dw:.3f} ms) (block) / {snr_val_candidate:.1f}σ (candidate)"
                    )
                else:
                    title_text = (
                        f"Dedispersed SNR DM={dm_val_consistent:.2f} pc cm⁻³\n"
                        f"Peak={peak_snr_dw:.1f}σ (block) / {snr_val_candidate:.1f}σ (candidate)"
                    )
        else:
            if peak_time_dw_abs is not None:
                if width_ms_dw is not None:
                    title_text = (
                        f"Dedispersed SNR DM={dm_val_consistent:.2f} pc cm⁻³\n"
                        f"Peak={peak_snr_dw:.1f}σ (w≈{width_ms_dw:.3f} ms) -> {peak_time_dw_abs:.6f}s"
                    )
                else:
                    title_text = (
                        f"Dedispersed SNR DM={dm_val_consistent:.2f} pc cm⁻³\n"
                        f"Peak={peak_snr_dw:.1f}σ -> {peak_time_dw_abs:.6f}s"
                    )
            else:
                if width_ms_dw is not None:
                    title_text = f"Dedispersed SNR DM={dm_val_consistent:.2f} pc cm⁻³\nPeak={peak_snr_dw:.1f}σ (w≈{width_ms_dw:.3f} ms)"
                else:
                    title_text = f"Dedispersed SNR DM={dm_val_consistent:.2f} pc cm⁻³\nPeak={peak_snr_dw:.1f}σ"
        ax_prof_dw.set_title(title_text, fontsize=9, fontweight="bold")
    elif not _skip_standard_panels:
        ax_prof_dw.text(0.5, 0.5, 'No dedispersed\ndata available', 
                       transform=ax_prof_dw.transAxes, 
                       ha='center', va='center', fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        ax_prof_dw.set_ylabel('SNR (σ)', fontsize=8, fontweight='bold')
        ax_prof_dw.grid(True, alpha=0.3)
        ax_prof_dw.set_xticks([])
        ax_prof_dw.set_title("No Dedispersed Data", fontsize=9, fontweight="bold")

    if not _skip_standard_panels:                             
        ax_dw = fig.add_subplot(gs_dedisp_nested[1, 0])
    
    if not _skip_standard_panels and dw_block is not None and dw_block.size > 0:
        im_dw = ax_dw.imshow(
            dw_block.T,
            origin="lower",
            cmap="mako",
            aspect="auto",
            vmin=np.nanpercentile(dw_block, 1),
            vmax=np.nanpercentile(dw_block, 99),
            extent=[slice_start_abs, slice_end_abs, freq_ds.min(), freq_ds.max()],
        )
        ax_dw.set_xlim(slice_start_abs, slice_end_abs)
        ax_dw.set_ylim(freq_ds.min(), freq_ds.max())

        ax_dw.set_yticks(freq_tick_positions)
        ax_dw.set_yticklabels([f"{f:.0f}" for f in freq_tick_positions])
        ax_dw.set_xticks(time_tick_positions)
        ax_dw.set_xticklabels([f"{t:.6f}" for t in time_tick_positions])
        ax_dw.set_xlabel("Time (s)", fontsize=9)
        ax_dw.set_ylabel("Frequency (MHz)", fontsize=9)
        
        if 'peak_snr_dw' in locals() and config.SNR_SHOW_PEAK_LINES:
            ax_dw.axvline(x=time_axis_dw[peak_idx_dw], color=config.SNR_HIGHLIGHT_COLOR, 
                         linestyle='-', alpha=0.8, linewidth=2)
    elif not _skip_standard_panels:
        ax_dw.text(0.5, 0.5, 'No dedispersed data available', 
                  transform=ax_dw.transAxes, 
                  ha='center', va='center', fontsize=12, 
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        ax_dw.set_xticks([])
        ax_dw.set_yticks([])
        ax_dw.set_xlabel("Time (s)", fontsize=9)
        ax_dw.set_ylabel("Frequency (MHz)", fontsize=9)

    if not _skip_standard_panels:                                   
        gs_patch_nested = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs_bottom_row[0, 2], height_ratios=[1, 4], hspace=0.05
        )
        ax_patch_prof = fig.add_subplot(gs_patch_nested[0, 0])
    
    if not _skip_standard_panels and patch_img is not None and patch_img.size > 0:
        snr_patch, sigma_patch, best_w_patch = compute_snr_profile(patch_img, off_regions)
        peak_snr_patch, peak_time_patch, peak_idx_patch = find_snr_peak(snr_patch)
        width_ms_patch = float(best_w_patch[int(peak_idx_patch)]) * time_reso_ds * 1000.0 if len(best_w_patch) == len(snr_patch) else None
        
        if absolute_start_time is not None:
            patch_start_abs = absolute_start_time + (patch_start - (slice_idx * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE))
        else:
            patch_start_abs = patch_start
        
        patch_time_axis = np.linspace(patch_start_abs, patch_start_abs + len(snr_patch) * time_reso_ds, len(snr_patch))
        ax_patch_prof.plot(patch_time_axis, snr_patch, color="orange", alpha=0.8, lw=1.5, label='Candidate SNR')
        
        if thresh_snr is not None and config.SNR_SHOW_PEAK_LINES:
            above_thresh_patch = snr_patch >= thresh_snr
            if np.any(above_thresh_patch):
                ax_patch_prof.plot(patch_time_axis[above_thresh_patch], snr_patch[above_thresh_patch], 
                                color=config.SNR_HIGHLIGHT_COLOR, lw=2, alpha=0.9)
            ax_patch_prof.axhline(y=thresh_snr, color=config.SNR_HIGHLIGHT_COLOR, 
                                 linestyle='--', alpha=0.7, linewidth=1)
        
        ax_patch_prof.plot(patch_time_axis[peak_idx_patch], peak_snr_patch, 'ro', markersize=5)
        ax_patch_prof.text(patch_time_axis[peak_idx_patch], peak_snr_patch + 0.1 * (ax_patch_prof.get_ylim()[1] - ax_patch_prof.get_ylim()[0]), 
                          f'{peak_snr_patch:.1f}σ', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax_patch_prof.set_xlim(patch_time_axis[0], patch_time_axis[-1])
        ax_patch_prof.set_ylabel('SNR (σ)', fontsize=8, fontweight='bold')
        ax_patch_prof.grid(True, alpha=0.3)
        ax_patch_prof.set_xticks([])
        if width_ms_patch is not None:
            ax_patch_prof.set_title(f"Candidate Patch SNR\nPeak={peak_snr_patch:.1f}σ (w≈{width_ms_patch:.3f} ms)", fontsize=9, fontweight="bold")
        else:
            ax_patch_prof.set_title(f"Candidate Patch SNR\nPeak={peak_snr_patch:.1f}σ", fontsize=9, fontweight="bold")
    elif not _skip_standard_panels:
        ax_patch_prof.text(0.5, 0.5, 'No candidate patch\navailable', 
                          transform=ax_patch_prof.transAxes, 
                          ha='center', va='center', fontsize=10, 
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        ax_patch_prof.set_ylabel('SNR (σ)', fontsize=8, fontweight='bold')
        ax_patch_prof.grid(True, alpha=0.3)
        ax_patch_prof.set_xticks([])
        ax_patch_prof.set_title("No Candidate Patch", fontsize=9, fontweight="bold")

    if not _skip_standard_panels:                       
        ax_patch = fig.add_subplot(gs_patch_nested[1, 0])
    
    if not _skip_standard_panels and patch_img is not None and patch_img.size > 0:
        ax_patch.imshow(
            patch_img.T,
            origin="lower",
            aspect="auto",
            cmap="mako",
            vmin=np.nanpercentile(patch_img, 1),
            vmax=np.nanpercentile(patch_img, 99),
            extent=[patch_time_axis[0], patch_time_axis[-1], freq_ds.min(), freq_ds.max()],
        )
        ax_patch.set_xlim(patch_time_axis[0], patch_time_axis[-1])
        ax_patch.set_ylim(freq_ds.min(), freq_ds.max())

        n_patch_time_ticks = 5
        patch_tick_positions = np.linspace(patch_time_axis[0], patch_time_axis[-1], n_patch_time_ticks)
        ax_patch.set_xticks(patch_tick_positions)
        ax_patch.set_xticklabels([f"{t:.6f}" for t in patch_tick_positions])

        ax_patch.set_yticks(freq_tick_positions)
        ax_patch.set_yticklabels([f"{f:.0f}" for f in freq_tick_positions])
        ax_patch.set_xlabel("Time (s)", fontsize=9)
        ax_patch.set_ylabel("Frequency (MHz)", fontsize=9)
        
        if 'peak_snr_patch' in locals() and config.SNR_SHOW_PEAK_LINES:
            ax_patch.axvline(x=patch_time_axis[peak_idx_patch], color=config.SNR_HIGHLIGHT_COLOR, 
                           linestyle='-', alpha=0.8, linewidth=2)
    elif not _skip_standard_panels:
        ax_patch.text(0.5, 0.5, 'No candidate patch available', 
                     transform=ax_patch.transAxes, 
                     ha='center', va='center', fontsize=12, 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        ax_patch.set_xticks([])
        ax_patch.set_yticks([])
        ax_patch.set_xlabel("Time (s)", fontsize=9)
        ax_patch.set_ylabel("Frequency (MHz)", fontsize=9)

                    
    idx_start_ds = int(round(slice_start_abs / (config.TIME_RESO * config.DOWN_TIME_RATE)))
    idx_end_ds = idx_start_ds + real_samples - 1
    start_center = slice_start_abs
    end_center = slice_end_abs
    
    if chunk_idx is not None:
        title = (
            f"Composite: {fits_stem} - {band_name_with_freq} - Chunk {chunk_idx:03d} - Slice {slice_idx:03d} | "
            f"start={start_center:.6f}s end={end_center:.6f}s Δt={(config.TIME_RESO * config.DOWN_TIME_RATE):.9f}s "
            f"| [idx {idx_start_ds}→{idx_end_ds}]"
        )
    else:
        title = (
            f"Composite: {fits_stem} - {band_name_with_freq} - Slice {slice_idx:03d} | "
            f"start={start_center:.6f}s end={end_center:.6f}s Δt={(config.TIME_RESO * config.DOWN_TIME_RATE):.9f}s "
            f"| [idx {idx_start_ds}→{idx_end_ds}]"
        )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)
    
                              
    try:
        dt_ds = config.TIME_RESO * config.DOWN_TIME_RATE
        global_start_sample = int(round(slice_start_abs / dt_ds))
        global_end_sample = global_start_sample + real_samples - 1

        info_lines = [
            f"Samples (decimated): {global_start_sample} → {global_end_sample} (N={real_samples})",
            f"Δt (effective): {dt_ds:.9f} s",
            f"Time span (centers): {start_center:.6f}s → {end_center:.6f}s (Δ={(real_samples - 1) * dt_ds:.6f}s)",
        ]
        fig.text(
            0.01,
            0.01,
            "\n".join(info_lines),
            ha="left",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
    except Exception:
        pass

    return fig


def save_composite_plot(
    waterfall_block: np.ndarray,
    dedispersed_block: np.ndarray,
    img_rgb: np.ndarray,
    patch_img: np.ndarray,
    patch_start: float,
    dm_val: float,
    top_conf: Iterable,
    top_boxes: Iterable | None,
    class_probs: Iterable | None,
    out_path: Path,
    slice_idx: int,
    time_slice: int,
    band_name: str,
    band_suffix: str,
    fits_stem: str,
    slice_len: int,
    normalize: bool = False,
    off_regions: Optional[List[Tuple[int, int]]] = None,
    thresh_snr: Optional[float] = None,
    band_idx: int = 0,
    absolute_start_time: Optional[float] = None, 
    chunk_idx: Optional[int] = None,  
    slice_samples: Optional[int] = None,  
    candidate_times_abs: Optional[Iterable[float]] = None,
    generate_individual_plots: bool = True,
    individual_plots_dir: str = "individual_plots",
    dedisp_block_linear: Optional[np.ndarray] = None,
    dedisp_block_circular: Optional[np.ndarray] = None,
    class_probs_linear: Optional[Iterable[float]] = None,  # NEW: Linear classification probs
) -> None:
    """Save composite plot by creating the figure and saving it to file.
    
    Args:
        ... (existing parameters) ...
        generate_individual_plots: If True, also generate individual plot components
        individual_plots_dir: Directory name for individual plots (relative to composite plot location)
        dedisp_block_linear: Dedispersed waterfall in Linear polarization (for HF pipeline)
        dedisp_block_circular: Dedispersed waterfall in Circular polarization (for HF pipeline)
    """
    
                                 
    fig = create_composite_plot(
        waterfall_block=waterfall_block,
        dedispersed_block=dedispersed_block,
        img_rgb=img_rgb,
        patch_img=patch_img,
        patch_start=patch_start,
        dm_val=dm_val,
        top_conf=top_conf,
        top_boxes=top_boxes,
        class_probs=class_probs,
        slice_idx=slice_idx,
        time_slice=time_slice,
        band_name=band_name,
        band_suffix=band_suffix,
        fits_stem=fits_stem,
        slice_len=slice_len,
        normalize=normalize,
        off_regions=off_regions,
        thresh_snr=thresh_snr,
        band_idx=band_idx,
        absolute_start_time=absolute_start_time,
        chunk_idx=chunk_idx,
        slice_samples=slice_samples,
        candidate_times_abs=candidate_times_abs,
        dedisp_block_linear=dedisp_block_linear,
        dedisp_block_circular=dedisp_block_circular,
        class_probs_linear=class_probs_linear,  # NEW: Pass Linear probs
    )
    
                                    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
                               
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    
                                            
    if generate_individual_plots:
        try:
            from .plot_individual_components import generate_individual_plots
            
            generate_individual_plots(
                waterfall_block=waterfall_block,
                dedispersed_block=dedispersed_block,
                img_rgb=img_rgb,
                patch_img=patch_img,
                patch_start=patch_start,
                dm_val=dm_val,
                top_conf=top_conf,
                top_boxes=top_boxes,
                class_probs=class_probs,
                base_out_path=out_path,
                slice_idx=slice_idx,
                time_slice=time_slice,
                band_name=band_name,
                band_suffix=band_suffix,
                fits_stem=fits_stem,
                slice_len=slice_len,
                normalize=normalize,
                off_regions=off_regions,
                thresh_snr=thresh_snr,
                band_idx=band_idx,
                absolute_start_time=absolute_start_time,
                chunk_idx=chunk_idx,
                slice_samples=slice_samples,
                candidate_times_abs=candidate_times_abs,
                output_dir=individual_plots_dir,
                class_probs_linear=class_probs_linear,  # NEW: Pass Linear probs
            )
        except Exception as e:
            logger.warning(f"Could not generate individual plots: {e}")
                                                                                  
