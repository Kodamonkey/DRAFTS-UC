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
from ..core.mjd_utils import calculate_candidate_mjd
from .visualization_ranges import get_dynamic_dm_range_for_candidate

              
logger = logging.getLogger(__name__)


def _calculate_label_positions(
    ax,
    candidate_boxes: list,
    initial_offset: float = 10.0,
    min_spacing: float = 5.0,
    title_bottom: float = None
) -> list:
    """
    Calculate label positions avoiding collisions with title and between labels.
    
    Args:
        ax: Matplotlib axis
        candidate_boxes: List of (x1, y1, x2, y2) tuples for each candidate
        initial_offset: Initial vertical offset from box top
        min_spacing: Minimum spacing between labels
        title_bottom: Y coordinate of title bottom (in data coordinates)
    
    Returns:
        List of (x, y) tuples for label positions
    """
    if not candidate_boxes:
        return []
    
    # Get axis limits to work in data coordinates
    ylim = ax.get_ylim()
    
    # Estimate label height in data coordinates
    # A typical label has ~6-8 lines, estimate ~100-150 pixels total height
    # Convert to data coordinates
    fig = ax.figure
    bbox_ax = ax.get_window_extent()
    height_pixels = bbox_ax.height
    data_height = ylim[1] - ylim[0]
    pixels_per_data_unit = height_pixels / data_height if data_height > 0 else 1.0
    
    # Estimate label height in data coordinates (assuming ~120-150 pixels for full label)
    estimated_label_height = 130.0 / pixels_per_data_unit
    
    label_positions = []
    
    for idx, box in enumerate(candidate_boxes):
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2.0
        
        # Start position: above the box
        y_start = float(y2) + initial_offset
        
        # If title position is provided, ensure label is below title
        if title_bottom is not None:
            y_start = max(y_start, float(title_bottom) - min_spacing)
        
        # Check collisions with existing labels
        y_pos = y_start
        max_iterations = 100
        iteration = 0
        
        while iteration < max_iterations:
            # Check if this position collides with any existing label
            collision = False
            label_top = y_pos + estimated_label_height / 2
            label_bottom = y_pos - estimated_label_height / 2
            
            for existing_x, existing_y in label_positions:
                existing_top = existing_y + estimated_label_height / 2
                existing_bottom = existing_y - estimated_label_height / 2
                
                # Check vertical overlap (horizontal overlap is OK, we stack vertically)
                if not (label_bottom > existing_top or label_top < existing_bottom):
                    collision = True
                    break
            
            if not collision:
                break
            
            # Move down
            y_pos += estimated_label_height + min_spacing
            iteration += 1
        
        # Ensure label doesn't go below plot area (with some margin)
        if y_pos > ylim[1]:
            y_pos = float(ylim[1]) - estimated_label_height / 2 - min_spacing
        
        label_positions.append((center_x, y_pos))
    
    return label_positions


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
    snr_waterfall_linear: Optional[Iterable[float | None]] = None,  # NEW: SNR from Linear waterfall
    snr_patch_linear: Optional[Iterable[float | None]] = None,  # NEW: SNR from dedispersed Linear patch
    snr_waterfall_intensity: Optional[Iterable[float | None]] = None,  # NEW: SNR from Intensity waterfall
    snr_patch_intensity: Optional[Iterable[float | None]] = None,  # NEW: SNR from dedispersed Intensity patch
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
    
    # Convert iterables to lists to ensure we can index them multiple times
    if snr_waterfall_linear is not None:
        if not isinstance(snr_waterfall_linear, (list, tuple)):
            snr_waterfall_linear = list(snr_waterfall_linear)
        logger.info("create_composite_plot: Received snr_waterfall_linear with %d values: %s", 
                   len(snr_waterfall_linear), 
                   [f"{v:.2f}" if v is not None else "None" for v in snr_waterfall_linear[:5]])
    else:
        logger.warning("create_composite_plot: snr_waterfall_linear is None")
    if snr_patch_linear is not None:
        if not isinstance(snr_patch_linear, (list, tuple)):
            snr_patch_linear = list(snr_patch_linear)
        logger.info("create_composite_plot: Received snr_patch_linear with %d values: %s", 
                   len(snr_patch_linear),
                   [f"{v:.2f}" if v is not None else "None" for v in snr_patch_linear[:5]])
    else:
        logger.warning("create_composite_plot: snr_patch_linear is None")
    if snr_waterfall_intensity is not None:
        if not isinstance(snr_waterfall_intensity, (list, tuple)):
            snr_waterfall_intensity = list(snr_waterfall_intensity)
        logger.info("create_composite_plot: Received snr_waterfall_intensity with %d values: %s", 
                   len(snr_waterfall_intensity), 
                   [f"{v:.2f}" if v is not None else "None" for v in snr_waterfall_intensity[:5]])
    else:
        logger.warning("create_composite_plot: snr_waterfall_intensity is None")
    if snr_patch_intensity is not None:
        if not isinstance(snr_patch_intensity, (list, tuple)):
            snr_patch_intensity = list(snr_patch_intensity)
        logger.info("create_composite_plot: Received snr_patch_intensity with %d values: %s", 
                   len(snr_patch_intensity),
                   [f"{v:.2f}" if v is not None else "None" for v in snr_patch_intensity[:5]])
    else:
        logger.warning("create_composite_plot: snr_patch_intensity is None")
    
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
    
    # Use actual size of wf_block if available, otherwise use slice_samples or slice_len
    # This handles residual slices correctly (last slice of chunk with different size)
    if wf_block is not None and wf_block.size > 0 and wf_block.ndim >= 1:
        real_samples = wf_block.shape[0]
    else:
        real_samples = slice_samples if slice_samples is not None else slice_len
    
    # IMPORTANT: slice_end_abs will be recalculated later based on actual SNR profile size
    # This initial calculation is just for setup, the actual time axis will use len(snr_wf)
    slice_end_abs = slice_start_abs + real_samples * config.TIME_RESO * config.DOWN_TIME_RATE

                                
    fig = plt.figure(figsize=(14, 12))
    gs_main = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.3, figure=fig)
    
                            
    ax_det = fig.add_subplot(gs_main[0, 0])
    ax_det.imshow(img_rgb, origin="lower", aspect="auto")
    
    # Set title first so we can calculate its position later
    title_det = "Detection Results"
    ax_det.set_title(title_det, fontsize=10, fontweight="bold")
    ax_det.set_xlabel("Time (s)", fontsize=9)
    ax_det.set_ylabel("Dispersion Measure (pc cm⁻³)", fontsize=9)

                                              
    n_time_ticks_det = 8
    time_positions_det = np.linspace(0, img_rgb.shape[1] - 1, n_time_ticks_det)
    n_px = img_rgb.shape[1]
    denom = float(max(n_px - 1, 1))
    time_values_det = slice_start_abs + (time_positions_det / denom) * (slice_end_abs - slice_start_abs)
    ax_det.set_xticks(time_positions_det)
    ax_det.set_xticklabels([f"{t:.6f}" for t in time_values_det], rotation=45)
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

    # Calculate label positions first to avoid collisions
    label_positions = []
    if top_boxes is not None:
        # Get title position after axis is set up
        # We need to render the figure to get accurate title position
        fig.canvas.draw()
        title_bbox = ax_det.title.get_window_extent(fig.canvas.get_renderer())
        bbox_ax = ax_det.get_window_extent()
        ylim = ax_det.get_ylim()
        
        # Convert title bottom from figure coordinates to data coordinates
        # Title is at the top of the axis, so we calculate its bottom edge
        title_bottom_fig = title_bbox.y0
        ax_top_fig = bbox_ax.y1
        ax_bottom_fig = bbox_ax.y0
        title_height_fig = title_bbox.height
        
        # Calculate title bottom in data coordinates
        # Title is above the axis, so we need to find where it ends
        # If title extends into axis area, calculate where it ends
        title_bottom_data = None
        if title_bottom_fig <= ax_top_fig:
            # Title overlaps with or is above axis
            # Calculate how much of the title is inside the axis
            overlap_fig = ax_top_fig - title_bottom_fig
            data_range = ylim[1] - ylim[0]
            fig_range = ax_top_fig - ax_bottom_fig
            if fig_range > 0:
                title_bottom_data = ylim[1] - (overlap_fig / fig_range) * data_range
            else:
                title_bottom_data = ylim[1]
        else:
            # Title is completely above axis, use top of axis as reference
            title_bottom_data = ylim[1]
        
        # Calculate label positions for all candidates
        label_positions = _calculate_label_positions(
            ax_det,
            top_boxes,
            initial_offset=10.0,
            min_spacing=5.0,
            title_bottom=title_bottom_data
        )
    
    # Now create labels and boxes
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
            
            # Calculate MJD for the label
            mjd_data = calculate_candidate_mjd(
                t_sec=detection_time,
                compute_bary=True,
                dm=dm_val_cand,
            )
            mjd_utc = mjd_data.get('mjd_utc', 0.0)
            mjd_bary_utc_inf = mjd_data.get('mjd_bary_utc_inf')
            
            # CRITICAL: Use SNR from detection (PRESTO-style) - NOT recalculated from dedispersed waterfall
            # This ensures consistency with PRESTO methodology and between DM-time label and waterfall title
            snr_I_wf = None
            snr_I_patch = None
            snr_L_wf = None
            snr_L_patch = None
            
            # Get Intensity SNR from detection (calculated during Phase 1 in HF pipeline or during detection in classic)
            if snr_waterfall_intensity is not None:
                if idx < len(snr_waterfall_intensity):
                    snr_I_wf_val = snr_waterfall_intensity[idx]
                    if snr_I_wf_val is not None:
                        snr_I_wf = float(snr_I_wf_val)
                        logger.info("Candidate idx=%d: SNR_I from detection=%.2f (PRESTO-style)", idx, snr_I_wf)
                    else:
                        logger.warning("snr_waterfall_intensity[%d] is None", idx)
                else:
                    logger.warning("idx=%d >= len(snr_waterfall_intensity)=%d", idx, len(snr_waterfall_intensity))
            else:
                logger.warning("snr_waterfall_intensity is None for candidate idx=%d", idx)
            
            if snr_patch_intensity is not None:
                if idx < len(snr_patch_intensity):
                    snr_I_patch_val = snr_patch_intensity[idx]
                    if snr_I_patch_val is not None:
                        snr_I_patch = float(snr_I_patch_val)
                else:
                    logger.warning("idx=%d >= len(snr_patch_intensity)=%d", idx, len(snr_patch_intensity))
            else:
                logger.warning("snr_patch_intensity is None for candidate idx=%d", idx)
            
            # Dual-polarization classification color logic (HF pipeline)
            if class_probs is not None and idx < len(class_probs):
                class_prob_I = class_probs[idx]
                is_burst_I = class_prob_I >= config.CLASS_PROB
                
                # Check if we have Linear classification (HF pipeline)
                # Show Linear prob if available, even if it's 0.0 (means classification was performed)
                is_burst_L = False
                class_prob_L = 0.0
                has_linear_classification = (class_probs_linear is not None and 
                                            idx < len(class_probs_linear))
                
                logger.info("Candidate idx=%d: has_linear_classification=%s, snr_waterfall_linear is not None=%s", 
                           idx, has_linear_classification, snr_waterfall_linear is not None)
                
                if has_linear_classification:
                    class_prob_L = class_probs_linear[idx]
                    # Use CLASS_PROB_LINEAR threshold for Linear classification
                    class_prob_linear_thresh = getattr(config, 'CLASS_PROB_LINEAR', config.CLASS_PROB)
                    is_burst_L = class_prob_L >= class_prob_linear_thresh
                    
                    # HF Mode: Verde solo si AMBAS clasifican BURST
                    color = "lime" if (is_burst_I and is_burst_L) else "orange"
                    
                    if is_burst_I and is_burst_L:
                        burst_status = "BURST (I+L)"
                    elif is_burst_I and not is_burst_L:
                        burst_status = "I:BURST L:NO"
                    else:
                        burst_status = "NO BURST"
                    
                    # Get Linear SNR (only for HF pipeline)
                    if snr_waterfall_linear is not None:
                        if idx < len(snr_waterfall_linear):
                            snr_L_wf_val = snr_waterfall_linear[idx]
                            # Handle None values in the list
                            if snr_L_wf_val is not None:
                                snr_L_wf = float(snr_L_wf_val)
                                logger.info("Candidate idx=%d: snr_L_wf=%.2f", idx, snr_L_wf)
                            else:
                                logger.warning("snr_waterfall_linear[%d] is None", idx)
                        else:
                            logger.warning("idx=%d >= len(snr_waterfall_linear)=%d", idx, len(snr_waterfall_linear))
                    else:
                        logger.warning("snr_waterfall_linear is None for candidate idx=%d", idx)
                    if snr_patch_linear is not None:
                        if idx < len(snr_patch_linear):
                            snr_L_patch_val = snr_patch_linear[idx]
                            # Handle None values in the list
                            if snr_L_patch_val is not None:
                                snr_L_patch = float(snr_L_patch_val)
                        else:
                            logger.warning("idx=%d >= len(snr_patch_linear)=%d", idx, len(snr_patch_linear))
                    else:
                        logger.warning("snr_patch_linear is None for candidate idx=%d", idx)
                    
                    # Get thresholds for display
                    thresh_snr_I = config.SNR_THRESH
                    thresh_snr_L = getattr(config, 'SNR_THRESH_LINEAR', config.SNR_THRESH)
                    thresh_class_I = config.CLASS_PROB
                    thresh_class_L = getattr(config, 'CLASS_PROB_LINEAR', config.CLASS_PROB)
                    
                    mjd_str = f"MJD_topo: {mjd_utc:.8f}"
                    if mjd_bary_utc_inf is not None:
                        mjd_str += f"\nMJD_bary_inf: {mjd_bary_utc_inf:.8f}"
                    
                    # Build label with SNR and classification info (clean format)
                    # Show actual SNR values (not thresholds) for both I and L
                    if snr_I_wf is not None:
                        snr_I_str = f"{snr_I_wf:.1f}σ (≥{thresh_snr_I:.1f})"
                    else:
                        snr_I_str = f"N/A (≥{thresh_snr_I:.1f})"
                    
                    label = (
                        f"#{idx+1}\n"
                        f"DM: {dm_val_cand:.1f} | Time: {detection_time:.3f}s\n"
                        f"{mjd_str}\n"
                        f"SNR_I: {snr_I_str} | Class_I: {class_prob_I:.2f} (≥{thresh_class_I:.2f})"
                    )
                    if snr_L_wf is not None:
                        label += f"\nSNR_L: {snr_L_wf:.1f}σ (≥{thresh_snr_L:.1f}) | Class_L: {class_prob_L:.2f} (≥{thresh_class_L:.2f})"
                    else:
                        label += f"\nSNR_L: N/A (≥{thresh_snr_L:.1f}) | Class_L: {class_prob_L:.2f} (≥{thresh_class_L:.2f})"
                    label += f"\n{burst_status}"
                else:
                    # Standard mode: Solo Intensity
                    color = "lime" if is_burst_I else "orange"
                    burst_status = "BURST" if is_burst_I else "NO BURST"
                    
                    # Get thresholds for display
                    thresh_snr_I = config.SNR_THRESH
                    thresh_class_I = config.CLASS_PROB
                    
                    mjd_str = f"MJD_topo: {mjd_utc:.8f}"
                    if mjd_bary_utc_inf is not None:
                        mjd_str += f"\nMJD_bary_inf: {mjd_bary_utc_inf:.8f}"
                    
                    # Build label with SNR (use snr_I_wf if available)
                    if snr_I_wf is not None:
                        snr_I_str = f"{snr_I_wf:.1f}σ (≥{thresh_snr_I:.1f})"
                    else:
                        snr_I_str = f"N/A (≥{thresh_snr_I:.1f})"
                    
                    label = (
                        f"#{idx+1}\n"
                        f"DM: {dm_val_cand:.1f} | Time: {detection_time:.3f}s\n"
                        f"{mjd_str}\n"
                        f"SNR_I: {snr_I_str} | Class_I: {class_prob_I:.2f} (≥{thresh_class_I:.2f})\n"
                        f"{burst_status}"
                    )
            else:
                color = "lime"
                mjd_str = f"MJD_topo: {mjd_utc:.8f}"
                if mjd_bary_utc_inf is not None:
                    mjd_str += f"\nMJD_bary_inf: {mjd_bary_utc_inf:.8f}"
                label = f"#{idx+1}\nDM: {dm_val_cand:.1f}\nTime: {detection_time:.3f}s\n{mjd_str}\nDet: {conf:.2f}"
            
                                      
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, 
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax_det.add_patch(rect)
            
            # Use calculated label position (avoids collisions with title and other labels)
            if idx < len(label_positions):
                label_x, label_y = label_positions[idx]
            else:
                # Fallback: use position above box
                label_x = center_x
                label_y = y2 + 10
            
            ax_det.annotate(
                label,
                xy=(center_x, center_y),
                xytext=(label_x, label_y),
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                fontsize=7,
                ha="center",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                zorder=10,  # High zorder so labels appear above title
            )
    
                               
    # Title was already set above, no need to set it again


    # =============================================================================
    # BOTTOM PANELS LAYOUT
    # =============================================================================
    # Multi-pol mode (HF pipeline): Show 3 dedispersed waterfalls (I, L, V)
    # Standard mode (classic pipeline): Show only dispersed and dedispersed waterfalls (NO patch)
    # =============================================================================
    
    if multi_pol_mode:
        # HF pipeline: 3 panels for I, L, V
        gs_bottom_row = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs_main[1, 0], width_ratios=[1, 1, 1], wspace=0.3
        )
    else:
        # Classic pipeline: 2 panels for raw and dedispersed waterfalls only
        gs_bottom_row = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_main[1, 0], width_ratios=[1, 1], wspace=0.3
        )
    
    if multi_pol_mode:
        # Use multi-polarization panel layout
        from .plot_multi_pol_panels import create_multi_pol_panels
        
        # Get candidate time and SNR from detection (PRESTO-style) for marking position in waterfalls
        candidate_time_abs = None
        candidate_time_intensity = None
        candidate_snr_intensity_wf = None
        candidate_snr_linear_wf = None
        if top_boxes is not None and len(top_boxes) > 0:
            # Use time and SNR from best candidate (highest confidence)
            if top_conf is not None and len(top_conf) > 0:
                best_idx = int(np.argmax(top_conf))
                # Get candidate time for marking position
                if candidate_times_abs is not None and best_idx < len(candidate_times_abs):
                    candidate_time_abs = candidate_times_abs[best_idx]
                    candidate_time_intensity = candidate_times_abs[best_idx]
                # Get SNR from detection (PRESTO-style) for waterfall titles
                if snr_waterfall_intensity is not None and best_idx < len(snr_waterfall_intensity):
                    candidate_snr_intensity_wf = snr_waterfall_intensity[best_idx]
                if snr_waterfall_linear is not None and best_idx < len(snr_waterfall_linear):
                    candidate_snr_linear_wf = snr_waterfall_linear[best_idx]
            # If no best candidate, use first candidate
            elif candidate_times_abs is not None and len(candidate_times_abs) > 0:
                candidate_time_abs = candidate_times_abs[0]
                candidate_time_intensity = candidate_times_abs[0]
                if snr_waterfall_intensity is not None and len(snr_waterfall_intensity) > 0:
                    candidate_snr_intensity_wf = snr_waterfall_intensity[0]
                if snr_waterfall_linear is not None and len(snr_waterfall_linear) > 0:
                    candidate_snr_linear_wf = snr_waterfall_linear[0]
        
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
            candidate_time_abs=candidate_time_abs,  # Pass candidate time for Linear waterfall
            candidate_time_intensity=candidate_time_intensity,  # Pass candidate time for Intensity waterfall
            candidate_snr_intensity_wf=candidate_snr_intensity_wf,  # Pass SNR from detection (PRESTO-style) for Intensity
            candidate_snr_linear_wf=candidate_snr_linear_wf,  # Pass SNR from detection (PRESTO-style) for Linear
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
    
    # Calculate dispersion correction BEFORE creating SNR profile
    # This will be used for both the SNR profile AND the waterfall
    raw_waterfall_start = slice_start_abs
    raw_waterfall_end = slice_end_abs
    delta_t_max_correction = 0.0
    
    # Pre-calculate DM correction if we have candidates
    # Also get SNR from best candidate for waterfall title (consistent with HF pipeline)
    candidate_snr_intensity_wf = None
    if top_boxes is not None and len(top_boxes) > 0 and top_conf is not None and len(top_conf) > 0:
        best_idx = int(np.argmax(top_conf))
        best_box = top_boxes[best_idx]
        x1, y1, x2, y2 = map(int, best_box)
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        effective_len = slice_samples if slice_samples is not None else slice_len
        dm_best, _, _ = extract_candidate_dm(center_x, center_y, effective_len)
        
        # Get SNR from best candidate for waterfall title (consistent with HF pipeline)
        if snr_waterfall_intensity is not None and best_idx < len(snr_waterfall_intensity):
            candidate_snr_intensity_wf = snr_waterfall_intensity[best_idx]
        elif snr_waterfall_intensity is not None and len(snr_waterfall_intensity) > 0:
            candidate_snr_intensity_wf = snr_waterfall_intensity[0]
        
        if dm_best > 0:
            K_DM = 4.148808e3
            freq_min, freq_max = get_band_frequency_range(band_idx)
            delta_t_max = K_DM * dm_best * (1.0/(freq_min**2) - 1.0/(freq_max**2))
            margin = 0.1 * delta_t_max
            delta_t_max_correction = delta_t_max + margin
            raw_waterfall_start = slice_start_abs - delta_t_max_correction
            raw_waterfall_end = slice_end_abs - delta_t_max_correction
            
            logger.debug(f"[RAW WATERFALL CORRECTION] DM={dm_best:.1f}, Δt_max={delta_t_max:.3f}s")
    
    if not _skip_standard_panels and wf_block is not None and wf_block.size > 0:
        snr_wf, sigma_wf, best_w_wf = compute_snr_profile(wf_block, off_regions)
        peak_snr_wf, peak_time_wf, peak_idx_wf = find_snr_peak(snr_wf)
        
        # Use SNR from detection (PRESTO-style) if available, otherwise use global peak
        # This ensures consistency with PRESTO methodology
        display_snr_wf = candidate_snr_intensity_wf if candidate_snr_intensity_wf is not None else peak_snr_wf
        
        # Use corrected time axis for raw waterfall SNR profile
        snr_samples = len(snr_wf)
        raw_waterfall_end_snr = raw_waterfall_start + snr_samples * config.TIME_RESO * config.DOWN_TIME_RATE
        time_axis_wf = np.linspace(raw_waterfall_start, raw_waterfall_end_snr, snr_samples)
        
        # Validate peak_idx_wf is within bounds
        if peak_idx_wf < len(time_axis_wf):
            peak_time_wf_abs = float(time_axis_wf[peak_idx_wf])
        else:
            peak_time_wf_abs = raw_waterfall_start + peak_idx_wf * config.TIME_RESO * config.DOWN_TIME_RATE
        
        ax_prof_wf.plot(time_axis_wf, snr_wf, color="royalblue", alpha=0.8, lw=1.5, label='SNR Profile')
        
        if thresh_snr is not None and config.SNR_SHOW_PEAK_LINES:
            above_thresh_wf = snr_wf >= thresh_snr
            if np.any(above_thresh_wf):
                ax_prof_wf.plot(time_axis_wf[above_thresh_wf], snr_wf[above_thresh_wf], 
                              color=config.SNR_HIGHLIGHT_COLOR, lw=2, alpha=0.9)
            ax_prof_wf.axhline(y=thresh_snr, color=config.SNR_HIGHLIGHT_COLOR, 
                             linestyle='--', alpha=0.7, linewidth=1)
        
        # CRITICAL: Use SNR from detection (PRESTO-style) for BOTH the dot position and the text
        # This ensures consistency with DM-time labels - they MUST be the same value
        display_snr_wf = candidate_snr_intensity_wf if candidate_snr_intensity_wf is not None else peak_snr_wf
        
        # Mark position: use candidate time if available, otherwise use peak time
        # But ALWAYS use display_snr_wf (SNR from detection) for the Y position
        if candidate_snr_intensity_wf is not None and top_boxes is not None and len(top_boxes) > 0:
            # Find candidate time for marking position
            best_idx = int(np.argmax(top_conf)) if top_conf is not None and len(top_conf) > 0 else 0
            if candidate_times_abs is not None and best_idx < len(candidate_times_abs):
                candidate_time_wf = candidate_times_abs[best_idx]
                # Find closest time index in time_axis_wf
                candidate_idx_wf = np.argmin(np.abs(time_axis_wf - candidate_time_wf))
                if 0 <= candidate_idx_wf < len(time_axis_wf):
                    mark_time_wf = float(time_axis_wf[candidate_idx_wf])
                else:
                    mark_time_wf = float(time_axis_wf[peak_idx_wf])
            else:
                mark_time_wf = float(time_axis_wf[peak_idx_wf])
        else:
            mark_time_wf = float(time_axis_wf[peak_idx_wf])
        
        # Mark position with red dot using display_snr_wf (SNR from detection)
        ax_prof_wf.plot(mark_time_wf, display_snr_wf, 'ro', markersize=5)
        
        # Position SNR text to the right of the peak to avoid collision with title
        # Calculate text position: slightly to the right and above the peak
        y_range = ax_prof_wf.get_ylim()[1] - ax_prof_wf.get_ylim()[0]
        x_range = ax_prof_wf.get_xlim()[1] - ax_prof_wf.get_xlim()[0]
        text_x = mark_time_wf + 0.02 * x_range  # 2% to the right
        text_y = display_snr_wf + 0.15 * y_range  # 15% above peak (increased from 10% to avoid title)
        # Use display_snr_wf (SNR from detection, PRESTO-style) - MUST match DM-time label
        ax_prof_wf.text(text_x, text_y, 
                       f'{display_snr_wf:.1f}σ', ha='left', va='bottom', fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='red', linewidth=0.5))
        
        ax_prof_wf.set_xlim(time_axis_wf[0], time_axis_wf[-1])
        ax_prof_wf.set_ylabel('SNR (σ)', fontsize=8, fontweight='bold')
        ax_prof_wf.grid(True, alpha=0.3)
        ax_prof_wf.set_xticks([])
        
        # Simplified title: use candidate-specific SNR and time if available (consistent with HF pipeline)
        if candidate_snr_intensity_wf is not None and top_boxes is not None and len(top_boxes) > 0:
            best_idx = int(np.argmax(top_conf)) if top_conf is not None and len(top_conf) > 0 else 0
            if candidate_times_abs is not None and best_idx < len(candidate_times_abs):
                candidate_time_wf_title = candidate_times_abs[best_idx]
                ax_prof_wf.set_title(f"Raw Waterfall\nTime: {candidate_time_wf_title:.6f}s | SNR: {display_snr_wf:.1f}σ", fontsize=9, fontweight="bold", pad=12)
            else:
                ax_prof_wf.set_title(f"Raw Waterfall\nPeak SNR: {display_snr_wf:.1f}σ", fontsize=9, fontweight="bold", pad=12)
        else:
            ax_prof_wf.set_title(f"Raw Waterfall\nPeak SNR: {display_snr_wf:.1f}σ", fontsize=9, fontweight="bold", pad=12)
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
        # Use the already calculated raw_waterfall_start/end from above
        # (calculated before the SNR profile to ensure consistency)
        
        im_wf = ax_wf.imshow(
            wf_block.T,
            origin="lower",
            cmap="mako",
            aspect="auto",
            vmin=np.nanpercentile(wf_block, 1),
            vmax=np.nanpercentile(wf_block, 99),
            extent=[raw_waterfall_start, raw_waterfall_end, freq_ds.min(), freq_ds.max()],
        )
        ax_wf.set_xlim(raw_waterfall_start, raw_waterfall_end)
        ax_wf.set_ylim(freq_ds.min(), freq_ds.max())

        n_freq_ticks = 6
        freq_tick_positions = np.linspace(freq_ds.min(), freq_ds.max(), n_freq_ticks)
        ax_wf.set_yticks(freq_tick_positions)

        n_time_ticks = 5
        time_tick_positions = np.linspace(raw_waterfall_start, raw_waterfall_end, n_time_ticks)
        ax_wf.set_xticks(time_tick_positions)
        ax_wf.set_xticklabels([f"{t:.6f}" for t in time_tick_positions], rotation=45)
        ax_wf.set_xlabel("Time (s)", fontsize=9)
        ax_wf.set_ylabel("Frequency (MHz)", fontsize=9)
        
        if 'peak_snr_wf' in locals() and config.SNR_SHOW_PEAK_LINES:
            # Use peak time from corrected time axis (already calculated above)
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
        
        # Use SNR from detection (PRESTO-style) if available, otherwise use global peak
        # This ensures consistency with PRESTO methodology
        display_snr_dw = candidate_snr_intensity_wf if candidate_snr_intensity_wf is not None else peak_snr_dw
        
        # Use len(snr_dw) which matches the actual processed size
        time_axis_dw = np.linspace(slice_start_abs, slice_end_abs, len(snr_dw))
        # Validate peak_idx_dw is within bounds
        if len(snr_dw) > 0 and peak_idx_dw < len(time_axis_dw):
            peak_time_dw_abs = float(time_axis_dw[peak_idx_dw])
        else:
            # Fallback: calculate directly from peak index
            peak_time_dw_abs = slice_start_abs + peak_idx_dw * config.TIME_RESO * config.DOWN_TIME_RATE if len(snr_dw) > 0 else None
        ax_prof_dw.plot(time_axis_dw, snr_dw, color="green", alpha=0.8, lw=1.5, label='Dedispersed SNR')
        
        if thresh_snr is not None and config.SNR_SHOW_PEAK_LINES:
            above_thresh_dw = snr_dw >= thresh_snr
            if np.any(above_thresh_dw):
                ax_prof_dw.plot(time_axis_dw[above_thresh_dw], snr_dw[above_thresh_dw], 
                               color=config.SNR_HIGHLIGHT_COLOR, lw=2.5, alpha=0.9)
            ax_prof_dw.axhline(y=thresh_snr, color=config.SNR_HIGHLIGHT_COLOR, 
                             linestyle='--', alpha=0.7, linewidth=1)
        
        # CRITICAL: Use SNR from detection (PRESTO-style) for BOTH the dot position and the text
        # This ensures consistency with DM-time labels - they MUST be the same value
        # Mark position: use candidate time if available, otherwise use peak time
        # But ALWAYS use display_snr_dw (SNR from detection) for the Y position
        if candidate_snr_intensity_wf is not None and top_boxes is not None and len(top_boxes) > 0:
            # Find candidate time for marking position
            best_idx = int(np.argmax(top_conf)) if top_conf is not None and len(top_conf) > 0 else 0
            if candidate_times_abs is not None and best_idx < len(candidate_times_abs):
                candidate_time_dw = candidate_times_abs[best_idx]
                # Find closest time index in time_axis_dw
                candidate_idx_dw = np.argmin(np.abs(time_axis_dw - candidate_time_dw))
                if 0 <= candidate_idx_dw < len(time_axis_dw):
                    mark_time_dw = float(time_axis_dw[candidate_idx_dw])
                else:
                    mark_time_dw = float(time_axis_dw[peak_idx_dw])
            else:
                mark_time_dw = float(time_axis_dw[peak_idx_dw])
        else:
            mark_time_dw = float(time_axis_dw[peak_idx_dw])
        
        # Mark position with red dot using display_snr_dw (SNR from detection)
        ax_prof_dw.plot(mark_time_dw, display_snr_dw, 'ro', markersize=5)
        
        # Position SNR text to the right of the peak to avoid collision with title
        y_range_dw = ax_prof_dw.get_ylim()[1] - ax_prof_dw.get_ylim()[0]
        x_range_dw = ax_prof_dw.get_xlim()[1] - ax_prof_dw.get_xlim()[0]
        text_x_dw = mark_time_dw + 0.02 * x_range_dw  # 2% to the right
        text_y_dw = display_snr_dw + 0.15 * y_range_dw  # 15% above peak (increased from 10% to avoid title)
        # Use display_snr_dw (SNR from detection, PRESTO-style) - MUST match DM-time label
        ax_prof_dw.text(text_x_dw, text_y_dw, 
                       f'{display_snr_dw:.1f}σ', ha='left', va='bottom', fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='red', linewidth=0.5))
        
        ax_prof_dw.set_xlim(slice_start_abs, slice_end_abs)
        ax_prof_dw.set_ylabel('SNR (σ)', fontsize=8, fontweight='bold')
        ax_prof_dw.grid(True, alpha=0.3)
        ax_prof_dw.set_xticks([])
        
        # Simplified title: use candidate-specific SNR and time if available (consistent with HF pipeline)
        if candidate_snr_intensity_wf is not None and top_boxes is not None and len(top_boxes) > 0:
            best_idx = int(np.argmax(top_conf)) if top_conf is not None and len(top_conf) > 0 else 0
            if candidate_times_abs is not None and best_idx < len(candidate_times_abs):
                candidate_time_dw_title = candidate_times_abs[best_idx]
                ax_prof_dw.set_title(f"Dedispersed Waterfall\nTime: {candidate_time_dw_title:.6f}s | SNR: {display_snr_dw:.1f}σ", fontsize=9, fontweight="bold", pad=12)
            else:
                ax_prof_dw.set_title(f"Dedispersed Waterfall\nPeak SNR: {display_snr_dw:.1f}σ", fontsize=9, fontweight="bold", pad=12)
        else:
            ax_prof_dw.set_title(f"Dedispersed Waterfall\nPeak SNR: {display_snr_dw:.1f}σ", fontsize=9, fontweight="bold", pad=12)
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

        # Dedispersed waterfall uses ORIGINAL time axis (no dispersion correction)
        n_time_ticks_dw = 5
        time_tick_positions_dw = np.linspace(slice_start_abs, slice_end_abs, n_time_ticks_dw)

        ax_dw.set_yticks(freq_tick_positions)
        ax_dw.set_yticklabels([f"{f:.0f}" for f in freq_tick_positions])
        ax_dw.set_xticks(time_tick_positions_dw)
        ax_dw.set_xticklabels([f"{t:.6f}" for t in time_tick_positions_dw], rotation=45)
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

    # NOTE: Candidate Patch SNR plot removed for classic pipeline
    # Only HF pipeline (multi_pol_mode) uses the third panel
    # Classic pipeline now shows only raw and dedispersed waterfalls (2 panels)

                    
    idx_start_ds = int(round(slice_start_abs / (config.TIME_RESO * config.DOWN_TIME_RATE)))
    idx_end_ds = idx_start_ds + real_samples - 1
    start_center = slice_start_abs
    end_center = slice_end_abs
    
    # Simplified title: less verbose
    if chunk_idx is not None:
        title = f"{fits_stem} - Slice {slice_idx:03d}"
    else:
        title = f"{fits_stem} - Slice {slice_idx:03d}"

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)

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
    snr_waterfall_linear: Optional[Iterable[float | None]] = None,  # NEW: SNR from Linear waterfall
    snr_patch_linear: Optional[Iterable[float | None]] = None,  # NEW: SNR from dedispersed Linear patch
    snr_waterfall_intensity: Optional[Iterable[float | None]] = None,  # NEW: SNR from Intensity waterfall
    snr_patch_intensity: Optional[Iterable[float | None]] = None,  # NEW: SNR from dedispersed Intensity patch
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
        snr_waterfall_linear=snr_waterfall_linear,  # NEW: Pass SNR from Linear waterfall
        snr_patch_linear=snr_patch_linear,  # NEW: Pass SNR from dedispersed Linear patch
        snr_waterfall_intensity=snr_waterfall_intensity,  # NEW: Pass SNR from Intensity waterfall
        snr_patch_intensity=snr_patch_intensity,  # NEW: Pass SNR from dedispersed Intensity patch
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
                dedisp_block_linear=dedisp_block_linear,  # NEW: Pass Linear polarization block
                dedisp_block_circular=dedisp_block_circular,  # NEW: Pass Circular polarization block
            )
        except Exception as e:
            logger.warning(f"Could not generate individual plots: {e}")
    
    # Generate polarization time series plots for each candidate
    if candidate_times_abs is not None and len(candidate_times_abs) > 0:
        try:
            from .plot_polarization_timeseries import save_polarization_timeseries_plot
            from ..config import config
            
            logger.info(f"Generating polarization time series plots for {len(candidate_times_abs)} candidate(s)")
            
            # Calculate frequency array
            freq_ds = np.mean(
                config.FREQ.reshape(
                    config.FREQ_RESO // config.DOWN_FREQ_RATE,
                    config.DOWN_FREQ_RATE,
                ),
                axis=1,
            )
            time_reso_ds = config.TIME_RESO * config.DOWN_TIME_RATE
            
            # Calculate slice time boundaries
            if absolute_start_time is not None:
                slice_start_abs = absolute_start_time
            else:
                slice_start_abs = slice_idx * slice_len * time_reso_ds
            
            real_samples = slice_samples if slice_samples is not None else slice_len
            slice_end_abs = slice_start_abs + real_samples * time_reso_ds
            
            # Determine polarization mode
            has_multipol = (dedisp_block_linear is not None) or (dedisp_block_circular is not None)
            pol_mode = "all" if has_multipol else "intensity"
            
            # CRITICAL: Prepare normalized blocks EXACTLY as in create_composite_plot (lines 178-204)
            # This ensures the waterfall data matches exactly what's shown in the composite plot
            dw_block = dedispersed_block.copy() if dedispersed_block is not None and dedispersed_block.size > 0 else None
            dw_linear = None
            dw_circular = None
            if has_multipol:
                if dedisp_block_linear is not None and dedisp_block_linear.size > 0:
                    dw_linear = dedisp_block_linear.copy()
                if dedisp_block_circular is not None and dedisp_block_circular.size > 0:
                    dw_circular = dedisp_block_circular.copy()
            
            # Apply normalization EXACTLY as in create_composite_plot (lines 192-204)
            if normalize:
                blocks_to_norm = [dw_block]
                if has_multipol:
                    blocks_to_norm.extend([dw_linear, dw_circular])
                
                for block in blocks_to_norm:
                    if block is not None:
                        block += 1
                        block /= np.mean(block, axis=0)
                        vmin, vmax = np.nanpercentile(block, [5, 95])
                        block[:] = np.clip(block, vmin, vmax)
                        block -= block.min()
                        block /= block.max() - block.min()
            
            # Generate plot for each candidate
            for cand_idx, cand_time_abs in enumerate(candidate_times_abs):
                # Determine output path - EXACTLY same structure as individual_plots
                # individual_plots uses: base_out_path.parent / output_dir / f"chunk_{chunk_idx:03d}" / f"slice_{slice_idx:03d}"
                # where base_out_path is the composite plot path
                # So polarization_timeseries should be at the same level as individual_plots
                if chunk_idx is not None:
                    # Match exact format: chunk_000 (not chunk_0)
                    pol_dir = out_path.parent / "polarization_timeseries" / f"chunk_{chunk_idx:03d}" / f"slice_{slice_idx:03d}"
                else:
                    pol_dir = out_path.parent / "polarization_timeseries" / f"slice_{slice_idx:03d}"
                
                pol_dir.mkdir(parents=True, exist_ok=True)
                
                # Create filename
                pol_filename = f"{fits_stem}_slice{slice_idx:03d}_cand{cand_idx:02d}_t{cand_time_abs:.3f}s_pol.png"
                pol_path = pol_dir / pol_filename
                
                # Generate and save plot
                # IMPORTANT: Pass the normalized blocks (dw_block, dw_linear, dw_circular)
                # that were prepared above using EXACTLY the same normalization as create_composite_plot
                # This ensures the waterfall matches exactly what's shown in the composite plot
                save_polarization_timeseries_plot(
                    dedisp_intensity=dw_block if dw_block is not None else dedispersed_block,
                    dedisp_linear=dw_linear,
                    dedisp_circular=dw_circular,
                    dm_val=dm_val,
                    candidate_time_abs=cand_time_abs,
                    slice_start_abs=slice_start_abs,
                    slice_end_abs=slice_end_abs,
                    freq_ds=freq_ds,
                    time_reso_ds=time_reso_ds,
                    fits_filename=fits_stem,
                    slice_idx=slice_idx,
                    pol_mode=pol_mode,
                    out_path=pol_path,
                    normalize=False,  # Data is already normalized above using same logic as create_composite_plot
                )
                logger.info(f"✓ Polarization time series plot saved: {pol_path}")
        except Exception as e:
            logger.warning(f"Could not generate polarization time series plots: {e}", exc_info=True)
    else:
        logger.debug(f"No candidate_times_abs provided or empty (candidate_times_abs={candidate_times_abs})")
                                                                                  
