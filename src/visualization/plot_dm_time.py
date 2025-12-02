# This module plots dispersion measure versus time.

"""DM-Time plot generation module for FRB pipeline - identical to the detection panel in composite plot."""
from __future__ import annotations

                          
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

                     
import matplotlib.pyplot as plt
import numpy as np

               
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


def create_dm_time_plot(
    img_rgb: np.ndarray,
    top_conf: Iterable,
    top_boxes: Iterable | None,
    class_probs: Iterable | None,
    slice_idx: int,
    time_slice: int,
    band_name: str,
    band_suffix: str,
    fits_stem: str,
    slice_len: int,
    band_idx: int = 0,
    absolute_start_time: Optional[float] = None, 
    chunk_idx: Optional[int] = None,  
    slice_samples: Optional[int] = None,
    candidate_times_abs: Optional[Iterable[float]] = None,
    class_probs_linear: Optional[Iterable[float]] = None,  # NEW: Linear classification probs
) -> plt.Figure:
    """Create DM-Time plot identical to the detection panel in composite plot."""
    
                                          
    band_name_with_freq = get_band_name_with_freq_range(band_idx, band_name)
    
                                    
    if absolute_start_time is not None:
        slice_start_abs = absolute_start_time
    else:
        slice_start_abs = slice_idx * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
    
    real_samples = slice_samples if slice_samples is not None else slice_len
    slice_end_abs = slice_start_abs + real_samples * config.TIME_RESO * config.DOWN_TIME_RATE

                   
    fig = plt.figure(figsize=(14, 8))
    ax_det = fig.add_subplot(111)
    
                                                    
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

                                                                           
    if top_boxes is not None:
        for idx, (conf, box) in enumerate(zip(top_conf, top_boxes)):
            x1, y1, x2, y2 = map(int, box)
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            effective_len_det = slice_samples if slice_samples is not None else slice_len
            dm_val_cand, t_sec_real, t_sample_real = extract_candidate_dm(center_x, center_y, effective_len_det)
            
            # Calculate detection time
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
                # Show Linear prob if available, even if it's 0.0 (means classification was performed)
                is_burst_L = False
                class_prob_L = 0.0
                has_linear_classification = (class_probs_linear is not None and 
                                            idx < len(class_probs_linear))
                
                if has_linear_classification:
                    class_prob_L = class_probs_linear[idx]
                    is_burst_L = class_prob_L >= config.CLASS_PROB
                    
                    # HF Mode: Lógica de colores mejorada
                    # Verde (lime): Ambas clasificaciones pasan (I y L)
                    # Morado (purple): I pasa pero L NO pasa
                    # Naranja (orange): Ninguna pasa (ni I ni L)
                    if is_burst_I and is_burst_L:
                        color = "lime"
                        burst_status = "BURST (I+L)"
                    elif is_burst_I and not is_burst_L:
                        color = "purple"
                        burst_status = "I:BURST L:NO"
                    else:  # not is_burst_I (y posiblemente not is_burst_L)
                        color = "orange"
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

                                            
    idx_start_ds = int(round(slice_start_abs / (config.TIME_RESO * config.DOWN_TIME_RATE)))
    idx_end_ds = idx_start_ds + real_samples - 1
    start_center = slice_start_abs
    end_center = slice_end_abs
    
    if chunk_idx is not None:
        title = (
            f"DM-Time Plot: {fits_stem} - {band_name_with_freq} - Chunk {chunk_idx:03d} - Slice {slice_idx:03d} | "
            f"start={start_center:.6f}s end={end_center:.6f}s Δt={(config.TIME_RESO * config.DOWN_TIME_RATE):.9f}s "
            f"| [idx {idx_start_ds}→{idx_end_ds}]"
        )
    else:
        title = (
            f"DM-Time Plot: {fits_stem} - {band_name_with_freq} - Slice {slice_idx:03d} | "
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


def save_dm_time_plot(
    img_rgb: np.ndarray,
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
    band_idx: int = 0,
    absolute_start_time: Optional[float] = None, 
    chunk_idx: Optional[int] = None,  
    slice_samples: Optional[int] = None,  
    candidate_times_abs: Optional[Iterable[float]] = None,
    class_probs_linear: Optional[Iterable[float]] = None,  # NEW: Linear classification probs
) -> None:
    """Save DM-Time plot by creating the figure and saving it to file."""
    
    fig = create_dm_time_plot(
        img_rgb=img_rgb,
        top_conf=top_conf,
        top_boxes=top_boxes,
        class_probs=class_probs,
        slice_idx=slice_idx,
        time_slice=time_slice,
        band_name=band_name,
        band_suffix=band_suffix,
        fits_stem=fits_stem,
        slice_len=slice_len,
        band_idx=band_idx,
        absolute_start_time=absolute_start_time,
        chunk_idx=chunk_idx,
        slice_samples=slice_samples,
        candidate_times_abs=candidate_times_abs,
        class_probs_linear=class_probs_linear,  # NEW: Pass Linear probs
    )
    
                                    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
                     
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

