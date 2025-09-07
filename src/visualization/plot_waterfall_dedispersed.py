"""Waterfall dedispersed plot generation module for FRB pipeline - identical to the center panel in composite plot."""
from __future__ import annotations

# Standard library imports
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

# Local imports
from ..analysis.snr_utils import compute_snr_profile, find_snr_peak
from ..config import config
from ..preprocessing.dm_candidate_extractor import extract_candidate_dm

# Setup logger
logger = logging.getLogger(__name__)


def get_band_frequency_range(band_idx: int) -> Tuple[float, float]:
    """Get the frequency range (min, max) for a specific band."""
    freq_ds = np.mean(
        config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE),
        axis=1,
    )
    
    if band_idx == 0:  # Full Band
        return freq_ds.min(), freq_ds.max()
    elif band_idx == 1:  # Low Band 
        mid_channel = len(freq_ds) // 2
        return freq_ds.min(), freq_ds[mid_channel]
    elif band_idx == 2:  # High Band
        mid_channel = len(freq_ds) // 2  
        return freq_ds[mid_channel], freq_ds.max()
    else:
        logger.warning(f"Invalid band index {band_idx}, using Full Band range")
        return freq_ds.min(), freq_ds.max()


def get_band_name_with_freq_range(band_idx: int, band_name: str) -> str:
    """Get band name with frequency range information."""
    freq_min, freq_max = get_band_frequency_range(band_idx)
    return f"{band_name} ({freq_min:.0f}-{freq_max:.0f} MHz)"


def create_waterfall_dedispersed_plot(
    dedispersed_block: np.ndarray,
    waterfall_block: np.ndarray,
    top_conf: Iterable,
    top_boxes: Iterable | None,
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
) -> plt.Figure:
    """Create waterfall dedispersed plot identical to the center panel in composite plot."""
    
    # Get band frequency range for display
    band_name_with_freq = get_band_name_with_freq_range(band_idx, band_name)
    
    freq_ds = np.mean(
        config.FREQ.reshape(
            config.FREQ_RESO // config.DOWN_FREQ_RATE,
            config.DOWN_FREQ_RATE,
        ),
        axis=1,
    )
    time_reso_ds = config.TIME_RESO * config.DOWN_TIME_RATE

    # Check if dedispersed_block is valid
    if dedispersed_block is not None and dedispersed_block.size > 0:
        dw_block = dedispersed_block.copy()
    else:
        dw_block = None
    
    if normalize:
        if dw_block is not None:
            dw_block += 1
            dw_block /= np.mean(dw_block, axis=0)
            vmin, vmax = np.nanpercentile(dw_block, [5, 95])
            dw_block[:] = np.clip(dw_block, vmin, vmax)
            dw_block -= dw_block.min()
            dw_block /= dw_block.max() - dw_block.min()

    # Calculate absolute time ranges - IDÉNTICO al composite
    if absolute_start_time is not None:
        slice_start_abs = absolute_start_time
    else:
        slice_start_abs = slice_idx * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
    
    real_samples = slice_samples if slice_samples is not None else slice_len
    slice_end_abs = slice_start_abs + real_samples * config.TIME_RESO * config.DOWN_TIME_RATE

    # Create figure and gridspec - IDÉNTICO al composite
    fig = plt.figure(figsize=(8, 10))
    gs_dedisp_nested = gridspec.GridSpec(2, 1, height_ratios=[1, 4], hspace=0.05)
    
    # Panel 1: SNR Profile - IDÉNTICO al composite
    ax_prof_dw = fig.add_subplot(gs_dedisp_nested[0, 0])
    
    # Calculate consistent DM value - IDÉNTICO al composite
    if top_boxes is not None and len(top_boxes) > 0:
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
    else:
        dm_val_consistent = 0.0  # Default value if no boxes
        snr_val_candidate = 0.0
    
    if dw_block is not None and dw_block.size > 0:
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
        
        ax_prof_dw.set_xlim(time_axis_dw[0], time_axis_dw[-1])
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
    else:
        ax_prof_dw.text(0.5, 0.5, 'No dedispersed\ndata available', 
                       transform=ax_prof_dw.transAxes, 
                       ha='center', va='center', fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        ax_prof_dw.set_ylabel('SNR (σ)', fontsize=8, fontweight='bold')
        ax_prof_dw.grid(True, alpha=0.3)
        ax_prof_dw.set_xticks([])
        ax_prof_dw.set_title("No Dedispersed Data", fontsize=9, fontweight="bold")

    # Dedispersed waterfall image - IDÉNTICO al composite
    ax_dw = fig.add_subplot(gs_dedisp_nested[1, 0])
    
    if dw_block is not None and dw_block.size > 0:
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

        n_freq_ticks = 6
        freq_tick_positions = np.linspace(freq_ds.min(), freq_ds.max(), n_freq_ticks)
        ax_dw.set_yticks(freq_tick_positions)
        ax_dw.set_yticklabels([f"{f:.0f}" for f in freq_tick_positions])

        n_time_ticks = 5
        time_tick_positions = np.linspace(slice_start_abs, slice_end_abs, n_time_ticks)
        ax_dw.set_xticks(time_tick_positions)
        ax_dw.set_xticklabels([f"{t:.6f}" for t in time_tick_positions])
        ax_dw.set_xlabel("Time (s)", fontsize=9)
        ax_dw.set_ylabel("Frequency (MHz)", fontsize=9)
        
        if 'peak_snr_dw' in locals() and config.SNR_SHOW_PEAK_LINES:
            ax_dw.axvline(x=time_axis_dw[peak_idx_dw], color=config.SNR_HIGHLIGHT_COLOR, 
                         linestyle='-', alpha=0.8, linewidth=2)
    else:
        ax_dw.text(0.5, 0.5, 'No dedispersed data available', 
                  transform=ax_dw.transAxes, 
                  ha='center', va='center', fontsize=12, 
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        ax_dw.set_xticks([])
        ax_dw.set_yticks([])
        ax_dw.set_xlabel("Time (s)", fontsize=9)
        ax_dw.set_ylabel("Frequency (MHz)", fontsize=9)

    # Set main title - IDÉNTICO al composite
    idx_start_ds = int(round(slice_start_abs / (config.TIME_RESO * config.DOWN_TIME_RATE)))
    idx_end_ds = idx_start_ds + real_samples - 1
    start_center = slice_start_abs
    end_center = slice_end_abs
    
    if chunk_idx is not None:
        title = (
            f"Waterfall Dedispersed: {fits_stem} - {band_name_with_freq} - Chunk {chunk_idx:03d} - Slice {slice_idx:03d} | "
            f"start={start_center:.6f}s end={end_center:.6f}s Δt={(config.TIME_RESO * config.DOWN_TIME_RATE):.9f}s "
            f"| [idx {idx_start_ds}→{idx_end_ds}]"
        )
    else:
        title = (
            f"Waterfall Dedispersed: {fits_stem} - {band_name_with_freq} - Slice {slice_idx:03d} | "
            f"start={start_center:.6f}s end={end_center:.6f}s Δt={(config.TIME_RESO * config.DOWN_TIME_RATE):.9f}s "
            f"| [idx {idx_start_ds}→{idx_end_ds}]"
        )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)
    
    # Add temporal information - IDÉNTICO al composite
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


def save_waterfall_dedispersed_plot(
    dedispersed_block: np.ndarray,
    waterfall_block: np.ndarray,
    top_conf: Iterable,
    top_boxes: Iterable | None,
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
) -> None:
    """Save waterfall dedispersed plot by creating the figure and saving it to file."""
    
    # Create the waterfall dedispersed figure
    fig = create_waterfall_dedispersed_plot(
        dedispersed_block=dedispersed_block,
        waterfall_block=waterfall_block,
        top_conf=top_conf,
        top_boxes=top_boxes,
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
    )
    
    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the figure
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

