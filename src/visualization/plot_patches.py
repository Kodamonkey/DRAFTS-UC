# This module creates patch-level diagnostic plots.

"""Patches plot generation module for FRB pipeline - identical to the right panel in composite plot."""
from __future__ import annotations

                          
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

                     
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

               
from ..analysis.snr_utils import compute_snr_profile, find_snr_peak
from ..config import config

              
logger = logging.getLogger(__name__)


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


def create_patches_plot(
    patch_img: np.ndarray,
    patch_start: float,
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
    """Create patches plot identical to the right panel in composite plot."""
    
                                          
    band_name_with_freq = get_band_name_with_freq_range(band_idx, band_name)
    
    freq_ds = np.mean(
        config.FREQ.reshape(
            config.FREQ_RESO // config.DOWN_FREQ_RATE,
            config.DOWN_FREQ_RATE,
        ),
        axis=1,
    )
    time_reso_ds = config.TIME_RESO * config.DOWN_TIME_RATE

                                 
    if patch_img is not None and patch_img.size > 0:
        patch_data = patch_img.copy()
    else:
        patch_data = None
    
    if normalize:
        if patch_data is not None:
            patch_data += 1
            patch_data /= np.mean(patch_data, axis=0)
            vmin, vmax = np.nanpercentile(patch_data, [5, 95])
            patch_data[:] = np.clip(patch_data, vmin, vmax)
            patch_data -= patch_data.min()
            patch_data /= patch_data.max() - patch_data.min()

                                                            
    if absolute_start_time is not None:
        slice_start_abs = absolute_start_time
    else:
        slice_start_abs = slice_idx * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
    
    real_samples = slice_samples if slice_samples is not None else slice_len
    slice_end_abs = slice_start_abs + real_samples * config.TIME_RESO * config.DOWN_TIME_RATE

                                                        
    fig = plt.figure(figsize=(8, 10))
    gs_patch_nested = gridspec.GridSpec(2, 1, height_ratios=[1, 4], hspace=0.05)
    
                                                  
    ax_patch_prof = fig.add_subplot(gs_patch_nested[0, 0])
    
    if patch_data is not None and patch_data.size > 0:
        snr_patch, sigma_patch, best_w_patch = compute_snr_profile(patch_data, off_regions)
        peak_snr_patch, peak_time_patch, peak_idx_patch = find_snr_peak(snr_patch)
        
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
        ax_patch_prof.set_title(f"Candidate Patch SNR\nPeak={peak_snr_patch:.1f}σ", fontsize=9, fontweight="bold")
    else:
        ax_patch_prof.text(0.5, 0.5, 'No candidate patch\navailable', 
                          transform=ax_patch_prof.transAxes, 
                          ha='center', va='center', fontsize=10, 
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        ax_patch_prof.set_ylabel('SNR (σ)', fontsize=8, fontweight='bold')
        ax_patch_prof.grid(True, alpha=0.3)
        ax_patch_prof.set_xticks([])
        ax_patch_prof.set_title("No Candidate Patch", fontsize=9, fontweight="bold")

                                                   
    ax_patch = fig.add_subplot(gs_patch_nested[1, 0])
    
    if patch_data is not None and patch_data.size > 0:
        ax_patch.imshow(
            patch_data.T,
            origin="lower",
            aspect="auto",
            cmap="mako",
            vmin=np.nanpercentile(patch_data, 1),
            vmax=np.nanpercentile(patch_data, 99),
            extent=[patch_time_axis[0], patch_time_axis[-1], freq_ds.min(), freq_ds.max()],
        )
        ax_patch.set_xlim(patch_time_axis[0], patch_time_axis[-1])
        ax_patch.set_ylim(freq_ds.min(), freq_ds.max())

        n_patch_time_ticks = 5
        patch_tick_positions = np.linspace(patch_time_axis[0], patch_time_axis[-1], n_patch_time_ticks)
        ax_patch.set_xticks(patch_tick_positions)
        ax_patch.set_xticklabels([f"{t:.6f}" for t in patch_tick_positions], rotation=45)

        n_freq_ticks = 6
        freq_tick_positions = np.linspace(freq_ds.min(), freq_ds.max(), n_freq_ticks)
        ax_patch.set_yticks(freq_tick_positions)
        ax_patch.set_yticklabels([f"{f:.0f}" for f in freq_tick_positions])
        ax_patch.set_xlabel("Time (s)", fontsize=9)
        ax_patch.set_ylabel("Frequency (MHz)", fontsize=9)
        
        if 'peak_snr_patch' in locals() and config.SNR_SHOW_PEAK_LINES:
            ax_patch.axvline(x=patch_time_axis[peak_idx_patch], color=config.SNR_HIGHLIGHT_COLOR, 
                           linestyle='-', alpha=0.8, linewidth=2)
    else:
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
            f"Patches Plot: {fits_stem} - {band_name_with_freq} - Chunk {chunk_idx:03d} - Slice {slice_idx:03d} | "
            f"start={start_center:.6f}s end={end_center:.6f}s Δt={(config.TIME_RESO * config.DOWN_TIME_RATE):.9f}s "
            f"| [idx {idx_start_ds}→{idx_end_ds}]"
        )
    else:
        title = (
            f"Patches Plot: {fits_stem} - {band_name_with_freq} - Slice {slice_idx:03d} | "
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


def save_patches_plot(
    patch_img: np.ndarray,
    patch_start: float,
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
    """Save patches plot by creating the figure and saving it to file."""
    
                               
    fig = create_patches_plot(
        patch_img=patch_img,
        patch_start=patch_start,
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
    
                                    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
                     
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

