# This module creates multi-polarization waterfall panels for HF pipeline.

"""Multi-polarization waterfall panel generation for high-frequency pipeline.

For high-frequency observations with full Stokes data, this module creates
visualization panels showing dedispersed waterfalls in three polarizations:
- Intensity (Stokes I)
- Linear Polarization (sqrt(Q^2 + U^2))
- Circular Polarization (|V|)
"""

from __future__ import annotations
import logging
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from ..analysis.snr_utils import compute_snr_profile, find_snr_peak
from ..config import config

logger = logging.getLogger(__name__)


def create_multi_pol_panels(
    fig: plt.Figure,
    gs_bottom_row: gridspec.GridSpecFromSubplotSpec,
    dedisp_intensity: np.ndarray,
    dedisp_linear: Optional[np.ndarray],
    dedisp_circular: Optional[np.ndarray],
    dm_val: float,
    slice_start_abs: float,
    slice_end_abs: float,
    freq_ds: np.ndarray,
    time_reso_ds: float,
    thresh_snr: Optional[float],
    off_regions: Optional[list] = None,
    candidate_time_abs: Optional[float] = None,  # Absolute time of candidate for marking position (Linear)
    candidate_time_intensity: Optional[float] = None,  # Absolute time of candidate for marking position (Intensity)
    candidate_snr_intensity_wf: Optional[float] = None,  # SNR from detection (PRESTO-style) for Intensity waterfall title
    candidate_snr_linear_wf: Optional[float] = None,  # SNR from detection (PRESTO-style) for Linear waterfall title
) -> None:
    """Create three bottom panels showing dedispersed waterfalls in different polarizations.
    
    Args:
        fig: Matplotlib figure
        gs_bottom_row: GridSpec for bottom row (3 columns)
        dedisp_intensity: Dedispersed waterfall in Intensity
        dedisp_linear: Dedispersed waterfall in Linear polarization
        dedisp_circular: Dedispersed waterfall in Circular polarization
        dm_val: Dispersion measure used for dedispersion
        slice_start_abs: Absolute start time of slice
        slice_end_abs: Absolute end time of slice
        freq_ds: Downsampled frequency axis
        time_reso_ds: Temporal resolution after downsampling
        thresh_snr: SNR threshold for highlighting
        off_regions: Off-pulse regions for SNR computation
    """
    
    # Common time tick positions for all panels
    n_time_ticks = 5
    time_tick_positions = np.linspace(slice_start_abs, slice_end_abs, n_time_ticks)
    
    # Common frequency tick positions
    n_freq_ticks = 6
    freq_tick_positions = np.linspace(freq_ds.min(), freq_ds.max(), n_freq_ticks)
    
    polarizations = [
        (dedisp_intensity, "Intensity", "green", 0, candidate_snr_intensity_wf, None),  # Intensity uses SNR from detection (PRESTO-style)
        (dedisp_linear, "Linear Polarization", "purple", 1, candidate_snr_linear_wf, None),  # Linear uses SNR from detection (PRESTO-style)
        (dedisp_circular, "Circular Polarization", "orange", 2, None, None),  # Circular uses global peak
    ]
    
    for pol_data, pol_label, pol_color, panel_idx, candidate_snr_wf, candidate_snr_patch in polarizations:
        # Create nested grid for SNR profile + waterfall
        gs_nested = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs_bottom_row[0, panel_idx], height_ratios=[1, 4], hspace=0.05
        )
        
        # SNR Profile subplot
        ax_prof = fig.add_subplot(gs_nested[0, 0])
        
        # Waterfall subplot  
        ax_waterfall = fig.add_subplot(gs_nested[1, 0])
        
        if pol_data is not None and pol_data.size > 0:
            # Compute SNR profile
            snr_prof, _, best_w = compute_snr_profile(pol_data, off_regions)
            peak_snr, _, peak_idx = find_snr_peak(snr_prof)
            
            time_axis = np.linspace(slice_start_abs, slice_end_abs, len(snr_prof))
            peak_time_abs = float(time_axis[peak_idx]) if len(snr_prof) > 0 else None
            
            # CRITICAL: Use SNR from detection (PRESTO-style) - NOT recalculated from waterfall
            # This ensures consistency with PRESTO methodology and between waterfall title, DM-time label, and red box
            # Determine position to mark: use candidate time if available, otherwise use global peak
            if candidate_time_abs is not None and panel_idx == 1:  # Linear polarization with candidate time
                # Find closest time index to candidate time for marking position
                candidate_idx = np.argmin(np.abs(time_axis - candidate_time_abs))
                if 0 <= candidate_idx < len(snr_prof):
                    mark_idx = candidate_idx
                    mark_time = float(time_axis[candidate_idx])
                else:
                    # Fallback to global peak
                    mark_idx = peak_idx
                    mark_time = peak_time_abs
                # ALWAYS use SNR from detection (PRESTO-style) - this is the correct SNR value
                # This ensures consistency with DM-time labels
                display_snr = candidate_snr_linear_wf if candidate_snr_linear_wf is not None else peak_snr
            else:
                # For Intensity and Circular: use candidate time if available, otherwise global peak
                if candidate_time_intensity is not None and panel_idx == 0:  # Intensity with candidate time
                    candidate_idx = np.argmin(np.abs(time_axis - candidate_time_intensity))
                    if 0 <= candidate_idx < len(snr_prof):
                        mark_idx = candidate_idx
                        mark_time = float(time_axis[candidate_idx])
                    else:
                        mark_idx = peak_idx
                        mark_time = peak_time_abs
                    # ALWAYS use SNR from detection (PRESTO-style) - this is the correct SNR value
                    # This ensures consistency with DM-time labels
                    display_snr = candidate_snr_intensity_wf if candidate_snr_intensity_wf is not None else peak_snr
                else:
                    # Use global peak for Circular, or if no candidate time
                    mark_idx = peak_idx
                    mark_time = peak_time_abs
                    display_snr = peak_snr
            
            # Plot SNR profile
            ax_prof.plot(time_axis, snr_prof, color=pol_color, alpha=0.8, lw=1.5)
            
            if thresh_snr is not None and config.SNR_SHOW_PEAK_LINES:
                above_thresh = snr_prof >= thresh_snr
                if np.any(above_thresh):
                    ax_prof.plot(time_axis[above_thresh], snr_prof[above_thresh],
                               color=config.SNR_HIGHLIGHT_COLOR, lw=2, alpha=0.9)
                ax_prof.axhline(y=thresh_snr, color=config.SNR_HIGHLIGHT_COLOR,
                              linestyle='--', alpha=0.7, linewidth=1)
            
            # Mark position with red dot and show display_snr in red box (consistent with classic pipeline)
            # CRITICAL: Use display_snr (SNR from detection, PRESTO-style) for BOTH the dot position and the text
            # This ensures consistency with DM-time labels - they MUST be the same value
            ax_prof.plot(mark_time, display_snr, 'ro', markersize=5)
            # Calculate text position: slightly to the right and above the peak (like classic pipeline)
            y_range = ax_prof.get_ylim()[1] - ax_prof.get_ylim()[0]
            x_range = ax_prof.get_xlim()[1] - ax_prof.get_xlim()[0]
            text_x = mark_time + 0.02 * x_range  # 2% to the right
            text_y = display_snr + 0.15 * y_range  # 15% above peak
            # Use display_snr (SNR from detection, PRESTO-style) - MUST match DM-time label
            ax_prof.text(text_x, text_y, 
                       f'{display_snr:.1f}σ', ha='left', va='bottom', fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='red', linewidth=0.5))
            
            ax_prof.set_xlim(time_axis[0], time_axis[-1])
            ax_prof.set_ylabel('SNR (σ)', fontsize=8, fontweight='bold')
            ax_prof.grid(True, alpha=0.3)
            ax_prof.set_xticks([])
            
            # Simplified title: polarization name + candidate time (if available)
            # SNR is shown in red box in the time series profile (below)
            if panel_idx == 0:  # Intensity
                if candidate_time_intensity is not None:
                    title = f"Intensity Waterfall\nTime: {candidate_time_intensity:.6f}s"
                else:
                    title = "Intensity Waterfall"
            elif panel_idx == 1:  # Linear
                if candidate_time_abs is not None:
                    title = f"Linear Polarization\nTime: {candidate_time_abs:.6f}s"
                else:
                    title = "Linear Polarization"
            else:  # Circular
                title = "Circular Polarization"
            ax_prof.set_title(title, fontsize=9, fontweight="bold", pad=12)
            
            # Plot waterfall
            ax_waterfall.imshow(
                pol_data.T,
                origin="lower",
                cmap="mako",
                aspect="auto",
                vmin=np.nanpercentile(pol_data, 1),
                vmax=np.nanpercentile(pol_data, 99),
                extent=[slice_start_abs, slice_end_abs, freq_ds.min(), freq_ds.max()],
            )
            ax_waterfall.set_xlim(slice_start_abs, slice_end_abs)
            ax_waterfall.set_ylim(freq_ds.min(), freq_ds.max())
            
            ax_waterfall.set_yticks(freq_tick_positions)
            ax_waterfall.set_yticklabels([f"{f:.0f}" for f in freq_tick_positions])
            ax_waterfall.set_xticks(time_tick_positions)
            ax_waterfall.set_xticklabels([f"{t:.6f}" for t in time_tick_positions], rotation=45)
            ax_waterfall.set_xlabel("Time (s)", fontsize=9)
            ax_waterfall.set_ylabel("Frequency (MHz)", fontsize=9)
            
            # Mark peak time (use candidate time for Linear if available, otherwise global peak)
            mark_time_for_line = candidate_time_abs if (candidate_time_abs is not None and panel_idx == 1) else peak_time_abs
            if config.SNR_SHOW_PEAK_LINES and mark_time_for_line is not None:
                ax_waterfall.axvline(x=mark_time_for_line, color=config.SNR_HIGHLIGHT_COLOR,
                                   linestyle='-', alpha=0.8, linewidth=2)
        
        else:
            # No data available for this polarization
            ax_prof.text(0.5, 0.5, f'No {pol_label}\ndata available',
                        transform=ax_prof.transAxes,
                        ha='center', va='center', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
            ax_prof.set_ylabel('SNR (σ)', fontsize=8, fontweight='bold')
            ax_prof.grid(True, alpha=0.3)
            ax_prof.set_xticks([])
            ax_prof.set_title(f"No {pol_label} Data", fontsize=9, fontweight="bold")
            
            ax_waterfall.text(0.5, 0.5, f'No {pol_label} data available',
                            transform=ax_waterfall.transAxes,
                            ha='center', va='center', fontsize=12,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
            ax_waterfall.set_xticks([])
            ax_waterfall.set_yticks([])
            ax_waterfall.set_xlabel("Time (s)", fontsize=9)
            ax_waterfall.set_ylabel("Frequency (MHz)", fontsize=9)

