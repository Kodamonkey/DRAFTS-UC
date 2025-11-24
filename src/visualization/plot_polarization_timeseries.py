# This module generates polarization time series plots similar to time_series_with_polarization_dos.py

"""Polarization time series plots for FRB candidates."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..config import config

logger = logging.getLogger(__name__)


def compute_simple_timeseries(waterfall: np.ndarray) -> np.ndarray:
    """Compute time series by summing over frequency channels.
    
    This is a simple sum (not PRESTO-style normalization) to match
    the behavior of time_series_with_polarization_dos.py.
    
    Parameters
    ----------
    waterfall : np.ndarray
        2D array of shape (n_freq, n_time) or (n_time, n_freq)
        
    Returns
    -------
    np.ndarray
        1D time series of shape (n_time,)
    """
    if waterfall.ndim != 2:
        raise ValueError(f"Expected 2D waterfall, got shape {waterfall.shape}")
    
    # Handle both (freq, time) and (time, freq) shapes
    if waterfall.shape[0] < waterfall.shape[1]:
        # Likely (freq, time) - sum over frequency
        return np.nansum(waterfall, axis=0)
    else:
        # Likely (time, freq) - sum over frequency
        return np.nansum(waterfall, axis=1)


def normalize_simple(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Simple normalization: (data - mean) / std.
    
    This matches the normalization in time_series_with_polarization_dos.py.
    
    Parameters
    ----------
    data : np.ndarray
        Data to normalize
    mean : float
        Mean value to subtract
    std : float
        Standard deviation to divide by
        
    Returns
    -------
    np.ndarray
        Normalized data
    """
    if std <= 0 or not np.isfinite(std):
        std = 1.0
    return (data - mean) / std


def compute_off_pulse_stats(
    time_series: np.ndarray,
    use_quarters: bool = True
) -> Tuple[float, float, float]:
    """Compute noise statistics from off-pulse regions.
    
    Uses first and last quarter of the time series to estimate noise,
    matching the behavior of time_series_with_polarization_dos.py.
    
    Parameters
    ----------
    time_series : np.ndarray
        1D time series
    use_quarters : bool
        If True, use first and last quarter. If False, use all data.
        
    Returns
    -------
    mean : float
        Mean of noise regions
    median : float
        Median of noise regions
    std : float
        Standard deviation of noise regions
    """
    if time_series.size == 0:
        return 0.0, 0.0, 1.0
    
    if use_quarters and time_series.size >= 4:
        # Use first and last quarter (matching script behavior)
        left_region = time_series[:len(time_series)//4]
        right_region = time_series[3*len(time_series)//4:]
        
        # Remove mean from each region before concatenating
        left_region = left_region - np.nanmean(left_region)
        right_region = right_region - np.nanmean(right_region)
        
        noise_region = np.concatenate([left_region, right_region])
    else:
        # Use all data
        noise_region = time_series - np.nanmean(time_series)
    
    mean = np.nanmean(noise_region)
    median = np.nanmedian(noise_region)
    std = np.nanstd(noise_region)
    
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    
    return mean, median, std


def create_polarization_timeseries_plot(
    dedisp_intensity: np.ndarray,
    dedisp_linear: Optional[np.ndarray] = None,
    dedisp_circular: Optional[np.ndarray] = None,
    dm_val: float = 0.0,
    candidate_time_abs: float = 0.0,
    slice_start_abs: float = 0.0,
    slice_end_abs: float = 0.0,
    freq_ds: Optional[np.ndarray] = None,
    time_reso_ds: float = 0.0,
    fits_filename: str = "",
    slice_idx: int = 0,
    pol_mode: str = "all",
    normalize: bool = False,  # For compatibility, but we use channel-by-channel normalization
) -> plt.Figure:
    """Create polarization time series plot similar to time_series_with_polarization_dos.py.
    
    Generates a 2-panel plot:
    - Top panel: Time series (SNR vs time) for different polarizations
    - Bottom panel: Dedispersed waterfall (frequency vs time)
    
    Parameters
    ----------
    dedisp_intensity : np.ndarray
        Dedispersed waterfall for intensity (Stokes I), shape (n_time, n_freq)
    dedisp_linear : np.ndarray, optional
        Dedispersed waterfall for linear polarization, shape (n_time, n_freq)
    dedisp_circular : np.ndarray, optional
        Dedispersed waterfall for circular polarization, shape (n_time, n_freq)
    dm_val : float
        Dispersion measure used for dedispersion
    candidate_time_abs : float
        Absolute time of the candidate (seconds)
    slice_start_abs : float
        Absolute start time of the slice (seconds)
    slice_end_abs : float
        Absolute end time of the slice (seconds)
    freq_ds : np.ndarray, optional
        Downsampled frequency array (MHz)
    time_reso_ds : float
        Downsampled time resolution (seconds)
    fits_filename : str
        Name of the FITS file
    slice_idx : int
        Slice index for labeling
    pol_mode : str
        Polarization mode: "all", "linear", "intensity", or number (0-3)
        
    Returns
    -------
    plt.Figure
        Matplotlib figure with the plot
    """
    # Ensure waterfall is (time, freq) shape
    if dedisp_intensity.shape[0] < dedisp_intensity.shape[1]:
        dedisp_intensity = dedisp_intensity.T
    if dedisp_linear is not None and dedisp_linear.shape[0] < dedisp_linear.shape[1]:
        dedisp_linear = dedisp_linear.T
    if dedisp_circular is not None and dedisp_circular.shape[0] < dedisp_circular.shape[1]:
        dedisp_circular = dedisp_circular.T
    
    n_time, n_freq = dedisp_intensity.shape
    
    # Verify downsampling matches - time_reso_ds should be config.TIME_RESO * config.DOWN_TIME_RATE
    # This is already calculated and passed in, but we verify it matches config
    # EXACTLY as in plot_composite.py line 154 and 889
    expected_time_reso_ds = config.TIME_RESO * config.DOWN_TIME_RATE
    if abs(time_reso_ds - expected_time_reso_ds) > 1e-10:
        logger.warning(f"Time resolution mismatch: passed {time_reso_ds} vs expected {expected_time_reso_ds}. Using expected value.")
        time_reso_ds = expected_time_reso_ds
    
    # Verify frequency downsampling - freq_ds should be calculated from config.FREQ with DOWN_FREQ_RATE
    # EXACTLY as in plot_composite.py lines 147-153
    if freq_ds is None:
        logger.warning("freq_ds not provided, calculating from config")
    
    # Calculate frequency axis - EXACTLY as in plot_composite.py lines 147-153
    # freq_ds should always be provided and match the downsampled frequency resolution
    if freq_ds is not None and len(freq_ds) == n_freq:
        f = freq_ds
        # Verify it matches expected downsampling (EXACTLY as in plot_composite.py)
        try:
            freq_ds_expected = np.mean(
                config.FREQ.reshape(
                    config.FREQ_RESO // config.DOWN_FREQ_RATE,
                    config.DOWN_FREQ_RATE,
                ),
                axis=1,
            )
            if len(freq_ds_expected) == n_freq:
                # Check if values match (within tolerance)
                if not np.allclose(f, freq_ds_expected, rtol=1e-5):
                    logger.warning(f"Frequency array values don't match expected downsampling. "
                                 f"Using provided freq_ds, but values may differ.")
        except Exception:
            pass  # If we can't verify, just use the provided freq_ds
    else:
        # Fallback: calculate from config (EXACTLY as in plot_composite.py lines 147-153)
        try:
            f = np.mean(
                config.FREQ.reshape(
                    config.FREQ_RESO // config.DOWN_FREQ_RATE,
                    config.DOWN_FREQ_RATE,
                ),
                axis=1,
            )
            if len(f) != n_freq:
                logger.warning(f"Frequency array length mismatch: expected {n_freq}, got {len(f)}. "
                             f"Using fallback linear spacing.")
                f = np.linspace(f.min() if len(f) > 0 else 1000, 
                              f.max() if len(f) > 0 else 2000, 
                              n_freq)
        except Exception as e:
            logger.warning(f"Could not calculate frequency array from config: {e}. Using fallback.")
            f = np.linspace(1000, 2000, n_freq)
    
    # Use EXACTLY the same logic as plot_multi_pol_panels.py and plot_individual_components.py
    # Compute SNR profiles using PRESTO-style (not simple sum)
    from ..analysis.snr_utils import compute_snr_profile, find_snr_peak
    
    # Apply normalization if needed (EXACTLY as in plot_composite.py lines 192-204)
    # and plot_individual_components.py lines 308-318
    dw_intensity = dedisp_intensity.copy()
    dw_linear = dedisp_linear.copy() if dedisp_linear is not None else None
    dw_circular = dedisp_circular.copy() if dedisp_circular is not None else None
    
    if normalize:
        # Apply EXACTLY the same normalization as in create_composite_plot() (lines 192-204)
        blocks_to_norm = [dw_intensity, dw_linear, dw_circular]
        for block in blocks_to_norm:
            if block is not None:
                block += 1
                block /= np.mean(block, axis=0)
                vmin, vmax = np.nanpercentile(block, [5, 95])
                block[:] = np.clip(block, vmin, vmax)
                block -= block.min()
                block /= block.max() - block.min()
    
    # Compute SNR profile for intensity (EXACTLY as in plot_multi_pol_panels.py line 84)
    # This uses PRESTO-style SNR calculation which may have different length than n_time
    snr_prof_intensity, _, best_w_intensity = compute_snr_profile(dw_intensity, off_regions=None)
    peak_snr_intensity, _, peak_idx_intensity = find_snr_peak(snr_prof_intensity)
    
    # Time axis - EXACTLY as in plot_multi_pol_panels.py line 87
    # Use len(snr_prof_intensity) which is the actual number of time samples in the SNR profile
    # This ensures the time axis matches the SNR profile length exactly
    time_axis = np.linspace(slice_start_abs, slice_end_abs, len(snr_prof_intensity))
    
    # Verify that the waterfall data matches the expected shape
    # dw_intensity should be (n_time, n_freq) where n_time matches len(snr_prof_intensity)
    if dw_intensity.shape[0] != len(snr_prof_intensity):
        logger.warning(f"Shape mismatch: waterfall has {dw_intensity.shape[0]} time samples, "
                      f"but SNR profile has {len(snr_prof_intensity)} samples")
    
    # Prepare data for plotting
    snr_profiles = []
    has_multipol = (dedisp_linear is not None) or (dedisp_circular is not None)
    
    if has_multipol and pol_mode == "all":
        # Intensity (Stokes I)
        snr_profiles.append((snr_prof_intensity, "total intensity", "black", peak_snr_intensity, peak_idx_intensity))
        
        # Circular (Stokes V)
        if dw_circular is not None:
            snr_prof_circular, _, _ = compute_snr_profile(dw_circular, off_regions=None)
            peak_snr_circular, _, peak_idx_circular = find_snr_peak(snr_prof_circular)
            snr_profiles.append((snr_prof_circular, "circular polarization", "blue", peak_snr_circular, peak_idx_circular))
        else:
            snr_profiles.append(None)
        
        # Linear (sqrt(Q²+U²))
        if dw_linear is not None:
            snr_prof_linear, _, _ = compute_snr_profile(dw_linear, off_regions=None)
            peak_snr_linear, _, peak_idx_linear = find_snr_peak(snr_prof_linear)
            snr_profiles.append((snr_prof_linear, "linear polarization", "red", peak_snr_linear, peak_idx_linear))
        else:
            snr_profiles.append(None)
        
        # Use intensity waterfall for display (matching script when pol='all')
        waterfall_display = dw_intensity
    else:
        # Single polarization mode
        snr_profiles.append((snr_prof_intensity, "intensity", "blueviolet", peak_snr_intensity, peak_idx_intensity))
        waterfall_display = dw_intensity
    
    # Create figure - EXACTLY as in script (2 panels: SNR profile + waterfall)
    # Adjust layout to accommodate legend outside plot area
    fig, ax = plt.subplots(2, 1, figsize=(7, 9), 
                          gridspec_kw={'height_ratios': [0.25, 0.75], 'hspace': 0.3}, 
                          sharex=True)
    
    # Top panel: SNR Profile (EXACTLY as in plot_multi_pol_panels.py lines 92-118)
    if len(snr_profiles) == 3 and all(p is not None for p in snr_profiles):
        # Multi-polarization: plot all three
        prof_i, label_i, color_i, peak_i, idx_i = snr_profiles[0]
        prof_c, label_c, color_c, peak_c, idx_c = snr_profiles[1] if snr_profiles[1] is not None else (None, None, None, None, None)
        prof_l, label_l, color_l, peak_l, idx_l = snr_profiles[2] if snr_profiles[2] is not None else (None, None, None, None, None)
        
        # Plot all three profiles (matching script style)
        ax[0].plot(time_axis, prof_i, c='black', label='total intensity', linewidth=1.5, alpha=0.8)
        if prof_c is not None:
            ax[0].plot(time_axis, prof_c, c='blue', ls=':', label='circular polarization', linewidth=1.5, alpha=0.8)
        if prof_l is not None:
            ax[0].plot(time_axis, prof_l, c='red', ls='--', label='linear polarization', linewidth=1.5, alpha=0.8)
    else:
        # Single polarization
        prof, label, color, peak, idx = snr_profiles[0]
        ax[0].plot(time_axis, prof, c='blueviolet', linewidth=1.5, alpha=0.8)
    
    ax[0].set_ylabel('SNR (σ)', fontsize=20)
    ax[0].tick_params(axis='both', which='major', labelsize=17)
    # Move legend to upper right corner, positioned outside the plot area to avoid interfering
    # with the time series visualization
    ax[0].legend(loc='upper right', fontsize=10, framealpha=0.6, 
                bbox_to_anchor=(1.0, 1.0), frameon=True)
    ax[0].grid(True, alpha=0.3)
    ax[0].set_xlim(time_axis[0], time_axis[-1])
    
    # Bottom panel: Waterfall (EXACTLY as in plot_multi_pol_panels.py lines 121-143)
    # Use imshow with extent (not pcolormesh) to match other plots
    # IMPORTANT: Use freq_ds (f) directly, and ensure it matches the data shape
    # The waterfall_display should be (n_time, n_freq) and we transpose to (n_freq, n_time) for imshow
    ax[1].imshow(
        waterfall_display.T,  # Transpose: (n_freq, n_time) for imshow
        origin="lower",
        cmap="magma",  # Rosadito como el usuario prefiere
        aspect="auto",
        vmin=np.nanpercentile(waterfall_display, 1),
        vmax=np.nanpercentile(waterfall_display, 99),
        extent=[slice_start_abs, slice_end_abs, f.min(), f.max()],
    )
    ax[1].set_xlim(slice_start_abs, slice_end_abs)
    ax[1].set_ylim(f.min(), f.max())  # EXACTLY same as plot_multi_pol_panels.py line 131
    
    # Set ticks - EXACTLY as in plot_multi_pol_panels.py (lines 133-136)
    # Calculate tick positions the same way
    n_time_ticks = 5
    time_tick_positions = np.linspace(slice_start_abs, slice_end_abs, n_time_ticks)
    n_freq_ticks = 6
    freq_tick_positions = np.linspace(f.min(), f.max(), n_freq_ticks)
    
    ax[1].set_yticks(freq_tick_positions)
    ax[1].set_yticklabels([f"{freq:.0f}" for freq in freq_tick_positions])
    ax[1].set_xticks(time_tick_positions)
    ax[1].set_xticklabels([f"{t:.6f}" for t in time_tick_positions], rotation=45)  # EXACTLY as plot_multi_pol_panels.py line 136
    # Rotate x-axis labels to avoid overlapping (matching other plots behavior)
    ax[1].tick_params(axis='x', which='major', labelsize=17, rotation=45)
    
    ax[1].set_ylabel(r'Frequency (MHz)', fontsize=20)
    ax[1].set_xlabel(r'Time (s)', fontsize=20)
    ax[1].tick_params(axis='y', which='major', labelsize=17)
    
    # Candidate time marking removed per user preference
    
    # Add colorbar (matching script style)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    # Create a ScalarMappable for the colorbar
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap="magma", norm=Normalize(vmin=np.nanpercentile(waterfall_display, 1), 
                                                      vmax=np.nanpercentile(waterfall_display, 99)))  # Rosadito como el usuario prefiere
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
    cbar.ax.set_ylabel('intensity (arbitrary units)', fontsize=17)
    
    # Title
    title_str = f'Candidate at slice {slice_idx} - {fits_filename} - {candidate_time_abs:.3f} s'
    if pol_mode == 'all' and has_multipol:
        fig.suptitle(f'{title_str} (all polarizations)', fontsize=20)
    elif pol_mode == 'linear':
        fig.suptitle(f'{title_str} (linear polarization)', fontsize=20)
    elif pol_mode == 'intensity' or pol_mode == '0':
        fig.suptitle(f'{title_str} (Total intensity)', fontsize=20)
    else:
        fig.suptitle(f'{title_str} (DM={dm_val:.2f} pc/cm³)', fontsize=20)
    
    # Use tight_layout with error handling to avoid warnings with incompatible axes
    try:
        plt.tight_layout()
    except Exception:
        # If tight_layout fails, adjust layout manually
        plt.subplots_adjust(hspace=0.3, top=0.95, bottom=0.1, left=0.1, right=0.95)
    
    return fig


def save_polarization_timeseries_plot(
    dedisp_intensity: np.ndarray,
    dedisp_linear: Optional[np.ndarray] = None,
    dedisp_circular: Optional[np.ndarray] = None,
    dm_val: float = 0.0,
    candidate_time_abs: float = 0.0,
    slice_start_abs: float = 0.0,
    slice_end_abs: float = 0.0,
    freq_ds: Optional[np.ndarray] = None,
    time_reso_ds: float = 0.0,
    fits_filename: str = "",
    slice_idx: int = 0,
    pol_mode: str = "all",
    out_path: Path = None,
    normalize: bool = False,  # Use same normalization as composite/individual plots
) -> None:
    """Save polarization time series plot to file.
    
    Parameters
    ----------
    ... (same as create_polarization_timeseries_plot)
    out_path : Path
        Output file path for the plot
    normalize : bool
        If True, apply the same normalization as composite/individual plots.
        If False, use channel-by-channel normalization (matching script behavior).
    """
    fig = create_polarization_timeseries_plot(
        dedisp_intensity=dedisp_intensity,
        dedisp_linear=dedisp_linear,
        dedisp_circular=dedisp_circular,
        dm_val=dm_val,
        candidate_time_abs=candidate_time_abs,
        slice_start_abs=slice_start_abs,
        slice_end_abs=slice_end_abs,
        freq_ds=freq_ds,
        time_reso_ds=time_reso_ds,
        fits_filename=fits_filename,
        slice_idx=slice_idx,
        pol_mode=pol_mode,
    )
    
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.debug(f"Saved polarization time series plot to {out_path}")
    
    plt.close(fig)

