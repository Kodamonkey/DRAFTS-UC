# This module generates individual diagnostic plots.

"""Individual plot components generator for FRB pipeline - creates separate plots for each composite component."""
from __future__ import annotations

                          
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

               
from .plot_dm_time import save_dm_time_plot
from .plot_waterfall_dispersed import save_waterfall_dispersed_plot
from .plot_waterfall_dedispersed import save_waterfall_dedispersed_plot
from .plot_patches import save_patches_plot
from ..analysis.snr_utils import compute_snr_profile, find_snr_peak
from ..config import config

              
logger = logging.getLogger(__name__)


def save_polarization_waterfall_plot(
    pol_data: np.ndarray,
    pol_label: str,
    pol_color: str,
    dm_val: float,
    slice_start_abs: float,
    slice_end_abs: float,
    freq_ds: np.ndarray,
    time_reso_ds: float,
    thresh_snr: Optional[float],
    off_regions: Optional[List[Tuple[int, int]]],
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
) -> None:
    """Save an individual polarization waterfall plot.
    
    Creates a standalone plot showing SNR profile + waterfall for a specific polarization.
    This function uses LITERALLY the same code as the loop in create_multi_pol_panels()
    (lines 70-163 of plot_multi_pol_panels.py) to guarantee 100% identical plots.
    
    Args:
        pol_data: Polarization data array (time, freq)
        pol_label: Polarization label (e.g., "Intensity (I)", "Linear (√(Q²+U²))")
        pol_color: Color for SNR profile plot
        dm_val: Dispersion measure used for dedispersion
        slice_start_abs: Absolute start time of slice
        slice_end_abs: Absolute end time of slice
        freq_ds: Downsampled frequency axis
        time_reso_ds: Temporal resolution after downsampling
        thresh_snr: SNR threshold for highlighting
        off_regions: Off-pulse regions for SNR computation
        out_path: Output file path
        ... (other parameters for metadata)
    """
    
    # Create figure with SNR profile + waterfall layout
    # Using gridspec to match exactly the composite plot structure
    # In composite: gs_bottom_row is 1x3, then gs_nested is 2x1 inside each column
    # For individual plot: create equivalent structure
    fig = plt.figure(figsize=(7, 9))
    
    # Create a main grid (equivalent to gs_bottom_row but with 1 column instead of 3)
    gs_main = gridspec.GridSpec(1, 1, figure=fig)
    
    # Create nested grid for SNR profile + waterfall (EXACTLY as in create_multi_pol_panels line 72-74)
    # In composite: gs_nested uses gs_bottom_row[0, panel_idx], here we use gs_main[0, 0]
    gs_nested = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_main[0, 0], height_ratios=[1, 4], hspace=0.05
    )
    
    # SNR Profile subplot (EXACTLY as in create_multi_pol_panels line 77)
    ax_prof = fig.add_subplot(gs_nested[0, 0])
    
    # Waterfall subplot (EXACTLY as in create_multi_pol_panels line 80)
    ax_waterfall = fig.add_subplot(gs_nested[1, 0])
    
    # Common time tick positions for all panels (EXACTLY as in create_multi_pol_panels lines 56-58)
    n_time_ticks = 5
    time_tick_positions = np.linspace(slice_start_abs, slice_end_abs, n_time_ticks)
    
    # Common frequency tick positions (EXACTLY as in create_multi_pol_panels lines 60-62)
    n_freq_ticks = 6
    freq_tick_positions = np.linspace(freq_ds.min(), freq_ds.max(), n_freq_ticks)
    
    # LITERALLY the same code as create_multi_pol_panels lines 82-163
    if pol_data is not None and pol_data.size > 0:
        # Compute SNR profile (EXACTLY line 84)
        snr_prof, _, best_w = compute_snr_profile(pol_data, off_regions)
        peak_snr, _, peak_idx = find_snr_peak(snr_prof)
        
        time_axis = np.linspace(slice_start_abs, slice_end_abs, len(snr_prof))
        peak_time_abs = float(time_axis[peak_idx]) if len(snr_prof) > 0 else None
        width_ms = float(best_w[int(peak_idx)]) * time_reso_ds * 1000.0 if len(best_w) == len(snr_prof) else None
        
        # Plot SNR profile (EXACTLY line 92)
        ax_prof.plot(time_axis, snr_prof, color=pol_color, alpha=0.8, lw=1.5)
        
        if thresh_snr is not None and config.SNR_SHOW_PEAK_LINES:
            above_thresh = snr_prof >= thresh_snr
            if np.any(above_thresh):
                ax_prof.plot(time_axis[above_thresh], snr_prof[above_thresh],
                           color=config.SNR_HIGHLIGHT_COLOR, lw=2, alpha=0.9)
            ax_prof.axhline(y=thresh_snr, color=config.SNR_HIGHLIGHT_COLOR,
                          linestyle='--', alpha=0.7, linewidth=1)
        
        ax_prof.plot(time_axis[peak_idx], peak_snr, 'ro', markersize=5)
        ax_prof.text(time_axis[peak_idx], peak_snr + 0.1 * (ax_prof.get_ylim()[1] - ax_prof.get_ylim()[0]),
                    f'{peak_snr:.1f}σ', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax_prof.set_xlim(time_axis[0], time_axis[-1])
        ax_prof.set_ylabel('SNR (σ)', fontsize=8, fontweight='bold')
        ax_prof.grid(True, alpha=0.3)
        ax_prof.set_xticks([])
        
        # Title with peak info (EXACTLY lines 112-118)
        if peak_time_abs is not None and width_ms is not None:
            title = f"Dedispersed {pol_label} (DM={dm_val:.1f})\nPeak={peak_snr:.1f}σ (w≈{width_ms:.3f} ms) @ {peak_time_abs:.6f}s"
        elif peak_time_abs is not None:
            title = f"Dedispersed {pol_label} (DM={dm_val:.1f})\nPeak={peak_snr:.1f}σ @ {peak_time_abs:.6f}s"
        else:
            title = f"Dedispersed {pol_label} (DM={dm_val:.1f})\nPeak={peak_snr:.1f}σ"
        ax_prof.set_title(title, fontsize=9, fontweight="bold")
        
        # Plot waterfall (EXACTLY lines 121-129)
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
        
        # Mark peak time (EXACTLY lines 141-143)
        if config.SNR_SHOW_PEAK_LINES and peak_time_abs is not None:
            ax_waterfall.axvline(x=peak_time_abs, color=config.SNR_HIGHLIGHT_COLOR,
                               linestyle='-', alpha=0.8, linewidth=2)
    
    else:
        # No data available for this polarization (EXACTLY lines 147-163)
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
    
    # Save the plot
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def generate_individual_plots(
    waterfall_block,
    dedispersed_block,
    img_rgb,
    patch_img,
    patch_start,
    dm_val,
    top_conf,
    top_boxes,
    class_probs,
    base_out_path: Path,
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
    output_dir: str = "individual_plots",
    class_probs_linear: Optional[Iterable[float]] = None,  # NEW: Linear classification probs
    dedisp_block_linear: Optional[np.ndarray] = None,  # NEW: Linear polarization waterfall
    dedisp_block_circular: Optional[np.ndarray] = None,  # NEW: Circular polarization waterfall
) -> None:
    """Generate individual plots for each component of the composite plot.
    
    For High Frequency pipeline (when dedisp_block_linear and dedisp_block_circular are provided):
    1. DM-Time plot (detections)
    2. Waterfall dedispersed Intensity plot
    3. Waterfall dedispersed Linear polarization plot  
    4. Waterfall dedispersed Circular polarization plot
    
    For Classic pipeline (when polarization blocks are not provided):
    1. DM-Time plot (detections)
    2. Waterfall dispersed plot
    3. Waterfall dedispersed plot
    4. Patches plot (candidate centered)
    
    All plots are IDENTICAL to their corresponding panels in the composite plot.
    """
    
                                                  
    if chunk_idx is not None:
        individual_dir = base_out_path.parent / output_dir / f"chunk_{chunk_idx:03d}" / f"slice_{slice_idx:03d}"
    else:
        individual_dir = base_out_path.parent / output_dir / f"slice_{slice_idx:03d}"
    
    individual_dir.mkdir(parents=True, exist_ok=True)
    
                            
    base_filename = f"{fits_stem}_slice_{slice_idx:03d}"
    if chunk_idx is not None:
        base_filename = f"{fits_stem}_chunk_{chunk_idx:03d}_slice_{slice_idx:03d}"
    
    logger.info(f"Generating individual plots in: {individual_dir}")
    
    try:
                                                                                 
        dm_time_path = individual_dir / f"{base_filename}_dm_time.png"
        save_dm_time_plot(
            img_rgb=img_rgb,
            top_conf=top_conf,
            top_boxes=top_boxes,
            class_probs=class_probs,
            out_path=dm_time_path,
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
        logger.info(f"✓ DM-Time plot saved: {dm_time_path}")
        
        # Detect High Frequency mode (multi-polarization)
        multi_pol_mode = (dedisp_block_linear is not None and dedisp_block_circular is not None)
        
        if multi_pol_mode:
            # HIGH FREQUENCY PIPELINE: Generate 3 polarization waterfalls
            logger.info("High Frequency mode detected: generating polarization waterfalls")
            
            # Prepare common parameters for polarization plots
            freq_ds = np.mean(
                config.FREQ.reshape(
                    config.FREQ_RESO // config.DOWN_FREQ_RATE,
                    config.DOWN_FREQ_RATE,
                ),
                axis=1,
            )
            time_reso_ds = config.TIME_RESO * config.DOWN_TIME_RATE
            
            # Calculate slice time boundaries (EXACTLY as in create_composite_plot)
            if absolute_start_time is not None:
                slice_start_abs = absolute_start_time
            else:
                slice_start_abs = slice_idx * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
            
            real_samples = slice_samples if slice_samples is not None else slice_len
            slice_end_abs = slice_start_abs + real_samples * config.TIME_RESO * config.DOWN_TIME_RATE
            
            # CRITICAL: Apply the SAME normalization as in create_composite_plot()
            # This ensures the plots are EXACTLY identical to the composite panels
            dw_intensity = dedispersed_block.copy() if dedispersed_block is not None and dedispersed_block.size > 0 else None
            dw_linear = dedisp_block_linear.copy() if dedisp_block_linear is not None and dedisp_block_linear.size > 0 else None
            dw_circular = dedisp_block_circular.copy() if dedisp_block_circular is not None and dedisp_block_circular.size > 0 else None
            
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
            
            # Generate Intensity waterfall (using normalized data)
            intensity_path = individual_dir / f"{base_filename}_waterfall_dedispersed_intensity.png"
            save_polarization_waterfall_plot(
                pol_data=dw_intensity,
                pol_label="Intensity (I)",
                pol_color="green",
                dm_val=dm_val,
                slice_start_abs=slice_start_abs,
                slice_end_abs=slice_end_abs,
                freq_ds=freq_ds,
                time_reso_ds=time_reso_ds,
                thresh_snr=thresh_snr,
                off_regions=off_regions,
                out_path=intensity_path,
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
            )
            logger.info(f"✓ Intensity waterfall plot saved: {intensity_path}")
            
            # Generate Linear polarization waterfall (using normalized data)
            linear_path = individual_dir / f"{base_filename}_waterfall_dedispersed_linear.png"
            save_polarization_waterfall_plot(
                pol_data=dw_linear,
                pol_label="Linear (√(Q²+U²))",
                pol_color="purple",
                dm_val=dm_val,
                slice_start_abs=slice_start_abs,
                slice_end_abs=slice_end_abs,
                freq_ds=freq_ds,
                time_reso_ds=time_reso_ds,
                thresh_snr=thresh_snr,
                off_regions=off_regions,
                out_path=linear_path,
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
            )
            logger.info(f"✓ Linear polarization waterfall plot saved: {linear_path}")
            
            # Generate Circular polarization waterfall (using normalized data)
            circular_path = individual_dir / f"{base_filename}_waterfall_dedispersed_circular.png"
            save_polarization_waterfall_plot(
                pol_data=dw_circular,
                pol_label="Circular (|V|)",
                pol_color="orange",
                dm_val=dm_val,
                slice_start_abs=slice_start_abs,
                slice_end_abs=slice_end_abs,
                freq_ds=freq_ds,
                time_reso_ds=time_reso_ds,
                thresh_snr=thresh_snr,
                off_regions=off_regions,
                out_path=circular_path,
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
            )
            logger.info(f"✓ Circular polarization waterfall plot saved: {circular_path}")
            
            logger.info(f"All High Frequency individual plots generated successfully in: {individual_dir}")
            
        else:
            # CLASSIC PIPELINE: Generate traditional 4 plots
            logger.info("Classic mode detected: generating traditional individual plots")
        
                                                                                 
        waterfall_dispersed_path = individual_dir / f"{base_filename}_waterfall_dispersed.png"
        save_waterfall_dispersed_plot(
            waterfall_block=waterfall_block,
            out_path=waterfall_dispersed_path,
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
            dm_value=dm_val,                                            
        )
        logger.info(f"✓ Waterfall dispersed plot saved: {waterfall_dispersed_path}")
        
                                                                                 
        waterfall_dedispersed_path = individual_dir / f"{base_filename}_waterfall_dedispersed.png"
        save_waterfall_dedispersed_plot(
            dedispersed_block=dedispersed_block,
            waterfall_block=waterfall_block,
            top_conf=top_conf,
            top_boxes=top_boxes,
            out_path=waterfall_dedispersed_path,
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
        logger.info(f"✓ Waterfall dedispersed plot saved: {waterfall_dedispersed_path}")
        
                                                                                        
        patches_path = individual_dir / f"{base_filename}_patches.png"
        save_patches_plot(
            patch_img=patch_img,
            patch_start=patch_start,
            out_path=patches_path,
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
        logger.info(f"✓ Patches plot saved: {patches_path}")
        
        logger.info(f"All classic individual plots generated successfully in: {individual_dir}")
        
    except Exception as e:
        logger.error(f"Error generating individual plots: {e}")
        raise


def generate_individual_plots_from_composite_params(
    composite_params: dict,
    base_out_path: Path,
    output_dir: str = "individual_plots",
) -> None:
    """Generate individual plots using parameters from a composite plot call.
    
    This is a convenience function that extracts parameters from a dictionary
    and calls generate_individual_plots.
    """
    
                                 
    required_params = [
        'waterfall_block', 'dedispersed_block', 'img_rgb', 'patch_img',
        'patch_start', 'dm_val', 'top_conf', 'top_boxes', 'class_probs',
        'slice_idx', 'time_slice', 'band_name', 'band_suffix', 'fits_stem', 'slice_len'
    ]
    
                                                  
    missing_params = [param for param in required_params if param not in composite_params]
    if missing_params:
        raise ValueError(f"Missing required parameters: {missing_params}")
    
                                               
    normalize = composite_params.get('normalize', False)
    off_regions = composite_params.get('off_regions', None)
    thresh_snr = composite_params.get('thresh_snr', None)
    band_idx = composite_params.get('band_idx', 0)
    absolute_start_time = composite_params.get('absolute_start_time', None)
    chunk_idx = composite_params.get('chunk_idx', None)
    slice_samples = composite_params.get('slice_samples', None)
    candidate_times_abs = composite_params.get('candidate_times_abs', None)
    class_probs_linear = composite_params.get('class_probs_linear', None)
    dedisp_block_linear = composite_params.get('dedisp_block_linear', None)
    dedisp_block_circular = composite_params.get('dedisp_block_circular', None)
    
                            
    generate_individual_plots(
        waterfall_block=composite_params['waterfall_block'],
        dedispersed_block=composite_params['dedispersed_block'],
        img_rgb=composite_params['img_rgb'],
        patch_img=composite_params['patch_img'],
        patch_start=composite_params['patch_start'],
        dm_val=composite_params['dm_val'],
        top_conf=composite_params['top_conf'],
        top_boxes=composite_params['top_boxes'],
        class_probs=composite_params['class_probs'],
        base_out_path=base_out_path,
        slice_idx=composite_params['slice_idx'],
        time_slice=composite_params['time_slice'],
        band_name=composite_params['band_name'],
        band_suffix=composite_params['band_suffix'],
        fits_stem=composite_params['fits_stem'],
        slice_len=composite_params['slice_len'],
        normalize=normalize,
        off_regions=off_regions,
        thresh_snr=thresh_snr,
        band_idx=band_idx,
        absolute_start_time=absolute_start_time,
        chunk_idx=chunk_idx,
        slice_samples=slice_samples,
        candidate_times_abs=candidate_times_abs,
        output_dir=output_dir,
        class_probs_linear=class_probs_linear,
        dedisp_block_linear=dedisp_block_linear,
        dedisp_block_circular=dedisp_block_circular,
    )

