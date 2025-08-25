"""Waterfall dispersed plot generation module for FRB pipeline - identical to the left panel in composite plot."""
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

# Setup logger
logger = logging.getLogger(__name__)


def calculate_undispersed_burst_time(
    observed_time: float, 
    dm: float, 
    freq_low: float, 
    freq_high: float
) -> float:
    """
    Calcula el tiempo de llegada no dispersado del burst.
    
    La fórmula para el retraso de dispersión entre dos frecuencias es:
    Δt = k × DM × (1/f_low² - 1/f_high²)
    
    Donde:
    - k ≈ 4.15 × 10³ s·MHz²·pc⁻¹·cm³ (constante de dispersión en unidades cgs)
    - DM es la Medida de Dispersión en pc·cm⁻³
    - f_low y f_high son las frecuencias más baja y más alta en MHz
    
    Para obtener el tiempo no dispersado:
    t_0 = t_obs - Δt
    
    Parameters
    ----------
    observed_time : float
        Tiempo de llegada observado de la señal dispersada (segundos)
    dm : float
        Medida de Dispersión en pc·cm⁻³
    freq_low : float
        Frecuencia más baja en MHz
    freq_high : float
        Frecuencia más alta en MHz
    
    Returns
    -------
    float
        Tiempo de llegada no dispersado del burst (segundos)
    """
    # Constante de dispersión en unidades cgs: k ≈ 4.15 × 10³ s·MHz²·pc⁻¹·cm³
    # Convertir a unidades más convenientes: s·MHz²·pc⁻¹·cm³
    k_dispersion = 4.15e3
    
    # Calcular retraso de dispersión
    delta_t = k_dispersion * dm * ((1.0 / (freq_low**2)) - (1.0 / (freq_high**2)))
    
    # Tiempo no dispersado
    undispersed_time = observed_time - delta_t
    
    logger.debug(f"[CORRECCIÓN DISPERSIÓN] DM: {dm:.2f} pc cm⁻³")
    logger.debug(f"[CORRECCIÓN DISPERSIÓN] Frecuencias: {freq_low:.1f} - {freq_high:.1f} MHz")
    logger.debug(f"[CORRECCIÓN DISPERSIÓN] Retraso calculado: {delta_t:.6f} s")
    logger.debug(f"[CORRECCIÓN DISPERSIÓN] Tiempo observado: {observed_time:.6f} s")
    logger.debug(f"[CORRECCIÓN DISPERSIÓN] Tiempo no dispersado: {undispersed_time:.6f} s")
    
    return undispersed_time


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


def create_waterfall_dispersed_plot(
    waterfall_block: np.ndarray,
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
    dm_value: Optional[float] = None,  # Nuevo parámetro para DM
) -> plt.Figure:
    """Create waterfall dispersed plot identical to the left panel in composite plot."""
    
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

    # Check if waterfall_block is valid
    if waterfall_block is not None and waterfall_block.size > 0:
        wf_block = waterfall_block.copy()
    else:
        wf_block = None
    
    if normalize:
        if wf_block is not None:
            wf_block += 1
            wf_block /= np.mean(wf_block, axis=0)
            vmin, vmax = np.nanpercentile(wf_block, [5, 95])
            wf_block[:] = np.clip(wf_block, vmin, vmax)
            wf_block -= wf_block.min()
            wf_block /= wf_block.max() - wf_block.min()

    # Calculate absolute time ranges - IDÉNTICO al composite
    if absolute_start_time is not None:
        slice_start_abs = absolute_start_time
    else:
        slice_start_abs = slice_idx * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
    
    real_samples = slice_samples if slice_samples is not None else slice_len
    slice_end_abs = slice_start_abs + real_samples * config.TIME_RESO * config.DOWN_TIME_RATE

    # CORRECCIÓN PARA MOSTRAR LA PARÁBOLA NATURAL DEL BURST
    # Si tenemos un valor de DM, calculamos el tiempo no dispersado del burst
    burst_start_time_corrected = slice_start_abs
    if dm_value is not None and dm_value > 0:
        freq_min, freq_max = get_band_frequency_range(band_idx)
        # Calcular el tiempo no dispersado del burst
        burst_start_time_corrected = calculate_undispersed_burst_time(
            observed_time=slice_start_abs,
            dm=dm_value,
            freq_low=freq_min,
            freq_high=freq_max
        )
        logger.info(f"🎯 [CORRECCIÓN BURST] Tiempo original del burst: {burst_start_time_corrected:.6f}s")
        logger.info(f"🎯 [CORRECCIÓN BURST] Tiempo observado: {slice_start_abs:.6f}s")
        logger.info(f"🎯 [CORRECCIÓN BURST] Corrección aplicada para mostrar parábola natural")

    # Create figure and gridspec - IDÉNTICO al composite
    fig = plt.figure(figsize=(8, 10))
    gs_waterfall_nested = gridspec.GridSpec(2, 1, height_ratios=[1, 4], hspace=0.05)
    
    # Panel 1: SNR Profile - IDÉNTICO al composite
    ax_prof_wf = fig.add_subplot(gs_waterfall_nested[0, 0])
    
    if wf_block is not None and wf_block.size > 0:
        snr_wf, sigma_wf = compute_snr_profile(wf_block, off_regions)
        peak_snr_wf, peak_time_wf, peak_idx_wf = find_snr_peak(snr_wf)
        
        # Usar el tiempo corregido para el eje temporal
        time_axis_wf = np.linspace(burst_start_time_corrected, 
                                  burst_start_time_corrected + real_samples * time_reso_ds, 
                                  len(snr_wf))
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
        
        if peak_time_wf_abs is not None:
            ax_prof_wf.set_title(
                f"Raw Waterfall SNR\nPeak={peak_snr_wf:.1f}σ -> {peak_time_wf_abs:.6f}s",
                fontsize=9, fontweight="bold",
            )
        else:
            ax_prof_wf.set_title(f"Raw Waterfall SNR\nPeak={peak_snr_wf:.1f}σ", fontsize=9, fontweight="bold")
    else:
        ax_prof_wf.text(0.5, 0.5, 'No waterfall data\navailable', 
                       transform=ax_prof_wf.transAxes, 
                       ha='center', va='center', fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        ax_prof_wf.set_ylabel('SNR (σ)', fontsize=8, fontweight='bold')
        ax_prof_wf.grid(True, alpha=0.3)
        ax_prof_wf.set_xticks([])
        ax_prof_wf.set_title("No Raw Waterfall Data", fontsize=9, fontweight="bold")

    # Raw waterfall image - IDÉNTICO al composite
    ax_wf = fig.add_subplot(gs_waterfall_nested[1, 0])
    
    if wf_block is not None and wf_block.size > 0:
        # Usar el tiempo corregido para el extent del waterfall
        waterfall_end_time = burst_start_time_corrected + real_samples * time_reso_ds
        
        im_wf = ax_wf.imshow(
            wf_block.T,
            origin="lower",
            cmap="mako",
            aspect="auto",
            vmin=np.nanpercentile(wf_block, 1),
            vmax=np.nanpercentile(wf_block, 99),
            extent=[burst_start_time_corrected, waterfall_end_time, freq_ds.min(), freq_ds.max()],
        )
        ax_wf.set_xlim(burst_start_time_corrected, waterfall_end_time)
        ax_wf.set_ylim(freq_ds.min(), freq_ds.max())

        n_freq_ticks = 6
        freq_tick_positions = np.linspace(freq_ds.min(), freq_ds.max(), n_freq_ticks)
        ax_wf.set_yticks(freq_tick_positions)

        n_time_ticks = 5
        time_tick_positions = np.linspace(burst_start_time_corrected, waterfall_end_time, n_time_ticks)
        ax_wf.set_xticks(time_tick_positions)
        ax_wf.set_xticklabels([f"{t:.6f}" for t in time_tick_positions])
        ax_wf.set_xlabel("Time (s)", fontsize=9)
        ax_wf.set_ylabel("Frequency (MHz)", fontsize=9)
        
        if 'peak_snr_wf' in locals() and config.SNR_SHOW_PEAK_LINES:
            ax_wf.axvline(x=time_axis_wf[peak_idx_wf], color=config.SNR_HIGHLIGHT_COLOR, 
                         linestyle='-', alpha=0.8, linewidth=2)
        
        # Agregar información sobre la corrección de dispersión si se aplicó
        if dm_value is not None and dm_value > 0:
            correction_info = f"DM={dm_value:.1f} pc cm⁻³ | Tiempo corregido para parábola natural"
            ax_wf.text(0.02, 0.98, correction_info, transform=ax_wf.transAxes, 
                      ha='left', va='top', fontsize=8, fontweight='bold',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    else:
        ax_wf.text(0.5, 0.5, 'No waterfall data available', 
                  transform=ax_wf.transAxes, 
                  ha='center', va='center', fontsize=12, 
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        ax_wf.set_xticks([])
        ax_wf.set_yticks([])
        ax_wf.set_xlabel("Time (s)", fontsize=9)
        ax_wf.set_ylabel("Frequency (MHz)", fontsize=9)

    # Set main title - IDÉNTICO al composite
    idx_start_ds = int(round(slice_start_abs / (config.TIME_RESO * config.DOWN_TIME_RATE)))
    idx_end_ds = idx_start_ds + real_samples - 1
    start_center = slice_start_abs
    end_center = slice_end_abs
    
    # Agregar información sobre la corrección en el título
    correction_suffix = ""
    if dm_value is not None and dm_value > 0:
        correction_suffix = f" | DM={dm_value:.1f} pc cm⁻³ (parábola natural)"
    
    if chunk_idx is not None:
        title = (
            f"Waterfall Dispersed: {fits_stem} - {band_name_with_freq} - Chunk {chunk_idx:03d} - Slice {slice_idx:03d} | "
            f"start={start_center:.6f}s end={end_center:.6f}s Δt={(config.TIME_RESO * config.DOWN_TIME_RATE):.9f}s "
            f"| [idx {idx_start_ds}→{idx_end_ds}]{correction_suffix}"
        )
    else:
        title = (
            f"Waterfall Dispersed: {fits_stem} - {band_name_with_freq} - Slice {slice_idx:03d} | "
            f"start={start_center:.6f}s end={end_center:.6f}s Δt={(config.TIME_RESO * config.DOWN_TIME_RATE):.9f}s "
            f"| [idx {idx_start_ds}→{idx_end_ds}]{correction_suffix}"
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
        
        # Agregar información sobre la corrección de dispersión
        if dm_value is not None and dm_value > 0:
            info_lines.append(f"Burst start (corregido): {burst_start_time_corrected:.6f}s")
            info_lines.append(f"Corrección DM: {dm_value:.1f} pc cm⁻³ → parábola natural visible")
        
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


def save_waterfall_dispersed_plot(
    waterfall_block: np.ndarray,
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
    dm_value: Optional[float] = None,  # Nuevo parámetro para DM
) -> None:
    """Save waterfall dispersed plot by creating the figure and saving it to file."""
    
    # Create the waterfall dispersed figure
    fig = create_waterfall_dispersed_plot(
        waterfall_block=waterfall_block,
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
        dm_value=dm_value,  # Pasar el valor de DM
    )
    
    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the figure
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

