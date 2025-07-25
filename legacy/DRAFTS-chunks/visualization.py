"""Helper functions for visualizations used in the pipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from . import config
from .astro_conversions import pixel_to_physical
from .image_utils import postprocess_img, preprocess_img, save_detection_plot, plot_waterfall_block, _calculate_dynamic_dm_range
from .dedispersion import dedisperse_block
from .snr_utils import compute_snr_profile, find_snr_peak

logger = logging.getLogger(__name__)


def get_band_frequency_range(band_idx: int) -> Tuple[float, float]:
    """Get the frequency range (min, max) for a specific band.
    
    Parameters
    ----------
    band_idx : int
        Band index (0=Full, 1=Low, 2=High)
        
    Returns
    -------
    Tuple[float, float]
        (freq_min, freq_max) in MHz
    """
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
    """Get band name with frequency range information.
    
    Parameters
    ----------
    band_idx : int
        Band index (0=Full, 1=Low, 2=High)
    band_name : str
        Original band name (e.g., "Full Band", "Low Band", "High Band")
        
    Returns
    -------
    str
        Band name with frequency range (e.g., "Full Band (1200-1500 MHz)")
    """
    freq_min, freq_max = get_band_frequency_range(band_idx)
    return f"{band_name} ({freq_min:.0f}-{freq_max:.0f} MHz)"


# ...existing code...

def save_detection_plot(
    img_rgb: np.ndarray,
    top_conf: Iterable,
    top_boxes: Iterable | None,
    class_probs: Iterable | None, 
    out_img_path: Path,
    slice_idx: int,
    time_slice: int,
    band_name: str,
    band_suffix: str,
    det_prob: float,
    fits_stem: str,
    slice_len: Optional[int] = None,
    band_idx: int = 0,  # Para calcular el rango de frecuencias de la banda
    absolute_start_time: Optional[float] = None,  # <-- NUEVO PARÁMETRO
) -> None:
    """Save detection plot with both detection and classification probabilities."""

    # Usar slice_len específico o del config
    if slice_len is None:
        slice_len = config.SLICE_LEN

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(img_rgb, origin="lower", aspect="auto")

    # Time axis labels
    n_time_ticks = 6
    time_positions = np.linspace(0, 512, n_time_ticks)
    # --- CAMBIO: Usar tiempo absoluto si se proporciona ---
    if absolute_start_time is not None:
        time_start_slice = absolute_start_time
    else:
        time_start_slice = slice_idx * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
    time_values = time_start_slice + (
        time_positions / 512.0
    ) * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
    ax.set_xticks(time_positions)
    ax.set_xticklabels([f"{t:.3f}" for t in time_values])
    ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")

    # ...rest of function unchanged...

# ...existing code...

def save_plot(
    img_rgb: np.ndarray,
    top_conf: Iterable,
    top_boxes: Iterable | None,
    class_probs: Iterable | None,
    out_img_path: Path,
    slice_idx: int,
    time_slice: int,
    band_name: str,
    band_suffix: str,
    fits_stem: str,
    slice_len: int,
    band_idx: int = 0,  # Para calcular el rango de frecuencias
    absolute_start_time: Optional[float] = None,  # <-- NUEVO PARÁMETRO
) -> None:
    """Wrapper around :func:`save_detection_plot` with dynamic slice length."""

    prev_len = config.SLICE_LEN
    config.SLICE_LEN = slice_len
    
    # Agregar información de rango de frecuencias al nombre de la banda
    band_name_with_freq = get_band_name_with_freq_range(band_idx, band_name)
    
    save_detection_plot(
        img_rgb,
        top_conf,
        top_boxes,
        class_probs,
        out_img_path,
        slice_idx,
        time_slice,
        band_name_with_freq,
        band_suffix,
        config.DET_PROB,
        fits_stem,
        slice_len=slice_len,  # Pasar slice_len explícitamente
        band_idx=band_idx,    # Pasar band_idx para el cálculo de frecuencias
        absolute_start_time=absolute_start_time,  # <-- PASAR TIEMPO ABSOLUTO
    )
    config.SLICE_LEN = prev_len

# ...existing code...


def save_patch_plot(
    patch: np.ndarray,
    out_path: Path,
    freq: np.ndarray,
    time_reso: float,
    start_time: float,
    off_regions: Optional[List[Tuple[int, int]]] = None,
    thresh_snr: Optional[float] = None,
    band_idx: int = 0,  # Para mostrar el rango de frecuencias
    band_name: str = "Unknown Band",  # Nombre de la banda
) -> None:
    """Save a visualization of the classification patch with SNR profile.
    
    Parameters
    ----------
    patch : np.ndarray
        2D array with shape (n_time, n_freq)
    out_path : Path
        Output file path
    freq : np.ndarray
        Frequency axis values
    time_reso : float
        Time resolution in seconds
    start_time : float
        Start time in seconds
    off_regions : Optional[List[Tuple[int, int]]]
        Off-pulse regions for SNR calculation
    thresh_snr : Optional[float]
        SNR threshold for highlighting
    band_idx : int
        Band index for frequency range calculation
    band_name : str
        Name of the band for display
    """

    # Check if patch is valid
    if patch is None or patch.size == 0:
        logger.warning(f"Cannot create patch plot: patch is None or empty. Skipping {out_path}")
        return

    # Calculate SNR profile
    snr_profile, sigma = compute_snr_profile(patch, off_regions)
    peak_snr, peak_time_rel, peak_idx = find_snr_peak(snr_profile)
    
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4], hspace=0.05)

    time_axis = start_time + np.arange(patch.shape[0]) * time_reso
    peak_time_abs = start_time + peak_idx * time_reso

    # Get band frequency range for title
    band_name_with_freq = get_band_name_with_freq_range(band_idx, band_name)
    
    # Top panel: SNR profile
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(time_axis, snr_profile, color="royalblue", alpha=0.8, lw=1.5)
    
    # Highlight regions above threshold
    if thresh_snr is not None and config.SNR_SHOW_PEAK_LINES:
        above_thresh = snr_profile >= thresh_snr
        if np.any(above_thresh):
            ax0.plot(time_axis[above_thresh], snr_profile[above_thresh], 
                    color=config.SNR_HIGHLIGHT_COLOR, lw=2, alpha=0.9)
        
        # Add threshold line
        ax0.axhline(y=thresh_snr, color=config.SNR_HIGHLIGHT_COLOR, 
                   linestyle='--', alpha=0.7, label=f'Thresh = {thresh_snr:.1f}σ')
    
    # Mark peak
    ax0.plot(peak_time_abs, peak_snr, 'ro', markersize=6, alpha=0.8)
    ax0.text(peak_time_abs, peak_snr + 0.1 * (ax0.get_ylim()[1] - ax0.get_ylim()[0]), 
             f'SNR = {peak_snr:.1f}σ', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax0.set_xlim(time_axis[0], time_axis[-1])
    ax0.set_ylabel('SNR (σ)', fontsize=10, fontweight='bold')
    ax0.grid(True, alpha=0.3)
    if thresh_snr is not None and config.SNR_SHOW_PEAK_LINES:
        ax0.legend(fontsize=8, loc='upper right')
    
    # Remove x-axis labels for top panel
    ax0.set_xticks([])

    # Bottom panel: Waterfall
    ax1 = fig.add_subplot(gs[1, 0])
    im = ax1.imshow(
        patch.T,
        origin="lower",
        aspect="auto",
        cmap=config.SNR_COLORMAP,
        vmin=np.nanpercentile(patch, 1),
        vmax=np.nanpercentile(patch, 99),
        extent=[time_axis[0], time_axis[-1], freq.min(), freq.max()],
    )

    ax1.set_xlim(time_axis[0], time_axis[-1])
    ax1.set_ylim(freq.min(), freq.max())

    n_freq_ticks = 6
    freq_tick_positions = np.linspace(freq.min(), freq.max(), n_freq_ticks)
    ax1.set_yticks(freq_tick_positions)
    ax1.set_yticklabels([f"{f:.0f}" for f in freq_tick_positions])

    n_time_ticks = 6
    time_tick_positions = np.linspace(time_axis[0], time_axis[-1], n_time_ticks)
    ax1.set_xticks(time_tick_positions)
    ax1.set_xticklabels([f"{t:.3f}" for t in time_tick_positions])

    # Mark peak position on waterfall
    if config.SNR_SHOW_PEAK_LINES:
        ax1.axvline(x=peak_time_abs, color=config.SNR_HIGHLIGHT_COLOR, 
                   linestyle='-', alpha=0.8, linewidth=2)

    ax1.set_xlabel("Time (s)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Frequency (MHz)", fontsize=10, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Normalized Intensity', fontsize=9, fontweight='bold')

    # Add title with band frequency range information
    plt.suptitle(f"Candidate Patch - {band_name_with_freq}", fontsize=12, fontweight='bold')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08, hspace=0.2, wspace=0.2)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    plt.close('all')  # Cerrar todas las figuras para liberar memoria


def save_slice_summary(
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
    band_idx: int = 0,  # Para mostrar el rango de frecuencias
    absolute_start_time: Optional[float] = None,  # 🕐 NUEVO PARÁMETRO PARA TIEMPO ABSOLUTO
) -> None:
    """Save a composite figure summarising detections and waterfalls with SNR analysis.

    Parameters
    ----------
    normalize : bool, optional
        If ``True``, apply per-channel normalization to ``waterfall_block`` and
        ``dedispersed_block`` before plotting, matching the behaviour of
        :func:`plot_waterfall_block`.
    off_regions : Optional[List[Tuple[int, int]]]
        Off-pulse regions for SNR calculation
    thresh_snr : Optional[float]
        SNR threshold for highlighting
    band_idx : int
        Band index for frequency range calculation
    """

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

    # DEBUG: Verificar configuración de plots
    if config.DEBUG_FREQUENCY_ORDER:
        print(f"🔍 [DEBUG PLOTS] Composite summary para: {fits_stem}")
        print(f"🔍 [DEBUG PLOTS] Band: {band_name_with_freq}")
        print(f"🔍 [DEBUG PLOTS] freq_ds shape: {freq_ds.shape}")
        print(f"🔍 [DEBUG PLOTS] freq_ds.min(): {freq_ds.min():.2f} MHz")
        print(f"🔍 [DEBUG PLOTS] freq_ds.max(): {freq_ds.max():.2f} MHz")
        print(f"🔍 [DEBUG PLOTS] waterfall_block shape: {waterfall_block.shape if waterfall_block is not None else 'None'}")
        print(f"🔍 [DEBUG PLOTS] dedispersed_block shape: {dedispersed_block.shape if dedispersed_block is not None else 'None'}")
        print(f"🔍 [DEBUG PLOTS] DM value: {dm_val:.2f} pc cm⁻³")
        print(f"🔍 [DEBUG PLOTS] imshow origin='lower' significa: freq_ds.min() en parte inferior, freq_ds.max() en parte superior")
        print(f"🔍 [DEBUG PLOTS] extent será: [tiempo_inicio, tiempo_fin, {freq_ds.min():.1f}, {freq_ds.max():.1f}]")
        print("🔍 [DEBUG PLOTS] " + "="*60)

    # Check if waterfall_block is valid
    if waterfall_block is not None and waterfall_block.size > 0:
        wf_block = waterfall_block.copy()
    else:
        wf_block = None
    
    # Check if dedispersed_block is valid
    if dedispersed_block is not None and dedispersed_block.size > 0:
        dw_block = dedispersed_block.copy()
    else:
        dw_block = None

    # DEBUG: Verificar datos de entrada
    if config.DEBUG_FREQUENCY_ORDER:
        print(f"🔍 [DEBUG DATOS] Entrada a save_slice_summary:")
        print(f"🔍 [DEBUG DATOS] waterfall_block válido: {wf_block is not None}")
        print(f"🔍 [DEBUG DATOS] dedispersed_block válido: {dw_block is not None}")
        if wf_block is not None and dw_block is not None:
            print(f"🔍 [DEBUG DATOS] ¿Son iguales raw y dedispersed? {np.array_equal(wf_block, dw_block)}")
            print(f"🔍 [DEBUG DATOS] Diferencia máxima: {np.max(np.abs(wf_block - dw_block)):.6f}")
        print("🔍 [DEBUG DATOS] " + "="*50)
    
    if normalize:
        for block in (wf_block, dw_block):
            if block is not None:
                block += 1
                block /= np.mean(block, axis=0)
                vmin, vmax = np.nanpercentile(block, [5, 95])
                block[:] = np.clip(block, vmin, vmax)
                block -= block.min()
                block /= block.max() - block.min()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 🕐 CALCULAR TIEMPOS ABSOLUTOS PARA TODO EL COMPOSITE
    if absolute_start_time is not None:
        slice_start_abs = absolute_start_time
    else:
        slice_start_abs = slice_idx * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
    
    slice_end_abs = slice_start_abs + slice_len * config.TIME_RESO * config.DOWN_TIME_RATE

    # Create figure with GridSpec for better control
    # 🎯 AJUSTE DE DIMENSIONES: Plot DM-tiempo más grande, waterfalls más pequeños
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[4, 3, 3], width_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)

    # === Panel 1: Detection Plot (DM vs Time) ===
    ax_det = fig.add_subplot(gs[0, :])
    
    if img_rgb is not None and img_rgb.size > 0:
        ax_det.imshow(img_rgb, origin="lower", aspect="auto", cmap="mako")
        
        # Time axis labels - USAR TIEMPO ABSOLUTO
        n_time_ticks = 6
        time_positions = np.linspace(0, img_rgb.shape[1] - 1, n_time_ticks)
        time_values = slice_start_abs + (time_positions / img_rgb.shape[1]) * (slice_end_abs - slice_start_abs)
        ax_det.set_xticks(time_positions)
        ax_det.set_xticklabels([f"{t:.3f}" for t in time_values])
        ax_det.set_xlabel("Time (s)", fontsize=10, fontweight="bold")

        # DM axis labels - USAR RANGO DM DINÁMICO
        n_dm_ticks = 8
        dm_positions = np.linspace(0, img_rgb.shape[0] - 1, n_dm_ticks)
        
        # Calcular rango DM dinámico basado en candidatos detectados
        dm_plot_min, dm_plot_max = _calculate_dynamic_dm_range(
            top_boxes=top_boxes,
            slice_len=slice_len,
            fallback_dm_min=config.DM_min,
            fallback_dm_max=config.DM_max,
            confidence_scores=top_conf if top_conf is not None else None
        )
        
        # Usar el rango dinámico para las etiquetas del eje DM
        dm_values = dm_plot_min + (dm_positions / img_rgb.shape[0]) * (dm_plot_max - dm_plot_min)
        ax_det.set_yticks(dm_positions)
        ax_det.set_yticklabels([f"{dm:.0f}" for dm in dm_values])
        ax_det.set_ylabel("Dispersion Measure (pc cm⁻³)", fontsize=10, fontweight="bold")

        # Bounding boxes con información completa - UNA SOLA ETIQUETA INTEGRADA
        if top_boxes is not None:
            for idx, (conf, box) in enumerate(zip(top_conf, top_boxes)):
                x1, y1, x2, y2 = map(int, box)
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                
                # ✅ CORRECCIÓN: Usar el DM REAL (mismo cálculo que pixel_to_physical)
                # Este es el DM que se usa en dedispersion y se guarda en CSV
                from .astro_conversions import pixel_to_physical
                dm_val_cand, t_sec_real, t_sample_real = pixel_to_physical(center_x, center_y, slice_len)
                
                # Determinar si tenemos probabilidades de clasificación
                if class_probs is not None and idx < len(class_probs):
                    class_prob = class_probs[idx]
                    is_burst = class_prob >= config.CLASS_PROB
                    color = "lime" if is_burst else "orange"
                    burst_status = "BURST" if is_burst else "NO BURST"
                    
                    # Etiqueta completa con toda la información - USANDO DM REAL
                    label = (
                        f"#{idx+1}\n"
                        f"DM: {dm_val_cand:.1f}\n"
                        f"Det: {conf:.2f}\n"
                        f"Cls: {class_prob:.2f}\n"
                        f"{burst_status}"
                    )
                else:
                    # Fallback si no hay probabilidades de clasificación
                    color = "lime"
                    label = f"#{idx+1}\nDM: {dm_val_cand:.1f}\nDet: {conf:.2f}"
                
                # Dibujar rectángulo
                rect = plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, 
                    linewidth=2, edgecolor=color, facecolor="none", alpha=0.8
                )
                ax_det.add_patch(rect)
                
                # Agregar etiqueta integrada
                ax_det.annotate(
                    label,
                    xy=(center_x, center_y),
                    xytext=(center_x, y2 + 15),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                    fontsize=8,
                    ha="center",
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=color, lw=1),
                )

        # Indicar si se está usando DM dinámico
        dm_range_info = f"{dm_plot_min:.0f}\u2013{dm_plot_max:.0f}"
        if getattr(config, 'DM_DYNAMIC_RANGE_ENABLE', True) and top_boxes is not None and len(top_boxes) > 0:
            dm_range_info += " (auto)"
        else:
            dm_range_info += " (full)"
            
        title = (
            f"Detection Plot: {fits_stem} - {band_name_with_freq} - Slice {slice_idx + 1} | "
            f"DM Range: {dm_range_info} pc cm⁻³ | "
            f"Time: {slice_start_abs:.3f}\u2013{slice_end_abs:.3f}s"
        )
        ax_det.set_title(title, fontsize=11, fontweight="bold")
        ax_det.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    else:
        ax_det.text(0.5, 0.5, 'No detection data available', 
                   transform=ax_det.transAxes, 
                   ha='center', va='center', fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        ax_det.set_xticks([])
        ax_det.set_yticks([])
        ax_det.set_title("No Detection Data", fontsize=11, fontweight="bold")

    # === Bottom Row: Waterfalls ===
    gs_bottom_row = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1:, :], wspace=0.3)

    # === Panel 1: Raw Waterfall con SNR ===
    gs_wf_nested = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_bottom_row[0, 0], height_ratios=[1, 4], hspace=0.05
    )
    ax_prof_wf = fig.add_subplot(gs_wf_nested[0, 0])
    
    # Verificar si hay datos de waterfall válidos
    if wf_block is not None and wf_block.size > 0:
        # Calcular perfil SNR para waterfall raw
        snr_wf, sigma_wf = compute_snr_profile(wf_block, off_regions)
        peak_snr_wf, peak_time_wf, peak_idx_wf = find_snr_peak(snr_wf)
        
        time_axis_wf = np.linspace(slice_start_abs, slice_end_abs, len(snr_wf))
        ax_prof_wf.plot(time_axis_wf, snr_wf, color="blue", alpha=0.8, lw=1.5, label='Raw SNR')
        
        # Resaltar regiones sobre threshold
        if thresh_snr is not None and config.SNR_SHOW_PEAK_LINES:
            above_thresh_wf = snr_wf >= thresh_snr
            if np.any(above_thresh_wf):
                ax_prof_wf.plot(time_axis_wf[above_thresh_wf], snr_wf[above_thresh_wf], 
                               color=config.SNR_HIGHLIGHT_COLOR, lw=2.5, alpha=0.9)
            ax_prof_wf.axhline(y=thresh_snr, color=config.SNR_HIGHLIGHT_COLOR, 
                              linestyle='--', alpha=0.7, linewidth=1)
        
        # Marcar pico
        ax_prof_wf.plot(time_axis_wf[peak_idx_wf], peak_snr_wf, 'ro', markersize=5)
        ax_prof_wf.text(time_axis_wf[peak_idx_wf], peak_snr_wf + 0.1 * (ax_prof_wf.get_ylim()[1] - ax_prof_wf.get_ylim()[0]), 
                       f'{peak_snr_wf:.1f}σ', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax_prof_wf.set_xlim(slice_start_abs, slice_end_abs)
        ax_prof_wf.set_ylabel('SNR (σ)', fontsize=8, fontweight="bold")
        ax_prof_wf.grid(True, alpha=0.3)
        ax_prof_wf.set_xticks([])
        ax_prof_wf.set_title(f"Raw SNR\nPeak={peak_snr_wf:.1f}σ", fontsize=9, fontweight="bold")
    else:
        ax_prof_wf.text(0.5, 0.5, 'No waterfall\ndata available', 
                       transform=ax_prof_wf.transAxes, 
                       ha='center', va='center', fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        ax_prof_wf.set_ylabel('SNR (σ)', fontsize=8, fontweight="bold")
        ax_prof_wf.grid(True, alpha=0.3)
        ax_prof_wf.set_xticks([])
        ax_prof_wf.set_title("No Waterfall Data", fontsize=9, fontweight="bold")

    ax_wf = fig.add_subplot(gs_wf_nested[1, 0])
    
    if wf_block is not None and wf_block.size > 0:
        # DEBUG: Verificar waterfall raw
        if config.DEBUG_FREQUENCY_ORDER:
            print(f"🔍 [DEBUG RAW WF] Raw waterfall shape: {wf_block.shape}")
            print(f"🔍 [DEBUG RAW WF] Transpose para imshow: {wf_block.T.shape}")
            print(f"🔍 [DEBUG RAW WF] .T[0, :] (primera freq) primeras 5 muestras: {wf_block.T[0, :5]}")
            print(f"🔍 [DEBUG RAW WF] .T[-1, :] (última freq) primeras 5 muestras: {wf_block.T[-1, :5]}")
        
        ax_wf.imshow(
            wf_block.T,
            origin="lower",
            cmap="viridis",
            aspect="auto",
            vmin=np.nanpercentile(wf_block, 1),
            vmax=np.nanpercentile(wf_block, 99),
            extent=[slice_start_abs, slice_end_abs, freq_ds.min(), freq_ds.max()],
        )
        ax_wf.set_xlim(slice_start_abs, slice_end_abs)
        ax_wf.set_ylim(freq_ds.min(), freq_ds.max())

        n_freq_ticks = 6
        freq_tick_positions = np.linspace(freq_ds.min(), freq_ds.max(), n_freq_ticks)
        n_time_ticks = 5
        time_tick_positions = np.linspace(slice_start_abs, slice_end_abs, n_time_ticks)
        ax_wf.set_xticks(time_tick_positions)
        ax_wf.set_xticklabels([f"{t:.3f}" for t in time_tick_positions])
        ax_wf.set_xlabel("Time (s)", fontsize=9)
        ax_wf.set_ylabel("Frequency (MHz)", fontsize=9)
        
        # Marcar posición del pico SNR en el waterfall
        if 'peak_snr_wf' in locals() and config.SNR_SHOW_PEAK_LINES:
            ax_wf.axvline(x=time_axis_wf[peak_idx_wf], color=config.SNR_HIGHLIGHT_COLOR, 
                         linestyle='-', alpha=0.8, linewidth=2)
    else:
        ax_wf.text(0.5, 0.5, 'No waterfall data available', 
                  transform=ax_wf.transAxes, 
                  ha='center', va='center', fontsize=12, 
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        ax_wf.set_xticks([])
        ax_wf.set_yticks([])
        ax_wf.set_xlabel("Time (s)", fontsize=9)
        ax_wf.set_ylabel("Frequency (MHz)", fontsize=9)

    # === Panel 2: Dedispersed Waterfall con SNR ===
    gs_dedisp_nested = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_bottom_row[0, 1], height_ratios=[1, 4], hspace=0.05
    )
    ax_prof_dw = fig.add_subplot(gs_dedisp_nested[0, 0])
    
    # Verificar si hay datos de waterfall dedispersado válidos
    if dw_block is not None and dw_block.size > 0:
        # Calcular perfil SNR para dedispersed waterfall
        snr_dw, sigma_dw = compute_snr_profile(dw_block, off_regions)
        peak_snr_dw, peak_time_dw, peak_idx_dw = find_snr_peak(snr_dw)
        
        time_axis_dw = np.linspace(slice_start_abs, slice_end_abs, len(snr_dw))
        ax_prof_dw.plot(time_axis_dw, snr_dw, color="green", alpha=0.8, lw=1.5, label='Dedispersed SNR')
        
        # Resaltar regiones sobre threshold
        if thresh_snr is not None and config.SNR_SHOW_PEAK_LINES:
            above_thresh_dw = snr_dw >= thresh_snr
            if np.any(above_thresh_dw):
                ax_prof_dw.plot(time_axis_dw[above_thresh_dw], snr_dw[above_thresh_dw], 
                               color=config.SNR_HIGHLIGHT_COLOR, lw=2.5, alpha=0.9)
            ax_prof_dw.axhline(y=thresh_snr, color=config.SNR_HIGHLIGHT_COLOR, 
                              linestyle='--', alpha=0.7, linewidth=1)
        
        # Marcar pico
        ax_prof_dw.plot(time_axis_dw[peak_idx_dw], peak_snr_dw, 'ro', markersize=5)
        ax_prof_dw.text(time_axis_dw[peak_idx_dw], peak_snr_dw + 0.1 * (ax_prof_dw.get_ylim()[1] - ax_prof_dw.get_ylim()[0]), 
                       f'{peak_snr_dw:.1f}σ', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax_prof_dw.set_xlim(slice_start_abs, slice_end_abs)
        ax_prof_dw.set_ylabel('SNR (σ)', fontsize=8, fontweight="bold")
        ax_prof_dw.grid(True, alpha=0.3)
        ax_prof_dw.set_xticks([])
        ax_prof_dw.set_title(f"Dedispersed SNR DM={dm_val:.2f} pc cm⁻³\nPeak={peak_snr_dw:.1f}σ", fontsize=9, fontweight="bold")
    else:
        ax_prof_dw.text(0.5, 0.5, 'No dedispersed\ndata available', 
                       transform=ax_prof_dw.transAxes, 
                       ha='center', va='center', fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        ax_prof_dw.set_ylabel('SNR (σ)', fontsize=8, fontweight="bold")
        ax_prof_dw.grid(True, alpha=0.3)
        ax_prof_dw.set_xticks([])
        ax_prof_dw.set_title("No Dedispersed Data", fontsize=9, fontweight="bold")

    ax_dw = fig.add_subplot(gs_dedisp_nested[1, 0])
    
    if dw_block is not None and dw_block.size > 0:
        # DEBUG: Verificar dedispersed waterfall
        if config.DEBUG_FREQUENCY_ORDER:
            print(f"🔍 [DEBUG DED WF] Dedispersed waterfall shape: {dw_block.shape}")
            print(f"🔍 [DEBUG DED WF] Transpose para imshow: {dw_block.T.shape}")
            print(f"🔍 [DEBUG DED WF] .T[0, :] (primera freq) primeras 5 muestras: {dw_block.T[0, :5]}")
            print(f"🔍 [DEBUG DED WF] .T[-1, :] (última freq) primeras 5 muestras: {dw_block.T[-1, :5]}")
            print(f"🔍 [DEBUG DED WF] ¿Es diferente al raw? Diff promedio: {np.mean(np.abs(dw_block - wf_block)) if wf_block is not None else 'N/A'}")
        
        ax_dw.imshow(
            dw_block.T,
            origin="lower",
            cmap="viridis",
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
        ax_dw.set_xticklabels([f"{t:.3f}" for t in time_tick_positions])
        ax_dw.set_xlabel("Time (s)", fontsize=9)
        ax_dw.set_ylabel("Frequency (MHz)", fontsize=9)
        
        # Marcar posición del pico SNR en el waterfall dedispersado
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

    # === Panel 3: Candidate Patch (AHORA USANDO DEDISPERSED WATERFALL) ===
    gs_patch_nested = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_bottom_row[0, 2], height_ratios=[1, 4], hspace=0.05
    )
    ax_patch_prof = fig.add_subplot(gs_patch_nested[0, 0])
    
    # 🎯 CAMBIO PRINCIPAL: Usar el waterfall dedispersado como candidate patch
    # Esto centraliza el candidato en el medio de la imagen
    candidate_patch = dw_block if dw_block is not None and dw_block.size > 0 else wf_block
    
    # Verificar si hay un patch válido
    if candidate_patch is not None and candidate_patch.size > 0:
        # Calcular perfil SNR para el patch del candidato (dedispersed waterfall)
        snr_patch, sigma_patch = compute_snr_profile(candidate_patch, off_regions)
        peak_snr_patch, peak_time_patch, peak_idx_patch = find_snr_peak(snr_patch)
        
        patch_time_axis = np.linspace(slice_start_abs, slice_end_abs, len(snr_patch))
        ax_patch_prof.plot(patch_time_axis, snr_patch, color="orange", alpha=0.8, lw=1.5, label='Candidate SNR')
        
        # Resaltar regiones sobre threshold
        if thresh_snr is not None and config.SNR_SHOW_PEAK_LINES:
            above_thresh_patch = snr_patch >= thresh_snr
            if np.any(above_thresh_patch):
                ax_patch_prof.plot(patch_time_axis[above_thresh_patch], snr_patch[above_thresh_patch], 
                                color=config.SNR_HIGHLIGHT_COLOR, lw=2, alpha=0.9)
            ax_patch_prof.axhline(y=thresh_snr, color=config.SNR_HIGHLIGHT_COLOR, 
                                 linestyle='--', alpha=0.7, linewidth=1)
        
        # Marcar pico
        ax_patch_prof.plot(patch_time_axis[peak_idx_patch], peak_snr_patch, 'ro', markersize=5)
        ax_patch_prof.text(patch_time_axis[peak_idx_patch], peak_snr_patch + 0.1 * (ax_patch_prof.get_ylim()[1] - ax_patch_prof.get_ylim()[0]), 
                          f'{peak_snr_patch:.1f}σ', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax_patch_prof.set_xlim(patch_time_axis[0], patch_time_axis[-1])
        ax_patch_prof.set_ylabel('SNR (σ)', fontsize=8, fontweight="bold")
        ax_patch_prof.grid(True, alpha=0.3)
        ax_patch_prof.set_xticks([])
        ax_patch_prof.set_title(f"Candidate Patch SNR (Dedispersed)\nPeak={peak_snr_patch:.1f}σ", fontsize=9, fontweight="bold")
    else:
        # Sin patch válido, mostrar mensaje
        ax_patch_prof.text(0.5, 0.5, 'No candidate patch\navailable', 
                          transform=ax_patch_prof.transAxes, 
                          ha='center', va='center', fontsize=10, 
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        ax_patch_prof.set_ylabel('SNR (σ)', fontsize=8, fontweight="bold")
        ax_patch_prof.grid(True, alpha=0.3)
        ax_patch_prof.set_xticks([])
        ax_patch_prof.set_title("No Candidate Patch", fontsize=9, fontweight="bold")

    ax_patch = fig.add_subplot(gs_patch_nested[1, 0])
    
    if candidate_patch is not None and candidate_patch.size > 0:
        # DEBUG: Verificar candidate patch (ahora dedispersed waterfall)
        if config.DEBUG_FREQUENCY_ORDER:
            print(f"🔍 [DEBUG PATCH] Candidate patch (dedispersed) shape: {candidate_patch.shape}")
            print(f"🔍 [DEBUG PATCH] Transpose para imshow: {candidate_patch.T.shape}")
            print(f"🔍 [DEBUG PATCH] .T[0, :] (primera freq) primeras 5 muestras: {candidate_patch.T[0, :5]}")
            print(f"🔍 [DEBUG PATCH] .T[-1, :] (última freq) primeras 5 muestras: {candidate_patch.T[-1, :5]}")
        
        ax_patch.imshow(
            candidate_patch.T,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=np.nanpercentile(candidate_patch, 1),
            vmax=np.nanpercentile(candidate_patch, 99),
            extent=[slice_start_abs, slice_end_abs, freq_ds.min(), freq_ds.max()],
        )
        ax_patch.set_xlim(slice_start_abs, slice_end_abs)
        ax_patch.set_ylim(freq_ds.min(), freq_ds.max())

        n_patch_time_ticks = 5
        patch_tick_positions = np.linspace(slice_start_abs, slice_end_abs, n_patch_time_ticks)
        ax_patch.set_xticks(patch_tick_positions)
        ax_patch.set_xticklabels([f"{t:.3f}" for t in patch_tick_positions])

        ax_patch.set_yticks(freq_tick_positions)
        ax_patch.set_yticklabels([f"{f:.0f}" for f in freq_tick_positions])
        ax_patch.set_xlabel("Time (s)", fontsize=9)
        ax_patch.set_ylabel("Frequency (MHz)", fontsize=9)
        
        # Marcar posición del pico SNR en el patch
        if 'peak_snr_patch' in locals() and config.SNR_SHOW_PEAK_LINES:
            ax_patch.axvline(x=patch_time_axis[peak_idx_patch], color=config.SNR_HIGHLIGHT_COLOR, 
                           linestyle='-', alpha=0.8, linewidth=2)
    else:
        # Mostrar mensaje de que no hay patch
        ax_patch.text(0.5, 0.5, 'No candidate patch available', 
                     transform=ax_patch.transAxes, 
                     ha='center', va='center', fontsize=12, 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        ax_patch.set_xticks([])
        ax_patch.set_yticks([])
        ax_patch.set_xlabel("Time (s)", fontsize=9)
        ax_patch.set_ylabel("Frequency (MHz)", fontsize=9)

    # 🚀 OPTIMIZACIÓN DE MEMORIA: Usar layout manual para evitar warnings
    # Evitar tight_layout que puede causar problemas con Axes complejas
    plt.subplots_adjust(
        left=0.08, right=0.92, 
        top=0.92, bottom=0.08, 
        hspace=0.25, wspace=0.25
    )
    
    fig.suptitle(
        f"Composite Summary: {fits_stem} - {band_name_with_freq} - Slice {slice_idx + 1}",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )
    
    # 🚀 OPTIMIZACIÓN: Reducir DPI para ahorrar memoria (sin parámetros incompatibles)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    
    # 🧹 LIMPIEZA EXPLÍCITA DE MEMORIA
    import gc
    gc.collect()


def plot_waterfalls(
    data: np.ndarray,
    slice_len: int,
    time_slice: int,
    fits_stem: str,
    out_dir: Path,
    absolute_start_time: Optional[float] = None,
) -> None:
    """Save frequency--time waterfall plots for each time block.
    
    Parameters
    ----------
    data : np.ndarray
        Data array to process
    slice_len : int
        Length of each slice
    time_slice : int
        Number of slices
    fits_stem : str
        Base filename
    out_dir : Path
        Output directory
    absolute_start_time : Optional[float]
        Absolute start time in seconds. If provided, plots will show real file times.
    """

    freq_ds = np.mean(
        config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE),
        axis=1,
    )
    time_reso_ds = config.TIME_RESO * config.DOWN_TIME_RATE
    out_dir.mkdir(parents=True, exist_ok=True)

    for j in range(time_slice):
        t0, t1 = j * slice_len, (j + 1) * slice_len
        block = data[t0:t1]
        if block.size == 0:
            continue
        plot_waterfall_block(
            data_block=block,
            freq=freq_ds,
            time_reso=time_reso_ds,
            block_size=block.shape[0],
            block_idx=j,
            save_dir=out_dir,
            filename=fits_stem,
            normalize=True,
            absolute_start_time=absolute_start_time,
        )


def plot_dedispersed_waterfalls(
    data: np.ndarray,
    freq_down: np.ndarray,
    dm: float,
    slice_len: int,
    time_slice: int,
    fits_stem: str,
    out_dir: Path,
    absolute_start_time: Optional[float] = None,
) -> None:
    """Save dedispersed frequency--time waterfall plots for each time block.
    
    Parameters
    ----------
    data : np.ndarray
        Data array to process
    freq_down : np.ndarray
        Downsampled frequency array
    dm : float
        Dispersion measure value
    slice_len : int
        Length of each slice
    time_slice : int
        Number of slices
    fits_stem : str
        Base filename
    out_dir : Path
        Output directory
    absolute_start_time : Optional[float]
        Absolute start time in seconds. If provided, plots will show real file times.
    """

    time_reso_ds = config.TIME_RESO * config.DOWN_TIME_RATE
    out_dir.mkdir(parents=True, exist_ok=True)

    for j in range(time_slice):
        start = j * slice_len
        block = dedisperse_block(data, freq_down, dm, start, slice_len)
        if block.size == 0:
            continue
        plot_waterfall_block(
            data_block=block,
            freq=freq_down,
            time_reso=time_reso_ds,
            block_size=block.shape[0],
            block_idx=j,
            save_dir=out_dir,
            filename=f"{fits_stem}_dm{dm:.2f}",
            normalize=True,
            absolute_start_time=absolute_start_time,
        )

