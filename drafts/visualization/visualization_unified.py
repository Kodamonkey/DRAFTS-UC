"""Unified visualization module for FRB pipeline."""
from __future__ import annotations

# Standard library imports
import logging
from pathlib import Path
from typing import Iterable, Optional, List, Tuple

# Third-party imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import ListedColormap

# Local imports
from .. import config
from ..analysis.snr_utils import compute_snr_profile, find_snr_peak
from ..preprocessing.dm_candidate_extractor import extract_candidate_dm
from ..preprocessing.dedispersion import dedisperse_block
from .visualization_ranges import get_dynamic_dm_range_for_candidate

# Setup logger
logger = logging.getLogger(__name__)

# Register custom colormap if not already registered
if "mako" not in plt.colormaps():
    plt.register_cmap(
        name="mako",
        cmap=ListedColormap(sns.color_palette("mako", as_cmap=True)(np.linspace(0, 1, 256)))
    )

def _calculate_dynamic_dm_range(
    top_boxes: Iterable | None,
    slice_len: int,
    fallback_dm_min: int = None,
    fallback_dm_max: int = None,
    confidence_scores: Iterable | None = None
) -> Tuple[float, float]:
    """Delegado unificado: usa visualization_ranges para rango DM dinámico."""
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
        print(f"[WARNING] Error calculando rango DM dinámico: {e}")
        dm_min = fallback_dm_min if fallback_dm_min is not None else config.DM_min
        dm_max = fallback_dm_max if fallback_dm_max is not None else config.DM_max
        return float(dm_min), float(dm_max)


def preprocess_img(img: np.ndarray) -> np.ndarray:
    img = (img - img.min()) / np.ptp(img)
    img = (img - img.mean()) / img.std()
    img = cv2.resize(img, (512, 512))
    img = np.clip(img, *np.percentile(img, (0.1, 99.9)))
    img = (img - img.min()) / np.ptp(img)
    img = plt.get_cmap("mako")(img)[..., :3]
    img -= [0.485, 0.456, 0.406]
    img /= [0.229, 0.224, 0.225]
    return img.transpose(2, 0, 1)


def postprocess_img(img_tensor: np.ndarray) -> np.ndarray:
    img = img_tensor.transpose(1, 2, 0)
    img *= [0.229, 0.224, 0.225]
    img += [0.485, 0.456, 0.406]
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def plot_waterfall_block(
    data_block: np.ndarray,
    freq: np.ndarray,
    time_reso: float,
    block_size: int,
    block_idx: int,
    save_dir: Path,
    filename: str,
    normalize: bool = False,
    absolute_start_time: float = None,  # Tiempo absoluto de inicio del bloque
    integrate_ts: bool = True,
    integrate_spec: bool = True,
) -> Path:
    """Plot a single waterfall block.

    Parameters
    ----------
    data_block : np.ndarray
        Frequency--time slice to plot.
    freq : np.ndarray
        Frequency axis in MHz.
    time_reso : float
        Time resolution of ``data_block`` in seconds.
    block_size : int
        Number of time samples in ``data_block``.
    block_idx : int
        Index of the block within the full observation.
    save_dir : Path
        Directory where the image will be saved.
    filename : str
        Base filename for the output image.
    normalize : bool, optional
        If ``True``, each frequency channel is scaled to unit mean and clipped
        between the 5th and 95th percentiles prior to plotting. This keeps the
        dynamic range consistent across different ``SLICE_LEN`` and DM ranges.
    absolute_start_time : float, optional
        Tiempo absoluto de inicio del bloque en segundos. Si se proporciona,
        se usa en lugar del cálculo relativo para mostrar tiempos reales del archivo.
    """

    block = data_block.copy() if normalize else data_block
    if normalize:
        # Normalización estilo PRESTO: restar mediana por canal y dividir por std global
        medians = np.median(block, axis=0, keepdims=True)
        block = block - medians
        std_global = float(np.std(block))
        if std_global > 0:
            block = block / std_global

    # Perfil temporal (integrado en frecuencia)
    profile = block.mean(axis=1)

    # Tiempo absoluto de inicio como en PRESTO (start)
    if absolute_start_time is not None:
        time_start = absolute_start_time
    else:
        time_start = block_idx * block_size * time_reso

    # Tiempo final del bloque
    time_end = time_start + block_size * time_reso

    # Figura al estilo PRESTO: imagen principal + time series arriba, spec a la derecha
    im_width = 0.6 if integrate_spec else 0.8
    im_height = 0.6 if integrate_ts else 0.8
    fig = plt.figure(figsize=(6, 5))
    ax_im = plt.axes((0.15, 0.15, im_width, im_height))
    ax_ts = plt.axes((0.15, 0.78, im_width, 0.18), sharex=ax_im) if integrate_ts else None
    ax_spec = plt.axes((0.78, 0.15, 0.17, im_height), sharey=ax_im) if integrate_spec else None

    # Imagen con extent centrado en muestras: usar centros temporales
    cmap_name = getattr(config, 'WATERFALL_CMAP', 'mako')
    origin_mode = getattr(config, 'WATERFALL_ORIGIN', 'lower')
    img = ax_im.imshow(
        block.T,
        origin=origin_mode,
        cmap=cmap_name,
        aspect="auto",
        # DEJAR QUE MATPLOTLIB ESCALA A PARTIR DE LOS VALORES NORMALIZADOS
        extent=(time_start, time_end, float(freq.min()), float(freq.max())),
        interpolation="nearest",
    )
    ax_im.set_xlabel("Time", fontsize=11)
    ax_im.set_ylabel("Observing frequency (MHz)", fontsize=11)

    # Time series integrada (como integrate_ts de PRESTO)
    if integrate_ts and ax_ts is not None:
        times = time_start + (np.arange(block_size) + 0.5) * time_reso
        ax_ts.plot(times, block.sum(axis=1), "k", lw=0.8)
        ax_ts.set_xlim([times.min(), times.max()])
        plt.setp(ax_ts.get_xticklabels(), visible=False)
        plt.setp(ax_ts.get_yticklabels(), visible=False)

    # Espectro integrado alrededor del centro (ventana 10% de la duración)
    if integrate_spec and ax_spec is not None:
        nbinlim = block_size
        window_width = max(1, int(0.05 * nbinlim))
        burst_bin = nbinlim // 2
        on_spec = block[burst_bin - window_width: burst_bin + window_width, :]
        Dedisp_spec = on_spec.sum(axis=0)
        ax_spec.plot(Dedisp_spec[::-1], np.linspace(freq.min(), freq.max(), len(Dedisp_spec)), "k", lw=0.8)
        plt.setp(ax_spec.get_xticklabels(), visible=False)
        plt.setp(ax_spec.get_yticklabels(), visible=False)
        ax_spec.set_ylim([float(freq.min()), float(freq.max())])

    peak_time = time_start + np.argmax(profile) * time_reso
    # Añadir anotación de auditoría en el título
    idx_start_ds = int(round((time_start) / time_reso))
    idx_end_ds = idx_start_ds + block_size - 1
    plt.suptitle(
        f"{filename} | start={time_start:.6f}s end={time_end:.6f}s Δt={time_reso:.9f}s | [idx {idx_start_ds}→{idx_end_ds}]",
        fontsize=10
    )
    out_path = save_dir / f"{filename}-block{block_idx:03d}-peak{peak_time:.2f}.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    return out_path


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
    absolute_start_time: Optional[float] = None,  
    slice_samples: Optional[int] = None,
    candidate_times_abs: Optional[Iterable[float]] = None,
) -> None:
    """Save detection plot with both detection and classification probabilities."""

    # Usar slice_len específico o del config
    if slice_len is None:
        slice_len = config.SLICE_LEN
    # Determinar muestras reales del slice (último slice puede ser más corto)
    if slice_samples is None:
        slice_samples = slice_len

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(img_rgb, origin="lower", aspect="auto")

    # Time axis labels
    n_time_ticks = 6
    time_positions = np.linspace(0, 512, n_time_ticks)

    if absolute_start_time is not None:
        time_start_slice = absolute_start_time
        duration_samples = slice_samples if slice_samples is not None else slice_len
    else:
        time_start_slice = slice_idx * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
        duration_samples = slice_len
    
    # Duración real del slice (respetando el último slice truncado)
    slice_duration_sec = duration_samples * config.TIME_RESO * config.DOWN_TIME_RATE
    # Usar centros de píxel para coherencia con el Composite
    time_values = time_start_slice + ((time_positions + 0.5) / 512.0) * slice_duration_sec
    ax.set_xticks(time_positions)
    ax.set_xticklabels([f"{t:.6f}" for t in time_values])
    ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")

    # DM axis labels - AQUÍ ESTÁ LA INTEGRACIÓN DEL DM DINÁMICO
    n_dm_ticks = 8
    dm_positions = np.linspace(0, 512, n_dm_ticks)
    
    # Calcular rango DM dinámico basado en candidatos detectados
    dm_plot_min, dm_plot_max = _calculate_dynamic_dm_range(
        top_boxes=top_boxes,
        slice_len=slice_len,
        fallback_dm_min=config.DM_min,
        fallback_dm_max=config.DM_max,
        confidence_scores=top_conf if top_conf is not None else None
    )
    
    # Usar el rango dinámico para las etiquetas del eje DM
    dm_values = dm_plot_min + (dm_positions / 512.0) * (dm_plot_max - dm_plot_min)
    ax.set_yticks(dm_positions)
    ax.set_yticklabels([f"{dm:.0f}" for dm in dm_values])
    ax.set_ylabel("Dispersion Measure (pc cm⁻³)", fontsize=12, fontweight="bold")
    
    freq_min, freq_max = get_band_frequency_range(band_idx)
    freq_range = f"{freq_min:.{config.PLOT_FREQ_PRECISION}f}\u2013{freq_max:.{config.PLOT_FREQ_PRECISION}f} MHz"
        
    # Indicar si se está usando DM dinámico
    dm_range_info = f"{dm_plot_min:.{config.PLOT_DM_PRECISION}f}\u2013{dm_plot_max:.{config.PLOT_DM_PRECISION}f}"
    if getattr(config, 'DM_DYNAMIC_RANGE_ENABLE', True) and top_boxes is not None and len(top_boxes) > 0:
        dm_range_info += " (auto)"
    else:
        dm_range_info += " (full)"
        
    # Mostrar duración exacta del slice en el título (ms con alta precisión)
    exact_slice_ms = slice_duration_sec * 1000.0
    title = (
        f"{fits_stem} - {band_name} ({freq_range})\n"
        f"Slice {slice_idx:03d}/{time_slice} | "
        f"Slice duration: {exact_slice_ms:.{config.PLOT_TIME_PRECISION}f} ms | "
        f"Time Resolution: {config.TIME_RESO * config.DOWN_TIME_RATE * 1e6:.{config.PLOT_TIME_PRECISION}f} \u03bcs | "
        f"DM Range: {dm_range_info} pc cm⁻\u00b3"
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=20)

    # Detection info
    if top_boxes is not None and len(top_boxes) > 0:
        detection_info = f"Detections: {len(top_boxes)}"
        ax.text(
            0.02,
            0.98,
            detection_info,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            fontsize=10,
            verticalalignment="top",
            fontweight="bold",
        )
    else:
        # Mostrar información de debug cuando no hay detecciones
        debug_info = "No detections found (debug mode)"
        ax.text(
            0.02,
            0.98,
            debug_info,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
            fontsize=10,
            verticalalignment="top",
            fontweight="bold",
        )

    tech_info = (
        f"Model: {config.MODEL_NAME.upper()}\n"
        f"Confidence: {det_prob:.1f}\n"
        f"Channels: {config.FREQ_RESO}\u2192{config.FREQ_RESO // config.DOWN_FREQ_RATE}\n"
        f"Time samples: {config.FILE_LENG}\u2192{config.FILE_LENG // config.DOWN_TIME_RATE}"
    )
    ax.text(
        0.98,
        0.02,
        tech_info,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    # Bounding boxes - UNA SOLA ETIQUETA INTEGRADA
    if top_boxes is not None:
        for idx, (conf, box) in enumerate(zip(top_conf, top_boxes)):
            x1, y1, x2, y2 = map(int, box)
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Usar el DM REAL (mismo cálculo que extract_candidate_dm)
            # Este es el DM que se usa en dedispersion y se guarda en CSV
            from ..preprocessing.dm_candidate_extractor import extract_candidate_dm
            # Usar el número REAL de muestras del slice para mapear posición X a tiempo
            effective_len = slice_samples if slice_samples is not None else slice_len
            dm_val_real, t_sec_real, t_sample_real = extract_candidate_dm(center_x, center_y, effective_len)
            
            # UNIFICACIÓN: Usar SOLO candidate_times_abs para consistencia
            if candidate_times_abs is not None and idx < len(candidate_times_abs):
                detection_time = float(candidate_times_abs[idx])
            else:
                # UNIFICACIÓN: NO usar fallbacks inconsistentes
                if candidate_times_abs is None:
                    logger.warning(f"[DETECTION] candidate_times_abs es None para candidato {idx}")
                elif idx >= len(candidate_times_abs):
                    logger.warning(f"[DETECTION] Índice {idx} fuera de rango para candidate_times_abs (len={len(candidate_times_abs)})")
                else:
                    logger.warning(f"[DETECTION] candidate_times_abs[{idx}] no es válido")
                detection_time = None
            
            # Determinar si tenemos probabilidades de clasificación
            if class_probs is not None and idx < len(class_probs):
                class_prob = class_probs[idx]
                is_burst = class_prob >= config.CLASS_PROB
                color = "lime" if is_burst else "orange"
                burst_status = "BURST" if is_burst else "NO BURST"
                
                # Etiqueta completa con tiempo absoluto - USANDO DM REAL
                if detection_time is not None:
                    label = (
                        f"#{idx+1}\n"
                        f"DM: {dm_val_real:.{config.PLOT_DM_PRECISION}f}\n"
                        f"Time: {detection_time:.{config.PLOT_TIME_PRECISION}f}s\n"
                        f"Det: {conf:.2f}\n"
                        f"Cls: {class_prob:.2f}\n"
                        f"{burst_status}"
                    )
                else:
                    label = (
                        f"#{idx+1}\n"
                        f"DM: {dm_val_real:.{config.PLOT_DM_PRECISION}f}\n"
                        f"Time: N/A\n"
                        f"Det: {conf:.2f}\n"
                        f"Cls: {class_prob:.2f}\n"
                        f"{burst_status}"
                    )
            else:
                # Fallback si no hay probabilidades de clasificación
                color = "lime"
                if detection_time is not None:
                    label = f"#{idx+1}\nDM: {dm_val_real:.{config.PLOT_DM_PRECISION}f}\nTime: {detection_time:.{config.PLOT_TIME_PRECISION}f}s\nDet: {conf:.2f}"
                else:
                    label = f"#{idx+1}\nDM: {dm_val_real:.{config.PLOT_DM_PRECISION}f}\nTime: N/A\nDet: {conf:.2f}"
            
            # Dibujar rectángulo
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, 
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            
            # Agregar etiqueta integrada
            ax.annotate(
                label,
                xy=(center_x, center_y),
                xytext=(center_x, y2 + 15),
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                fontsize=8,
                ha="center",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color, lw=1),
            )

    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    plt.savefig(out_img_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()

    if band_suffix == "fullband":
        fig_cb, ax_cb = plt.subplots(figsize=(13, 8))
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        im_cb = ax_cb.imshow(img_gray, origin="lower", aspect="auto", cmap="mako")
        ax_cb.set_xticks(time_positions)
        ax_cb.set_xticklabels([f"{t:.{config.PLOT_TIME_PRECISION}f}s" for t in time_values])
        ax_cb.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax_cb.set_yticks(dm_positions)
        ax_cb.set_yticklabels([f"{dm:.{config.PLOT_DM_PRECISION}f}" for dm in dm_values])
        ax_cb.set_ylabel("Dispersion Measure (pc cm⁻³)", fontsize=12, fontweight="bold")
        ax_cb.set_title(title, fontsize=11, fontweight="bold", pad=20)
        if top_boxes is not None:
            for box in top_boxes:
                x1, y1, x2, y2 = map(int, box)
                rect = plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="cyan", facecolor="none"
                )
                ax_cb.add_patch(rect)
        cbar = plt.colorbar(im_cb, ax=ax_cb, shrink=0.8, pad=0.02)
        cbar.set_label("Normalized Intensity", fontsize=10, fontweight="bold")
        ax_cb.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        cb_path = out_img_path.parent / f"{out_img_path.stem}_colorbar{out_img_path.suffix}"
        plt.savefig(cb_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close()

def save_all_plots(
    waterfall_block,
    dedisp_block,
    img_rgb,
    first_patch,
    first_start,
    first_dm,
    top_conf,
    top_boxes,
    class_probs_list,
    comp_path,
    j,
    time_slice,
    band_name,
    band_suffix,
    fits_stem,
    slice_len,
    normalize,
    off_regions,
    thresh_snr,
    band_idx,
    patch_path,
    waterfall_dedispersion_dir,
    freq_down,
    time_reso_ds,
    detections_dir,
    out_img_path,
    absolute_start_time=None,
    chunk_idx=None,  # PARÁMETRO PARA CHUNK
    force_plots: bool = False,
    candidate_times_abs: Optional[Iterable[float]] = None,
    plot_context: Optional[dict] = None,
):
    """Guarda todos los plots con tiempo absoluto para continuidad temporal.
    
    Args:
        absolute_start_time: Tiempo absoluto de inicio del slice en segundos desde el inicio del archivo
        force_plots: Si True, genera todos los plots independientemente de si hay candidatos
    """
    # Crear carpetas solo cuando se van a generar plots
    # Composite plot - crear carpeta solo si se va a generar
    if comp_path is not None:
        comp_path.parent.mkdir(parents=True, exist_ok=True)
        # Calcular muestras reales del slice (por si el último slice está truncado)
        real_slice_samples = (
            waterfall_block.shape[0]
            if waterfall_block is not None and hasattr(waterfall_block, "shape")
            else slice_len
        )
        # (Quitado ajuste extra PRESTO en visual: ya se calcula en detección si aplica)
        presto_dt = 0.0

        save_slice_summary(
            waterfall_block,
            dedisp_block if dedisp_block is not None and dedisp_block.size > 0 else waterfall_block,
            img_rgb,
            first_patch,
            first_start if first_start is not None else 0.0,
            first_dm if first_dm is not None else 0.0,
            top_conf if len(top_conf) > 0 else [],
            top_boxes if len(top_boxes) > 0 else [],
            class_probs_list,
            comp_path,
            j,
            time_slice,
            band_name,
            band_suffix,
            fits_stem,
            slice_len,
            normalize=normalize,
            off_regions=off_regions,
            thresh_snr=thresh_snr,
            band_idx=band_idx,
            absolute_start_time=absolute_start_time,  
            chunk_idx=chunk_idx,  
            slice_samples=real_slice_samples,
            candidate_times_abs=candidate_times_abs,
            plot_context=plot_context,
        )
    
    # Patch plot - crear carpeta solo si hay patch o si se fuerza en modo debug
    if patch_path is not None and (first_patch is not None or force_plots):
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        # Si force_plots=True pero no hay patch, crear un patch vacío para debug
        if first_patch is not None:
            patch_data = first_patch
        elif force_plots:
            # Crear un patch de ejemplo para debug cuando no hay candidatos
            patch_data = np.zeros((10, 10)) if waterfall_block is None else waterfall_block[:10, :10]
        else:
            patch_data = np.zeros((10, 10))
        
        save_patch_plot(
            patch_data,
            patch_path,
            freq_down,
            time_reso_ds,
            first_start if first_start is not None else 0.0,
            off_regions=off_regions,
            thresh_snr=thresh_snr,
            band_idx=band_idx,
            band_name=band_name,
        )
    
    # Waterfall dedispersed - crear carpeta solo si hay datos dedispersados o si se fuerza
    if (dedisp_block is not None and dedisp_block.size > 0) or force_plots:
        waterfall_dedispersion_dir.mkdir(parents=True, exist_ok=True)
        # Si force_plots=True pero no hay dedisp_block, usar waterfall_block
        plot_data = dedisp_block if dedisp_block is not None and dedisp_block.size > 0 else waterfall_block
        if plot_data is not None and plot_data.size > 0:
            wf_dedisp_path = plot_waterfall_block(
                data_block=plot_data,
                freq=freq_down,
                time_reso=time_reso_ds,
                block_size=plot_data.shape[0],
                block_idx=j,
                save_dir=waterfall_dedispersion_dir,
                filename=f"{fits_stem}_dm{first_dm:.2f}_{band_suffix}" if first_dm is not None else f"{fits_stem}_dm0.00_{band_suffix}",
                normalize=normalize,
                absolute_start_time=absolute_start_time,  
                integrate_ts=True,
                integrate_spec=True,
            )
            # Adjuntar ruta al contexto para auditoría
            if plot_context is not None:
                plot_context['waterfall_dedispersion_path'] = str(wf_dedisp_path)
    
    # Detections plot - crear carpeta solo si se va a generar
    if out_img_path is not None:
        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        # Aplicar el mismo ajuste PRESTO en detections plot
        save_plot(
            img_rgb,
            top_conf if len(top_conf) > 0 else [],
            top_boxes if len(top_boxes) > 0 else [],
            class_probs_list,
            out_img_path,
            j,
            time_slice,
            band_name,
            band_suffix,
            fits_stem,
            slice_len,
            band_idx=band_idx,
            absolute_start_time=absolute_start_time,
            slice_samples=(
                waterfall_block.shape[0]
                if waterfall_block is not None and hasattr(waterfall_block, "shape")
                else slice_len
            ),
            candidate_times_abs=candidate_times_abs,
        )

def validate_candidate_times(candidate_times_abs: Optional[Iterable[float]], 
                           top_conf: Iterable, 
                           top_boxes: Iterable | None) -> Tuple[bool, str]:
    """
    Valida que candidate_times_abs sea consistente y válido.
    
    Args:
        candidate_times_abs: Lista de tiempos absolutos de candidatos
        top_conf: Lista de confianzas de detección
        top_boxes: Lista de bounding boxes
        
    Returns:
        Tuple[is_valid, error_message]
    """
    if candidate_times_abs is None:
        return False, "candidate_times_abs es None"
    
    if len(candidate_times_abs) == 0:
        return False, "candidate_times_abs está vacío"
    
    if len(candidate_times_abs) != len(top_conf):
        return False, f"candidate_times_abs ({len(candidate_times_abs)}) no coincide con top_conf ({len(top_conf)})"
    
    if top_boxes is not None and len(candidate_times_abs) != len(top_boxes):
        return False, f"candidate_times_abs ({len(candidate_times_abs)}) no coincide con top_boxes ({len(top_boxes)})"
    
    # Verificar que todos los tiempos sean números válidos
    for i, time in enumerate(candidate_times_abs):
        if not isinstance(time, (int, float)) or np.isnan(time) or np.isinf(time):
            return False, f"candidate_times_abs[{i}] = {time} no es un número válido"
    
    return True, "OK"


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
    absolute_start_time: Optional[float] = None,  # TIEMPO ABSOLUTO
    slice_samples: Optional[int] = None,  # muestras reales en el slice
    candidate_times_abs: Optional[Iterable[float]] = None,
) -> None:
    """Wrapper fino sin mutar estado global; pasa slice_len explícito."""
    
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
        slice_len=slice_len,
        band_idx=band_idx,
        absolute_start_time=absolute_start_time, 
        slice_samples=slice_samples,
        candidate_times_abs=candidate_times_abs,
    )
    # No modificar config.SLICE_LEN global


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
    detection_time: Optional[float] = None,  # Tiempo exacto de detección del candidato
) -> None:
    """Save a visualization of the classification patch with SNR profile.
    
    FUNCIONALIDAD ESPECIAL: Si se proporciona detection_time, el patch se centra
    automáticamente en ese tiempo para máxima precisión. El patch se muestra desde
    (detection_time - patch_duration/2) hasta (detection_time + patch_duration/2),
    con el candidato perfectamente centrado en detection_time.
    
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
        Start time in seconds (tiempo base del slice)
    off_regions : Optional[List[Tuple[int, int]]]
        Off-pulse regions for SNR calculation
    thresh_snr : Optional[float]
        SNR threshold for highlighting
    band_idx : int
        Band index for frequency range calculation
    band_name : str
        Name of the band for display
    detection_time : Optional[float]
        Tiempo exacto de detección del candidato para centrar el patch
    """

    # Check if patch is valid
    if patch is None or patch.size == 0:
        logger.warning(f"Cannot create patch plot: patch is None or empty. Skipping {out_path}")
        return

    # Calculate SNR profile
    snr_profile, sigma = compute_snr_profile(patch, off_regions)
    peak_snr, peak_time_rel, peak_idx = find_snr_peak(snr_profile)
    
    # Centrar el patch en el tiempo de detección del candidato
    if detection_time is not None:
        # Calcular el tiempo de inicio del patch para que el candidato quede centrado
        patch_duration = patch.shape[0] * time_reso
        patch_start_time = detection_time - (patch_duration / 2)
        patch_end_time = detection_time + (patch_duration / 2)
        
        # Log de la funcionalidad especial de centrado temporal
        logger.info(f"[PATCH] Patch centrado en candidato: tiempo={detection_time:.{config.PLOT_TIME_PRECISION}f}s, "
                   f"ventana=[{patch_start_time:.{config.PLOT_TIME_PRECISION}f}s, {patch_end_time:.{config.PLOT_TIME_PRECISION}f}s], duración={patch_duration:.{config.PLOT_TIME_PRECISION}f}s")
    else:
        # Fallback: usar el tiempo de inicio original
        patch_start_time = start_time
        patch_end_time = start_time + patch.shape[0] * time_reso
    
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4], hspace=0.05)

    # Usar los tiempos corregidos para el eje temporal
    # CORRECCIÓN CRÍTICA: Garantizar centrado exacto en el tiempo de detección
    if detection_time is not None:
        # MÉTODO EXACTO: Construir el eje temporal directamente desde el centro
        # Esto garantiza que detection_time esté exactamente en el centro
        n_samples = patch.shape[0]
        center_idx = n_samples // 2
        
        if n_samples % 2 == 0:
            # Número par de muestras: centrar entre dos muestras centrales
            half_samples = n_samples // 2
            # Construir eje temporal desde el centro hacia afuera
            time_axis = np.zeros(n_samples)
            for i in range(n_samples):
                if i < center_idx:
                    # Muestras antes del centro
                    time_axis[i] = detection_time - (center_idx - i) * time_reso
                else:
                    # Muestras desde el centro en adelante
                    time_axis[i] = detection_time + (i - center_idx) * time_reso
        else:
            # Número impar de muestras: centrar exactamente en la muestra central
            half_samples = n_samples // 2
            # Construir eje temporal desde el centro hacia afuera
            time_axis = np.zeros(n_samples)
            for i in range(n_samples):
                if i < center_idx:
                    # Muestras antes del centro
                    time_axis[i] = detection_time - (center_idx - i) * time_reso
                else:
                    # Muestras desde el centro en adelante
                    time_axis[i] = detection_time + (i - center_idx) * time_reso
        
        # Verificación del centrado temporal
        center_time = time_axis[center_idx]
        logger.info(f"[PATCH] Centrado temporal verificado:")
        logger.info(f"   - Tiempo objetivo: {detection_time:.9f}s")
        logger.info(f"   - Tiempo en el centro: {center_time:.9f}s")
        logger.info(f"   - Diferencia: {abs(center_time - detection_time):.9f}s")
        logger.info(f"   - Límites del patch: [{time_axis[0]:.9f}s, {time_axis[-1]:.9f}s]")
        
        # Verificar que el centrado sea perfecto
        if abs(center_time - detection_time) > 1e-9:
            logger.error(f" [PATCH] ERROR: Centrado temporal falló!")
            logger.error(f"   Diferencia: {abs(center_time - detection_time):.9f}s")
            raise ValueError(f"Centrado temporal falló: diferencia={abs(center_time - detection_time):.9f}s")
        else:
            logger.info(f"[PATCH] Centrado temporal perfecto!")
        
        peak_time_abs = detection_time
    else:
        # Fallback: usar linspace normal
        time_axis = np.linspace(patch_start_time, patch_end_time, patch.shape[0])
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

    # Marcar la posición del pico SNR en el waterfall
    if config.SNR_SHOW_PEAK_LINES:
        ax1.axvline(x=peak_time_abs, color=config.SNR_HIGHLIGHT_COLOR, 
                    linestyle='-', alpha=0.8, linewidth=2)

    ax1.set_xlabel("Time (s)", fontsize=10)
    ax1.set_ylabel("Frequency (MHz)", fontsize=10)
    
    # Título con información del centrado temporal
    if detection_time is not None:
        title = f"Candidate Patch - {band_name_with_freq}\nCentered at {detection_time:.{config.PLOT_TIME_PRECISION}f}s | Peak SNR: {peak_snr:.{config.PLOT_SNR_PRECISION}f}σ"
    else:
        title = f"Candidate Patch - {band_name_with_freq}\nPeak SNR: {peak_snr:.1f}σ"
    
    ax1.set_title(title, fontsize=11, fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


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
    absolute_start_time: Optional[float] = None, 
    chunk_idx: Optional[int] = None,  
    slice_samples: Optional[int] = None,  
    candidate_times_abs: Optional[Iterable[float]] = None,
    plot_context: Optional[dict] = None,
) -> None:
    print(f"[COMPOSITE] save_slice_summary INICIADA para slice {slice_idx}")
    print(f"[COMPOSITE] candidate_times_abs = {candidate_times_abs}")
    """Save a composite figure summarising detections and waterfalls with SNR analysis.
    
    FUNCIONALIDAD ESPECIAL: Los patches de waterfall (tanto individuales como en el composite)
    se centran automáticamente en el tiempo de detección del candidato para máxima precisión.
    Si el candidato se detectó en 9.55s y la ventana es de 500ms, el patch se muestra
    desde 9.30s hasta 9.80s, con el candidato perfectamente centrado en 9.55s.
    
    Esta funcionalidad NO afecta a los otros plots: waterfall dispersed, waterfall dedispersed,
    detection, DM-time.

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
    absolute_start_time : Optional[float]
        Tiempo absoluto de inicio del slice en segundos desde el inicio del archivo
    chunk_idx : Optional[int]
        Índice del chunk al que pertenece este slice
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
        print(f"[DEBUG PLOTS] Composite summary para: {fits_stem}")
        print(f"[DEBUG PLOTS] Band: {band_name_with_freq}")
        print(f"[DEBUG PLOTS] freq_ds shape: {freq_ds.shape}")
        print(f"[DEBUG PLOTS] freq_ds.min(): {freq_ds.min():.2f} MHz")
        print(f"[DEBUG PLOTS] freq_ds.max(): {freq_ds.max():.2f} MHz")
        print(f"[DEBUG PLOTS] waterfall_block shape: {waterfall_block.shape if waterfall_block is not None else 'None'}")
        print(f"[DEBUG PLOTS] dedispersed_block shape: {dedispersed_block.shape if dedispersed_block is not None else 'None'}")
        print(f"[DEBUG PLOTS] DM value: {dm_val:.2f} pc cm⁻³")
        print(f"[DEBUG PLOTS] imshow origin='lower' significa: freq_ds.min() en parte inferior, freq_ds.max() en parte superior")
        print(f"[DEBUG PLOTS] extent será: [tiempo_inicio, tiempo_fin, {freq_ds.min():.1f}, {freq_ds.max():.1f}]")
        print("[DEBUG PLOTS] " + "="*60)

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
        print(f"[DEBUG DATOS] Entrada a save_slice_summary:")
        print(f"[DEBUG DATOS] waterfall_block válido: {wf_block is not None}")
        print(f"[DEBUG DATOS] dedispersed_block válido: {dw_block is not None}")
        if wf_block is not None and dw_block is not None:
            print(f"[DEBUG DATOS] ¿Son iguales raw y dedispersed? {np.array_equal(wf_block, dw_block)}")
            print(f"[DEBUG DATOS] Diferencia máxima: {np.max(np.abs(wf_block - dw_block)):.6f}")
        print("[DEBUG DATOS] " + "="*50)
    
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


    if absolute_start_time is not None:
        slice_start_abs = absolute_start_time
    else:
        slice_start_abs = slice_idx * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
    
    # Duración real del slice (por si el último slice quedó truncado)
    real_samples = slice_samples if slice_samples is not None else slice_len
    slice_end_abs = slice_start_abs + real_samples * config.TIME_RESO * config.DOWN_TIME_RATE

    fig = plt.figure(figsize=(14, 12))


    gs_main = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.3, figure=fig)
    # Subplot para detecciones (parte superior izquierda)
    ax_det = fig.add_subplot(gs_main[0, 0])
    ax_det.imshow(img_rgb, origin="lower", aspect="auto")
    ax_det.set_title("Detection Results", fontsize=10, fontweight="bold")
    ax_det.set_xlabel("Time (s)", fontsize=9)
    ax_det.set_ylabel("Dispersion Measure (pc cm⁻³)", fontsize=9)

    prev_len_config = config.SLICE_LEN
    config.SLICE_LEN = slice_len

    n_time_ticks_det = 8
    time_positions_det = np.linspace(0, img_rgb.shape[1] - 1, n_time_ticks_det)

    # Etiquetar por BORDES: queremos [start, end] exactos
    real_samples = slice_samples if slice_samples is not None else slice_len
    n_px = img_rgb.shape[1]
    denom = float(max(n_px - 1, 1))
    time_values_det = slice_start_abs + (time_positions_det / denom) * (slice_end_abs - slice_start_abs)
    ax_det.set_xticks(time_positions_det)
    ax_det.set_xticklabels([f"{t:.{config.PLOT_TIME_PRECISION}f}" for t in time_values_det])
    ax_det.set_xlabel("Time (s)", fontsize=10, fontweight="bold")

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
    ax_det.set_yticklabels([f"{dm:.{config.PLOT_DM_PRECISION}f}" for dm in dm_values])
    ax_det.set_ylabel("Dispersion Measure (pc cm⁻³)", fontsize=10, fontweight="bold")

    # Bounding boxes con información completa - UNA SOLA ETIQUETA INTEGRADA
    if top_boxes is not None:
        for idx, (conf, box) in enumerate(zip(top_conf, top_boxes)):
            x1, y1, x2, y2 = map(int, box)
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            #  Usar el DM REAL (mismo cálculo que extract_candidate_dm)
            # Este es el DM que se usa en dedispersion y se guarda en CSV
            from ..preprocessing.dm_candidate_extractor import extract_candidate_dm
            effective_len_det = slice_samples if slice_samples is not None else slice_len
            dm_val_cand, t_sec_real, t_sample_real = extract_candidate_dm(center_x, center_y, effective_len_det)
            
            # UNIFICACIÓN: Usar SOLO candidate_times_abs para consistencia
            if candidate_times_abs is not None and idx < len(candidate_times_abs):
                detection_time = float(candidate_times_abs[idx])
            else:
                # UNIFICACIÓN: NO usar fallbacks inconsistentes
                if candidate_times_abs is None:
                    logger.warning(f"[DETECTION] candidate_times_abs es None para candidato {idx}")
                elif idx >= len(candidate_times_abs):
                    logger.warning(f"[DETECTION] Índice {idx} fuera de rango para candidate_times_abs (len={len(candidate_times_abs)})")
                else:
                    logger.warning(f"[DETECTION] candidate_times_abs[{idx}] no es válido")
                detection_time = None
            
            # Determinar si tenemos probabilidades de clasificación
            if class_probs is not None and idx < len(class_probs):
                class_prob = class_probs[idx]
                is_burst = class_prob >= config.CLASS_PROB
                color = "lime" if is_burst else "orange"
                burst_status = "BURST" if is_burst else "NO BURST"
                
                # Etiqueta completa con tiempo absoluto - USANDO DM REAL
                if detection_time is not None:
                    label = (
                        f"#{idx+1}\n"
                        f"DM: {dm_val_cand:.{config.PLOT_DM_PRECISION}f}\n"
                        f"Time: {detection_time:.{config.PLOT_TIME_PRECISION}f}s\n"
                        f"Det: {conf:.2f}\n"
                        f"Cls: {class_prob:.2f}\n"
                        f"{burst_status}"
                    )
                else:
                    label = (
                        f"#{idx+1}\n"
                        f"DM: {dm_val_cand:.{config.PLOT_DM_PRECISION}f}\n"
                        f"Time: N/A\n"
                        f"Det: {conf:.2f}\n"
                        f"Cls: {class_prob:.2f}\n"
                        f"{burst_status}"
                    )
            else:
                # Fallback si no hay probabilidades de clasificación
                color = "lime"
                if detection_time is not None:
                    label = f"#{idx+1}\nDM: {dm_val_cand:.{config.PLOT_DM_PRECISION}f}\nTime: {detection_time:.{config.PLOT_TIME_PRECISION}f}s\nDet: {conf:.2f}"
                else:
                    label = f"#{idx+1}\nDM: {dm_val_cand:.{config.PLOT_DM_PRECISION}f}\nTime: N/A\nDet: {conf:.2f}"
            
            # Dibujar rectángulo
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, 
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax_det.add_patch(rect)
            
            # Agregar etiqueta integrada
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
    # Indicar si se está usando DM dinámico
    dm_range_info = f"{dm_plot_min:.{config.PLOT_DM_PRECISION}f}\u2013{dm_plot_max:.{config.PLOT_DM_PRECISION}f}"
    if getattr(config, 'DM_DYNAMIC_RANGE_ENABLE', True) and top_boxes is not None and len(top_boxes) > 0:
        dm_range_info += " (auto)"
    else:
        dm_range_info += " (full)"
    
    # Precisión: mostrar duración exacta del slice considerando los índices reales usados
    exact_slice_ms_det = (slice_samples * (config.TIME_RESO * config.DOWN_TIME_RATE)) * 1000.0
    title_det = (
        f"Detection Map - {fits_stem} ({band_name_with_freq})\n"
        f"Slice {slice_idx:03d} of {time_slice} | Duration: {exact_slice_ms_det:.{config.PLOT_TIME_PRECISION}f} ms | "
        f"DM Range: {dm_range_info} pc cm⁻³"
    )
    ax_det.set_title(title_det, fontsize=11, fontweight="bold")
    config.SLICE_LEN = prev_len_config

    gs_bottom_row = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_main[1, 0], width_ratios=[1, 1, 1], wspace=0.3
    )

    # Auditoría opcional: persistir contexto de rutas usadas para este composite
    try:
        if getattr(config, 'SAVE_PLOT_CONTEXT', False) and plot_context is not None:
            import json
            base_out = out_path.with_suffix("")
            ctx_dir = Path(getattr(config, 'PLOT_CONTEXT_DIR', '') or base_out.parent)
            ctx_dir.mkdir(parents=True, exist_ok=True)
            ctx_file = ctx_dir / (base_out.name + ".ctx.json")
            payload = {
                'slice_start_abs': slice_start_abs,
                'slice_end_abs': slice_end_abs,
                'slice_samples': real_samples,
                'waterfall_dispersion_path': plot_context.get('waterfall_dispersion_path'),
                'waterfall_dedispersion_path': plot_context.get('waterfall_dedispersion_path'),
            }
            with open(ctx_file, 'w') as f:
                json.dump(payload, f, indent=2)
    except Exception:
        pass

    # Usar los mismos slice_start_abs y slice_end_abs ya calculados arriba con real_samples
    # para evitar inconsistencias entre paneles

    # === Panel 1: Raw Waterfall con SNR ===
    gs_waterfall_nested = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_bottom_row[0, 0], height_ratios=[1, 4], hspace=0.05
    )
    ax_prof_wf = fig.add_subplot(gs_waterfall_nested[0, 0])
    
    # Verificar si hay datos de waterfall válidos
    if wf_block is not None and wf_block.size > 0:
        # Calcular perfil SNR para raw waterfall
        snr_wf, sigma_wf = compute_snr_profile(wf_block, off_regions)
        peak_snr_wf, peak_time_wf, peak_idx_wf = find_snr_peak(snr_wf)
        
        # Bordes uniformes para etiquetas y extent
        time_axis_wf = np.linspace(slice_start_abs, slice_end_abs, len(snr_wf))
        peak_time_wf_abs = float(time_axis_wf[peak_idx_wf]) if len(snr_wf) > 0 else None
        ax_prof_wf.plot(time_axis_wf, snr_wf, color="royalblue", alpha=0.8, lw=1.5, label='SNR Profile')
        
        # Resaltar regiones sobre threshold
        if thresh_snr is not None and config.SNR_SHOW_PEAK_LINES:
            above_thresh_wf = snr_wf >= thresh_snr
            if np.any(above_thresh_wf):
                ax_prof_wf.plot(time_axis_wf[above_thresh_wf], snr_wf[above_thresh_wf], 
                              color=config.SNR_HIGHLIGHT_COLOR, lw=2, alpha=0.9)
            ax_prof_wf.axhline(y=thresh_snr, color=config.SNR_HIGHLIGHT_COLOR, 
                             linestyle='--', alpha=0.7, linewidth=1)
        
        # Marcar pico
        ax_prof_wf.plot(time_axis_wf[peak_idx_wf], peak_snr_wf, 'ro', markersize=5)
        ax_prof_wf.text(time_axis_wf[peak_idx_wf], peak_snr_wf + 0.1 * (ax_prof_wf.get_ylim()[1] - ax_prof_wf.get_ylim()[0]), 
                       f'{peak_snr_wf:.1f}σ', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax_prof_wf.set_xlim(time_axis_wf[0], time_axis_wf[-1])
        ax_prof_wf.set_ylabel('SNR (σ)', fontsize=8, fontweight='bold')
        ax_prof_wf.grid(True, alpha=0.3)
        ax_prof_wf.set_xticks([])
        if peak_time_wf_abs is not None:
            ax_prof_wf.set_title(
                f"Raw Waterfall SNR\nPeak={peak_snr_wf:.{config.PLOT_SNR_PRECISION}f}σ -> {peak_time_wf_abs:.{config.PLOT_TIME_PRECISION}f}s",
                fontsize=9,
                fontweight="bold",
            )
        else:
            ax_prof_wf.set_title(f"Raw Waterfall SNR\nPeak={peak_snr_wf:.{config.PLOT_SNR_PRECISION}f}σ", fontsize=9, fontweight="bold")
    else:
        ax_prof_wf.text(0.5, 0.5, 'No waterfall data\navailable', 
                       transform=ax_prof_wf.transAxes, 
                       ha='center', va='center', fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        ax_prof_wf.set_ylabel('SNR (σ)', fontsize=8, fontweight='bold')
        ax_prof_wf.grid(True, alpha=0.3)
        ax_prof_wf.set_xticks([])
        ax_prof_wf.set_title("No Raw Waterfall Data", fontsize=9, fontweight="bold")

    ax_wf = fig.add_subplot(gs_waterfall_nested[1, 0])
    
    if wf_block is not None and wf_block.size > 0:
        # DEBUG: Verificar raw waterfall
        if config.DEBUG_FREQUENCY_ORDER:
            print(f"[DEBUG RAW WF] Raw waterfall shape: {wf_block.shape}")
            print(f"[DEBUG RAW WF] Transpose para imshow: {wf_block.T.shape}")
            print(f"[DEBUG RAW WF] .T[0, :] (primera freq) primeras 5 muestras: {wf_block.T[0, :5]}")
            print(f"[DEBUG RAW WF] .T[-1, :] (última freq) primeras 5 muestras: {wf_block.T[-1, :5]}")
        
        cmap_name = getattr(config, 'WATERFALL_CMAP', 'mako')
        origin_mode = getattr(config, 'WATERFALL_ORIGIN', 'lower')
        im_wf = ax_wf.imshow(
            wf_block.T,
            origin=origin_mode,
            cmap=cmap_name,
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
        ax_wf.set_yticklabels([f"{f:.{config.PLOT_FREQ_PRECISION}f}" for f in freq_tick_positions])

        n_time_ticks = 5
        time_tick_positions = np.linspace(slice_start_abs, slice_end_abs, n_time_ticks)
        ax_wf.set_xticks(time_tick_positions)
        # Mostrar tiempo con mayor precisión para que el usuario vea el tiempo exacto
        ax_wf.set_xticklabels([f"{t:.{config.PLOT_TIME_PRECISION}f}" for t in time_tick_positions])
        ax_wf.set_xlabel("Time (s)", fontsize=9)
        ax_wf.set_ylabel("Frequency (MHz)", fontsize=9)
        
        # Marcar posición del pico SNR en el waterfall
        if 'peak_snr_wf' in locals() and config.SNR_SHOW_PEAK_LINES:
            ax_wf.axvline(x=time_axis_wf[peak_idx_wf], color=config.SNR_HIGHLIGHT_COLOR, 
                         linestyle='-', alpha=0.8, linewidth=2)

        # Sombrear zona inválida por falta de solape (opcional)
        try:
            if getattr(config, 'SHADE_INVALID_TAIL', True):
                # Calcular muestras inválidas máximas por Δt_max
                nu_min = freq_ds.min()
                nu_max = freq_ds.max()
                dt_max_ms = 4.1488 * max(config.DM_min, 0) * 0.0  # placeholder para claridad
                # usar DM_max real del plot para sobreestimar
                dm_max_plot = getattr(config, 'DM_max', 0)
                dt_max_sec = 4.1488e-3 * dm_max_plot * (nu_min**-2 - nu_max**-2)
                invalid_samp = max(0, int(np.ceil(dt_max_sec / (config.TIME_RESO * config.DOWN_TIME_RATE))))
                if invalid_samp > 0:
                    tail_start = slice_end_abs - invalid_samp * (config.TIME_RESO * config.DOWN_TIME_RATE)
                    if tail_start < slice_end_abs:
                        ax_wf.axvspan(tail_start, slice_end_abs, color='k', alpha=0.08, label='zona inválida')
        except Exception:
            pass
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
    
    # Usar el DM del candidato más fuerte para consistencia
    # En lugar de usar first_dm (que puede ser de cualquier candidato)
    # usar el DM del candidato con mayor confianza
    if top_boxes is not None and len(top_boxes) > 0:
        # Encontrar el candidato con mayor confianza
        best_candidate_idx = np.argmax(top_conf)
        best_box = top_boxes[best_candidate_idx]
        center_x, center_y = (best_box[0] + best_box[2]) / 2, (best_box[1] + best_box[3]) / 2
        
        # Calcular DM usando el mismo método que en pipeline_utils.py
        from ..preprocessing.dm_candidate_extractor import extract_candidate_dm
        dm_val_consistent, _, _ = extract_candidate_dm(center_x, center_y, slice_len)
        
        #  Calcular SNR del candidato más fuerte (como en CSV)
        # Extraer región del candidato para cálculo de SNR consistente
        x1, y1, x2, y2 = map(int, best_box)
        # Usar waterfall_block en lugar de band_img para consistencia
        candidate_region = waterfall_block[:, y1:y2] if waterfall_block is not None else None
        if candidate_region is not None and candidate_region.size > 0:
            snr_profile_candidate, _ = compute_snr_profile(candidate_region)
            snr_val_candidate = np.max(snr_profile_candidate)  # Tomar el pico del SNR
        else:
            snr_val_candidate = 0.0
    else:
        dm_val_consistent = dm_val  # Fallback al valor original
        snr_val_candidate = 0.0
        # Agregar información de debug cuando no hay candidatos
        if config.FORCE_PLOTS:
            logger.debug(f"🔍 [DEBUG] No candidatos detectados en slice {slice_idx}, pero force_plots=True")
    
    # Verificar si hay datos de waterfall dedispersado válidos
    if dw_block is not None and dw_block.size > 0:
        # Calcular perfil SNR para dedispersed waterfall
        snr_dw, sigma_dw = compute_snr_profile(dw_block, off_regions)
        peak_snr_dw, peak_time_dw, peak_idx_dw = find_snr_peak(snr_dw)
        
        time_axis_dw = np.linspace(slice_start_abs, slice_end_abs, len(snr_dw))
        peak_time_dw_abs = float(time_axis_dw[peak_idx_dw]) if len(snr_dw) > 0 else None
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
        ax_prof_dw.set_ylabel('SNR (σ)', fontsize=8, fontweight='bold')
        ax_prof_dw.grid(True, alpha=0.3)
        ax_prof_dw.set_xticks([])
        # Usar DM consistente en el título y mostrar ambos SNRs
        if snr_val_candidate > 0:
            if peak_time_dw_abs is not None:
                title_text = (
                    f"Dedispersed SNR DM={dm_val_consistent:.2f} pc cm⁻³\n"
                    f"Peak={peak_snr_dw:.{config.PLOT_SNR_PRECISION}f}σ -> {peak_time_dw_abs:.{config.PLOT_TIME_PRECISION}f}s (block) / {snr_val_candidate:.{config.PLOT_SNR_PRECISION}f}σ (candidate)"
                )
            else:
                title_text = (
                    f"Dedispersed SNR DM={dm_val_consistent:.2f} pc cm⁻³\n"
                    f"Peak={peak_snr_dw:.{config.PLOT_SNR_PRECISION}f}σ (block) / {snr_val_candidate:.{config.PLOT_SNR_PRECISION}f}σ (candidate)"
                )
        else:
            if peak_time_dw_abs is not None:
                title_text = (
                    f"Dedispersed SNR DM={dm_val_consistent:.2f} pc cm⁻³\n"
                    f"Peak={peak_snr_dw:.{config.PLOT_SNR_PRECISION}f}σ -> {peak_time_dw_abs:.{config.PLOT_TIME_PRECISION}f}s"
                )
            else:
                title_text = f"Dedispersed SNR DM={dm_val_consistent:.{config.PLOT_DM_PRECISION}f} pc cm⁻³\nPeak={peak_snr_dw:.{config.PLOT_SNR_PRECISION}f}σ"
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

    ax_dw = fig.add_subplot(gs_dedisp_nested[1, 0])
    
    if dw_block is not None and dw_block.size > 0:
        # Verificar dedispersed waterfall
        if config.DEBUG_FREQUENCY_ORDER:
            print(f"[DEBUG DED WF] Dedispersed waterfall shape: {dw_block.shape}")
            print(f"[DEBUG DED WF] Transpose para imshow: {dw_block.T.shape}")
            print(f"[DEBUG DED WF] .T[0, :] (primera freq) primeras 5 muestras: {dw_block.T[0, :5]}")
            print(f"[DEBUG DED WF] .T[-1, :] (última freq) primeras 5 muestras: {dw_block.T[-1, :5]}")
            print(f"[DEBUG DED WF] ¿Es diferente al raw? Diff promedio: {np.mean(np.abs(dw_block - wf_block)) if wf_block is not None else 'N/A'}")
        
        cmap_name = getattr(config, 'WATERFALL_CMAP', 'mako')
        origin_mode = getattr(config, 'WATERFALL_ORIGIN', 'lower')
        im_dw = ax_dw.imshow(
            dw_block.T,
            origin=origin_mode,
            cmap=cmap_name,
            aspect="auto",
            vmin=np.nanpercentile(dw_block, 1),
            vmax=np.nanpercentile(dw_block, 99),
            extent=[slice_start_abs, slice_end_abs, freq_ds.min(), freq_ds.max()],
        )
        ax_dw.set_xlim(slice_start_abs, slice_end_abs)
        ax_dw.set_ylim(freq_ds.min(), freq_ds.max())

        ax_dw.set_yticks(freq_tick_positions)
        ax_dw.set_yticklabels([f"{f:.{config.PLOT_FREQ_PRECISION}f}" for f in freq_tick_positions])
        ax_dw.set_xticks(time_tick_positions)
        ax_dw.set_xticklabels([f"{t:.{config.PLOT_TIME_PRECISION}f}" for t in time_tick_positions])
        ax_dw.set_xlabel("Time (s)", fontsize=9)
        ax_dw.set_ylabel("Frequency (MHz)", fontsize=9)
        
        # Marcar posición del pico SNR en el waterfall dedispersado
        if 'peak_snr_dw' in locals() and config.SNR_SHOW_PEAK_LINES:
            ax_dw.axvline(x=time_axis_dw[peak_idx_dw], color=config.SNR_HIGHLIGHT_COLOR, 
                         linestyle='-', alpha=0.8, linewidth=2)

        # Sombrear cola inválida igual que en raw
        try:
            if getattr(config, 'SHADE_INVALID_TAIL', True):
                nu_min = freq_ds.min()
                nu_max = freq_ds.max()
                dm_max_plot = getattr(config, 'DM_max', 0)
                dt_max_sec = 4.1488e-3 * dm_max_plot * (nu_min**-2 - nu_max**-2)
                invalid_samp = max(0, int(np.ceil(dt_max_sec / (config.TIME_RESO * config.DOWN_TIME_RATE))))
                if invalid_samp > 0:
                    tail_start = slice_end_abs - invalid_samp * (config.TIME_RESO * config.DOWN_TIME_RATE)
                    if tail_start < slice_end_abs:
                        ax_dw.axvspan(tail_start, slice_end_abs, color='k', alpha=0.08, label='zona inválida')
        except Exception:
            pass
    else:
        ax_dw.text(0.5, 0.5, 'No dedispersed data available', 
                  transform=ax_dw.transAxes, 
                  ha='center', va='center', fontsize=12, 
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        ax_dw.set_xticks([])
        ax_dw.set_yticks([])
        ax_dw.set_xlabel("Time (s)", fontsize=9)
        ax_dw.set_ylabel("Frequency (MHz)", fontsize=9)

    # === Panel 3: Candidate Patch con SNR ===
    gs_patch_nested = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_bottom_row[0, 2], height_ratios=[1, 4], hspace=0.05
    )
    ax_patch_prof = fig.add_subplot(gs_patch_nested[0, 0])
    
    # Verificar si hay un patch válido
    if patch_img is not None and patch_img.size > 0:
        # Calcular perfil SNR para el patch del candidato
        snr_patch, sigma_patch = compute_snr_profile(patch_img, off_regions)
        peak_snr_patch, peak_time_patch, peak_idx_patch = find_snr_peak(snr_patch)
        
        # Centrar el patch en el tiempo de detección del candidato
        # UNIFICACIÓN: Usar SOLO candidate_times_abs para consistencia
        detection_time_patch = None  # Inicializar variable
        
        # Validar candidate_times_abs antes de usarlo
        is_valid, error_msg = validate_candidate_times(candidate_times_abs, top_conf, top_boxes)
        
        if is_valid and top_boxes is not None and len(top_boxes) > 0:
            # UNIFICACIÓN: Seleccionar candidato usando SOLO candidate_times_abs
            # Usar el candidato con MAYOR CONFIANZA (más fuerte detectado)
            best_candidate_idx = np.argmax(top_conf)
            
            # Log de la selección del candidato
            logger.info(f"[COMPOSITE] Seleccionando candidato para patch:")
            logger.info(f"   - Candidatos disponibles: {[f'{t:.{config.PLOT_TIME_PRECISION}f}s' for t in candidate_times_abs]}")
            logger.info(f"   - Confianzas: {[f'{c:.3f}' for c in top_conf]}")
            logger.info(f"   - Candidato seleccionado: {candidate_times_abs[best_candidate_idx]:.{config.PLOT_TIME_PRECISION}f}s (índice {best_candidate_idx}, confianza {top_conf[best_candidate_idx]:.3f})")
            
            detection_time_patch = float(candidate_times_abs[best_candidate_idx])
            
            # Calcular el tiempo de inicio del patch para que el candidato quede centrado
            patch_duration = len(snr_patch) * time_reso_ds
            patch_start_abs = detection_time_patch - (patch_duration / 2)
            patch_end_abs = detection_time_patch + (patch_duration / 2)
            
            # Log de la funcionalidad especial de centrado temporal
            logger.info(f"[COMPOSITE] Patch centrado en candidato: tiempo={detection_time_patch:.{config.PLOT_TIME_PRECISION}f}s, "
                       f"ventana=[{patch_start_abs:.{config.PLOT_TIME_PRECISION}f}s, {patch_end_abs:.{config.PLOT_TIME_PRECISION}f}s], duración={patch_duration:.{config.PLOT_TIME_PRECISION}f}s")
        else:
            # UNIFICACIÓN: NO usar fallbacks inconsistentes
            # Si no hay candidate_times_abs, NO generar patch centrado
            if candidate_times_abs is not None:
                logger.warning(f"[COMPOSITE] candidate_times_abs no es válido: {error_msg}")
            else:
                logger.warning(f"[COMPOSITE] candidate_times_abs es None, NO se puede centrar el patch")
            detection_time_patch = None
            patch_start_abs = None
            patch_end_abs = None
        
        # Para el patch, usar bordes también para coherencia
        # Garantizar centrado exacto en el tiempo de detección
        print(f"[COMPOSITE] DEBUG: detection_time_patch = {detection_time_patch}")
        print(f"[COMPOSITE] DEBUG: type(detection_time_patch) = {type(detection_time_patch)}")
        print(f"[COMPOSITE] DEBUG: detection_time_patch == None = {detection_time_patch == None}")
        print(f"[COMPOSITE] DEBUG: detection_time_patch is None = {detection_time_patch is None}")
        
        logger.info(f"[COMPOSITE] DEBUG: detection_time_patch = {detection_time_patch}")
        logger.info(f"[COMPOSITE] DEBUG: type(detection_time_patch) = {type(detection_time_patch)}")
        logger.info(f"[COMPOSITE] DEBUG: detection_time_patch == None = {detection_time_patch == None}")
        logger.info(f"[COMPOSITE] DEBUG: detection_time_patch is None = {detection_time_patch is None}")
        
        if detection_time_patch is not None:
            print(f"[COMPOSITE] Usando lógica de centrado exacto")
            logger.info(f"[COMPOSITE] Usando lógica de centrado exacto")
            # Crear eje temporal que centre EXACTAMENTE en el tiempo de detección
            n_samples = len(snr_patch)
            
            # Construir el eje temporal directamente desde el centro
            # Esto garantiza que detection_time_patch esté exactamente en el centro
            center_idx = n_samples // 2
            
            if n_samples % 2 == 0:
                # Número par de muestras: centrar entre dos muestras centrales
                half_samples = n_samples // 2
                # Construir eje temporal desde el centro hacia afuera
                patch_time_axis = np.zeros(n_samples)
                for i in range(n_samples):
                    if i < center_idx:
                        # Muestras antes del centro
                        patch_time_axis[i] = detection_time_patch - (center_idx - i) * time_reso_ds
                    else:
                        # Muestras desde el centro en adelante
                        patch_time_axis[i] = detection_time_patch + (i - center_idx) * time_reso_ds
            else:
                # Número impar de muestras: centrar exactamente en la muestra central
                half_samples = n_samples // 2
                # Construir eje temporal desde el centro hacia afuera
                patch_time_axis = np.zeros(n_samples)
                for i in range(n_samples):
                    if i < center_idx:
                        # Muestras antes del centro
                        patch_time_axis[i] = detection_time_patch - (center_idx - i) * time_reso_ds
                    else:
                        # Muestras desde el centro en adelante
                        patch_time_axis[i] = detection_time_patch + (i - center_idx) * time_reso_ds
            
            # Verificación del centrado temporal
            center_time = patch_time_axis[center_idx]
            logger.info(f"[COMPOSITE] Centrado temporal verificado:")
            logger.info(f"   - Tiempo objetivo: {detection_time_patch:.9f}s")
            logger.info(f"   - Tiempo en el centro: {center_time:.9f}s")
            logger.info(f"   - Diferencia: {abs(center_time - detection_time_patch):.9f}s")
            logger.info(f"   - Límites del patch: [{patch_time_axis[0]:.9f}s, {patch_time_axis[-1]:.9f}s]")
            logger.info(f"   - VERIFICACIÓN FINAL: ¿Está centrado en 4.725s? {'✅ SÍ' if abs(center_time - 4.725) < 1e-6 else '❌ NO'}")
            logger.info(f"   - DIFERENCIA DEL OBJETIVO 4.725s: {abs(center_time - 4.725):.9f}s")
            
            # Verificar que el centrado sea perfecto
            if abs(center_time - detection_time_patch) > 1e-9:
                logger.error(f"[COMPOSITE] ERROR: Centrado temporal falló!")
                logger.error(f"   Diferencia: {abs(center_time - detection_time_patch):.9f}s")
                raise ValueError(f"Centrado temporal falló: diferencia={abs(center_time - detection_time_patch):.9f}s")
            else:
                logger.info(f"[COMPOSITE] Centrado temporal perfecto!")
                
        else:
            # Fallback: usar linspace normal cuando no hay candidatos
            print(f"[COMPOSITE] No hay candidatos válidos, usando fallback temporal")
            print(f"[COMPOSITE] ESTO NO DEBERÍA PASAR SI HAY CANDIDATOS!")
            logger.info(f"[COMPOSITE] No hay candidatos válidos, usando fallback temporal")
            logger.info(f"[COMPOSITE] ESTO NO DEBERÍA PASAR SI HAY CANDIDATOS!")
            patch_time_axis = np.linspace(patch_start_abs, patch_end_abs, len(snr_patch))
        
        # Log del eje temporal final usado
        logger.info(f"[COMPOSITE] Eje temporal final usado:")
        logger.info(f"   - Tipo: {'Centrado exacto' if detection_time_patch is not None else 'Fallback linspace'}")
        logger.info(f"   - Límites: [{patch_time_axis[0]:.9f}s, {patch_time_axis[-1]:.9f}s]")
        logger.info(f"   - Centro: {patch_time_axis[len(patch_time_axis)//2]:.9f}s")
        logger.info(f"   - VERIFICACIÓN FINAL: ¿Está centrado en 4.725s? {' SÍ' if abs(patch_time_axis[len(patch_time_axis)//2] - 4.725) < 1e-6 else '❌ NO'}")
        
        ax_patch_prof.plot(patch_time_axis, snr_patch, color="orange", alpha=0.8, lw=1.5, label='Candidate SNR')
        
        # Resaltar regiones sobre threshold
        if thresh_snr is not None and config.SNR_SHOW_PEAK_LINES:
            above_thresh_patch = snr_patch >= thresh_snr
            if np.any(above_thresh_patch):
                ax_patch_prof.plot(patch_time_axis[above_thresh_patch], snr_patch[above_thresh_patch], 
                                color=config.SNR_HIGHLIGHT_COLOR, lw=2, alpha=0.9)
            ax_patch_prof.axhline(y=thresh_snr, color=config.SNR_HIGHLIGHT_COLOR, 
                                 linestyle='--', alpha=0.7, linewidth=1)
        
        # Marcar pico - usar el tiempo de detección real del candidato
        if detection_time_patch is not None:
            peak_time_abs_patch = detection_time_patch
        else:
            peak_time_abs_patch = patch_time_axis[peak_idx_patch]
            
        ax_patch_prof.plot(peak_time_abs_patch, peak_snr_patch, 'ro', markersize=5)
        ax_patch_prof.text(peak_time_abs_patch, peak_snr_patch + 0.1 * (ax_patch_prof.get_ylim()[1] - ax_patch_prof.get_ylim()[0]), 
                          f'{peak_snr_patch:.1f}σ', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax_patch_prof.set_xlim(patch_time_axis[0], patch_time_axis[-1])
        ax_patch_prof.set_ylabel('SNR (σ)', fontsize=8, fontweight='bold')
        ax_patch_prof.grid(True, alpha=0.3)
        ax_patch_prof.set_xticks([])
        
        # Título con información del centrado temporal
        if detection_time_patch is not None:
            ax_patch_prof.set_title(f"Candidate Patch SNR\nCentered at {detection_time_patch:.{config.PLOT_TIME_PRECISION}f}s | Peak={peak_snr_patch:.{config.PLOT_SNR_PRECISION}f}σ", fontsize=9, fontweight="bold")
        else:
            ax_patch_prof.set_title(f"Candidate Patch SNR\nPeak={peak_snr_patch:.{config.PLOT_SNR_PRECISION}f}σ", fontsize=9, fontweight="bold")
    else:
        # Sin patch válido, mostrar mensaje
        ax_patch_prof.text(0.5, 0.5, 'No candidate patch\navailable', 
                          transform=ax_patch_prof.transAxes, 
                          ha='center', va='center', fontsize=10, 
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        ax_patch_prof.set_ylabel('SNR (σ)', fontsize=8, fontweight='bold')
        ax_patch_prof.grid(True, alpha=0.3)
        ax_patch_prof.set_xticks([])
        ax_patch_prof.set_title("No Candidate Patch", fontsize=9, fontweight="bold")

    ax_patch = fig.add_subplot(gs_patch_nested[1, 0])
    
    if patch_img is not None and patch_img.size > 0:
        # DEBUG: Verificar candidate patch
        if config.DEBUG_FREQUENCY_ORDER:
            print(f"[DEBUG PATCH] Candidate patch shape: {patch_img.shape}")
            print(f"[DEBUG PATCH] Transpose para imshow: {patch_img.T.shape}")
            print(f"[DEBUG PATCH] .T[0, :] (primera freq) primeras 5 muestras: {patch_img.T[0, :5]}")
            print(f"[DEBUG PATCH] .T[-1, :] (última freq) primeras 5 muestras: {patch_img.T[-1, :5]}")
        
        cmap_name = getattr(config, 'WATERFALL_CMAP', 'mako')
        origin_mode = getattr(config, 'WATERFALL_ORIGIN', 'lower')
        ax_patch.imshow(
            patch_img.T,
            origin=origin_mode,
            aspect="auto",
            cmap=cmap_name,
            vmin=np.nanpercentile(patch_img, 1),
            vmax=np.nanpercentile(patch_img, 99),
            extent=[patch_time_axis[0], patch_time_axis[-1], freq_ds.min(), freq_ds.max()],
        )
        ax_patch.set_xlim(patch_time_axis[0], patch_time_axis[-1])
        ax_patch.set_ylim(freq_ds.min(), freq_ds.max())

        n_patch_time_ticks = 5
        # Usar el eje temporal ya centrado correctamente
        # NO usar np.linspace aquí, ya que reintroduce el error de centrado
        if detection_time_patch is not None:
            # Usar el eje temporal que ya está perfectamente centrado
            patch_tick_positions = patch_time_axis[::len(patch_time_axis)//(n_patch_time_ticks-1)][:n_patch_time_ticks]
            # Asegurar que el último tick esté en el final exacto
            if len(patch_tick_positions) < n_patch_time_ticks:
                patch_tick_positions = np.append(patch_tick_positions, patch_time_axis[-1])
            
            # Log de verificación de ticks
            center_tick_idx = len(patch_tick_positions) // 2
            center_tick = patch_tick_positions[center_tick_idx]
            logger.info(f"[COMPOSITE] Ticks generados:")
            logger.info(f"   - Ticks: {[f'{t:.9f}s' for t in patch_tick_positions]}")
            logger.info(f"   - Tick central: {center_tick:.9f}s")
            logger.info(f"   - Tiempo objetivo: {detection_time_patch:.9f}s")
            logger.info(f"   - VERIFICACIÓN TICKS: ¿Está centrado en 4.725s? {'✅ SÍ' if abs(center_tick - 4.725) < 1e-6 else '❌ NO'}")
            logger.info(f"   - DIFERENCIA DEL OBJETIVO 4.725s: {abs(center_tick - 4.725):.9f}s")
        else:
            # Fallback: usar linspace solo si no hay detection_time_patch
            patch_tick_positions = np.linspace(patch_time_axis[0], patch_time_axis[-1], n_patch_time_ticks)
        
        ax_patch.set_xticks(patch_tick_positions)
        ax_patch.set_xticklabels([f"{t:.{config.PLOT_TIME_PRECISION}f}" for t in patch_tick_positions])

        ax_patch.set_yticks(freq_tick_positions)
        ax_patch.set_yticklabels([f"{f:.{config.PLOT_FREQ_PRECISION}f}" for f in freq_tick_positions])
        ax_patch.set_xlabel("Time (s)", fontsize=9)
        ax_patch.set_ylabel("Frequency (MHz)", fontsize=9)
        
        # Marcar posición del pico SNR en el patch - usar el tiempo de detección real
        if 'peak_snr_patch' in locals() and config.SNR_SHOW_PEAK_LINES:
            if detection_time_patch is not None:
                ax_patch.axvline(x=detection_time_patch, color=config.SNR_HIGHLIGHT_COLOR, 
                               linestyle='-', alpha=0.8, linewidth=2)
            else:
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

    # Crear título con información de auditoría temporal
    # Usar los tiempos absolutos del slice calculados arriba
    idx_start_ds = int(round(slice_start_abs / (config.TIME_RESO * config.DOWN_TIME_RATE)))
    idx_end_ds = idx_start_ds + real_samples - 1
    # Usar tiempos de BORDES en el título para coincidir con etiquetas
    start_center = slice_start_abs
    end_center = slice_end_abs
    if chunk_idx is not None:
        title = (
            f"Composite: {fits_stem} - {band_name_with_freq} - Chunk {chunk_idx:03d} - Slice {slice_idx:03d} | "
            f"start={start_center:.{config.PLOT_TIME_PRECISION}f}s end={end_center:.{config.PLOT_TIME_PRECISION}f}s Δt={(config.TIME_RESO * config.DOWN_TIME_RATE):.9f}s "
            f"| [idx {idx_start_ds}→{idx_end_ds}]"
        )
    else:
        title = (
            f"Composite: {fits_stem} - {band_name_with_freq} - Slice {slice_idx:03d} | "
            f"start={start_center:.{config.PLOT_TIME_PRECISION}f}s end={end_center:.{config.PLOT_TIME_PRECISION}f}s Δt={(config.TIME_RESO * config.DOWN_TIME_RATE):.9f}s "
            f"| [idx {idx_start_ds}→{idx_end_ds}]"
        )

    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )
    # === Información temporal exacta del slice (decimado) ===
    try:
        dt_ds = config.TIME_RESO * config.DOWN_TIME_RATE
        # Índice global de la primera muestra del slice en dominio decimado
        global_start_sample = int(round(slice_start_abs / dt_ds))
        global_end_sample = global_start_sample + real_samples - 1
        time_range_sec = real_samples * dt_ds

        info_lines = [
            f"Samples (decimated): {global_start_sample} → {global_end_sample} (N={real_samples})",
            f"Δt (effective): {dt_ds:.9f} s",
            f"Time span (centers): {start_center:.{config.PLOT_TIME_PRECISION}f}s → {end_center:.{config.PLOT_TIME_PRECISION}f}s (Δ={(real_samples - 1) * dt_ds:.{config.PLOT_TIME_PRECISION}f}s)",
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

    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()

