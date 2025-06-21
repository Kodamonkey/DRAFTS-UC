"""High level pipeline for FRB detection with CenterNet."""
from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec

from ObjectDet.centernet_model import centernet
from ObjectDet.centernet_utils import get_res
from BinaryClass.binary_model import BinaryNet

from . import config
from .candidate import Candidate
from .dedispersion import d_dm_time_g, dedisperse_patch, dedisperse_block
from .metrics import compute_snr
from .astro_conversions import pixel_to_physical
from .preprocessing import downsample_data
from .image_utils import (
    postprocess_img,
    preprocess_img,
    save_detection_plot,
    plot_waterfall_block,
)
from .io import get_obparams, load_fits_file

logger = logging.getLogger(__name__)

def _load_model() -> torch.nn.Module:
    """Load the CenterNet model configured in :mod:`config`."""

    model = centernet(model_name=config.MODEL_NAME).to(config.DEVICE)
    state = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def _load_class_model() -> torch.nn.Module:
    """Load the binary classification model configured in :mod:`config`."""

    model = BinaryNet(config.CLASS_MODEL_NAME, num_classes=2).to(config.DEVICE)
    state = torch.load(config.CLASS_MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

def _find_fits_files(frb: str) -> List[Path]:
    """Return FITS files matching ``frb`` within ``config.DATA_DIR``."""

    return sorted(f for f in config.DATA_DIR.glob("*.fits") if frb in f.name)


def _ensure_csv_header(csv_path: Path) -> None:
    """Create ``csv_path`` with the standard candidate header if needed."""

    # Verificar si el directorio padre existe y crearlo si no
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    if csv_path.exists():
        return

    try:
        with csv_path.open("w", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(
                [
                    "file",
                    "slice",
                    "band",
                    "prob",
                    "dm_pc_cm-3",
                    "t_sec",
                    "t_sample",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "snr",
                    "class_prob",
                    "is_burst",
                    "patch_file",
                ]
            )
    except PermissionError as e:
        logger.error("Error de permisos al crear CSV %s: %s", csv_path, e)
        raise


def _write_candidate_to_csv(csv_file: Path, candidate: Candidate) -> None:
    """Write a single candidate to the CSV file with error handling."""
    
    try:
        with csv_file.open("a", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(candidate.to_row())
    except PermissionError as e:
        logger.error("Error de permisos al escribir en CSV %s: %s", csv_file, e)
        # Intentar crear un archivo alternativo
        alt_csv = csv_file.with_suffix(f".{int(time.time())}.csv")
        logger.info("Intentando escribir en archivo alternativo: %s", alt_csv)
        with alt_csv.open("a", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(candidate.to_row())
    except Exception as e:
        logger.error("Error inesperado al escribir CSV: %s", e)
        raise


def _slice_parameters(width_total: int, slice_len: int) -> tuple[int, int]:
    """Return adjusted ``slice_len`` and number of slices for ``width_total``."""

    if width_total == 0:
        return 0, 0
    if width_total < slice_len:
        return width_total, 1
    return slice_len, width_total // slice_len


def _detect(model: torch.nn.Module, img_tensor: np.ndarray) -> tuple[list, list | None]:
    """Run the detection model and return confidences and boxes."""

    with torch.no_grad():
        hm, wh, offset = model(
            torch.from_numpy(img_tensor)
            .to(config.DEVICE)
            .float()
            .unsqueeze(0)
        )
    return get_res(hm, wh, offset, confidence=config.DET_PROB)


def _save_plot(
    img_rgb: np.ndarray,
    top_conf: Iterable,
    top_boxes: Iterable | None,
    out_img_path: Path,
    slice_idx: int,
    time_slice: int,
    band_name: str,
    band_suffix: str,
    fits_stem: str,
    slice_len: int,
) -> None:
    """Wrapper around :func:`save_detection_plot` with dynamic slice length."""

    prev_len = config.SLICE_LEN
    config.SLICE_LEN = slice_len
    save_detection_plot(
        img_rgb,
        top_conf,
        top_boxes,
        out_img_path,
        slice_idx,
        time_slice,
        band_name,
        band_suffix,
        config.DET_PROB,
        fits_stem,
    )
    config.SLICE_LEN = prev_len


def _save_patch_plot(
    patch: np.ndarray,
    out_path: Path,
    freq: np.ndarray,
    time_reso: float,
    start_time: float,
) -> None:
    """Save a visualization of the classification patch with a profile and
    axes in physical units."""

    profile = patch.mean(axis=1)
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4], hspace=0.05)

    time_axis = start_time + np.arange(patch.shape[0]) * time_reso

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(time_axis, profile, color="royalblue", alpha=0.8, lw=1)
    ax0.set_xlim(time_axis[0], time_axis[-1])
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = fig.add_subplot(gs[1, 0])
    im = ax1.imshow(
        patch.T,
        origin="lower",
        aspect="auto",
        cmap="mako",
        vmin=np.nanpercentile(patch, 1),
        vmax=np.nanpercentile(patch, 99),
        extent=[time_axis[0], time_axis[-1], freq.min(), freq.max()],
    )
    
    ax1.set_xlim(time_axis[0], time_axis[-1])
    ax1.set_ylim(freq.min(), freq.max())
    
    # Configurar ticks de frecuencia correctamente
    n_freq_ticks = 6
    freq_tick_positions = np.linspace(freq.min(), freq.max(), n_freq_ticks)
    ax1.set_yticks(freq_tick_positions)
    ax1.set_yticklabels([f"{f:.0f}" for f in freq_tick_positions])
    
    # Configurar ticks de tiempo correctamente
    n_time_ticks = 6
    time_tick_positions = np.linspace(time_axis[0], time_axis[-1], n_time_ticks)
    ax1.set_xticks(time_tick_positions)
    ax1.set_xticklabels([f"{t:.3f}" for t in time_tick_positions])
    
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (MHz)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _save_slice_summary(
    waterfall_block: np.ndarray,
    dedispersed_block: np.ndarray,
    img_rgb: np.ndarray,
    patch_img: np.ndarray,
    patch_start: float,  # Absolute start time of the patch in seconds
    dm_val: float,
    top_conf: Iterable,
    top_boxes: Iterable | None,
    out_path: Path,
    slice_idx: int,
    time_slice: int,
    band_name: str,
    band_suffix: str,
    fits_stem: str,
    slice_len: int,
) -> None:
    """Save a composite figure with detection plot as main,
    and waterfall and patch plots below."""

    freq_ds = np.mean(
        config.FREQ.reshape(
            config.FREQ_RESO // config.DOWN_FREQ_RATE,
            config.DOWN_FREQ_RATE,
        ),
        axis=1,
    )
    time_reso_ds = config.TIME_RESO * config.DOWN_TIME_RATE

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 12))  # Increased figure size

    # Main GridSpec: 2 rows. Top row for detection, bottom row for waterfall & patch
    gs_main = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.3, figure=fig)

    # --- Top Row: Detection Plot ---
    ax_det = fig.add_subplot(gs_main[0, 0])
    ax_det.imshow(img_rgb, origin="lower", aspect="auto")

    prev_len_config = config.SLICE_LEN
    config.SLICE_LEN = slice_len

    n_time_ticks_det = 8
    time_positions_det = np.linspace(0, img_rgb.shape[1]-1, n_time_ticks_det)
    time_start_det_slice_abs = slice_idx * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
    time_values_det = time_start_det_slice_abs + (time_positions_det / img_rgb.shape[1]) * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
    ax_det.set_xticks(time_positions_det)
    ax_det.set_xticklabels([f"{t:.3f}" for t in time_values_det])
    ax_det.set_xlabel("Time (s)", fontsize=10, fontweight="bold")

    n_dm_ticks = 8
    dm_positions = np.linspace(0, img_rgb.shape[0]-1, n_dm_ticks)
    dm_values = config.DM_min + (dm_positions / img_rgb.shape[0]) * (config.DM_max - config.DM_min)
    ax_det.set_yticks(dm_positions)
    ax_det.set_yticklabels([f"{dm:.0f}" for dm in dm_values])
    ax_det.set_ylabel("Dispersion Measure (pc cm⁻³)", fontsize=10, fontweight="bold")

    if top_boxes is not None:
        for idx, (conf, box) in enumerate(zip(top_conf, top_boxes)):
            x1, y1, x2, y2 = map(int, box)
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none"
            )
            ax_det.add_patch(rect)
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            dm_val_cand, _, _ = pixel_to_physical(center_x, center_y, slice_len)
            label = f"#{idx+1}\nDM: {dm_val_cand:.1f}\nP: {conf:.2f}"
            ax_det.annotate(
                label,
                xy=(center_x, center_y),
                xytext=(center_x, y2 + 20),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lime", alpha=0.8),
                fontsize=7,
                ha="center",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="lime", lw=1.5),
            )
    title_det = (
        f"Detection Map - {fits_stem} ({band_name})\n"
        f"Slice {slice_idx + 1} of {time_slice}"
    )
    ax_det.set_title(title_det, fontsize=11, fontweight="bold")
    config.SLICE_LEN = prev_len_config

    # --- Bottom Row: Waterfalls and Patch Plots ---
    gs_bottom_row = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_main[1, 0], width_ratios=[1, 1, 1], wspace=0.3
    )

    # Calculate time arrays for proper extent
    block_size_wf_samples = waterfall_block.shape[0]
    slice_start_abs = slice_idx * block_size_wf_samples * time_reso_ds
    slice_end_abs = slice_start_abs + block_size_wf_samples * time_reso_ds

    # --- Bottom Row, Left Column: Waterfall with profile ---
    gs_waterfall_nested = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_bottom_row[0, 0], height_ratios=[1, 4], hspace=0.05
    )
    ax_prof_wf = fig.add_subplot(gs_waterfall_nested[0, 0])
    profile_wf = waterfall_block.mean(axis=1)
    time_axis_wf = np.linspace(slice_start_abs, slice_end_abs, len(profile_wf))
    ax_prof_wf.plot(time_axis_wf, profile_wf, color="royalblue", alpha=0.8, lw=1)
    ax_prof_wf.set_xlim(slice_start_abs, slice_end_abs)
    ax_prof_wf.set_xticks([])
    ax_prof_wf.set_yticks([])
    ax_prof_wf.set_title(f"Raw Waterfall\nSlice {slice_idx+1}", fontsize=9, fontweight="bold")

    ax_wf = fig.add_subplot(gs_waterfall_nested[1, 0])
    ax_wf.imshow(
        waterfall_block.T,
        origin="lower",
        cmap="mako",
        aspect="auto",
        vmin=np.nanpercentile(waterfall_block, 1),
        vmax=np.nanpercentile(waterfall_block, 99),
        extent=[slice_start_abs, slice_end_abs, freq_ds.min(), freq_ds.max()],
    )
    # Set proper axis limits to match the extent
    ax_wf.set_xlim(slice_start_abs, slice_end_abs)
    ax_wf.set_ylim(freq_ds.min(), freq_ds.max())
    
    # Configure ticks properly
    n_freq_ticks = 6
    freq_tick_positions = np.linspace(freq_ds.min(), freq_ds.max(), n_freq_ticks)
    ax_wf.set_yticks(freq_tick_positions)
    ax_wf.set_yticklabels([f"{f:.0f}" for f in freq_tick_positions])
    
    n_time_ticks = 5
    time_tick_positions = np.linspace(slice_start_abs, slice_end_abs, n_time_ticks)
    ax_wf.set_xticks(time_tick_positions)
    ax_wf.set_xticklabels([f"{t:.3f}" for t in time_tick_positions])
    ax_wf.set_xlabel("Time (s)", fontsize=9)
    ax_wf.set_ylabel("Frequency (MHz)", fontsize=9)

    # --- Bottom Row, Middle Column: Dedispersed waterfall ---
    gs_dedisp_nested = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_bottom_row[0, 1], height_ratios=[1, 4], hspace=0.05
    )
    ax_prof_dw = fig.add_subplot(gs_dedisp_nested[0, 0])
    prof_dw = dedispersed_block.mean(axis=1)
    time_axis_dw = np.linspace(slice_start_abs, slice_end_abs, len(prof_dw))
    ax_prof_dw.plot(time_axis_dw, prof_dw, color="royalblue", alpha=0.8, lw=1)
    ax_prof_dw.set_xlim(slice_start_abs, slice_end_abs)
    ax_prof_dw.set_xticks([])
    ax_prof_dw.set_yticks([])
    ax_prof_dw.set_title(f"Dedispersed DM={dm_val:.2f}", fontsize=9, fontweight="bold")

    ax_dw = fig.add_subplot(gs_dedisp_nested[1, 0])
    ax_dw.imshow(
        dedispersed_block.T,
        origin="lower",
        cmap="mako",
        aspect="auto",
        vmin=np.nanpercentile(dedispersed_block, 1),
        vmax=np.nanpercentile(dedispersed_block, 99),
        extent=[slice_start_abs, slice_end_abs, freq_ds.min(), freq_ds.max()],
    )
    # Set proper axis limits
    ax_dw.set_xlim(slice_start_abs, slice_end_abs)
    ax_dw.set_ylim(freq_ds.min(), freq_ds.max())
    
    ax_dw.set_yticks(freq_tick_positions)
    ax_dw.set_yticklabels([f"{f:.0f}" for f in freq_tick_positions])
    ax_dw.set_xticks(time_tick_positions)
    ax_dw.set_xticklabels([f"{t:.3f}" for t in time_tick_positions])
    ax_dw.set_xlabel("Time (s)", fontsize=9)
    ax_dw.set_ylabel("Frequency (MHz)", fontsize=9)

    # --- Bottom Row, Right Column: Patch image with profile ---
    gs_patch_nested = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_bottom_row[0, 2], height_ratios=[1, 4], hspace=0.05
    )
    ax_patch_prof = fig.add_subplot(gs_patch_nested[0, 0])
    patch_prof_val = patch_img.mean(axis=1)

    patch_duration = len(patch_prof_val) * time_reso_ds
    patch_time_axis = patch_start + np.arange(len(patch_prof_val)) * time_reso_ds

    ax_patch_prof.plot(patch_time_axis, patch_prof_val, color="royalblue", alpha=0.8, lw=1)
    ax_patch_prof.set_xlim(patch_time_axis[0], patch_time_axis[-1])
    ax_patch_prof.set_xticks([])
    ax_patch_prof.set_yticks([])
    ax_patch_prof.set_title("Dedispersed Candidate Patch", fontsize=9, fontweight="bold")

    ax_patch = fig.add_subplot(gs_patch_nested[1, 0])
    ax_patch.imshow(
        patch_img.T,
        origin="lower",
        aspect="auto",
        cmap="mako",
        vmin=np.nanpercentile(patch_img, 1),
        vmax=np.nanpercentile(patch_img, 99),
        extent=[patch_time_axis[0], patch_time_axis[-1], freq_ds.min(), freq_ds.max()],
    )
    # Set proper axis limits
    ax_patch.set_xlim(patch_time_axis[0], patch_time_axis[-1])
    ax_patch.set_ylim(freq_ds.min(), freq_ds.max())

    # Configurar ticks específicos para el patch
    n_patch_time_ticks = 5
    patch_tick_positions = np.linspace(patch_time_axis[0], patch_time_axis[-1], n_patch_time_ticks)
    ax_patch.set_xticks(patch_tick_positions)
    ax_patch.set_xticklabels([f"{t:.3f}" for t in patch_tick_positions])

    ax_patch.set_yticks(freq_tick_positions)
    ax_patch.set_yticklabels([f"{f:.0f}" for f in freq_tick_positions])
    ax_patch.set_xlabel("Time (s)", fontsize=9)
    ax_patch.set_ylabel("Frequency (MHz)", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(f"Composite Summary: {fits_stem} - {band_name} - Slice {slice_idx + 1}", 
                 fontsize=14, fontweight='bold', y=0.97)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def _write_summary(summary: dict, save_path: Path) -> None:
    summary_path = save_path / "summary.json"
    with summary_path.open("w") as f_json:
        json.dump(summary, f_json, indent=2)
    logger.info("Resumen global escrito en %s", summary_path)


def _plot_waterfalls(
    data: np.ndarray,
    slice_len: int,
    time_slice: int,
    fits_stem: str,
    out_dir: Path,
) -> None:
    """Save frequency--time waterfall plots for each time block."""

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
        )


def _plot_dedispersed_waterfalls(
    data: np.ndarray,
    freq_down: np.ndarray,
    dm: float,
    slice_len: int,
    time_slice: int,
    fits_stem: str,
    out_dir: Path,
) -> None:
    """Save dedispersed frequency--time waterfall plots for each time block.

    The dedispersion is performed at ``dm`` while preserving the original
    slice boundaries.
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
        )

def _prep_patch(patch: np.ndarray) -> np.ndarray:
    """Normalize patch for classification."""

    patch = patch.copy()
    patch += 1
    patch /= np.mean(patch, axis=0)
    vmin, vmax = np.nanpercentile(patch, [5, 95])
    patch = np.clip(patch, vmin, vmax)
    patch = (patch - patch.min()) / (patch.max() - patch.min())
    return patch


def _classify_patch(model: torch.nn.Module, patch: np.ndarray) -> tuple[float, np.ndarray]:
    """Return probability from binary model for ``patch`` along with the processed patch."""

    proc = _prep_patch(patch)
    tensor = torch.from_numpy(proc[None, None, :, :]).float().to(config.DEVICE)
    with torch.no_grad():
        out = model(tensor)
        prob = out.softmax(dim=1)[0, 1].item()
    return prob, proc


def _process_file(
    det_model: torch.nn.Module,
    cls_model: torch.nn.Module,
    fits_path: Path,
    save_dir: Path,
) -> dict:
    """Process a single FITS file and return summary information."""

    t_start = time.time()
    logger.info("Procesando %s", fits_path.name)

    data = load_fits_file(str(fits_path))
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    data = np.vstack([data, data[::-1, :]])

    data = downsample_data(data)

    freq_down = np.mean(
        config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE),
        axis=1,
    )

    height = config.DM_max - config.DM_min + 1
    width_total = config.FILE_LENG // config.DOWN_TIME_RATE

    slice_len, time_slice = _slice_parameters(width_total, config.SLICE_LEN)

    dm_time = d_dm_time_g(data, height=height, width=width_total)

    slice_duration = slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
    logger.info(
        "Análisis de %s con %d slices de %d muestras (%.3f s cada uno)",
        fits_path.name,
        time_slice,
        slice_len,
        slice_duration,
    )

    csv_file = save_dir / f"{fits_path.stem}.candidates.csv"
    _ensure_csv_header(csv_file)

    cand_counter = 0
    prob_max = 0.0
    snr_list: List[float] = []

    band_configs = (
        [
            (0, "fullband", "Full Band"),
            (1, "lowband", "Low Band"),
            (2, "highband", "High Band"),
        ]
        if config.USE_MULTI_BAND
        else [(0, "fullband", "Full Band")]
    )

    # Preparar directorios para waterfalls individuales
    waterfall_dispersion_dir = save_dir / "waterfall_dispersion" / fits_path.stem
    waterfall_dedispersion_dir = save_dir / "waterfall_dedispersion" / fits_path.stem
    freq_ds = np.mean(
        config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE),
        axis=1,
    )
    time_reso_ds = config.TIME_RESO * config.DOWN_TIME_RATE

    for j in range(time_slice):
        slice_cube = dm_time[:, :, slice_len * j : slice_len * (j + 1)]
        waterfall_block = data[j * slice_len : (j + 1) * slice_len]
        
        # 2) Generar waterfall sin dedispersar para este slice
        waterfall_dispersion_dir.mkdir(parents=True, exist_ok=True)
        if waterfall_block.size > 0:
            plot_waterfall_block(
                data_block=waterfall_block,
                freq=freq_ds,
                time_reso=time_reso_ds,
                block_size=waterfall_block.shape[0],
                block_idx=j,
                save_dir=waterfall_dispersion_dir,
                filename=fits_path.stem,
            )

        for band_idx, band_suffix, band_name in band_configs:
            band_img = slice_cube[band_idx]
            img_tensor = preprocess_img(band_img)
            top_conf, top_boxes = _detect(det_model, img_tensor)
            if top_boxes is None:
                continue

            img_rgb = postprocess_img(img_tensor)

            first_patch: np.ndarray | None = None
            first_start: float | None = None
            first_dm: float | None = None
            patch_dir = save_dir / "Patches" / fits_path.stem
            patch_path = patch_dir / f"patch_slice{j}_band{band_idx}.png"

            for conf, box in zip(top_conf, top_boxes):
                dm_val, t_sec, t_sample = pixel_to_physical(
                    (box[0] + box[2]) / 2,
                    (box[1] + box[3]) / 2,
                    slice_len,
                )
                snr_val = compute_snr(band_img, tuple(map(int, box)))
                snr_list.append(snr_val)
                global_sample = j * slice_len + int(t_sample)
                patch, start_sample = dedisperse_patch(
                    data, freq_down, dm_val, global_sample
                )
                class_prob, proc_patch = _classify_patch(cls_model, patch)
                is_burst = class_prob >= config.CLASS_PROB
                if first_patch is None:
                    first_patch = proc_patch
                    first_start = start_sample * config.TIME_RESO * config.DOWN_TIME_RATE
                    first_dm = dm_val
                cand = Candidate(
                    fits_path.name,
                    j,
                    band_idx,
                    float(conf),
                    dm_val,
                    t_sec,
                    t_sample,
                    tuple(map(int, box)),
                    snr_val,
                    class_prob,
                    is_burst,
                    patch_path.name,
                )
                cand_counter += 1
                prob_max = max(prob_max, float(conf))
                
                # Usar la función con manejo de errores
                _write_candidate_to_csv(csv_file, cand)
                
                logger.info(
                    "Candidato DM %.2f t=%.3f s conf=%.2f class=%.2f -> %s",
                    dm_val,
                    t_sec,
                    conf,
                    class_prob,
                    "BURST" if is_burst else "no burst",
                )

            if first_patch is not None:
                # 3) Generar waterfall dedispersado para este slice con el primer candidato
                waterfall_dedispersion_dir.mkdir(parents=True, exist_ok=True)
                start = j * slice_len
                dedisp_block = dedisperse_block(data, freq_down, first_dm, start, slice_len)
                if dedisp_block.size > 0:
                    plot_waterfall_block(
                        data_block=dedisp_block,
                        freq=freq_down,
                        time_reso=time_reso_ds,
                        block_size=dedisp_block.shape[0],
                        block_idx=j,
                        save_dir=waterfall_dedispersion_dir,
                        filename=f"{fits_path.stem}_dm{first_dm:.2f}",
                    )

                _save_patch_plot(
                    first_patch,
                    patch_path,
                    freq_down,
                    config.TIME_RESO * config.DOWN_TIME_RATE,
                    first_start,
                )

                # 1) Generar composite
                composite_dir = save_dir / "Composite" / fits_path.stem
                comp_path = composite_dir / f"slice{j}_band{band_idx}.png"
                _save_slice_summary(
                    waterfall_block,
                    dedisp_block,
                    img_rgb,
                    first_patch,
                    first_start,
                    first_dm,
                    top_conf,
                    top_boxes,
                    comp_path,
                    j,
                    time_slice,
                    band_name,
                    band_suffix,
                    fits_path.stem,
                    slice_len,
                )

            # 4) Generar detecciones de Bow ties (detections)
            out_img_path = save_dir / f"{fits_path.stem}_slice{j}_{band_suffix}.png"
            _save_plot(
                img_rgb,
                top_conf,
                top_boxes,
                out_img_path,
                j,
                time_slice,
                band_name,
                band_suffix,
                fits_path.stem,
                slice_len,
            )

    runtime = time.time() - t_start
    logger.info(
        "\u25b6 %s: %d candidatos, max prob %.2f, \u23f1 %.1f s",
        fits_path.name,
        cand_counter,
        prob_max,
        runtime,
    )

    return {
        "n_candidates": cand_counter,
        "runtime_s": runtime,
        "max_prob": float(prob_max),
        "mean_snr": float(np.mean(snr_list)) if snr_list else 0.0,
    }


def run_pipeline() -> None:
    """Run the full FRB detection pipeline."""

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    save_dir = config.RESULTS_DIR / config.MODEL_NAME
    save_dir.mkdir(parents=True, exist_ok=True)
    det_model = _load_model()
    cls_model = _load_class_model()

    summary: dict[str, dict] = {}
    for frb in config.FRB_TARGETS:
        file_list = _find_fits_files(frb)
        if not file_list:
            continue

        get_obparams(str(file_list[0]))
        for fits_path in file_list:
            summary[fits_path.name] = _process_file(det_model, cls_model, fits_path, save_dir)

    _write_summary(summary, save_dir)
