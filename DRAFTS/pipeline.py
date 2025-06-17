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
from .dedispersion import d_dm_time_g
from .image_utils import (
    compute_snr,
    pixel_to_physical,
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

    if csv_path.exists():
        return

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

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(profile, color="royalblue", alpha=0.8, lw=1)
    ax0.set_xlim(0, patch.shape[0])
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
    )
    nchan = patch.shape[1]
    ax1.set_yticks(np.linspace(0, nchan, 6))
    ax1.set_yticklabels(np.round(np.linspace(freq.min(), freq.max(), 6)).astype(int))
    ax1.set_xticks(np.linspace(0, patch.shape[0], 6))
    ax1.set_xticklabels(
        np.round(start_time + np.linspace(0, patch.shape[0], 6) * time_reso, 2)
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (MHz)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _save_slice_summary(
    waterfall_block: np.ndarray,
    img_rgb: np.ndarray,
    patch_img: np.ndarray,
    patch_start: float,
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
    """Save a composite figure with waterfall, detection and patch."""

    freq_ds = np.mean(
        config.FREQ.reshape(
            config.FREQ_RESO // config.DOWN_FREQ_RATE,
            config.DOWN_FREQ_RATE,
        ),
        axis=1,
    )
    time_reso_ds = config.TIME_RESO * config.DOWN_TIME_RATE

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[1.3, 1],
        height_ratios=[1.3, 1],
        wspace=0.3,
        hspace=0.3,
    )

    # Waterfall with profile
    gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, 0], height_ratios=[1, 4], hspace=0.05)
    ax_prof = fig.add_subplot(gs_left[0, 0])
    profile = waterfall_block.mean(axis=1)
    ax_prof.plot(profile, color="royalblue", alpha=0.8, lw=1)
    block_size = waterfall_block.shape[0]
    ax_prof.set_xlim(0, block_size)
    ax_prof.set_xticks([])
    ax_prof.set_yticks([])

    ax_wf = fig.add_subplot(gs_left[1, 0])
    ax_wf.imshow(
        waterfall_block.T,
        origin="lower",
        cmap="mako",
        aspect="auto",
        vmin=np.nanpercentile(waterfall_block, 1),
        vmax=np.nanpercentile(waterfall_block, 99),
    )
    nchan = waterfall_block.shape[1]
    time_start = slice_idx * block_size * time_reso_ds
    ax_wf.set_yticks(np.linspace(0, nchan, 6))
    ax_wf.set_yticklabels(np.round(np.linspace(freq_ds.min(), freq_ds.max(), 6)).astype(int))
    ax_wf.set_xticks(np.linspace(0, block_size, 6))
    ax_wf.set_xticklabels(np.round(time_start + np.linspace(0, block_size, 6) * time_reso_ds, 2))
    ax_wf.set_xlabel("Time (s)")
    ax_wf.set_ylabel("Frequency (MHz)")

    # Detection image
    ax_det = fig.add_subplot(gs[0, 1])
    ax_det.imshow(img_rgb, origin="lower", aspect="auto")

    prev_len = config.SLICE_LEN
    config.SLICE_LEN = slice_len

    n_time_ticks = 6
    time_positions = np.linspace(0, 512, n_time_ticks)
    time_start_slice = slice_idx * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
    time_values = time_start_slice + (time_positions / 512.0) * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
    ax_det.set_xticks(time_positions)
    ax_det.set_xticklabels([f"{t:.3f}" for t in time_values])
    ax_det.set_xlabel("Time (s)", fontsize=10, fontweight="bold")

    n_dm_ticks = 8
    dm_positions = np.linspace(0, 512, n_dm_ticks)
    dm_values = config.DM_min + (dm_positions / 512.0) * (config.DM_max - config.DM_min)
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
            dm_val, _, _ = pixel_to_physical(center_x, center_y, slice_len)
            label = f"#{idx+1}\nDM: {dm_val:.1f}\nP: {conf:.2f}"
            ax_det.annotate(
                label,
                xy=(center_x, center_y),
                xytext=(center_x, y2 + 15),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lime", alpha=0.8),
                fontsize=7,
                ha="center",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="lime", lw=1),
            )

    title = (
        f"{fits_stem} - {band_name}\nSlice {slice_idx + 1}/{time_slice}"
    )
    ax_det.set_title(title, fontsize=9, fontweight="bold")

    # Patch image with profile
    gs_patch = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[1, 1], height_ratios=[1, 4], hspace=0.05
    )
    ax_patch_prof = fig.add_subplot(gs_patch[0, 0])
    patch_prof = patch_img.mean(axis=1)
    ax_patch_prof.plot(patch_prof, color="royalblue", alpha=0.8, lw=1)
    block_patch = patch_img.shape[0]
    ax_patch_prof.set_xlim(0, block_patch)
    ax_patch_prof.set_xticks([])
    ax_patch_prof.set_yticks([])

    ax_patch = fig.add_subplot(gs_patch[1, 0])
    ax_patch.imshow(
        patch_img.T,
        origin="lower",
        aspect="auto",
        cmap="mako",
        vmin=np.nanpercentile(patch_img, 1),
        vmax=np.nanpercentile(patch_img, 99),
    )
    nchan_patch = patch_img.shape[1]
    ax_patch.set_yticks(np.linspace(0, nchan_patch, 6))
    ax_patch.set_yticklabels(
        np.round(np.linspace(freq_ds.min(), freq_ds.max(), 6)).astype(int)
    )
    ax_patch.set_xticks(np.linspace(0, block_patch, 6))
    ax_patch.set_xticklabels(
        np.round(
            patch_start + np.linspace(0, block_patch, 6) * time_reso_ds, 2
        )
    )
    ax_patch.set_xlabel("Time (s)")
    ax_patch.set_ylabel("Frequency (MHz)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    config.SLICE_LEN = prev_len

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


def _dedisperse_patch(
    data: np.ndarray,
    freq_down: np.ndarray,
    dm: float,
    sample: int,
    patch_len: int = 512,
) -> tuple[np.ndarray, int]:
    """Dedisperse ``data`` at ``dm`` around ``sample`` and return a patch and
    the start sample used."""

    delays = (
        4.15
        * dm
        * (freq_down ** -2 - freq_down.max() ** -2)
        * 1e3
        / config.TIME_RESO
        / config.DOWN_TIME_RATE
    ).astype(np.int64)
    max_delay = int(delays.max())
    start = sample - patch_len // 2
    if start < 0:
        start = 0
    if start + patch_len + max_delay > data.shape[0]:
        start = max(0, data.shape[0] - (patch_len + max_delay))
    segment = data[start : start + patch_len + max_delay]
    patch = np.zeros((patch_len, freq_down.size), dtype=np.float32)
    for idx in range(freq_down.size):
        patch[:, idx] = segment[delays[idx] : delays[idx] + patch_len, idx]
    return patch, start


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

    n_time = (data.shape[0] // config.DOWN_TIME_RATE) * config.DOWN_TIME_RATE
    n_freq = (data.shape[2] // config.DOWN_FREQ_RATE) * config.DOWN_FREQ_RATE
    data = data[:n_time, :, :n_freq]
    data = (
        np.mean(
            data.reshape(
                n_time // config.DOWN_TIME_RATE,
                config.DOWN_TIME_RATE,
                2,
                n_freq // config.DOWN_FREQ_RATE,
                config.DOWN_FREQ_RATE,
            ),
            axis=(1, 4),
        )
        .mean(axis=1)
        .astype(np.float32)
    )

    freq_down = np.mean(
        config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE),
        axis=1,
    )

    freq_down = np.mean(
        config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE),
        axis=1,
    )

    height = config.DM_max - config.DM_min + 1
    width_total = config.FILE_LENG // config.DOWN_TIME_RATE

    slice_len, time_slice = _slice_parameters(width_total, config.SLICE_LEN)

    # Waterfalls will be generated as part of the slice summaries

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

    for j in range(time_slice):
        slice_cube = dm_time[:, :, slice_len * j : slice_len * (j + 1)]
        waterfall_block = data[j * slice_len : (j + 1) * slice_len]
        for band_idx, band_suffix, band_name in band_configs:
            band_img = slice_cube[band_idx]
            img_tensor = preprocess_img(band_img)
            top_conf, top_boxes = _detect(det_model, img_tensor)
            if top_boxes is None:
                continue

            img_rgb = postprocess_img(img_tensor)

            first_patch: np.ndarray | None = None
            first_start: float | None = None
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
                patch, start_sample = _dedisperse_patch(
                    data, freq_down, dm_val, global_sample
                )
                class_prob, proc_patch = _classify_patch(cls_model, patch)
                is_burst = class_prob >= config.CLASS_PROB
                if first_patch is None:
                    first_patch = proc_patch
                    first_start = start_sample * config.TIME_RESO * config.DOWN_TIME_RATE
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
                with csv_file.open("a", newline="") as f_csv:
                    writer = csv.writer(f_csv)
                    writer.writerow(cand.to_row())
                logger.info(
                    "Candidato DM %.2f t=%.3f s conf=%.2f class=%.2f -> %s",
                    dm_val,
                    t_sec,
                    conf,
                    class_prob,
                    "BURST" if is_burst else "no burst",
                )

            if first_patch is not None:
                _save_patch_plot(
                    first_patch,
                    patch_path,
                    freq_down,
                    config.TIME_RESO * config.DOWN_TIME_RATE,
                    first_start,
                )

                composite_dir = save_dir / "Composite" / fits_path.stem
                comp_path = composite_dir / f"slice{j}_band{band_idx}.png"
                _save_slice_summary(
                    waterfall_block,
                    img_rgb,
                    first_patch,
                    first_start,
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
