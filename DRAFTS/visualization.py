"""Helper functions for visualizations used in the pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from . import config
from .astro_conversions import pixel_to_physical
from .image_utils import postprocess_img, preprocess_img, save_detection_plot, plot_waterfall_block
from .dedispersion import dedisperse_block


def save_plot(
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


def save_patch_plot(
    patch: np.ndarray,
    out_path: Path,
    freq: np.ndarray,
    time_reso: float,
    start_time: float,
) -> None:
    """Save a visualization of the classification patch."""

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
    ax1.imshow(
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

    n_freq_ticks = 6
    freq_tick_positions = np.linspace(freq.min(), freq.max(), n_freq_ticks)
    ax1.set_yticks(freq_tick_positions)
    ax1.set_yticklabels([f"{f:.0f}" for f in freq_tick_positions])

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


def save_slice_summary(
    waterfall_block: np.ndarray,
    dedispersed_block: np.ndarray,
    img_rgb: np.ndarray,
    patch_img: np.ndarray,
    patch_start: float,
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
    """Save a composite figure summarising detections and waterfalls."""

    freq_ds = np.mean(
        config.FREQ.reshape(
            config.FREQ_RESO // config.DOWN_FREQ_RATE,
            config.DOWN_FREQ_RATE,
        ),
        axis=1,
    )
    time_reso_ds = config.TIME_RESO * config.DOWN_TIME_RATE

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 12))

    gs_main = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.3, figure=fig)
    ax_det = fig.add_subplot(gs_main[0, 0])
    ax_det.imshow(img_rgb, origin="lower", aspect="auto")

    prev_len_config = config.SLICE_LEN
    config.SLICE_LEN = slice_len

    n_time_ticks_det = 8
    time_positions_det = np.linspace(0, img_rgb.shape[1] - 1, n_time_ticks_det)
    time_start_det_slice_abs = slice_idx * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
    time_values_det = time_start_det_slice_abs + (time_positions_det / img_rgb.shape[1]) * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
    ax_det.set_xticks(time_positions_det)
    ax_det.set_xticklabels([f"{t:.3f}" for t in time_values_det])
    ax_det.set_xlabel("Time (s)", fontsize=10, fontweight="bold")

    n_dm_ticks = 8
    dm_positions = np.linspace(0, img_rgb.shape[0] - 1, n_dm_ticks)
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
    title_det = f"Detection Map - {fits_stem} ({band_name})\nSlice {slice_idx + 1} of {time_slice}"
    ax_det.set_title(title_det, fontsize=11, fontweight="bold")
    config.SLICE_LEN = prev_len_config

    gs_bottom_row = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_main[1, 0], width_ratios=[1, 1, 1], wspace=0.3
    )

    block_size_wf_samples = waterfall_block.shape[0]
    slice_start_abs = slice_idx * block_size_wf_samples * time_reso_ds
    slice_end_abs = slice_start_abs + block_size_wf_samples * time_reso_ds

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
    ax_wf.set_xlim(slice_start_abs, slice_end_abs)
    ax_wf.set_ylim(freq_ds.min(), freq_ds.max())

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
    ax_dw.set_xlim(slice_start_abs, slice_end_abs)
    ax_dw.set_ylim(freq_ds.min(), freq_ds.max())

    ax_dw.set_yticks(freq_tick_positions)
    ax_dw.set_yticklabels([f"{f:.0f}" for f in freq_tick_positions])
    ax_dw.set_xticks(time_tick_positions)
    ax_dw.set_xticklabels([f"{t:.3f}" for t in time_tick_positions])
    ax_dw.set_xlabel("Time (s)", fontsize=9)
    ax_dw.set_ylabel("Frequency (MHz)", fontsize=9)

    gs_patch_nested = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_bottom_row[0, 2], height_ratios=[1, 4], hspace=0.05
    )
    ax_patch_prof = fig.add_subplot(gs_patch_nested[0, 0])
    patch_prof_val = patch_img.mean(axis=1)

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
    ax_patch.set_xlim(patch_time_axis[0], patch_time_axis[-1])
    ax_patch.set_ylim(freq_ds.min(), freq_ds.max())

    n_patch_time_ticks = 5
    patch_tick_positions = np.linspace(patch_time_axis[0], patch_time_axis[-1], n_patch_time_ticks)
    ax_patch.set_xticks(patch_tick_positions)
    ax_patch.set_xticklabels([f"{t:.3f}" for t in patch_tick_positions])

    ax_patch.set_yticks(freq_tick_positions)
    ax_patch.set_yticklabels([f"{f:.0f}" for f in freq_tick_positions])
    ax_patch.set_xlabel("Time (s)", fontsize=9)
    ax_patch.set_ylabel("Frequency (MHz)", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(
        f"Composite Summary: {fits_stem} - {band_name} - Slice {slice_idx + 1}",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_waterfalls(
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


def plot_dedispersed_waterfalls(
    data: np.ndarray,
    freq_down: np.ndarray,
    dm: float,
    slice_len: int,
    time_slice: int,
    fits_stem: str,
    out_dir: Path,
) -> None:
    """Save dedispersed frequency--time waterfall plots for each time block."""

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

