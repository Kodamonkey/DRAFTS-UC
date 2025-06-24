"""Utility functions for image handling and visualization."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
from pyparsing import Iterable
import seaborn as sns
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import ListedColormap

from . import config
from .astro_conversions import pixel_to_physical

if "mako" not in plt.colormaps():
    plt.register_cmap(
        name="mako",
        cmap=ListedColormap(sns.color_palette("mako", as_cmap=True)(np.linspace(0, 1, 256)))
    )


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
) -> None:
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
    """

    block = data_block.copy() if normalize else data_block
    if normalize:
        block += 1
        block /= np.mean(block, axis=0)
        vmin, vmax = np.nanpercentile(block, [5, 95])
        block = np.clip(block, vmin, vmax)
        block = (block - block.min()) / (block.max() - block.min())

    profile = block.mean(axis=1)
    time_start = block_idx * block_size * time_reso
    peak_time = time_start + np.argmax(profile) * time_reso

    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 4, 4, 4], hspace=0.05)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(profile, color="royalblue", alpha=0.8, lw=1)
    ax0.set_xlim(0, block_size)
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = fig.add_subplot(gs[1:, 0])
    ax1.imshow(
        block.T,
        origin="lower",
        cmap="mako",
        aspect="auto",
        vmin=np.nanpercentile(block, 1),
        vmax=np.nanpercentile(block, 99),
    )
    nchan = block.shape[1]
    ax1.set_yticks(np.linspace(0, nchan, 6))
    ax1.set_yticklabels(np.round(np.linspace(freq.min(), freq.max(), 6)).astype(int))
    ax1.set_xticks(np.linspace(0, block_size, 6))
    ax1.set_xticklabels(np.round(time_start + np.linspace(0, block_size, 6) * time_reso, 2))
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (MHz)")

    out_path = save_dir / f"{filename}-block{block_idx:03d}-peak{peak_time:.2f}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


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
) -> None:
    """Save detection plot with both detection and classification probabilities."""

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(img_rgb, origin="lower", aspect="auto")

    # Time axis labels
    n_time_ticks = 6
    time_positions = np.linspace(0, 512, n_time_ticks)
    time_start_slice = slice_idx * config.SLICE_LEN * config.TIME_RESO * config.DOWN_TIME_RATE
    time_values = time_start_slice + (
        time_positions / 512.0
    ) * config.SLICE_LEN * config.TIME_RESO * config.DOWN_TIME_RATE
    ax.set_xticks(time_positions)
    ax.set_xticklabels([f"{t:.3f}" for t in time_values])
    ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")

    # DM axis labels
    n_dm_ticks = 8
    dm_positions = np.linspace(0, 512, n_dm_ticks)
    dm_values = config.DM_min + (dm_positions / 512.0) * (config.DM_max - config.DM_min)
    ax.set_yticks(dm_positions)
    ax.set_yticklabels([f"{dm:.0f}" for dm in dm_values])
    ax.set_ylabel("Dispersion Measure (pc cm⁻³)", fontsize=12, fontweight="bold")

    # Title
    if config.FREQ is not None:
        freq_range = f"{config.FREQ.min():.1f}\u2013{config.FREQ.max():.1f} MHz"
    else:
        freq_range = ""
    title = (
        f"{fits_stem} - {band_name} ({freq_range})\n"
        f"Slice {slice_idx + 1}/{time_slice} | "
        f"Time Resolution: {config.TIME_RESO * config.DOWN_TIME_RATE * 1e6:.1f} \u03bcs | "
        f"DM Range: {config.DM_min}\u2013{config.DM_max} pc cm⁻\u00b3"
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
            dm_val, t_sec, _ = pixel_to_physical(center_x, center_y, config.SLICE_LEN)
            
            # Determinar si tenemos probabilidades de clasificación
            if class_probs is not None and idx < len(class_probs):
                class_prob = class_probs[idx]
                is_burst = class_prob >= config.CLASS_PROB
                color = "lime" if is_burst else "orange"
                burst_status = "BURST" if is_burst else "NO BURST"
                
                # Etiqueta completa con toda la información
                label = (
                    f"#{idx+1}\n"
                    f"DM: {dm_val:.1f}\n"
                    f"Det: {conf:.2f}\n"
                    f"Cls: {class_prob:.2f}\n"
                    f"{burst_status}"
                )
            else:
                # Fallback si no hay probabilidades de clasificación
                color = "lime"
                label = f"#{idx+1}\nDM: {dm_val:.1f}\nDet: {conf:.2f}"
            
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
    plt.tight_layout()
    plt.savefig(out_img_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()

    if band_suffix == "fullband":
        fig_cb, ax_cb = plt.subplots(figsize=(13, 8))
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        im_cb = ax_cb.imshow(img_gray, origin="lower", aspect="auto", cmap="mako")
        ax_cb.set_xticks(time_positions)
        ax_cb.set_xticklabels([f"{t:.3f}" for t in time_values])
        ax_cb.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax_cb.set_yticks(dm_positions)
        ax_cb.set_yticklabels([f"{dm:.0f}" for dm in dm_values])
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
        plt.tight_layout()
        cb_path = out_img_path.parent / f"{out_img_path.stem}_colorbar{out_img_path.suffix}"
        plt.savefig(cb_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close()
