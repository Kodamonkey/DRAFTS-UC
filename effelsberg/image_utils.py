"""Utility functions for image handling and visualization."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from . import config


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
) -> None:
    profile = data_block.mean(axis=1)
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
        data_block.T,
        origin="lower",
        cmap="mako",
        aspect="auto",
        vmin=np.nanpercentile(data_block, 1),
        vmax=np.nanpercentile(data_block, 99),
    )
    nchan = data_block.shape[1]
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


def pixel_to_physical(px: float, py: float, slice_len: int) -> Tuple[float, float, int]:
    dm_range = config.DM_max - config.DM_min + 1
    scale_dm = dm_range / 512.0
    scale_time = slice_len / 512.0
    dm_val = config.DM_min + py * scale_dm
    sample_off = px * scale_time
    t_sample = int(sample_off)
    t_seconds = t_sample * config.TIME_RESO * config.DOWN_TIME_RATE
    return dm_val, t_seconds, t_sample


def compute_snr(slice_band: np.ndarray, box: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = map(int, box)
    box_data = slice_band[y1:y2, x1:x2]
    if box_data.size == 0:
        return 0.0
    signal = box_data.mean()
    noise = np.median(slice_band)
    std = slice_band.std(ddof=1)
    return (signal - noise) / (std + 1e-6)
