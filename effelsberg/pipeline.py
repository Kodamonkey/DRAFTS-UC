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

from ObjectDet.centernet_model import centernet
from ObjectDet.centernet_utils import get_res

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


def _process_file(model: torch.nn.Module, fits_path: Path, save_dir: Path) -> dict:
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

    height = config.DM_max - config.DM_min + 1
    width_total = config.FILE_LENG // config.DOWN_TIME_RATE

    slice_len, time_slice = _slice_parameters(width_total, config.SLICE_LEN)

    waterfall_dir = save_dir / "Waterfalls" / fits_path.stem
    _plot_waterfalls(data, slice_len, time_slice, fits_path.stem, waterfall_dir)

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
        for band_idx, band_suffix, band_name in band_configs:
            band_img = slice_cube[band_idx]
            img_tensor = preprocess_img(band_img)

            top_conf, top_boxes = _detect(model, img_tensor)
            if top_boxes is None:
                continue

            img_rgb = postprocess_img(img_tensor)
            for conf, box in zip(top_conf, top_boxes):
                dm_val, t_sec, t_sample = pixel_to_physical(
                    (box[0] + box[2]) / 2,
                    (box[1] + box[3]) / 2,
                    slice_len,
                )
                snr_val = compute_snr(band_img, tuple(map(int, box)))
                snr_list.append(snr_val)
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
                )
                cand_counter += 1
                prob_max = max(prob_max, float(conf))
                with csv_file.open("a", newline="") as f_csv:
                    writer = csv.writer(f_csv)
                    writer.writerow(cand.to_row())

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

    model = _load_model()

    summary: dict[str, dict] = {}
    for frb in config.FRB_TARGETS:
        file_list = _find_fits_files(frb)
        if not file_list:
            continue

        get_obparams(str(file_list[0]))
        for fits_path in file_list:
            summary[fits_path.name] = _process_file(model, fits_path, save_dir)

    _write_summary(summary, save_dir)