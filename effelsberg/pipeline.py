"""High level pipeline for FRB detection with CenterNet."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import List

import cv2

import numpy as np
import torch

from . import config
from .candidate import Candidate
from .dedispersion import d_dm_time_g
from .image_utils import compute_snr, pixel_to_physical, postprocess_img, preprocess_img
from .io import get_obparams, load_fits_file
from ObjectDet.centernet_utils import get_res
from ObjectDet.centernet_model import centernet


def run_pipeline() -> None:
    det_prob = config.DET_PROB
    save_path = config.RESULTS_DIR / config.MODEL_NAME
    save_path.mkdir(parents=True, exist_ok=True)

    model = centernet(model_name=config.MODEL_NAME).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.eval()

    summary: dict[str, dict] = {}

    for frb in config.FRB_TARGETS:
        file_list = sorted([f for f in config.DATA_DIR.glob("*.fits") if frb in f.name])
        if not file_list:
            continue
        get_obparams(str(file_list[0]))
        for fits_path in file_list:
            t_start = time.time()
            print(f"Procesando {fits_path.name}")

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
            dm_time = d_dm_time_g(data, height=height, width=width_total)

            if width_total == 0:
                time_slice = 0
            elif width_total < config.SLICE_LEN:
                config.SLICE_LEN = width_total
                time_slice = 1
            else:
                time_slice = width_total // config.SLICE_LEN
            if time_slice == 0 and width_total > 0:
                time_slice = 1
                config.SLICE_LEN = width_total
            print(f"Análisis de {fits_path.name} con {time_slice} slices de {config.SLICE_LEN} muestras")

            csv_file = save_path / f"{fits_path.stem}.candidates.csv"
            if not csv_file.exists():
                with csv_file.open("w", newline="") as f_csv:
                    writer = csv.writer(f_csv)
                    writer.writerow([
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
                    ])
            cand_counter = 0
            prob_max = 0.0
            snr_list: List[float] = []

            band_configs = [
                (0, "fullband", "Full Band"),
                (1, "lowband", "Low Band"),
                (2, "highband", "High Band"),
            ] if config.USE_MULTI_BAND else [(0, "fullband", "Full Band")]

            for j in range(time_slice):
                slice_cube = dm_time[:, :, config.SLICE_LEN * j : config.SLICE_LEN * (j + 1)]
                for band_idx, band_suffix, _ in band_configs:
                    band_img = slice_cube[band_idx]
                    img_tensor = preprocess_img(band_img)
                    with torch.no_grad():
                        hm, wh, offset = model(torch.from_numpy(img_tensor).to(config.DEVICE).float().unsqueeze(0))
                    top_conf, top_boxes = get_res(hm, wh, offset, confidence=det_prob)
                    if top_boxes is None:
                        continue
                    img_rgb = postprocess_img(img_tensor)
                    for conf, box in zip(top_conf, top_boxes):
                        dm_val, t_sec, t_sample = pixel_to_physical((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, config.SLICE_LEN)
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
                        prob_max = max(prob_max, conf)
                        with csv_file.open("a", newline="") as f_csv:
                            writer = csv.writer(f_csv)
                            writer.writerow(cand.to_row())
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 220, 0), 1)
                    out_img_path = save_path / f"{fits_path.stem}_slice{j}_{band_suffix}.png"
                    cv2.imwrite(str(out_img_path), img_rgb)
            runtime = time.time() - t_start
            summary[fits_path.name] = {
                "n_candidates": cand_counter,
                "runtime_s": runtime,
                "max_prob": prob_max,
                "mean_snr": float(np.mean(snr_list)) if snr_list else 0.0,
            }
            print(f"▶ {fits_path.name}: {cand_counter} candidatos, max prob {prob_max:.2f}, ⏱ {runtime:.1f} s")

    summary_path = save_path / "summary.json"
    with summary_path.open("w") as f_json:
        json.dump(summary, f_json, indent=2)
    print(f"Resumen global escrito en {summary_path}")
