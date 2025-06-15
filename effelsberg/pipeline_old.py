"""High level pipeline for FRB detection with CenterNet."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import config
from .candidate import Candidate
from .dedispersion import d_dm_time_g
from .image_utils import (
    compute_snr,
    pixel_to_physical,
    postprocess_img,
    preprocess_img,
    save_detection_plot,
)
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
            
            # Debug de dimensiones
            print(">> data.shape:", data.shape)
            print(">> FILE_LENG =", config.FILE_LENG)
            print(">> FREQ_RESO =", config.FREQ_RESO)
            print(">> DOWN_TIME_RATE =", config.DOWN_TIME_RATE)
            print(">> DOWN_FREQ_RATE =", config.DOWN_FREQ_RATE)
            print(">> FREQ =", config.FREQ)
            print(">> FREQ.shape =", config.FREQ.shape)
            print(">> DM_min =", config.DM_min)
            print(">> DM_max =", config.DM_max)
            print(">> TIME_RESO =", config.TIME_RESO)
            
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
            print("height =", height)
            print("width_total =", width_total)
            dm_time = d_dm_time_g(data, height=height, width=width_total)
            print("dm_time.shape =", dm_time.shape)

            slice_len = config.SLICE_LEN
            
            if width_total == 0:
                print("Advertencia: width_total es 0, no se pueden crear slices.")
                time_slice = 0
            elif width_total < slice_len:
                slice_len = width_total
                time_slice = 1
            else:
                time_slice = width_total // slice_len
            
            if time_slice == 0 and width_total > 0:
                time_slice = 1
                slice_len = width_total
            
            # Duración de cada slice
            slice_duration = slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
            print(f"Duración de cada slice: {slice_duration:.3f} segundos")
            print(f"Análisis de {fits_path.name} con {time_slice} slices de {slice_len} muestras cada uno.")

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
                slice_cube = dm_time[:, :, slice_len * j : slice_len * (j + 1)]
                for band_idx, band_suffix, band_name in band_configs:
                    band_img = slice_cube[band_idx]
                    img_tensor = preprocess_img(band_img)
                    with torch.no_grad():
                        hm, wh, offset = model(torch.from_numpy(img_tensor).to(config.DEVICE).float().unsqueeze(0))
                    top_conf, top_boxes = get_res(hm, wh, offset, confidence=det_prob)
                    if top_boxes is None:
                        continue
                    img_rgb = postprocess_img(img_tensor)
                    for conf, box in zip(top_conf, top_boxes):
                        dm_val, t_sec, t_sample = pixel_to_physical((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, slice_len)
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
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 220, 0), 1)
                    
                    # Enhanced plotting with matplotlib
                    out_img_path = save_path / f"{fits_path.stem}_slice{j}_{band_suffix}.png"
                    
                    # Create figure with scientific labels
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Show the image
                    im = ax.imshow(img_rgb, origin="lower", aspect='auto')
                    
                    # Configure axis labels with physical values
                    # X-axis (Time)
                    n_time_ticks = 6
                    time_positions = np.linspace(0, 512, n_time_ticks)
                    time_start_slice = j * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
                    time_values = time_start_slice + (time_positions / 512.0) * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
                    ax.set_xticks(time_positions)
                    ax.set_xticklabels([f"{t:.3f}" for t in time_values])
                    ax.set_xlabel("Time (s)", fontsize=12, fontweight='bold')
                    
                    # Y-axis (DM)
                    n_dm_ticks = 8
                    dm_positions = np.linspace(0, 512, n_dm_ticks)
                    dm_values = config.DM_min + (dm_positions / 512.0) * (config.DM_max - config.DM_min)
                    ax.set_yticks(dm_positions)
                    ax.set_yticklabels([f"{dm:.0f}" for dm in dm_values])
                    ax.set_ylabel("Dispersion Measure (pc cm⁻³)", fontsize=12, fontweight='bold')
                    
                    # Detailed scientific title
                    freq_range = f"{config.FREQ.min():.1f}–{config.FREQ.max():.1f} MHz"
                    title = (f"{fits_path.stem} - {band_name} ({freq_range})\n"
                            f"Slice {j+1}/{time_slice} | "
                            f"Time Resolution: {config.TIME_RESO*config.DOWN_TIME_RATE*1e6:.1f} μs | "
                            f"DM Range: {config.DM_min}–{config.DM_max} pc cm⁻³")
                    ax.set_title(title, fontsize=11, fontweight='bold', pad=20)
                    
                    # Add detection information as text
                    if top_boxes is not None and len(top_boxes) > 0:
                        detection_info = f"Detections: {len(top_boxes)}"
                        ax.text(0.02, 0.98, detection_info, transform=ax.transAxes, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                               fontsize=10, verticalalignment='top', fontweight='bold')
                    
                    # Add technical info in bottom right corner
                    tech_info = (f"Model: {config.MODEL_NAME.upper()}\n"
                                f"Confidence: {det_prob:.1f}\n"
                                f"Channels: {config.FREQ_RESO}→{config.FREQ_RESO//config.DOWN_FREQ_RATE}\n"
                                f"Time samples: {config.FILE_LENG}→{config.FILE_LENG//config.DOWN_TIME_RATE}")
                    ax.text(0.98, 0.02, tech_info, transform=ax.transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                           fontsize=8, verticalalignment='bottom', horizontalalignment='right')
                    
                    # Improve detection box annotations
                    if top_boxes is not None:
                        for idx, (conf, box) in enumerate(zip(top_conf, top_boxes)):
                            x1, y1, x2, y2 = map(int, box)
                            # Draw detection rectangle
                            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                               linewidth=2, edgecolor='lime', facecolor='none')
                            ax.add_patch(rect)
                            
                            # Calculate physical values for label
                            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                            dm_val, t_sec, t_sample = pixel_to_physical(center_x, center_y, slice_len)
                            
                            # Label with detection information
                            label = f"#{idx+1}\nDM: {dm_val:.1f}\nP: {conf:.2f}"
                            ax.annotate(label, xy=(center_x, center_y), 
                                       xytext=(center_x, y2 + 15),
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lime", alpha=0.8),
                                       fontsize=8, ha='center', fontweight='bold',
                                       arrowprops=dict(arrowstyle='->', color='lime', lw=1))
                    
                    # Add subtle grid for better readability
                    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                    
                    # Improve spacing and layout
                    plt.tight_layout()
                    
                    # Save with high resolution
                    plt.savefig(out_img_path, dpi=300, bbox_inches="tight", 
                               facecolor='white', edgecolor='none')
                    plt.close()
                    
                    # Optional: Save version with colorbar for full band
                    if band_suffix == "fullband":
                        fig_cb, ax_cb = plt.subplots(figsize=(13, 8))
                        
                        # Convert from RGB to grayscale for colorbar
                        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                        im_cb = ax_cb.imshow(img_gray, origin="lower", aspect='auto', cmap='mako')
                        
                        # Apply same labels
                        ax_cb.set_xticks(time_positions)
                        ax_cb.set_xticklabels([f"{t:.3f}" for t in time_values])
                        ax_cb.set_xlabel("Time (s)", fontsize=12, fontweight='bold')
                        
                        ax_cb.set_yticks(dm_positions)
                        ax_cb.set_yticklabels([f"{dm:.0f}" for dm in dm_values])
                        ax_cb.set_ylabel("Dispersion Measure (pc cm⁻³)", fontsize=12, fontweight='bold')
                        
                        ax_cb.set_title(title, fontsize=11, fontweight='bold', pad=20)
                        
                        # Add colorbar
                        cbar = plt.colorbar(im_cb, ax=ax_cb, shrink=0.8, pad=0.02)
                        cbar.set_label('Normalized Intensity', fontsize=10, fontweight='bold')
                        
                        # Add detections
                        if top_boxes is not None:
                            for idx, (conf, box) in enumerate(zip(top_conf, top_boxes)):
                                x1, y1, x2, y2 = map(int, box)
                                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                   linewidth=2, edgecolor='cyan', facecolor='none')
                                ax_cb.add_patch(rect)
                        
                        ax_cb.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                        plt.tight_layout()
                        
                        # Save version with colorbar
                        cb_path = save_path / f"{fits_path.stem}_slice{j}_{band_suffix}_colorbar.png"
                        plt.savefig(cb_path, dpi=300, bbox_inches="tight", 
                                   facecolor='white', edgecolor='none')
                        plt.close()

            runtime = time.time() - t_start
            summary[fits_path.name] = {
                "n_candidates": cand_counter,
                "runtime_s": runtime,
                "max_prob": float(prob_max),
                "mean_snr": float(np.mean(snr_list)) if snr_list else 0.0,
            }
            print(f"▶ {fits_path.name}: {cand_counter} candidatos, max prob {prob_max:.2f}, ⏱ {runtime:.1f} s")

    summary_path = save_path / "summary.json"
    with summary_path.open("w") as f_json:
        json.dump(summary, f_json, indent=2)
    print(f"Resumen global escrito en {summary_path}")