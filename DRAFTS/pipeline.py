"""High level pipeline for FRB detection with CenterNet."""
from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import List

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None
import numpy as np

from . import config
from .candidate import Candidate
from .dedispersion import d_dm_time_g, dedisperse_patch, dedisperse_block
from .metrics import compute_snr
from .astro_conversions import pixel_to_physical
from .preprocessing import downsample_data
from .image_utils import (
    preprocess_img,
    postprocess_img,
    plot_waterfall_block,
)
from .io import get_obparams, load_fits_file
from .filterbank_io import load_fil_file, get_obparams_fil
from .visualization import (
    save_plot,
    save_patch_plot,
    save_slice_summary,
    plot_waterfalls,
)
logger = logging.getLogger(__name__)


def _load_model() -> torch.nn.Module:
    """Load the CenterNet model configured in :mod:`config`."""
    if torch is None:
        raise ImportError("torch is required to load models")

    from ObjectDet.centernet_model import centernet
    model = centernet(model_name=config.MODEL_NAME).to(config.DEVICE)
    state = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

def _load_class_model() -> torch.nn.Module:
    """Load the binary classification model configured in :mod:`config`."""
    if torch is None:
        raise ImportError("torch is required to load models")

    from BinaryClass.binary_model import BinaryNet
    model = BinaryNet(config.CLASS_MODEL_NAME, num_classes=2).to(config.DEVICE)
    state = torch.load(config.CLASS_MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

def _find_data_files(frb: str) -> List[Path]:
    """Return FITS or filterbank files matching ``frb`` within ``config.DATA_DIR``."""

    files = list(config.DATA_DIR.glob("*.fits")) + list(config.DATA_DIR.glob("*.fil"))
    return sorted(f for f in files if frb in f.name)


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
        logger.info("Usando archivo alternativo: %s", alt_csv)
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
    from ObjectDet.centernet_utils import get_res

    with torch.no_grad():
        hm, wh, offset = model(
            torch.from_numpy(img_tensor)
            .to(config.DEVICE)
            .float()
            .unsqueeze(0)
        )
    return get_res(hm, wh, offset, confidence=config.DET_PROB)


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

def _write_summary(summary: dict, save_path: Path) -> None:
    """Write global summary information to ``summary.json``.

    Each entry in ``summary`` now includes ``n_bursts`` and ``n_no_bursts``
    indicating how many classified bursts and non-bursts were found in a
    given FITS file.
    """

    summary_path = save_path / "summary.json"
    with summary_path.open("w") as f_json:
        json.dump(summary, f_json, indent=2)
    logger.info("Resumen global escrito en %s", summary_path)

def _process_file(
    det_model: torch.nn.Module,
    cls_model: torch.nn.Module,
    fits_path: Path,
    save_dir: Path,
) -> dict:
    """Process a single FITS file and return summary information."""

    t_start = time.time()
    logger.info("Procesando %s", fits_path.name)

    try:
        if fits_path.suffix.lower() == ".fits":
            data = load_fits_file(str(fits_path))
        else:
            data = load_fil_file(str(fits_path))
    except ValueError as e:
        if "corrupto" in str(e).lower():
            logger.error("Archivo corrupto detectado: %s - SALTANDO", fits_path.name)
            return {
                "n_candidates": 0,
                "n_bursts": 0,
                "n_no_bursts": 0,
                "runtime_s": time.time() - t_start,
                "max_prob": 0.0,
                "mean_snr": 0.0,
                "status": "CORRUPTED_FILE"
            }
        else:
            raise  # Re-lanzar si es otro tipo de error
    
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
    n_bursts = 0
    n_no_bursts = 0
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
                normalize=True,
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
            
            # Lista para almacenar las probabilidades de clasificación
            class_probs_list = []

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
                class_probs_list.append(class_prob)  # Agregar a la lista
                
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
                if is_burst:
                    n_bursts += 1
                else:
                    n_no_bursts += 1
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
                        normalize=True,
                    )

                save_patch_plot(
                    first_patch,
                    patch_path,
                    freq_down,
                    config.TIME_RESO * config.DOWN_TIME_RATE,
                    first_start,
                )

                # 1) Generar composite
                composite_dir = save_dir / "Composite" / fits_path.stem
                comp_path = composite_dir / f"slice{j}_band{band_idx}.png"
                save_slice_summary(
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
                    fits_path.stem,
                    slice_len,
                    normalize=True,
                )

            # 4) Generar detecciones de Bow ties (detections)
            out_img_path = save_dir / f"{fits_path.stem}_slice{j}_{band_suffix}.png"
            save_plot(
                img_rgb,
                top_conf,
                top_boxes,
                class_probs_list,   
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
        "n_bursts": n_bursts,
        "n_no_bursts": n_no_bursts,
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
        file_list = _find_data_files(frb)
        if not file_list:
            continue

        try:
            first_file = file_list[0]
            if first_file.suffix.lower() == ".fits":
                get_obparams(str(first_file))
            else:
                get_obparams_fil(str(first_file))
        except Exception as e:
            logger.error("Error obteniendo parámetros de observación: %s", e)
            continue
            
        for fits_path in file_list:
            try:
                summary[fits_path.name] = _process_file(det_model, cls_model, fits_path, save_dir)
            except Exception as e:
                logger.error("Error procesando %s: %s", fits_path.name, e)
                summary[fits_path.name] = {
                    "n_candidates": 0,
                    "n_bursts": 0,
                    "n_no_bursts": 0,
                    "runtime_s": 0,
                    "max_prob": 0.0,
                    "mean_snr": 0.0,
                    "status": "ERROR"
                }

    _write_summary(summary, save_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FRB detection pipeline")
    parser.add_argument("--data-dir", type=Path, help="Directory with FITS files")
    parser.add_argument("--results-dir", type=Path, help="Directory for results")
    parser.add_argument("--det-model", type=Path, help="Detection model path")
    parser.add_argument("--class-model", type=Path, help="Classification model path")
    args = parser.parse_args()

    if args.data_dir:
        config.DATA_DIR = args.data_dir
    if args.results_dir:
        config.RESULTS_DIR = args.results_dir
    if args.det_model:
        config.MODEL_PATH = args.det_model
    if args.class_model:
        config.CLASS_MODEL_PATH = args.class_model

    run_pipeline()