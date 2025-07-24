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
except ImportError:  
    torch = None
import numpy as np

from . import config
from ..io.candidate import Candidate
from ..io.candidate_utils import ensure_csv_header, append_candidate
from ..detection.dedispersion import d_dm_time_g, dedisperse_patch, dedisperse_block
from ..detection.metrics import compute_snr
from ..detection.astro_conversions import pixel_to_physical
from ..preprocessing.preprocessing import downsample_data
from ..visualization.image_utils import (
    preprocess_img,
    postprocess_img,
    plot_waterfall_block,
)
from ..io.io import get_obparams, load_fits_file
from ..io.filterbank_io import load_fil_file, get_obparams_fil
from ..preprocessing.slice_len_utils import update_slice_len_dynamic, get_slice_duration_info
from ..visualization.visualization import plot_waterfalls
from ..visualization.plot_manager import save_all_plots
from .summary_utils import (
    _write_summary,
    _update_summary_with_results,
    _update_summary_with_file_debug,
)
from ..detection.utils import detect, classify_patch
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





def _process_file(
    det_model: torch.nn.Module,
    cls_model: torch.nn.Module,
    fits_path: Path,
    save_dir: Path,
) -> dict:
    """Process a single FITS file and return summary information."""

    t_start = time.time()
    logger.info("Procesando %s", fits_path.name)

    # Inicialización de contadores para candidatos y bursts
    cand_counter = 0
    n_bursts = 0
    n_no_bursts = 0
    prob_max = 0.0


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

    # Definición de variables necesarias antes del bucle principal
    if config.FREQ_RESO == 0 or config.DOWN_FREQ_RATE == 0:
        raise ValueError(f"Parámetros de frecuencia inválidos: FREQ_RESO={config.FREQ_RESO}, DOWN_FREQ_RATE={config.DOWN_FREQ_RATE}")

    freq_down = np.mean(
        config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE),
        axis=1,
    )

    height = config.DM_max - config.DM_min + 1
    width_total = config.FILE_LENG // config.DOWN_TIME_RATE

    slice_len, real_duration_ms = update_slice_len_dynamic()
    time_slice = (width_total + slice_len - 1) // slice_len

    logger.info("✅ Sistema de slice simplificado:")
    logger.info(f"   🎯 Duración objetivo: {config.SLICE_DURATION_MS:.1f} ms")
    logger.info(f"   � SLICE_LEN calculado: {slice_len} muestras")
    logger.info(f"   ⏱️  Duración real obtenida: {real_duration_ms:.1f} ms")
    logger.info(f"   📊 Archivo: {config.FILE_LENG} muestras → {time_slice} slices")

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
    ensure_csv_header(csv_file)

    band_configs = (
        [
            (0, "fullband", "Full Band"),
            (1, "lowband", "Low Band"),
            (2, "highband", "High Band"),
        ]
        if config.USE_MULTI_BAND
        else [(0, "fullband", "Full Band")]
    )

    waterfall_dispersion_dir = save_dir / "waterfall_dispersion" / fits_path.stem
    waterfall_dedispersion_dir = save_dir / "waterfall_dedispersion" / fits_path.stem
    freq_ds = freq_down
    time_reso_ds = config.TIME_RESO * config.DOWN_TIME_RATE

    snr_list: List[float] = []
    for j in range(time_slice):
        slice_cube = dm_time[:, :, slice_len * j : slice_len * (j + 1)]
        waterfall_block = data[j * slice_len : (j + 1) * slice_len]
        # Verificación básica para arrays válidos
        if slice_cube.size == 0:
            logger.warning("Slice %d: slice_cube vacío, saltando...", j)
            continue
        if waterfall_block.size == 0:
            logger.warning("Slice %d: waterfall_block vacío, saltando...", j)
            continue
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
                absolute_start_time=None,
            )
        band_results = []
        slice_has_candidates = False
        for band_idx, band_suffix, band_name in band_configs:
            band_img = slice_cube[band_idx]
            img_tensor = preprocess_img(band_img)
            top_conf, top_boxes = detect(det_model, img_tensor)
            img_rgb = postprocess_img(img_tensor)
            if top_boxes is None:
                top_conf = []
                top_boxes = []
            first_patch = None
            first_start = None
            first_dm = None
            patch_dir = save_dir / "Patches" / fits_path.stem
            patch_path = patch_dir / f"patch_slice{j}_band{band_idx}.png"
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
                class_prob, proc_patch = classify_patch(cls_model, patch)
                class_probs_list.append(class_prob)
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
                append_candidate(csv_file, cand.to_row())
                logger.info(
                    "Candidato DM %.2f t=%.3f s conf=%.2f class=%.2f -> %s",
                    dm_val,
                    t_sec,
                    conf,
                    class_prob,
                    "BURST" if is_burst else "no burst",
                )
            if len(top_conf) > 0:
                slice_has_candidates = True
            # Modularizada: llamar a save_all_plots para cada banda
            composite_dir = save_dir / "Composite" / fits_path.stem
            comp_path = composite_dir / f"slice{j}_band{band_idx}.png"
            detections_dir = save_dir / "Detections" / fits_path.stem
            detections_dir.mkdir(parents=True, exist_ok=True)
            out_img_path = detections_dir / f"slice{j}_{band_suffix}.png"
            dedisp_block = None
            if slice_has_candidates:
                if first_patch is not None:
                    waterfall_dedispersion_dir.mkdir(parents=True, exist_ok=True)
                    start = j * slice_len
                    dedisp_block = dedisperse_block(data, freq_down, first_dm, start, slice_len)
                else:
                    waterfall_dedispersion_dir.mkdir(parents=True, exist_ok=True)
                    start = j * slice_len
                    dedisp_block = dedisperse_block(data, freq_down, 0.0, start, slice_len)
                save_all_plots(
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
                    off_regions=None,
                    thresh_snr=config.SNR_THRESH,
                    band_idx=band_idx,
                    patch_path=patch_path,
                    waterfall_dedispersion_dir=waterfall_dedispersion_dir,
                    freq_down=freq_down,
                    time_reso_ds=time_reso_ds,
                    detections_dir=detections_dir,
                    out_img_path=out_img_path
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

    print("=== INICIANDO PIPELINE DE DETECCIÓN DE FRB ===")
    print(f"Directorio de datos: {config.DATA_DIR}")
    print(f"Directorio de resultados: {config.RESULTS_DIR}")
    print(f"Targets FRB: {config.FRB_TARGETS}")
    
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    save_dir = config.RESULTS_DIR / config.MODEL_NAME
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Directorio de guardado: {save_dir}")
    
    print("Cargando modelos...")
    det_model = _load_model()
    cls_model = _load_class_model()
    print("Modelos cargados exitosamente")

    summary: dict[str, dict] = {}
    for frb in config.FRB_TARGETS:
        print(f"\nBuscando archivos para target: {frb}")
        file_list = _find_data_files(frb)
        print(f"Archivos encontrados: {[f.name for f in file_list]}")
        if not file_list:
            print(f"No se encontraron archivos para {frb}")
            continue

        try:
            first_file = file_list[0]
            print(f"Leyendo parámetros de observación desde: {first_file.name}")
            if first_file.suffix.lower() == ".fits":
                get_obparams(str(first_file))
            else:
                get_obparams_fil(str(first_file))
            print("Parámetros de observación cargados exitosamente")
        except Exception as e:
            print(f"[ERROR] Error obteniendo parámetros de observación: {e}")
            logger.error("Error obteniendo parámetros de observación: %s", e)
            continue
            
        for fits_path in file_list:
            try:
                print(f"Procesando archivo: {fits_path.name}")
                results = _process_file(det_model, cls_model, fits_path, save_dir)
                summary[fits_path.name] = results
                
                # *** GUARDAR RESULTADOS INMEDIATAMENTE EN SUMMARY.JSON ***
                _update_summary_with_results(save_dir, fits_path.stem, {
                    "n_candidates": results.get("n_candidates", 0),
                    "n_bursts": results.get("n_bursts", 0),
                    "n_no_bursts": results.get("n_no_bursts", 0),
                    "processing_time": results.get("runtime_s", 0.0),
                    "max_detection_prob": results.get("max_prob", 0.0),
                    "mean_snr": results.get("mean_snr", 0.0),
                    "status": results.get("status", "completed")
                })
                
                print(f"Archivo {fits_path.name} procesado exitosamente")
            except Exception as e:
                print(f"[ERROR] Error procesando {fits_path.name}: {e}")
                logger.error("Error procesando %s: %s", fits_path.name, e)
                error_results = {
                    "n_candidates": 0,
                    "n_bursts": 0,
                    "n_no_bursts": 0,
                    "runtime_s": 0,
                    "max_prob": 0.0,
                    "mean_snr": 0.0,
                    "status": "ERROR"
                }
                summary[fits_path.name] = error_results
                
                # *** GUARDAR RESULTADOS DE ERROR INMEDIATAMENTE ***
                _update_summary_with_results(save_dir, fits_path.stem, {
                    "n_candidates": 0,
                    "n_bursts": 0,
                    "n_no_bursts": 0,
                    "processing_time": 0.0,
                    "max_detection_prob": 0.0,
                    "mean_snr": 0.0,
                    "status": "ERROR",
                    "error_message": str(e)
                })

    print("Escribiendo resumen final...")
    _write_summary(summary, save_dir)
    print("=== PIPELINE COMPLETADO ===")

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
    
    