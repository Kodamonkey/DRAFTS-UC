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
from .rfi_mitigation import RFIMitigator
from .image_utils import (
    preprocess_img,
    postprocess_img,
    plot_waterfall_block,
)
from .io import get_obparams, load_fits_file
from .filterbank_io import load_fil_file, get_obparams_fil
from .slice_len_utils import update_slice_len_dynamic, get_slice_duration_info
from .visualization import (
    save_plot,
    save_patch_plot,
    save_slice_summary,
    plot_waterfalls,
)
logger = logging.getLogger(__name__)


def apply_rfi_cleaning(
    waterfall: np.ndarray,
    stokes_v: np.ndarray | None = None,
    output_dir: Path | None = None
) -> tuple[np.ndarray, dict]:
    """
    Aplica limpieza de RFI al waterfall DM vs. Time.
    
    Parameters
    ----------
    waterfall : np.ndarray
        Datos waterfall (tiempo, frecuencia/DM)
    stokes_v : np.ndarray, optional
        Datos de polarización circular
    output_dir : Path, optional
        Directorio para guardar diagnósticos
        
    Returns
    -------
    tuple[np.ndarray, dict]
        Waterfall limpio y estadísticas de RFI
    """
    if not hasattr(config, 'RFI_ENABLE_ALL_FILTERS') or not config.RFI_ENABLE_ALL_FILTERS:
        # RFI cleaning deshabilitado
        return waterfall, {}
    
    print("[INFO] Aplicando limpieza de RFI...")
    
    # Configura el mitigador de RFI
    rfi_mitigator = RFIMitigator(
        freq_sigma_thresh=getattr(config, 'RFI_FREQ_SIGMA_THRESH', 5.0),
        time_sigma_thresh=getattr(config, 'RFI_TIME_SIGMA_THRESH', 5.0),
        zero_dm_sigma_thresh=getattr(config, 'RFI_ZERO_DM_SIGMA_THRESH', 4.0),
        impulse_sigma_thresh=getattr(config, 'RFI_IMPULSE_SIGMA_THRESH', 6.0),
        polarization_thresh=getattr(config, 'RFI_POLARIZATION_THRESH', 0.8)
    )
    
    # Aplica limpieza
    cleaned_waterfall, rfi_stats = rfi_mitigator.clean_waterfall(
        waterfall,
        stokes_v=stokes_v,
        apply_all_filters=True
    )
    
    # Guarda diagnósticos si está configurado
    if (getattr(config, 'RFI_SAVE_DIAGNOSTICS', False) and 
        output_dir and 
        rfi_stats.get('total_flagged_fraction', 0) > 0.001):
        
        rfi_dir = output_dir / "rfi_diagnostics"
        rfi_dir.mkdir(parents=True, exist_ok=True)
        
        diagnostic_path = rfi_dir / "rfi_cleaning_diagnostics.png"
        rfi_mitigator.plot_rfi_diagnostics(
            waterfall, cleaned_waterfall, diagnostic_path
        )
    
    return cleaned_waterfall, rfi_stats


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
                    "t_sec_absolute",  # ✅ Actualizado: tiempo absoluto desde inicio archivo
                    "t_sample",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "snr_peak",       # ✅ Actualizado: SNR calculado igual que plots
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


def _detect(model, img_tensor: np.ndarray) -> tuple[list, list | None]:
    """Run the detection model and return confidences and boxes."""
    from ObjectDet.centernet_utils import get_res

    try:
        with torch.no_grad():
            hm, wh, offset = model(
                torch.from_numpy(img_tensor)
                .to(config.DEVICE)
                .float()
                .unsqueeze(0)
            )
        
        top_conf, top_boxes = get_res(hm, wh, offset, confidence=config.DET_PROB)
        
        # Validar resultados
        if top_boxes is None:
            return [], None
            
        if not isinstance(top_conf, (list, np.ndarray)):
            logger.warning(f"top_conf no es lista/array: {type(top_conf)}")
            return [], None
            
        if not isinstance(top_boxes, (list, np.ndarray)):
            logger.warning(f"top_boxes no es lista/array: {type(top_boxes)}")
            return [], None
            
        # Convertir a listas si es necesario
        if isinstance(top_conf, np.ndarray):
            top_conf = top_conf.tolist()
        if isinstance(top_boxes, np.ndarray):
            top_boxes = top_boxes.tolist()
            
        return top_conf, top_boxes
        
    except Exception as e:
        logger.error(f"Error en _detect: {e}")
        return [], None


def _prep_patch(patch: np.ndarray) -> np.ndarray:
    """Normalize patch for classification."""

    patch = patch.copy()
    patch += 1
    patch /= np.mean(patch, axis=0)
    vmin, vmax = np.nanpercentile(patch, [5, 95])
    patch = np.clip(patch, vmin, vmax)
    patch = (patch - patch.min()) / (patch.max() - patch.min())
    return patch


def _classify_patch(model, patch: np.ndarray) -> tuple[float, np.ndarray]:
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

    # Detectar si necesitamos procesamiento por chunks
    if getattr(config, 'ENABLE_CHUNK_PROCESSING', True):
        # Check if we have the original file size stored from parameter reading
        total_samples_estimated = getattr(config, '_ORIGINAL_FILE_SAMPLES', 0)
        
        if total_samples_estimated > config.MAX_SAMPLES_LIMIT:
            logger.info(f"Archivo grande detectado: {total_samples_estimated:,} muestras")
            logger.info(f"Usando procesamiento por chunks (límite: {config.MAX_SAMPLES_LIMIT:,})")
            
            return _process_file_in_chunks(
                det_model, cls_model, fits_path, save_dir,
                total_samples_estimated, 
                config.MAX_SAMPLES_LIMIT,
                getattr(config, 'CHUNK_OVERLAP_SAMPLES', 1000)
            )
        
        # Fallback: try to read header directly if not stored
        if fits_path.suffix.lower() == ".fil":
            from .filterbank_io import _read_header
            try:
                with open(str(fits_path), "rb") as f:
                    header, _ = _read_header(f)
                total_samples_estimated = header.get("nsamples", 0)
                
                if total_samples_estimated > config.MAX_SAMPLES_LIMIT:
                    logger.info(f"Archivo grande detectado: {total_samples_estimated:,} muestras")
                    logger.info(f"Usando procesamiento por chunks (límite: {config.MAX_SAMPLES_LIMIT:,})")
                    
                    return _process_file_in_chunks(
                        det_model, cls_model, fits_path, save_dir,
                        total_samples_estimated, 
                        config.MAX_SAMPLES_LIMIT,
                        getattr(config, 'CHUNK_OVERLAP_SAMPLES', 1000)
                    )
            except Exception as e:
                logger.warning(f"No se pudo determinar tamaño del archivo {fits_path.name}: {e}")
                logger.info("Continuando con procesamiento estándar")

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

    # Verificación básica para evitar errores con arrays vacíos
    if config.FREQ_RESO == 0 or config.DOWN_FREQ_RATE == 0:
        raise ValueError(f"Parámetros de frecuencia inválidos: FREQ_RESO={config.FREQ_RESO}, DOWN_FREQ_RATE={config.DOWN_FREQ_RATE}")
    
    freq_down = np.mean(
        config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE),
        axis=1,
    )

    height = config.DM_max - config.DM_min + 1
    width_total = config.FILE_LENG // config.DOWN_TIME_RATE

    # 🚀 NUEVO SISTEMA SIMPLIFICADO: SIEMPRE usar SLICE_DURATION_MS
    # Calcular SLICE_LEN dinámicamente después de cargar metadatos del archivo
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
    
    # Reutilizar freq_down ya calculado anteriormente para evitar problemas
    freq_ds = freq_down
    time_reso_ds = config.TIME_RESO * config.DOWN_TIME_RATE

    for j in range(time_slice):
        slice_cube = dm_time[:, :, slice_len * j : slice_len * (j + 1)]
        waterfall_block = data[j * slice_len : (j + 1) * slice_len]
        
        # Verificación básica para arrays válidos
        if waterfall_block.size == 0 or slice_cube.size == 0:
            logger.warning("Slice %d tiene arrays vacíos, saltando...", j)
            continue
        
        # Aplicar limpieza de RFI si está habilitada
        rfi_stats = {}
        if hasattr(config, 'RFI_ENABLE_ALL_FILTERS') and config.RFI_ENABLE_ALL_FILTERS:
            try:
                logger.info("Aplicando limpieza de RFI al slice %d", j)
                waterfall_block, rfi_stats = apply_rfi_cleaning(
                    waterfall_block,
                    stokes_v=None,  # No hay datos de polarización en este punto
                    output_dir=save_dir / "rfi_diagnostics" if getattr(config, 'RFI_SAVE_DIAGNOSTICS', False) else None
                )
                logger.info("RFI cleaning completado para slice %d: %.2f%% datos flagged", 
                           j, rfi_stats.get('total_flagged_fraction', 0) * 100)
            except Exception as e:
                logger.warning("Error en limpieza de RFI para slice %d: %s", j, e)
                # Continúa con datos originales si hay error
        
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

        # Recopilar información de todas las bandas primero
        band_results = []
        slice_has_candidates = False
        
        for band_idx, band_suffix, band_name in band_configs:
            band_img = slice_cube[band_idx]
            img_tensor = preprocess_img(band_img)
            top_conf, top_boxes = _detect(det_model, img_tensor)
            
            # Siempre generar img_rgb para visualización, incluso sin detecciones
            img_rgb = postprocess_img(img_tensor)
            
            # Si no hay detecciones, crear listas vacías pero continuar con visualizaciones
            if top_boxes is None:
                top_conf = []
                top_boxes = []

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
                
                # ✅ CORRECCIÓN: Calcular SNR mejorado y tiempo absoluto
                snr_peak, t_sec_absolute = _calculate_improved_snr_and_time(
                    waterfall_block,
                    tuple(map(int, box)),
                    j,
                    int(t_sample),
                    slice_len
                )
                
                # Mantener cálculo original para compatibilidad
                snr_val = compute_snr(band_img, tuple(map(int, box)))
                snr_list.append(snr_peak)  # Usar SNR mejorado para la lista
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
                    snr_val,  # SNR original para compatibilidad
                    class_prob,
                    is_burst,
                    patch_path.name,
                    t_sec_absolute,  # ✅ Tiempo absoluto
                    snr_peak,       # ✅ SNR mejorado (mismo que plots)
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
            
            # Marcar si este slice tiene candidatos
            if len(top_conf) > 0:
                slice_has_candidates = True
            
            # Almacenar información de esta banda para procesamiento posterior
            band_results.append({
                'band_idx': band_idx,
                'band_suffix': band_suffix,
                'band_name': band_name,
                'band_img': band_img,
                'img_rgb': img_rgb,
                'top_conf': top_conf,
                'top_boxes': top_boxes,
                'class_probs_list': class_probs_list,
                'first_patch': first_patch,
                'first_start': first_start,
                'first_dm': first_dm,
                'patch_path': patch_path
            })

        # Ahora generar visualizaciones para todas las bandas si AL MENOS UNA tiene candidatos
        for band_result in band_results:
            band_idx = band_result['band_idx']
            band_suffix = band_result['band_suffix']
            band_name = band_result['band_name']
            img_rgb = band_result['img_rgb']
            top_conf = band_result['top_conf']
            top_boxes = band_result['top_boxes']
            class_probs_list = band_result['class_probs_list']
            first_patch = band_result['first_patch']
            first_start = band_result['first_start']
            first_dm = band_result['first_dm']
            patch_path = band_result['patch_path']

            
            # Solo generar visualizaciones complejas si AL MENOS UNA banda en este slice tiene candidatos
            if slice_has_candidates:
                # Preparar valores por defecto para casos sin detecciones en esta banda específica
                dedisp_block = None
                if first_patch is not None:
                    # 3) Generar waterfall dedispersado para este slice con el primer candidato
                    waterfall_dedispersion_dir.mkdir(parents=True, exist_ok=True)
                    start = j * slice_len
                    dedisp_block = dedisperse_block(data, freq_down, first_dm, start, slice_len)
                    
                    # Aplicar limpieza de RFI al bloque dedispersado si está habilitada
                    if hasattr(config, 'RFI_ENABLE_ALL_FILTERS') and config.RFI_ENABLE_ALL_FILTERS and dedisp_block.size > 0:
                        try:
                            logger.debug("Aplicando limpieza de RFI al bloque dedispersado slice %d", j)
                            dedisp_block_clean, rfi_stats_dedisp = apply_rfi_cleaning(
                                dedisp_block,
                                stokes_v=None,
                                output_dir=None  # No guardar diagnósticos para cada bloque
                            )
                            dedisp_block = dedisp_block_clean
                            logger.debug("RFI cleaning en bloque dedispersado completado: %.2f%% datos flagged", 
                                       rfi_stats_dedisp.get('total_flagged_fraction', 0) * 100)
                        except Exception as e:
                            logger.warning("Error en limpieza de RFI para bloque dedispersado slice %d: %s", j, e)
                            # Continúa con datos originales si hay error
                
                if dedisp_block is not None and dedisp_block.size > 0:
                    plot_waterfall_block(
                        data_block=dedisp_block,
                        freq=freq_down,
                        time_reso=time_reso_ds,
                        block_size=dedisp_block.shape[0],
                        block_idx=j,
                        save_dir=waterfall_dedispersion_dir,
                        filename=f"{fits_path.stem}_dm{first_dm:.2f}_{band_suffix}",
                        normalize=True,
                    )

                if first_patch is not None:
                    save_patch_plot(
                        first_patch,
                        patch_path,
                        freq_down,
                        config.TIME_RESO * config.DOWN_TIME_RATE,
                        first_start,
                        off_regions=None,  # Use IQR method for robust estimation
                        thresh_snr=config.SNR_THRESH,
                        band_idx=band_idx,  # Pasar el índice de la banda
                        band_name=band_name,  # Pasar el nombre de la banda
                    )
                else:
                    # Para bandas sin detecciones, crear un parche dummy
                    waterfall_dedispersion_dir.mkdir(parents=True, exist_ok=True)
                    start = j * slice_len
                    # Usar DM=0 para banda sin detecciones
                    dedisp_block = dedisperse_block(data, freq_down, 0.0, start, slice_len)
                    
                    if dedisp_block.size > 0:
                        plot_waterfall_block(
                            data_block=dedisp_block,
                            freq=freq_down,
                            time_reso=time_reso_ds,
                            block_size=dedisp_block.shape[0],
                            block_idx=j,
                            save_dir=waterfall_dedispersion_dir,
                            filename=f"{fits_path.stem}_dm0.00_{band_suffix}",
                            normalize=True,
                        )

                # 1) Generar composite - SIEMPRE para comparativas si hay candidatos en este slice
                composite_dir = save_dir / "Composite" / fits_path.stem
                comp_path = composite_dir / f"slice{j}_band{band_idx}.png"
                save_slice_summary(
                    waterfall_block,
                    dedisp_block if dedisp_block is not None and dedisp_block.size > 0 else waterfall_block,  # fallback a waterfall original
                    img_rgb,
                    first_patch,
                    first_start if first_start is not None else 0.0,
                    first_dm if first_dm is not None else 0.0,
                    top_conf if len(top_conf) > 0 else [],
                    top_boxes if len(top_boxes) > 0 else [],
                    class_probs_list, 
                    comp_path,
                    j,
                    time_slice,
                    band_name,
                    band_suffix,
                    fits_path.stem,
                    slice_len,
                    normalize=True,
                    off_regions=None,  # Use IQR method
                    thresh_snr=config.SNR_THRESH,
                    band_idx=band_idx,  # Pasar el índice de la banda
                    )

                # 4) Generar detecciones de Bow ties (detections) - SIEMPRE
                detections_dir = save_dir / "Detections" / fits_path.stem
                detections_dir.mkdir(parents=True, exist_ok=True)
                out_img_path = detections_dir / f"slice{j}_{band_suffix}.png"
                save_plot(
                    img_rgb,
                    top_conf if len(top_conf) > 0 else [],
                    top_boxes if len(top_boxes) > 0 else [],
                    class_probs_list,   
                    out_img_path,
                    j,
                    time_slice,
                    band_name,
                    band_suffix,
                    fits_path.stem,
                    slice_len,
                    band_idx=band_idx,  # Pasar el índice de la banda
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

def _load_extended_data_for_dedispersion(fits_path: Path, start_sample_global: int, chunk_size: int, dm_val: float) -> np.ndarray:
    """
    Cargar datos extendidos del archivo original para dedispersión correcta.
    La dedispersión requiere contexto temporal más amplio que un chunk.
    """
    # Calcular rango extendido necesario para dedispersión
    # El delay máximo de dispersión determina cuántos datos adicionales necesitamos
    max_freq = np.max(config.FREQ) if config.FREQ is not None else 1500.0  # MHz
    min_freq = np.min(config.FREQ) if config.FREQ is not None else 1200.0  # MHz
    
    # Calcular delay máximo (en muestras)
    delay_constant = 4.148808  # ms * MHz^2 / (pc cm^-3)
    max_delay_ms = delay_constant * dm_val * (1.0/(min_freq**2) - 1.0/(max_freq**2))
    max_delay_samples = int(max_delay_ms / (config.TIME_RESO * 1000)) + 100  # +100 buffer
    
    # Extender el rango para incluir el delay
    extended_start = max(0, start_sample_global - max_delay_samples)
    extended_end = min(config.FILE_LENG, start_sample_global + chunk_size + max_delay_samples)
    extended_size = extended_end - extended_start
    
    logger.debug(f"Cargando datos extendidos: muestras {extended_start} a {extended_end} (DM={dm_val:.1f})")
    
    try:
        if fits_path.suffix.lower() == ".fits":
            # Para FITS, cargar todo y extraer región
            full_data = load_fits_file(str(fits_path))
            extended_data = full_data[extended_start:extended_end]
        else:
            # Para .fil, cargar solo la región extendida
            extended_data = _load_fil_chunk(str(fits_path), extended_start, extended_size)
        
        return extended_data, extended_start
        
    except Exception as e:
        logger.warning(f"Error cargando datos extendidos: {e}. Usando chunk original.")
        # Fallback al chunk original si hay problemas
        return None, extended_start

def _load_fil_chunk(file_path: str, start_sample: int, chunk_size: int) -> np.ndarray:
    """Load a specific chunk from a filterbank file."""
    from .filterbank_io import _read_header
    
    # Leer header para obtener parámetros
    with open(file_path, "rb") as f:
        header, hdr_len = _read_header(f)
    
    nchans = header["nchans"]
    nbits = header["nbits"]
    nifs = header.get("nifs", 1)
    
    # Calcular bytes por muestra
    bytes_per_sample = nifs * nchans * (nbits // 8)
    
    # Calcular offset en el archivo
    data_start_offset = hdr_len + start_sample * bytes_per_sample
    bytes_to_read = chunk_size * bytes_per_sample
    
    # Determinar dtype
    dtype = np.uint8
    if nbits == 16:
        dtype = np.int16
    elif nbits == 32:
        dtype = np.float32
    elif nbits == 64:
        dtype = np.float64
    
    # Leer chunk específico
    with open(file_path, "rb") as f:
        f.seek(data_start_offset)
        raw_data = f.read(bytes_to_read)
    
    # Convertir a array numpy
    data_flat = np.frombuffer(raw_data, dtype=dtype)
    
    # Reorganizar en formato (tiempo, nifs, frecuencia) para coincidir con load_fil_file
    if len(data_flat) == chunk_size * nchans * nifs:
        data = data_flat.reshape(chunk_size, nifs, nchans)
    else:
        # Manejar caso donde no se pudo leer la cantidad exacta
        available_samples = len(data_flat) // (nchans * nifs)
        data = data_flat[:available_samples * nchans * nifs].reshape(available_samples, nifs, nchans)
    
    # Apply reversal if needed (same logic as load_fil_file)
    if getattr(config, 'DATA_NEEDS_REVERSAL', False):
        data = data[:, :, ::-1]
    
    return data

def _process_single_chunk(
    det_model,
    cls_model, 
    data_chunk: np.ndarray,
    fits_path: Path,
    save_dir: Path,
    chunk_idx: int,
    start_sample_global: int,
    csv_file: Path,
    total_samples: int = None,  # ✅ NUEVO: Total de muestras del archivo completo
) -> dict:
    """Process a single chunk of data with GLOBAL context for correct plotting."""
    
    # Esta función contiene la lógica de procesamiento principal
    # extraída de _process_file, adaptada para chunks
    
    from .preprocessing import downsample_data
    from .dedispersion import d_dm_time_g, dedisperse_patch, dedisperse_block
    from .image_utils import preprocess_img, postprocess_img, plot_waterfall_block
    from .visualization import save_plot, save_patch_plot, save_slice_summary
    import time
    
    t_start = time.time()
    
    # Aplicar el mismo preprocessing que en _process_file
    data_chunk = np.vstack([data_chunk, data_chunk[::-1, :]])
    data_chunk = downsample_data(data_chunk)
    
    # Calcular parámetros para este chunk
    height = config.DM_max - config.DM_min + 1
    width_total = data_chunk.shape[0] // config.DOWN_TIME_RATE
    
    # 🚀 NUEVO SISTEMA SIMPLIFICADO: usar SLICE_LEN ya calculado dinámicamente
    slice_len = config.SLICE_LEN  # Ya actualizado por update_slice_len_dynamic()
    time_slice = (width_total + slice_len - 1) // slice_len
    
    # ✅ CORRECCIÓN CRÍTICA 1: Calcular contexto global del archivo completo
    if total_samples is not None:
        total_samples_ds = total_samples // config.DOWN_TIME_RATE
        total_time_slice = (total_samples_ds + slice_len - 1) // slice_len
    else:
        total_time_slice = time_slice  # Fallback si no se proporciona
    
    # ✅ CORRECCIÓN CRÍTICA 2: Calcular slice inicial global
    start_sample_global_ds = start_sample_global // config.DOWN_TIME_RATE  
    start_slice_global = start_sample_global_ds // slice_len
    
    duration_ms, duration_text = get_slice_duration_info(slice_len)
    logger.info(f"Chunk {chunk_idx}: usando {slice_len} muestras = {duration_text}")
    logger.info(f"Chunk {chunk_idx}: slice global inicial = {start_slice_global}, total slices archivo = {total_time_slice}")
    
    # Generar DM vs tiempo para este chunk
    dm_time = d_dm_time_g(data_chunk, height=height, width=width_total)
    
    # Contadores para este chunk
    chunk_candidates = 0
    chunk_bursts = 0
    chunk_no_bursts = 0
    chunk_max_prob = 0.0
    chunk_snr_list = []
    
    # Configurar bandas
    band_configs = (
        [
            (0, "fullband", "Full Band"),
            (1, "lowband", "Low Band"), 
            (2, "highband", "High Band"),
        ]
        if config.USE_MULTI_BAND
        else [(0, "fullband", "Full Band")]
    )
    
    # Procesar slices en este chunk
    for j in range(time_slice):
        # ✅ CORRECCIÓN CRÍTICA 3: Calcular slice_idx GLOBAL para visualizaciones
        slice_idx_global = start_slice_global + j
        
        slice_cube = dm_time[:, :, slice_len * j : slice_len * (j + 1)]
        waterfall_block = data_chunk[j * slice_len : (j + 1) * slice_len]
        
        if slice_cube.size == 0:
            continue
            
        # Verificación básica para arrays válidos
        if waterfall_block.size == 0 or slice_cube.size == 0:
            logger.warning(f"Chunk {chunk_idx}, slice {j} (global: {slice_idx_global}) tiene arrays vacíos, saltando...")
            continue
        
        # Preparar directorios para waterfalls individuales  
        waterfall_dispersion_dir = save_dir / "waterfall_dispersion" / f"{fits_path.stem}_chunk{chunk_idx}"
        waterfall_dedispersion_dir = save_dir / "waterfall_dedispersion" / f"{fits_path.stem}_chunk{chunk_idx}"
        
        # Calcular freq_down para este chunk
        freq_ds = np.mean(
            config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE),
            axis=1,
        )
        time_reso_ds = config.TIME_RESO * config.DOWN_TIME_RATE
        
        # ✅ CORRECCIÓN: Generar waterfall SIN dedispersar (datos crudos con dispersión visible)
        waterfall_dispersion_dir.mkdir(parents=True, exist_ok=True)
        if waterfall_block.size > 0:
            plot_waterfall_block(
                data_block=waterfall_block,  # ✅ CORRECTO - Datos crudos del chunk (con dispersión)
                freq=freq_ds,
                time_reso=time_reso_ds,
                block_size=waterfall_block.shape[0],
                block_idx=slice_idx_global,  # ✅ Usar índice global
                save_dir=waterfall_dispersion_dir,
                filename=f"{fits_path.stem}_chunk{chunk_idx}",
                normalize=True,
            )

        # Recopilar información de todas las bandas primero
        band_results = []
        slice_has_candidates = False
        
        for band_idx, band_suffix, band_name in band_configs:
            band_img = slice_cube[band_idx]
            img_tensor = preprocess_img(band_img)
            top_conf, top_boxes = _detect(det_model, img_tensor)
            
            # Siempre generar img_rgb para visualización, incluso sin detecciones
            img_rgb = postprocess_img(img_tensor)
            
            if top_boxes is None:
                top_conf = []
                top_boxes = []
            
            # Debug: Log detection info
            logger.debug(f"Chunk {chunk_idx}, slice {j}, band {band_idx}: Detecciones encontradas: {len(top_boxes)}")
            if len(top_boxes) > 0:
                logger.debug(f"Primera detección - conf: {top_conf[0]}, box: {top_boxes[0]}, box_type: {type(top_boxes[0])}")
                
            # Validar que top_conf y top_boxes tengan la misma longitud
            if len(top_conf) != len(top_boxes):
                logger.warning(f"Chunk {chunk_idx}, slice {j}, band {band_idx}: Longitud de confianzas ({len(top_conf)}) != longitud de boxes ({len(top_boxes)})")
                continue

            first_patch: np.ndarray | None = None
            first_start: float | None = None
            first_dm: float | None = None
            patch_dir = save_dir / "Patches" / f"{fits_path.stem}_chunk{chunk_idx}"
            patch_path = patch_dir / f"patch_slice{j}_band{band_idx}.png"
            
            # Lista para almacenar las probabilidades de clasificación
            class_probs_list = []
                
            for conf_idx, (conf, box) in enumerate(zip(top_conf, top_boxes)):
                try:
                    # Validar formato de box
                    if len(box) != 4:
                        logger.warning(f"Chunk {chunk_idx}, slice {j}, band {band_idx}, box {conf_idx}: Box inválido (len={len(box)}): {box}")
                        continue
                    
                    # Calcular posición global considerando el chunk
                    box_center_x = (box[0] + box[2]) / 2
                    box_center_y = (box[1] + box[3]) / 2
                    
                    dm_val, t_sec, t_sample = pixel_to_physical(
                        box_center_x,
                        box_center_y,
                        slice_len,
                    )
                    
                    # Ajustar tiempo global
                    global_sample = start_sample_global + j * slice_len + int(t_sample)
                    global_t_sec = global_sample * config.TIME_RESO * config.DOWN_TIME_RATE
                    
                    # Validar box para compute_snr
                    box_int = tuple(map(int, box))
                    if len(box_int) != 4:
                        logger.warning(f"Chunk {chunk_idx}, slice {j}, band {band_idx}, box {conf_idx}: Box_int inválido: {box_int}")
                        continue
                    
                    # ✅ CORRECCIÓN: Calcular SNR usando método compatible
                    # NO usar _calculate_improved_snr_and_time en chunks por ahora
                    # Usar método original más estable
                    snr_val = compute_snr(band_img, box_int)
                    chunk_snr_list.append(snr_val)  # Usar SNR original
                    
                    # ✅ CORRECCIÓN CRÍTICA: Extraer patch usando datos extendidos del archivo original
                    # Cargar datos extendidos para patch correcto
                    extended_data_patch, extended_start_patch = _load_extended_data_for_dedispersion(
                        fits_path, start_sample_global, data_chunk.shape[0], dm_val
                    )
                    
                    if extended_data_patch is not None:
                        # Calcular posición global del candidato en datos extendidos
                        global_sample_in_extended = (start_sample_global + j * slice_len + int(t_sample)) - extended_start_patch
                        
                        # Extraer patch para clasificación usando datos extendidos
                        patch, start_sample_patch = dedisperse_patch(
                            extended_data_patch, config.FREQ, dm_val, global_sample_in_extended
                        )
                        # Ajustar start_sample_patch al contexto global
                        start_sample_patch += extended_start_patch
                    else:
                        # Fallback: usar chunk si no se pueden cargar datos extendidos
                        patch, start_sample_patch = dedisperse_patch(
                            data_chunk, config.FREQ, dm_val, j * slice_len + int(t_sample)
                        )
                        start_sample_patch += start_sample_global
                    
                    class_prob, proc_patch = _classify_patch(cls_model, patch)
                    class_probs_list.append(class_prob)  # Agregar a la lista
                    
                    is_burst = class_prob >= config.CLASS_PROB
                    
                    # ✅ CORRECCIÓN CRÍTICA 4: Calcular patch_start GLOBAL
                    if first_patch is None:
                        first_patch = proc_patch
                        # Calcular tiempo de inicio GLOBAL del patch
                        global_patch_start_sample = start_sample_global + start_sample_patch
                        first_start = global_patch_start_sample * config.TIME_RESO * config.DOWN_TIME_RATE
                        first_dm = dm_val
                    
                    # Crear candidato con información global
                    cand = Candidate(
                        fits_path.name,
                        j + chunk_idx * 10000,  # ID único considerando chunk
                        band_idx,
                        float(conf),
                        dm_val,
                        global_t_sec,  # Tiempo global (ya es absoluto)
                        global_sample,  # Muestra global
                        box_int,
                        snr_val,  # SNR original para compatibilidad
                        class_prob,
                        is_burst,
                        f"chunk{chunk_idx}_slice{j}_band{band_idx}.png",
                        global_t_sec,  # ✅ Tiempo absoluto (ya calculado correctamente)
                        snr_val,       # ✅ SNR calculado
                    )
                    
                    chunk_candidates += 1
                    if is_burst:
                        chunk_bursts += 1
                    else:
                        chunk_no_bursts += 1
                        
                    chunk_max_prob = max(chunk_max_prob, float(conf))
                    
                    # Escribir a CSV
                    _write_candidate_to_csv(csv_file, cand)
                    
                    logger.info(
                        f"Chunk {chunk_idx} - Candidato DM {dm_val:.2f} t={global_t_sec:.3f}s conf={conf:.2f} -> {'BURST' if is_burst else 'no burst'}"
                    )
                    
                except Exception as e:
                    logger.error(f"Error procesando candidato en chunk {chunk_idx}, slice {j}, band {band_idx}, box {conf_idx}: {e}")
                    logger.error(f"Detalles - conf: {conf}, box: {box}")
                    continue
            
            # Marcar si este slice tiene candidatos
            if len(top_conf) > 0:
                slice_has_candidates = True
            
            # Almacenar información de esta banda para procesamiento posterior
            band_results.append({
                'band_idx': band_idx,
                'band_suffix': band_suffix,
                'band_name': band_name,
                'band_img': band_img,
                'img_rgb': img_rgb,
                'top_conf': top_conf,
                'top_boxes': top_boxes,
                'class_probs_list': class_probs_list,
                'first_patch': first_patch,
                'first_start': first_start,
                'first_dm': first_dm,
                'patch_path': patch_path
            })

        # Ahora generar visualizaciones para todas las bandas si AL MENOS UNA tiene candidatos
        for band_result in band_results:
            band_idx = band_result['band_idx']
            band_suffix = band_result['band_suffix']
            band_name = band_result['band_name']
            img_rgb = band_result['img_rgb']
            top_conf = band_result['top_conf']
            top_boxes = band_result['top_boxes']
            class_probs_list = band_result['class_probs_list']
            first_patch = band_result['first_patch']
            first_start = band_result['first_start']
            first_dm = band_result['first_dm']
            patch_path = band_result['patch_path']

            # Solo generar visualizaciones complejas si AL MENOS UNA banda en este slice tiene candidatos
            if slice_has_candidates:
                # ✅ CORRECCIÓN: Inicializar dedisp_block en todos los casos
                dedisp_block = None
                
                # ✅ CORRECCIÓN CRÍTICA: Usar datos apropiados para cada tipo de plot
                if first_patch is not None:
                    # 3) Generar waterfall dedispersado - USAR DATOS EXTENDIDOS DEL ARCHIVO ORIGINAL
                    waterfall_dedispersion_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Cargar datos extendidos para dedispersión correcta
                    extended_data, extended_start = _load_extended_data_for_dedispersion(
                        fits_path, start_sample_global, data_chunk.shape[0], first_dm
                    )
                    
                    if extended_data is not None:
                        # Calcular posición relativa del slice en los datos extendidos
                        relative_start = start_sample_global - extended_start + j * slice_len
                        
                        # Aplicar mismo preprocessing que al chunk
                        extended_preprocessed = np.vstack([extended_data, extended_data[::-1, :]])
                        extended_preprocessed = downsample_data(extended_preprocessed)
                        
                        # Dedispersar usando datos extendidos (CORRECTO)
                        dedisp_block = dedisperse_block(
                            extended_preprocessed, 
                            freq_ds, 
                            first_dm, 
                            relative_start, 
                            slice_len
                        )
                    else:
                        # Fallback: usar chunk si no se pueden cargar datos extendidos
                        logger.warning(f"Usando chunk para dedispersión (no ideal) - slice {slice_idx_global}")
                        start = j * slice_len
                        dedisp_block = dedisperse_block(data_chunk, freq_ds, first_dm, start, slice_len)
                    
                if dedisp_block is not None and dedisp_block.size > 0:
                    plot_waterfall_block(
                        data_block=dedisp_block,
                        freq=freq_ds,
                        time_reso=time_reso_ds,
                        block_size=dedisp_block.shape[0],
                        block_idx=slice_idx_global,  # ✅ Usar índice global
                        save_dir=waterfall_dedispersion_dir,
                        filename=f"{fits_path.stem}_chunk{chunk_idx}_dm{first_dm:.2f}_{band_suffix}",
                        normalize=True,
                    )

                if first_patch is not None:
                    save_patch_plot(
                        first_patch,
                        patch_path,
                        freq_ds,
                        config.TIME_RESO * config.DOWN_TIME_RATE,
                        first_start,
                        off_regions=None,  # Use IQR method for robust estimation
                        thresh_snr=config.SNR_THRESH,
                        band_idx=band_idx,  # Pasar el índice de la banda
                        band_name=band_name,  # Pasar el nombre de la banda
                    )
                else:
                    # ✅ CORRECCIÓN: Para bandas sin detecciones, usar datos extendidos con DM=0
                    waterfall_dedispersion_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Cargar datos extendidos para dedispersión con DM=0
                    extended_data, extended_start = _load_extended_data_for_dedispersion(
                        fits_path, start_sample_global, data_chunk.shape[0], 0.0  # DM=0 para sin detecciones
                    )
                    
                    if extended_data is not None:
                        # Calcular posición relativa del slice en los datos extendidos
                        relative_start = start_sample_global - extended_start + j * slice_len
                        
                        # Aplicar mismo preprocessing
                        extended_preprocessed = np.vstack([extended_data, extended_data[::-1, :]])
                        extended_preprocessed = downsample_data(extended_preprocessed)
                        
                        # Usar DM=0 para banda sin detecciones
                        dedisp_block = dedisperse_block(
                            extended_preprocessed, 
                            freq_ds, 
                            0.0, 
                            relative_start, 
                            slice_len
                        )
                    else:
                        # Fallback
                        start = j * slice_len
                        dedisp_block = dedisperse_block(data_chunk, freq_ds, 0.0, start, slice_len)
                    
                    if dedisp_block is not None and dedisp_block.size > 0:
                        plot_waterfall_block(
                            data_block=dedisp_block,
                            freq=freq_ds,
                            time_reso=time_reso_ds,
                            block_size=dedisp_block.shape[0],
                            block_idx=slice_idx_global,  # ✅ Usar índice global
                            save_dir=waterfall_dedispersion_dir,
                            filename=f"{fits_path.stem}_chunk{chunk_idx}_dm0.00_{band_suffix}",
                            normalize=True,
                        )

                # ✅ CORRECCIÓN CRÍTICA 5: save_slice_summary con parámetros GLOBALES
                composite_dir = save_dir / "Composite" / f"{fits_path.stem}_chunk{chunk_idx}"
                comp_path = composite_dir / f"slice{slice_idx_global}_band{band_idx}.png"
                save_slice_summary(
                    waterfall_block,
                    dedisp_block if dedisp_block is not None and dedisp_block.size > 0 else waterfall_block,  # fallback a waterfall original
                    img_rgb,
                    first_patch,
                    first_start if first_start is not None else 0.0,
                    first_dm if first_dm is not None else 0.0,
                    top_conf if len(top_conf) > 0 else [],
                    top_boxes if len(top_boxes) > 0 else [],
                    class_probs_list, 
                    comp_path,
                    slice_idx_global,        # ✅ CORRECCIÓN: Usar índice GLOBAL del slice
                    total_time_slice,        # ✅ CORRECCIÓN: Total de slices del archivo COMPLETO
                    band_name,
                    band_suffix,
                    fits_path.stem,
                    slice_len,
                    normalize=True,
                    off_regions=None,  # Use IQR method
                    thresh_snr=config.SNR_THRESH,
                    band_idx=band_idx,  # Pasar el índice de la banda
                )

                # ✅ CORRECCIÓN CRÍTICA 6: save_plot con parámetros GLOBALES
                detections_dir = save_dir / "Detections" / f"{fits_path.stem}_chunk{chunk_idx}"
                detections_dir.mkdir(parents=True, exist_ok=True)
                out_img_path = detections_dir / f"slice{slice_idx_global}_{band_suffix}.png"
                save_plot(
                    img_rgb,
                    top_conf if len(top_conf) > 0 else [],
                    top_boxes if len(top_boxes) > 0 else [],
                    class_probs_list,   
                    out_img_path,
                    slice_idx_global,        # ✅ CORRECCIÓN: Usar índice GLOBAL del slice
                    total_time_slice,        # ✅ CORRECCIÓN: Total de slices del archivo COMPLETO
                    band_name,
                    band_suffix,
                    fits_path.stem,
                    slice_len,
                    band_idx=band_idx,  # Pasar el índice de la banda
                )
    
    runtime = time.time() - t_start
    
    return {
        "n_candidates": chunk_candidates,
        "n_bursts": chunk_bursts,
        "n_no_bursts": chunk_no_bursts,
        "runtime_s": runtime,
        "max_prob": chunk_max_prob,
        "snr_list": chunk_snr_list
    }

def _process_file_in_chunks(
    det_model,
    cls_model,
    fits_path: Path,
    save_dir: Path,
    total_samples: int,
    chunk_size: int,
    overlap: int = 1000
) -> dict:
    """Process a large file in chunks to handle memory constraints."""
    
    import gc
    import time
    
    logger.info(f"Procesando archivo grande {fits_path.name} en chunks")
    logger.info(f"Total de muestras: {total_samples:,}")
    logger.info(f"Tamaño de chunk: {chunk_size:,}")
    logger.info(f"Overlap entre chunks: {overlap}")
    
    # Calcular número de chunks necesarios
    effective_chunk_size = chunk_size - overlap
    num_chunks = (total_samples + effective_chunk_size - 1) // effective_chunk_size
    
    logger.info(f"Se procesarán {num_chunks} chunks")
    
    # Acumuladores para resultados globales
    total_candidates = 0
    total_bursts = 0 
    total_no_bursts = 0
    max_prob_global = 0.0
    snr_list_global = []
    total_start_time = time.time()
    
    # Preparar CSV global
    csv_file = save_dir / f"{fits_path.stem}.candidates.csv"
    _ensure_csv_header(csv_file)
    
    for chunk_idx in range(num_chunks):
        logger.info(f"Procesando chunk {chunk_idx + 1}/{num_chunks}")
        
        # Calcular rango de muestras para este chunk
        start_sample = chunk_idx * effective_chunk_size
        end_sample = min(start_sample + chunk_size, total_samples)
        
        if chunk_idx > 0:
            start_sample -= overlap  # Agregar overlap con chunk anterior
            
        actual_chunk_size = end_sample - start_sample
        logger.info(f"Chunk {chunk_idx + 1}: muestras {start_sample:,} a {end_sample:,} ({actual_chunk_size:,} muestras)")
        
        # Modificar temporalmente la configuración para este chunk
        original_file_leng = config.FILE_LENG
        config.FILE_LENG = actual_chunk_size
        
        try:
            # Cargar solo este chunk del archivo
            if fits_path.suffix.lower() == ".fits":
                # Para archivos FITS, necesitaríamos modificar load_fits_file
                # Por ahora, usamos el método existente
                data_chunk = load_fits_file(str(fits_path))
                data_chunk = data_chunk[start_sample:end_sample]
            else:
                # Para archivos .fil, podemos cargar solo el chunk específico
                data_chunk = _load_fil_chunk(str(fits_path), start_sample, actual_chunk_size)
            
            # ✅ CORRECCIÓN CRÍTICA 7: Pasar total_samples para contexto global
            chunk_results = _process_single_chunk(
                det_model, cls_model, data_chunk, fits_path, save_dir, 
                chunk_idx, start_sample, csv_file, total_samples  # ✅ NUEVO parámetro
            )
            
            # Acumular resultados
            total_candidates += chunk_results["n_candidates"]
            total_bursts += chunk_results["n_bursts"] 
            total_no_bursts += chunk_results["n_no_bursts"]
            max_prob_global = max(max_prob_global, chunk_results["max_prob"])
            snr_list_global.extend(chunk_results.get("snr_list", []))
            
            # Liberar memoria
            del data_chunk
            gc.collect()
            
            logger.info(f"Chunk {chunk_idx + 1} completado: {chunk_results['n_candidates']} candidatos")
            
        except Exception as e:
            logger.error(f"Error procesando chunk {chunk_idx + 1}: {e}")
            continue
        finally:
            # Restaurar configuración original
            config.FILE_LENG = original_file_leng
    
    total_runtime = time.time() - total_start_time
    
    logger.info(f"Procesamiento completo: {total_candidates} candidatos totales en {total_runtime:.1f}s")
    
    return {
        "n_candidates": total_candidates,
        "n_bursts": total_bursts,
        "n_no_bursts": total_no_bursts,
        "runtime_s": total_runtime,
        "max_prob": float(max_prob_global),
        "mean_snr": float(np.mean(snr_list_global)) if snr_list_global else 0.0,
        "chunks_processed": num_chunks
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
                summary[fits_path.name] = _process_file(det_model, cls_model, fits_path, save_dir)
                print(f"Archivo {fits_path.name} procesado exitosamente")
            except Exception as e:
                print(f"[ERROR] Error procesando {fits_path.name}: {e}")
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

def _calculate_improved_snr_and_time(
    waterfall_block: np.ndarray,
    box: tuple[int, int, int, int],
    slice_idx: int,
    t_sample_relative: int,
    slice_len: int
) -> tuple[float, float]:
    """
    Calcula SNR mejorado usando compute_snr_profile y tiempo absoluto.
    
    Parameters
    ----------
    waterfall_block : np.ndarray
        Bloque de waterfall para calcular SNR
    box : tuple
        Bounding box (x1, y1, x2, y2)
    slice_idx : int
        Índice del slice actual
    t_sample_relative : int
        Muestra de tiempo relativa al slice
    slice_len : int
        Longitud del slice
        
    Returns
    -------
    tuple
        (snr_peak, t_sec_absolute)
    """
    from .snr_utils import compute_snr_profile, find_snr_peak
    
    try:
        # Calcular SNR usando el mismo método que los plots
        snr_profile, sigma = compute_snr_profile(waterfall_block, off_regions=None)
        
        # Encontrar el pico SNR en la región de la bounding box
        x1, y1, x2, y2 = box
        
        # Mapear coordenadas de la bounding box al perfil SNR
        # La bounding box está en coordenadas de imagen (512x512)
        # El perfil SNR tiene la longitud del waterfall_block
        if len(snr_profile) > 0:
            # Escalar coordenada X de la bounding box al perfil SNR
            center_x = (x1 + x2) / 2
            snr_idx = int((center_x / 512.0) * len(snr_profile))
            snr_idx = max(0, min(len(snr_profile) - 1, snr_idx))
            
            # Tomar SNR en la posición del candidato
            snr_peak = snr_profile[snr_idx]
        else:
            snr_peak = 0.0
            
    except Exception as e:
        logger.warning(f"Error calculando SNR mejorado: {e}")
        # Fallback al método original
        from .metrics import compute_snr
        band_img = waterfall_block  # Asumir que es la imagen de banda
        snr_peak = compute_snr(band_img, box)
    
    # Calcular tiempo absoluto desde el inicio del archivo
    # t_sample_relative es relativo al slice actual
    # Necesitamos sumar el offset del slice
    t_sample_absolute = slice_idx * slice_len + t_sample_relative
    t_sec_absolute = t_sample_absolute * config.TIME_RESO * config.DOWN_TIME_RATE
    
    return snr_peak, t_sec_absolute