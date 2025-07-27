"""High level pipeline for FRB detection with CenterNet."""
from __future__ import annotations

import csv
import json
import logging
import time
import gc
from pathlib import Path
from typing import List

try:
    import torch
except ImportError:  
    torch = None
import numpy as np

# Importar matplotlib al nivel del módulo para evitar problemas de scope
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from . import config
from .candidate import Candidate
from .dedispersion import d_dm_time_g, dedisperse_patch, dedisperse_patch_centered, dedisperse_block
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
from .slice_len_utils import update_slice_len_dynamic, get_slice_duration_info
from .visualization import (
    save_plot,
    save_patch_plot,
    save_slice_summary,
    plot_waterfalls,
)
from .summary_utils import (
    _write_summary,
    _update_summary_with_results,
    _update_summary_with_file_debug,
)
logger = logging.getLogger(__name__)


def _optimize_memory(aggressive: bool = False) -> None:
    """Optimiza el uso de memoria del sistema.
    
    Args:
        aggressive: Si True, realiza limpieza más agresiva
    """
    # Limpieza básica de Python
    gc.collect()
    
    # Limpieza de matplotlib para evitar acumulación de figuras
    if plt is not None:
        plt.close('all')  # Cerrar todas las figuras de matplotlib
    
    # Limpieza de CUDA si está disponible
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        if aggressive:
            # Limpieza más agresiva de CUDA
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    
    # Pausa breve para permitir liberación de memoria
    if aggressive:
        time.sleep(0.05)  # 50ms para limpieza agresiva
    else:
        time.sleep(0.01)  # 10ms para limpieza normal


def _load_model() -> torch.nn.Module:
    """Load the CenterNet model configured in :mod:`config`."""
    if torch is None:
        raise ImportError("torch is required to load models")

    from training.ObjectDet.centernet_model import centernet
    model = centernet(model_name=config.MODEL_NAME).to(config.DEVICE)
    state = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

def _load_class_model() -> torch.nn.Module:
    """Load the binary classification model configured in :mod:`config`."""
    if torch is None:
        raise ImportError("torch is required to load models")

    from training.BinaryClass.binary_model import BinaryNet
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

def _detect(model, img_tensor: np.ndarray) -> tuple[list, list | None]:
    """Run the detection model and return confidences and boxes."""
    from training.ObjectDet.centernet_utils import get_res

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


def _process_block(
    det_model: torch.nn.Module,
    cls_model: torch.nn.Module,
    block: np.ndarray,
    metadata: dict,
    fits_path: Path,
    save_dir: Path,
    chunk_idx: int,
) -> dict:
    """Procesa un bloque de datos y retorna estadísticas del bloque."""
    
    # Configurar parámetros temporales para este bloque
    original_file_leng = config.FILE_LENG # Guardar longitud original del archivo
    config.FILE_LENG = metadata["actual_chunk_size"] # Actualizar longitud del archivo al tamaño del bloque actual
    
    # 🕐 CALCULAR TIEMPO ABSOLUTO DESDE INICIO DEL ARCHIVO
    # Tiempo de inicio del chunk en segundos desde el inicio del archivo
    chunk_start_time_sec = metadata["start_sample"] * config.TIME_RESO
    chunk_duration_sec = metadata["actual_chunk_size"] * config.TIME_RESO
    
    logger.info(f"🕐 Chunk {chunk_idx:03d}: Tiempo {chunk_start_time_sec:.2f}s - {chunk_start_time_sec + chunk_duration_sec:.2f}s "
               f"(duración: {chunk_duration_sec:.2f}s)")
    
    try:
        # Aplicar downsampling al bloque
        block = downsample_data(block) # Aplica downsampling según la configuración
        
        # Calcular parámetros para este bloque
        freq_down = np.mean(
            config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE),
            axis=1,
        ) # Promedio de frecuencias después del downsampling
        
        height = config.DM_max - config.DM_min + 1 # Altura del cubo DM
        width_total = config.FILE_LENG // config.DOWN_TIME_RATE # Ancho total del cubo DM-time
        
        # Calcular slice_len dinámicamente
        slice_len, real_duration_ms = update_slice_len_dynamic() # Actualiza slice_len según la configuración
        time_slice = (width_total + slice_len - 1) // slice_len # Número de slices por chunk
        
        logger.info(f"🧩 Chunk {chunk_idx:03d}: {metadata['actual_chunk_size']} muestras → {time_slice} slices")
        
        # Generar DM-time cube
        dm_time = d_dm_time_g(block, height=height, width=width_total) # dedispersion del bloque
        
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
        
        # Procesar slices
        cand_counter = 0
        n_bursts = 0
        n_no_bursts = 0
        prob_max = 0.0
        snr_list = []
        
        for j in range(time_slice):
            # Verificar que tenemos suficientes datos para este slice
            start_idx = slice_len * j
            end_idx = slice_len * (j + 1)
            
            # Verificar que no excedemos los límites del bloque
            if start_idx >= block.shape[0]:
                logger.warning(f"Slice {j}: start_idx ({start_idx}) >= block.shape[0] ({block.shape[0]}), saltando...")
                continue
                
            if end_idx > block.shape[0]:
                logger.warning(f"Slice {j}: end_idx ({end_idx}) > block.shape[0] ({block.shape[0]}), ajustando...")
                end_idx = block.shape[0]
                # Si el slice es muy pequeño, saltarlo
                if end_idx - start_idx < slice_len // 2:
                    logger.warning(f"Slice {j}: muy pequeño ({end_idx - start_idx} muestras), saltando...")
                    continue
            
            slice_cube = dm_time[:, :, start_idx : end_idx]
            waterfall_block = block[start_idx : end_idx]

            if slice_cube.size == 0 or waterfall_block.size == 0:
                logger.warning(f"Slice {j}: datos vacíos, saltando...")
                continue

            # 🕐 Calcular tiempo absoluto para este slice específico
            slice_start_time_sec = chunk_start_time_sec + (j * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE)

            # 🎯 VARIABLES PARA ACUMULAR RESULTADOS DE TODAS LAS BANDAS
            all_candidates = []
            all_top_conf = []
            all_top_boxes = []
            all_class_probs = []
            first_patch = None
            first_dm = None
            img_rgb_fullband = None

            # Procesar cada banda para detecciones
            for band_idx, band_suffix, band_name in band_configs:
                band_img = slice_cube[band_idx]
                img_tensor = preprocess_img(band_img)
                top_conf, top_boxes = _detect(det_model, img_tensor)

                if top_boxes is None:
                    top_conf = []
                    top_boxes = []

                # Guardar img_rgb de la banda principal (fullband) para visualizaciones
                if band_idx == 0:
                    img_rgb_fullband = postprocess_img(img_tensor)

                # Procesar detecciones de esta banda
                class_probs_list = []
                for conf, box in zip(top_conf, top_boxes):
                    dm_val, t_sec, t_sample = pixel_to_physical(
                        (box[0] + box[2]) / 2,
                        (box[1] + box[3]) / 2,
                        slice_len,
                    )
                    snr_val = compute_snr(band_img, tuple(map(int, box)))
                    snr_list.append(snr_val)

                    # Ajustar tiempo global considerando el chunk
                    global_sample = metadata["start_sample"] + j * slice_len + int(t_sample)
                    # Usar patch centralizado para mejor visualización del candidato
                    patch, start_sample = dedisperse_patch_centered(
                        block, freq_down, dm_val, global_sample
                    )
                    class_prob, proc_patch = _classify_patch(cls_model, patch)
                    class_probs_list.append(class_prob)

                    is_burst = class_prob >= config.CLASS_PROB
                    cand_counter += 1
                    if is_burst:
                        n_bursts += 1
                    else:
                        n_no_bursts += 1
                    prob_max = max(prob_max, float(conf))

                    # Guardar primer patch para visualizaciones (de cualquier banda)
                    if first_patch is None:
                        first_patch = proc_patch
                        first_dm = dm_val

                    # 🕐 Calcular tiempo absoluto del candidato
                    absolute_candidate_time = slice_start_time_sec + t_sec

                    # Crear candidato y escribir al CSV
                    cand = Candidate(
                        f"{fits_path.name}_chunk{chunk_idx:03d}",
                        j,
                        band_idx,
                        float(conf),
                        dm_val,
                        absolute_candidate_time,  # 🕐 Tiempo absoluto desde inicio del archivo
                        t_sample,
                        tuple(map(int, box)),
                        snr_val,
                        class_prob,
                        is_burst,
                        f"patch_slice{j}_band{band_idx}_chunk{chunk_idx:03d}.png",
                    )

                    # Escribir al CSV
                    csv_file = save_dir / f"{fits_path.stem}_chunk{chunk_idx:03d}.candidates.csv"
                    _ensure_csv_header(csv_file)
                    _write_candidate_to_csv(csv_file, cand)

                    logger.info(
                        f"🧩 Chunk {chunk_idx:03d} - Candidato DM {dm_val:.2f} t={absolute_candidate_time:.3f}s (chunk: {t_sec:.3f}s) conf={conf:.2f} class={class_prob:.2f} → {'BURST' if is_burst else 'no burst'}"
                    )

                # Acumular resultados de esta banda
                all_candidates.extend(zip(top_conf, top_boxes, class_probs_list))
                all_top_conf.extend(top_conf)
                all_top_boxes.extend(top_boxes)
                all_class_probs.extend(class_probs_list)

            # 🎨 GENERAR VISUALIZACIONES UNA SOLA VEZ POR SLICE (fuera del bucle de bandas)
            custom_mode = not config.PLOT_CONTROL_DEFAULT
            plot_wf_disp = (
                (not custom_mode and len(all_top_conf) > 0)
                or (custom_mode and config.PLOT_WATERFALL_DISPERSION)
            )
            plot_wf_dedisp = (
                (not custom_mode and len(all_top_conf) > 0)
                or (custom_mode and config.PLOT_WATERFALL_DEDISPERSION)
            )
            plot_patch = (
                first_patch is not None
                and (
                    (not custom_mode and len(all_top_conf) > 0)
                    or (custom_mode and config.PLOT_PATCH_CANDIDATE)
                )
            )
            plot_comp = (
                (not custom_mode and len(all_top_conf) > 0)
                or (custom_mode and config.PLOT_COMPOSITE)
            )
            plot_det = (
                (not custom_mode and len(all_top_conf) > 0)
                or (custom_mode and config.PLOT_DETECTION_DM_TIME)
            )

            if any([plot_wf_disp, plot_wf_dedisp, plot_patch, plot_comp, plot_det]):
                # Preparar directorios con sufijo de chunk
                chunk_suffix = f"_chunk{chunk_idx:03d}"
                
                # 1. Generar waterfall sin dedispersar
                if plot_wf_disp:
                    waterfall_dispersion_dir = save_dir / "waterfall_dispersion" / f"{fits_path.stem}{chunk_suffix}"
                    waterfall_dispersion_dir.mkdir(parents=True, exist_ok=True)
                    
                    plot_waterfall_block(
                        data_block=waterfall_block,
                        freq=freq_down,
                        time_reso=config.TIME_RESO * config.DOWN_TIME_RATE,
                        block_size=waterfall_block.shape[0],
                        block_idx=j,
                        save_dir=waterfall_dispersion_dir,
                        filename=f"{fits_path.stem}{chunk_suffix}",
                        normalize=True,
                        absolute_start_time=slice_start_time_sec,  # 🕐 TIEMPO ABSOLUTO
                    )
                
                # 2. Generar waterfall dedispersado
                if plot_wf_dedisp:
                    waterfall_dedispersion_dir = save_dir / "waterfall_dedispersion" / f"{fits_path.stem}{chunk_suffix}"
                    waterfall_dedispersion_dir.mkdir(parents=True, exist_ok=True)

                    # Usar DM del primer candidato o DM=0 si no hay candidatos
                    dm_for_dedisp = first_dm if first_dm is not None else 0.0
                    dedisp_block = dedisperse_block(block, freq_down, dm_for_dedisp, j * slice_len, slice_len)

                    if dedisp_block.size > 0:
                        plot_waterfall_block(
                            data_block=dedisp_block,
                            freq=freq_down,
                            time_reso=config.TIME_RESO * config.DOWN_TIME_RATE,
                            block_size=dedisp_block.shape[0],
                            block_idx=j,
                            save_dir=waterfall_dedispersion_dir,
                            filename=f"{fits_path.stem}{chunk_suffix}_dm{dm_for_dedisp:.2f}",
                            normalize=True,
                            absolute_start_time=slice_start_time_sec,  # 🕐 TIEMPO ABSOLUTO
                        )

                # 3. Generar patch plot
                if plot_patch and first_patch is not None:
                    patch_dir = save_dir / "Patches" / f"{fits_path.stem}{chunk_suffix}"
                    patch_dir.mkdir(parents=True, exist_ok=True)
                    patch_path = patch_dir / f"patch_slice{j}{chunk_suffix}.png"
                    
                    save_patch_plot(
                        first_patch,
                        patch_path,
                        freq_down,
                        config.TIME_RESO * config.DOWN_TIME_RATE,
                        slice_start_time_sec,  # 🕐 TIEMPO ABSOLUTO
                        off_regions=None,
                        thresh_snr=config.SNR_THRESH,
                        band_idx=0,  # Usar banda principal
                        band_name="Full Band",
                    )

                # 4. Generar composite
                if plot_comp and img_rgb_fullband is not None:
                    composite_dir = save_dir / "Composite" / f"{fits_path.stem}{chunk_suffix}"
                    composite_dir.mkdir(parents=True, exist_ok=True)
                    comp_path = composite_dir / f"slice{j}{chunk_suffix}.png"
                    
                    dedisp_block = dedisperse_block(block, freq_down, first_dm if first_dm is not None else 0.0, j * slice_len, slice_len) if first_dm is not None else None
                    
                    save_slice_summary(
                        waterfall_block,
                        dedisp_block if dedisp_block is not None and dedisp_block.size > 0 else waterfall_block,
                        img_rgb_fullband,
                        first_patch,
                        slice_start_time_sec,  # 🕐 TIEMPO ABSOLUTO
                        first_dm if first_dm is not None else 0.0,
                        all_top_conf,
                        all_top_boxes,
                        all_class_probs,
                        comp_path,
                        j,
                        time_slice,
                        "Full Band",
                        "fullband",
                        f"{fits_path.stem}{chunk_suffix}",
                        slice_len,
                        normalize=True,
                        off_regions=None,
                        thresh_snr=config.SNR_THRESH,
                        band_idx=0,
                        absolute_start_time=slice_start_time_sec,  # 🕐 TIEMPO ABSOLUTO
                    )
                
                # 5. Generar detecciones (SIEMPRE generar, incluso sin candidatos)
                if plot_det:
                    detections_dir = save_dir / "Detections" / f"{fits_path.stem}{chunk_suffix}"
                    detections_dir.mkdir(parents=True, exist_ok=True)
                    out_img_path = detections_dir / f"slice{j}{chunk_suffix}.png"

                    # Usar img_rgb_fullband si está disponible, sino crear uno vacío
                    detection_img = img_rgb_fullband if img_rgb_fullband is not None else np.zeros((512, 512, 3), dtype=np.float32)
                    
                    save_plot(
                        detection_img,
                        all_top_conf if all_top_conf else [],
                        all_top_boxes if all_top_boxes else [],
                        all_class_probs if all_class_probs else [],
                        out_img_path,
                        j,
                        time_slice,
                        "Full Band",
                        "fullband",
                        f"{fits_path.stem}{chunk_suffix}",
                        slice_len,
                        band_idx=0,
                        absolute_start_time=slice_start_time_sec,  # 🕐 TIEMPO ABSOLUTO
                    )

            # 🧹 LIMPIEZA DE MEMORIA DESPUÉS DE CADA SLICE
            if j % 10 == 0:  # Cada 10 slices
                _optimize_memory(aggressive=False)
            else:
                # Limpieza básica después de cada slice
                if plt is not None:
                    plt.close('all')
                gc.collect()
                
            # 🧹 LIMPIEZA ADICIONAL DESPUÉS DE GENERAR VISUALIZACIONES
            if any([plot_wf_disp, plot_wf_dedisp, plot_patch, plot_comp, plot_det]):
                if plt is not None:
                    plt.close('all')
                gc.collect()

        return {
            "n_candidates": cand_counter,
            "n_bursts": n_bursts,
            "n_no_bursts": n_no_bursts,
            "max_prob": prob_max,
            "mean_snr": float(np.mean(snr_list)) if snr_list else 0.0,
            "time_slice": time_slice,
        }
        
    finally:
        # Restaurar configuración original
        config.FILE_LENG = original_file_leng


def _process_file_chunked(
    det_model: torch.nn.Module,
    cls_model: torch.nn.Module,
    fits_path: Path,
    save_dir: Path,
    chunk_samples: int,
) -> dict:
    """Procesa un archivo .fil en bloques usando stream_fil."""
    
    import gc
    from .filterbank_io import stream_fil
    
    # 📊 CALCULAR INFORMACIÓN DEL ARCHIVO DE MANERA EFICIENTE
    logger.info(f"📁 Analizando estructura del archivo: {fits_path.name}")
    
    # Calcular información basada en config.FILE_LENG (ya cargado por get_obparams_fil)
    total_samples = config.FILE_LENG
    chunk_count = (total_samples + chunk_samples - 1) // chunk_samples  # Redondear hacia arriba
    total_duration_sec = total_samples * config.TIME_RESO
    chunk_duration_sec = chunk_samples * config.TIME_RESO
    
    logger.info(f"📊 RESUMEN DEL ARCHIVO:")
    logger.info(f"   🧩 Total de chunks estimado: {chunk_count}")
    logger.info(f"   📊 Muestras totales: {total_samples:,}")
    logger.info(f"   🕐 Duración total: {total_duration_sec:.2f} segundos ({total_duration_sec/60:.1f} minutos)")
    logger.info(f"   📦 Tamaño de chunk: {chunk_samples:,} muestras ({chunk_duration_sec:.2f}s)")
    logger.info(f"   🔄 Iniciando procesamiento...")
    
    t_start = time.time()
    cand_counter_total = 0
    n_bursts_total = 0
    n_no_bursts_total = 0
    prob_max_total = 0.0
    snr_list_total = []
    actual_chunk_count = 0
    
    try:
        # Procesar cada bloque (UNA SOLA PASADA)
        for block, metadata in stream_fil(str(fits_path), chunk_samples):
            actual_chunk_count += 1
            logger.info(f"🧩 Procesando chunk {metadata['chunk_idx']:03d} "
                       f"({metadata['start_sample']:,} - {metadata['end_sample']:,})")
            
            # Procesar bloque
            block_results = _process_block(
                det_model, cls_model, block, metadata, 
                fits_path, save_dir, metadata['chunk_idx']
            )
            
            # Acumular resultados
            cand_counter_total += block_results["n_candidates"]
            n_bursts_total += block_results["n_bursts"]
            n_no_bursts_total += block_results["n_no_bursts"]
            prob_max_total = max(prob_max_total, block_results["max_prob"])
            
            # 🧹 LIMPIEZA AGRESIVA DE MEMORIA
            del block
            _optimize_memory(aggressive=(actual_chunk_count % 5 == 0))  # Limpieza agresiva cada 5 chunks
        
        runtime = time.time() - t_start
        logger.info(
            f"🧩 Archivo completado: {actual_chunk_count} chunks procesados, "
            f"{cand_counter_total} candidatos, max prob {prob_max_total:.2f}, ⏱️ {runtime:.1f}s"
        )
        
        return {
            "n_candidates": cand_counter_total,
            "n_bursts": n_bursts_total,
            "n_no_bursts": n_no_bursts_total,
            "runtime_s": runtime,
            "max_prob": prob_max_total,
            "mean_snr": float(np.mean(snr_list_total)) if snr_list_total else 0.0,
            "status": "completed_chunked"
        }
        
    except Exception as e:
        logger.error(f"Error procesando archivo por bloques: {e}")
        return {
            "n_candidates": 0,
            "n_bursts": 0,
            "n_no_bursts": 0,
            "runtime_s": time.time() - t_start,
            "max_prob": 0.0,
            "mean_snr": 0.0,
            "status": "ERROR_CHUNKED"
        }

def _process_file(
    det_model: torch.nn.Module,
    cls_model: torch.nn.Module,
    fits_path: Path,
    save_dir: Path,
    chunk_samples: int = 0,
) -> dict:
    """Process a single FITS file and return summary information."""

    t_start = time.time()
    logger.info("Procesando %s", fits_path.name)

    # Determinar si usar procesamiento por bloques
    use_chunking = chunk_samples > 0 and fits_path.suffix.lower() == ".fil"
    
    if use_chunking:
        logger.info("🧩 Procesando archivo .fil en bloques de %d muestras", chunk_samples)
        return _process_file_chunked(det_model, cls_model, fits_path, save_dir, chunk_samples)
    
    # Procesamiento normal (modo antiguo)
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
        if slice_cube.size == 0:
            logger.warning("Slice %d: slice_cube vacío, saltando...", j)
            continue
            
        if waterfall_block.size == 0:
            logger.warning("Slice %d: waterfall_block vacío, saltando...", j)
            continue
        
        
        # 2) Generar waterfall sin dedispersar para este slice
        custom_mode = not config.PLOT_CONTROL_DEFAULT
        if (not custom_mode) or config.PLOT_WATERFALL_DISPERSION:
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

            custom_mode = not config.PLOT_CONTROL_DEFAULT
            plot_wf_dedisp = (
                (slice_has_candidates and not custom_mode) or (custom_mode and config.PLOT_WATERFALL_DEDISPERSION)
            )
            plot_patch = (
                first_patch is not None
                and (
                    (slice_has_candidates and not custom_mode)
                    or (custom_mode and config.PLOT_PATCH_CANDIDATE)
                )
            )
            plot_comp = (
                (slice_has_candidates and not custom_mode) or (custom_mode and config.PLOT_COMPOSITE)
            )
            plot_det = (not custom_mode) or config.PLOT_DETECTION_DM_TIME

            if plot_wf_dedisp or plot_patch or plot_comp:
                # Preparar valores por defecto para casos sin detecciones en esta banda específica
                dedisp_block = None
                if first_patch is not None:
                    # 3) Generar waterfall dedispersado para este slice con el primer candidato
                    waterfall_dedispersion_dir.mkdir(parents=True, exist_ok=True)
                    start = j * slice_len
                    
                    # DEBUG: Verificar DM usado para dedispersión
                    if config.DEBUG_FREQUENCY_ORDER:
                        print(f"🔍 [DEBUG DM] Slice {j} CON candidatos - usando DM={first_dm:.2f}")
                    
                    dedisp_block = dedisperse_block(data, freq_down, first_dm, start, slice_len)
                    
                
                if plot_wf_dedisp and dedisp_block is not None and dedisp_block.size > 0:
                    plot_waterfall_block(
                        data_block=dedisp_block,
                        freq=freq_down,
                        time_reso=time_reso_ds,
                        block_size=dedisp_block.shape[0],
                        block_idx=j,
                        save_dir=waterfall_dedispersion_dir,
                        filename=f"{fits_path.stem}_dm{first_dm:.2f}_{band_suffix}",
                        normalize=True,
                        absolute_start_time=None,  # 🕐 Usar tiempo relativo para procesamiento estándar
                    )

                if plot_patch:
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
                elif plot_wf_dedisp:
                    # Para bandas sin detecciones, crear un parche dummy
                    waterfall_dedispersion_dir.mkdir(parents=True, exist_ok=True)
                    start = j * slice_len
                    
                    # DEBUG: Verificar DM usado cuando no hay candidatos
                    if config.DEBUG_FREQUENCY_ORDER:
                        print(f"🔍 [DEBUG DM] Slice {j} SIN candidatos - usando DM=0.0 (¡Sin dedispersión!)")
                        print(f"🔍 [DEBUG DM] ❌ PROBLEMA: DM=0 significa waterfall dedispersado = waterfall raw")
                    
                    # Usar DM=0 para banda sin detecciones
                    dedisp_block = dedisperse_block(data, freq_down, 0.0, start, slice_len)
                    
                    if plot_wf_dedisp and dedisp_block.size > 0:
                        plot_waterfall_block(
                            data_block=dedisp_block,
                            freq=freq_down,
                            time_reso=time_reso_ds,
                            block_size=dedisp_block.shape[0],
                            block_idx=j,
                            save_dir=waterfall_dedispersion_dir,
                            filename=f"{fits_path.stem}_dm0.00_{band_suffix}",
                            normalize=True,
                            absolute_start_time=None,  # 🕐 Usar tiempo relativo para procesamiento estándar
                        )

            if plot_comp:
                # 1) Generar composite
                composite_dir = save_dir / "Composite" / fits_path.stem
                comp_path = composite_dir / f"slice{j}_band{band_idx}.png"

                # DEBUG: Verificar datos antes de save_slice_summary
                if config.DEBUG_FREQUENCY_ORDER:
                    print(f"🔍 [DEBUG PIPELINE] Slice {j}, Band {band_idx}")
                    print(f"🔍 [DEBUG PIPELINE] waterfall_block shape: {waterfall_block.shape}")
                    print(f"🔍 [DEBUG PIPELINE] dedisp_block existe: {dedisp_block is not None}")
                    if dedisp_block is not None:
                        print(f"🔍 [DEBUG PIPELINE] dedisp_block shape: {dedisp_block.shape}")
                        print(f"🔍 [DEBUG PIPELINE] ¿Son iguales?: {np.array_equal(waterfall_block, dedisp_block)}")
                        if not np.array_equal(waterfall_block, dedisp_block):
                            print(f"🔍 [DEBUG PIPELINE] ✅ DIFERENTES - Diff max: {np.max(np.abs(waterfall_block - dedisp_block)):.6f}")
                        else:
                            print(f"🔍 [DEBUG PIPELINE] ❌ IGUALES - Esto indica problema!")
                    else:
                        print(f"🔍 [DEBUG PIPELINE] ❌ dedisp_block es None - usando fallback!")
                
                save_slice_summary(
                    waterfall_block,
                    dedisp_block if dedisp_block is not None and dedisp_block.size > 0 else waterfall_block,
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
                    off_regions=None,
                    thresh_snr=config.SNR_THRESH,
                    band_idx=band_idx,
                )

            if plot_det:
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
                    band_idx=band_idx,
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

def run_pipeline(chunk_samples: int = 0) -> None:
    """Run the full FRB detection pipeline.
    
    Args:
        chunk_samples: Número de muestras por bloque para archivos .fil (0 = modo antiguo)
    """

    print("=== INICIANDO PIPELINE DE DETECCIÓN DE FRB ===")
    print(f"Directorio de datos: {config.DATA_DIR}")
    print(f"Directorio de resultados: {config.RESULTS_DIR}")
    print(f"Targets FRB: {config.FRB_TARGETS}")
    if chunk_samples > 0:
        print(f"🧩 Modo chunking habilitado: {chunk_samples:,} muestras por bloque")
    else:
        print("📁 Modo normal: carga completa en memoria")
    
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
                results = _process_file(det_model, cls_model, fits_path, save_dir, chunk_samples)
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
    parser.add_argument("--chunk-samples", type=int, default=0, 
                       help="Número de muestras por bloque para archivos .fil (0 = modo antiguo)")
    args = parser.parse_args()

    if args.data_dir:
        config.DATA_DIR = args.data_dir
    if args.results_dir:
        config.RESULTS_DIR = args.results_dir
    if args.det_model:
        config.MODEL_PATH = args.det_model
    if args.class_model:
        config.CLASS_MODEL_PATH = args.class_model

    run_pipeline(chunk_samples=args.chunk_samples)
