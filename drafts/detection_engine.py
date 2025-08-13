"""Detection engine for FRB pipeline - orchestrates detection, classification, and visualization."""
from __future__ import annotations

from .visualization.visualization_unified import save_all_plots, plot_waterfall_block, preprocess_img, postprocess_img
from .preprocessing.dedispersion import dedisperse_patch
from .preprocessing.dedispersion import dedisperse_block
from .detection.model_interface import detect, classify_patch
# from .detection.metrics import compute_snr  # Archivo eliminado, no se usa
from .analysis.snr_utils import compute_snr_profile  
from .preprocessing.dm_candidate_extractor import extract_candidate_dm
from .output.candidate_manager import append_candidate, Candidate
from .preprocessing.slice_len_calculator import update_slice_len_dynamic
import numpy as np
import logging
import os
from pathlib import Path
logger = logging.getLogger(__name__)

def _presto_time_ref_correction(dm_val: float, freq_ref_used_mhz: float, freq_ref_global_mhz: float) -> float:
    """Corrección de referencia temporal al estilo PRESTO (segundos).

    PRESTO ajusta el inicio del eje temporal cuando se cambia la frecuencia de
    referencia (por ejemplo, tras subbandeo). La corrección es:
        Δt = 4.1488e3 * DM * (|1/ref_global^2 - 1/ref_used^2|)
    con frecuencias en MHz y Δt en segundos.

    Args:
        dm_val: DM del candidato
        freq_ref_used_mhz: frecuencia de referencia efectiva usada para dedispersión/plot
        freq_ref_global_mhz: frecuencia de referencia global (tope de banda original)
    Returns:
        float: corrección en segundos (sumar al tiempo reportado)
    """
    try:
        f1 = float(freq_ref_global_mhz)
        f2 = float(freq_ref_used_mhz)
        if f1 <= 0 or f2 <= 0 or dm_val is None:
            return 0.0
        return 4.1488e3 * float(dm_val) * abs(1.0 / (f1 * f1) - 1.0 / (f2 * f2))
    except Exception:
        return 0.0

def get_pipeline_parameters(config):
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
    slice_duration = slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
    return freq_down, height, width_total, slice_len, real_duration_ms, time_slice, slice_duration

def process_band(
    det_model,
    cls_model,
    band_img,
    slice_len,
    j,
    fits_path,
    save_dir,
    data,
    freq_down,
    csv_file,
    time_reso_ds,
    snr_list,
    config,
    absolute_start_time=None,
    patches_dir=None,
    chunk_idx=None,  # ID del chunk
    band_idx=None,  # ID de la banda
    slice_start_idx: int | None = None,  # NUEVO: inicio del slice (decimado) para índice global
):
    """Procesa una banda con tiempo absoluto para continuidad temporal.
    
    Args:
        absolute_start_time: Tiempo absoluto de inicio del slice en segundos desde el inicio del archivo
        chunk_idx: ID del chunk donde se encuentra este slice
        band_idx: ID de la banda (0=fullband, 1=lowband, 2=highband)
    """
    # Obtener el logger global para mensajes informativos
    try:
        from .logging.logging_config import get_global_logger, Colors
        global_logger = get_global_logger()
    except ImportError:
        global_logger = None
    
    # Mensaje de inicio de procesamiento de banda
    band_names = ["Full Band", "Low Band", "High Band"]
    band_name = band_names[band_idx] if band_idx is not None and band_idx < len(band_names) else f"Band {band_idx}"
    if global_logger:
        global_logger.processing_band(band_name, j)
    
    img_tensor = preprocess_img(band_img)
    top_conf, top_boxes = detect(det_model, img_tensor)
    img_rgb = postprocess_img(img_tensor)
    if top_boxes is None:
        top_conf = []
        top_boxes = []
    
    # Mensaje sobre detecciones encontradas
    if global_logger:
        global_logger.band_candidates(band_name, len(top_conf))
    
    first_patch = None
    first_start = None
    first_dm = None
    if patches_dir is not None:
        patch_path = patches_dir / f"patch_slice{j}_band{band_img.shape[0] if hasattr(band_img, 'shape') else 0}.png"
    else:
        patch_dir = save_dir / "Patches" / fits_path.stem
        patch_path = patch_dir / f"patch_slice{j}_band{band_img.shape[0] if hasattr(band_img, 'shape') else 0}.png"
    class_probs_list = []
    candidate_times_abs: list[float] = []
    # Trabajos PRESTO: lista de tuplas (t_abs, dm)
    burst_jobs: list[tuple[float, float]] = []
    cand_counter = 0
    n_bursts = 0
    n_no_bursts = 0
    prob_max = 0.0
    
    # Variables para seleccionar el mejor candidato para el Composite
    best_patch = None
    best_start = None
    best_dm = None
    best_is_burst = False
    first_patch = None  # Mantener para compatibilidad (primer candidato)
    first_start = None
    first_dm = None
    
    # Lista para almacenar todos los candidatos procesados
    all_candidates = []
    
    for conf, box in zip(top_conf, top_boxes):
        dm_val, t_sec, t_sample = extract_candidate_dm(
            (box[0] + box[2]) / 2,
            (box[1] + box[3]) / 2,
            slice_len,
        )
        
        # Usar compute_snr_profile para SNR consistente con composite
        # Extraer región del candidato para cálculo de SNR
        x1, y1, x2, y2 = map(int, box)
        candidate_region = band_img[y1:y2, x1:x2]
        if candidate_region.size > 0:
            # Usar compute_snr_profile para consistencia con composite
            snr_profile, _ = compute_snr_profile(candidate_region)
            snr_val_raw = np.max(snr_profile)  # Tomar el pico del SNR
        else:
            snr_val_raw = 0.0
        
        snr_list.append(snr_val_raw)  # Guardar SNR raw para estadísticas
        # Índice global correcto dentro del bloque decimado: inicio real del slice + offset detectado
        if slice_start_idx is not None:
            global_sample = int(slice_start_idx) + int(t_sample)
        else:
            # Fallback: modo antiguo (puede ser inexacto si los slices no son uniformes)
            global_sample = j * slice_len + int(t_sample)
        patch, start_sample = dedisperse_patch(
            data, freq_down, dm_val, global_sample
        )
        
        # Calcular SNR del patch dedispersado al estilo PRESTO (matched filter)
        snr_val = 0.0  # Valor por defecto
        peak_idx_patch = None
        if patch is not None and patch.size > 0:
            from .analysis.snr_utils import compute_presto_matched_snr
            # patch: (time, freq)
            dt_ds = config.TIME_RESO * config.DOWN_TIME_RATE
            snr_profile_pre, best_w = compute_presto_matched_snr(patch, dt_seconds=dt_ds)
            # pico y su índice
            peak_idx_patch = int(np.argmax(snr_profile_pre)) if snr_profile_pre.size > 0 else None
            snr_val = float(np.max(snr_profile_pre)) if snr_profile_pre.size > 0 else 0.0
        else:
            # Si no hay patch, usar el SNR raw como fallback
            snr_val = snr_val_raw
        class_prob, proc_patch = classify_patch(cls_model, patch)
        class_probs_list.append(class_prob)
        is_burst = class_prob >= config.CLASS_PROB
        
        # LÓGICA DE SELECCIÓN DEL MEJOR CANDIDATO PARA COMPOSITE
        # Guardar el primer candidato para compatibilidad
        if first_patch is None:
            first_patch = proc_patch
            # Convertir a tiempo ABSOLUTO del archivo: inicio del slice + offset dentro del slice
            offset_within_slice = (start_sample - (slice_start_idx if slice_start_idx is not None else 0))
            first_start = (absolute_start_time if absolute_start_time is not None else 0.0) 
            first_start += offset_within_slice * config.TIME_RESO * config.DOWN_TIME_RATE
            # Ajuste opcional PRESTO: si la frecuencia de referencia efectiva difiere de la global
            try:
                # Alinear referencia con PRESTO: usar la misma frecuencia tope global
                freq_ref_used = float(np.max(config.FREQ)) if getattr(config, 'FREQ', None) is not None else None
                freq_ref_global = float(np.max(config.FREQ)) if getattr(config, 'FREQ', None) is not None else None
                if freq_ref_used and freq_ref_global:
                    first_start += _presto_time_ref_correction(dm_val, freq_ref_used, freq_ref_global)
            except Exception:
                pass
            first_dm = dm_val
        
        # Almacenar información del candidato para análisis posterior
        candidate_info = {
            'patch': proc_patch,
            'start': first_start,  # tiempo ABSOLUTO del patch
            'dm': dm_val,
            'is_burst': is_burst,
            'confidence': conf,
            'class_prob': class_prob
        }
        all_candidates.append(candidate_info)
        
        # SELECCIÓN INTELIGENTE: Priorizar candidatos BURST sobre NO BURST
        if best_patch is None:
            # Primer candidato siempre se guarda como fallback
            best_patch = proc_patch
            best_start = first_start
            best_dm = dm_val
            best_is_burst = is_burst
        elif is_burst and not best_is_burst:
            # Si encontramos un BURST y el mejor actual es NO BURST, actualizar
            best_patch = proc_patch
            best_start = first_start
            best_dm = dm_val
            best_is_burst = is_burst
        # Si ambos son BURST o ambos son NO BURST, mantener el primero (orden de detección)
        
        # CALCULAR TIEMPO ABSOLUTO DEL CANDIDATO PRIORIZANDO PROCESAMIENTO (pico SNR del patch)
        dt_ds = config.TIME_RESO * config.DOWN_TIME_RATE
        if peak_idx_patch is not None:
            # Tiempo absoluto de inicio del patch dentro del archivo
            slice_offset_samples = (start_sample - (slice_start_idx if slice_start_idx is not None else 0))
            patch_start_abs = (absolute_start_time if absolute_start_time is not None else 0.0) + slice_offset_samples * dt_ds
            absolute_candidate_time = patch_start_abs + peak_idx_patch * dt_ds
        else:
            # Fallback: usar tiempo basado en centro del bbox
            if absolute_start_time is not None:
                absolute_candidate_time = absolute_start_time + t_sec
            else:
                absolute_candidate_time = t_sec

        # Ajuste opcional PRESTO: corregir por frecuencia de referencia si aplica
        try:
            # Alinear referencia con PRESTO: usar la misma frecuencia tope global
            freq_ref_used = float(np.max(config.FREQ)) if getattr(config, 'FREQ', None) is not None else None
            freq_ref_global = float(np.max(config.FREQ)) if getattr(config, 'FREQ', None) is not None else None
            if freq_ref_used and freq_ref_global:
                absolute_candidate_time += _presto_time_ref_correction(dm_val, freq_ref_used, freq_ref_global)
        except Exception:
            pass
        candidate_times_abs.append(float(absolute_candidate_time))
        # Si es BURST, agendar ejecución de PRESTO para esta detección
        if is_burst:
            try:
                burst_jobs.append((float(absolute_candidate_time), float(dm_val)))
            except Exception:
                pass
        
        # Usar chunk_idx en el candidato
        cand = Candidate(
            fits_path.name,
            chunk_idx if chunk_idx is not None else 0,  # AGREGAR CHUNK_ID
            j,  # slice_id
            band_idx if band_idx is not None else 0,  # BAND_ID CORRECTO
            float(conf),
            dm_val,
            absolute_candidate_time,  # USAR TIEMPO ABSOLUTO
            t_sample,
            tuple(map(int, box)),
            snr_val,  # SNR CORREGIDO
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
        
        # Mensaje informativo sobre el candidato encontrado
        try:
            from .logging.logging_config import get_global_logger
            global_logger = get_global_logger()
            global_logger.candidate_detected(dm_val, absolute_candidate_time, conf, class_prob, is_burst, snr_val_raw, snr_val)
        except ImportError:
            # Fallback al logger original
            logger.info(
                f"Candidato DM {dm_val:.2f} t={absolute_candidate_time:.3f}s conf={conf:.2f} class={class_prob:.2f} → {'BURST' if is_burst else 'no burst'}"
            )
            logger.info(
                f"SNR Raw: {snr_val_raw:.2f}σ, SNR Patch Dedispersado: {snr_val:.2f}σ (guardado en CSV)"
            )
    # SELECCIONAR EL CANDIDATO FINAL PARA EL COMPOSITE
    # Si hay múltiples candidatos, priorizar BURST sobre NO BURST
    # Si no hay BURST, usar el primer candidato
    final_patch = best_patch if best_patch is not None else first_patch
    final_start = best_start if best_start is not None else first_start
    final_dm = best_dm if best_dm is not None else first_dm
    
    # Log informativo sobre la selección del candidato
    if len(all_candidates) > 1:
        burst_count = sum(1 for c in all_candidates if c['is_burst'])
        if global_logger:
            global_logger.logger.info(
                f"{Colors.OKCYAN}Slice {j} - {band_name}: {len(all_candidates)} candidatos "
                f"({burst_count} BURST, {len(all_candidates) - burst_count} NO BURST). "
                f"Seleccionado: {'BURST' if best_is_burst else 'NO BURST'} (DM={final_dm:.2f}){Colors.ENDC}"
            )
    
    return {
        "top_conf": top_conf,
        "top_boxes": top_boxes,
        "class_probs_list": class_probs_list,
        "first_patch": final_patch,  # USAR EL MEJOR CANDIDATO SELECCIONADO
        "first_start": final_start,  # USAR EL MEJOR CANDIDATO SELECCIONADO
        "first_dm": final_dm,        # USAR EL MEJOR CANDIDATO SELECCIONADO
        "img_rgb": img_rgb,
        "cand_counter": cand_counter,
        "n_bursts": n_bursts,
        "n_no_bursts": n_no_bursts,
        "prob_max": prob_max,
        "patch_path": patch_path,
        "best_is_burst": best_is_burst,  # INFORMACIÓN ADICIONAL PARA DEBUG
        "total_candidates": len(all_candidates),  # INFORMACIÓN ADICIONAL PARA DEBUG
        "candidate_times_abs": candidate_times_abs,
        "burst_jobs": burst_jobs,
    }

def process_slice(
    j,
    dm_time,
    block,
    slice_len,
    det_model,
    cls_model,
    fits_path,
    save_dir,
    freq_down,
    csv_file,
    time_reso_ds,
    band_configs,
    snr_list,
    waterfall_dispersion_dir,
    waterfall_dedispersion_dir,
    config,
    absolute_start_time=None,
    composite_dir=None,
    detections_dir=None,
    patches_dir=None,
    chunk_idx=None,  #  ID del chunk
    force_plots: bool = False,
    slice_start_idx: int | None = None,  # NUEVO: inicio del slice en muestras (dominio decimado)
    slice_end_idx: int | None = None,    # NUEVO: fin exclusivo del slice (dominio decimado)
):
    """Procesa un slice con tiempo absoluto para continuidad temporal entre chunks.
    
    Args:
        absolute_start_time: Tiempo absoluto de inicio del slice en segundos desde el inicio del archivo
        chunk_idx: ID del chunk donde se encuentra este slice
    """
    # Obtener el logger global para mensajes informativos
    try:
        from .logging.logging_config import get_global_logger, Colors
        global_logger = get_global_logger()
    except ImportError:
        global_logger = None
    
    # Mensaje de inicio de procesamiento del slice
    if global_logger:
        chunk_info = f" (chunk {chunk_idx:03d})" if chunk_idx is not None else ""
        global_logger.logger.info(f"{Colors.PROCESSING} Procesando slice {j:03d}{chunk_info}{Colors.ENDC}")
    
    # Permitir slicing dinámico: si se proveen índices usar esos; si no, usar esquema uniforme
    if slice_start_idx is not None and slice_end_idx is not None:
        start_idx = int(slice_start_idx)
        end_idx = int(slice_end_idx)
    else:
        start_idx = slice_len * j
        end_idx = slice_len * (j + 1)

    # Log: muestras exactas usadas en el slice (dominio decimado)
    try:
        real_samples = end_idx - start_idx
        if global_logger:
            chunk_info = f" (chunk {chunk_idx:03d})" if chunk_idx is not None else ""
            global_logger.logger.info(
                f"{Colors.PROCESSING} Slice {j:03d}{chunk_info}: {real_samples} muestras reales "
                f"(decimado) [{start_idx}→{end_idx}){Colors.ENDC}"
            )
        else:
            logger.info(
                f"Slice {j:03d}: {end_idx - start_idx} muestras reales (decimado) "
                f"[{start_idx}→{end_idx})"
            )
    except Exception:
        # Fallback silencioso si falla el logger global/Colors
        logger.info(
            f"Slice {j:03d}: {end_idx - start_idx} muestras reales (decimado) "
            f"[{start_idx}→{end_idx})"
        )

    slice_cube = dm_time[:, :, start_idx:end_idx]
    waterfall_block = block[start_idx:end_idx]
    if slice_cube.size == 0 or waterfall_block.size == 0:
        logger.warning(f"Slice {j}: slice_cube o waterfall_block vacío, saltando...")
        return 0, 0, 0, 0.0
    
    # CALCULAR TIEMPO ABSOLUTO DEL SLICE SI NO SE PROPORCIONA
    if absolute_start_time is None:
        # Tiempo relativo al chunk (modo antiguo)
        absolute_start_time = start_idx * config.TIME_RESO * config.DOWN_TIME_RATE
    
    # Mensaje sobre creación de waterfall dispersado
    if global_logger:
        global_logger.logger.debug(f"{Colors.OKCYAN} Creando waterfall dispersado para slice {j}{Colors.ENDC}")
    
    # Crear carpeta de waterfall dispersado solo si hay datos para procesar
    if waterfall_block.size > 0:
        waterfall_dispersion_dir.mkdir(parents=True, exist_ok=True)
        wf_disp_path = plot_waterfall_block(
            data_block=waterfall_block, # Bloque de datos
            freq=freq_down, # Frecuencia
            time_reso=time_reso_ds, # Resolución temporal
            block_size=waterfall_block.shape[0], # Tamaño del bloque
            block_idx=j, # Índice del bloque
            save_dir=waterfall_dispersion_dir, # Directorio de guardado
            filename=fits_path.stem, # Nombre del archivo
            normalize=True, # Normalizar
            absolute_start_time=absolute_start_time, # Tiempo absoluto
        ) # Guardar el plot y obtener ruta
    
    slice_has_candidates = False # Indica si el slice tiene candidatos
    cand_counter = 0 # Contador de candidatos
    n_bursts = 0 # Contador de candidatos de tipo burst
    n_no_bursts = 0 # Contador de candidatos de tipo no burst
    prob_max = 0.0 # Probabilidad máxima de detección
    
    fits_stem = fits_path.stem # Nombre del archivo
    # Use chunked directories if provided
    if composite_dir is not None:
        comp_path = composite_dir / f"{fits_stem}_slice{j:03d}.png" # Directorio de guardado
    else:
        comp_path = save_dir / "Composite" / f"{fits_stem}_slice{j:03d}.png"

    if patches_dir is not None:
        patch_path = patches_dir / f"{fits_stem}_slice{j:03d}.png"
    else:
        patch_path = save_dir / "Patches" / f"{fits_stem}_slice{j:03d}.png"

    if detections_dir is not None:
        out_img_path = detections_dir / f"{fits_stem}_slice{j:03d}.png"
    else:
        out_img_path = save_dir / "Detections" / f"{fits_stem}_slice{j:03d}.png"

    # Calcular cantidad de slices total en este bloque para propósitos de visualización
    time_slice = block.shape[0] // slice_len
    if block.shape[0] % slice_len != 0:
        time_slice += 1

    # Recolector de trabajos PRESTO a nivel de slice
    presto_burst_jobs: list[tuple[float, float]] = []

    for band_idx, band_suffix, band_name in band_configs:
        band_img = slice_cube[band_idx]
        band_result = process_band(
            det_model,
            cls_model,
            band_img,
            end_idx - start_idx,  # usar muestras REALES del slice
            j,
            fits_path,
            save_dir,
            block,
            freq_down,
            csv_file,
            time_reso_ds,
            snr_list,
            config,
            absolute_start_time=absolute_start_time,  # PASAR TIEMPO ABSOLUTO
            patches_dir=patches_dir,  # PASAR CARPETA DE PATCHES POR CHUNK
            chunk_idx=chunk_idx,  # PASAR CHUNK_ID
            band_idx=band_idx,  # PASAR BAND_ID CORRECTO
            slice_start_idx=start_idx,  # PASAR INICIO REAL DEL SLICE
        )
        cand_counter += band_result["cand_counter"]
        n_bursts += band_result["n_bursts"]
        n_no_bursts += band_result["n_no_bursts"]
        prob_max = max(prob_max, band_result["prob_max"])
        if len(band_result["top_conf"]) > 0:
            slice_has_candidates = True

        dedisp_block = None

        if slice_has_candidates or force_plots:
            if slice_has_candidates and global_logger:
                global_logger.slice_completed(j, cand_counter, n_bursts, n_no_bursts)

            dm_to_use = band_result["first_dm"] if band_result["first_dm"] is not None else 0.0
            waterfall_dedispersion_dir.mkdir(parents=True, exist_ok=True)
            start = start_idx
            block_len = end_idx - start_idx
            dedisp_block = dedisperse_block(block, freq_down, dm_to_use, start, block_len)
            if global_logger:
                global_logger.creating_waterfall("dedispersado", j, dm_to_use)
                global_logger.generating_plots()

            comp_ctx = {
                'waterfall_dispersion_path': str(wf_disp_path) if 'wf_disp_path' in locals() else None,
            }
            save_all_plots(
                waterfall_block,
                dedisp_block,
                band_result["img_rgb"],
                band_result["first_patch"],
                band_result["first_start"],
                band_result["first_dm"],
                band_result["top_conf"],
                band_result["top_boxes"],
                band_result["class_probs_list"],
                comp_path,
                j,
                time_slice,
                band_name,
                band_suffix,
                fits_stem,
                end_idx - start_idx,
                normalize=True,
                off_regions=None,
                thresh_snr=config.SNR_THRESH,
                band_idx=band_idx,
                patch_path=band_result["patch_path"],
                waterfall_dedispersion_dir=waterfall_dedispersion_dir,
                freq_down=freq_down,
                time_reso_ds=time_reso_ds,
                detections_dir=detections_dir,
                out_img_path=out_img_path,
                absolute_start_time=absolute_start_time,  # PASAR TIEMPO ABSOLUTO
                chunk_idx=chunk_idx,  # PASAR CHUNK_ID
                force_plots=force_plots,
                plot_context=comp_ctx,
            )
        else:
            if global_logger:
                global_logger.logger.debug(f"{Colors.OKCYAN} Slice {j}: Sin candidatos detectados{Colors.ENDC}")

        # Acumular trabajos PRESTO (tiempos absolutos, DM) para candidatos BURST de esta banda
        try:
            presto_burst_jobs.extend(band_result.get("burst_jobs", []))
        except Exception:
            pass
    
    # PRESTO deshabilitado por solicitud del usuario

    return cand_counter, n_bursts, n_no_bursts, prob_max 


def _run_presto_for_burst_jobs(*args, **kwargs):
    return None
