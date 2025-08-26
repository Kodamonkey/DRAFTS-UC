"""Motor de detección FRB - detecta candidatos, los clasifica y coordina visualizaciones.

Este módulo contiene las funciones principales para:
1. detect_and_classify_candidates_in_band(): Detección y clasificación en una banda específica
2. process_slice_with_multiple_bands(): Coordinación de múltiples bandas y generación de visualizaciones

Cada función tiene responsabilidades claras y nombres descriptivos que explican su propósito.
"""
from __future__ import annotations

# Standard library imports
import logging

# Third-party imports
import numpy as np

# Local imports
from ..analysis.snr_utils import compute_presto_matched_snr, compute_snr_profile
from ..detection.model_interface import classify_patch, detect
from ..logging.logging_config import Colors, get_global_logger
from ..output.candidate_manager import Candidate, append_candidate
from ..preprocessing.dm_candidate_extractor import extract_candidate_dm
from ..preprocessing.dedispersion import dedisperse_block, dedisperse_patch
from ..preprocessing.slice_len_calculator import update_slice_len_dynamic
from ..visualization.visualization_unified import (
    postprocess_img,
    preprocess_img,
    save_all_plots
)

# Setup logger
logger = logging.getLogger(__name__)

def detect_and_classify_candidates_in_band(
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
    """Detecta candidatos FRB en una banda de frecuencia, los clasifica y selecciona el mejor.
    
    Esta función realiza el flujo completo de detección y clasificación para una banda específica:
    1. Detecta candidatos usando el modelo de detección
    2. Clasifica cada candidato como BURST o NO BURST
    3. Calcula SNR y parámetros de cada candidato
    4. Selecciona el mejor candidato para visualización
    5. Guarda todos los candidatos en CSV
    
    Args:
        det_model: Modelo de detección de objetos
        cls_model: Modelo de clasificación binaria
        band_img: Imagen de la banda de frecuencia
        slice_len: Longitud del slice en muestras
        j: Índice del slice
        fits_path: Path del archivo FITS
        save_dir: Directorio de guardado
        data: Datos del bloque completo
        freq_down: Frecuencias decimadas
        csv_file: Archivo CSV para guardar candidatos
        time_reso_ds: Resolución temporal decimada
        snr_list: Lista para acumular valores SNR
        config: Configuración del pipeline
        absolute_start_time: Tiempo absoluto de inicio del slice
        patches_dir: Directorio para guardar patches
        chunk_idx: ID del chunk donde se encuentra este slice
        band_idx: ID de la banda (0=fullband, 1=lowband, 2=highband)
        slice_start_idx: Inicio del slice en muestras decimadas
    """
    # Obtener el logger global para mensajes informativos
    try:
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
                freq_ref_used = float(freq_down.max()) if hasattr(freq_down, 'max') else None
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
            freq_ref_used = float(freq_down.max()) if hasattr(freq_down, 'max') else None
            freq_ref_global = float(np.max(config.FREQ)) if getattr(config, 'FREQ', None) is not None else None
            if freq_ref_used and freq_ref_global:
                absolute_candidate_time += _presto_time_ref_correction(dm_val, freq_ref_used, freq_ref_global)
        except Exception:
            pass
        candidate_times_abs.append(float(absolute_candidate_time))
        
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
        
        # FILTRADO: Solo guardar candidatos BURST si SAVE_ONLY_BURST está activado
        if not config.SAVE_ONLY_BURST or is_burst:
            append_candidate(csv_file, cand.to_row())
            
            # Mensaje informativo sobre el candidato encontrado
            try:
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
        else:
            # Si SAVE_ONLY_BURST está activado y no es BURST, solo contar pero no guardar
            logger.debug(
                f"Candidato NO BURST filtrado (SAVE_ONLY_BURST=True): DM {dm_val:.2f} t={absolute_candidate_time:.3f}s "
                f"conf={conf:.2f} class={class_prob:.2f} → NO BURST (no guardado)"
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
    }

def process_slice_with_multiple_bands(
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
    """Coordina el procesamiento de múltiples bandas en un slice temporal y genera visualizaciones.
    
    Esta función es el coordinador principal para un slice:
    1. Procesa cada banda de frecuencia (fullband, lowband, highband)
    2. Llama a detect_and_classify_candidates_in_band() para cada banda
    3. Genera visualizaciones (waterfall, composite, patches)
    4. Maneja la dedispersión del bloque completo
    5. Retorna estadísticas consolidadas del slice
    
    Args:
        j: Índice del slice
        dm_time: Cubo DM-tiempo del bloque
        block: Bloque de datos completo
        slice_len: Longitud del slice en muestras
        det_model: Modelo de detección de objetos
        cls_model: Modelo de clasificación binaria
        fits_path: Path del archivo FITS
        save_dir: Directorio de guardado
        freq_down: Frecuencias decimadas
        csv_file: Archivo CSV para guardar candidatos
        time_reso_ds: Resolución temporal decimada
        band_configs: Configuración de bandas a procesar
        snr_list: Lista para acumular valores SNR
        config: Configuración del pipeline
        absolute_start_time: Tiempo absoluto de inicio del slice
        composite_dir: Directorio para plots compuestos
        detections_dir: Directorio para plots de detecciones
        patches_dir: Directorio para patches
        chunk_idx: ID del chunk donde se encuentra este slice
        force_plots: Forzar generación de plots incluso sin candidatos
        slice_start_idx: Inicio del slice en muestras decimadas
        slice_end_idx: Fin del slice en muestras decimadas
    """
    # Obtener el logger global para mensajes informativos
    try:
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

    for band_idx, band_suffix, band_name in band_configs:
        band_img = slice_cube[band_idx]
        band_result = detect_and_classify_candidates_in_band(
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

        # FILTRADO: Si SAVE_ONLY_BURST está activado, solo generar visualizaciones si hay candidatos BURST
        should_generate_plots = False
        if config.SAVE_ONLY_BURST:
            # Solo generar plots si hay candidatos BURST o si force_plots está activado
            should_generate_plots = (n_bursts > 0) or force_plots
        else:
            # Comportamiento normal: generar plots si hay candidatos o force_plots
            should_generate_plots = slice_has_candidates or force_plots
        
        if should_generate_plots:
            if slice_has_candidates and global_logger:
                global_logger.slice_completed(j, cand_counter, n_bursts, n_no_bursts)

            dm_to_use = band_result["first_dm"] if band_result["first_dm"] is not None else 0.0
       
            start = start_idx
            block_len = end_idx - start_idx
            dedisp_block = dedisperse_block(block, freq_down, dm_to_use, start, block_len)
            if global_logger:
                global_logger.creating_waterfall("dedispersado", j, dm_to_use)
                global_logger.generating_plots()

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
                absolute_start_time=absolute_start_time,
                chunk_idx=chunk_idx, 
                force_plots=force_plots,
            )
        else:
            if global_logger:
                if config.SAVE_ONLY_BURST and n_no_bursts > 0:
                    global_logger.logger.debug(f"{Colors.OKCYAN} Slice {j}: Solo candidatos NO BURST detectados (SAVE_ONLY_BURST=True, no plots){Colors.ENDC}")
                else:
                    global_logger.logger.debug(f"{Colors.OKCYAN} Slice {j}: Sin candidatos detectados{Colors.ENDC}")
    
    # FILTRADO: Si SAVE_ONLY_BURST está activado, solo retornar candidatos BURST para estadísticas
    if config.SAVE_ONLY_BURST:
        # Solo contar candidatos BURST para estadísticas y visualizaciones
        effective_cand_counter = n_bursts
        effective_n_bursts = n_bursts
        effective_n_no_bursts = 0  # No contar NO BURST cuando solo se guardan BURST
    else:
        # Comportamiento normal: contar todos los candidatos
        effective_cand_counter = cand_counter
        effective_n_bursts = n_bursts
        effective_n_no_bursts = n_no_bursts
    
    return effective_cand_counter, effective_n_bursts, effective_n_no_bursts, prob_max 