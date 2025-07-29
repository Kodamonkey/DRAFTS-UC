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
logger = logging.getLogger(__name__)

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
):
    """Procesa una banda con tiempo absoluto para continuidad temporal.
    
    Args:
        absolute_start_time: Tiempo absoluto de inicio del slice en segundos desde el inicio del archivo
        chunk_idx: ID del chunk donde se encuentra este slice
        band_idx: ID de la banda (0=fullband, 1=lowband, 2=highband)
    """
    img_tensor = preprocess_img(band_img)
    top_conf, top_boxes = detect(det_model, img_tensor)
    img_rgb = postprocess_img(img_tensor)
    if top_boxes is None:
        top_conf = []
        top_boxes = []
    first_patch = None
    first_start = None
    first_dm = None
    if patches_dir is not None:
        patch_path = patches_dir / f"patch_slice{j}_band{band_img.shape[0] if hasattr(band_img, 'shape') else 0}.png"
    else:
        patch_dir = save_dir / "Patches" / fits_path.stem
        patch_path = patch_dir / f"patch_slice{j}_band{band_img.shape[0] if hasattr(band_img, 'shape') else 0}.png"
    class_probs_list = []
    cand_counter = 0
    n_bursts = 0
    n_no_bursts = 0
    prob_max = 0.0
    
    for conf, box in zip(top_conf, top_boxes):
        dm_val, t_sec, t_sample = extract_candidate_dm(
            (box[0] + box[2]) / 2,
            (box[1] + box[3]) / 2,
            slice_len,
        )
        
        # 🧩 CORRECCIÓN: Usar compute_snr_profile para SNR consistente con composite
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
        global_sample = j * slice_len + int(t_sample)
        patch, start_sample = dedisperse_patch(
            data, freq_down, dm_val, global_sample
        )
        
        # Calcular SNR del patch dedispersado (SNR final para CSV)
        snr_val = 0.0  # Valor por defecto
        if patch is not None and patch.size > 0:
            from .analysis.snr_utils import find_snr_peak
            snr_patch_profile, _ = compute_snr_profile(patch)
            snr_val, _, _ = find_snr_peak(snr_patch_profile)
            # ✅ IMPORTANTE: Este es el SNR que se guarda en CSV (patch dedispersado)
        else:
            # Si no hay patch, usar el SNR raw como fallback
            snr_val = snr_val_raw
        class_prob, proc_patch = classify_patch(cls_model, patch)
        class_probs_list.append(class_prob)
        is_burst = class_prob >= config.CLASS_PROB
        if first_patch is None:
            first_patch = proc_patch
            first_start = start_sample * config.TIME_RESO * config.DOWN_TIME_RATE
            first_dm = dm_val
        
        # 🕐 CALCULAR TIEMPO ABSOLUTO DEL CANDIDATO
        if absolute_start_time is not None:
            absolute_candidate_time = absolute_start_time + t_sec
        else:
            absolute_candidate_time = t_sec  # Tiempo relativo al slice
        
        # 🧩 NUEVO: Usar chunk_idx en el candidato
        cand = Candidate(
            fits_path.name,
            chunk_idx if chunk_idx is not None else 0,  # AGREGAR CHUNK_ID
            j,  # slice_id
            band_idx if band_idx is not None else 0,  # BAND_ID CORRECTO
            float(conf),
            dm_val,
            absolute_candidate_time,  # 🕐 USAR TIEMPO ABSOLUTO
            t_sample,
            tuple(map(int, box)),
            snr_val,  # 🧩 SNR CORREGIDO
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
            f"Candidato DM {dm_val:.2f} t={absolute_candidate_time:.3f}s conf={conf:.2f} class={class_prob:.2f} → {'BURST' if is_burst else 'no burst'}"
        )
        # ✅ LOGGING DETALLADO: Mostrar valores de SNR para transparencia
        logger.info(
            f"  📊 SNR Raw: {snr_val_raw:.2f}σ, SNR Patch Dedispersado: {snr_val:.2f}σ (guardado en CSV)"
        )
    return {
        "top_conf": top_conf,
        "top_boxes": top_boxes,
        "class_probs_list": class_probs_list,
        "first_patch": first_patch,
        "first_start": first_start,
        "first_dm": first_dm,
        "img_rgb": img_rgb,
        "cand_counter": cand_counter,
        "n_bursts": n_bursts,
        "n_no_bursts": n_no_bursts,
        "prob_max": prob_max,
        "patch_path": patch_path,
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
):
    """Procesa un slice con tiempo absoluto para continuidad temporal entre chunks.
    
    Args:
        absolute_start_time: Tiempo absoluto de inicio del slice en segundos desde el inicio del archivo
        chunk_idx: ID del chunk donde se encuentra este slice
    """
    slice_cube = dm_time[:, :, slice_len * j : slice_len * (j + 1)]
    waterfall_block = block[j * slice_len : (j + 1) * slice_len]
    if slice_cube.size == 0 or waterfall_block.size == 0:
        logger.warning(f"Slice {j}: slice_cube o waterfall_block vacío, saltando...")
        return 0, 0, 0, 0.0
    
    # CALCULAR TIEMPO ABSOLUTO DEL SLICE SI NO SE PROPORCIONA
    if absolute_start_time is None:
        # Tiempo relativo al chunk (modo antiguo)
        absolute_start_time = j * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
    
    waterfall_dispersion_dir.mkdir(parents=True, exist_ok=True)
    if waterfall_block.size > 0:
        plot_waterfall_block(
            data_block=waterfall_block, # Bloque de datos
            freq=freq_down, # Frecuencia
            time_reso=time_reso_ds, # Resolución temporal
            block_size=waterfall_block.shape[0], # Tamaño del bloque
            block_idx=j, # Índice del bloque
            save_dir=waterfall_dispersion_dir, # Directorio de guardado
            filename=fits_path.stem, # Nombre del archivo
            normalize=True, # Normalizar
            absolute_start_time=absolute_start_time, # Tiempo absoluto
        ) # Guardar el plot
    
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

    # Calculate time_slice if not provided
    time_slice = block.shape[0] // slice_len # Tamaño del slice
    if block.shape[0] % slice_len != 0: # Si el tamaño del bloque no es divisible por el tamaño del slice
        time_slice += 1 # Incrementar el tamaño del slice

    for band_idx, band_suffix, band_name in band_configs:
        band_img = slice_cube[band_idx]
        band_result = process_band(
            det_model,
            cls_model,
            band_img,
            slice_len,
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
        )
        cand_counter += band_result["cand_counter"]
        n_bursts += band_result["n_bursts"]
        n_no_bursts += band_result["n_no_bursts"]
        prob_max = max(prob_max, band_result["prob_max"])
        if len(band_result["top_conf"]) > 0:
            slice_has_candidates = True

        dedisp_block = None

        if slice_has_candidates:
            if band_result["first_patch"] is not None:
                waterfall_dedispersion_dir.mkdir(parents=True, exist_ok=True)
                start = j * slice_len
                dedisp_block = dedisperse_block(block, freq_down, band_result["first_dm"], start, slice_len)
            else:
                waterfall_dedispersion_dir.mkdir(parents=True, exist_ok=True)
                start = j * slice_len
                dedisp_block = dedisperse_block(block, freq_down, 0.0, start, slice_len)

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
                slice_len,
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
            )
    
    return cand_counter, n_bursts, n_no_bursts, prob_max 