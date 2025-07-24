from ..visualization.plot_manager import save_all_plots
from ..visualization.image_utils import plot_waterfall_block, preprocess_img, postprocess_img
from ..detection.dedispersion import dedisperse_patch
from ..detection.dedispersion import dedisperse_block
from ..detection.utils import detect, classify_patch
from ..detection.metrics import compute_snr
from ..detection.astro_conversions import pixel_to_physical
from ..io.candidate_utils import append_candidate
from ..io.candidate import Candidate
from ..preprocessing.slice_len_utils import update_slice_len_dynamic
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
    absolute_start_time=None
):
    """Procesa una banda con tiempo absoluto para continuidad temporal.
    
    Args:
        absolute_start_time: Tiempo absoluto de inicio del slice en segundos desde el inicio del archivo
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
    patch_dir = save_dir / "Patches" / fits_path.stem
    patch_path = patch_dir / f"patch_slice{j}_band{band_img.shape[0] if hasattr(band_img, 'shape') else 0}.png"
    class_probs_list = []
    cand_counter = 0
    n_bursts = 0
    n_no_bursts = 0
    prob_max = 0.0
    
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
        
        # 🕐 CALCULAR TIEMPO ABSOLUTO DEL CANDIDATO
        if absolute_start_time is not None:
            absolute_candidate_time = absolute_start_time + t_sec
        else:
            absolute_candidate_time = t_sec  # Tiempo relativo al slice
        
        cand = Candidate(
            fits_path.name,
            j,
            band_img.shape[0] if hasattr(band_img, 'shape') else 0,
            float(conf),
            dm_val,
            absolute_candidate_time,  # 🕐 USAR TIEMPO ABSOLUTO
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
            f"Candidato DM {dm_val:.2f} t={absolute_candidate_time:.3f}s conf={conf:.2f} class={class_prob:.2f} → {'BURST' if is_burst else 'no burst'}"
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

def process_slice(j, dm_time, data, slice_len, det_model, cls_model, fits_path, save_dir, freq_down, csv_file, time_reso_ds, band_configs, snr_list, waterfall_dispersion_dir, waterfall_dedispersion_dir, config, absolute_start_time=None):
    """Procesa un slice con tiempo absoluto para continuidad temporal entre chunks.
    
    Args:
        absolute_start_time: Tiempo absoluto de inicio del slice en segundos desde el inicio del archivo
    """
    slice_cube = dm_time[:, :, slice_len * j : slice_len * (j + 1)]
    waterfall_block = data[j * slice_len : (j + 1) * slice_len]
    if slice_cube.size == 0 or waterfall_block.size == 0:
        logger.warning(f"Slice {j}: slice_cube o waterfall_block vacío, saltando...")
        return 0, 0, 0, 0.0
    
    # 🕐 CALCULAR TIEMPO ABSOLUTO DEL SLICE SI NO SE PROPORCIONA
    if absolute_start_time is None:
        # Tiempo relativo al chunk (modo antiguo)
        absolute_start_time = j * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
    
    waterfall_dispersion_dir.mkdir(parents=True, exist_ok=True)
    if waterfall_block.size > 0:
        plot_waterfall_block(
            data_block=waterfall_block,
            freq=freq_down,
            time_reso=time_reso_ds,
            block_size=waterfall_block.shape[0],
            block_idx=j,
            save_dir=waterfall_dispersion_dir,
            filename=fits_path.stem,
            normalize=True,
            absolute_start_time=absolute_start_time,  # 🕐 PASAR TIEMPO ABSOLUTO
        )
    
    slice_has_candidates = False
    cand_counter = 0
    n_bursts = 0
    n_no_bursts = 0
    prob_max = 0.0
    
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
            data,
            freq_down,
            csv_file,
            time_reso_ds,
            snr_list,
            config,
            absolute_start_time=absolute_start_time,  # 🕐 PASAR TIEMPO ABSOLUTO
        )
        cand_counter += band_result["cand_counter"]
        n_bursts += band_result["n_bursts"]
        n_no_bursts += band_result["n_no_bursts"]
        prob_max = max(prob_max, band_result["prob_max"])
        if len(band_result["top_conf"]) > 0:
            slice_has_candidates = True
        
        composite_dir = save_dir / "Composite" / fits_path.stem
        comp_path = composite_dir / f"slice{j}_band{band_idx}.png"
        detections_dir = save_dir / "Detections" / fits_path.stem
        detections_dir.mkdir(parents=True, exist_ok=True)
        out_img_path = detections_dir / f"slice{j}_{band_suffix}.png"
        dedisp_block = None
        
        if slice_has_candidates:
            if band_result["first_patch"] is not None:
                waterfall_dedispersion_dir.mkdir(parents=True, exist_ok=True)
                start = j * slice_len
                dedisp_block = dedisperse_block(data, freq_down, band_result["first_dm"], start, slice_len)
            else:
                waterfall_dedispersion_dir.mkdir(parents=True, exist_ok=True)
                start = j * slice_len
                dedisp_block = dedisperse_block(data, freq_down, 0.0, start, slice_len)
            
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
                len(band_configs),
                band_name,
                band_suffix,
                fits_path.stem,
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
                absolute_start_time=absolute_start_time,  # 🕐 PASAR TIEMPO ABSOLUTO
            )
    
    return cand_counter, n_bursts, n_no_bursts, prob_max
