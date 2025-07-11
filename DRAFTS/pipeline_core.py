"""Helper utilities for the FRB detection pipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Tuple, List

import numpy as np

from . import config
from .io import load_fits_file
from .filterbank_io import load_fil_file
from .preprocessing import downsample_data
from .slice_len_utils import update_slice_len_dynamic
from .detection_utils import (
    detect_candidates as _detect,
    classify_patch as _classify_patch,
)
from .image_utils import preprocess_img, postprocess_img
from .astro_conversions import pixel_to_physical
from .metrics import compute_snr
from .dedispersion import dedisperse_patch, dedisperse_block
from .visualization import save_plot, save_patch_plot, save_slice_summary
from .rfi_utils import apply_rfi_cleaning
from .candidate import Candidate
from .csv_utils import write_candidate as _write_candidate_to_csv

logger = logging.getLogger(__name__)


def load_and_prepare_data(fits_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a FITS or filterbank file and return preprocessed data and downsampled frequency."""
    if fits_path.suffix.lower() == ".fits":
        data = load_fits_file(str(fits_path))
    else:
        data = load_fil_file(str(fits_path))

    data = np.vstack([data, data[::-1, :]])
    data = downsample_data(data)

    if config.FREQ_RESO == 0 or config.DOWN_FREQ_RATE == 0:
        raise ValueError(
            f"Par\u00e1metros de frecuencia inv\u00e1lidos: FREQ_RESO={config.FREQ_RESO}, DOWN_FREQ_RATE={config.DOWN_FREQ_RATE}"
        )

    freq_down = np.mean(
        config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE),
        axis=1,
    )
    return data, freq_down


def compute_slice_info(width_total: int) -> Tuple[int, int, float]:
    """Compute slice length and number of slices using dynamic configuration."""
    slice_len, real_duration_ms = update_slice_len_dynamic()
    time_slice = (width_total + slice_len - 1) // slice_len
    return slice_len, time_slice, real_duration_ms


def process_slice(
    det_model,
    cls_model,
    j: int,
    data: np.ndarray,
    slice_cube: np.ndarray,
    waterfall_block: np.ndarray,
    freq_down: np.ndarray,
    csv_file: Path,
    fits_path: Path,
    slice_len: int,
    time_slice: int,
    save_dir: Path,
) -> Tuple[int, int, int, float, List[float]]:
    """Process a single time slice and return aggregated metrics."""
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

    waterfall_dispersion_dir = save_dir / "waterfall_dispersion" / fits_path.stem
    waterfall_dedispersion_dir = save_dir / "waterfall_dedispersion" / fits_path.stem
    freq_ds = freq_down
    time_reso_ds = config.TIME_RESO * config.DOWN_TIME_RATE

    # RFI cleaning
    if hasattr(config, "RFI_ENABLE_ALL_FILTERS") and config.RFI_ENABLE_ALL_FILTERS:
        try:
            logger.info("Aplicando limpieza de RFI al slice %d", j)
            waterfall_block, _ = apply_rfi_cleaning(
                waterfall_block,
                stokes_v=None,
                output_dir=save_dir / "rfi_diagnostics" if getattr(config, "RFI_SAVE_DIAGNOSTICS", False) else None,
            )
        except Exception as e:  # pragma: no cover - sanity
            logger.warning("Error en limpieza de RFI para slice %d: %s", j, e)

    if waterfall_block.size > 0:
        waterfall_dispersion_dir.mkdir(parents=True, exist_ok=True)
        from .image_utils import plot_waterfall_block

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
        img_rgb = postprocess_img(img_tensor)

        if top_boxes is None:
            top_conf = []
            top_boxes = []

        first_patch = None
        first_start = None
        first_dm = None
        patch_dir = save_dir / "Patches" / fits_path.stem
        patch_path = patch_dir / f"patch_slice{j}_band{band_idx}.png"
        class_probs_list: List[float] = []

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
                t_sec,
                snr_val,
            )
            cand_counter += 1
            if is_burst:
                n_bursts += 1
            else:
                n_no_bursts += 1
            prob_max = max(prob_max, float(conf))
            _write_candidate_to_csv(csv_file, cand)
            logger.info(
                "Candidato DM %.2f t=%.3f s conf=%.2f class=%.2f -> %s",
                dm_val,
                t_sec,
                conf,
                class_prob,
                "BURST" if is_burst else "no burst",
            )

        if first_patch is not None:
            waterfall_dedispersion_dir.mkdir(parents=True, exist_ok=True)
            start = j * slice_len
            dedisp_block = dedisperse_block(data, freq_down, first_dm, start, slice_len)
            if dedisp_block.size > 0:
                from .image_utils import plot_waterfall_block

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

            save_patch_plot(
                first_patch,
                patch_path,
                freq_down,
                config.TIME_RESO * config.DOWN_TIME_RATE,
                first_start,
                off_regions=None,
                thresh_snr=config.SNR_THRESH,
                band_idx=band_idx,
                band_name=band_name,
            )

        composite_dir = save_dir / "Composite" / fits_path.stem
        comp_path = composite_dir / f"slice{j}_band{band_idx}.png"
        save_slice_summary(
            waterfall_block,
            dedisp_block if first_patch is not None else waterfall_block,
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

    return cand_counter, n_bursts, n_no_bursts, prob_max, snr_list
