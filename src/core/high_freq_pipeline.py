"""High-frequency FRB pipeline integrating Yong-Kun Zhang's strategies."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from ..config import config
from ..logging.logging_config import get_global_logger
from ..output.candidate_manager import Candidate, append_candidate
from ..visualization.visualization_unified import preprocess_img, postprocess_img, save_all_plots
from .high_freq_strategies import (
    ExpandedDMRange,
    HighFreqCandidate,
    calculate_expanded_dm_range,
    deduplicate_and_rank_candidates,
    dm_aware_validation,
    generate_time_dm_plane,
    monitor_pipeline_health,
    run_bow_tie_strategy,
    zero_dm_classifier,
)


logger = logging.getLogger(__name__)


def _append_candidates_to_csv(
    candidates: list[HighFreqCandidate],
    csv_file: Path,
    fits_path: Path,
    chunk_idx: int,
    slice_idx: int,
    patch_name: str | None,
) -> tuple[int, int]:
    """Persist candidates into the CSV file and return burst counts."""

    n_bursts = 0
    n_no_bursts = 0
    for cand in candidates:
        is_burst = cand.is_burst()
        if not config.SAVE_ONLY_BURST or is_burst:
            candidate_row = Candidate(
                file=fits_path.name,
                chunk_id=int(chunk_idx),
                slice_id=int(slice_idx),
                band_id=int(cand.band_idx),
                prob=float(cand.confidence),
                dm=float(cand.effective_dm()),
                t_sec=float(cand.absolute_time),
                t_sample=int(cand.global_sample),
                box=tuple(map(int, cand.candidate_box or (0, 0, 0, 0))),
                snr=float(cand.snr),
                class_prob=float(cand.class_prob),
                is_burst=bool(is_burst),
                patch_file=patch_name,
                width_ms=(
                    float(cand.width_samples * config.TIME_RESO * config.DOWN_TIME_RATE * 1000.0)
                    if cand.width_samples is not None
                    else None
                ),
            )
            append_candidate(csv_file, candidate_row.to_row())
            try:
                gl = get_global_logger()
                gl.candidate_detected(
                    cand.effective_dm(),
                    cand.absolute_time,
                    cand.confidence,
                    cand.class_prob,
                    is_burst,
                    cand.snr,
                    cand.snr,
                )
            except Exception:
                pass
        if is_burst:
            n_bursts += 1
        else:
            n_no_bursts += 1
    return n_bursts, n_no_bursts


def process_slice_with_multiple_bands_high_freq(
    j: int,
    dm_time: np.ndarray,
    block: np.ndarray,
    slice_len: int,
    cls_model,
    fits_path: Path,
    save_dir: Path,
    freq_down: np.ndarray,
    csv_file: Path,
    time_reso_ds: float,
    band_configs: list[tuple[int, str, str]],
    snr_list: list,
    absolute_start_time: float | None,
    composite_dir: Path | None,
    detections_dir: Path | None,
    patches_dir: Path | None,
    chunk_idx: int | None,
    slice_start_idx: int,
    slice_end_idx: int,
    dm_range: ExpandedDMRange,
) -> dict:
    """Process a slice combining bow-tie recovery and zero-DM validation."""

    try:
        global_logger = get_global_logger()
    except Exception:
        global_logger = None

    start_idx = int(slice_start_idx)
    end_idx = int(slice_end_idx)
    if end_idx <= start_idx:
        return {
            "cand_counter": 0,
            "n_bursts": 0,
            "n_no_bursts": 0,
            "prob_max": 0.0,
            "strategy_1_count": 0,
            "strategy_2_initial_count": 0,
            "strategy_2_validated_count": 0,
            "final_candidate_count": 0,
            "all_candidates_initial": 0,
        }

    slice_cube = dm_time[:, :, start_idx:end_idx]
    waterfall_block = block[start_idx:end_idx]
    if slice_cube.size == 0 or waterfall_block.size == 0:
        return {
            "cand_counter": 0,
            "n_bursts": 0,
            "n_no_bursts": 0,
            "prob_max": 0.0,
            "strategy_1_count": 0,
            "strategy_2_initial_count": 0,
            "strategy_2_validated_count": 0,
            "final_candidate_count": 0,
            "all_candidates_initial": 0,
        }

    chunk_idx_int = int(chunk_idx) if chunk_idx is not None else 0
    fits_stem = fits_path.stem
    if composite_dir is not None:
        comp_path = composite_dir / f"{fits_stem}_slice{j:03d}.png"
    else:
        comp_path = save_dir / "Composite" / f"{fits_stem}_slice{j:03d}.png"

    abs_start = float(absolute_start_time or 0.0)

    zero_dm_candidates = zero_dm_classifier(
        block,
        freq_down,
        start_idx,
        end_idx,
        cls_model=cls_model,
        absolute_start_time=abs_start,
        chunk_idx=chunk_idx_int,
        slice_idx=j,
        dt_ds=time_reso_ds,
        dm_range=dm_range,
    )
    validated_candidates = dm_aware_validation(zero_dm_candidates, block, freq_down, time_reso_ds, dm_range)

    strategy1_candidates: list[HighFreqCandidate] = []
    for band_idx, band_suffix, band_name in band_configs:
        band_img = slice_cube[band_idx]
        band_candidates = run_bow_tie_strategy(
            band_img,
            dm_range,
            block=block,
            freq_down=freq_down,
            start_idx=start_idx,
            absolute_start_time=abs_start,
            cls_model=cls_model,
            chunk_idx=chunk_idx_int,
            slice_idx=j,
            band_idx=band_idx,
            dt_ds=time_reso_ds,
        )
        strategy1_candidates.extend(band_candidates)
        if global_logger:
            try:
                global_logger.band_candidates(band_name, len(band_candidates))
            except Exception:
                pass

    all_candidates = strategy1_candidates + validated_candidates
    final_candidates = deduplicate_and_rank_candidates(all_candidates)

    for cand in final_candidates:
        snr_list.append(float(cand.snr))

    best_candidate = final_candidates[0] if final_candidates else None
    prob_max = max((cand.confidence for cand in final_candidates), default=0.0)
    cand_counter = len(final_candidates)
    n_bursts = sum(1 for cand in final_candidates if cand.is_burst())
    n_no_bursts = cand_counter - n_bursts

    if final_candidates:
        band_idx_plot = int(best_candidate.band_idx)
        band_map = {idx: (suffix, name) for idx, suffix, name in band_configs}
        band_suffix = band_map.get(band_idx_plot, ("fullband", "Full Band"))[0]
        band_name = band_map.get(band_idx_plot, ("fullband", "Full Band"))[1]
        band_img = slice_cube[band_idx_plot]
        img_tensor = preprocess_img(band_img)
        img_rgb = postprocess_img(img_tensor)
        dedisp_dm = best_candidate.effective_dm()
        dedisp_block = generate_dedispersed_block(block, freq_down, start_idx, end_idx, dedisp_dm)
        patch_dir = patches_dir if patches_dir is not None else (save_dir / "Patches" / fits_stem)
        patch_dir.mkdir(parents=True, exist_ok=True)
        patch_path = patch_dir / f"patch_slice{j}_band{band_idx_plot}.png"
        first_patch = best_candidate.processed_patch
        if first_patch is None:
            first_patch = regenerate_patch(block, freq_down, best_candidate, cls_model)
            best_candidate.processed_patch = first_patch
        if global_logger:
            try:
                global_logger.generating_plots()
            except Exception:
                pass
        should_generate_plots = bool(final_candidates) or config.FORCE_PLOTS
        if config.SAVE_ONLY_BURST:
            should_generate_plots = (n_bursts > 0) or config.FORCE_PLOTS
        if should_generate_plots:
            save_all_plots(
                waterfall_block,
                dedisp_block,
                img_rgb,
                first_patch,
                best_candidate.absolute_time,
                dedisp_dm,
                [cand.confidence for cand in final_candidates],
                [cand.candidate_box or (0, 0, 0, 0) for cand in final_candidates],
                [cand.class_prob for cand in final_candidates],
                comp_path,
                j,
                block.shape[0] // slice_len + (1 if block.shape[0] % slice_len != 0 else 0),
                band_name,
                band_suffix,
                fits_stem,
                end_idx - start_idx,
                normalize=True,
                off_regions=None,
                thresh_snr=getattr(config, "SNR_THRESH", 4.0),
                band_idx=band_idx_plot,
                patch_path=patch_path,
                absolute_start_time=abs_start,
                chunk_idx=chunk_idx_int,
                force_plots=config.FORCE_PLOTS,
                candidate_times_abs=[cand.absolute_time for cand in final_candidates],
            )
    else:
        patch_path = None

    extra_bursts, extra_non_bursts = _append_candidates_to_csv(
        final_candidates,
        csv_file,
        fits_path,
        chunk_idx_int,
        j,
        patch_path.name if patch_path is not None else None,
    )
    # Ensure counts align with actual persisted rows when SAVE_ONLY_BURST is enabled.
    if config.SAVE_ONLY_BURST:
        n_bursts = extra_bursts
        n_no_bursts = 0

    return {
        "cand_counter": cand_counter,
        "n_bursts": n_bursts,
        "n_no_bursts": n_no_bursts,
        "prob_max": prob_max,
        "strategy_1_count": len(strategy1_candidates),
        "strategy_2_initial_count": len(zero_dm_candidates),
        "strategy_2_validated_count": len(validated_candidates),
        "final_candidate_count": len(final_candidates),
        "all_candidates_initial": len(strategy1_candidates) + len(validated_candidates),
    }


def regenerate_patch(block: np.ndarray, freq_down: np.ndarray, candidate: HighFreqCandidate, cls_model):
    from ..preprocessing.dedispersion import dedisperse_patch
    patch, _ = dedisperse_patch(block, freq_down, candidate.effective_dm(), candidate.global_sample)
    _, proc_patch = classify_patch_safe(cls_model, patch)
    return proc_patch


def generate_dedispersed_block(
    block: np.ndarray,
    freq_down: np.ndarray,
    start_idx: int,
    end_idx: int,
    dm_val: float,
) -> np.ndarray:
    from ..preprocessing.dedispersion import dedisperse_block

    return dedisperse_block(block, freq_down, dm_val, start_idx, end_idx - start_idx)


def classify_patch_safe(model, patch: np.ndarray):
    from ..detection.model_interface import classify_patch

    if patch is None or patch.size == 0:
        return 0.0, np.zeros((1, 1), dtype=np.float32)
    return classify_patch(model, patch)


def _log_chunk_start(metadata: dict) -> None:
    if not config.DEBUG_FREQUENCY_ORDER:
        return
    start = metadata.get("start_sample")
    chunk_idx = metadata.get("chunk_idx")
    logger.debug("Processing chunk %s starting at sample %s", chunk_idx, start)


def _process_file_chunked_high_freq(
    cls_model,
    fits_path: Path,
    save_dir: Path,
    chunk_samples: int,
    streaming_func,
) -> dict:
    """Run the high-frequency pipeline over a file in streaming chunks."""

    from .data_flow_manager import (
        create_chunk_directories,
        downsample_chunk,
        get_chunk_processing_parameters,
        plan_slices,
        trim_valid_window,
    )
    from ..output.candidate_manager import ensure_csv_header
    from ..logging import log_block_processing, log_streaming_parameters

    csv_file = save_dir / f"{fits_path.stem}.candidates.csv"
    ensure_csv_header(csv_file)

    t_start = time.time()
    cand_counter_total = 0
    n_bursts_total = 0
    n_no_bursts_total = 0
    prob_max_total = 0.0
    snr_list_total: list[float] = []
    actual_chunk_count = 0
    strategy1_total = 0
    strategy2_initial_total = 0
    strategy2_validated_total = 0
    final_candidate_total = 0
    all_candidates_initial_total = 0

    total_samples = config.FILE_LENG
    freq_ds = np.mean(
        config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE),
        axis=1,
    )
    nu_min = float(freq_ds.min())
    nu_max = float(freq_ds.max())
    dt_max_sec = 4.1488e3 * config.DM_max * (nu_min ** -2 - nu_max ** -2)
    overlap_raw = int(np.ceil(dt_max_sec / config.TIME_RESO))
    log_streaming_parameters(chunk_samples, overlap_raw, total_samples, chunk_samples, streaming_func, "fits/fil")

    dm_range = calculate_expanded_dm_range(
        traditional_max_dm=config.DM_max,
        multiplier=getattr(config, "HF_DM_EXPANSION_FACTOR", 3.0),
        step_coarse=getattr(config, "HF_DM_COARSE_STEP", 5.0),
        min_dm=config.DM_min,
        min_step=getattr(config, "HF_DM_MIN_STEP", 1.0),
    )

    for block, metadata in streaming_func(str(fits_path), chunk_samples, overlap_samples=overlap_raw):
        actual_chunk_count += 1
        log_block_processing(actual_chunk_count, block.shape, str(block.dtype), metadata)
        _log_chunk_start(metadata)

        block_ds, dt_ds = downsample_chunk(block)
        chunk_params = get_chunk_processing_parameters(metadata)
        freq_down = chunk_params["freq_down"]
        slice_len = chunk_params["slice_len"]
        overlap_left_ds = chunk_params["overlap_left_ds"]
        overlap_right_ds = chunk_params["overlap_right_ds"]

        dm_time_full = generate_time_dm_plane(block_ds, dm_range)
        block_ds, dm_time, valid_start_ds, valid_end_ds = trim_valid_window(
            block_ds,
            dm_time_full,
            overlap_left_ds,
            overlap_right_ds,
        )

        slices_to_process = plan_slices(block_ds, slice_len, metadata["chunk_idx"])
        composite_dir, detections_dir, patches_dir = create_chunk_directories(save_dir, fits_path, metadata["chunk_idx"])
        chunk_start_time_sec = metadata["start_sample"] * config.TIME_RESO

        for j, start_idx, end_idx in slices_to_process:
            slice_result = process_slice_with_multiple_bands_high_freq(
                j=j,
                dm_time=dm_time,
                block=block_ds,
                slice_len=slice_len,
                cls_model=cls_model,
                fits_path=fits_path,
                save_dir=save_dir,
                freq_down=freq_down,
                csv_file=csv_file,
                time_reso_ds=dt_ds,
                band_configs=config.get_band_configs(),
                snr_list=snr_list_total,
                absolute_start_time=chunk_start_time_sec + start_idx * dt_ds,
                composite_dir=composite_dir,
                detections_dir=detections_dir,
                patches_dir=patches_dir,
                chunk_idx=metadata["chunk_idx"],
                slice_start_idx=start_idx,
                slice_end_idx=end_idx,
                dm_range=dm_range,
            )

            cand_counter_total += slice_result["cand_counter"]
            n_bursts_total += slice_result["n_bursts"]
            n_no_bursts_total += slice_result["n_no_bursts"]
            prob_max_total = max(prob_max_total, slice_result["prob_max"])
            strategy1_total += slice_result["strategy_1_count"]
            strategy2_initial_total += slice_result["strategy_2_initial_count"]
            strategy2_validated_total += slice_result["strategy_2_validated_count"]
            final_candidate_total += slice_result["final_candidate_count"]
            all_candidates_initial_total += slice_result["all_candidates_initial"]

    runtime = time.time() - t_start
    if config.SAVE_ONLY_BURST:
        effective_cand_counter_total = n_bursts_total
        effective_n_bursts_total = n_bursts_total
        effective_n_no_bursts_total = 0
    else:
        effective_cand_counter_total = cand_counter_total
        effective_n_bursts_total = n_bursts_total
        effective_n_no_bursts_total = n_no_bursts_total

    result = {
        "n_candidates": effective_cand_counter_total,
        "n_bursts": effective_n_bursts_total,
        "n_no_bursts": effective_n_no_bursts_total,
        "runtime_s": runtime,
        "max_prob": prob_max_total,
        "mean_snr": float(np.mean(snr_list_total)) if snr_list_total else 0.0,
        "status": "SUCCESS_CHUNKED_HIGH_FREQ",
        "processing_stats": {
            "total_processing_time": runtime,
            "strategy_1_count": strategy1_total,
            "strategy_2_initial_count": strategy2_initial_total,
            "strategy_2_validated_count": strategy2_validated_total,
            "final_unique_count": final_candidate_total,
            "all_candidates_initial": all_candidates_initial_total,
            "actual_chunks": actual_chunk_count,
        },
    }

    alerts = monitor_pipeline_health(result, config)
    for alert in alerts:
        level = alert.get("level", "INFO").upper()
        message = alert.get("message", "")
        recommendation = alert.get("recommendation")
        if level == "ERROR":
            logger.error("[HF Pipeline] %s", message)
        elif level == "WARNING":
            logger.warning("[HF Pipeline] %s", message)
        else:
            logger.info("[HF Pipeline] %s", message)
        if recommendation:
            logger.info("[HF Pipeline] Recommendation: %s", recommendation)

    return result


__all__ = ["_process_file_chunked_high_freq", "process_slice_with_multiple_bands_high_freq"]

