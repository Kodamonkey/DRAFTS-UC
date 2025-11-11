# This module orchestrates the end-to-end FRB processing pipeline.

from __future__ import annotations

                          
import gc
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

                     
import numpy as np

                              
try:
    import torch
except ImportError:
    torch = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

               
from ..config import config
from .detection_engine import process_slice_with_multiple_bands
from .data_flow_manager import (
    build_dm_time_cube,
    create_chunk_directories,
    downsample_chunk,
    get_chunk_processing_parameters,
    plan_slices,
    trim_valid_window,
    validate_slice_indices,
)
from .pipeline_parameters import calculate_absolute_slice_time, calculate_frequency_downsampled
from ..input.parameter_extractor import extract_parameters_auto
from ..input.streaming_orchestrator import get_streaming_function
from .high_freq_pipeline import _process_file_chunked_high_freq
from ..input.file_finder import find_data_files
from ..logging import (
    log_block_processing,
    log_pipeline_file_completion,
    log_pipeline_file_processing,
    log_processing_summary,
    log_streaming_parameters,
)
from ..output.candidate_manager import ensure_csv_header
from ..output.summary_manager import _update_summary_with_results, _write_summary_with_timestamp

              
logger = logging.getLogger(__name__)


@dataclass
class DetectionStats:
    """Accumulate detection metrics for a chunk or file."""

    n_candidates: int = 0
    n_bursts: int = 0
    n_no_bursts: int = 0
    max_prob: float = 0.0
    snr_values: list[float] = field(default_factory=list)

    def update(self, candidates: int, bursts: int, no_bursts: int, prob_max: float) -> None:
        """Update counters with the result of a slice or chunk."""

        self.n_candidates += candidates
        self.n_bursts += bursts
        self.n_no_bursts += no_bursts
        self.max_prob = max(self.max_prob, float(prob_max))

    def merge(self, other: "DetectionStats") -> None:
        """Merge metrics coming from another :class:`DetectionStats` instance."""

        self.update(other.n_candidates, other.n_bursts, other.n_no_bursts, other.max_prob)
        if other.snr_values:
            self.snr_values.extend(other.snr_values)

    def mean_snr(self) -> float:
        return float(np.mean(self.snr_values)) if self.snr_values else 0.0

    def effective_counts(self, save_only_burst: bool) -> tuple[int, int, int]:
        """Return counts respecting the SAVE_ONLY_BURST flag."""

        if save_only_burst:
            return self.n_bursts, self.n_bursts, 0
        return self.n_candidates, self.n_bursts, self.n_no_bursts

def _trace_info(message: str, *args) -> None:
    try:
        from ..logging.logging_config import get_global_logger
        gl = get_global_logger()
        gl.logger.info(message % args if args else message)
    except Exception:
        logger.info(message, *args)

def _optimize_memory(aggressive: bool = False) -> None:
    """Release cached resources to keep the pipeline within memory limits.

    Args:
        aggressive: When ``True`` also clear GPU caches and pause briefly.
    """

    gc.collect()


    if plt is not None:
        plt.close('all')                                          
    
                                         
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        if aggressive:
                                           
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    
                                                     
    if aggressive:
        time.sleep(0.05)                               
    else:
        time.sleep(0.01)                             


def _load_detection_model() -> torch.nn.Module:
    """Load the CenterNet model configured in :mod:`config`."""
    if torch is None:
        raise ImportError("torch is required to load models")

    from ..models.ObjectDet.centernet_model import centernet
    model = centernet(model_name=config.MODEL_NAME).to(config.DEVICE)
    state = torch.load(config.MODEL_PATH, map_location=config.DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

def _load_class_model() -> torch.nn.Module:
    """Load the binary classification model configured in :mod:`config`."""
    if torch is None:
        raise ImportError("torch is required to load models")

    from ..models.BinaryClass.binary_model import BinaryNet
    model = BinaryNet(config.CLASS_MODEL_NAME, num_classes=2).to(config.DEVICE)
    state = torch.load(config.CLASS_MODEL_PATH, map_location=config.DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

def _process_block(
    det_model: torch.nn.Module,
    cls_model: torch.nn.Module,
    block: np.ndarray,
    metadata: dict,
    fits_path: Path,
    save_dir: Path,
    chunk_idx: int,
    csv_file: Path,
) -> DetectionStats:
    """Process a data block and return aggregated detection statistics."""

    chunk_samples = int(metadata.get("actual_chunk_size", block.shape[0]))
    total_samples = int(metadata.get("total_samples", 0)) or chunk_samples
    start_sample = int(metadata.get("start_sample", 0))
    end_sample = int(metadata.get("end_sample", start_sample + chunk_samples))

    chunk_start_time_sec = start_sample * config.TIME_RESO
    chunk_duration_sec = chunk_samples * config.TIME_RESO

    logger.info(
        "Chunk %03d • samples=%s/%s • range=[%s→%s] • time=%.2fs-%.2fs (%.2fs) • progress=%.1f%%",
        chunk_idx,
        f"{chunk_samples:,}",
        f"{total_samples:,}",
        f"{start_sample:,}",
        f"{end_sample:,}",
        chunk_start_time_sec,
        chunk_start_time_sec + chunk_duration_sec,
        chunk_duration_sec,
        (start_sample / max(total_samples, 1)) * 100,
    )

    block, dt_ds = downsample_chunk(block)

    _trace_info(
        "[TRACE] Chunk %03d: tsamp=%.9fs DOWN_TIME_RATE=%dx Δt=%.9fs start_sample_raw=%d end_sample_raw=%d",
        chunk_idx,
        config.TIME_RESO,
        int(config.DOWN_TIME_RATE),
        dt_ds,
        metadata.get("start_sample", -1),
        metadata.get("end_sample", -1),
    )

    chunk_params = get_chunk_processing_parameters(metadata)
    freq_down = chunk_params['freq_down']
    height = chunk_params['height']
    slice_len = chunk_params['slice_len']
    time_slice = chunk_params['time_slice']
    overlap_left_ds = chunk_params['overlap_left_ds']
    overlap_right_ds = chunk_params['overlap_right_ds']

    logger.debug(
        "Overlap raw→ds • left=%d→%d (rate=%d) • right=%d→%d",
        int(metadata.get("overlap_left", 0)),
        overlap_left_ds,
        int(config.DOWN_TIME_RATE),
        int(metadata.get("overlap_right", 0)),
        overlap_right_ds,
    )

    dm_time_full = build_dm_time_cube(block, height=height, dm_min=config.DM_min, dm_max=config.DM_max)
    block, dm_time, valid_start_ds, valid_end_ds = trim_valid_window(
        block, dm_time_full, overlap_left_ds, overlap_right_ds
    )

    _trace_info(
        "[TRACE] Chunk %03d: valid_start_ds=%d valid_end_ds=%d (N_valid=%d)",
        chunk_idx,
        valid_start_ds,
        valid_end_ds,
        (valid_end_ds - valid_start_ds),
    )

    band_configs = config.get_band_configs()
    chunk_stats = DetectionStats()
    snr_list = chunk_stats.snr_values
    slices_to_process = plan_slices(block, slice_len, chunk_idx)

    logger.info(
        "Chunk %03d • planned slices=%d (slice_len=%d)",
        chunk_idx,
        len(slices_to_process),
        slice_len,
    )

    for j, start_idx, end_idx in slices_to_process:
        if j % 10 == 0 or j == 0:
            try:
                from ..logging.logging_config import get_global_logger

                global_logger = get_global_logger()
                global_logger.slice_progress(j, time_slice, chunk_idx)
            except ImportError:
                pass

        es_valido, start_idx_ajustado, end_idx_ajustado, razon = validate_slice_indices(
            start_idx, end_idx, block.shape[0], slice_len, j, chunk_idx
        )

        if not es_valido:
            logger.warning("Skipping slice %d (chunk %d): %s", j, chunk_idx, razon)
            continue

        start_idx, end_idx = start_idx_ajustado, end_idx_ajustado

        dt_ds_local = config.TIME_RESO * config.DOWN_TIME_RATE
        slice_abs_start_preview = chunk_start_time_sec + (start_idx * dt_ds_local)
        slice_info = {
            'slice_idx': j,
            'slice_len': slice_len,
            'start_idx': start_idx,
            'end_idx_calculado': end_idx,
            'block_shape': block.shape[0],
            'chunk_idx': chunk_idx,
            'tiempo_absoluto_inicio': slice_abs_start_preview,
            'duracion_slice_esperada_ms': slice_len * config.TIME_RESO * config.DOWN_TIME_RATE * 1000,
        }

        slice_cube = dm_time[:, :, start_idx:end_idx]
        waterfall_block = block[start_idx:end_idx]

        slice_tiempo_real_ms = (end_idx - start_idx) * config.TIME_RESO * config.DOWN_TIME_RATE * 1000
        logger.debug(
            "Slice %03d (chunk %03d) • samples=%d • abs=%.3fs • duration=%.1f ms • cube=%s • waterfall=%s",
            j,
            chunk_idx,
            end_idx - start_idx,
            slice_info['tiempo_absoluto_inicio'],
            slice_tiempo_real_ms,
            slice_cube.shape,
            waterfall_block.shape,
        )

        if slice_cube.size == 0 or waterfall_block.size == 0:
            logger.warning(
                "Skipping slice %d (chunk %d) because the data window is empty: cube=%d waterfall=%d",
                j,
                chunk_idx,
                slice_cube.size,
                waterfall_block.size,
            )
            continue

        slice_start_time_sec = calculate_absolute_slice_time(
            chunk_start_time_sec, start_idx, dt_ds
        )

        _trace_info(
            "[TRACE] Slice %03d (chunk %03d): start_idx=%d end_idx=%d N=%d | abs_start=%.9fs abs_end=%.9fs Δt=%.9fs",
            j,
            chunk_idx,
            start_idx,
            end_idx,
            (end_idx - start_idx),
            slice_start_time_sec,
            slice_start_time_sec + (end_idx - start_idx) * dt_ds,
            dt_ds,
        )

        composite_dir, detections_dir, patches_dir = create_chunk_directories(
            save_dir, fits_path, chunk_idx
        )

        cands, bursts, no_bursts, max_prob = process_slice_with_multiple_bands(
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
            config.TIME_RESO * config.DOWN_TIME_RATE,
            band_configs,
            snr_list,
            config,
            absolute_start_time=slice_start_time_sec,
            composite_dir=composite_dir,
            detections_dir=detections_dir,
            patches_dir=patches_dir,
            chunk_idx=chunk_idx,
            force_plots=config.FORCE_PLOTS,
            slice_start_idx=start_idx,
            slice_end_idx=end_idx,
        )

        chunk_stats.update(cands, bursts, no_bursts, max_prob)

        if j % 10 == 0:
            _optimize_memory(aggressive=False)
        else:
            if plt is not None:
                plt.close('all')
            gc.collect()

    try:
        from ..logging.logging_config import get_global_logger

        global_logger = get_global_logger()
        global_logger.chunk_completed(
            chunk_idx, chunk_stats.n_candidates, chunk_stats.n_bursts, chunk_stats.n_no_bursts
        )
    except ImportError:
        pass

    if not config.SAVE_ONLY_BURST and chunk_stats.n_bursts > 0:
        file_folder_name = fits_path.stem
        chunk_folder_name = f"chunk{chunk_idx:03d}"
        try:
            chunks_with_frbs_dir = save_dir / "Composite" / file_folder_name / "ChunksWithFRBs"
            chunks_with_frbs_dir.mkdir(parents=True, exist_ok=True)

            chunk_dir = save_dir / "Composite" / file_folder_name / chunk_folder_name
            if chunk_dir.exists():
                png_files = list(chunk_dir.glob("*.png"))
                if png_files:
                    destination_dir = chunks_with_frbs_dir / chunk_folder_name
                    if destination_dir.exists():
                        shutil.rmtree(destination_dir)
                    shutil.move(str(chunk_dir), str(destination_dir))
                    logger.info(
                        "Chunk %03d moved to ChunksWithFRBs (contains %d burst candidates)",
                        chunk_idx,
                        chunk_stats.n_bursts,
                    )
                else:
                    logger.warning(
                        "Chunk %03d has %d burst candidates but no plots were produced, leaving in place",
                        chunk_idx,
                        chunk_stats.n_bursts,
                    )
            else:
                logger.warning("Chunk directory %s is missing; cannot move chunk %03d", chunk_dir, chunk_idx)
        except Exception as e:
            logger.error("Failed to move chunk %03d to ChunksWithFRBs: %s", chunk_idx, e)
    elif config.SAVE_ONLY_BURST and chunk_stats.n_bursts > 0:
        logger.info(
            "Chunk %03d contains %d burst candidates (SAVE_ONLY_BURST=True, no reorganisation)",
            chunk_idx,
            chunk_stats.n_bursts,
        )

    return chunk_stats


def _process_file_chunked(
    det_model: torch.nn.Module,
    cls_model: torch.nn.Module,
    fits_path: Path,
    save_dir: Path,
    chunk_samples: int,
) -> dict:
    """Process a file in streaming chunks using ``stream_fil`` or ``stream_fits``."""
    
                                                          
    logger.info("Inspecting file structure: %s", fits_path.name)
    
                                                                                       
    total_samples = config.FILE_LENG                              
    
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be greater than zero")


    if total_samples <= chunk_samples:
        logger.info(
            "Small file detected (%s samples); running in a single optimised chunk",
            f"{total_samples:,}",
        )
                                                                    
        effective_chunk_samples = total_samples
        chunk_count = 1
        logger.info(
            "Using single chunk optimisation • chunk_samples=%s (entire file)",
            f"{effective_chunk_samples:,}",
        )
    else:
        effective_chunk_samples = chunk_samples
        chunk_count = (total_samples + chunk_samples - 1) // chunk_samples
        logger.info("Standard chunking • estimated chunks=%d", chunk_count)

    total_duration_sec = total_samples * config.TIME_RESO
    chunk_duration_sec = effective_chunk_samples * config.TIME_RESO

    logger.info(
        "File summary • chunks=%d • samples=%s • duration=%.2fs (%.1f min) • chunk_size=%s (%.2fs)",
        chunk_count,
        f"{total_samples:,}",
        total_duration_sec,
        total_duration_sec / 60,
        f"{effective_chunk_samples:,}",
        chunk_duration_sec,
    )
    logger.info("Starting streaming processing...")
    
    csv_file = save_dir / f"{fits_path.stem}.candidates.csv"                       
    ensure_csv_header(csv_file)                                               
    
    t_start = time.time()                                     
    actual_chunk_count = 0                                
    file_stats = DetectionStats()
    
    try:
        if total_samples <= 0:
            raise ValueError(f"Invalid file length: {total_samples} samples")
        if total_samples > 1_000_000_000:                      
            logger.warning(
                "Large file detected (%s samples); processing may take longer",
                f"{total_samples:,}",
            )
        
                                                               
        try:
            freq_ds = calculate_frequency_downsampled()
        except ValueError as exc:
            logger.warning(
                "Failed to compute frequency downsampling (%s); using original axis.",
                exc,
            )
            if config.FREQ is None or len(config.FREQ) == 0:
                raise
            freq_ds = config.FREQ
        nu_min = float(freq_ds.min())
        nu_max = float(freq_ds.max())
        dt_max_sec = 4.1488e3 * config.DM_max * (nu_min**-2 - nu_max**-2)

        if config.TIME_RESO <= 0:
            logger.warning(
                "Invalid TIME_RESO (%s); using default overlap window",
                config.TIME_RESO,
            )
            overlap_raw = 1024
        else:
            overlap_raw = max(0, int(np.ceil(dt_max_sec / config.TIME_RESO)))

                                                                        
        streaming_func, file_type = get_streaming_function(fits_path) 
        logger.info(
            "Detected %s file • using streaming reader %s",
            file_type.upper(),
            streaming_func.__name__,
        )
        
        log_streaming_parameters(effective_chunk_samples, overlap_raw, total_samples, chunk_samples, streaming_func, file_type)
        
        # Switch to the high-frequency pipeline when the configuration allows it
        try:
            # FIX: Use calculate_frequency_downsampled to avoid errors with non-divisible channels
            freq_ds_local = calculate_frequency_downsampled()
            center_mhz = float(np.median(freq_ds_local))
            exceeds_threshold = center_mhz >= float(getattr(config, 'HIGH_FREQ_THRESHOLD_MHZ', 8000.0))
        except Exception:
            exceeds_threshold = False

        auto_high_freq_enabled = bool(getattr(config, 'AUTO_HIGH_FREQ_PIPELINE', True))

        if auto_high_freq_enabled and exceeds_threshold:
            logger.info(
                "Switching to high-frequency pipeline (SNR-based detection)")
            logger.info(
                "Reason: centre frequency %.1f MHz ≥ threshold %.1f MHz",
                center_mhz,
                float(getattr(config, 'HIGH_FREQ_THRESHOLD_MHZ', 8000.0)),
            )
            return _process_file_chunked_high_freq(
                cls_model=cls_model,
                fits_path=fits_path,
                save_dir=save_dir,
                chunk_samples=effective_chunk_samples,
                streaming_func=streaming_func,
            )

        # Process each block with overlap
        for block, metadata in streaming_func(str(fits_path), effective_chunk_samples, overlap_samples=overlap_raw):
            actual_chunk_count += 1                                               
            
            log_block_processing(actual_chunk_count, block.shape, str(block.dtype), metadata)
            
            logger.info(
                "Processing chunk %03d • samples %s→%s",
                metadata['chunk_idx'],
                f"{metadata['start_sample']:,}",
                f"{metadata['end_sample']:,}",
            )
            
                                                           
            try:
                block_stats = _process_block(
                    det_model,
                    cls_model,
                    block,
                    metadata,
                    fits_path,
                    save_dir,
                    metadata['chunk_idx'],
                    csv_file,
                )
                file_stats.merge(block_stats)
            except Exception as chunk_error:
                logger.exception(f"Error processing chunk {metadata['chunk_idx']:03d}: {chunk_error}")

                                          
            del block                             
            _optimize_memory(aggressive=(actual_chunk_count % 5 == 0))                                   

                                                        
        log_processing_summary(actual_chunk_count, chunk_count, file_stats.n_candidates, file_stats.n_bursts)

        runtime = time.time() - t_start                      
        logger.info(
            "File completed • chunks=%d • candidates=%d • max_prob=%.2f • runtime=%.1fs",
            actual_chunk_count,
            file_stats.n_candidates,
            file_stats.max_prob,
            runtime,
        )

        n_candidates, n_bursts, n_no_bursts = file_stats.effective_counts(config.SAVE_ONLY_BURST)

        return {
            "n_candidates": n_candidates,
            "n_bursts": n_bursts,
            "n_no_bursts": n_no_bursts,
            "runtime_s": runtime,
            "max_prob": file_stats.max_prob,
            "mean_snr": file_stats.mean_snr,
            "status": "SUCCESS_CHUNKED",
            "chunks_processed": actual_chunk_count,
            "total_chunks": chunk_count,
            "file_size_samples": total_samples,
            "processing_mode": "small_file_optimized" if total_samples <= chunk_samples else "standard_chunking"
        }
        
    except Exception as e:
                                                         
        error_msg = str(e).lower()
        if "corrupted" in error_msg or "invalid" in error_msg or "corrupt" in error_msg:
            status = "ERROR_CORRUPTED_FILE"
            logger.error("Corrupted file detected: %s - %s", fits_path.name, e)
        elif "memory" in error_msg or "out of memory" in error_msg or "oom" in error_msg:
            status = "ERROR_MEMORY"
            logger.error("Memory error while processing %s: %s", fits_path.name, e)
        elif "file not found" in error_msg or "no such file" in error_msg:
            status = "ERROR_FILE_NOT_FOUND"
            logger.error("File not found: %s - %s", fits_path.name, e)
        elif "permission" in error_msg or "access denied" in error_msg:
            status = "ERROR_PERMISSION"
            logger.error("Permission error processing %s: %s", fits_path.name, e)
        else:
            status = "ERROR_CHUNKED"
            logger.error("Unhandled error processing %s: %s", fits_path.name, e)
        
        return {
            "n_candidates": 0,
            "n_bursts": 0,
            "n_no_bursts": 0,
            "runtime_s": time.time() - t_start,
            "max_prob": 0.0,
            "mean_snr": 0.0,
            "status": status,
            "error_details": str(e)
        }

def run_pipeline(chunk_samples: int = 0, config_dict: dict | None = None) -> None:
    # Inject configuration if provided
    if config_dict is not None:
        config.inject_config(config_dict)
    
    from ..logging.logging_config import setup_logging, set_global_logger
    
    logger = setup_logging(level="INFO", use_colors=True)                     
    set_global_logger(logger)                              
    
                                
    pipeline_config = {
        'data_dir': str(config.DATA_DIR),                      
        'results_dir': str(config.RESULTS_DIR),                           
        'targets': config.FRB_TARGETS,                     
        'chunk_samples': chunk_samples                                                   
    }
    
    logger.pipeline_start(pipeline_config) 

    # Log candidate filtering mode
    if config.SAVE_ONLY_BURST:
        logger.logger.info("Output mode: BURST ONLY - Non-burst detections will be discarded")
        logger.logger.info("  → Only candidates classified as BURST by ResNet will be saved")
    else:
        logger.logger.info("Output mode: ALL DETECTIONS - Saving all CenterNet detections")
        logger.logger.info("  → Both BURST and NON-BURST candidates will be saved to CSV")

    save_dir = config.RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.logger.info("Loading models...")
    det_model = _load_detection_model()
    cls_model = _load_class_model()
    logger.logger.info("Models loaded successfully")

    summary: dict[str, dict] = {}
    for frb in config.FRB_TARGETS:                                 
        logger.logger.info("Searching files for target: %s", frb)
        file_list = find_data_files(frb)
        logger.logger.info("Files found: %s", [f.name for f in file_list])
        if not file_list:
            logger.logger.warning("No files found for %s", frb)
            continue
        try:

            first_file = file_list[0]
            logger.logger.info("Extracting parameters from %s", first_file.name)

            extraction_result = extract_parameters_auto(first_file)
            if extraction_result['success']:
                logger.logger.info(
                    "Parameters extracted: %s",
                    ", ".join(extraction_result['parameters_extracted']),
                )
            else:
                logger.logger.error(
                    "Parameter extraction failed: %s",
                    ", ".join(extraction_result['errors']),
                )
                continue
            
                                                                  
            from ..preprocessing.slice_len_calculator import get_processing_parameters, validate_processing_parameters
            from ..logging.chunking_logging import display_detailed_chunking_info
            
            if chunk_samples == 0:
                processing_params = get_processing_parameters()
                if validate_processing_parameters(processing_params):
                    chunk_samples = processing_params['chunk_samples']

                    display_detailed_chunking_info(processing_params)
                else:
                    logger.logger.error("Calculated processing parameters are invalid; falling back to defaults")
                    chunk_samples = 2_097_152
            else:
                logger.logger.info("Using manual chunk_samples override: %s", f"{chunk_samples:,}")

        except Exception as e:
            logger.logger.error("Failed to obtain parameters: %s", e)
            continue
            
        for fits_path in file_list:                                 
            try:
                                                      
                file_info = {
                    'samples': config.FILE_LENG,
                    'duration_min': (config.FILE_LENG * config.TIME_RESO) / 60,
                    'channels': config.FREQ_RESO
                }
                logger.file_processing_start(fits_path.name, file_info) 
                
                log_pipeline_file_processing(fits_path.name, fits_path.suffix.lower(), config.FILE_LENG, chunk_samples) 
                
                results = _process_file_chunked(det_model, cls_model, fits_path, save_dir, chunk_samples)                               
                summary[fits_path.name] = results                                       
                
                log_pipeline_file_completion(fits_path.name, results)
                
                                                                           
                _update_summary_with_results(save_dir, fits_path.stem, {
                    "n_candidates": results.get("n_candidates", 0),
                    "n_bursts": results.get("n_bursts", 0),
                    "n_no_bursts": results.get("n_no_bursts", 0),
                    "processing_time": results.get("runtime_s", 0.0),
                    "max_detection_prob": results.get("max_prob", 0.0),
                    "mean_snr": results.get("mean_snr", 0.0),
                    "status": "completed"
                })
                
                logger.file_processing_end(fits_path.name, results)
            except Exception as e:
                logger.logger.error("Error processing %s: %s", fits_path.name, e)
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

    logger.logger.info("Writing final summary...")
    _write_summary_with_timestamp(summary, save_dir)
    logger.pipeline_end(summary)

if __name__ == "__main__":
                                                     
    run_pipeline()
