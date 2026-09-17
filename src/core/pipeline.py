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
    from ..visualization.mpl_backend import select_headless_backend
    select_headless_backend()
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
    release_dm_cube_buffer,
    trim_valid_window,
    validate_slice_indices,
)
from .contracts import ChunkPlan, DMGrid, ObservationMetadata, PipelineConfigSnapshot
from .file_driver import (
    ChunkLoopState,
    begin_chunk,
    check_file_length,
    compute_overlap_raw,
    export_validation_metrics,
    finalize_file_status,
    finish_chunk,
    log_chunk_arrival_latency,
    optimize_memory as _optimize_memory,
    prepare_chunked_run,
    record_chunk_failure,
    record_oom,
    select_pipeline_path,
    start_arrival_clock,
)
from .pipeline_parameters import calculate_absolute_slice_time, calculate_dm_values, calculate_frequency_downsampled
from ..input.parameter_extractor import extract_parameters_auto
from ..input.streaming_orchestrator import get_streaming_function
from .high_freq_pipeline import _process_file_chunked_high_freq
from ..input.file_finder import find_data_files
from ..log_utils import (
    log_pipeline_file_completion,
    log_pipeline_file_processing,
    log_processing_summary,
    log_streaming_parameters,
)
from ..output.candidate_manager import CandidateWriter, rotate_previous_candidates
from ..output.phase_metrics import PhaseMetricsTracker

              
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

def _error_result(
    status: str,
    error: Exception,
    t_start: float,
    stats: "DetectionStats",
    chunks_processed: int = 0,
    failed_chunks: int = 0,
) -> dict:
    """Per-file result for a run that ended in an error.

    The counts are what the run actually produced and wrote before failing, not
    zeros. Reporting zero while the CSV already held those rows led straight to
    the wrong conclusion -- that the file had no detections -- and to real
    candidates being discarded with it.
    """
    effective_candidates, effective_bursts, effective_no_bursts = stats.effective_counts(
        bool(getattr(config, "SAVE_ONLY_BURST", False))
    )
    return {
        "n_candidates": effective_candidates,
        "n_bursts": effective_bursts,
        "n_no_bursts": effective_no_bursts,
        "runtime_s": time.time() - t_start,
        "max_prob": stats.max_prob,
        "mean_snr": stats.mean_snr(),
        "status": status,
        "error_details": str(error),
        "chunks_processed": chunks_processed,
        "failed_chunks": failed_chunks,
    }


def _trace_info(message: str, *args) -> None:
    try:
        from ..log_utils.logging_config import get_global_logger
        gl = get_global_logger()
        gl.logger.info(message % args if args else message)
    except Exception:
        logger.info(message, *args)

# ``finalize_file_status`` and ``_optimize_memory`` moved to ``core.file_driver``
# so the high-frequency driver can reach them without importing this module,
# which imports it. They stay importable from here: callers and tests use
# ``src.core.pipeline.finalize_file_status``.


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
    collector=None,  # ValidationMetricsCollector
) -> DetectionStats:
    """Process a data block and return aggregated detection statistics."""

    block_wall_start = time.time()
    raw_block_nbytes = int(getattr(block, "nbytes", 0))
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

    obs_meta = ObservationMetadata.from_config(config)
    pipe_snap = PipelineConfigSnapshot.from_config(config)
    dm_grid = DMGrid.from_config(config)
    chunk_plan = ChunkPlan(
        chunk_idx=int(chunk_idx),
        start_sample=int(start_sample),
        end_sample=int(end_sample),
        overlap_left=int(metadata.get("overlap_left", 0)),
        overlap_right=int(metadata.get("overlap_right", 0)),
    )
    logger.debug(
        "Chunk %03d contracts: band=[%.1f–%.1f] MHz duration=%.2fs DM_grid=%d rows",
        chunk_idx,
        obs_meta.freq_low,
        obs_meta.freq_high,
        obs_meta.duration_s,
        dm_grid.size,
    )
    _trace_info(
        "[TRACE] Chunk %03d ChunkPlan valid=%d block=[%d,%d)",
        chunk_idx,
        chunk_plan.valid_length,
        chunk_plan.block_start,
        chunk_plan.block_end,
    )

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

    # PRESTO-style: Build DM-time cube with memory validation
    # For very large chunks, we could use DoubleBufferDedispersion, but for now
    # build_dm_time_cube with immediate trimming provides good memory control
    logger.info(
        f"Chunk %03d: Starting dedispersion (DM range: %.1f-%.1f pc cm⁻³, height=%d, width=%d)",
        chunk_idx, config.DM_min, config.DM_max, height, block.shape[0]
    )
    dm_time_full = build_dm_time_cube(block, height=height, dm_min=config.DM_min, dm_max=config.DM_max, collector=collector)
    logger.info(f"Chunk %03d: Dedispersion complete, cube shape: {dm_time_full.shape}", chunk_idx)
    block, dm_time, valid_start_ds, valid_end_ds = trim_valid_window(
        block, dm_time_full, overlap_left_ds, overlap_right_ds
    )
    
    # Record chunk processing for validation metrics
    if collector is not None:
        collector.record_chunk_processing(
            chunk_idx=chunk_idx,
            overlap_left=overlap_left_ds,
            overlap_right=overlap_right_ds,
            valid_start=valid_start_ds,
            valid_end=valid_end_ds,
            chunk_samples=block.shape[0],
        )
    
    # CRITICAL: Free the full cube immediately after trimming (PRESTO-style)
    # This ensures we never keep more than the trimmed cube in memory
    # We only keep what we need for processing slices
    # Frees the temporary file when the cube was memmap-backed; a no-op
    # otherwise. Without it each large chunk leaves a multi-GB file behind.
    release_dm_cube_buffer(dm_time_full)
    del dm_time_full
    gc.collect()  # gc is imported at module scope; a local import here would
                  # make the name function-local for the whole function.

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
                from ..log_utils.logging_config import get_global_logger

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

        composite_dir, detections_dir, patches_dir, summary_dir = create_chunk_directories(
            save_dir, fits_path, chunk_idx
        )

        # PRESTO-style: Process and write immediately (no accumulation)
        # Candidates are written immediately via append_candidate() in process_slice_with_multiple_bands
        # Plots are saved immediately during processing
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
            dm_values=dm_grid.values,
        )

        # Update stats immediately (PRESTO-style: process → write → update stats)
        chunk_stats.update(cands, bursts, no_bursts, max_prob)

        # CRITICAL: Explicitly free slice arrays immediately after processing (PRESTO-style)
        # This ensures we never accumulate more than necessary
        del slice_cube, waterfall_block

        if j % 10 == 0:
            _optimize_memory(aggressive=False)
        else:
            # plt.close('all') stays per slice: matplotlib keeps every figure
            # alive in its own registry until it is closed, so skipping it leaks.
            #
            # gc.collect() does not. The `del` above releases the slice arrays by
            # refcount, immediately and without a traversal; a full generational
            # collection per slice only buys the cyclic garbage, and the
            # _optimize_memory call every tenth slice already does that. At the
            # scale this pipeline targets -- about 1.9 million slices for a
            # 5.5 TiB file -- the per-slice collection was measured by the audit
            # at roughly 11 hours of pure GC (PERF-04).
            if plt is not None:
                plt.close('all')

    # CRITICAL: Free all chunk-level arrays after processing all slices
    del block, dm_time
    _optimize_memory(aggressive=True)

    try:
        from ..log_utils.logging_config import get_global_logger

        global_logger = get_global_logger()
        chunk_runtime_s = time.time() - block_wall_start
        global_logger.chunk_completed(
            chunk_idx,
            chunk_stats.n_candidates,
            chunk_stats.n_bursts,
            chunk_stats.n_no_bursts,
            runtime_s=chunk_runtime_s,
            sample_count=chunk_samples,
            mean_snr=chunk_stats.mean_snr(),
            throughput_sps=(chunk_samples / max(chunk_runtime_s, 1e-9)),
            data_rate_mib_s=(raw_block_nbytes / max(chunk_runtime_s, 1e-9) / (1024 ** 2)),
            slice_count=len(slices_to_process),
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


#: Per-file error status by exception type, in the order ``_process_file_chunked``
#: tests them in its handler chain. ``_run_high_freq_file`` reports failures under
#: the same names, because the high-frequency driver used to be called from inside
#: that chain and its failures surfaced through it.
_FILE_ERROR_STATUS: tuple[tuple[type[BaseException], str, str], ...] = (
    (MemoryError, "ERROR_MEMORY", "Memory error while processing %s: %s"),
    (FileNotFoundError, "ERROR_FILE_NOT_FOUND", "File not found: %s - %s"),
    (PermissionError, "ERROR_PERMISSION", "Permission error processing %s: %s"),
    (ValueError, "ERROR_CORRUPTED_FILE", "Invalid/corrupted file %s: %s"),
    (Exception, "ERROR_CHUNKED", "Unhandled error processing %s: %s"),
)


def _run_high_freq_file(
    cls_model: torch.nn.Module,
    fits_path: Path,
    save_dir: Path,
    chunk_samples: int,
    reason: str,
) -> dict:
    """Hand *fits_path* to the high-frequency driver and report what it returned.

    This is the whole low-frequency involvement in a high-frequency run. It used
    to be an early ``return`` some 190 lines into ``_process_file_chunked``,
    reached only after that function had built a validation collector, run the
    adaptive memory budget, planned the chunk geometry and created the candidate
    CSV -- all of which ``_process_file_chunked_high_freq`` then built again for
    itself. REF-02 moved the decision above that setup; this function keeps the
    two things the early return was still providing.

    The first is the error mapping: the high-frequency driver re-raises rather
    than returning a result, so its failures were converted to a per-file status
    by the low-frequency handler chain. They still are, by the same table.

    The second is ``CandidateWriter.flush_all()`` on every exit. The
    high-frequency driver flushes on its success path only, so without this a
    failed run would drop up to a buffer's worth of rows that were already
    counted. Both are stop-gaps for a driver that should return its own result;
    that remains open and deliberately untouched here.

    ``chunk_samples`` is the size this file asked for, not a planned one. It used
    to be ``effective_chunk_samples`` -- the output of the low-frequency chunk
    plan -- which the high-frequency driver then planned a second time. Feeding a
    plan's output back in as its input is what let ``MAX_CHUNK_SAMPLES`` reach a
    driver documented as never applying it, and only for one shape of input: a
    file shorter than the requested chunk size but longer than the cap. That is
    now consistently not applied, which is what every comment about it already
    claimed.
    """

    logger.info("Switching to high-frequency pipeline (SNR-based detection)")
    logger.info("Reason: %s", reason)

    t_start = time.time()
    try:
        return _process_file_chunked_high_freq(
            cls_model=cls_model,
            fits_path=fits_path,
            save_dir=save_dir,
            chunk_samples=chunk_samples,
        )
    except Exception as error:
        for error_type, status, message in _FILE_ERROR_STATUS:
            if isinstance(error, error_type):
                logger.error(message, fits_path.name, error)
                # The counts are zero because the high-frequency driver keeps its
                # own and loses them when it raises; that is the open defect, not
                # a claim that nothing was written.
                return _error_result(status, error, t_start, DetectionStats())
        raise
    finally:
        CandidateWriter.flush_all()


def _process_file_chunked(
    det_model: torch.nn.Module,
    cls_model: torch.nn.Module,
    fits_path: Path,
    save_dir: Path,
    chunk_samples: int,
) -> dict:
    """Process a file in streaming chunks using ``stream_fil`` or ``stream_fits``."""


    logger.info("Inspecting file structure: %s", fits_path.name)

    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be greater than zero")

    # REF-02: decide which driver runs this file before either one's setup, not
    # after this one's. Everything below here is low-frequency setup, and the
    # high-frequency driver builds its own equivalent of all of it.
    use_hf, hf_reason = select_pipeline_path()
    if use_hf:
        return _run_high_freq_file(cls_model, fits_path, save_dir, chunk_samples, hf_reason)

    logger.info("Using standard pipeline. %s", hf_reason)

    # ===== VALIDATION METRICS COLLECTOR =====
    from ..output.validation_metrics import ValidationMetricsCollector
    collector = ValidationMetricsCollector(fits_path.name)
    collector.record_data_characteristics()

    plan = prepare_chunked_run(
        fits_path,
        save_dir,
        chunk_samples,
        collector,
        # Check performance limits to prevent system hangs.
        max_chunk_limit=getattr(config, 'MAX_CHUNK_SAMPLES', 1000000),  # Default 1M samples
    )
    chunk_samples = plan.chunk_samples
    total_samples = plan.total_samples
    effective_chunk_samples = plan.effective_chunk_samples
    chunk_count = plan.chunk_count
    csv_file = plan.csv_file
    
    t_start = time.time()
    state = ChunkLoopState()
    file_stats = DetectionStats()

    try:
        check_file_length(total_samples)

        # PRESTO-style: the overlap covers the worst-case dispersion delay, so a
        # burst that straddles a chunk boundary is still whole in one of them.
        overlap_raw = compute_overlap_raw()

                                                                        
        streaming_func, file_type = get_streaming_function(fits_path) 
        logger.info(
            "Detected %s file • using streaming reader %s",
            file_type.upper(),
            streaming_func.__name__,
        )
        
        # Log streaming parameters with the adjusted chunk size (after adaptive budgeting)
        log_streaming_parameters(effective_chunk_samples, overlap_raw, total_samples, effective_chunk_samples, streaming_func, file_type)

        # PRESTO-style: Process each block immediately (read → process → write → free)
        # This ensures we never accumulate multiple chunks in memory

        start_arrival_clock(state)

        # Checkpoint/resume: skip already-completed chunks
        from ..core.checkpoint import (
            save_checkpoint,
            load_checkpoint,
            clear_checkpoint,
            should_skip_chunk,
            compute_run_fingerprint,
        )
        # Ties the checkpoint to this search and this input file, so a resume
        # after a configuration change starts over instead of splicing two
        # different searches into one CSV.
        run_fingerprint = compute_run_fingerprint(fits_path, config)
        resume_after = load_checkpoint(save_dir, fits_path.stem, run_fingerprint)
        if resume_after < 0:
            # Fresh run, not a resume: do not append to a previous run's CSV.
            rotate_previous_candidates(csv_file)

        logger.info(
            "Starting to read chunks from file.%s",
            f" Resuming from chunk {resume_after + 1}." if resume_after >= 0 else ""
        )

        for chunk_idx, (block, metadata) in enumerate(streaming_func(str(fits_path), effective_chunk_samples, overlap_samples=overlap_raw), 1):
            if should_skip_chunk(chunk_idx, resume_after):
                logger.debug("Skipping chunk %d (already completed)", chunk_idx)
                continue
            begin_chunk(state, block, metadata)

            logger.info(
                f"Processing chunk {chunk_idx}/{chunk_count} ({chunk_idx/chunk_count*100:.1f}%) • "
                f"samples {metadata['start_sample']:,}→{metadata['end_sample']:,} • "
                f"shape={block.shape}"
            )

            # Gated on the stream sequence number, which is what this driver has
            # always used; the HF driver gates on chunks actually processed.
            log_chunk_arrival_latency(state, chunk_idx, is_first=chunk_idx <= 1)

            # PRESTO-style: Process immediately, write results immediately, then free
            # Results (candidates, plots) are written during _process_block via append_candidate()
            chunk_succeeded = False
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
                    collector=collector,  # Pass collector for validation metrics
                )
                # Merge stats immediately (results already written to CSV/plots)
                file_stats.merge(block_stats)
                chunk_succeeded = True
            except MemoryError as mem_error:
                record_oom(collector, metadata['chunk_idx'], mem_error)
                raise
            except Exception as chunk_error:
                record_chunk_failure(state, metadata['chunk_idx'], chunk_error)

            # This driver knows the exact chunk count up front, so its ETA is
            # exact; the HF driver has to estimate one.
            finish_chunk(
                state,
                chunk_idx,
                remaining_chunks=chunk_count - chunk_idx,
                report_eta=chunk_idx > 1,
            )

            # Checkpoint ONLY after a chunk that actually completed. A failed
            # chunk must stay un-checkpointed, otherwise a later resume skips it
            # for good and the recovery run reports success without recovering
            # anything. Flush first: the checkpoint claims the rows this chunk
            # produced are durable, so they have to be on disk before it lands.
            if chunk_succeeded:
                CandidateWriter.flush_buffers()
                save_checkpoint(
                    save_dir, fits_path.stem, chunk_idx, chunk_count,
                    fingerprint=run_fingerprint,
                )

            # CRITICAL: Free block immediately after processing (PRESTO-style)
            del block
            _optimize_memory(aggressive=(state.actual_chunk_count % 5 == 0))

                                                        
        log_processing_summary(state.actual_chunk_count, chunk_count, file_stats.n_candidates, file_stats.n_bursts)

        # Flush any buffered CSV rows
        CandidateWriter.flush_all()

        # Clear checkpoint on successful completion
        clear_checkpoint(save_dir, fits_path.stem)

        export_validation_metrics(collector, save_dir, fits_path)

        runtime = time.time() - t_start                      
        logger.info(
            "File completed • chunks=%d • candidates=%d • max_prob=%.2f • runtime=%.1fs",
            state.actual_chunk_count,
            file_stats.n_candidates,
            file_stats.max_prob,
            runtime,
        )

        n_candidates, n_bursts, n_no_bursts = file_stats.effective_counts(config.SAVE_ONLY_BURST)

        successful_chunks = state.actual_chunk_count - state.failed_chunk_count
        status = finalize_file_status("SUCCESS_CHUNKED", state.failed_chunk_count, successful_chunks)
        if state.failed_chunk_count > 0:
            logger.warning(
                "File %s completed with %d/%d chunks failed -> status=%s",
                fits_path.name, state.failed_chunk_count, state.actual_chunk_count, status,
            )

        return {
            "n_candidates": n_candidates,
            "n_bursts": n_bursts,
            "n_no_bursts": n_no_bursts,
            "runtime_s": runtime,
            "max_prob": file_stats.max_prob,
            "mean_snr": file_stats.mean_snr(),
            "status": status,
            "failed_chunks": state.failed_chunk_count,
            "chunks_processed": state.actual_chunk_count,
            "total_chunks": chunk_count,
            "file_size_samples": total_samples,
            "processing_mode": "small_file_optimized" if total_samples <= chunk_samples else "standard_chunking"
        }
        
    except MemoryError as e:
        logger.error("Memory error while processing %s: %s", fits_path.name, e)
        status = "ERROR_MEMORY"
        return _error_result(
            status, e, t_start, file_stats,
            chunks_processed=state.actual_chunk_count,
            failed_chunks=state.failed_chunk_count,
        )
    except FileNotFoundError as e:
        logger.error("File not found: %s - %s", fits_path.name, e)
        status = "ERROR_FILE_NOT_FOUND"
        return _error_result(
            status, e, t_start, file_stats,
            chunks_processed=state.actual_chunk_count,
            failed_chunks=state.failed_chunk_count,
        )
    except PermissionError as e:
        logger.error("Permission error processing %s: %s", fits_path.name, e)
        status = "ERROR_PERMISSION"
        return _error_result(
            status, e, t_start, file_stats,
            chunks_processed=state.actual_chunk_count,
            failed_chunks=state.failed_chunk_count,
        )
    except ValueError as e:
        logger.error("Invalid/corrupted file %s: %s", fits_path.name, e)
        status = "ERROR_CORRUPTED_FILE"
        return _error_result(
            status, e, t_start, file_stats,
            chunks_processed=state.actual_chunk_count,
            failed_chunks=state.failed_chunk_count,
        )
    except Exception as e:
        logger.error("Unhandled error processing %s: %s", fits_path.name, e)
        status = "ERROR_CHUNKED"
        return _error_result(
            status, e, t_start, file_stats,
            chunks_processed=state.actual_chunk_count,
            failed_chunks=state.failed_chunk_count,
        )
    finally:
        # Every exit path -- success, early return, or any of the handlers above --
        # must leave the CSV on disk. Without this, a failure after N candidates
        # were counted drops the last buffered rows while the summary still
        # reports them.
        CandidateWriter.flush_all()


def _prepare_file_parameters(fits_path: Path, manual_chunk_override: int) -> tuple[dict, int]:
    """Extract observation parameters for a single file and resolve its chunk size.

    SPEC-IO-001: parameters must be extracted per file (not once per target), so
    heterogeneous files never inherit another file's TIME_RESO/FREQ/FILE_LENG.

    Returns
    -------
    (extraction_result, chunk_samples)
        ``extraction_result`` is the dict returned by ``extract_parameters_auto``.
        ``chunk_samples`` is the manual override when > 0, otherwise the value
        computed from this file's freshly-extracted parameters.
    """
    from ..preprocessing.slice_len_calculator import get_processing_parameters, validate_processing_parameters
    from ..log_utils.chunking_logging import display_detailed_chunking_info

    extraction_result = extract_parameters_auto(fits_path)
    if not extraction_result.get('success'):
        return extraction_result, 0

    obs_meta = ObservationMetadata.from_config(config)
    logger.debug(
        "Per-file metadata %s: Δt=%.3e s channels=%d samples=%d band=[%.1f–%.1f] MHz",
        fits_path.name,
        obs_meta.time_reso,
        obs_meta.freq_reso,
        obs_meta.file_leng,
        obs_meta.freq_low,
        obs_meta.freq_high,
    )

    if manual_chunk_override and manual_chunk_override > 0:
        return extraction_result, int(manual_chunk_override)

    processing_params = get_processing_parameters()
    if validate_processing_parameters(processing_params):
        try:
            display_detailed_chunking_info(processing_params)
        except Exception:
            pass
        return extraction_result, int(processing_params['chunk_samples'])

    # Fall back to a safe default when the computed parameters are invalid.
    return extraction_result, 2_097_152


def run_pipeline(chunk_samples: int = 0, config_dict: dict | None = None) -> None:
    # Inject configuration if provided
    if config_dict is not None:
        config.inject_config(config_dict)

    from ..log_utils.logging_config import setup_logging, set_global_logger

    # Level and colours come from advanced-config/logging.yaml; they used to be
    # literals here, so raising verbosity in production meant editing code.
    logger = setup_logging(
        level=str(getattr(config, 'LOG_LEVEL', 'INFO')),
        use_colors=bool(getattr(config, 'LOG_COLORS', True)),
    )
    set_global_logger(logger)

    # ===== HARDWARE DETECTION & STARTUP VALIDATION =====
    from ..core.hardware_profile import apply_thread_settings, detect_hardware
    from ..core.system_validator import SystemRequirements

    hw = detect_hardware(str(config.RESULTS_DIR))
    config.inject_config({"_hardware_profile": hw})
    # advanced-config/performance.yaml exposes cpu_threads and it was loaded
    # and then ignored: the override never reached the thread settings.
    apply_thread_settings(hw, user_threads=int(getattr(config, 'CPU_THREADS', 0)))
    logger.logger.info(
        "Hardware: %s, %d cores, %.1f GB RAM (%.1f GB free), GPU: %s",
        hw.platform_system, hw.cpu_cores_physical,
        hw.ram_total_gb, hw.ram_available_gb,
        hw.gpu_name or "none",
    )

    issues = SystemRequirements.validate(hw, config)
    for issue in issues:
        logger.logger.warning("System check: %s", issue)

    pipeline_config = {
        'data_dir': str(config.DATA_DIR),
        'results_dir': str(config.RESULTS_DIR),
        'targets': config.FRB_TARGETS,
        'chunk_samples': chunk_samples,
        'dm_min': config.DM_min,
        'dm_max': config.DM_max,
        'dm_trials': int(calculate_dm_values().size),
        'dm_grid_mode': getattr(config, 'DM_GRID_MODE', 'legacy_uniform'),
        'trial_correction': getattr(config, 'TRIAL_CORRECTION', 'gaussian_extreme'),
        'slice_duration_ms': getattr(config, 'SLICE_DURATION_MS', 0.0),
        'down_time_rate': getattr(config, 'DOWN_TIME_RATE', 1),
        'down_freq_rate': getattr(config, 'DOWN_FREQ_RATE', 1),
        'polarization_mode': getattr(config, 'POLARIZATION_MODE', 'intensity'),
        'save_only_burst': getattr(config, 'SAVE_ONLY_BURST', False),
        'device': str(getattr(config, 'DEVICE', 'cpu')),
        'multi_band': getattr(config, 'USE_MULTI_BAND', False),
        'auto_high_freq': getattr(config, 'AUTO_HIGH_FREQ_PIPELINE', True),
        'hardware_summary': hw.summary(),
    }

    logger.pipeline_start(pipeline_config)

    # Log High-Frequency Pipeline Configuration
    enable_phase2_snr = getattr(config, 'ENABLE_LINEAR_VALIDATION', False)
    enable_intensity_class = getattr(config, 'ENABLE_INTENSITY_CLASSIFICATION', True)
    enable_linear_class = getattr(config, 'ENABLE_LINEAR_CLASSIFICATION', True)
    
    logger.logger.info("=" * 80)
    logger.logger.info("HF PIPELINE CONTROL (collapse_ratio=%.1f):", getattr(config, 'BOWTIE_COLLAPSE_RATIO', 2.0))
    logger.logger.info("  Phase 1 (SNR Detection - I): ALWAYS ENABLED (threshold=%.1f)", config.SNR_THRESH)
    logger.logger.info("  Phase 2 (SNR Validation - L): %s%s", 
                      "ENABLED" if enable_phase2_snr else "DISABLED",
                      f" (threshold={getattr(config, 'SNR_THRESH_LINEAR', config.SNR_THRESH):.1f})" if enable_phase2_snr else "")
    logger.logger.info("  Phase 3a (Classification - I): %s%s",
                      "ENABLED" if enable_intensity_class else "DISABLED",
                      f" (threshold={config.CLASS_PROB:.2f})" if enable_intensity_class else "")
    logger.logger.info("  Phase 3b (Classification - L): %s%s",
                      "ENABLED" if enable_linear_class else "DISABLED",
                      f" (threshold={getattr(config, 'CLASS_PROB_LINEAR', config.CLASS_PROB):.2f})" if enable_linear_class else "")
    logger.logger.info("=" * 80)
    
    # Log output mode
    if config.SAVE_ONLY_BURST:
        logger.logger.info("Output mode: STRICT - Save only if ALL enabled phases classify as BURST")
    else:
        logger.logger.info("Output mode: PERMISSIVE - Save if ANY enabled phase classifies as BURST")

    save_dir = config.RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.logger.info("Loading models...")
    det_model = _load_detection_model()
    cls_model = _load_class_model()
    logger.logger.info("Models loaded successfully")

    summary: dict[str, dict] = {}
    # Dictionary to store phase metrics trackers by filename
    phase_metrics_by_file: dict[str, 'PhaseMetricsTracker'] = {}
    
    # Preserve the original manual override so it does not leak across files/targets.
    manual_chunk_override = int(chunk_samples)

    # Every output path -- candidate CSV, checkpoint, plot directories, summary
    # key -- is derived from the file stem alone, so two inputs with the same
    # basename in different directories write over each other: the CSVs merge
    # (append mode), the checkpoints collide and the plots are overwritten. The
    # result is two observations blended into one dataset with nothing marking
    # the seam. Detect it up front rather than discover it in the data later.
    # The structural fix is to derive output identity from the full path
    # (audit REF-17); this guard is the safe stop-gap.
    seen_stems: dict[str, Path] = {}
    for frb in config.FRB_TARGETS:
        for candidate_path in find_data_files(frb):
            previous = seen_stems.get(candidate_path.stem)
            if previous is not None and previous != candidate_path:
                raise ValueError(
                    "Two input files share the basename "
                    f"'{candidate_path.stem}':\n  {previous}\n  {candidate_path}\n"
                    "They would write to the same CSV, checkpoint and plot "
                    "directory and their results would be merged silently. "
                    "Rename one, or process them into separate results_dir."
                )
            seen_stems[candidate_path.stem] = candidate_path

    for frb in config.FRB_TARGETS:
        logger.logger.info("Searching files for target: %s", frb)
        file_list = find_data_files(frb)
        logger.logger.info("Files found: %s", [f.name for f in file_list])
        if not file_list:
            logger.logger.warning("No files found for %s", frb)
            continue

        for fits_path in file_list:
            try:
                # SPEC-IO-001: extract parameters PER FILE (not once per target).
                logger.logger.info("Extracting parameters from %s", fits_path.name)
                try:
                    extraction_result, file_chunk_samples = _prepare_file_parameters(
                        fits_path, manual_chunk_override
                    )
                except Exception as e:
                    logger.logger.error("Failed to obtain parameters for %s: %s", fits_path.name, e)
                    continue

                if not extraction_result.get('success'):
                    logger.logger.error(
                        "Parameter extraction failed for %s: %s",
                        fits_path.name,
                        ", ".join(extraction_result.get('errors', [])),
                    )
                    continue

                logger.logger.info(
                    "Parameters extracted: %s",
                    ", ".join(extraction_result.get('parameters_extracted', [])),
                )
                if manual_chunk_override and manual_chunk_override > 0:
                    logger.logger.info("Using manual chunk_samples override: %s", f"{file_chunk_samples:,}")

                try:
                    freq_ds = calculate_frequency_downsampled()
                    freq_min_mhz = float(freq_ds.min())
                    freq_max_mhz = float(freq_ds.max())
                except Exception:
                    freq_min_mhz = float(np.min(config.FREQ)) if getattr(config, "FREQ", None) is not None else 0.0
                    freq_max_mhz = float(np.max(config.FREQ)) if getattr(config, "FREQ", None) is not None else 0.0
                file_info = {
                    'samples': config.FILE_LENG,
                    'duration_min': (config.FILE_LENG * config.TIME_RESO) / 60,
                    'channels': config.FREQ_RESO,
                    'freq_min_mhz': freq_min_mhz,
                    'freq_max_mhz': freq_max_mhz,
                    'bandwidth_mhz': max(0.0, freq_max_mhz - freq_min_mhz),
                    'time_reso_ms': float(config.TIME_RESO) * 1000.0,
                    'time_reso_ds_ms': float(config.TIME_RESO) * max(1, int(config.DOWN_TIME_RATE)) * 1000.0,
                    'dm_min': float(config.DM_min),
                    'dm_max': float(config.DM_max),
                    'dm_trials': int(calculate_dm_values().size),
                    'dm_grid_mode': getattr(config, 'DM_GRID_MODE', 'legacy_uniform'),
                }
                logger.file_processing_start(fits_path.name, file_info) 
                
                log_pipeline_file_processing(fits_path.name, fits_path.suffix.lower(), config.FILE_LENG, file_chunk_samples) 
                
                results = _process_file_chunked(det_model, cls_model, fits_path, save_dir, file_chunk_samples)                               
                summary[fits_path.name] = results
                
                # Capture phase metrics tracker if available (from high-freq pipeline)
                if "phase_metrics" in results and results["phase_metrics"] is not None:
                    phase_metrics_by_file[fits_path.name] = results["phase_metrics"]
                                       
                log_pipeline_file_completion(fits_path.name, results)
                
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
                
    logger.pipeline_end(summary)
    
    # Generate execution summary with phase metrics
    if phase_metrics_by_file:
        from ..output.execution_summary import generate_execution_summary
        try:
            generate_execution_summary(phase_metrics_by_file, save_dir, summary)
            logger.logger.info("Execution summary with phase metrics generated successfully")
        except Exception as e:
            logger.logger.warning(f"Failed to generate execution summary: {e}", exc_info=True)

if __name__ == "__main__":
                                                     
    run_pipeline()
