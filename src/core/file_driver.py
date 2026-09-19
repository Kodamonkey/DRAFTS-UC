# The orchestration both per-file drivers share, written once.
#
# ``_process_file_chunked`` (low frequency) and
# ``_process_file_chunked_high_freq`` were the same file driver written twice:
# the audit measured about half of their normalised lines byte-identical. That
# duplication had already produced real defects, each fixed in one copy long
# before the other -- the missing ``CandidateWriter.flush_all()`` that silently
# dropped up to 49 candidates per file (P0-2), checkpoint/resume, and the
# flush-before-checkpoint discipline. One divergence is still open:
# ``config.MAX_CHUNK_SAMPLES`` is applied by the LF driver and not by the HF
# one. That is preserved here as the ``max_chunk_limit`` parameter rather than
# closed, because closing it would change the untested path.
#
# This module holds the shared steps so the two paths can no longer drift. Every
# place where the two copies genuinely differ is a parameter here, not a choice:
# the low-frequency path has a golden-CSV regression test and the
# high-frequency path has none, so a "cleanup" that quietly picked one
# behaviour would be invisible exactly where it matters most.
#
# It is also where ``finalize_file_status`` and ``_optimize_memory`` now live.
# ``high_freq_pipeline`` used to reach up into ``core.pipeline`` for both with
# late imports while ``core.pipeline`` imported ``high_freq_pipeline`` at module
# scope -- a genuine cycle hidden behind function-local imports. Nothing here
# imports either driver, so the cycle is gone.

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from ..domain.physics import K_DM_MS
from ..log_utils import log_block_processing
from ..output.candidate_manager import ensure_csv_header
from .contracts import PipelineConfigSnapshot
from .pipeline_parameters import calculate_frequency_downsampled, should_use_hf_pipeline

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# what a driver reports about a file
# --------------------------------------------------------------------------- #
# These three lived in ``pipeline.py``. They are here because BOTH drivers need
# them and ``pipeline`` imports ``high_freq_pipeline`` at module level, so the
# high-frequency driver could not reach them without a cycle -- which is the
# mechanical reason it re-raised instead of returning a result, and the reason
# its caller rebuilt that result from an empty ``DetectionStats`` and reported
# zero candidates for a run that had already written rows to disk.
#
# ``pipeline`` re-imports all three under their old private names, so
# ``from src.core.pipeline import DetectionStats, _error_result`` still works.


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


def error_result(
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


#: Per-file error status by exception type, in the order the drivers test them
#: in their handler chains. Both drivers report failures under the same names.
FILE_ERROR_STATUS: tuple[tuple[type[BaseException], str, str], ...] = (
    (MemoryError, "ERROR_MEMORY", "Memory error while processing %s: %s"),
    (FileNotFoundError, "ERROR_FILE_NOT_FOUND", "File not found: %s - %s"),
    (PermissionError, "ERROR_PERMISSION", "Permission error processing %s: %s"),
    (ValueError, "ERROR_CORRUPTED_FILE", "Invalid/corrupted file %s: %s"),
    (Exception, "ERROR_CHUNKED", "Unhandled error processing %s: %s"),
)


def status_for_error(error: BaseException) -> tuple[str, str]:
    """``(status, log message)`` for *error*, by the table above."""
    for error_type, status, message in FILE_ERROR_STATUS:
        if isinstance(error, error_type):
            return status, message
    return "ERROR_CHUNKED", "Unhandled error processing %s: %s"


# --------------------------------------------------------------------------- #
# which driver runs this file
# --------------------------------------------------------------------------- #

def select_pipeline_path() -> tuple[bool, str]:
    """Decide, once per file, whether this run takes the high-frequency path.

    Returns ``(use_hf, reason)``; *reason* is the sentence the caller logs.

    This decision used to be taken about 190 lines into ``_process_file_chunked``
    -- after the validation collector, the adaptive memory budget, the chunk plan
    and the candidate CSV had all been built -- and the high-frequency driver then
    built every one of them again for itself. Taken here, before any of that, the
    driver that does not run does no setup at all (audit REF-02).

    Two behaviours are preserved deliberately. Any failure computing the bow-tie
    criterion falls back to the low-frequency pipeline, because that is the path
    with a golden-CSV regression test behind it. And ``AUTO_HIGH_FREQ_PIPELINE``
    is folded into the boolean rather than left for the caller to re-test, so
    there is one answer and not two conditions that have to be kept in step.
    """

    try:
        freq_ds = calculate_frequency_downsampled()
        snapshot = PipelineConfigSnapshot.from_config(config)
        use_hf, reason = should_use_hf_pipeline(
            freq_low_mhz=float(np.min(freq_ds)),
            freq_high_mhz=float(np.max(freq_ds)),
            dm_max=float(snapshot.dm_max),
            time_reso_s=float(config.TIME_RESO),
            down_time_rate=int(config.DOWN_TIME_RATE),
            collapse_ratio=float(snapshot.bowtie_collapse_ratio),
        )
    except Exception:
        return False, "error computing bow-tie criterion — falling back to standard pipeline"

    if use_hf and not bool(getattr(config, "AUTO_HIGH_FREQ_PIPELINE", True)):
        return False, f"automatic high-frequency selection is disabled ({reason})"
    return use_hf, reason


# --------------------------------------------------------------------------- #
# helpers that used to live in core.pipeline
# --------------------------------------------------------------------------- #

def finalize_file_status(base_status: str, failed_chunks: int, processed_chunks: int) -> str:
    """Resolve the per-file status accounting for swallowed chunk errors.

    SPEC-IO-002: a file with failed chunks must NOT report success silently.
    - failed > 0 and some chunks processed -> ``<base>_PARTIAL``
    - failed > 0 and nothing processed     -> ``ERROR_ALL_CHUNKS_FAILED``
    - failed == 0                          -> ``base_status``
    """
    if failed_chunks <= 0:
        return base_status
    if processed_chunks <= 0:
        return "ERROR_ALL_CHUNKS_FAILED"
    return f"{base_status}_PARTIAL"


def optimize_memory(aggressive: bool = False) -> None:
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

    # No sleep — gc.collect() and torch.cuda.empty_cache() are synchronous.


# --------------------------------------------------------------------------- #
# per-run state
# --------------------------------------------------------------------------- #

@dataclass
class ChunkLoopState:
    """Bookkeeping the chunk loop carries across iterations.

    It is a mutable object rather than a pile of locals because the callers keep
    their own ``try``/``except`` around the loop and their error returns have to
    report the counts the run actually reached (see ``_error_result``).
    """

    actual_chunk_count: int = 0
    failed_chunk_count: int = 0
    chunk_processing_times: list[float] = field(default_factory=list)
    chunk_start_time: float = 0.0
    last_chunk_arrival_time: float = 0.0


@dataclass
class ChunkedRunPlan:
    """What the shared setup resolved before the first byte was read."""

    total_samples: int
    chunk_samples: int
    effective_chunk_samples: int
    chunk_count: int
    csv_file: Path


# --------------------------------------------------------------------------- #
# setup, before the stream is opened
# --------------------------------------------------------------------------- #

def resolve_memory_safe_chunk_size(chunk_samples: int, collector: Any, log_prefix: str = "") -> int:
    """Clamp *chunk_samples* between the physical minimum and what RAM allows.

    ``log_prefix`` is the only difference between the two callers; the HF driver
    tags these lines with ``"[HF Pipeline] "``.
    """

    # ===== ADAPTIVE MEMORY BUDGETING: Calculate memory-safe chunk size =====
    # This ensures we never exceed available RAM, even with large DM ranges
    from ..preprocessing.slice_len_calculator import calculate_memory_safe_chunk_size

    try:
        safe_chunk_samples, budget_diagnostics = calculate_memory_safe_chunk_size()

        # Record budget diagnostics
        collector.record_memory_budget(budget_diagnostics)
        collector.record_dm_cube(budget_diagnostics)
        collector.record_chunk_calculation(budget_diagnostics)

        # Calculate physical lower bound (minimum samples required for overlap/decimation)
        # budget_diagnostics['required_min_size'] is in DECIMATED domain
        min_required_raw = budget_diagnostics.get('required_min_size', 0) * max(1, config.DOWN_TIME_RATE)

        # Logic to determine final chunk_samples:
        # 1. Upper Bound: Must not exceed available RAM (safe_chunk_samples)
        # 2. Lower Bound: Must meet physical constraints (min_required_raw)

        if chunk_samples < min_required_raw:
            logger.warning(
                f"{log_prefix}Requested chunk size ({chunk_samples:,}) is too small for physical constraints "
                f"(overlap + slice_len requires {min_required_raw:,} raw samples). "
                f"Upgrading to memory-safe calculated size: {safe_chunk_samples:,}."
            )
            chunk_samples = safe_chunk_samples

        elif chunk_samples > safe_chunk_samples:
            logger.info(
                f"{log_prefix}Adaptive budgeting: Reducing chunk size from {chunk_samples:,} to {safe_chunk_samples:,} samples "
                f"to fit in available memory ({budget_diagnostics['usable_bytes_gb']:.2f} GB usable). "
                f"Scenario: {budget_diagnostics['scenario']}."
            )
            if budget_diagnostics['will_use_dm_chunking']:
                logger.info(
                    f"Expected DM-time cube size: {budget_diagnostics['expected_cube_gb']:.2f} GB. "
                    f"DM chunking will activate automatically."
                )
            chunk_samples = safe_chunk_samples
        else:
            logger.debug(
                f"{log_prefix}Requested chunk size ({chunk_samples:,}) is within safe limits "
                f"(min={min_required_raw:,}, max={safe_chunk_samples:,}). Proceeding with requested size."
            )
    except Exception as e:
        logger.warning(
            f"{log_prefix}Failed to calculate memory-safe chunk size: {e}. "
            f"Using requested chunk_samples={chunk_samples:,} (may cause OOM with large DM ranges)."
        )

    return chunk_samples


def plan_chunking(
    total_samples: int,
    chunk_samples: int,
    max_chunk_limit: int | None = None,
) -> tuple[int, int]:
    """Return ``(effective_chunk_samples, chunk_count)`` for this file.

    ``max_chunk_limit`` is ``config.MAX_CHUNK_SAMPLES`` for the low-frequency
    driver and ``None`` for the high-frequency one, which has never had the
    limit. Passing ``None`` keeps the two-branch behaviour HF has today rather
    than quietly giving it a cap nothing has tested.
    """

    if total_samples <= chunk_samples and (max_chunk_limit is None or total_samples <= max_chunk_limit):
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
    elif max_chunk_limit is not None and total_samples <= chunk_samples:
        logger.info(
            "File size (%s samples) exceeds maximum chunk limit (%s samples); using chunked processing",
            f"{total_samples:,}",
            f"{max_chunk_limit:,}",
        )
        effective_chunk_samples = min(chunk_samples, max_chunk_limit)
        chunk_count = (total_samples + effective_chunk_samples - 1) // effective_chunk_samples
    else:
        effective_chunk_samples = chunk_samples
        chunk_count = (total_samples + chunk_samples - 1) // chunk_samples
        logger.info("Standard chunking • estimated chunks=%d", chunk_count)

    return effective_chunk_samples, chunk_count


def log_file_summary(chunk_count: int, total_samples: int, effective_chunk_samples: int) -> None:
    """Announce the chunk geometry the run will use."""

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


def prepare_candidate_csv(save_dir: Path, fits_path: Path) -> Path:
    """Create ``Summary/<file>/`` and the candidate CSV with its header."""

    # Create Summary directory structure: Summary/(file_name)/
    summary_dir = save_dir / "Summary" / fits_path.stem
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_file = summary_dir / f"{fits_path.stem}.candidates.csv"
    ensure_csv_header(csv_file)
    return csv_file


def prepare_chunked_run(
    fits_path: Path,
    save_dir: Path,
    chunk_samples: int,
    collector: Any,
    *,
    log_prefix: str = "",
    max_chunk_limit: int | None = None,
) -> ChunkedRunPlan:
    """The whole pre-stream setup both drivers run, in the order both ran it.

    Memory budget, chunk geometry, the file-summary log line and the candidate
    CSV. Deliberately outside the callers' ``try``: that is where both copies
    had it, so a failure here still propagates to ``run_pipeline`` rather than
    being reported as a per-file error status.
    """

    total_samples = config.FILE_LENG

    chunk_samples = resolve_memory_safe_chunk_size(chunk_samples, collector, log_prefix=log_prefix)
    effective_chunk_samples, chunk_count = plan_chunking(
        total_samples, chunk_samples, max_chunk_limit=max_chunk_limit
    )
    log_file_summary(chunk_count, total_samples, effective_chunk_samples)
    csv_file = prepare_candidate_csv(save_dir, fits_path)

    return ChunkedRunPlan(
        total_samples=total_samples,
        chunk_samples=chunk_samples,
        effective_chunk_samples=effective_chunk_samples,
        chunk_count=chunk_count,
        csv_file=csv_file,
    )


# --------------------------------------------------------------------------- #
# pre-stream checks, inside the callers' try
# --------------------------------------------------------------------------- #

def check_file_length(total_samples: int) -> None:
    """Reject an empty/unreadable file and warn about a very large one."""

    if total_samples <= 0:
        raise ValueError(f"Invalid file length: {total_samples} samples")
    if total_samples > 1_000_000_000:
        logger.warning(
            "Large file detected (%s samples); processing may take longer",
            f"{total_samples:,}",
        )


def compute_overlap_raw() -> int:
    """Raw samples of overlap needed to cover the worst-case dispersion delay."""

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
    dt_max_sec = K_DM_MS * config.DM_max * (nu_min**-2 - nu_max**-2)

    if config.TIME_RESO <= 0:
        logger.warning(
            "Invalid TIME_RESO (%s); using default overlap window",
            config.TIME_RESO,
        )
        return 1024
    return max(0, int(np.ceil(dt_max_sec / config.TIME_RESO)))


def start_arrival_clock(state: ChunkLoopState) -> None:
    """Start the clock the per-chunk arrival latency is measured against.

    The checkpoint/resume block that follows it in both drivers is deliberately
    NOT shared: ``tests/test_p2_reliability.py`` asserts on the source text of
    ``compute_run_fingerprint`` / ``load_checkpoint`` / ``rotate_previous_candidates``
    in each driver file, so moving those lines here would make the pipelines
    look un-checkpointed to the tests that exist to prove they are not.
    """

    state.last_chunk_arrival_time = time.time()


# --------------------------------------------------------------------------- #
# the per-chunk steps
# --------------------------------------------------------------------------- #

def begin_chunk(state: ChunkLoopState, block: Any, metadata: dict) -> None:
    """Start this chunk's clock, count it, and log the block that arrived."""

    state.chunk_start_time = time.time()
    state.actual_chunk_count += 1
    log_block_processing(state.actual_chunk_count, block.shape, str(block.dtype), metadata)


def log_chunk_arrival_latency(state: ChunkLoopState, chunk_label: Any, is_first: bool) -> None:
    """Log how long this chunk took to arrive from the file.

    ``is_first`` is supplied by the caller because the two drivers gate this on
    different counters: LF on the stream sequence number, HF on the number of
    chunks it actually processed. With a resume those differ, and neither has
    been shown to be the intended one, so both are kept as they were.
    """

    # Log time since last chunk arrived from file
    chunk_arrival_time = time.time()
    if not is_first:
        time_since_last = chunk_arrival_time - state.last_chunk_arrival_time
        if time_since_last > 10:
            logger.warning(
                f"Chunk {chunk_label} took {time_since_last:.1f}s to arrive from file. "
                f"This may indicate slow I/O or large buffer concatenation. "
                f"Consider reducing chunk size if this is frequent."
            )
        elif time_since_last > 5:
            logger.info(
                f"Chunk {chunk_label} arrived after {time_since_last:.1f}s. "
                f"File I/O is proceeding normally."
            )
    state.last_chunk_arrival_time = chunk_arrival_time


def record_oom(collector: Any, chunk_idx: int, error: BaseException) -> None:
    """Record an out-of-memory failure. The caller still re-raises."""

    collector.record_oom_error()
    logger.exception(f"Out of memory processing chunk {chunk_idx:03d}: {error}")


def record_chunk_failure(state: ChunkLoopState, chunk_idx: int, error: BaseException) -> None:
    """SPEC-IO-002: do not silently drop chunks.

    Counting the failure is what turns the file's status into ``*_PARTIAL``
    instead of ``SUCCESS``.
    """

    state.failed_chunk_count += 1
    logger.exception(f"Error processing chunk {chunk_idx:03d}: {error}")


def finish_chunk(
    state: ChunkLoopState,
    chunk_label: Any,
    remaining_chunks: int,
    report_eta: bool,
) -> float:
    """Record this chunk's wall time and, when asked, log the ETA.

    ``remaining_chunks`` is the caller's: LF knows the exact chunk count up
    front, HF estimates it from the file length because its stream does not
    announce one.
    """

    # Track chunk processing time
    chunk_processing_time = time.time() - state.chunk_start_time
    state.chunk_processing_times.append(chunk_processing_time)

    if chunk_processing_time > 30:
        logger.warning(
            f"Chunk {chunk_label} processing took {chunk_processing_time:.1f}s. "
            f"This is unusually long. Check system resources."
        )

    # Estimate remaining time
    if report_eta:
        avg_time = sum(state.chunk_processing_times) / len(state.chunk_processing_times)
        eta_seconds = remaining_chunks * avg_time
        if eta_seconds > 60:
            logger.info(
                f"Chunk {chunk_label} processed in {chunk_processing_time:.1f}s. "
                f"Average: {avg_time:.1f}s/chunk. ETA: {eta_seconds/60:.1f} min"
            )
        else:
            logger.debug(
                f"Chunk {chunk_label} processed in {chunk_processing_time:.1f}s. "
                f"Average: {avg_time:.1f}s/chunk. ETA: {eta_seconds:.1f}s"
            )

    return chunk_processing_time


# --------------------------------------------------------------------------- #
# after the loop
# --------------------------------------------------------------------------- #

def export_validation_metrics(collector: Any, save_dir: Path, fits_path: Path) -> None:
    """Write the run's validation metrics; never fatal to the file's result."""

    try:
        validation_dir = save_dir / "Validation" / fits_path.stem
        collector.export_to_json(validation_dir)
        logger.info(f"Validation metrics exported to: {validation_dir}")
    except Exception as e:
        logger.warning(f"Failed to export validation metrics: {e}")
