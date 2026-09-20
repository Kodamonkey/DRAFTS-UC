"""Chunk-level checkpoint/resume for pipeline recovery.

Writes an atomic JSON checkpoint after each chunk completes.  On restart,
``load_checkpoint`` returns the index of the last fully completed chunk so
the pipeline can skip ahead without reprocessing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_CHECKPOINT_FILENAME = ".drafts_checkpoint.json"

#: Bumped whenever the payload changes shape. An older checkpoint is discarded
#: rather than misread.
_SCHEMA_VERSION = 2

#: Settings that change what a chunk means. Resuming across a change in any of
#: them splices two different searches into one CSV with nothing to mark the
#: seam, so the checkpoint is refused instead.
_FINGERPRINT_KEYS = (
    "DM_min", "DM_max", "DM_GRID_MODE", "MAX_DM_SMEARING_MS",
    "DOWN_TIME_RATE", "DOWN_FREQ_RATE", "SLICE_DURATION_MS",
    "TEMPORAL_DOWNSAMPLING_MODE", "PREWHITEN_BEFORE_DM",
    "DET_PROB", "CLASS_PROB", "SAVE_ONLY_BURST", "MAX_CHUNK_SAMPLES",
)


def compute_run_fingerprint(input_path, config_module) -> str:
    """Identify the search this checkpoint belongs to.

    Covers both the parameters that define the search and the identity of the
    input file, so a checkpoint cannot be applied to a different observation
    that happens to share a file stem.
    """
    parts = [f"{key}={getattr(config_module, key, None)!r}" for key in _FINGERPRINT_KEYS]
    try:
        stat = Path(input_path).stat()
        parts.append(f"size={stat.st_size}")
        parts.append(f"mtime={int(stat.st_mtime)}")
    except OSError:
        parts.append("stat=unavailable")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _checkpoint_path(results_dir: Path, file_stem: str) -> Path:
    return results_dir / file_stem / _CHECKPOINT_FILENAME


def save_checkpoint(
    results_dir: Path,
    file_stem: str,
    chunk_idx: int,
    total_chunks: int,
    extra: dict | None = None,
    fingerprint: str | None = None,
) -> None:
    """Atomically write a checkpoint after *chunk_idx* completes.

    Uses write-to-temp + rename to avoid corruption on crash.
    """
    cp_path = _checkpoint_path(results_dir, file_stem)
    cp_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": _SCHEMA_VERSION,
        "file_stem": file_stem,
        "last_completed_chunk": chunk_idx,
        "total_chunks": total_chunks,
        "fingerprint": fingerprint,
    }
    if extra:
        payload.update(extra)

    # Atomic write: temp file in same directory, then rename
    def _write() -> None:
        fd, tmp = tempfile.mkstemp(
            dir=str(cp_path.parent), suffix=".tmp", prefix=".ckpt_"
        )
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        Path(tmp).replace(cp_path)

    try:
        # Safe to retry: temp file plus rename, so a failed attempt leaves the
        # previous checkpoint intact.
        from .retry import with_retry

        with_retry(_write, description=f"checkpoint for {file_stem}")
    except OSError as e:
        logger.warning("Failed to write checkpoint for %s chunk %d: %s", file_stem, chunk_idx, e)


def should_skip_chunk(chunk_idx: int, resume_after: int) -> bool:
    """True when *chunk_idx* was already completed in an earlier run.

    ``chunk_idx`` is 1-based (the streaming loop enumerates from 1) and
    :func:`save_checkpoint` stores that same 1-based index, so ``resume_after``
    IS the last completed chunk. Skipping must therefore stop AT it, not one
    past it: with ``resume_after=5`` the next chunk to process is 6.

    ``resume_after == -1`` means no checkpoint, so nothing is skipped.
    """
    return chunk_idx <= resume_after


def load_checkpoint(
    results_dir: Path,
    file_stem: str,
    expected_fingerprint: str | None = None,
) -> int:
    """Return the index of the last completed chunk, or -1 if none applies.

    When *expected_fingerprint* is given, a checkpoint written for a different
    search or a different input file is refused. Without that check, resuming
    after changing (say) ``--dm-max`` produced a single CSV holding chunks
    searched over two different DM ranges, with nothing recording the seam.
    """
    cp_path = _checkpoint_path(results_dir, file_stem)
    if not cp_path.exists():
        return -1
    try:
        with cp_path.open(encoding="utf-8") as f:
            data = json.load(f)

        version = int(data.get("schema_version", 1))
        if version != _SCHEMA_VERSION:
            logger.warning(
                "Checkpoint for %s was written by schema v%d (this is v%d); "
                "starting from scratch", file_stem, version, _SCHEMA_VERSION,
            )
            return -1

        if expected_fingerprint is not None:
            stored = data.get("fingerprint")
            if stored != expected_fingerprint:
                logger.warning(
                    "Checkpoint for %s belongs to a different run (configuration "
                    "or input file changed); starting from scratch instead of "
                    "mixing two searches in one output", file_stem,
                )
                return -1

        last = int(data.get("last_completed_chunk", -1))
        logger.info(
            "Resuming %s from chunk %d (checkpoint found)", file_stem, last + 1
        )
        return last
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning("Corrupt checkpoint for %s (%s), starting from scratch", file_stem, e)
        return -1


def clear_checkpoint(results_dir: Path, file_stem: str) -> None:
    """Remove the checkpoint file after the file finishes successfully."""
    cp_path = _checkpoint_path(results_dir, file_stem)
    try:
        cp_path.unlink(missing_ok=True)
    except OSError:
        pass
