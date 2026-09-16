"""Chunk-level checkpoint/resume for pipeline recovery.

Writes an atomic JSON checkpoint after each chunk completes.  On restart,
``load_checkpoint`` returns the index of the last fully completed chunk so
the pipeline can skip ahead without reprocessing.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_CHECKPOINT_FILENAME = ".drafts_checkpoint.json"


def _checkpoint_path(results_dir: Path, file_stem: str) -> Path:
    return results_dir / file_stem / _CHECKPOINT_FILENAME


def save_checkpoint(
    results_dir: Path,
    file_stem: str,
    chunk_idx: int,
    total_chunks: int,
    extra: dict | None = None,
) -> None:
    """Atomically write a checkpoint after *chunk_idx* completes.

    Uses write-to-temp + rename to avoid corruption on crash.
    """
    cp_path = _checkpoint_path(results_dir, file_stem)
    cp_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "file_stem": file_stem,
        "last_completed_chunk": chunk_idx,
        "total_chunks": total_chunks,
    }
    if extra:
        payload.update(extra)

    # Atomic write: temp file in same directory, then rename
    try:
        fd, tmp = tempfile.mkstemp(
            dir=str(cp_path.parent), suffix=".tmp", prefix=".ckpt_"
        )
        with open(fd, "w") as f:
            json.dump(payload, f)
        Path(tmp).replace(cp_path)
    except OSError as e:
        logger.warning("Failed to write checkpoint for %s chunk %d: %s", file_stem, chunk_idx, e)


def load_checkpoint(results_dir: Path, file_stem: str) -> int:
    """Return the index of the last completed chunk, or -1 if none."""
    cp_path = _checkpoint_path(results_dir, file_stem)
    if not cp_path.exists():
        return -1
    try:
        with cp_path.open() as f:
            data = json.load(f)
        last = int(data.get("last_completed_chunk", -1))
        logger.info(
            "Resuming %s from chunk %d (checkpoint found)", file_stem, last + 1
        )
        return last
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("Corrupt checkpoint for %s (%s), starting from scratch", file_stem, e)
        return -1


def clear_checkpoint(results_dir: Path, file_stem: str) -> None:
    """Remove the checkpoint file after the file finishes successfully."""
    cp_path = _checkpoint_path(results_dir, file_stem)
    try:
        cp_path.unlink(missing_ok=True)
    except OSError:
        pass
