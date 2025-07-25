"""Candidate CSV utilities for FRB pipeline."""
import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CANDIDATE_HEADER = [
    "file",
    "chunk_id",  # 🧩 NUEVO: ID del chunk
    "slice_id",  # 🧩 RENOMBRADO: Más claro que "slice"
    "band_id",   # 🧩 RENOMBRADO: Más claro que "band"
    "detection_prob",  # 🧩 RENOMBRADO: Más claro que "prob"
    "dm_pc_cm-3",
    "t_sec",
    "t_sample",
    "x1",
    "y1",
    "x2",
    "y2",
    "snr",
    "class_prob",
    "is_burst",
    "patch_file",
]

def ensure_csv_header(csv_path: Path) -> None:
    """Create csv_path with the standard candidate header if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        return
    try:
        with csv_path.open("w", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(CANDIDATE_HEADER)
    except PermissionError as e:
        logger.error("Error de permisos al crear CSV %s: %s", csv_path, e)
        raise

def append_candidate(csv_path: Path, candidate_row: list) -> None:
    """Append a candidate row to the CSV file."""
    with csv_path.open("a", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(candidate_row)
