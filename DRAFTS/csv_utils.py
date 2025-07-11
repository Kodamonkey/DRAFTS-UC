from __future__ import annotations

import csv
import time
from pathlib import Path

from .candidate import Candidate

__all__ = ["ensure_csv_header", "write_candidate"]


HEADER = [
    "file",
    "slice",
    "band",
    "prob",
    "dm_pc_cm-3",
    "t_sec_absolute",
    "t_sample",
    "x1",
    "y1",
    "x2",
    "y2",
    "snr_peak",
    "class_prob",
    "is_burst",
    "patch_file",
]


def ensure_csv_header(csv_path: Path) -> None:
    """Create ``csv_path`` with the standard candidate header if missing."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        return

    with csv_path.open("w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(HEADER)


def write_candidate(csv_file: Path, candidate: Candidate) -> None:
    """Append ``candidate`` to ``csv_file``."""
    try:
        with csv_file.open("a", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(candidate.to_row())
    except PermissionError:
        alt_csv = csv_file.with_suffix(f".{int(time.time())}.csv")
        with alt_csv.open("a", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(candidate.to_row())
