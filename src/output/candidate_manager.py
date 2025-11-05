# This module handles candidate persistence and CSV output.

"""Candidate management for FRB pipeline - handles CSV output and candidate serialization."""
from __future__ import annotations

                          
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

              
logger = logging.getLogger(__name__)

                                 
CANDIDATE_HEADER = [
    "file",
    "chunk_id",                
    "slice_id",                         
    "band_id",                         
    "detection_prob",                        
    "dm_pc_cm-3",
    "t_sec",
    "t_sample",
    "x1",
    "y1",
    "x2",
    "y2",
    "snr",
    "width_ms",
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
        logger.error("Permission error while creating CSV %s: %s", csv_path, e)
        raise


def append_candidate(csv_path: Path, candidate_row: list) -> None:
    """Append a candidate row to the CSV file."""
    with csv_path.open("a", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(candidate_row)


@dataclass(slots=True)
class Candidate:
    """Data structure for detected FRB candidates."""
    file: str
    chunk_id: int                                               
    slice_id: int
    band_id: int
    prob: float
    dm: float
    t_sec: float
    t_sample: int
    box: Tuple[int, int, int, int]
    snr: float
    class_prob: float | None = None
    is_burst: bool | None = None
    patch_file: str | None = None
    width_ms: float | None = None

    def to_row(self) -> List:
        """Convert candidate to CSV row format."""
        row = [
            self.file,
            self.chunk_id,                           
            self.slice_id,
            self.band_id,
            f"{self.prob:.3f}",
            f"{self.dm:.2f}",
            f"{self.t_sec:.6f}",
            self.t_sample,
            *self.box,
            f"{self.snr:.2f}",
        ]
        if self.width_ms is not None:
            row.append(f"{self.width_ms:.3f}")
        else:
            row.append("")
        if self.class_prob is not None:
            row.append(f"{self.class_prob:.3f}")
        if self.is_burst is not None:
            row.append("burst" if self.is_burst else "no_burst")
        if self.patch_file is not None:
            row.append(self.patch_file)
        return row 
