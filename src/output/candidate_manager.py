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
    "file", # Name of the file containing the candidate
    "chunk_id", # Index of the chunk in the file
    "slice_id", # Index of the slice in the chunk
    "band_id", # Index of the band in the chunk
    "detection_prob", # Probability of the candidate being a detection
    "dm_pc_cm-3", # Dispersion measure in pc/cm^3
    "t_sec_dm_time",  # Time from DM-time plot (same as shown in plot label)
    "t_sec_waterfall",  # Time from waterfall SNR peak (different method, also valuable)
    "t_sample", # Time of the sample in the chunk
    "mjd_utc",
    "mjd_bary_utc",
    "mjd_bary_tdb",
    "mjd_bary_utc_inf",
    "mjd_bary_tdb_inf",
    "x1",
    "y1",
    "x2",
    "y2",
    "snr_waterfall",  # SNR from waterfall raw (peak_snr_wf from plot) - Intensity
    "snr_patch_dedispersed",  # SNR from dedispersed patch (what was previously "snr") - Intensity
    "snr_waterfall_linear",  # SNR from waterfall raw in Linear polarization - HF pipeline only
    "snr_patch_dedispersed_linear",  # SNR from dedispersed patch in Linear polarization - HF pipeline only
    "width_ms",
    "class_prob_intensity",  # Classification probability in Intensity (I) - always present
    "is_burst_intensity",  # BURST classification in Intensity (I) - always present
    "class_prob_linear",  # Classification probability in Linear (L) - HF pipeline only
    "is_burst_linear",  # BURST classification in Linear (L) - HF pipeline only
    "is_burst",  # Final classification (I for classic, I+L for HF when SAVE_ONLY_BURST=True)
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
    """Data structure for detected FRB candidates.
    
    Supports both classic and HF pipelines:
    - Classic pipeline: class_prob_linear and is_burst_linear will be None
    - HF pipeline: All fields are populated, including Linear polarization classification
    """
    file: str
    chunk_id: int                                               
    slice_id: int
    band_id: int
    prob: float
    dm: float
    t_sec_dm_time: float  # Time from DM-time plot (same as shown in plot label)
    t_sec_waterfall: float | None = None  # Time from waterfall SNR peak (different method)
    t_sample: int = 0
    box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    snr_waterfall: float | None = None  # SNR from waterfall raw (peak_snr_wf) - Intensity
    snr_patch_dedispersed: float = 0.0  # SNR from dedispersed patch - Intensity
    snr_waterfall_linear: float | None = None  # SNR from waterfall raw in Linear - HF pipeline only
    snr_patch_dedispersed_linear: float | None = None  # SNR from dedispersed patch in Linear - HF pipeline only
    width_ms: float | None = None
    class_prob_intensity: float | None = None  # Classification probability in Intensity (I)
    is_burst_intensity: bool | None = None  # BURST classification in Intensity (I)
    class_prob_linear: float | None = None  # Classification probability in Linear (L) - HF only
    is_burst_linear: bool | None = None  # BURST classification in Linear (L) - HF only
    is_burst: bool | None = None  # Final classification (I for classic, I+L for HF when SAVE_ONLY_BURST=True)
    patch_file: str | None = None
    mjd_utc: float | None = None
    mjd_bary_utc: float | None = None
    mjd_bary_tdb: float | None = None
    mjd_bary_utc_inf: float | None = None
    mjd_bary_tdb_inf: float | None = None

    def to_row(self) -> List:
        """Convert candidate to CSV row format."""
        row = [
            self.file,
            self.chunk_id,                           
            self.slice_id,
            self.band_id,
            f"{self.prob:.3f}",
            f"{self.dm:.2f}",  # DM calculated with extract_candidate_dm (same as plot)
            f"{self.t_sec_dm_time:.6f}",  # Time from DM-time plot (same as shown in plot label)
            f"{self.t_sec_waterfall:.6f}" if self.t_sec_waterfall is not None else "",  # Time from waterfall SNR peak
            self.t_sample,
        ]
        # Add MJD columns
        row.append(f"{self.mjd_utc:.12f}" if self.mjd_utc is not None else "")
        row.append(f"{self.mjd_bary_utc:.12f}" if self.mjd_bary_utc is not None else "")
        row.append(f"{self.mjd_bary_tdb:.12f}" if self.mjd_bary_tdb is not None else "")
        row.append(f"{self.mjd_bary_utc_inf:.12f}" if self.mjd_bary_utc_inf is not None else "")
        row.append(f"{self.mjd_bary_tdb_inf:.12f}" if self.mjd_bary_tdb_inf is not None else "")
        # Add box coordinates
        row.extend(self.box)
        # Add SNR values (Intensity)
        row.append(f"{self.snr_waterfall:.2f}" if self.snr_waterfall is not None else "")  # SNR from waterfall - Intensity
        row.append(f"{self.snr_patch_dedispersed:.2f}")  # SNR from dedispersed patch - Intensity
        # Add SNR values (Linear - HF pipeline only)
        row.append(f"{self.snr_waterfall_linear:.2f}" if self.snr_waterfall_linear is not None else "")  # SNR from waterfall - Linear
        row.append(f"{self.snr_patch_dedispersed_linear:.2f}" if self.snr_patch_dedispersed_linear is not None else "")  # SNR from dedispersed patch - Linear
        if self.width_ms is not None:
            row.append(f"{self.width_ms:.3f}")
        else:
            row.append("")
        # Add classification probabilities (Intensity - always present)
        if self.class_prob_intensity is not None:
            row.append(f"{self.class_prob_intensity:.3f}")
        else:
            row.append("")
        if self.is_burst_intensity is not None:
            row.append("burst" if self.is_burst_intensity else "no_burst")
        else:
            row.append("")
        # Add Linear polarization classification (HF pipeline only)
        if self.class_prob_linear is not None:
            row.append(f"{self.class_prob_linear:.3f}")
        else:
            row.append("")  # Empty for classic pipeline
        if self.is_burst_linear is not None:
            row.append("burst" if self.is_burst_linear else "no_burst")
        else:
            row.append("")  # Empty for classic pipeline
        # Add final classification
        if self.is_burst is not None:
            row.append("burst" if self.is_burst else "no_burst")
        else:
            row.append("")
        if self.patch_file is not None:
            row.append(self.patch_file)
        else:
            row.append("")
        return row 
