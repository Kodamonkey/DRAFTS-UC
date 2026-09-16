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
    "dm_uncertainty",
    "dm_status",
    "best_width_ms",
    "n_trials",
    "post_trials_sigma",
    "snr_pre_dedisp",
    "snr_post_dedisp",
    "linear_fraction",
    "physical_score",
    "rank_score",
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
        try:
            with csv_path.open("r", newline="") as f_csv:
                rows = list(csv.reader(f_csv))
            if rows and rows[0] != CANDIDATE_HEADER:
                width = len(CANDIDATE_HEADER)
                padded = [CANDIDATE_HEADER]
                for row in rows[1:]:
                    padded.append((row + [""] * width)[:width])
                with csv_path.open("w", newline="") as f_csv:
                    writer = csv.writer(f_csv)
                    writer.writerows(padded)
        except Exception as e:
            logger.warning("Could not validate/update CSV header for %s: %s", csv_path, e)
        return
    try:
        with csv_path.open("w", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(CANDIDATE_HEADER)
    except PermissionError as e:
        logger.error("Permission error while creating CSV %s: %s", csv_path, e)
        raise


def append_candidate(csv_path: Path, candidate_row: list) -> None:
    """Append a candidate row to the CSV file.

    Uses a global ``CandidateWriter`` per path to buffer writes and avoid
    opening/closing the file for every single candidate.
    """
    CandidateWriter.get(csv_path).write(candidate_row)


class CandidateWriter:
    """Buffered CSV writer that batches candidate rows.

    Keeps the file handle open and flushes every ``flush_interval`` rows
    or when explicitly flushed / closed.
    """

    _instances: dict[Path, "CandidateWriter"] = {}

    def __init__(self, csv_path: Path, flush_interval: int = 50):
        self._path = csv_path
        self._flush_interval = flush_interval
        self._buffer: list[list] = []
        self._fh = None
        self._writer = None

    @classmethod
    def get(cls, csv_path: Path) -> "CandidateWriter":
        resolved = csv_path.resolve()
        if resolved not in cls._instances:
            cls._instances[resolved] = cls(csv_path)
        return cls._instances[resolved]

    @classmethod
    def flush_all(cls) -> None:
        """Flush and close all open writers (call at pipeline end)."""
        for w in list(cls._instances.values()):
            w.close()
        cls._instances.clear()

    def _ensure_open(self):
        if self._fh is None or self._fh.closed:
            self._fh = self._path.open("a", newline="")
            self._writer = csv.writer(self._fh)

    def write(self, row: list) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self._flush_interval:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        self._ensure_open()
        self._writer.writerows(self._buffer)
        self._fh.flush()
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        if self._fh is not None and not self._fh.closed:
            self._fh.close()
            self._fh = None


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
    dm_uncertainty: float | None = None
    dm_status: str = "measured"
    best_width_ms: float | None = None
    n_trials: int | None = None
    post_trials_sigma: float | None = None
    snr_pre_dedisp: float | None = None
    snr_post_dedisp: float | None = None
    linear_fraction: float | None = None
    physical_score: float | None = None
    rank_score: float | None = None
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
        row.append(f"{self.dm_uncertainty:.3f}" if self.dm_uncertainty is not None else "")
        row.append(self.dm_status)
        row.append(f"{self.best_width_ms:.3f}" if self.best_width_ms is not None else "")
        row.append(self.n_trials if self.n_trials is not None else "")
        row.append(f"{self.post_trials_sigma:.3f}" if self.post_trials_sigma is not None else "")
        row.append(f"{self.snr_pre_dedisp:.2f}" if self.snr_pre_dedisp is not None else "")
        row.append(f"{self.snr_post_dedisp:.2f}" if self.snr_post_dedisp is not None else "")
        row.append(f"{self.linear_fraction:.4f}" if self.linear_fraction is not None else "")
        row.append(f"{self.physical_score:.4f}" if self.physical_score is not None else "")
        row.append(f"{self.rank_score:.4f}" if self.rank_score is not None else "")
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
