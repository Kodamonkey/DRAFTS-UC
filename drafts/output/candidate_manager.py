"""Candidate management for FRB pipeline - handles CSV output and candidate serialization."""
from __future__ import annotations

# Standard library imports
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# Third-party imports
import numpy as np

# Setup logger
logger = logging.getLogger(__name__)

# CSV header for candidate output
CANDIDATE_HEADER = [
    "file",
    "chunk_id",  # ID del chunk
    "slice_id",  # Más claro que "slice"
    "band_id",   # Más claro que "band"
    "detection_prob",  # Más claro que "prob"
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


@dataclass(slots=True)
class Candidate:
    """Data structure for detected FRB candidates."""
    file: str
    chunk_id: int  # ID del chunk donde se encontró el candidato
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
    
    # Nuevos campos para validación DM-aware (estrategias E1/E2)
    dm_star: float | None = None           # DM* óptimo encontrado
    dm_star_err: float | None = None       # Error en DM*
    snr_dm0: float | None = None           # SNR a DM=0
    snr_dmstar: float | None = None        # SNR a DM*
    delta_snr: float | None = None         # ΔSNR = SNR(DM*) - SNR(0)
    subband_agreement: float | None = None # Acuerdo entre sub-bandas (%)
    validation_passed: bool | None = None  # True si pasa validación DM-aware
    validation_reason: str | None = None   # Razón del fallo si validation_passed=False
    strategy: str | None = None            # E1_expand, E2_fish, o None

    def to_row(self) -> List:
        """Convert candidate to CSV row format."""
        row = [
            self.file,
            self.chunk_id,  # Incluir chunk_id en CSV
            self.slice_id,
            self.band_id,
            f"{self.prob:.3f}",
            f"{self.dm:.2f}",
            self.t_sec,
            self.t_sample,
            *self.box,
            f"{self.snr:.2f}",
        ]
        if self.class_prob is not None:
            row.append(f"{self.class_prob:.3f}")
        if self.is_burst is not None:
            row.append("burst" if self.is_burst else "no_burst")
        if self.patch_file is not None:
            row.append(self.patch_file)
        return row
    
    def calculate_priority_score(self) -> float:
        """
        Calcula un score interpretable para priorización de candidatos.
        
        Score = w1*delta_snr + w2*log_snr_star + w3*subband_agreement - w4*penalty_if_dm0_peak
        
        Returns:
            float: Score de prioridad (mayor = más prioritario)
        """
        # Pesos para diferentes componentes del score
        w1 = 2.0    # Peso para ΔSNR
        w2 = 1.0    # Peso para log(SNR*)
        w3 = 0.5    # Peso para acuerdo sub-bandas
        w4 = 3.0    # Penalización si el pico está en DM=0
        
        score = 0.0
        
        # Componente 1: ΔSNR (diferencia entre SNR(DM*) y SNR(0))
        if self.delta_snr is not None:
            score += w1 * self.delta_snr
        
        # Componente 2: log(SNR*) - favorece candidatos con SNR alto
        if self.snr_dmstar is not None and self.snr_dmstar > 0:
            score += w2 * np.log10(self.snr_dmstar)
        
        # Componente 3: Acuerdo entre sub-bandas
        if self.subband_agreement is not None:
            score += w3 * (self.subband_agreement / 100.0)  # Normalizar a [0,1]
        
        # Componente 4: Penalización si el pico está en DM=0 (posible RFI)
        if self.dm_star is not None and self.dm_star < 5.0:  # Umbral de 5 pc cm⁻³
            score -= w4
        
        # Bonus por validación exitosa
        if self.validation_passed:
            score += 1.0
        
        return score
    
    def get_validation_summary(self) -> str:
        """
        Genera un resumen legible del estado de validación.
        
        Returns:
            str: Resumen de validación
        """
        if self.validation_passed is None:
            return "No validado"
        
        if not self.validation_passed:
            return f"Falló: {self.validation_reason or 'Razón desconocida'}"
        
        summary = f"✓ DM*={self.dm_star:.1f}±{self.dm_star_err:.1f} pc cm⁻³"
        summary += f" | ΔSNR={self.delta_snr:.2f}"
        
        if self.subband_agreement is not None:
            summary += f" | Sub-bandas: {self.subband_agreement:.1f}%"
        
        if self.strategy:
            summary += f" | Estrategia: {self.strategy}"
        
        return summary 
