"""Candidate management for FRB pipeline - handles CSV output and candidate serialization."""
from __future__ import annotations

# Standard library imports
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# Setup logger
logger = logging.getLogger(__name__)

def is_file_locked(file_path: Path) -> bool:
    """Check if a file is locked by another process (Windows)."""
    try:
        if not file_path.exists():
            return False
        
        # Intentar abrir el archivo en modo exclusivo
        with file_path.open("r+b"):
            pass
        return False
    except (PermissionError, OSError):
        return True
    except Exception:
        return False

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
    try:
        # Crear directorio si no existe
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Si el archivo ya existe, no hacer nada
        if csv_path.exists():
            return
            
        # Crear archivo con header
        with csv_path.open("w", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(CANDIDATE_HEADER)
            logger.debug(f"Header CSV creado en: {csv_path}")
            
    except PermissionError as e:
        logger.error(f"Error de permisos al crear CSV {csv_path}: {e}")
        logger.error("Verificar permisos de escritura en el directorio")
        raise
    except Exception as e:
        logger.error(f"Error inesperado al crear CSV {csv_path}: {e}")
        raise


def append_candidate(csv_path: Path, candidate_row: list) -> None:
    """Append a candidate row to the CSV file."""
    try:
        # Verificar que el directorio existe
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Verificar si el archivo existe, si no, crear con header
        if not csv_path.exists():
            ensure_csv_header(csv_path)
        
        # Verificar si el archivo está bloqueado
        if is_file_locked(csv_path):
            logger.warning(f"Archivo CSV bloqueado: {csv_path}")
            logger.warning("Creando archivo alternativo...")
            _create_alternative_csv(csv_path, candidate_row)
            return
        
        # Intentar escribir el candidato
        with csv_path.open("a", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(candidate_row)
            
    except PermissionError as e:
        logger.error(f"Error de permisos al escribir en CSV {csv_path}: {e}")
        logger.error("El archivo puede estar siendo usado por otro proceso o no hay permisos de escritura")
        _create_alternative_csv(csv_path, candidate_row)
            
    except Exception as e:
        logger.error(f"Error inesperado al escribir candidato en CSV {csv_path}: {e}")
        raise


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

    def to_row(self) -> List:
        """Convert candidate to CSV row format."""
        row = [
            self.file,
            self.chunk_id,  # Incluir chunk_id en CSV
            self.slice_id,
            self.band_id,
            f"{self.prob:.3f}",
            f"{self.dm:.2f}",
            f"{self.t_sec:.6f}",
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


def _create_alternative_csv(original_path: Path, candidate_row: list) -> None:
    """Create an alternative CSV file when the original is locked or has permission issues."""
    try:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_path = original_path.parent / f"{original_path.stem}_{timestamp}.csv"
        logger.info(f"Creando archivo alternativo: {alt_path}")
        
        with alt_path.open("w", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(CANDIDATE_HEADER)
            writer.writerow(candidate_row)
        logger.info(f"Candidato guardado en archivo alternativo: {alt_path}")
        
    except Exception as e:
        logger.error(f"No se pudo crear archivo alternativo: {e}")
        raise 
