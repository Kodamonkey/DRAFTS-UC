"""Data structures for detected candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(slots=True)
class Candidate:
    file: str
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
    # Nuevos campos para información completa
    t_sec_absolute: float | None = None  # Tiempo absoluto desde inicio del archivo
    snr_peak: float | None = None        # SNR calculado con compute_snr_profile (mismo que plots)

    def to_row(self) -> List:
        row = [
            self.file,
            self.slice_id,
            self.band_id,
            f"{self.prob:.3f}",
            f"{self.dm:.2f}",
            f"{self.t_sec_absolute if self.t_sec_absolute is not None else self.t_sec:.6f}",  # Usar tiempo absoluto si disponible
            self.t_sample,
            *self.box,
            f"{self.snr_peak if self.snr_peak is not None else self.snr:.2f}",  # Usar SNR de plots si disponible
        ]
        if self.class_prob is not None:
            row.append(f"{self.class_prob:.3f}")
        if self.is_burst is not None:
            row.append("burst" if self.is_burst else "no_burst")
        if self.patch_file is not None:
            row.append(self.patch_file)
        return row
