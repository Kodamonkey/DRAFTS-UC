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

    def to_row(self) -> List:
        return [
            self.file,
            self.slice_id,
            self.band_id,
            f"{self.prob:.3f}",
            f"{self.dm:.2f}",
            f"{self.t_sec:.6f}",
            self.t_sample,
            *self.box,
            f"{self.snr:.2f}",
        ]
