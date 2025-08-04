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

    def to_row(self) -> List:
        row = [
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
        if self.class_prob is not None:
            row.append(f"{self.class_prob:.3f}")
        if self.is_burst is not None:
            row.append("burst" if self.is_burst else "no_burst")
        if self.patch_file is not None:
            row.append(self.patch_file)
        return row
