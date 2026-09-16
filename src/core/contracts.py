"""Lightweight, immutable data contracts for pipeline stages.

These dataclasses make stage inputs/outputs explicit (SDD Etapa 3). They wrap the
mutable global ``config`` so hot paths can receive immutable snapshots instead of
reading global state. They are intentionally dependency-free (numpy only) and do
NOT mutate ``config``.

Contracts:
    ObservationMetadata   — per-file observation parameters (from headers)
    PipelineConfigSnapshot— immutable snapshot of search configuration
    ChunkPlan             — boundaries/overlap of a streamed chunk
    DMGrid                — DM trial values + image-row <-> DM mapping
    CandidateRecord       — minimal candidate description
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ObservationMetadata:
    """Per-file observation parameters extracted from the file header."""

    time_reso: float
    freq_reso: int
    file_leng: int
    freq_mhz: Tuple[float, ...]  # ascending frequency axis (MHz)
    down_time_rate: int = 1
    down_freq_rate: int = 1
    data_needs_reversal: bool = False
    tstart_mjd: Optional[float] = None

    @property
    def freq_low(self) -> float:
        return float(min(self.freq_mhz)) if self.freq_mhz else 0.0

    @property
    def freq_high(self) -> float:
        return float(max(self.freq_mhz)) if self.freq_mhz else 0.0

    @property
    def effective_time_reso(self) -> float:
        return float(self.time_reso) * max(1, int(self.down_time_rate))

    @property
    def duration_s(self) -> float:
        return float(self.file_leng) * float(self.time_reso)

    @classmethod
    def from_config(cls, config) -> "ObservationMetadata":
        freq = getattr(config, "FREQ", None)
        freq_tuple = tuple(float(x) for x in np.asarray(freq).ravel()) if freq is not None else tuple()
        return cls(
            time_reso=float(getattr(config, "TIME_RESO", 0.0)),
            freq_reso=int(getattr(config, "FREQ_RESO", 0)),
            file_leng=int(getattr(config, "FILE_LENG", 0)),
            freq_mhz=freq_tuple,
            down_time_rate=int(getattr(config, "DOWN_TIME_RATE", 1) or 1),
            down_freq_rate=int(getattr(config, "DOWN_FREQ_RATE", 1) or 1),
            data_needs_reversal=bool(getattr(config, "DATA_NEEDS_REVERSAL", False)),
            tstart_mjd=getattr(config, "TSTART_MJD", None),
        )


@dataclass(frozen=True)
class PipelineConfigSnapshot:
    """Immutable snapshot of the search configuration for hot paths."""

    dm_min: float
    dm_max: float
    dm_grid_mode: str = "legacy_uniform"
    snr_thresh: float = 5.0
    class_prob: float = 0.5
    save_only_burst: bool = False
    prewhiten_before_dm: bool = False
    bowtie_collapse_ratio: float = 2.0

    @classmethod
    def from_config(cls, config) -> "PipelineConfigSnapshot":
        return cls(
            dm_min=float(getattr(config, "DM_min", 0.0)),
            dm_max=float(getattr(config, "DM_max", 0.0)),
            dm_grid_mode=str(getattr(config, "DM_GRID_MODE", "legacy_uniform")).lower(),
            snr_thresh=float(getattr(config, "SNR_THRESH", 5.0)),
            class_prob=float(getattr(config, "CLASS_PROB", 0.5)),
            save_only_burst=bool(getattr(config, "SAVE_ONLY_BURST", False)),
            prewhiten_before_dm=bool(getattr(config, "PREWHITEN_BEFORE_DM", False)),
            bowtie_collapse_ratio=float(getattr(config, "BOWTIE_COLLAPSE_RATIO", 2.0)),
        )


@dataclass(frozen=True)
class ChunkPlan:
    """Boundaries and overlap (in samples) for a single streamed chunk."""

    chunk_idx: int
    start_sample: int
    end_sample: int
    overlap_left: int = 0
    overlap_right: int = 0

    def __post_init__(self) -> None:
        if self.end_sample < self.start_sample:
            raise ValueError(
                f"ChunkPlan end_sample ({self.end_sample}) < start_sample ({self.start_sample})"
            )
        if self.overlap_left < 0 or self.overlap_right < 0:
            raise ValueError("ChunkPlan overlaps must be non-negative")

    @property
    def valid_length(self) -> int:
        return int(self.end_sample) - int(self.start_sample)

    @property
    def block_start(self) -> int:
        return max(0, int(self.start_sample) - int(self.overlap_left))

    @property
    def block_end(self) -> int:
        return int(self.end_sample) + int(self.overlap_right)


@dataclass(frozen=True)
class DMGrid:
    """DM trial grid with explicit image-row <-> DM mapping (SPEC-DM-005)."""

    values: np.ndarray

    @property
    def dm_min(self) -> float:
        return float(self.values[0]) if self.values.size else 0.0

    @property
    def dm_max(self) -> float:
        return float(self.values[-1]) if self.values.size else 0.0

    @property
    def size(self) -> int:
        return int(self.values.size)

    def dm_for_row(self, row: int, height: Optional[int] = None) -> float:
        """Map an image row index to a DM value, never exceeding ``dm_max``.

        When ``height`` equals the grid size, the grid is indexed directly.
        Otherwise the row fraction is resampled onto the grid.
        """
        if self.values.size == 0:
            return 0.0
        if height is None:
            height = self.values.size
        denom = max(int(height) - 1, 1)
        frac = min(max(float(row) / denom, 0.0), 1.0)
        idx = int(round(frac * (self.values.size - 1)))
        return float(self.values[idx])

    @classmethod
    def from_values(cls, values) -> "DMGrid":
        arr = np.asarray(values, dtype=np.float64).ravel()
        return cls(values=arr)

    @classmethod
    def from_config(cls, config) -> "DMGrid":
        from .pipeline_parameters import calculate_dm_values
        return cls(values=np.asarray(calculate_dm_values(), dtype=np.float64))


@dataclass
class CandidateRecord:
    """Minimal, serialisable candidate description."""

    file_name: str
    dm: float
    time_sec: float
    snr: float
    chunk_idx: int = 0
    slice_idx: int = 0
    band_idx: int = 0
    is_burst: bool = False
    box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    dm_status: str = "measured"
    extra: dict = field(default_factory=dict)
