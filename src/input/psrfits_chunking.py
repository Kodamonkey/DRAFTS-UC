# Pure chunk arithmetic for the PSRFITS readers. No file I/O, no module globals.

"""The chunking arithmetic the PSRFITS readers share, with the I/O taken out.

Audit REF-03. ``stream_fits`` used to be a single 1182-line generator holding
four readers, and this arithmetic -- buffer accumulation, the valid window, the
overlap bookkeeping, the emitted metadata -- was inlined inside each of them.
Every off-by-one this project has shipped lived here: P1-01 (the overlap trimmed
twice), P1-04/05/06 (``start_sample`` reporting the block's start, the emergency
path advancing by ``2 * overlap`` too much, a constant overlap in the metadata)
and the tail-handler bug that streamed whole files twice.

Nothing in this module touches a file, a header or :mod:`src.config`. Every
function takes integers and returns integers or a small frozen record, so the
geometry can be checked in a unit test without writing a PSRFITS first.

Still NOT here, though nothing forbids it any more: the first-chunk clamp
(``if emitted - out_buf.shape[0] <= 0 and valid_start > 0``) and the metadata
dict of the two buffered astropy readers. ``tests/test_p1_regressions.py``
used to assert on the literal source text of those expressions inside
``src/input/fits_handler.py`` and require exactly two copies of each, so moving
them here failed a test that refactor was not allowed to edit. That test asserts
on the readers' output now, so the move is unblocked -- it just has not been
made.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

#: Force a concatenation once the buffer holds more blocks than this: many small
#: blocks make the concatenation itself the bottleneck.
DEFAULT_MAX_BUFFER_BLOCKS = 200

#: Above this chunk size the memory-derived ceiling replaces the ``2 * chunk``
#: rule, because ``2 * chunk`` would no longer fit anywhere.
LARGE_CHUNK_SAMPLES = 1_000_000


# --------------------------------------------------------------------------- #
# buffer sizing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BufferLimits:
    """How much the subint buffer may hold before a chunk is forced out."""

    max_buffer_samples: int
    max_buffer_blocks: int
    bytes_per_sample: int
    max_buffer_gb: float
    large_chunk: bool


def compute_buffer_limits(
    chunk_samples: int,
    overlap_samples: int,
    nchan: int,
    available_ram_gb: float,
    max_buffer_blocks: int = DEFAULT_MAX_BUFFER_BLOCKS,
) -> BufferLimits:
    """Buffer ceiling for one file, from the RAM actually free right now.

    ``available_ram_gb`` is passed in rather than read from ``psutil`` here so
    that the sizing rule can be tested at a fixed memory figure.
    """
    max_buffer_gb = min(available_ram_gb * 0.3, 8.0)
    bytes_per_sample = 4 * int(nchan)
    max_buffer_bytes = max_buffer_gb * (1024 ** 3)
    max_buffer_samples = int(max_buffer_bytes / bytes_per_sample)

    large_chunk = chunk_samples > LARGE_CHUNK_SAMPLES
    if large_chunk:
        max_buffer_samples = min(
            max_buffer_samples, chunk_samples + overlap_samples * 2
        )
    else:
        max_buffer_samples = max(max_buffer_samples, chunk_samples * 2)

    return BufferLimits(
        max_buffer_samples=max_buffer_samples,
        max_buffer_blocks=max_buffer_blocks,
        bytes_per_sample=bytes_per_sample,
        max_buffer_gb=max_buffer_gb,
        large_chunk=large_chunk,
    )


# --------------------------------------------------------------------------- #
# the accumulator
# --------------------------------------------------------------------------- #
class SubintBuffer:
    """Accumulates ``(nsamp, 1, nchan)`` blocks and hands out a contiguous array.

    Kept as a list until the moment a chunk is emitted: concatenating on every
    subint is O(N^2) on a long observation. ``concatenate`` pre-allocates rather
    than calling ``np.concatenate``, and returns the single stored block
    untouched when there is only one -- both readers relied on that, and on the
    view semantics of ``advance``.
    """

    def __init__(self, nchan: int) -> None:
        self._nchan = int(nchan)
        self._blocks: list[np.ndarray] = []
        self._total = 0

    def __bool__(self) -> bool:
        return bool(self._blocks)

    @property
    def total_samples(self) -> int:
        return self._total

    @property
    def n_blocks(self) -> int:
        return len(self._blocks)

    def append(self, block: np.ndarray) -> None:
        self._blocks.append(block)
        self._total += int(block.shape[0])

    def pad(self, gap: int) -> None:
        """Append *gap* zero samples: a hole between two subints."""
        pad = np.empty((gap, 1, self._nchan), dtype=np.float32)
        pad.fill(0.0)
        self.append(pad)

    def concatenate(self) -> np.ndarray:
        if not self._blocks:
            return np.zeros((0, 1, self._nchan), dtype=np.float32)
        if len(self._blocks) == 1:
            return self._blocks[0]

        if len(self._blocks) > 100:
            logger.warning(
                f"Concatenating large buffer: {len(self._blocks):,} blocks "
                f"(~{self._total:,} samples). This may take a few seconds..."
            )

        start_time = time.time()
        result = np.empty((self._total, 1, self._nchan), dtype=np.float32)
        current_idx = 0
        for block in self._blocks:
            block_len = block.shape[0]
            result[current_idx:current_idx + block_len] = block
            current_idx += block_len
        elapsed = time.time() - start_time

        if elapsed > 1.0:
            logger.warning(
                f"Buffer concatenation took {elapsed:.2f}s for {len(self._blocks):,} blocks "
                f"({self._total:,} samples). Consider reducing chunk size if this is frequent."
            )
        return result

    def advance(self, out_buf: np.ndarray, samples_to_remove: int) -> None:
        """Drop the samples just emitted and keep the tail as the next buffer.

        The tail is a *view* into ``out_buf``, which is what both readers did.
        """
        if samples_to_remove >= out_buf.shape[0]:
            self._blocks = []
            self._total = 0
            return
        remaining_buf = out_buf[samples_to_remove:]
        if remaining_buf.shape[0] > 0:
            self._blocks = [remaining_buf]
            self._total = int(remaining_buf.shape[0])
        else:
            self._blocks = []
            self._total = 0


# --------------------------------------------------------------------------- #
# where a subint belongs in the file
# --------------------------------------------------------------------------- #
def expected_start_sample(
    offs_sub_seconds: Optional[float],
    sub_index: int,
    *,
    tsub: float,
    tbin: float,
    nsuboffs: int,
    nsblk: int,
) -> int:
    """First sample of subint *sub_index*, from ``OFFS_SUB`` when it is there.

    ``OFFS_SUB`` is the centre of the subint in seconds since the start of the
    *observation*, so on a continuation file this is an absolute position and
    the reader zero-pads everything before it (divergence D4). Without the
    column the position is counted from ``NSUBOFFS`` instead.
    """
    if offs_sub_seconds is not None:
        start_sec = float(offs_sub_seconds) - 0.5 * tsub
        return int(round(start_sec / tbin))
    return (nsuboffs + sub_index) * nsblk


# --------------------------------------------------------------------------- #
# chunk geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChunkWindow:
    """Where a chunk sits *inside the buffer*, in buffer-relative samples.

    ``[start_with_overlap, end_with_overlap)`` is what gets copied out and
    ``[valid_start, valid_end)`` is the part of it the consumer may search; the
    rest is context for dedispersion at the edges.
    """

    start_with_overlap: int
    end_with_overlap: int
    valid_start: int
    valid_end: int
    actual_chunk_size: int
    emergency: bool


def plan_buffered_window(
    buffer_len: int,
    chunk_samples: int,
    overlap_samples: int,
    *,
    buffer_too_large: bool,
    large_chunk: bool,
) -> ChunkWindow:
    """The window the astropy readers cut out of a full buffer.

    Normally that is ``chunk_samples`` of valid data with ``overlap_samples`` of
    context on each side. When the buffer has outgrown its ceiling the reader
    emits early instead, taking a fraction of what it holds so the next read
    still has somewhere to go.
    """
    if buffer_too_large:
        safe_chunk_ratio = 0.7 if large_chunk else 0.9
        max_safe_chunk = int(buffer_len * safe_chunk_ratio)
        actual_chunk_size = min(chunk_samples, max_safe_chunk, buffer_len)
        if actual_chunk_size < overlap_samples * 2:
            actual_chunk_size = min(overlap_samples * 2, buffer_len)

        start_with_overlap = 0
        end_with_overlap = min(actual_chunk_size + overlap_samples * 2, buffer_len)
        valid_start = overlap_samples if end_with_overlap > overlap_samples * 2 else 0
        return ChunkWindow(
            start_with_overlap=start_with_overlap,
            end_with_overlap=end_with_overlap,
            valid_start=valid_start,
            valid_end=valid_start + actual_chunk_size,
            actual_chunk_size=actual_chunk_size,
            emergency=True,
        )

    return ChunkWindow(
        start_with_overlap=0,
        end_with_overlap=chunk_samples + overlap_samples * 2,
        valid_start=overlap_samples,
        valid_end=overlap_samples + chunk_samples,
        actual_chunk_size=chunk_samples,
        emergency=False,
    )


@dataclass(frozen=True)
class ChunkSpan:
    """Where a chunk sits *in the file*, in absolute samples.

    ``[start_sample, end_sample)`` is the valid window and
    ``[block_start_sample, block_end_sample)`` the block that carries it.
    """

    start_sample: int
    end_sample: int
    block_start_sample: int
    block_end_sample: int

    @property
    def actual_chunk_size(self) -> int:
        return self.end_sample - self.start_sample

    @property
    def overlap_left(self) -> int:
        return self.start_sample - self.block_start_sample

    @property
    def overlap_right(self) -> int:
        return max(0, self.block_end_sample - self.end_sample)


def plan_sequential_span(
    start: int, step: int, overlap_samples: int, total_samples: int
) -> ChunkSpan:
    """Geometry of a reader that seeks straight to a sample, with no buffer.

    This tiles the file exactly: the valid windows are ``[0, step)``,
    ``[step, 2 * step)`` ... and the overlap only widens the block around them.
    The buffered astropy readers do NOT agree with this (divergence D1).
    """
    read_start = max(0, start - overlap_samples)
    read_end = min(total_samples, start + step + overlap_samples)
    return ChunkSpan(
        start_sample=start,
        end_sample=min(start + step, total_samples),
        block_start_sample=read_start,
        block_end_sample=read_end,
    )


def chunk_metadata(
    span: ChunkSpan,
    *,
    chunk_samples: int,
    total_samples: int,
    nchans: int,
    nifs: int,
    block: np.ndarray,
    tbin_sec: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The dict yielded beside every block, built from the geometry alone.

    The overlaps are *derived* from the span: a chunk at the start of a file has
    no left overlap and one at the end has no right overlap, and reporting a
    constant ``overlap_samples`` instead made the consumer trim data that was
    never overlap (audit P1-06).
    """
    metadata: Dict[str, Any] = {
        "chunk_idx": span.start_sample // max(1, chunk_samples),
        "start_sample": span.start_sample,
        "end_sample": span.end_sample,
        "actual_chunk_size": span.actual_chunk_size,
        "block_start_sample": span.block_start_sample,
        "block_end_sample": span.block_end_sample,
        "overlap_left": span.overlap_left,
        "overlap_right": span.overlap_right,
        "total_samples": total_samples,
        "nchans": nchans,
        "nifs": nifs,
        "dtype": str(block.dtype),
        "shape": block.shape,
        "file_type": "fits",
        "tbin_sec": tbin_sec,
        "t_rel_start_sec": span.start_sample * tbin_sec,
        "t_rel_end_sec": span.end_sample * tbin_sec,
    }
    if extra:
        metadata.update(extra)
    return metadata
