# This module holds the display normalization shared by every waterfall panel.

"""One definition of how a waterfall block is prepared for display.

Two recipes used to be copied across the package: the per-channel
normalization applied when ``normalize=True`` (seven copies) and the
percentile pair handed to ``imshow`` as ``vmin``/``vmax`` (nine copies).
Both live here now, so a panel drawn on its own and the same panel inside
the composite figure cannot drift apart.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["normalize_block", "percentile_limits"]


def normalize_block(block: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Per-channel normalize a ``(n_time, n_freq)`` block for display.

    The arithmetic is the one this package has always used: add 1, divide
    each channel by its own mean, clip to the 5th-95th percentile, then
    rescale the whole block to ``[0, 1]``. Three things that used to be
    decided per copy are decided here, once:

    NaN handling
        The block is made finite up front (``nan``/``+inf``/``-inf`` -> 0),
        which is what ``plot_composite._coerce_float_image`` already did for
        the composite figure. The other six copies did not, and there a
        single NaN anywhere poisoned ``block.min()`` and turned the whole
        panel into an all-NaN image (an all-NaN block also raised
        ``RuntimeWarning: All-NaN slice encountered``). Those panels are
        documented to be identical to the composite panels, so they now
        agree with it instead of blanking. Blocks that are already finite
        -- every input the composite path can produce -- are untouched, so
        their output is bit-identical to before.

    dtype
        Integer, boolean and object blocks are converted to float64;
        ``block /= np.mean(...)`` used to raise ``UFuncTypeError`` on an
        integer block at six of the seven call sites. Three cast nothing,
        and the three that cast only when the dtype is object or
        non-numeric skipped an integer block, because an integer dtype is
        numeric. Only the composite figure escaped, by going through
        ``_coerce_float_image`` first. A block that is already floating
        keeps its own precision (float32 stays float32) so the rendered
        result does not move.

    Division by zero
        Both divisions are guarded, because neither guard can change a
        result that was finite before:

        * a channel whose post-increment mean is exactly 0 is divided by 1
          instead, i.e. left at its own scale. Unguarded this produced
          ``inf``/``nan`` for that channel, and the later ``block.min()``
          spread it over the entire panel.
        * a block that ``np.clip`` collapsed to a constant (flat or heavily
          quantized data) has zero range after ``block -= block.min()``, so
          the final rescale is skipped and the block stays all-zero -- a
          uniform panel at the low end of the colormap. Unguarded this was
          ``0/0`` for every pixel, i.e. an all-NaN panel plus a
          ``RuntimeWarning``.

    The block is normalized in place when its dtype allows it (every caller
    passes a private copy), and is always returned: callers must use the
    return value, since a dtype conversion cannot be done in place.

    ``None`` passes through, which is what the call sites' ``if block is not
    None`` guards used to express.
    """

    if block is None:
        return None

    a = np.asarray(block)
    if not np.issubdtype(a.dtype, np.floating):
        a = a.astype(np.float64)

    np.nan_to_num(a, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    a += 1
    channel_mean = np.mean(a, axis=0)
    channel_mean = np.where(channel_mean == 0, 1, channel_mean).astype(
        channel_mean.dtype, copy=False
    )
    a /= channel_mean

    vmin, vmax = np.nanpercentile(a, [5, 95])
    np.clip(a, vmin, vmax, out=a)

    a -= a.min()
    # `a.min()` is now exactly 0, so the original `a.max() - a.min()` is `a.max()`.
    span = a.max()
    if span > 0:
        a /= span
    return a


def percentile_limits(
    data: np.ndarray, low: float = 1.0, high: float = 99.0
) -> Tuple[float, float]:
    """Return the ``(vmin, vmax)`` percentile pair an ``imshow`` panel displays with.

    One ``nanpercentile`` pass instead of the two (four, in the
    polarization time series, which asked for the same pair again for its
    colorbar) that every call site used to make. Asking for both quantiles
    at once returns them in float64 rather than in the input's own
    precision, so on a float32 block the limits can differ from the old
    pair in the last float32 digit; that is below what a 256-entry colormap
    can resolve.
    """

    vmin, vmax = np.nanpercentile(data, [low, high])
    return vmin, vmax
