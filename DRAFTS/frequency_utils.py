"""Utility helpers for frequency axis handling."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def normalize_frequency_order(freq: np.ndarray, step: float | None = None) -> Tuple[np.ndarray, bool]:
    """Return frequencies in ascending order.

    Parameters
    ----------
    freq : np.ndarray
        Frequency values.
    step : float, optional
        Frequency increment per channel. If negative, the array will be
        reversed automatically.

    Returns
    -------
    Tuple[np.ndarray, bool]
        Tuple with the (possibly reversed) frequency array and a flag
        indicating whether a reversal occurred.
    """
    freq = np.asarray(freq, dtype=float)
    inverted = False

    if step is not None:
        try:
            step_val = float(step)
            if step_val < 0:
                inverted = True
        except Exception:
            pass

    if not inverted and len(freq) > 1 and freq[0] > freq[-1]:
        inverted = True

    if inverted:
        freq = freq[::-1]

    return freq, inverted
