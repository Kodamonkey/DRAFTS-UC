from __future__ import annotations

import numpy as np


def compute_snr(slice_band: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """Compute the classical signal-to-noise ratio for a region of ``slice_band``.

    Parameters
    ----------
    slice_band : np.ndarray
        2D array representing the dedispersed image.
    box : tuple[int, int, int, int]
        Bounding box specified as (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = map(int, box)
    box_data = slice_band[y1:y2, x1:x2]
    if box_data.size == 0:
        return 0.0
    signal = box_data.mean()
    noise = np.median(slice_band)
    std = slice_band.std(ddof=1)
    return (signal - noise) / (std + 1e-6)
