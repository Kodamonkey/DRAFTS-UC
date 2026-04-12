"""Scientific search metrics for dispersion and candidate ranking."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

K_DM_MS = 4.148808e3  # Dispersion constant: gives SECONDS when freq in MHz, DM in pc/cm³


def dispersion_delay_ms(dm: float, freq_low_mhz: float, freq_high_mhz: float) -> float:
    """Delay across a band in milliseconds."""
    if freq_low_mhz <= 0 or freq_high_mhz <= 0:
        return 0.0
    nu_lo = min(float(freq_low_mhz), float(freq_high_mhz))
    nu_hi = max(float(freq_low_mhz), float(freq_high_mhz))
    return K_DM_MS * float(dm) * (nu_lo ** -2 - nu_hi ** -2) * 1000.0


def dm_step_for_smearing(max_smearing_ms: float, freq_low_mhz: float, freq_high_mhz: float) -> float:
    """DM spacing whose band-delay error is at most ``max_smearing_ms``."""
    denom = dispersion_delay_ms(1.0, freq_low_mhz, freq_high_mhz)
    if denom <= 0 or not np.isfinite(denom):
        return 1.0
    return max(float(max_smearing_ms) / denom, 1e-6)


def estimate_dm_uncertainty(dm_values: Iterable[float], peak_index: int) -> float:
    """Half local DM spacing around a peak; useful as conservative DM error."""
    vals = np.asarray(list(dm_values), dtype=np.float64)
    if vals.size <= 1:
        return 0.0
    idx = int(max(0, min(vals.size - 1, peak_index)))
    if idx == 0:
        return abs(vals[1] - vals[0]) / 2.0
    if idx == vals.size - 1:
        return abs(vals[-1] - vals[-2]) / 2.0
    return abs(vals[idx + 1] - vals[idx - 1]) / 2.0


def post_trials_sigma(snr: float, n_trials: int, mode: str = "gaussian_extreme") -> float:
    """Approximate post-trials significance in sigma units."""
    snr_f = float(snr)
    trials = max(1, int(n_trials))
    mode_l = (mode or "none").lower()
    if mode_l == "none" or trials <= 1:
        return max(0.0, snr_f)
    if mode_l in {"bonferroni", "gaussian_extreme"}:
        penalty = math.sqrt(2.0 * math.log(trials))
        return max(0.0, snr_f - penalty)
    return max(0.0, snr_f)


def physical_consistency_score(
    snr_post_trials: float,
    snr_pre_dedisp: float | None,
    snr_post_dedisp: float | None,
    dm_status: str,
    linear_fraction: float | None = None,
) -> float:
    """Compact 0..1 score from significance and simple physical checks."""
    score = min(1.0, max(0.0, float(snr_post_trials) / 8.0))
    if snr_pre_dedisp is not None and snr_post_dedisp is not None and snr_pre_dedisp > 0:
        gain = float(snr_post_dedisp) / max(float(snr_pre_dedisp), 1e-6)
        score *= min(1.25, max(0.5, gain)) / 1.25
    if dm_status.startswith("unresolved"):
        score *= 0.85
    if linear_fraction is not None and np.isfinite(linear_fraction):
        score *= min(1.1, max(0.8, 0.9 + 0.2 * min(abs(float(linear_fraction)), 1.0)))
    return float(min(1.0, max(0.0, score)))

