"""Expose SNR utility helpers for tests."""
from .analysis.snr_utils import (
    compute_snr_profile,
    find_snr_peak,
    inject_synthetic_frb,
    estimate_sigma_iqr,
    create_snr_regions_around_peak,
    compute_detection_significance,
)

__all__ = [
    "compute_snr_profile",
    "find_snr_peak",
    "inject_synthetic_frb",
    "estimate_sigma_iqr",
    "create_snr_regions_around_peak",
    "compute_detection_significance",
]
