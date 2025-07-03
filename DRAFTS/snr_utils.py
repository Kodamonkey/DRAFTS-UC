"""Utilities for SNR calculation and analysis in FRB detection."""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional

from . import config


def compute_snr_profile(
    waterfall: np.ndarray, 
    off_regions: Optional[List[Tuple[int, int]]] = None
) -> Tuple[np.ndarray, float]:
    """
    Calculate SNR profile from a waterfall array.
    
    Parameters
    ----------
    waterfall : np.ndarray
        2D array with shape (n_time, n_freq)
    off_regions : List[Tuple[int, int]], optional
        List of (start, end) bin ranges to use for noise estimation.
        If None, uses IQR method over the entire profile.
    
    Returns
    -------
    snr : np.ndarray
        1D SNR profile with shape (n_time,)
    sigma : float
        Estimated noise standard deviation
    """
    # Integrate over frequency axis
    profile = np.mean(waterfall, axis=1)
    
    if off_regions is None:
        # Use IQR method for robust noise estimation
        sigma = estimate_sigma_iqr(profile)
        mean_level = np.median(profile)
    else:
        # Use specified off-pulse regions
        off_data = []
        for start, end in off_regions:
            # Handle negative indices and bounds
            start_idx = max(0, start if start >= 0 else len(profile) + start)
            end_idx = min(len(profile), end if end >= 0 else len(profile) + end)
            if start_idx < end_idx:
                off_data.extend(profile[start_idx:end_idx])
        
        if off_data:
            off_data = np.array(off_data)
            sigma = np.std(off_data, ddof=1)
            mean_level = np.mean(off_data)
        else:
            # Fallback to IQR if no valid off regions
            sigma = estimate_sigma_iqr(profile)
            mean_level = np.median(profile)
    
    # Calculate SNR
    snr = (profile - mean_level) / (sigma + 1e-10)  # Add small epsilon to avoid division by zero
    
    return snr, sigma


def estimate_sigma_iqr(data: np.ndarray) -> float:
    """
    Estimate noise standard deviation using the Interquartile Range method.
    
    This method is robust to outliers (like FRB pulses).
    
    Parameters
    ----------
    data : np.ndarray
        1D data array
        
    Returns
    -------
    float
        Estimated standard deviation
    """
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    # For Gaussian distribution: sigma ≈ IQR / 1.349
    return iqr / 1.349


def find_snr_peak(snr: np.ndarray, time_axis: Optional[np.ndarray] = None) -> Tuple[float, float, int]:
    """
    Find the peak SNR value and its location.
    
    Parameters
    ----------
    snr : np.ndarray
        SNR profile
    time_axis : np.ndarray, optional
        Time axis values. If None, uses sample indices.
        
    Returns
    -------
    peak_snr : float
        Maximum SNR value
    peak_time : float
        Time of peak (or sample index if time_axis is None)
    peak_idx : int
        Sample index of peak
    """
    peak_idx = np.argmax(snr)
    peak_snr = snr[peak_idx]
    peak_time = time_axis[peak_idx] if time_axis is not None else peak_idx
    
    return peak_snr, peak_time, peak_idx


def create_snr_regions_around_peak(
    peak_idx: int, 
    profile_length: int, 
    region_width: int = 50
) -> List[Tuple[int, int]]:
    """
    Create off-pulse regions around a detected peak.
    
    Parameters
    ----------
    peak_idx : int
        Index of the detected peak
    profile_length : int
        Total length of the profile
    region_width : int
        Width of each off-pulse region
        
    Returns
    -------
    List[Tuple[int, int]]
        List of (start, end) indices for off-pulse regions
    """
    regions = []
    
    # Left region
    left_end = peak_idx - region_width
    left_start = left_end - region_width
    if left_start >= 0:
        regions.append((left_start, left_end))
    
    # Right region  
    right_start = peak_idx + region_width
    right_end = right_start + region_width
    if right_end < profile_length:
        regions.append((right_start, right_end))
        
    # Far left region if space allows
    if left_start >= region_width:
        far_left_end = left_start - region_width // 2
        far_left_start = far_left_end - region_width
        if far_left_start >= 0:
            regions.append((far_left_start, far_left_end))
    
    return regions


def inject_synthetic_frb(
    waterfall: np.ndarray,
    peak_time_idx: int,
    peak_freq_idx: int,
    amplitude: float,
    width_time: int = 5,
    width_freq: int = 20
) -> np.ndarray:
    """
    Inject a synthetic FRB pulse into a waterfall for testing.
    
    Parameters
    ----------
    waterfall : np.ndarray
        Background waterfall with shape (n_time, n_freq)
    peak_time_idx : int
        Time index for pulse peak
    peak_freq_idx : int
        Frequency index for pulse peak
    amplitude : float
        Peak amplitude of the pulse
    width_time : int
        Temporal width (samples)
    width_freq : int
        Spectral width (channels)
        
    Returns
    -------
    np.ndarray
        Waterfall with injected pulse
    """
    waterfall_with_pulse = waterfall.copy()
    n_time, n_freq = waterfall.shape
    
    # Create 2D Gaussian pulse
    t_indices = np.arange(n_time)
    f_indices = np.arange(n_freq)
    t_grid, f_grid = np.meshgrid(t_indices, f_indices, indexing='ij')
    
    # Gaussian profile in time and frequency
    pulse = amplitude * np.exp(
        -0.5 * ((t_grid - peak_time_idx) / width_time) ** 2
        -0.5 * ((f_grid - peak_freq_idx) / width_freq) ** 2
    )
    
    waterfall_with_pulse += pulse
    return waterfall_with_pulse


def compute_detection_significance(
    snr_peak: float, 
    n_samples: int, 
    n_trials: int = 1
) -> float:
    """
    Compute the statistical significance of a detection.
    
    Parameters
    ----------
    snr_peak : float
        Peak SNR value
    n_samples : int
        Number of independent samples searched
    n_trials : int
        Number of independent trials (e.g., DM trials)
        
    Returns
    -------
    float
        Significance level (number of sigma)
    """
    # Bonferroni correction for multiple testing
    effective_trials = n_samples * n_trials
    
    # For Gaussian statistics, convert to significance
    # This is a simplified calculation
    significance = snr_peak - np.sqrt(2 * np.log(effective_trials))
    
    return max(0, significance)
