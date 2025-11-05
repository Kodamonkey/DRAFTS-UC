# This module provides signal-to-noise ratio analysis helpers.

"""Utilities for SNR calculation and analysis in FRB detection."""
from __future__ import annotations

                          
from typing import List, Optional, Tuple

                     
import numpy as np

                                                                 


def compute_snr_profile(
    waterfall: np.ndarray,
    off_regions: Optional[List[Tuple[int, int]]] = None
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Compute a unified PRESTO-style SNR profile from a 2D waterfall.

    Pipeline:
    - Integrate over frequency to obtain a time series
    - Detrend/normalize in blocks (RMS≈1) with the 1.148 correction
    - Apply matched filtering with boxcars normalized by ``√width``
    - Return the per-sample maximum SNR and effective sigma≈1.0

    The ``off_regions`` parameter is kept for compatibility but is unused because
    block normalization already stabilizes the RMS.

    Returns
    -------
    snr : np.ndarray
        1D SNR profile (maximum over widths) with shape ``(n_time,)``
    sigma : float
        Effective noise standard deviation (≈1.0 after normalization)
    best_width : np.ndarray
        Boxcar width (in samples) that maximizes the SNR at each sample
    """
    if waterfall is None or waterfall.size == 0:
        raise ValueError("waterfall is None or empty in compute_snr_profile")
    if waterfall.ndim != 2:
        raise ValueError("waterfall must be 2D (time, freq)")

    # 1) Integrate in frequency → time series
    timeseries = np.mean(waterfall, axis=1).astype(np.float32)

    # 2) Detrend/normalization by blocks (RMS≈1)
    ts = _detrend_normalize_by_blocks(timeseries, block_len=1000, fast=True)

    # 3) Matched filtering with PRESTO width set (trimmed to 30 by default)
    widths = [1, 2, 3, 4, 6, 9, 14, 20, 30]
    n = ts.shape[0]
    # Avoid broadcasting: constrain widths to the available length
    widths = [w for w in widths if 1 <= w <= n]
    if not widths:
        widths = [1]
    snr_max = np.full(n, -np.inf, dtype=np.float32)
    best_width = np.zeros(n, dtype=np.int32)
    for w in widths:
        if w <= 0:
            continue
        kernel = np.ones(w, dtype=np.float32) / np.sqrt(float(w))
        conv = np.convolve(ts, kernel, mode="same").astype(np.float32)
        mask = conv > snr_max
        snr_max[mask] = conv[mask]
        best_width[mask] = w

    snr_max = np.nan_to_num(snr_max, nan=-np.inf,
                            posinf=np.max(snr_max[np.isfinite(snr_max)]) if np.isfinite(snr_max).any() else 0.0)
    return snr_max, 1.0, best_width

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


def _detrend_normalize_timeseries(timeseries: np.ndarray) -> np.ndarray:
    """Approximate PRESTO-style detrend and normalize to RMS≈1.

    - Remove median (fast mode equivalent)
    - Estimate std using central 95% and apply 1.148 correction
    """
    ts = timeseries.astype(np.float32)
    ts = ts - float(np.median(ts))
    sorted_ts = np.sort(ts.copy())
    n = len(sorted_ts)
    if n >= 20:
        lo = n // 40
        hi = n - lo
    else:
        lo = 0
        hi = n
    central = sorted_ts[lo:hi]
    if central.size == 0:
        sigma = float(np.std(ts))
    else:
        sigma = float(np.sqrt((central.astype(np.float64) ** 2.0).sum() / (0.95 * n)))
        sigma *= 1.148                             
    if sigma <= 0:
        sigma = 1.0
    return ts / sigma

def _detrend_normalize_by_blocks(timeseries: np.ndarray, block_len: int = 1000, fast: bool = True) -> np.ndarray:
    """Detrend and normalize by blocks in a PRESTO style, returning RMS≈1.

    - Divide the series into blocks of length ~block_len (last block may be smaller)
    - For each block: remove trend (median if fast=True), estimate robust σ
      with ~5% clipping and apply 1.148 correction; normalize block by σ
    - Optionally, suppress anomalous blocks (σ out of range) by setting them to 0
    """
    t = timeseries.astype(np.float32, copy=False)
    n = t.shape[0]
    if n == 0:
        return t
    blen = max(32, min(block_len, n))
    out = np.empty_like(t)

    # Estimate σ per block robustly
    block_sigmas: list[float] = []
    blocks: list[tuple[int, int]] = []
    for start in range(0, n, blen):
        end = min(n, start + blen)
        block = t[start:end].copy()
        if fast:
            # Median removal (PRESTO 'fast' mode)
            med = float(np.median(block))
            block -= med
            work = block.copy()
        else:
            # Approximate linear detrend (fallback)
            x = np.linspace(0.0, 1.0, block.shape[0], dtype=np.float32)
            p = np.polyfit(x, block.astype(np.float64), deg=1)
            trend = (p[0] * x + p[1]).astype(np.float32)
            block = block - trend
            work = block.copy()

        # Robust σ by central clipping (~95%) and 1.148 correction
        work.sort()
        m = work.shape[0]
        lo = m // 40
        hi = m - lo
        central = work[lo:hi]
        if central.size == 0:
            sigma = float(np.std(work))
        else:
            sigma = float(np.sqrt((central.astype(np.float64) ** 2.0).sum() / (0.95 * m)))
            sigma *= 1.148
        if sigma <= 0 or not np.isfinite(sigma):
            sigma = 1.0
        block /= sigma
        out[start:end] = block
        block_sigmas.append(sigma)
        blocks.append((start, end))

    # Identify anomalous blocks and suppress them (similar to bad blocks filtering)
    sigmas_arr = np.asarray(block_sigmas, dtype=np.float32)
    if sigmas_arr.size >= 3:
        med = float(np.median(sigmas_arr))
        sdev = float(np.std(sigmas_arr))
        lo_th = med - 4.0 * sdev
        hi_th = med + 4.0 * sdev
        for (start, end), s in zip(blocks, sigmas_arr):
            if s < lo_th or s > hi_th:
                out[start:end] = 0.0

    return out
  
def compute_presto_matched_snr(
    waterfall: np.ndarray,
    dt_seconds: float,
    max_downfact: int = 30,
    widths: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute SNR profile emulating PRESTO (matched filtering with boxcars).

    Args:
        waterfall: 2D matrix (n_time, n_freq)
        dt_seconds: temporal resolution (s) per sample
        max_downfact: maximum downfact (bins) to use
        widths: list of boxcar widths; if None, uses default PRESTO set

    Returns:
        snr_max: maximum SNR profile per sample
        best_width: boxcar width that maximizes SNR at each sample
    """
    if waterfall is None or waterfall.size == 0:
        raise ValueError("empty waterfall in compute_presto_matched_snr")

                                             
    if waterfall.ndim != 2:
        raise ValueError("waterfall must be 2D (time, freq)")
    timeseries = np.mean(waterfall, axis=1).astype(np.float32)

                                                
    ts = _detrend_normalize_timeseries(timeseries)

                                                        
                                                                                        
                                                                                        
                                                                                   
    if widths is None:
        widths = [2, 3, 4, 6, 9, 14, 20, 30, 45, 70, 100, 150, 220, 300]
    widths = [w for w in widths if w <= max_downfact and w >= 1]
    n = ts.shape[0]
    snr_max = np.full(n, -np.inf, dtype=np.float32)
    best_width = np.zeros(n, dtype=np.int32)

                                                        
    for w in widths:
        kernel = np.ones(w, dtype=np.float32) / np.sqrt(float(w))
        conv = np.convolve(ts, kernel, mode="same").astype(np.float32)
                                   
        mask = conv > snr_max
        snr_max[mask] = conv[mask]
        best_width[mask] = w

                      
    snr_max = np.nan_to_num(snr_max, nan=-np.inf, posinf=np.max(snr_max[np.isfinite(snr_max)]) if np.isfinite(snr_max).any() else 0.0)
    return snr_max, best_width


                                                                                     


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
    
                              
    t_indices = np.arange(n_time)
    f_indices = np.arange(n_freq)
    t_grid, f_grid = np.meshgrid(t_indices, f_indices, indexing='ij')
    
                                            
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
                                                
    effective_trials = n_samples * n_trials
    
                                                      
                                      
    significance = snr_peak - np.sqrt(2 * np.log(effective_trials))
    
    return max(0, significance)
