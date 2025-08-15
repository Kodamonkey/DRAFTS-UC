"""Utilities for SNR calculation and analysis in FRB detection."""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional

# Nota: este módulo no requiere configuración global del pipeline


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
    # Verificación básica para arrays válidos
    if waterfall is None or waterfall.size == 0:
        raise ValueError("waterfall is None or empty in compute_snr_profile")
    if waterfall.ndim < 2:
        raise ValueError(f"waterfall must be 2D, got {waterfall.ndim}D array")
    
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


def estimate_sigma_robust(data: np.ndarray) -> float:
    """
    Estimate noise standard deviation using robust method with central trimming.
    
    This method applies detrending, trims the central 95% of data, and applies
    correction factors for accurate noise estimation in presence of signals.
    
    Parameters
    ----------
    data : np.ndarray
        1D data array
        
    Returns
    -------
    float
        Estimated standard deviation
    """
    data = data.astype(np.float32)
    
    # 1. DETRENDING: Remove median to eliminate baseline drift
    median_val = float(np.median(data))
    data_detrended = data - median_val
    
    n = len(data_detrended)
    if n >= 20:
        # 2. CENTRAL TRIMMING: Use central 95% of data (trim 2.5% from each end)
        lo = n // 40      # 2.5% from bottom
        hi = n - lo       # 2.5% from top
        central = data_detrended[lo:hi]
        
        if central.size > 0:
            # 3. ROBUST SIGMA CALCULATION: RMS using only central data
            # This avoids contamination from signal peaks
            central_squared = central.astype(np.float64) ** 2.0
            sigma = float(np.sqrt(central_squared.sum() / (0.95 * n)))
            
            # 4. CORRECTION FACTOR: Compensate for 5% trimming
            try:
                from .. import config
                correction_factor = getattr(config, 'SNR_NOISE_CORRECTION_FACTOR', 1.148)
            except ImportError:
                correction_factor = 1.148  # Default correction factor for 5% trimming
            
            sigma *= correction_factor
        else:
            # Fallback: use standard deviation of detrended data
            sigma = float(np.std(data_detrended, ddof=1))
    else:
        # For small datasets, use standard deviation of detrended data
        sigma = float(np.std(data_detrended, ddof=1))
    
    return max(sigma, 1e-10)  # Avoid division by zero


def compute_snr_profile_enhanced(
    waterfall: np.ndarray, 
    time_reso: float,
    off_regions: Optional[List[Tuple[int, int]]] = None,
    use_enhanced: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Calculate enhanced SNR profile using matched filtering with multiple boxcar widths.
    
    This method applies matched filtering with different temporal widths to
    maximize signal detection across various pulse durations.
    
    Parameters
    ----------
    waterfall : np.ndarray
        2D array with shape (n_time, n_freq)
    time_reso : float
        Time resolution in seconds
    off_regions : List[Tuple[int, int]], optional
        List of (start, end) bin ranges for noise estimation
        
    Returns
    -------
    snr : np.ndarray
        1D SNR profile with maximum SNR across all boxcar widths
    sigma : float
        Estimated noise standard deviation
    """
    if waterfall is None or waterfall.size == 0:
        raise ValueError("waterfall is None or empty in compute_snr_profile_enhanced")
    if waterfall.ndim < 2:
        raise ValueError(f"waterfall must be 2D, got {waterfall.ndim}D array")
    
    # Integrate over frequency axis
    profile = np.mean(waterfall, axis=1)
    
    # Estimate noise using robust method
    if off_regions is None:
        sigma = estimate_sigma_robust(profile)
        mean_level = np.median(profile)
    else:
        # Use specified off-pulse regions
        off_data = []
        for start, end in off_regions:
            start_idx = max(0, start if start >= 0 else len(profile) + start)
            end_idx = min(len(profile), end if end >= 0 else len(profile) + end)
            if start_idx < end_idx:
                off_data.extend(profile[start_idx:end_idx])
        
        if off_data:
            off_data = np.array(off_data)
            sigma = np.std(off_data, ddof=1)
            mean_level = np.mean(off_data)
        else:
            sigma = estimate_sigma_robust(profile)
            mean_level = np.median(profile)
    
    # Normalize profile
    profile_normalized = (profile - mean_level) / (sigma + 1e-10)
    
    # Apply matched filtering with different boxcar widths if enhanced mode is enabled
    if use_enhanced:
        try:
            from .. import config
            widths = getattr(config, 'SNR_BOXCAR_WIDTHS', [1, 2, 3, 4, 6, 9, 14, 20, 30])
        except ImportError:
            widths = [1, 2, 3, 4, 6, 9, 14, 20, 30]  # Optimal widths for FRB detection
        
        snr_max = np.full(len(profile), -np.inf)
        
        for w in widths:
            if w <= len(profile):
                # Create normalized kernel
                kernel = np.ones(w, dtype=np.float32) / np.sqrt(float(w))
                # Apply convolution
                conv = np.convolve(profile_normalized, kernel, mode="same")
                # Update maximum SNR
                mask = conv > snr_max
                snr_max[mask] = conv[mask]
    else:
        # Use simple SNR without matched filtering
        snr_max = profile_normalized
    
    # Ensure finite values
    snr_max = np.nan_to_num(snr_max, nan=-np.inf, posinf=np.max(snr_max[np.isfinite(snr_max)]) if np.isfinite(snr_max).any() else 0.0)
    
    return snr_max, sigma


def compute_snr_profile_corrected(
    waterfall: np.ndarray, 
    time_reso: float,
    off_regions: Optional[List[Tuple[int, int]]] = None
) -> Tuple[np.ndarray, float]:
    """
    Calculate SNR profile using corrected method with proper detrending and noise estimation.
    
    This method implements the correct approach for signal detection:
    1. Detrending to remove baseline drift
    2. Robust noise estimation using central trimming
    3. Proper normalization for SNR calculation
    4. Optional matched filtering for enhanced detection
    
    Parameters
    ----------
    waterfall : np.ndarray
        2D array with shape (n_time, n_freq)
    time_reso : float
        Time resolution in seconds
    off_regions : List[Tuple[int, int]], optional
        List of (start, end) bin ranges for noise estimation
        
    Returns
    -------
    snr : np.ndarray
        1D SNR profile with maximum SNR across all boxcar widths
    sigma : float
        Estimated noise standard deviation
    """
    if waterfall is None or waterfall.size == 0:
        raise ValueError("waterfall is None or empty in compute_snr_profile_corrected")
    if waterfall.ndim < 2:
        raise ValueError(f"waterfall must be 2D, got {waterfall.ndim}D array")
    
    # 1. INTEGRATION: Integrate over frequency axis
    profile = np.mean(waterfall, axis=1)
    
    # 2. DETRENDING: Remove median to eliminate baseline drift
    median_val = float(np.median(profile))
    profile_detrended = profile - median_val
    
    # 3. ROBUST NOISE ESTIMATION: Use central trimming method
    sigma = estimate_sigma_robust(profile)
    
    # 4. NORMALIZATION: Normalize to unit RMS
    if sigma > 0:
        profile_normalized = profile_detrended / sigma
    else:
        profile_normalized = profile_detrended
        sigma = 1.0
    
    # 5. MATCHED FILTERING: Apply with different boxcar widths for enhanced detection
    try:
        from .. import config
        use_enhanced = getattr(config, 'ENHANCED_SNR_CALCULATION', True)
    except ImportError:
        use_enhanced = True
    
    if use_enhanced:
        try:
            widths = getattr(config, 'SNR_BOXCAR_WIDTHS', [1, 2, 3, 4, 6, 9, 14, 20, 30])
        except ImportError:
            widths = [1, 2, 3, 4, 6, 9, 14, 20, 30]  # Optimal widths for FRB detection
        
        snr_max = np.full(len(profile_normalized), -np.inf)
        
        for w in widths:
            if w <= len(profile_normalized):
                # Create normalized kernel
                kernel = np.ones(w, dtype=np.float32) / np.sqrt(float(w))
                # Apply convolution
                conv = np.convolve(profile_normalized, kernel, mode="same")
                # Update maximum SNR
                mask = conv > snr_max
                snr_max[mask] = conv[mask]
        
        # Ensure finite values
        snr_max = np.nan_to_num(snr_max, nan=-np.inf, posinf=np.max(snr_max[np.isfinite(snr_max)]) if np.isfinite(snr_max).any() else 0.0)
        
        return snr_max, sigma
    else:
        # Return simple normalized SNR without matched filtering
        return profile_normalized, sigma


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


def find_peak_enhanced(snr_profile: np.ndarray, time_axis: np.ndarray, 
                       min_snr_threshold: float = 3.0) -> Tuple[float, float, int]:
    """
    Find peak with enhanced timing precision and validation.
    
    This method applies SNR thresholding and can optionally use
    interpolation for sub-sample timing precision.
    
    Parameters
    ----------
    snr_profile : np.ndarray
        SNR profile array
    time_axis : np.ndarray
        Time axis corresponding to SNR profile
    min_snr_threshold : float
        Minimum SNR threshold for valid peaks
        
    Returns
    -------
    peak_snr : float
        Peak SNR value
    peak_time : float
        Peak time with enhanced precision
    peak_idx : int
        Peak index
    """
    # Apply SNR threshold
    valid_mask = snr_profile >= min_snr_threshold
    
    if not np.any(valid_mask):
        return 0.0, time_axis[0], 0
    
    # Find global peak
    peak_idx = np.argmax(snr_profile)
    peak_snr = snr_profile[peak_idx]
    peak_time = time_axis[peak_idx]
    
    # Enhanced timing precision using quadratic interpolation
    if peak_idx > 0 and peak_idx < len(snr_profile) - 1:
        y1, y2, y3 = snr_profile[peak_idx-1:peak_idx+2]
        if y2 > y1 and y2 > y3:
            # Real peak, calculate sub-sample offset
            offset = 0.5 * (y1 - y3) / (y1 - 2*y2 + y3)
            dt = time_axis[1] - time_axis[0]
            peak_time += offset * dt
    
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
        sigma *= 1.148  # corrección por recorte 5%
    if sigma <= 0:
        sigma = 1.0
    return ts / sigma


def compute_presto_matched_snr(
    waterfall: np.ndarray,
    dt_seconds: float,
    max_downfact: int = 30,
    widths: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute SNR profile emulando PRESTO (matched filtering con boxcars).

    Args:
        waterfall: matriz 2D (n_time, n_freq)
        dt_seconds: resolución temporal (s) por muestra
        max_downfact: downfact máximo (bins) a usar
        widths: lista de anchos de boxcar; si None, usa set por defecto PRESTO

    Returns:
        snr_max: perfil SNR máximo por muestra
        best_width: ancho de boxcar que maximiza SNR en cada muestra
    """
    if waterfall is None or waterfall.size == 0:
        raise ValueError("waterfall vacío en compute_presto_matched_snr")

    # Integrar en frecuencia → serie temporal
    if waterfall.ndim != 2:
        raise ValueError("waterfall debe ser 2D (tiempo, freq)")
    timeseries = np.mean(waterfall, axis=1).astype(np.float32)

    # Detrend + normalización aproximando PRESTO
    ts = _detrend_normalize_timeseries(timeseries)

    # Conjunto de anchos por defecto (subset de PRESTO)
    if widths is None:
        widths = [1, 2, 3, 4, 6, 9, 14, 20, 30]
    widths = [w for w in widths if w <= max_downfact and w >= 1]
    n = ts.shape[0]
    snr_max = np.full(n, -np.inf, dtype=np.float32)
    best_width = np.zeros(n, dtype=np.int32)

    # Convolución con kernel normalizado por sqrt(width)
    for w in widths:
        kernel = np.ones(w, dtype=np.float32) / np.sqrt(float(w))
        conv = np.convolve(ts, kernel, mode="same").astype(np.float32)
        # actualizar máximo y ancho
        mask = conv > snr_max
        snr_max[mask] = conv[mask]
        best_width[mask] = w

    # Asegurar finitos
    snr_max = np.nan_to_num(snr_max, nan=-np.inf, posinf=np.max(snr_max[np.isfinite(snr_max)]) if np.isfinite(snr_max).any() else 0.0)
    return snr_max, best_width


## Nota: se eliminaron utilidades de regiones off-pulse no utilizadas por el pipeline


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
