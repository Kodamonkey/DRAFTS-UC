"""Utilities for SNR calculation and analysis in FRB detection."""
from __future__ import annotations

# Standard library imports
from typing import List, Optional, Tuple

# Third-party imports
import numpy as np

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

    # Conjunto de anchos por defecto alineado con PRESTO
    # PRESTO usa: [1 (implícito), 2, 3, 4, 6, 9, 14, 20, 30, 45, 70, 100, 150, 220, 300]
    # En su búsqueda primero umbraliza sin suavizar (equiv. a 1) y luego aplica boxcars.
    # Aquí incluimos directamente el set completo y lo recortamos por max_downfact.
    if widths is None:
        widths = [2, 3, 4, 6, 9, 14, 20, 30, 45, 70, 100, 150, 220, 300]
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
