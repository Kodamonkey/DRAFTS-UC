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
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Calcula un único perfil SNR unificado (estilo PRESTO) desde un waterfall 2D.

    Flujo:
    - Integración en frecuencia → serie temporal
    - Detrend/normalización por bloques (RMS≈1) con corrección 1.148
    - Matched filtering con boxcars normalizados por √width
    - Devuelve el perfil SNR máximo por muestra y sigma≈1.0

    Nota: el parámetro off_regions se mantiene por compatibilidad pero no se usa
    en este esquema, ya que la normalización por bloques ya estabiliza la RMS.

    Returns
    -------
    snr : np.ndarray
        Perfil SNR 1D (máximo sobre anchos) con shape (n_time,)
    sigma : float
        Desviación estándar efectiva del ruido (≈1.0 tras normalización)
    best_width : np.ndarray
        Ancho de boxcar (en muestras) que maximiza el SNR en cada muestra
    """
    if waterfall is None or waterfall.size == 0:
        raise ValueError("waterfall is None or empty in compute_snr_profile")
    if waterfall.ndim != 2:
        raise ValueError("waterfall must be 2D (time, freq)")

    # 1) Integrar en frecuencia → serie temporal
    timeseries = np.mean(waterfall, axis=1).astype(np.float32)

    # 2) Detrend/normalización por bloques (RMS≈1)
    ts = _detrend_normalize_by_blocks(timeseries, block_len=1000, fast=True)

    # 3) Matched filtering con set de anchos PRESTO (recortado a 30 por defecto)
    widths = [1, 2, 3, 4, 6, 9, 14, 20, 30]
    n = ts.shape[0]
    # Evitar broadcasting: limitar anchos a la longitud disponible
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


def _detrend_normalize_by_blocks(timeseries: np.ndarray, block_len: int = 1000, fast: bool = True) -> np.ndarray:
    """Detrend y normaliza por bloques al estilo PRESTO, devolviendo RMS≈1.

    - Divide la serie en bloques de longitud ~block_len (último bloque puede ser menor)
    - Para cada bloque: remueve tendencia (mediana si fast=True), estima σ robusta
      con recorte ~5% y aplica corrección 1.148; normaliza el bloque por σ
    - Opcionalmente, suprime bloques anómalos (σ fuera de rango) poniéndolos en 0
    """
    t = timeseries.astype(np.float32, copy=False)
    n = t.shape[0]
    if n == 0:
        return t
    blen = max(32, min(block_len, n))
    out = np.empty_like(t)

    # Estimar σ por bloque de forma robusta
    block_sigmas: list[float] = []
    blocks: list[tuple[int, int]] = []
    for start in range(0, n, blen):
        end = min(n, start + blen)
        block = t[start:end].copy()
        if fast:
            # Remoción de mediana (modo 'fast' de PRESTO)
            med = float(np.median(block))
            block -= med
            work = block.copy()
        else:
            # Detrend lineal aproximado (fallback)
            x = np.linspace(0.0, 1.0, block.shape[0], dtype=np.float32)
            p = np.polyfit(x, block.astype(np.float64), deg=1)
            trend = (p[0] * x + p[1]).astype(np.float32)
            block = block - trend
            work = block.copy()

        # σ robusta por recorte central (~95%) y corrección 1.148
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

    # Identificar bloques anómalos y suprimirlos (similar al filtrado de bad blocks)
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
