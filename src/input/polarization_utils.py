# This module provides utilities for multi-polarization data extraction.

"""Polarization extraction utilities for high-frequency pipeline.

This module provides functions to extract specific polarization products
from raw multi-polarization data, enabling the HF pipeline to analyze
multiple Stokes parameters sequentially.
"""

from __future__ import annotations
import numpy as np
import logging

logger = logging.getLogger(__name__)


def debiased_linear_polarization(q: np.ndarray, u: np.ndarray, enabled: bool = True) -> np.ndarray:
    """Return L=sqrt(Q^2+U^2) with optional first-order noise-bias removal."""
    power = q * q + u * u
    if enabled:
        q_sigma = 1.4826 * np.median(np.abs(q - np.median(q, axis=0)), axis=0)
        u_sigma = 1.4826 * np.median(np.abs(u - np.median(u, axis=0)), axis=0)
        sigma2 = 0.5 * (q_sigma * q_sigma + u_sigma * u_sigma)
        power = np.maximum(power - sigma2[np.newaxis, :], 0.0)
    return np.sqrt(np.maximum(power, 0.0)).astype(q.dtype, copy=False)


def polarization_fractions(
    raw_data: np.ndarray,
    pol_type: str,
    eps: float = 1e-6,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return L/I and V/I for IQUV data; otherwise ``(None, None)``."""
    if not has_full_polarization_data(raw_data, pol_type):
        return None, None
    i = raw_data[:, 0, :]
    q = raw_data[:, 1, :]
    u = raw_data[:, 2, :]
    v = raw_data[:, 3, :]
    try:
        from ..config import config
        debias = bool(getattr(config, "POLARIZATION_LINEAR_DEBIAS", True))
    except Exception:
        debias = True
    l = debiased_linear_polarization(q, u, enabled=debias)
    denom = np.where(np.abs(i) < eps, np.nan, i)
    return l / denom, np.abs(v) / denom


def extract_polarization_from_raw(
    raw_data: np.ndarray,
    pol_type: str,
    mode: str,
    default_index: int = 0,
) -> np.ndarray:
    """Extract a specific polarization product from raw multi-pol data.
    
    Args:
        raw_data: Raw data array (time, npol, chan) with all polarizations
        pol_type: Polarization type from header (e.g., "IQUV", "AABB")
        mode: Desired polarization mode:
            - "intensity": Stokes I (total intensity)
            - "linear": sqrt(Q^2 + U^2) (linear polarization)
            - "circular": |V| (circular polarization)
            - "pol0", "pol1", "pol2", "pol3": Specific polarization index
        default_index: Default index when IQUV not available
    
    Returns:
        Data array (time, 1, chan) with selected polarization
    """
    if raw_data.ndim != 3:
        raise ValueError(f"Expected 3D array (time, pol, chan), got shape {raw_data.shape}")
    
    npol = raw_data.shape[1]
    if npol == 1:
        return raw_data[:, 0:1, :]
    
    mode_l = (mode or "").strip().lower()
    pol_type_u = (pol_type or "").strip().upper()
    
    # Handle IQUV polarization products
    if pol_type_u == "IQUV" and npol >= 4:
        if mode_l in ("intensity", "i", "stokes_i", "intensidad"):
            return raw_data[:, 0:1, :]
        
        if mode_l in ("linear", "l", "lineal"):
            q = raw_data[:, 1, :]
            u = raw_data[:, 2, :]
            try:
                from ..config import config
                debias = bool(getattr(config, "POLARIZATION_LINEAR_DEBIAS", True))
            except Exception:
                debias = True
            l = debiased_linear_polarization(q, u, enabled=debias)
            return l[:, np.newaxis, :]
        
        if mode_l in ("circular", "v", "c"):
            # Circular polarization: |V|
            v = np.abs(raw_data[:, 3, :]).astype(raw_data.dtype, copy=False)
            return v[:, np.newaxis, :]
        
        if mode_l.startswith("pol"):
            try:
                idx = int(mode_l.replace("pol", ""))
            except Exception:
                idx = default_index
            idx = max(0, min(npol - 1, idx))
            return raw_data[:, idx:idx + 1, :]
        
        # Default to intensity
        return raw_data[:, 0:1, :]
    
    # Handle other polarization types or direct index selection
    if mode_l.startswith("pol"):
        try:
            idx = int(mode_l.replace("pol", ""))
        except Exception:
            idx = default_index
        idx = max(0, min(npol - 1, idx))
        return raw_data[:, idx:idx + 1, :]
    
    # Default: return first polarization
    return raw_data[:, default_index:default_index + 1, :]


def has_full_polarization_data(raw_data: np.ndarray, pol_type: str) -> bool:
    """Check if the raw data contains full Stokes IQUV information.
    
    Args:
        raw_data: Raw data array (time, npol, chan)
        pol_type: Polarization type from header
    
    Returns:
        True if full IQUV data is available
    """
    if raw_data.ndim != 3:
        return False
    
    npol = raw_data.shape[1]
    pol_type_u = (pol_type or "").strip().upper()
    
    return pol_type_u == "IQUV" and npol >= 4

