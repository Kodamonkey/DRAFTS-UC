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
            # Linear polarization: sqrt(Q^2 + U^2)
            q = raw_data[:, 1, :]
            u = raw_data[:, 2, :]
            l = np.sqrt(np.maximum(0.0, q * q + u * u)).astype(raw_data.dtype, copy=False)
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

