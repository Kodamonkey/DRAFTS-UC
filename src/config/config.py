"""
System configuration for the FRB detection pipeline
===================================================

This file contains system configurations that are set automatically or are
specific to the internal operation of the pipeline.

IMPORTANT:
- Do NOT modify this file directly
- To configure user parameters, edit config.yaml
- This file maintains compatibility with the existing codebase
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

# Try to import torch (optional for CPU-only mode)
try:
    import torch
except ImportError:
    torch = None

# ==============================================================================
# USER CONFIGURATION IMPORT (from config.yaml via user_config.py)
# ==============================================================================

_config_injected: bool = False

# Import user configuration from user_config.py (which reads config.yaml)
try:
    from .user_config import (
        DATA_DIR,
        DEBUG_FREQUENCY_ORDER,
        DM_max,
        DM_min,
        DOWN_FREQ_RATE,
        DOWN_TIME_RATE,
        DET_PROB,
        CLASS_PROB,
        CLASS_PROB_LINEAR,
        FORCE_PLOTS,
        FRB_TARGETS,
        RESULTS_DIR,
        SLICE_DURATION_MS,
        SNR_THRESH,
        SNR_THRESH_LINEAR,
        USE_MULTI_BAND,
        SAVE_ONLY_BURST,
        AUTO_HIGH_FREQ_PIPELINE,
        HIGH_FREQ_THRESHOLD_MHZ,
        ENABLE_LINEAR_VALIDATION,
        ENABLE_INTENSITY_CLASSIFICATION,
        ENABLE_LINEAR_CLASSIFICATION,
        POLARIZATION_MODE,
        POLARIZATION_INDEX,
        MAX_CHUNK_SAMPLES,
        MAX_RAM_FRACTION_USER,
        MAX_DM_CUBE_SIZE_GB,
        DM_CHUNKING_THRESHOLD_GB_USER,
        MEMORY_OVERHEAD_FACTOR_USER,
    )
except ImportError:
    try:
        from user_config import (
            DATA_DIR,
            DEBUG_FREQUENCY_ORDER,
            DM_max,
            DM_min,
            DOWN_FREQ_RATE,
            DOWN_TIME_RATE,
            DET_PROB,
            CLASS_PROB,
            CLASS_PROB_LINEAR,
            FORCE_PLOTS,
            FRB_TARGETS,
            RESULTS_DIR,
            SLICE_DURATION_MS,
            SNR_THRESH,
            SNR_THRESH_LINEAR,
            USE_MULTI_BAND,
            SAVE_ONLY_BURST,
            AUTO_HIGH_FREQ_PIPELINE,
            HIGH_FREQ_THRESHOLD_MHZ,
            ENABLE_LINEAR_VALIDATION,
            ENABLE_INTENSITY_CLASSIFICATION,
            ENABLE_LINEAR_CLASSIFICATION,
            POLARIZATION_MODE,
            POLARIZATION_INDEX,
            MAX_CHUNK_SAMPLES,
            MAX_RAM_FRACTION_USER,
            MAX_DM_CUBE_SIZE_GB,
            DM_CHUNKING_THRESHOLD_GB_USER,
            MEMORY_OVERHEAD_FACTOR_USER,
        )
    except ImportError as e:
        raise ImportError(
            "Could not import user_config. Ensure config.yaml exists in the "
            "project root directory and run the pipeline from the project root. "
            f"Original error: {e}"
        ) from e

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================
              
MODEL_NAME = "resnet18"                                                           
MODEL_PATH = Path(__file__).parent.parent / "models" / f"cent_{MODEL_NAME}.pth"                            
                                
CLASS_MODEL_NAME = "resnet18"                                                         
CLASS_MODEL_PATH = Path(__file__).parent.parent / "models" / f"class_{CLASS_MODEL_NAME}.pth"                  
                   
# Device configuration (GPU/CPU/MPS)
if torch is not None:
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
else:
    DEVICE = "cpu"
                                                   
# ==============================================================================
# DATA PARAMETERS (set dynamically from file headers)
# ==============================================================================

FREQ: np.ndarray | None = None  # Frequency array
FREQ_RESO: int = 0               # Frequency resolution
TIME_RESO: float = 0.0           # Time resolution
FILE_LENG: int = 0               # File length in samples
DATA_NEEDS_REVERSAL: bool = False  # Frequency ordering flag

# ==============================================================================
# SLICE CONFIGURATION
# ==============================================================================

SLICE_LEN: int = 512              # Temporal slice length (calculated from SLICE_DURATION_MS)
SLICE_LEN_MIN: int = 32           # Minimum allowed slice length
SLICE_LEN_MAX: int = 2048         # Maximum allowed slice length
MAX_SLICE_COUNT: int = 5000       # Maximum number of slices
TIME_TOL_MS: float = 0.1          # Time tolerance for slice alignment (ms)

# ==============================================================================
# MEMORY AND CHUNKING CONFIGURATION
# ==============================================================================

USE_PLANNED_CHUNKING: bool = True       # Enable intelligent chunking
MAX_SAMPLES_LIMIT: int = 10_000_000     # Maximum samples to process at once
MAX_CHUNK_BYTES: int | None = None      # Maximum chunk size in bytes (None = auto)
MAX_RAM_FRACTION: float = MAX_RAM_FRACTION_USER  # Fraction of available RAM to use (from config.yaml)
OVERHEAD_FACTOR: float = MEMORY_OVERHEAD_FACTOR_USER  # Memory overhead factor for safety (from config.yaml)

# Performance limits (loaded from config.yaml)
# These will be overridden by the imported values from user_config.py
# MAX_DM_CUBE_SIZE_GB is imported from user_config.py above
DM_CHUNKING_THRESHOLD_GB: float = DM_CHUNKING_THRESHOLD_GB_USER  # DM chunking threshold (from config.yaml)

# ==============================================================================
# DM (DISPERSION MEASURE) CONFIGURATION
# ==============================================================================

# Adaptive DM range
DM_RANGE_ADAPTIVE: bool = False                                                                  
DM_RANGE_MIN_WIDTH: float = 80.0                                                  
DM_RANGE_MAX_WIDTH: float = 300.0                                                 
DM_RANGE_FACTOR: float = 0.3
DM_DYNAMIC_RANGE_ENABLE: bool = False
                                         
# DM plotting ranges
DM_PLOT_MARGIN_FACTOR: float = 0.25                                                    
DM_PLOT_MIN_RANGE: float = 120.0                                              
DM_PLOT_MAX_RANGE: float = 400.0                                              
DM_PLOT_DEFAULT_RANGE: float = 250.0                                          
DM_RANGE_DEFAULT_VISUALIZATION: str = "detailed"                                     
                               
# ==============================================================================
# SNR AND VISUALIZATION CONFIGURATION
# ==============================================================================

# SNR off-pulse regions (for noise estimation)
SNR_OFF_REGIONS = [(-250, -150), (-100, -50), (50, 100), (150, 250)]                       
SNR_HIGHLIGHT_COLOR = "red"                                                  
SNR_SHOW_PEAK_LINES: bool = False                                                              
SNR_COLORMAP = "viridis"                                                     

# ==============================================================================
# PREPROCESSING CONFIGURATION
# ==============================================================================

PREWHITEN_BEFORE_DM: bool = True   # Apply prewhitening before dedispersion
SHADE_INVALID_TAIL: bool = True    # Shade invalid tail regions in plots

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================
                                      
LOG_LEVEL: str = "INFO"                                                                    
LOG_COLORS: bool = True                                                 
LOG_FILE: bool = False                                               
GPU_VERBOSE: bool = False                                                       
SHOW_PROGRESS: bool = True                                              
                                                                               
# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

_KNOWN_CONFIG_KEYS = {
    "DATA_DIR", "RESULTS_DIR", "FRB_TARGETS",
    "SLICE_DURATION_MS", "DOWN_FREQ_RATE", "DOWN_TIME_RATE",
    "DM_min", "DM_max", "DET_PROB", "CLASS_PROB", "CLASS_PROB_LINEAR",
    "SNR_THRESH", "SNR_THRESH_LINEAR", "USE_MULTI_BAND", "SAVE_ONLY_BURST",
    "AUTO_HIGH_FREQ_PIPELINE", "HIGH_FREQ_THRESHOLD_MHZ",
    "ENABLE_LINEAR_VALIDATION", "ENABLE_INTENSITY_CLASSIFICATION",
    "ENABLE_LINEAR_CLASSIFICATION", "POLARIZATION_MODE", "POLARIZATION_INDEX",
    "DEBUG_FREQUENCY_ORDER", "FORCE_PLOTS",
    "MAX_CHUNK_SAMPLES", "MAX_RAM_FRACTION", "MAX_DM_CUBE_SIZE_GB",
    "DM_CHUNKING_THRESHOLD_GB", "OVERHEAD_FACTOR",
    "FREQ", "FREQ_RESO", "TIME_RESO", "FILE_LENG", "DATA_NEEDS_REVERSAL",
    "SLICE_LEN", "DEVICE", "TSTART_MJD", "TSTART_MJD_CORR",
    "_hardware_profile",
}


def inject_config(config_dict: dict):
    """Inject configuration from external source (CLI overrides).

    Only keys in ``_KNOWN_CONFIG_KEYS`` are accepted; unknown keys raise
    ``ValueError`` to prevent silent misconfiguration.
    """
    import logging
    import sys
    global _config_injected

    _logger = logging.getLogger(__name__)
    current_module = sys.modules[__name__]

    for key, value in config_dict.items():
        if key not in _KNOWN_CONFIG_KEYS:
            _logger.warning("Unknown config key ignored: %s", key)
            continue
        setattr(current_module, key, value)

    _config_injected = True


def get_band_configs():
    """Return band configuration tuples based on USE_MULTI_BAND.
    
    Returns:
        list: List of (band_id, band_name, band_label) tuples
    """
    return [
        (0, "fullband", "Full Band"),
        (1, "lowband", "Low Band"),
        (2, "highband", "High Band"),
    ] if USE_MULTI_BAND else [(0, "fullband", "Full Band")]
                                                                               

