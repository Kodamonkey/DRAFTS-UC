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

# Global variables for configuration injection
_injected_config: dict = {}
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
        FORCE_PLOTS,
        FRB_TARGETS,
        RESULTS_DIR,
        SLICE_DURATION_MS,
        SNR_THRESH,
        USE_MULTI_BAND,
        SAVE_ONLY_BURST,
        AUTO_HIGH_FREQ_PIPELINE,
        HIGH_FREQ_THRESHOLD_MHZ,
        ENABLE_LINEAR_VALIDATION,
        POLARIZATION_MODE,
        POLARIZATION_INDEX,
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
            FORCE_PLOTS,
            FRB_TARGETS,
            RESULTS_DIR,
            SLICE_DURATION_MS,
            SNR_THRESH,
            USE_MULTI_BAND,
            SAVE_ONLY_BURST,
            AUTO_HIGH_FREQ_PIPELINE,
            HIGH_FREQ_THRESHOLD_MHZ,
            ENABLE_LINEAR_VALIDATION,
            POLARIZATION_MODE,
            POLARIZATION_INDEX,
        )
    except:
        # Default values when user_config cannot be imported
        DATA_DIR = Path("./Data/raw/")
        RESULTS_DIR = Path("./Results/")
        FRB_TARGETS = ["FRB20201124_0009"]
        SLICE_DURATION_MS = 300.0
        DOWN_FREQ_RATE = 1
        DOWN_TIME_RATE = 8
        DM_min = 0
        DM_max = 1024
        DET_PROB = 0.3
        CLASS_PROB = 0.5
        SNR_THRESH = 5.0
        USE_MULTI_BAND = False
        SAVE_ONLY_BURST = True
        AUTO_HIGH_FREQ_PIPELINE = True
        HIGH_FREQ_THRESHOLD_MHZ = 8000.0
        ENABLE_LINEAR_VALIDATION = True
        POLARIZATION_MODE = "intensity"
        POLARIZATION_INDEX = 0
        DEBUG_FREQUENCY_ORDER = False
        FORCE_PLOTS = False

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================
              
MODEL_NAME = "resnet18"                                                           
MODEL_PATH = Path(__file__).parent.parent / "models" / f"cent_{MODEL_NAME}.pth"                            
                                
CLASS_MODEL_NAME = "resnet18"                                                         
CLASS_MODEL_PATH = Path(__file__).parent.parent / "models" / f"class_{CLASS_MODEL_NAME}.pth"                  
                   
# Device configuration (GPU/CPU)
if torch is not None:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
MAX_RAM_FRACTION: float = 0.25          # Fraction of available RAM to use
OVERHEAD_FACTOR: float = 1.3            # Memory overhead factor for safety

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

def inject_config(config_dict: dict):
    """Inject configuration from external source.
    
    Overrides module-level variables with values from config_dict.
    Used by main.py to inject configuration from config.yaml.
    
    Args:
        config_dict: Dictionary with configuration values to inject
    """
    import sys
    global _config_injected
    
    current_module = sys.modules[__name__]
    
    for key, value in config_dict.items():
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
                                                                               

def validate_configuration():
    """Validate system configuration and generate informative error messages.

    This function ensures all critical parameters are configured correctly
    before running the pipeline.
    
    Raises:
        ValueError: If any configuration parameter is invalid
        
    Returns:
        bool: True if all validations pass
    """
    errors = []
    
    # Validate data parameters
    if FREQ_RESO <= 0:
        errors.append(
            f"FREQ_RESO={FREQ_RESO} is invalid\n"
            f"  → FREQ_RESO must be > 0 to process frequency data\n"
            f"  → This value is extracted from the FITS/FIL file header\n"
            f"  → Recommendation: Verify that the data file is valid"
        )
    
    if TIME_RESO <= 0:
        errors.append(
            f"TIME_RESO={TIME_RESO} is invalid\n"
            f"  → TIME_RESO must be > 0 to process temporal data\n"
            f"  → This value is extracted from the FITS/FIL file header\n"
            f"  → Recommendation: Verify that the data file is valid"
        )
    
    if FILE_LENG <= 0:
        errors.append(
            f"FILE_LENG={FILE_LENG} is invalid\n"
            f"  → FILE_LENG must be > 0 to process data\n"
            f"  → This value indicates the total number of temporal samples\n"
            f"  → Recommendation: Verify that the data file is valid"
        )
    
    # Validate slice parameters
    if SLICE_LEN < SLICE_LEN_MIN or SLICE_LEN > SLICE_LEN_MAX:
        errors.append(
            f"SLICE_LEN={SLICE_LEN} is outside the valid range [{SLICE_LEN_MIN}, {SLICE_LEN_MAX}]\n"
            f"  → SLICE_LEN must lie between {SLICE_LEN_MIN} and {SLICE_LEN_MAX} samples\n"
            f"  → This value is calculated automatically from SLICE_DURATION_MS\n"
            f"  → Recommendation: Adjust SLICE_DURATION_MS in config.yaml"
        )
    
    # Validate DM ranges
    if DM_RANGE_MIN_WIDTH <= 0 or DM_RANGE_MAX_WIDTH <= 0:
        errors.append(
            f"Invalid DM ranges: MIN={DM_RANGE_MIN_WIDTH}, MAX={DM_RANGE_MAX_WIDTH}\n"
            f"  → Both values must be > 0\n"
            f"  → These values define the limits of the dynamic DM range\n"
            f"  → Recommendation: Verify DM range configuration"
        )
    
    if DM_RANGE_MIN_WIDTH >= DM_RANGE_MAX_WIDTH:
        errors.append(
            f"Inconsistent DM ranges: MIN={DM_RANGE_MIN_WIDTH} >= MAX={DM_RANGE_MAX_WIDTH}\n"
            f"  → DM_RANGE_MIN_WIDTH must be < DM_RANGE_MAX_WIDTH\n"
            f"  → Recommendation: Adjust DM range values"
        )
    
    # Validate model files
    if not MODEL_PATH.exists():
        errors.append(
            f"Detection model not found: {MODEL_PATH}\n"
            f"  → The model file does not exist at the specified path\n"
            f"  → Verify that the model is trained and saved\n"
            f"  → Recommendation: Train the model or verify the path"
        )
    
    if not CLASS_MODEL_PATH.exists():
        errors.append(
            f"Classification model not found: {CLASS_MODEL_PATH}\n"
            f"  → The model file does not exist at the specified path\n"
            f"  → Verify that the model is trained and saved\n"
            f"  → Recommendation: Train the model or verify the path"
        )
    
    # Validate CUDA if available
    if torch is not None and torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
        except Exception as e:
            errors.append(
                f"CUDA error: {e}\n"
                f"  → CUDA is available but not functioning correctly\n"
                f"  → Verify NVIDIA drivers and the PyTorch installation\n"
                f"  → Recommendation: Reinstall PyTorch with CUDA support or use CPU"
            )
    
    # Raise error if any validation failed
    if errors:
        error_message = "Invalid system configuration:\n\n"
        for i, error in enumerate(errors, 1):
            error_message += f"{i}. {error}\n\n"
        error_message += "Fix these errors before running the pipeline."
        raise ValueError(error_message)
    
    return True


def check_model_files():
    """Verify that model files exist and are accessible.

    Returns:
        dict: Dictionary with the status of each model (detection, classification)
    """
    model_status = {}
    
    # Check detection model
    if MODEL_PATH.exists():
        try:
            if torch is not None:
                state = torch.load(MODEL_PATH, map_location='cpu')
                model_status['detection'] = {
                    'exists': True,
                    'size_mb': MODEL_PATH.stat().st_size / (1024 * 1024),
                    'state_dict_keys': len(state.keys()) if isinstance(state, dict) else 0
                }
            else:
                model_status['detection'] = {
                    'exists': True,
                    'size_mb': MODEL_PATH.stat().st_size / (1024 * 1024),
                    'note': 'PyTorch not available for full verification'
                }
        except Exception as e:
            model_status['detection'] = {
                'exists': True,
                'error': f"Corrupted model: {e}",
                'recommendation': 'Retrain the model'
            }
    else:
        model_status['detection'] = {
            'exists': False,
            'error': f"File not found: {MODEL_PATH}",
            'recommendation': 'Train the model or verify the path'
        }
    
    # Check classification model
    if CLASS_MODEL_PATH.exists():
        try:
            if torch is not None:
                state = torch.load(CLASS_MODEL_PATH, map_location='cpu')
                model_status['classification'] = {
                    'exists': True,
                    'size_mb': CLASS_MODEL_PATH.stat().st_size / (1024 * 1024),
                    'state_dict_keys': len(state.keys()) if isinstance(state, dict) else 0
                }
            else:
                model_status['classification'] = {
                    'exists': True,
                    'size_mb': CLASS_MODEL_PATH.stat().st_size / (1024 * 1024),
                    'note': 'PyTorch not available for full verification'
                }
        except Exception as e:
            model_status['classification'] = {
                'exists': True,
                'error': f"Corrupted model: {e}",
                'recommendation': 'Retrain the model'
            }
    else:
        model_status['classification'] = {
            'exists': False,
            'error': f"File not found: {CLASS_MODEL_PATH}",
            'recommendation': 'Train the model or verify the path'
        }
    
    return model_status
