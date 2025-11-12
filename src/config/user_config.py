"""
User Configuration Module
===========================
This module loads configuration from config.yaml and makes it available
to the rest of the pipeline through config.py.

DO NOT hardcode values here. Edit config.yaml instead.
"""

from pathlib import Path
import yaml


def _load_config():
    """Load configuration from config.yaml file."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please create a config.yaml file in the project root.\n"
            f"You can use config.example.yaml as a template."
        )
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


# Load configuration from YAML
_config = _load_config()

# =============================================================================
# DATA AND FILE CONFIGURATION
# =============================================================================
DATA_DIR = Path(_config['data']['input_dir'])
RESULTS_DIR = Path(_config['data']['results_dir'])
FRB_TARGETS = _config['data']['targets']

# =============================================================================
# TEMPORAL ANALYSIS CONFIGURATION
# =============================================================================
SLICE_DURATION_MS = float(_config['temporal']['slice_duration_ms'])

# =============================================================================
# DOWNSAMPLING CONFIGURATION
# =============================================================================
DOWN_FREQ_RATE = int(_config['downsampling']['frequency_rate'])
DOWN_TIME_RATE = int(_config['downsampling']['time_rate'])

# =============================================================================
# DISPERSION MEASURE CONFIGURATION (DM)
# =============================================================================
DM_min = int(_config['dispersion']['dm_min'])
DM_max = int(_config['dispersion']['dm_max'])

# =============================================================================
# DETECTION THRESHOLDS
# =============================================================================
DET_PROB = float(_config['thresholds']['detection_probability'])
CLASS_PROB = float(_config['thresholds']['classification_probability'])
SNR_THRESH = float(_config['thresholds']['snr_threshold'])

# =============================================================================
# MULTI-BAND ANALYSIS CONFIGURATION
# =============================================================================
USE_MULTI_BAND = bool(_config['multiband']['enabled'])

# =============================================================================
# HIGH-FREQUENCY PIPELINE CONFIGURATION
# =============================================================================
AUTO_HIGH_FREQ_PIPELINE = bool(_config['high_frequency']['auto_enable'])
HIGH_FREQ_THRESHOLD_MHZ = float(_config['high_frequency']['threshold_mhz'])
ENABLE_LINEAR_VALIDATION = bool(_config['high_frequency'].get('enable_linear_validation', True))

# =============================================================================
# POLARIZATION CONFIGURATION (PSRFITS INPUT)
# =============================================================================
POLARIZATION_MODE = str(_config['polarization']['mode'])
POLARIZATION_INDEX = int(_config['polarization']['default_index'])

# =============================================================================
# LOGGING AND DEBUG CONFIGURATION
# =============================================================================
DEBUG_FREQUENCY_ORDER = bool(_config['debug']['show_frequency_info'])
FORCE_PLOTS = bool(_config['debug']['force_plots'])

# =============================================================================
# CANDIDATE FILTERING CONFIGURATION
# =============================================================================
SAVE_ONLY_BURST = bool(_config['output']['save_only_burst'])

