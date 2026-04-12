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
    """Load configuration from config.yaml and advanced config files."""
    project_root = Path(__file__).parent.parent.parent
    main_config_path = project_root / "config.yaml"
    
    if not main_config_path.exists():
        raise FileNotFoundError(
            f"Main configuration file not found: {main_config_path}\n"
            f"Please create a config.yaml file in the project root."
        )
    
    # Load main configuration
    with open(main_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Load advanced configurations from advanced-config/ directory
    advanced_config_dir = project_root / "advanced-config"
    if advanced_config_dir.exists():
        advanced_configs = {
            'performance_advanced': 'performance.yaml',
            'visualization_advanced': 'visualization.yaml', 
            'models_advanced': 'models.yaml',
            'logging_advanced': 'logging.yaml'
        }
        
        for key, filename in advanced_configs.items():
            config_file = advanced_config_dir / filename
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        advanced_config = yaml.safe_load(f)
                        config[key] = advanced_config
                except Exception as e:
                    print(f"Warning: Failed to load {filename}: {e}")
    
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
TEMPORAL_DOWNSAMPLING_MODE = str(_config.get('downsampling', {}).get('temporal_mode', 'sum')).lower()

# =============================================================================
# DISPERSION MEASURE CONFIGURATION (DM)
# =============================================================================
DM_min = int(_config['dispersion']['dm_min'])
DM_max = int(_config['dispersion']['dm_max'])
DM_GRID_MODE = str(_config.get('dispersion', {}).get('dm_grid_mode', 'legacy_uniform')).lower()
MAX_DM_SMEARING_MS = _config.get('dispersion', {}).get('max_dm_smearing_ms', 'auto')
# DM_CHUNKING_THRESHOLD_GB moved to performance section

# =============================================================================
# DETECTION THRESHOLDS
# =============================================================================
DET_PROB = float(_config['thresholds']['detection_probability'])
CLASS_PROB = float(_config['thresholds']['classification_probability'])
CLASS_PROB_LINEAR = float(_config['thresholds'].get('classification_probability_linear', 0.6))
SNR_THRESH = float(_config['thresholds']['snr_threshold'])
SNR_THRESH_LINEAR = float(_config['thresholds'].get('snr_threshold_linear', 5.0))

_detection_config = _config.get('detection', {})
DETECTION_WIDTHS_MS = _detection_config.get('widths_ms', [])
TRIAL_CORRECTION = str(_detection_config.get('trial_correction', 'gaussian_extreme')).lower()

# =============================================================================
# MULTI-BAND ANALYSIS CONFIGURATION
# =============================================================================
USE_MULTI_BAND = bool(_config['multiband']['enabled'])

# =============================================================================
# HIGH-FREQUENCY PIPELINE CONFIGURATION
# =============================================================================
AUTO_HIGH_FREQ_PIPELINE = bool(_config['high_frequency']['auto_enable'])
BOWTIE_COLLAPSE_RATIO = float(_config['high_frequency'].get('collapse_ratio', 2.0))
HIGH_FREQ_DM_POLICY = str(_config['high_frequency'].get('dm_policy', 'unresolved')).lower()

# Phase 2: Linear Polarization SNR Validation
ENABLE_LINEAR_VALIDATION = bool(_config['high_frequency'].get('enable_linear_validation', False))

# Phase 3: ResNet18 Classification Control
ENABLE_INTENSITY_CLASSIFICATION = bool(_config['high_frequency'].get('enable_intensity_classification', True))
ENABLE_LINEAR_CLASSIFICATION = bool(_config['high_frequency'].get('enable_linear_classification', True))

# Validation: At least one classification phase must be enabled
if not ENABLE_INTENSITY_CLASSIFICATION and not ENABLE_LINEAR_CLASSIFICATION:
    raise ValueError(
        "Invalid configuration: At least one classification phase must be enabled. "
        "Set enable_intensity_classification=true OR enable_linear_classification=true in config.yaml"
    )

# =============================================================================
# POLARIZATION CONFIGURATION (PSRFITS INPUT)
# =============================================================================
POLARIZATION_MODE = str(_config['polarization']['mode'])
POLARIZATION_INDEX = int(_config['polarization']['default_index'])
POLARIZATION_LINEAR_DEBIAS = bool(_config['polarization'].get('linear_debias', True))

# =============================================================================
# LOGGING AND DEBUG CONFIGURATION
# =============================================================================
DEBUG_FREQUENCY_ORDER = bool(_config['debug']['show_frequency_info'])
FORCE_PLOTS = bool(_config['debug']['force_plots'])

# =============================================================================
# CANDIDATE FILTERING CONFIGURATION
# =============================================================================
SAVE_ONLY_BURST = bool(_config['output']['save_only_burst'])

# =============================================================================
# PERFORMANCE AND MEMORY OPTIMIZATION
# =============================================================================
_performance_config = _config.get('performance', {})
_performance_advanced = _config.get('performance_advanced', {})

# Basic Performance Settings (from main config.yaml)
MAX_RAM_FRACTION_USER = float(_performance_config.get('max_ram_fraction', 0.25))
MAX_CHUNK_SAMPLES = int(_performance_config.get('max_chunk_samples', 1000000))
MAX_DM_CUBE_SIZE_GB = float(_performance_config.get('max_dm_cube_size_gb', 2.0))
DM_CHUNKING_THRESHOLD_GB_USER = float(_performance_config.get('dm_chunking_threshold_gb', 16.0))

# Advanced Performance Settings (from config/performance.yaml)
_memory_config = _performance_advanced.get('memory', {})
MEMORY_OVERHEAD_FACTOR_USER = float(_memory_config.get('overhead_factor', 1.3))

# Settings kept for Phase 2/3 implementation
_gpu_config = _performance_advanced.get('gpu', {})
ENABLE_MIXED_PRECISION = bool(_gpu_config.get('enable_mixed_precision', False))

_io_config = _performance_advanced.get('io', {})
ENABLE_ASYNC_IO = bool(_io_config.get('enable_async_io', True))

_parallel_config = _performance_advanced.get('parallel', {})
CPU_THREADS = int(_parallel_config.get('cpu_threads', 0))

