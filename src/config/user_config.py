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
# SOURCE AND OBSERVATORY (barycentric MJD only)
# =============================================================================
# Defaults match what src/core/mjd_utils.py used to hardcode, so an existing
# config.yaml without a `source:` section keeps its previous output.
SOURCE_RA = str(_config.get('source', {}).get('ra', "05:31:58.70"))
SOURCE_DEC = str(_config.get('source', {}).get('dec', "33:08:52.5"))
REF_FREQ_MHZ = float(_config.get('source', {}).get('reference_freq_mhz', 1400.0))
OBSERVATORY = str(_config.get('source', {}).get('observatory', "Effelsberg"))
EPHEMERIS = str(_config.get('source', {}).get('ephemeris', "de432s"))

# =============================================================================
# PREPROCESSING CONFIGURATION
# =============================================================================
# Scientific default is False: prewhitening changes the DM-cube physics.
PREWHITEN_BEFORE_DM = bool(_config.get('preprocessing', {}).get('prewhiten_before_dm', False))

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

_gpu_config = _performance_advanced.get('gpu', {})

# These two were the ones audit PERF-04 names: declared here, loaded into a
# dictionary, and read by nothing. An operator who set enable_mixed_precision
# or raised batch_size was changing a comment. They drive
# ``detection.model_interface`` now.
INFERENCE_BATCH_SIZE = max(1, int(_gpu_config.get('batch_size', 4)))
ENABLE_MIXED_PRECISION = bool(_gpu_config.get('enable_mixed_precision', False))
GPU_MEMORY_MANAGEMENT = bool(_gpu_config.get('enable_memory_management', True))
# cudnn.benchmark picks the fastest convolution algorithm for a given input
# shape by trying them once and caching the winner. It is a clear win when the
# shapes repeat, which they do here -- every patch is the same size -- and a
# loss when they vary, because every new shape pays the search. Defaulted on
# for that reason, and overridable because "the shapes repeat" is a property of
# the data, not a law.
CUDNN_BENCHMARK = bool(_gpu_config.get('cudnn_benchmark', True))

_io_config = _performance_advanced.get('io', {})

_parallel_config = _performance_advanced.get('parallel', {})
CPU_THREADS = int(_parallel_config.get('cpu_threads', 0))


# =============================================================================
# VISUALIZATION (from advanced-config/visualization.yaml)
# =============================================================================
# These values existed twice: as YAML nobody read, and as literals in config.py.
# Editing the YAML did nothing. The YAML is the source of truth now, and the
# defaults below are the literals config.py used to carry, so behaviour is
# unchanged for an existing installation.
_visualization_advanced = _config.get('visualization_advanced', {})
_snr_vis = _visualization_advanced.get('snr', {})
_adaptive_dm = _visualization_advanced.get('adaptive_dm', {})
_dm_plotting = _visualization_advanced.get('dm_plotting', {})
_styling = _visualization_advanced.get('styling', {})
_export = _visualization_advanced.get('export', {})

SNR_HIGHLIGHT_COLOR = str(_snr_vis.get('highlight_color', 'red'))
SNR_SHOW_PEAK_LINES = bool(_snr_vis.get('show_peak_lines', False))

DM_RANGE_MIN_WIDTH = float(_adaptive_dm.get('range_min_width', 80.0))
DM_RANGE_MAX_WIDTH = float(_adaptive_dm.get('range_max_width', 300.0))
DM_RANGE_FACTOR = float(_adaptive_dm.get('range_factor', 0.3))
DM_DYNAMIC_RANGE_ENABLE = bool(_adaptive_dm.get('dynamic_range_enable', False))
DM_RANGE_DEFAULT_VISUALIZATION = str(_dm_plotting.get('default_visualization', 'detailed'))

# Figure output. dpi dominates the cost of a render, and every plot was pinned
# at 300 with no way to lower it (audit P1-23 / PERF-01).
PLOT_DPI = int(_styling.get('dpi', 300))
PLOT_BBOX_INCHES = _export.get('bbox_inches', 'tight') or None
PLOT_PAD_INCHES = float(_export.get('pad_inches', 0.1))

# =============================================================================
# LOGGING (from advanced-config/logging.yaml)
# =============================================================================
# These keys were loaded and then never read by anything, so editing them had
# no effect. They drive the logger now.
_logging_advanced = _config.get('logging_advanced', {})
_logging_general = _logging_advanced.get('general', {})

LOG_LEVEL = str(_logging_general.get('level', 'INFO')).upper()
LOG_COLORS = bool(_logging_general.get('colors', True))
_enable_file_logging = bool(_logging_general.get('enable_file_logging', False))
# Only an explicit path counts; otherwise the logger picks its own destination
# (DRAFTS_LOG_DIR, or a directory next to the package).
LOG_FILE = str(_logging_general.get('log_file')) if _enable_file_logging and _logging_general.get('log_file') else None
LOG_MAX_BYTES = int(float(_logging_general.get('max_file_size_mb', 50)) * 1024 * 1024)
LOG_BACKUP_COUNT = int(_logging_general.get('backup_count', 5))

# =============================================================================
# MODELS (from advanced-config/models.yaml)
# =============================================================================
# The file was loaded and never read: 177 lines of architecture, inference and
# validation settings that nothing consumed (audit P2-29, REF-07). It has been
# cut down to the keys a run can actually honour -- which weights to load and
# under what backbone name -- and those keys are read here.
#
# The defaults are the literals config.py carried, so an installation without
# the file, or with the keys removed, behaves exactly as before.
def resolve_model_settings(models_advanced: dict, src_dir: Path) -> dict:
    """Turn the ``models.yaml`` mapping into the five values config.py exports.

    A function rather than inline statements so it can be tested with a
    configuration other than the installed one -- which is the only way to show
    that the file is read at all. The literals config.py used to carry are the
    defaults, so an installation without the file behaves exactly as before.

    *src_dir* is the ``src/`` package directory, which is what ``model_dir`` is
    relative to.
    """
    models_advanced = models_advanced or {}
    arch = models_advanced.get('architecture') or {}
    paths = models_advanced.get('paths') or {}

    detection_name = str((arch.get('detection') or {}).get('name') or 'resnet18')
    classification_name = str(
        (arch.get('classification') or {}).get('name') or 'resnet18'
    )
    model_dir = (Path(src_dir) / str(paths.get('model_dir') or 'models')).resolve()

    # An absent or empty file name falls back to the conventional one, which is
    # what config.py built unconditionally.
    detection_file = paths.get('detection_model') or f"cent_{detection_name}.pth"
    classification_file = (
        paths.get('classification_model') or f"class_{classification_name}.pth"
    )
    return {
        'MODEL_NAME': detection_name,
        'CLASS_MODEL_NAME': classification_name,
        'MODEL_DIR': model_dir,
        'MODEL_PATH': model_dir / str(detection_file),
        'CLASS_MODEL_PATH': model_dir / str(classification_file),
    }


_model_settings = resolve_model_settings(
    _config.get('models_advanced', {}), Path(__file__).parent.parent
)
MODEL_NAME = _model_settings['MODEL_NAME']
CLASS_MODEL_NAME = _model_settings['CLASS_MODEL_NAME']
MODEL_DIR = _model_settings['MODEL_DIR']
MODEL_PATH = _model_settings['MODEL_PATH']
CLASS_MODEL_PATH = _model_settings['CLASS_MODEL_PATH']

