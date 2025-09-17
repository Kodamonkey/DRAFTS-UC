from pathlib import Path

# =============================================================================
# DATA AND FILE CONFIGURATION
# =============================================================================

# Input and output directories
DATA_DIR = Path("./Data/raw")                        # Directory with input files (.fits, .fil)
RESULTS_DIR = Path("./fusion-tests")                      # Directory where results are stored

# List of files to process
FRB_TARGETS = [
   "2017-04-03-12_56_05_230_0002_t2.3_t17.395"
]

# =============================================================================
# TEMPORAL ANALYSIS CONFIGURATION
# =============================================================================

# Duration of each temporal slice (milliseconds)
SLICE_DURATION_MS: float = 300.0

# =============================================================================
# DOWNSAMPLING CONFIGURATION
# =============================================================================

# Reduction factors to optimize processing
DOWN_FREQ_RATE: int = 1                      # Frequency reduction factor (1 = no reduction)
DOWN_TIME_RATE: int = 8                     # Time reduction factor (1 = no reduction)


# =============================================================================
# DISPERSION MEASURE CONFIGURATION (DM)
# =============================================================================

# Dispersion Measure search range
DM_min: int = 0                             # Minimum DM in pc cm⁻³
DM_max: int = 1024                          # Maximum DM in pc cm⁻³

# =============================================================================
# DETECTION THRESHOLDS
# =============================================================================

# Minimum probabilities for detection and classification
DET_PROB: float = 0.3                       # Minimum probability to consider a detection valid
CLASS_PROB: float = 0.5                     # Minimum probability to classify as burst

# SNR threshold for highlighting in visualizations
SNR_THRESH: float = 4.0                     # SNR threshold used in plots

# =============================================================================
# MULTI-BAND ANALYSIS CONFIGURATION
# =============================================================================

# Multi-band analysis (Full/Low/High)
USE_MULTI_BAND: bool = False                # True = enable multi-band analysis, False = only full band

# =============================================================================
# HIGH-FREQUENCY PIPELINE CONFIGURATION
# =============================================================================

# Controls whether the high-frequency pipeline is triggered automatically
# based on the file's central frequency (default ≥ 8000 MHz)
AUTO_HIGH_FREQ_PIPELINE: bool = True

# Central frequency threshold (MHz) to consider "high frequency"
HIGH_FREQ_THRESHOLD_MHZ: float = 8000.0

# --- Yong-Kun Zhang high-frequency strategies --------------------------------

# Strategy 1 (bow-tie recovery) parameters
HF_DM_EXPANSION_FACTOR: float = 3.0          # Multiplier for the traditional DM range
HF_DM_COARSE_STEP: float = 5.0               # Coarse DM step (pc cm⁻³)
HF_DM_MIN_STEP: float = 1.0                  # Minimum DM step to guarantee coverage
HF_BOW_TIE_THRESHOLD: float = 2.0            # Minimum contrast required for bow-tie pattern
HF_BOW_TIE_WING_WIDTH: int = 20              # Samples used to estimate bow-tie wings
HF_BOW_TIE_MIN_DM_INDEX: int = 10            # Minimum DM index to accept a bow-tie candidate
HF_BOW_TIE_MIN_SNR: float = 5.0              # Minimum SNR for bow-tie detections

# Strategy 2 (zero-DM + validation) parameters
HF_ZERO_DM_TRIALS = [0.0, 1.0, 2.0, 5.0]     # DM trials around zero for permissive detection
HF_ZERO_DM_SENSITIVITY: float = 0.4          # Minimum class probability in permissive stage
HF_ZERO_DM_MIN_SNR: float = 3.5              # Minimum SNR to keep zero-DM candidates
HF_ZERO_DM_MAX_CANDIDATES: int = 1000        # Limit of candidates processed in validation
HF_MIN_SIGNIFICANT_DM: float = 15.0          # Minimum DM considered astrophysically valid
HF_VALIDATION_DM_MIN: float = 0.0            # DM search lower bound during validation
HF_VALIDATION_DM_MAX: float = 6000.0         # DM search upper bound during validation
HF_VALIDATION_DM_STEP: float = 5.0           # DM step during validation scan
HF_VALIDATION_PATCH_LEN: int = 256           # Samples used in validation patches
HF_SUBBAND_COUNT: int = 4                    # Sub-bands used to verify consistency
HF_SUBBAND_SNR_THRESHOLD: float = 5.0        # Minimum SNR per sub-band
HF_SUBBAND_CONSISTENCY_THRESHOLD: float = 0.75  # Required fraction of consistent sub-bands
HF_TEMPORAL_CHUNK_SEC: float = 30.0          # Temporal window for consistency checks
HF_TEMPORAL_SNR_THRESHOLD: float = 5.0       # Minimum SNR in independent chunk reprocessing
HF_DEDUP_TIME_TOL_SEC: float = 1.0           # Time tolerance for deduplication (seconds)
HF_DEDUP_DM_TOL: float = 50.0                # DM tolerance for deduplication (pc cm⁻³)

# Monitoring thresholds for the integrated pipeline
HF_MONITOR_BOW_TIE_RATIO: float = 0.1        # Expected minimum ratio of bow-tie recoveries

# =============================================================================
# POLARIZATION CONFIGURATION (PSRFITS INPUT)
# =============================================================================

# Polarization mode for PSRFITS with POL_TYPE=IQUV and npol>=4
# Options: "intensity" (I), "linear" (sqrt(Q^2+U^2)), "circular" (abs(V)),
#          "pol0", "pol1", "pol2", "pol3" to select a specific index
POLARIZATION_MODE: str = "intensity"

# Default index when IQUV is not available (e.g., AABB, two pols)
POLARIZATION_INDEX: int = 0

# =============================================================================
# LOGGING AND DEBUG CONFIGURATION
# =============================================================================

# Frequency and file debugging
DEBUG_FREQUENCY_ORDER: bool = True        # True = show detailed frequency and file information
                                           # False = quiet mode (recommended for batch processing)

# Force plot generation even when no candidates (debug mode)
FORCE_PLOTS: bool = False                  # True = always generate plots for inspection

# =============================================================================
# CANDIDATE FILTERING CONFIGURATION
# =============================================================================

# Only save and display candidates classified as BURST
SAVE_ONLY_BURST: bool = True             # True = keep only BURST candidates, False = keep all candidates

