from pathlib import Path

# =============================================================================
# DATA AND FILE CONFIGURATION
# =============================================================================

# Input and output directories
DATA_DIR = Path("./Data/raw/")                        # Directory with input files (.fits, .fil)
RESULTS_DIR = Path("./Tests-Pulse-big-new")                 # Directory where results are stored

# List of files to process
FRB_TARGETS = [
   "2017-04-03-08_55_22_153_0006_t23.444"
]

# =============================================================================
# TEMPORAL ANALYSIS CONFIGURATION
# =============================================================================

# Duration of each temporal slice (milliseconds)
SLICE_DURATION_MS: float = 5000.0

# =============================================================================
# DOWNSAMPLING CONFIGURATION
# =============================================================================

# Reduction factors to optimize processing
DOWN_FREQ_RATE: int = 1                      # Frequency reduction factor (1 = no reduction)
DOWN_TIME_RATE: int = 1080                     # Time reduction factor (1 = no reduction)

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
DEBUG_FREQUENCY_ORDER: bool = False        # True = show detailed frequency and file information
                                           # False = quiet mode (recommended for batch processing)

# Force plot generation even when no candidates (debug mode)
FORCE_PLOTS: bool = True                  # True = always generate plots for inspection

# =============================================================================
# CANDIDATE FILTERING CONFIGURATION
# =============================================================================

# Only save and display candidates classified as BURST
SAVE_ONLY_BURST: bool = True             # True = keep only BURST candidates, False = keep all candidates

