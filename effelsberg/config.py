"""Global configuration and runtime parameters for the Effelsberg pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Device selection -------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Observation parameters -------------------------------------------------------
FREQ: np.ndarray | None = None
FREQ_RESO: int = 0
TIME_RESO: float = 0.0
FILE_LENG: int = 0
DOWN_FREQ_RATE: int = 1
DOWN_TIME_RATE: int = 1
DATA_NEEDS_REVERSAL: bool = False

# Pipeline configuration ------------------------------------------------------
USE_MULTI_BAND: bool = False
SLICE_LEN: int = 512
DET_PROB: float = 0.5
DM_min: int = 0
DM_max: int = 129

# Paths -----------------------------------------------------------------------
DATA_DIR = Path("./Data")
RESULTS_DIR = Path("./Results/ObjectDetection")
MODEL_NAME = "resnet50"
MODEL_PATH = Path(f"cent_{MODEL_NAME}.pth")

# Default FRB targets --------------------------------------------------------
# List of substrings used to select FITS files corresponding to specific
# observations. Update this list to search for other FRBs.
FRB_TARGETS = ["B0355+54"]
