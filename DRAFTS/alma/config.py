"""Global configuration and runtime parameters for the ALMA pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Device configuration ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Observation parameters -------------------------------------------------------
FREQ: np.ndarray | None = None
FREQ_RESO: int = 0
TIME_RESO: float = 0.0
FILE_LENG: int = 0
DOWN_FREQ_RATE: int = 1
DOWN_TIME_RATE: int = 1  # Keep original time resolution for ALMA
DATA_NEEDS_REVERSAL: bool = False

# Pipeline configuration -------------------------------------------------------
USE_MULTI_BAND: bool = True
SLICE_LEN: int = 2
DET_PROB: float = 0.1
DM_min: int = 0
DM_max: int = 0

# Paths to data and models -----------------------------------------------------
DATA_DIR = Path("./Data")
RESULTS_DIR = Path("./Results/ObjectDetection")
MODEL_NAME = "resnet50"
MODEL_PATH = Path(f"./models/cent_{MODEL_NAME}.pth")

# Default FRB targets ----------------------------------------------------------
FRB_TARGETS = ["2017-04-03-08"]
