"""Pytest fixtures: isolate mutable global ``config`` between tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config as _config_module

# Only snapshot keys tests mutate; avoids copying modules (torch, etc.).
_CONFIG_SNAPSHOT_KEYS = [
    "TIME_RESO", "FREQ_RESO", "FILE_LENG", "FREQ", "DOWN_TIME_RATE", "DOWN_FREQ_RATE",
    "DM_min", "DM_max", "DM_GRID_MODE", "MAX_DM_SMEARING_MS", "DEVICE",
    "PREWHITEN_BEFORE_DM", "TEMPORAL_DOWNSAMPLING_MODE", "DETECTION_WIDTHS_MS",
    "BOWTIE_COLLAPSE_RATIO", "AUTO_HIGH_FREQ_PIPELINE", "DATA_NEEDS_REVERSAL",
    "DM_CHUNKING_THRESHOLD_GB", "MAX_DM_CUBE_SIZE_GB", "SLICE_LEN",
    "TSTART_MJD", "TSTART_MJD_CORR",
]


def _snapshot_config() -> dict:
    snap: dict = {}
    for key in _CONFIG_SNAPSHOT_KEYS:
        if not hasattr(_config_module, key):
            continue
        val = getattr(_config_module, key)
        if isinstance(val, np.ndarray):
            snap[key] = val.copy()
        elif val is None or isinstance(val, (bool, int, float, str)):
            snap[key] = val
        elif isinstance(val, (list, tuple)):
            snap[key] = type(val)(val)
        else:
            # Skip non-serialisable objects (modules, callables).
            continue
    return snap


def _restore_config(snap: dict) -> None:
    for key, val in snap.items():
        if isinstance(val, np.ndarray):
            setattr(_config_module, key, val.copy())
        else:
            setattr(_config_module, key, val)


@pytest.fixture(autouse=True)
def _isolate_config():
    """Restore config globals after each test to prevent order-dependent failures."""
    before = _snapshot_config()
    yield
    _restore_config(before)
