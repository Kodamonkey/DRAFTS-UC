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
    # Both decide chunk geometry, both are assigned directly by end-to-end
    # modules, and neither was restored: `calculate_memory_safe_chunk_size`
    # clamps the chunk to `(MAX_CHUNK_SAMPLES // (SLICE_LEN * DOWN_TIME_RATE))`
    # blocks, so a value left behind by an earlier module silently retiles a
    # later one's file -- 4096-sample chunks arriving as 3672, every
    # `start_sample` moved and every absolute time with it.
    "MAX_CHUNK_SAMPLES", "SLICE_DURATION_MS",
    "TSTART_MJD", "TSTART_MJD_CORR",
    "SOURCE_RA", "SOURCE_DEC", "REF_FREQ_MHZ", "OBSERVATORY", "EPHEMERIS",
    # Created by get_obparams from a PSRFITS header; absent until it runs.
    "NBITS", "NPOL", "POL_TYPE", "NSUBOFFS", "TSUBINT", "NEED_FLIPBAND",
]

# Marks a key that did not exist on the module before the test, so restoring
# means deleting it again rather than leaving the value the test created.
_ABSENT = object()


def _snapshot_config() -> dict:
    snap: dict = {}
    for key in _CONFIG_SNAPSHOT_KEYS:
        if not hasattr(_config_module, key):
            snap[key] = _ABSENT
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
        if val is _ABSENT:
            if hasattr(_config_module, key):
                delattr(_config_module, key)
        elif isinstance(val, np.ndarray):
            setattr(_config_module, key, val.copy())
        else:
            setattr(_config_module, key, val)


@pytest.fixture(autouse=True)
def _isolate_config():
    """Restore config globals after each test to prevent order-dependent failures."""
    before = _snapshot_config()
    yield
    _restore_config(before)


@pytest.fixture(autouse=True, scope="session")
def _keep_astropy_off_the_network():
    """Stop astropy reaching for the network mid-test.

    Two paths do. The site registry is one, and ``tests/observatory.py`` deals
    with it by pinning a position instead of a name. The other is the IERS
    Earth-orientation table: astropy tries two mirrors, waits out both
    timeouts, warns, and falls back to the IERS-A table bundled in
    ``astropy-iers-data``.

    The fallback is not an approximation worth avoiding here -- the bundled
    table reproduces every barycentric column of
    ``tests/golden/lf_candidates.csv`` to its full twelve decimals. So the
    download changes no assertion in this suite and costs two timeouts per
    call. Turning it off makes the run deterministic and faster, and it makes a
    genuine network dependency fail loudly rather than after a wait.
    """
    try:
        from astropy.utils import iers
    except ImportError:
        yield
        return
    before = iers.conf.auto_download
    iers.conf.auto_download = False
    try:
        yield
    finally:
        iers.conf.auto_download = before


@pytest.fixture(autouse=True)
def _forget_resolved_observatories():
    """Keep ``mjd_utils``' site cache from leaking between tests.

    It remembers failures as well as successes (deliberately -- see
    ``_resolve_site``), which within one process would otherwise make a test
    that pins a bad site name change the outcome of one that does not.
    """
    from src.core import mjd_utils

    mjd_utils.clear_site_cache()
    yield
    mjd_utils.clear_site_cache()
