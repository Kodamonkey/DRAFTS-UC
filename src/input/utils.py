# This module provides utility conversions for input processing.

"""Common utilities used while handling FITS and filterbank files."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict
import logging

import numpy as np

from ..config import config


logger = logging.getLogger(__name__)


def safe_float(value, default=0.0) -> float:
    """Return ``value`` as ``float`` or ``default`` if conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            cleaned = str(value).strip().replace("*", "").replace("UNSET", "")
            return float(cleaned)
        except (TypeError, ValueError):
            return default


def safe_int(value, default=0) -> int:
    """Return ``value`` as ``int`` or ``default`` if conversion fails."""
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            cleaned = str(value).strip().replace("*", "").replace("UNSET", "")
            return int(float(cleaned))
        except (TypeError, ValueError):
            return default


def auto_config_downsampling() -> None:
    """Configure downsampling factors when the user did not override them."""
    user_configured_freq = getattr(config, 'DOWN_FREQ_RATE', None)
    user_configured_time = getattr(config, 'DOWN_TIME_RATE', None)

    if user_configured_freq is None or user_configured_freq == 1:
        if config.FREQ_RESO >= 512:
            config.DOWN_FREQ_RATE = max(1, int(round(config.FREQ_RESO / 512)))
        else:
            config.DOWN_FREQ_RATE = 1

    if user_configured_time is None:
        if config.TIME_RESO > 1e-9:
            config.DOWN_TIME_RATE = max(1, int((49.152 * 16 / 1e6) / config.TIME_RESO))
        else:
            config.DOWN_TIME_RATE = 15


def print_debug_frequencies(prefix: str, file_name: str, freq_axis_inverted: bool) -> None:
    """Log a standard frequency debug block with the given prefix."""

    freq_values = getattr(config, "FREQ", None)
    try:
        freq_array = np.asarray(freq_values, dtype=float)
    except Exception:
        logger.debug("%s Frequency array is unavailable; skipping frequency debug block", prefix)
        return

    if freq_array.size == 0:
        logger.debug("%s Frequency array is empty; skipping frequency debug block", prefix)
        return

    first_values = freq_array[:5] if freq_array.size >= 5 else freq_array
    last_values = freq_array[-5:] if freq_array.size >= 5 else freq_array

    logger.debug("%s File: %s", prefix, file_name)
    logger.debug("%s Detected freq_axis_inverted: %s", prefix, freq_axis_inverted)
    logger.debug("%s DATA_NEEDS_REVERSAL: %s", prefix, getattr(config, "DATA_NEEDS_REVERSAL", 'N/A'))
    logger.debug("%s First frequencies: %s", prefix, first_values)
    logger.debug("%s Last frequencies: %s", prefix, last_values)
    logger.debug("%s Minimum frequency: %.2f MHz", prefix, float(np.min(freq_array)))
    logger.debug("%s Maximum frequency: %.2f MHz", prefix, float(np.max(freq_array)))
    logger.debug("%s Expected order: ascending frequencies", prefix)

    if freq_array.size >= 2:
        if freq_array[0] < freq_array[-1]:
            logger.debug("%s Order CORRECT: %.2f < %.2f", prefix, freq_array[0], freq_array[-1])
        else:
            logger.debug("%s Order INCORRECT: %.2f > %.2f", prefix, freq_array[0], freq_array[-1])

    logger.debug("%s DOWN_FREQ_RATE: %s", prefix, getattr(config, "DOWN_FREQ_RATE", 'N/A'))
    logger.debug("%s DOWN_TIME_RATE: %s", prefix, getattr(config, "DOWN_TIME_RATE", 'N/A'))
    logger.debug("%s %s", prefix, "=" * 50)


def save_file_debug_info(file_name: str, debug_info: Dict) -> None:
    """Persist debug information (FITS or FIL) into ``summary.json`` immediately.
    
    NOTE: JSON summary functionality has been disabled. This function now does nothing.
    """
    # JSON summary functionality disabled - no longer creating summary.json files
    pass
