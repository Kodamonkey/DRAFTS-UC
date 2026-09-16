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


def normalize_frequency_axis(freq: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return ascending MHz frequency axis and whether data channels must reverse."""
    arr = np.asarray(freq, dtype=np.float64)
    if arr.size <= 1:
        return arr, False
    descending = bool(arr[0] > arr[-1])
    if descending:
        return arr[::-1].copy(), True
    return arr.copy(), False


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
    """Configure downsampling factors when the user did not set them explicitly.

    Values set explicitly in config.yaml (via user_config) are always respected,
    including frequency_rate: 1 (no reduction). Auto-adjustment only kicks in
    when the pipeline runtime value is None (never set by user or code).
    """
    current_freq = getattr(config, 'DOWN_FREQ_RATE', None)
    current_time = getattr(config, 'DOWN_TIME_RATE', None)

    # Read the user's explicit setting from config.yaml to compare.
    # If the runtime value matches what user_config loaded, the user set it
    # explicitly and we must not override it.
    try:
        from ..config import user_config as _uc
        user_freq_explicit = int(_uc.DOWN_FREQ_RATE)
        user_time_explicit = int(_uc.DOWN_TIME_RATE)
    except Exception:
        user_freq_explicit = None
        user_time_explicit = None

    # Frequency downsampling: only auto-adjust when no explicit value was set
    # (current_freq is None) or when the value was never touched by user_config
    # (user_freq_explicit is None).
    freq_is_user_set = (
        current_freq is not None
        and user_freq_explicit is not None
        and current_freq == user_freq_explicit
    )
    if not freq_is_user_set:
        if current_freq is None:
            # Pipeline never got a value: derive from file channel count
            if config.FREQ_RESO >= 512:
                config.DOWN_FREQ_RATE = max(1, int(round(config.FREQ_RESO / 512)))
            else:
                config.DOWN_FREQ_RATE = 1
            logger.info(
                "[AUTO-DS] DOWN_FREQ_RATE auto-set to %d (FREQ_RESO=%d)",
                config.DOWN_FREQ_RATE, config.FREQ_RESO,
            )
    else:
        logger.debug(
            "[AUTO-DS] DOWN_FREQ_RATE=%d kept (explicit config.yaml value)",
            config.DOWN_FREQ_RATE,
        )

    # Time downsampling: only auto-adjust when no value was set
    time_is_user_set = (
        current_time is not None
        and user_time_explicit is not None
        and current_time == user_time_explicit
    )
    if not time_is_user_set:
        if current_time is None:
            if config.TIME_RESO > 1e-9:
                config.DOWN_TIME_RATE = max(1, int((49.152 * 16 / 1e6) / config.TIME_RESO))
            else:
                config.DOWN_TIME_RATE = 15
            logger.info(
                "[AUTO-DS] DOWN_TIME_RATE auto-set to %d (TIME_RESO=%.2e s)",
                config.DOWN_TIME_RATE, config.TIME_RESO,
            )
    else:
        logger.debug(
            "[AUTO-DS] DOWN_TIME_RATE=%d kept (explicit config.yaml value)",
            config.DOWN_TIME_RATE,
        )


def print_debug_frequencies(prefix: str, file_name: str, freq_axis_inverted: bool) -> None:
    """Log a standard frequency debug block with the given prefix."""

    logger.debug("%s File: %s", prefix, file_name)
    logger.debug("%s Detected freq_axis_inverted: %s", prefix, freq_axis_inverted)
    logger.debug("%s DATA_NEEDS_REVERSAL: %s", prefix, config.DATA_NEEDS_REVERSAL)
    logger.debug("%s First 5 frequencies: %s", prefix, config.FREQ[:5])
    logger.debug("%s Last 5 frequencies: %s", prefix, config.FREQ[-5:])
    logger.debug("%s Minimum frequency: %.2f MHz", prefix, config.FREQ.min())
    logger.debug("%s Maximum frequency: %.2f MHz", prefix, config.FREQ.max())
    logger.debug("%s Expected order: ascending frequencies", prefix)
    if config.FREQ[0] < config.FREQ[-1]:
        logger.debug("%s Order CORRECT: %.2f < %.2f", prefix, config.FREQ[0], config.FREQ[-1])
    else:
        logger.debug("%s Order INCORRECT: %.2f > %.2f", prefix, config.FREQ[0], config.FREQ[-1])
    logger.debug("%s DOWN_FREQ_RATE: %s", prefix, config.DOWN_FREQ_RATE)
    logger.debug("%s DOWN_TIME_RATE: %s", prefix, config.DOWN_TIME_RATE)
    logger.debug("%s %s", prefix, "=" * 50)


def save_file_debug_info(file_name: str, debug_info: Dict) -> None:
    """Persist debug information (FITS or FIL) into ``summary.json`` immediately.
    
    NOTE: JSON summary functionality has been disabled. This function now does nothing.
    """
    # JSON summary functionality disabled - no longer creating summary.json files
    pass
