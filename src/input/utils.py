# This module provides utility conversions for input processing.

"""Common utilities used while handling FITS and filterbank files."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict
import logging


from ..config import config
from ..output.summary_manager import _update_summary_with_file_debug


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
    """Persist debug information (FITS or FIL) into ``summary.json`` immediately."""
    try:
        results_dir = getattr(config, 'RESULTS_DIR', Path('./Results'))
        results_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(file_name).stem
        _update_summary_with_file_debug(results_dir, filename, debug_info)
    except Exception as e:
        logger.warning("Failed to save debug info for %s: %s", file_name, e)
