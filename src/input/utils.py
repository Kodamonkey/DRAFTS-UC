# This module provides utility conversions for input processing.

"""Common utilities used while handling FITS and filterbank files."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

               
from ..config import config
from ..output.summary_manager import _update_summary_with_file_debug


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
    """Print a standard frequency debug block with the given prefix."""
    print(f"{prefix} File: {file_name}")
    print(f"{prefix} Detected freq_axis_inverted: {freq_axis_inverted}")
    print(f"{prefix} DATA_NEEDS_REVERSAL: {config.DATA_NEEDS_REVERSAL}")
    print(f"{prefix} First 5 frequencies: {config.FREQ[:5]}")
    print(f"{prefix} Last 5 frequencies: {config.FREQ[-5:]}")
    print(f"{prefix} Minimum frequency: {config.FREQ.min():.2f} MHz")
    print(f"{prefix} Maximum frequency: {config.FREQ.max():.2f} MHz")
    print(f"{prefix} Expected order: ascending frequencies")
    if config.FREQ[0] < config.FREQ[-1]:
        print(f"{prefix} Order CORRECT: {config.FREQ[0]:.2f} < {config.FREQ[-1]:.2f}")
    else:
        print(f"{prefix} Order INCORRECT: {config.FREQ[0]:.2f} > {config.FREQ[-1]:.2f}")
    print(f"{prefix} DOWN_FREQ_RATE: {config.DOWN_FREQ_RATE}")
    print(f"{prefix} DOWN_TIME_RATE: {config.DOWN_TIME_RATE}")
    print(f"{prefix} " + "="*50)


def save_file_debug_info(file_name: str, debug_info: Dict) -> None:
    """Persist debug information (FITS or FIL) into ``summary.json`` immediately."""
    try:
        results_dir = getattr(config, 'RESULTS_DIR', Path('./Results'))
        results_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(file_name).stem
        _update_summary_with_file_debug(results_dir, filename, debug_info)
    except Exception as e:
        print(f"[WARNING] Error saving debug info for {file_name}: {e}")
