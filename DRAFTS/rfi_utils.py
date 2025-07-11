from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from . import config
from .rfi_mitigation import RFIMitigator

__all__ = ["apply_rfi_cleaning"]


def apply_rfi_cleaning(
    waterfall: np.ndarray,
    stokes_v: np.ndarray | None = None,
    output_dir: Path | None = None,
) -> Tuple[np.ndarray, dict]:
    """Apply RFI mitigation to ``waterfall`` if enabled."""
    if not getattr(config, "RFI_ENABLE_ALL_FILTERS", False):
        return waterfall, {}

    rfi_mitigator = RFIMitigator(
        freq_sigma_thresh=getattr(config, "RFI_FREQ_SIGMA_THRESH", 5.0),
        time_sigma_thresh=getattr(config, "RFI_TIME_SIGMA_THRESH", 5.0),
        zero_dm_sigma_thresh=getattr(config, "RFI_ZERO_DM_SIGMA_THRESH", 4.0),
        impulse_sigma_thresh=getattr(config, "RFI_IMPULSE_SIGMA_THRESH", 6.0),
        polarization_thresh=getattr(config, "RFI_POLARIZATION_THRESH", 0.8),
    )

    cleaned, stats = rfi_mitigator.clean_waterfall(
        waterfall, stokes_v=stokes_v, apply_all_filters=True
    )

    if (
        getattr(config, "RFI_SAVE_DIAGNOSTICS", False)
        and output_dir
        and stats.get("total_flagged_fraction", 0) > 0.001
    ):
        rfi_dir = output_dir / "rfi_diagnostics"
        rfi_dir.mkdir(parents=True, exist_ok=True)
        diagnostic_path = rfi_dir / "rfi_cleaning_diagnostics.png"
        rfi_mitigator.plot_rfi_diagnostics(waterfall, cleaned, diagnostic_path)

    return cleaned, stats
