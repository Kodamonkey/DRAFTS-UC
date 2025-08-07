"""Compatibility wrappers for image utility functions used in tests."""
from .visualization.visualization_unified import (
    save_detection_plot,
    plot_waterfall_block,
    _calculate_dynamic_dm_range,
)

__all__ = [
    "save_detection_plot",
    "plot_waterfall_block",
    "_calculate_dynamic_dm_range",
]
