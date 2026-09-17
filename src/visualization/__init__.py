# This module aggregates visualization entry points.

"""Visualization module for FRB pipeline."""

# Must come first: every module below imports matplotlib.pyplot at module
# scope, and importing pyplot is what instantiates the backend.
from . import mpl_backend  # noqa: F401

from . import plot_composite
from . import plot_dm_time
from . import plot_waterfall_dispersed
from . import plot_waterfall_dedispersed
# Patch plots disabled
# from . import plot_patches
from . import plot_individual_components
from . import visualization_ranges
from . import visualization_unified

__all__ = [
    "plot_composite",
    "plot_dm_time",
    "plot_waterfall_dispersed", 
    "plot_waterfall_dedispersed",
    "plot_patches",
    "plot_individual_components",
    "visualization_ranges", 
    "visualization_unified"
] 
