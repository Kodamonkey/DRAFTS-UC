"""Expose data loading utilities at top-level for tests."""
from .input.data_loader import load_fits_file

__all__ = ["load_fits_file"]
