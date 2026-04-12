"""Streaming input helpers for FITS, PSRFITS, and filterbank files."""

__all__ = [
    "get_obparams",
    "stream_fits",
    "get_obparams_fil",
    "stream_fil",
    "detect_file_type",
    "validate_file_compatibility",
    "extract_parameters_auto",
    "get_streaming_function",
    "get_streaming_info",
    "validate_streaming_parameters",
    "find_data_files",
    "safe_float",
    "safe_int",
    "auto_config_downsampling",
    "print_debug_frequencies",
    "save_file_debug_info",
    "normalize_frequency_axis",
]


def __getattr__(name):
    if name in {"safe_float", "safe_int", "auto_config_downsampling", "print_debug_frequencies", "save_file_debug_info", "normalize_frequency_axis"}:
        from . import utils
        return getattr(utils, name)
    if name in {"get_obparams", "stream_fits"}:
        from . import fits_handler
        return getattr(fits_handler, name)
    if name in {"get_obparams_fil", "stream_fil"}:
        from . import filterbank_handler
        return getattr(filterbank_handler, name)
    if name in {"detect_file_type", "validate_file_compatibility"}:
        from . import file_detector
        return getattr(file_detector, name)
    if name == "extract_parameters_auto":
        from .parameter_extractor import extract_parameters_auto
        return extract_parameters_auto
    if name in {"get_streaming_function", "get_streaming_info", "validate_streaming_parameters"}:
        from . import streaming_orchestrator
        return getattr(streaming_orchestrator, name)
    if name == "find_data_files":
        from .file_finder import find_data_files
        return find_data_files
    raise AttributeError(name)

