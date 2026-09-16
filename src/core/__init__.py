"""Core building blocks used to assemble the FRB detection pipeline."""

__all__ = [
    "process_slice_with_multiple_bands",
    "calculate_frequency_downsampled",
    "calculate_dm_height",
    "calculate_dm_values",
    "calculate_width_total",
    "calculate_slice_parameters",
    "calculate_time_slice",
    "calculate_overlap_decimated",
    "calculate_absolute_slice_time",
    "downsample_chunk",
    "build_dm_time_cube",
    "trim_valid_window",
    "plan_slices",
    "validate_slice_indices",
    "create_chunk_directories",
    "get_chunk_processing_parameters",
]


def __getattr__(name):
    if name == "process_slice_with_multiple_bands":
        from .detection_engine import process_slice_with_multiple_bands
        return process_slice_with_multiple_bands
    if name in {
        "calculate_frequency_downsampled",
        "calculate_dm_height",
        "calculate_dm_values",
        "calculate_width_total",
        "calculate_slice_parameters",
        "calculate_time_slice",
        "calculate_overlap_decimated",
        "calculate_absolute_slice_time",
    }:
        from . import pipeline_parameters
        return getattr(pipeline_parameters, name)
    if name in {
        "downsample_chunk",
        "build_dm_time_cube",
        "trim_valid_window",
        "plan_slices",
        "validate_slice_indices",
        "create_chunk_directories",
        "get_chunk_processing_parameters",
    }:
        from . import data_flow_manager
        return getattr(data_flow_manager, name)
    raise AttributeError(name)

