# This module logs high-level pipeline execution events.

"""
Logging management for the DRAFTS pipeline
==========================================

This module provides specialized functions to display detailed pipeline
information, especially for chunking operations and file processing.
"""

                          
from typing import Any, Callable, Dict, Optional

               
from .logging_config import get_global_logger


def log_streaming_parameters(
    effective_chunk_samples: int,
    overlap_raw: int,
    total_samples: int,
    chunk_samples: int,
    streaming_func: Callable,
    file_type: str,
) -> None:
    """Record streaming parameters used in the pipeline.

    Args:
        effective_chunk_samples: Effective chunk size
        overlap_raw: Overlap in raw samples
        total_samples: Total number of samples in the file
        chunk_samples: Configured chunk size
        streaming_func: Streaming function used
        file_type: Detected file type
    """
    logger = get_global_logger().logger
    logger.info(
        "Streaming • type=%s • reader=%s • chunk=%s (effective=%s) • overlap=%d • total=%s",
        file_type,
        getattr(streaming_func, "__name__", str(streaming_func)),
        f"{chunk_samples:,}",
        f"{effective_chunk_samples:,}",
        overlap_raw,
        f"{total_samples:,}",
    )


def log_block_processing(
    actual_chunk_count: int, block_shape: tuple, block_dtype: str, metadata: Dict[str, Any]
) -> None:
    """Record block processing information in the pipeline.

    Args:
        actual_chunk_count: Current chunk counter
        block_shape: Shape of the block
        block_dtype: Data type of the block
        metadata: Block metadata
    """
    logger = get_global_logger().logger
    metadata_str = str(metadata) if metadata else "{}"
    logger.debug(
        "Block %d • shape=%s • dtype=%s • metadata=%s",
        actual_chunk_count,
        block_shape,
        block_dtype,
        metadata_str,
    )


def log_processing_summary(
    actual_chunk_count: int,
    chunk_count: int,
    cand_counter_total: int,
    n_bursts_total: int,
) -> None:
    """Record the processing summary within the pipeline.

    Args:
        actual_chunk_count: Number of processed chunks
        chunk_count: Estimated total number of chunks
        cand_counter_total: Total candidates found
        n_bursts_total: Total bursts detected
    """
    logger = get_global_logger().logger
    logger.info(
        "Progress • chunks=%d/%d • candidates=%d • bursts=%d",
        actual_chunk_count,
        chunk_count,
        cand_counter_total,
        n_bursts_total,
    )


def log_pipeline_file_processing(
    fits_path_name: str, file_suffix: str, total_samples: int, chunk_samples: int
) -> None:
    """Record the start of file processing in the pipeline.

    Args:
        fits_path_name: File name
        file_suffix: File extension
        total_samples: Total number of samples in the file
        chunk_samples: Configured chunk size
    """
    logger = get_global_logger().logger
    logger.debug(
        "File processing • %s (%s) • samples=%s • chunk=%s",
        fits_path_name,
        file_suffix,
        f"{total_samples:,}",
        f"{chunk_samples:,}",
    )


def log_pipeline_file_completion(fits_path_name: str, results: Dict[str, Any]) -> None:
    """Record the completion of file processing in the pipeline.

    Args:
        fits_path_name: File name
        results: Processing results
    """
    logger = get_global_logger().logger
    logger.debug(
        "File completed • %s • status=%s • chunks=%s • mode=%s",
        fits_path_name,
        results.get("status", "N/A"),
        results.get("chunks_processed", "N/A"),
        results.get("processing_mode", "N/A"),
    )
