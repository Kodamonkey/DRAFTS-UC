# This module logs data loading and streaming metrics.

"""
Data Loader Logging Management for DRAFTS
========================================

This module provides specialized functions to display detailed information
from the data loader, especially for FITS and FIL file streaming operations.
"""

                          
from typing import Any, Dict, Optional

               
from .logging_config import get_global_logger


def log_stream_fil_parameters(nsamples: int, chunk_samples: int, overlap_samples: int, 
                             nchans: int, nifs: int, nbits: int, dtype: str) -> None:
    """
    Logs streaming parameters for FIL files.
    
    Args:
        nsamples: Total number of samples
        chunk_samples: Chunk size
        overlap_samples: Overlap samples
        nchans: Number of channels
        nifs: Number of IFs
        nbits: Bits per sample
        dtype: Data type
    """
    logger = get_global_logger()
    logger.logger.debug(f"[DEBUG] STREAM_FIL: nsamples={nsamples:,}, chunk_samples={chunk_samples:,}, overlap={overlap_samples}")
    logger.logger.debug(f"[DEBUG] STREAM_FIL: nchans={nchans}, nifs={nifs}, nbits={nbits}, dtype={dtype}")


def log_stream_fil_block_generation(chunk_counter: int, block_shape: tuple, block_dtype: str,
                                   valid_start: int, valid_end: int, 
                                   start_with_overlap: int, end_with_overlap: int,
                                   actual_chunk_size: int) -> None:
    """
    Logs generation of a block in FIL streaming.
    
    Args:
        chunk_counter: Chunk counter
        block_shape: Block shape
        block_dtype: Block data type
        valid_start: Start of valid region
        valid_end: End of valid region
        start_with_overlap: Inicio con solapamiento
        end_with_overlap: Fin con solapamiento
        actual_chunk_size: Tamaño real del chunk
    """
    logger = get_global_logger()
    logger.logger.debug(f"[DEBUG] STREAM_FIL BLOQUE {chunk_counter}: shape={block_shape}, dtype={block_dtype}")
    logger.logger.debug(f"[DEBUG] STREAM_FIL BLOQUE {chunk_counter}: valid_range=({valid_start:,}, {valid_end:,})")
    logger.logger.debug(f"[DEBUG] STREAM_FIL BLOQUE {chunk_counter}: overlap_range=({start_with_overlap:,}, {end_with_overlap:,})")
    logger.logger.debug(f"[DEBUG] STREAM_FIL BLOQUE {chunk_counter}: actual_chunk_size={actual_chunk_size:,}")


def log_stream_fil_summary(chunk_counter: int) -> None:
    """
    Logs FIL streaming summary.
    
    Args:
        chunk_counter: Total number of chunks generated
    """
    logger = get_global_logger()
    logger.logger.debug(f"[DEBUG] STREAM_FIL SUMMARY: {chunk_counter} blocks generated successfully")
    logger.logger.debug(f"[DEBUG] STREAM_FIL SUMMARY: file processed completely through streaming")


def log_stream_fits_parameters(nsamples: int, chunk_samples: int, overlap_samples: int,
                              nsubint: Optional[int], nchan: Optional[int], 
                              npol: Optional[int], nsblk: Optional[int]) -> None:
    """
    Logs streaming parameters for FITS files.
    
    Args:
        nsamples: Total number of samples
        chunk_samples: Chunk size
        overlap_samples: Overlap samples
        nsubint: Number of subintegrations
        nchan: Number of channels
        npol: Number of polarizations
        nsblk: Number of samples per block
    """
    logger = get_global_logger()
    logger.logger.debug(f"[DEBUG] STREAM_FITS: nsamples={nsamples:,}, chunk_samples={chunk_samples:,}, overlap={overlap_samples}")
    logger.logger.debug(f"[DEBUG] STREAM_FITS: nsubint={nsubint if nsubint is not None else 'N/A'}, nchan={nchan if nchan is not None else 'N/A'}")
    logger.logger.debug(f"[DEBUG] STREAM_FITS: npol={npol if npol is not None else 'N/A'}, nsblk={nsblk if nsblk is not None else 'N/A'}")


def log_stream_fits_load_strategy(use_memmap: bool, data_shape: tuple, data_dtype: str) -> None:
    """
    Logs loading strategy for FITS files.
    
    Args:
        use_memmap: Whether memmap is used
        data_shape: Data shape
        data_dtype: Data type
    """
    logger = get_global_logger()
    logger.logger.debug(f"[DEBUG] STREAM_FITS LOAD: use_memmap={use_memmap}, data_array.shape={data_shape}")
    logger.logger.debug(f"[DEBUG] STREAM_FITS CARGA: data_array.dtype={data_dtype}")


def log_stream_fits_block_generation(chunk_counter: int, block_shape: tuple, block_dtype: str,
                                    valid_start: int, valid_end: int,
                                    start_with_overlap: int, end_with_overlap: int,
                                    actual_chunk_size: int) -> None:
    """
    Logs generation of a block in FITS streaming.
    
    Args:
        chunk_counter: Chunk counter
        block_shape: Block shape
        block_dtype: Block data type
        valid_start: Start of valid region
        valid_end: End of valid region
        start_with_overlap: Start with overlap
        end_with_overlap: End with overlap
        actual_chunk_size: Actual chunk size
    """
    logger = get_global_logger()
    logger.logger.debug(f"[DEBUG] STREAM_FITS BLOQUE {chunk_counter}: shape={block_shape}, dtype={block_dtype}")
    logger.logger.debug(f"[DEBUG] STREAM_FITS BLOQUE {chunk_counter}: valid_range=({valid_start:,}, {valid_end:,})")
    logger.logger.debug(f"[DEBUG] STREAM_FITS BLOQUE {chunk_counter}: overlap_range=({start_with_overlap:,}, {end_with_overlap:,})")
    logger.logger.debug(f"[DEBUG] STREAM_FITS BLOQUE {chunk_counter}: actual_chunk_size={actual_chunk_size:,}")


def log_stream_fits_summary(chunk_counter: int) -> None:
    """
    Logs FITS streaming summary.
    
    Args:
        chunk_counter: Total number of chunks generated
    """
    logger = get_global_logger()
    logger.logger.debug(f"[DEBUG] STREAM_FITS RESUMEN: {chunk_counter} bloques generados exitosamente")
    logger.logger.debug(f"[DEBUG] STREAM_FITS RESUMEN: archivo procesado completamente a través de streaming")
