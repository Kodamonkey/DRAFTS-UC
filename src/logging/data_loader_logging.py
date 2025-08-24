"""
Manejo de Logging para Data Loader en DRAFTS
============================================

Este módulo proporciona funciones especializadas para mostrar información
detallada del data loader, especialmente para operaciones de streaming
de archivos FITS y FIL.
"""

# Standard library imports
from typing import Any, Dict, Optional

# Local imports
from .logging_config import get_global_logger


def log_stream_fil_parameters(nsamples: int, chunk_samples: int, overlap_samples: int, 
                             nchans: int, nifs: int, nbits: int, dtype: str) -> None:
    """
    Registra los parámetros de streaming para archivos FIL.
    
    Args:
        nsamples: Número total de muestras
        chunk_samples: Tamaño del chunk
        overlap_samples: Muestras de solapamiento
        nchans: Número de canales
        nifs: Número de IFs
        nbits: Bits por muestra
        dtype: Tipo de datos
    """
    logger = get_global_logger()
    logger.logger.debug(f"🔍 STREAM_FIL: nsamples={nsamples:,}, chunk_samples={chunk_samples:,}, overlap={overlap_samples}")
    logger.logger.debug(f"🔍 STREAM_FIL: nchans={nchans}, nifs={nifs}, nbits={nbits}, dtype={dtype}")


def log_stream_fil_block_generation(chunk_counter: int, block_shape: tuple, block_dtype: str,
                                   valid_start: int, valid_end: int, 
                                   start_with_overlap: int, end_with_overlap: int,
                                   actual_chunk_size: int) -> None:
    """
    Registra la generación de un bloque en streaming FIL.
    
    Args:
        chunk_counter: Contador del chunk
        block_shape: Forma del bloque
        block_dtype: Tipo de datos del bloque
        valid_start: Inicio de la región válida
        valid_end: Fin de la región válida
        start_with_overlap: Inicio con solapamiento
        end_with_overlap: Fin con solapamiento
        actual_chunk_size: Tamaño real del chunk
    """
    logger = get_global_logger()
    logger.logger.debug(f"🔍 STREAM_FIL BLOQUE {chunk_counter}: shape={block_shape}, dtype={block_dtype}")
    logger.logger.debug(f"🔍 STREAM_FIL BLOQUE {chunk_counter}: valid_range=({valid_start:,}, {valid_end:,})")
    logger.logger.debug(f"🔍 STREAM_FIL BLOQUE {chunk_counter}: overlap_range=({start_with_overlap:,}, {end_with_overlap:,})")
    logger.logger.debug(f"🔍 STREAM_FIL BLOQUE {chunk_counter}: actual_chunk_size={actual_chunk_size:,}")


def log_stream_fil_summary(chunk_counter: int) -> None:
    """
    Registra el resumen de streaming FIL.
    
    Args:
        chunk_counter: Número total de chunks generados
    """
    logger = get_global_logger()
    logger.logger.debug(f"🔍 STREAM_FIL RESUMEN: {chunk_counter} bloques generados exitosamente")
    logger.logger.debug(f"🔍 STREAM_FIL RESUMEN: archivo procesado completamente a través de streaming")


def log_stream_fits_parameters(nsamples: int, chunk_samples: int, overlap_samples: int,
                              nsubint: Optional[int], nchan: Optional[int], 
                              npol: Optional[int], nsblk: Optional[int]) -> None:
    """
    Registra los parámetros de streaming para archivos FITS.
    
    Args:
        nsamples: Número total de muestras
        chunk_samples: Tamaño del chunk
        overlap_samples: Muestras de solapamiento
        nsubint: Número de subintegraciones
        nchan: Número de canales
        npol: Número de polarizaciones
        nsblk: Número de muestras por bloque
    """
    logger = get_global_logger()
    logger.logger.debug(f"🔍 STREAM_FITS: nsamples={nsamples:,}, chunk_samples={chunk_samples:,}, overlap={overlap_samples}")
    logger.logger.debug(f"🔍 STREAM_FITS: nsubint={nsubint if nsubint is not None else 'N/A'}, nchan={nchan if nchan is not None else 'N/A'}")
    logger.logger.debug(f"🔍 STREAM_FITS: npol={npol if npol is not None else 'N/A'}, nsblk={nsblk if nsblk is not None else 'N/A'}")


def log_stream_fits_load_strategy(use_memmap: bool, data_shape: tuple, data_dtype: str) -> None:
    """
    Registra la estrategia de carga para archivos FITS.
    
    Args:
        use_memmap: Si se usa memmap
        data_shape: Forma de los datos
        data_dtype: Tipo de datos
    """
    logger = get_global_logger()
    logger.logger.debug(f"🔍 STREAM_FITS CARGA: use_memmap={use_memmap}, data_array.shape={data_shape}")
    logger.logger.debug(f"🔍 STREAM_FITS CARGA: data_array.dtype={data_dtype}")


def log_stream_fits_block_generation(chunk_counter: int, block_shape: tuple, block_dtype: str,
                                    valid_start: int, valid_end: int,
                                    start_with_overlap: int, end_with_overlap: int,
                                    actual_chunk_size: int) -> None:
    """
    Registra la generación de un bloque en streaming FITS.
    
    Args:
        chunk_counter: Contador del chunk
        block_shape: Forma del bloque
        block_dtype: Tipo de datos del bloque
        valid_start: Inicio de la región válida
        valid_end: Fin de la región válida
        start_with_overlap: Inicio con solapamiento
        end_with_overlap: Fin con solapamiento
        actual_chunk_size: Tamaño real del chunk
    """
    logger = get_global_logger()
    logger.logger.debug(f"🔍 STREAM_FITS BLOQUE {chunk_counter}: shape={block_shape}, dtype={block_dtype}")
    logger.logger.debug(f"🔍 STREAM_FITS BLOQUE {chunk_counter}: valid_range=({valid_start:,}, {valid_end:,})")
    logger.logger.debug(f"🔍 STREAM_FITS BLOQUE {chunk_counter}: overlap_range=({start_with_overlap:,}, {end_with_overlap:,})")
    logger.logger.debug(f"🔍 STREAM_FITS BLOQUE {chunk_counter}: actual_chunk_size={actual_chunk_size:,}")


def log_stream_fits_summary(chunk_counter: int) -> None:
    """
    Registra el resumen de streaming FITS.
    
    Args:
        chunk_counter: Número total de chunks generados
    """
    logger = get_global_logger()
    logger.logger.debug(f"🔍 STREAM_FITS RESUMEN: {chunk_counter} bloques generados exitosamente")
    logger.logger.debug(f"🔍 STREAM_FITS RESUMEN: archivo procesado completamente a través de streaming")
