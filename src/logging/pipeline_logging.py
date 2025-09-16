# This module logs high-level pipeline execution events.

"""
Manejo de Logging para Pipeline en DRAFTS
=========================================

Este módulo proporciona funciones especializadas para mostrar información
detallada del pipeline, especialmente para operaciones de chunking y
procesamiento de archivos.
"""

                          
from typing import Any, Callable, Dict, Optional

               
from .logging_config import get_global_logger


def log_streaming_parameters(effective_chunk_samples: int, overlap_raw: int,
                            total_samples: int, chunk_samples: int,
                            streaming_func: Callable, file_type: str) -> None:
    """
    Registra los parámetros de streaming en el pipeline.
    
    Args:
        effective_chunk_samples: Tamaño efectivo del chunk
        overlap_raw: Solapamiento en muestras raw
        total_samples: Total de muestras del archivo
        chunk_samples: Tamaño configurado del chunk
        streaming_func: Función de streaming utilizada
        file_type: Tipo de archivo detectado
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


def log_block_processing(actual_chunk_count: int, block_shape: tuple, block_dtype: str,
                         metadata: Dict[str, Any]) -> None:
    """
    Registra el procesamiento de un bloque en el pipeline.
    
    Args:
        actual_chunk_count: Contador del chunk actual
        block_shape: Forma del bloque
        block_dtype: Tipo de datos del bloque
        metadata: Metadatos del bloque
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


def log_processing_summary(actual_chunk_count: int, chunk_count: int,
                          cand_counter_total: int, n_bursts_total: int) -> None:
    """
    Registra el resumen del procesamiento en el pipeline.
    
    Args:
        actual_chunk_count: Número de chunks procesados
        chunk_count: Número total estimado de chunks
        cand_counter_total: Total de candidatos encontrados
        n_bursts_total: Total de bursts detectados
    """
    logger = get_global_logger().logger
    logger.info(
        "Progress • chunks=%d/%d • candidates=%d • bursts=%d",
        actual_chunk_count,
        chunk_count,
        cand_counter_total,
        n_bursts_total,
    )


def log_pipeline_file_processing(fits_path_name: str, file_suffix: str,
                                total_samples: int, chunk_samples: int) -> None:
    """
    Registra el inicio del procesamiento de un archivo en el pipeline.
    
    Args:
        fits_path_name: Nombre del archivo
        file_suffix: Extensión del archivo
        total_samples: Total de muestras del archivo
        chunk_samples: Tamaño configurado del chunk
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
    """
    Registra la finalización del procesamiento de un archivo en el pipeline.
    
    Args:
        fits_path_name: Nombre del archivo
        results: Resultados del procesamiento
    """
    logger = get_global_logger().logger
    logger.debug(
        "File completed • %s • status=%s • chunks=%s • mode=%s",
        fits_path_name,
        results.get("status", "N/A"),
        results.get("chunks_processed", "N/A"),
        results.get("processing_mode", "N/A"),
    )
