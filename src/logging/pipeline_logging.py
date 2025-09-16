"""
Manejo de Logging para Pipeline en DRAFTS
=========================================

Este módulo proporciona funciones especializadas para mostrar información
detallada del pipeline, especialmente para operaciones de chunking y
procesamiento de archivos.
"""

# Standard library imports
from typing import Any, Callable, Dict, Optional

# Local imports
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
    logger = get_global_logger()
    logger.logger.debug(f"🔍 DEBUG STREAMING: effective_chunk_samples={effective_chunk_samples:,}, overlap_raw={overlap_raw}")
    logger.logger.debug(f"🔍 DEBUG STREAMING: total_samples={total_samples:,}, chunk_samples={chunk_samples:,}")
    logger.logger.debug(f"🔍 DEBUG STREAMING: streaming_func={streaming_func.__name__}, file_type={file_type}")


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
    logger = get_global_logger()
    logger.logger.debug(f"🔍 DEBUG BLOQUE {actual_chunk_count}: shape={block_shape}, dtype={block_dtype}")
    logger.logger.debug(f"🔍 DEBUG BLOQUE {actual_chunk_count}: metadata={metadata}")
    logger.logger.debug(f"🔍 DEBUG BLOQUE {actual_chunk_count}: chunk_idx={metadata.get('chunk_idx', 'N/A')}")
    logger.logger.debug(f"🔍 DEBUG BLOQUE {actual_chunk_count}: start_sample={metadata.get('start_sample', 'N/A'):,}")
    logger.logger.debug(f"🔍 DEBUG BLOQUE {actual_chunk_count}: end_sample={metadata.get('end_sample', 'N/A'):,}")
    logger.logger.debug(f"🔍 DEBUG BLOQUE {actual_chunk_count}: overlap_left={metadata.get('overlap_left', 'N/A')}")
    logger.logger.debug(f"🔍 DEBUG BLOQUE {actual_chunk_count}: overlap_right={metadata.get('overlap_right', 'N/A')}")


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
    logger = get_global_logger()
    logger.logger.debug(f"🔍 DEBUG RESUMEN: chunks procesados={actual_chunk_count}, total estimado={chunk_count}")
    logger.logger.debug(f"🔍 DEBUG RESUMEN: candidatos totales={cand_counter_total}, bursts={n_bursts_total}")
    logger.logger.debug(f"🔍 DEBUG RESUMEN: archivo procesado completamente a través de streaming")


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
    logger = get_global_logger()
    logger.logger.debug(f"🔍 DEBUG PIPELINE: Procesando archivo: {fits_path_name}")
    logger.logger.debug(f"🔍 DEBUG PIPELINE: Tipo de archivo: {file_suffix}")
    logger.logger.debug(f"🔍 DEBUG PIPELINE: Muestras totales: {total_samples:,}")
    logger.logger.debug(f"🔍 DEBUG PIPELINE: chunk_samples configurado: {chunk_samples:,}")
    logger.logger.debug(f"🔍 DEBUG PIPELINE: SIEMPRE llamando a _process_file_chunked (nunca a _process_file)")


def log_pipeline_file_completion(fits_path_name: str, results: Dict[str, Any]) -> None:
    """
    Registra la finalización del procesamiento de un archivo en el pipeline.
    
    Args:
        fits_path_name: Nombre del archivo
        results: Resultados del procesamiento
    """
    logger = get_global_logger()
    logger.logger.debug(f"🔍 DEBUG PIPELINE: Archivo {fits_path_name} procesado exitosamente")
    logger.logger.debug(f"🔍 DEBUG PIPELINE: Status: {results.get('status', 'N/A')}")
    logger.logger.debug(f"🔍 DEBUG PIPELINE: Chunks procesados: {results.get('chunks_processed', 'N/A')}")
    logger.logger.debug(f"🔍 DEBUG PIPELINE: Modo de procesamiento: {results.get('processing_mode', 'N/A')}")
