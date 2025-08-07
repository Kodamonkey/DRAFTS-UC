"""
Manejo de Logging para el Sistema de Chunking en DRAFTS
======================================================

Este módulo proporciona funciones especializadas para mostrar información
detallada del sistema de chunking de manera organizada y profesional.
"""

from typing import Dict, Any
from .logging_config import get_global_logger


def display_detailed_chunking_info(parameters: Dict[str, Any]) -> None:
    """
    Muestra información detallada del sistema de chunking en consola.
    
    Args:
        parameters: Diccionario con parámetros de procesamiento calculados
    """
    logger = get_global_logger()
    
    logger.logger.info("=" * 80)
    logger.logger.info("SISTEMA DE CHUNKING DETALLADO")
    logger.logger.info("=" * 80)
    
    # =============================================================================
    # PARÁMETROS DEL USUARIO
    # =============================================================================
    logger.logger.info("PARÁMETROS DEL USUARIO:")
    logger.logger.info(f"   • Duración deseada de slice: {parameters['slice_duration_ms_target']:.1f} ms")
    logger.logger.info(f"   • Factor de downsampling temporal: {parameters['down_time_rate']}x")
    logger.logger.info(f"   • Factor de downsampling frecuencial: {parameters['down_freq_rate']}x")
    
    # =============================================================================
    # PARÁMETROS DEL ARCHIVO ORIGINAL
    # =============================================================================
    logger.logger.info("PARÁMETROS DEL ARCHIVO ORIGINAL:")
    logger.logger.info(f"   • Muestras totales: {parameters['total_samples_original']:,}")
    logger.logger.info(f"   • Canales totales: {parameters['total_channels_original']:,}")
    logger.logger.info(f"   • Resolución temporal: {parameters['time_reso_original']:.2e} s")
    logger.logger.info(f"   • Duración total: {parameters['total_duration_min']:.1f} min ({parameters['total_duration_sec']:.1f} s)")
    
    # =============================================================================
    # PARÁMETROS DESPUÉS DEL DOWNSAMPLING
    # =============================================================================
    logger.logger.info("PARÁMETROS DESPUÉS DEL DOWNSAMPLING:")
    logger.logger.info(f"   • Muestras con downsampling: {parameters['total_samples_downsampled']:,}")
    logger.logger.info(f"   • Canales con downsampling: {parameters['total_channels_downsampled']:,}")
    logger.logger.info(f"   • Resolución temporal: {parameters['time_reso_downsampled']:.2e} s")
    
    # =============================================================================
    # PARÁMETROS CALCULADOS
    # =============================================================================
    logger.logger.info("PARÁMETROS CALCULADOS:")
    logger.logger.info(f"   • Muestras por slice: {parameters['samples_per_slice']:,}")
    logger.logger.info(f"   • Muestras por chunk: {parameters['chunk_samples']:,}")
    logger.logger.info(f"   • Total de chunks: {parameters['total_chunks']}")
    logger.logger.info(f"   • Slices por chunk: {parameters['slices_per_chunk']}")
    logger.logger.info(f"   • Total de slices: {parameters['total_slices']:,}")
    
    # =============================================================================
    # DURACIONES
    # =============================================================================
    logger.logger.info("DURACIONES:")
    logger.logger.info(f"   • Duración real de slice: {parameters['slice_duration_ms_real']:.1f} ms")
    logger.logger.info(f"   • Duración de chunk: {parameters['chunk_duration_sec']:.1f} s")
    
    # =============================================================================
    # INFORMACIÓN DE MEMORIA
    # =============================================================================
    logger.logger.info("INFORMACIÓN DE MEMORIA:")
    logger.logger.info(f"   • Tamaño del archivo: {parameters['total_file_size_gb']:.2f} GB")
    logger.logger.info(f"   • Tamaño por chunk: {parameters['chunk_size_gb']:.2f} GB")
    logger.logger.info(f"   • Memoria disponible: {parameters['available_memory_gb']:.1f} GB")
    logger.logger.info(f"   • Memoria total: {parameters['total_memory_gb']:.1f} GB")
    logger.logger.info(f"   • ¿Puede cargar archivo completo?: {'✅ Sí' if parameters['can_load_full_file'] else '❌ No'}")
    
    # =============================================================================
    # ESTADO DEL SISTEMA
    # =============================================================================
    logger.logger.info("ESTADO DEL SISTEMA:")
    logger.logger.info(f"   • Modo optimizado de memoria: {'✅ Activado' if parameters['memory_optimized'] else '❌ Desactivado'}")
    
    # =============================================================================
    # MUESTRAS RESIDUALES
    # =============================================================================
    if parameters['has_leftover_samples']:
        leftover_time_ms = parameters['leftover_samples'] * parameters['time_reso_downsampled'] * 1000
        logger.logger.warning("ADVERTENCIA - MUESTRAS RESIDUALES:")
        logger.logger.warning(f"   • Muestras sobrantes en el último chunk: {parameters['leftover_samples']:,}")
        logger.logger.warning(f"   • Tiempo sobrante: {leftover_time_ms:.1f} ms")
        logger.logger.warning(f"   • Estas muestras NO formarán un slice completo")
    
    logger.logger.info("=" * 80)


def log_chunk_processing_start(chunk_idx: int, chunk_info: Dict[str, Any]) -> None:
    """
    Registra el inicio del procesamiento de un chunk.
    
    Args:
        chunk_idx: Índice del chunk
        chunk_info: Información del chunk
    """
    logger = get_global_logger()
    logger.logger.info(f"Iniciando procesamiento del chunk {chunk_idx:03d}/{chunk_info.get('total_chunks', 0)}")
    logger.logger.info(f"   • Muestras en chunk: {chunk_info.get('chunk_samples', 0):,}")
    logger.logger.info(f"   • Slices en chunk: {chunk_info.get('slices_per_chunk', 0)}")


def log_chunk_processing_end(chunk_idx: int, results: Dict[str, Any]) -> None:
    """
    Registra el fin del procesamiento de un chunk.
    
    Args:
        chunk_idx: Índice del chunk
        results: Resultados del procesamiento
    """
    logger = get_global_logger()
    logger.logger.info(f"Chunk {chunk_idx:03d} procesado exitosamente")
    logger.logger.info(f"   • Candidatos encontrados: {results.get('candidates', 0)}")
    logger.logger.info(f"   • Tiempo de procesamiento: {results.get('processing_time', 0):.2f} s")


def log_file_processing_summary(file_info: Dict[str, Any]) -> None:
    """
    Registra un resumen del procesamiento del archivo.
    
    Args:
        file_info: Información del archivo procesado
    """
    logger = get_global_logger()
    logger.logger.info("RESUMEN DEL PROCESAMIENTO:")
    logger.logger.info(f"   • Archivo: {file_info.get('filename', 'N/A')}")
    logger.logger.info(f"   • Chunks procesados: {file_info.get('total_chunks', 0)}")
    logger.logger.info(f"   • Slices procesados: {file_info.get('total_slices', 0):,}")
    logger.logger.info(f"   • Candidatos totales: {file_info.get('total_candidates', 0)}")
    logger.logger.info(f"   • Tiempo total: {file_info.get('total_time', 0):.2f} s")


def log_memory_optimization(optimization_info: Dict[str, Any]) -> None:
    """
    Registra información sobre optimizaciones de memoria.
    
    Args:
        optimization_info: Información de optimización
    """
    logger = get_global_logger()
    logger.logger.info("OPTIMIZACIÓN DE MEMORIA:")
    logger.logger.info(f"   • Estrategia: {optimization_info.get('strategy', 'N/A')}")
    logger.logger.info(f"   • Memoria utilizada: {optimization_info.get('memory_used_gb', 0):.2f} GB")
    logger.logger.info(f"   • Eficiencia: {optimization_info.get('efficiency_percent', 0):.1f}%")


def log_slice_configuration(slice_config: Dict[str, Any]) -> None:
    """
    Registra la configuración de slices.
    
    Args:
        slice_config: Configuración de slices
    """
    logger = get_global_logger()
    logger.logger.info("CONFIGURACIÓN DE SLICES:")
    logger.logger.info(f"   • Duración objetivo: {slice_config.get('target_duration_ms', 0):.1f} ms")
    logger.logger.info(f"   • Muestras por slice: {slice_config.get('samples_per_slice', 0):,}")
    logger.logger.info(f"   • Duración real: {slice_config.get('real_duration_ms', 0):.1f} ms")
    logger.logger.info(f"   • Precisión: {slice_config.get('precision_percent', 0):.1f}%")
