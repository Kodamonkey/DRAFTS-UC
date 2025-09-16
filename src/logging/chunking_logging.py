# This module logs chunk planning and slice configuration details.

"""
Manejo de Logging para el Sistema de Chunking en DRAFTS
======================================================

Este módulo proporciona funciones especializadas para mostrar información
detallada del sistema de chunking de manera organizada y profesional.
"""

                          
from typing import Any, Dict

               
from .logging_config import get_global_logger


# This function displays detailed chunking info.
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
    
                                                                                   
                            
                                                                                   
    logger.logger.info("PARÁMETROS DEL USUARIO:")
    logger.logger.info(f"   • Duración deseada de slice: {parameters['slice_duration_ms_target']:.1f} ms")
    logger.logger.info(f"   • Factor de downsampling temporal: {parameters['down_time_rate']}x")
    logger.logger.info(f"   • Factor de downsampling frecuencial: {parameters['down_freq_rate']}x")
    
                                                                                   
                                     
                                                                                   
    logger.logger.info("PARÁMETROS DEL ARCHIVO ORIGINAL:")
    logger.logger.info(f"   • Muestras totales: {parameters['total_samples_original']:,}")
    logger.logger.info(f"   • Canales totales: {parameters['total_channels_original']:,}")
    logger.logger.info(f"   • Resolución temporal: {parameters['time_reso_original']:.2e} s")
    logger.logger.info(f"   • Duración total: {parameters['total_duration_min']:.1f} min ({parameters['total_duration_sec']:.1f} s)")
    
                                                                                   
                                         
                                                                                   
    logger.logger.info("PARÁMETROS DESPUÉS DEL DOWNSAMPLING:")
    logger.logger.info(f"   • Muestras con downsampling: {parameters['total_samples_downsampled']:,}")
    logger.logger.info(f"   • Canales con downsampling: {parameters['total_channels_downsampled']:,}")
    logger.logger.info(f"   • Resolución temporal: {parameters['time_reso_downsampled']:.2e} s")
    
                                                                                   
                           
                                                                                   
    logger.logger.info("PARÁMETROS CALCULADOS (previos y alineados a streaming):")
    logger.logger.info(f"   • Muestras por slice (decimado): {parameters['samples_per_slice']:,}")
    logger.logger.info(f"   • Muestras por chunk (decimado teórico): {parameters['chunk_samples']:,}")
    logger.logger.info(f"   • Muestras por chunk (original para streaming): {parameters.get('chunk_samples_stream_original', 0):,}")
    logger.logger.info(f"   • Muestras por chunk (decimado efectivo en streaming): {parameters.get('chunk_samples_stream_decimated', 0):,}")
    logger.logger.info(f"   • Slices por chunk (preview teórico): {parameters['slices_per_chunk']}")
    logger.logger.info(f"   • Slices por chunk (preview alineado a streaming): {parameters.get('slices_per_chunk_stream_estimate', 0)}")
    logger.logger.info(f"   • Duración de chunk (teórica, decimado): {parameters['chunk_duration_sec']:.3f} s")
    logger.logger.info(f"   • Duración de chunk (streaming, original): {parameters.get('chunk_duration_stream_sec', 0.0):.3f} s")
    logger.logger.info(f"   • Total de chunks (estimado): {parameters['total_chunks']}")
    logger.logger.info(f"   • Total de slices (global): {parameters['total_slices']:,}")
    logger.logger.info("Nota: El planificador por chunk imprimirá el número real por chunk en ejecución.")
    
                                                                                   
                
                                                                                   
    logger.logger.info("DURACIONES:")
    logger.logger.info(f"   • Duración real de slice: {parameters['slice_duration_ms_real']:.1f} ms")
    logger.logger.info(f"   • Duración de chunk: {parameters['chunk_duration_sec']:.1f} s")
    
                                                                                   
                            
                                                                                   
    logger.logger.info("INFORMACIÓN DE MEMORIA:")
    logger.logger.info(f"   • Tamaño del archivo: {parameters['total_file_size_gb']:.2f} GB")
    logger.logger.info(f"   • Tamaño por chunk: {parameters['chunk_size_gb']:.2f} GB")
    logger.logger.info(f"   • Memoria disponible: {parameters['available_memory_gb']:.1f} GB")
    logger.logger.info(f"   • Memoria total: {parameters['total_memory_gb']:.1f} GB")
    logger.logger.info(f"   • ¿Puede cargar archivo completo?: {'✅ Sí' if parameters['can_load_full_file'] else '❌ No'}")
    
                                                                                   
                        
                                                                                   
    logger.logger.info("ESTADO DEL SISTEMA:")
    logger.logger.info(f"   • Modo optimizado de memoria: {'✅ Activado' if parameters['memory_optimized'] else '❌ Desactivado'}")
    
                                                                                   
                         
                                                                                   
    if parameters['has_leftover_samples']:
        leftover_time_ms = parameters['leftover_samples'] * parameters['time_reso_downsampled'] * 1000
        logger.logger.warning("ADVERTENCIA - MUESTRAS RESIDUALES:")
        logger.logger.warning(f"   • Muestras sobrantes en el último chunk: {parameters['leftover_samples']:,}")
        logger.logger.warning(f"   • Tiempo sobrante: {leftover_time_ms:.1f} ms")
        logger.logger.warning(f"   • Estas muestras NO formarán un slice completo")
    
    logger.logger.info("=" * 80)


# This function logs chunk processing start.
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


# This function logs chunk processing end.
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


# This function logs file processing summary.
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


# This function logs memory optimization.
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


# This function logs slice configuration.
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


# This function logs chunk budget.
def log_chunk_budget(budget: Dict[str, Any]) -> None:
    """
    Registra el presupuesto de memoria y cálculo de tamaño de chunk cuando el planificador está activo.
    
    Args:
        budget: Dict con llaves esperadas:
            - bytes_per_sample
            - available_bytes
            - usable_bytes
            - nsamp_max_raw
            - nsamp_max_aligned
            - slice_len
            - chunk_samples
            - down_time_rate
            - down_freq_rate
    """
    logger = get_global_logger()
    logger.logger.info("PRESUPUESTO DE CHUNKING (planificador)")
    logger.logger.info(f"   • Bytes por muestra: {budget.get('bytes_per_sample', 0):,}")
    logger.logger.info(f"   • Memoria disponible: {budget.get('available_bytes', 0) / (1024**3):.2f} GB")
    logger.logger.info(f"   • Usable tras overhead: {budget.get('usable_bytes', 0) / (1024**3):.2f} GB")
    logger.logger.info(f"   • nsamp_max (raw): {budget.get('nsamp_max_raw', 0):,}")
    logger.logger.info(f"   • nsamp_max alineado a slice_len: {budget.get('nsamp_max_aligned', 0):,}")
    logger.logger.info(f"   • slice_len: {budget.get('slice_len', 0):,} muestras")
    logger.logger.info(f"   • chunk_samples final: {budget.get('chunk_samples', 0):,} muestras")
    logger.logger.info(f"   • Downsampling: tiempo={budget.get('down_time_rate', 1)}x, freq={budget.get('down_freq_rate', 1)}x")


# This function logs slice plan summary.
def log_slice_plan_summary(chunk_idx: int, plan: Dict[str, Any]) -> None:
    """Registra un resumen del plan de slices para un chunk.

    Muestra número de slices, duración media y algunos ejemplos de límites.
    """
    logger = get_global_logger()
    n = plan.get("n_slices", 0)
    avg_ms = plan.get("avg_ms", 0.0)
    delta_ms = plan.get("delta_ms", 0.0)
    slices = plan.get("slices", [])
    logger.logger.info(f"PLAN DE SLICES (chunk {chunk_idx:03d})")
    logger.logger.info(f"   • n_slices: {n}")
    logger.logger.info(f"   • duración media por slice: {avg_ms:.6f} ms (Δ={delta_ms:+.6f} ms respecto a objetivo)")
    if slices:
        lengths = [sl.length for sl in slices]
        logger.logger.info(f"   • tamaño de slice (muestras): min={min(lengths)}, max={max(lengths)}, mediana={sorted(lengths)[len(lengths)//2]}")
                                                       
        preview = []
        for idx in [0, 1, len(slices) - 1]:
            if 0 <= idx < len(slices):
                sl = slices[idx]
                preview.append(f"[{idx:04d}] {sl.start_idx}→{sl.end_idx} ({sl.length} muestras, {sl.duration_ms:.6f} ms)")
        if preview:
            logger.logger.info("   • ejemplos: ")
            for line in preview:
                logger.logger.info(f"      {line}")
