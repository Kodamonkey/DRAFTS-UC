# This module logs chunk planning and slice configuration details.

"""
Manejo de Logging para el Sistema de Chunking en DRAFTS
======================================================

Este módulo proporciona funciones especializadas para mostrar información
detallada del sistema de chunking de manera organizada y profesional.
"""

                          
from typing import Any, Dict

               
from .logging_config import get_global_logger


def display_detailed_chunking_info(parameters: Dict[str, Any]) -> None:
    """Emit a concise overview of the chunking plan."""

    logger = get_global_logger().logger

    logger.info(
        "Chunking overview • target_slice=%.1f ms (%d samples) • downsample t=%dx f=%dx",
        parameters["slice_duration_ms_target"],
        parameters["samples_per_slice"],
        parameters["down_time_rate"],
        parameters["down_freq_rate"],
    )

    logger.info(
        "Input data • samples=%s • channels=%d • duration=%.1fs (%.1f min)",
        f"{parameters['total_samples_original']:,}",
        parameters["total_channels_original"],
        parameters["total_duration_sec"],
        parameters["total_duration_min"],
    )

    logger.info(
        "After downsampling • samples=%s • channels=%d • Δt=%.3f ms • slices/chunk≈%d",
        f"{parameters['total_samples_downsampled']:,}",
        parameters["total_channels_downsampled"],
        parameters["time_reso_downsampled"] * 1e3,
        parameters["slices_per_chunk"],
    )

    logger.info(
        "Chunks • size=%s samples (~%.1fs) • estimated=%d chunks • total_slices=%d",
        f"{parameters['chunk_samples']:,}",
        parameters["chunk_duration_sec"],
        parameters["total_chunks"],
        parameters["total_slices"],
    )

    logger.info(
        "Memory • file=%.2f GB • chunk=%.2f GB • available=%.1f/%.1f GB • optimized=%s",
        parameters["total_file_size_gb"],
        parameters["chunk_size_gb"],
        parameters["available_memory_gb"],
        parameters["total_memory_gb"],
        "yes" if parameters["memory_optimized"] else "no",
    )

    if parameters["has_leftover_samples"]:
        leftover_time_ms = parameters["leftover_samples"] * parameters["time_reso_downsampled"] * 1000
        logger.warning(
            "Residual samples • %s (~%.1f ms) will be skipped",
            f"{parameters['leftover_samples']:,}",
            leftover_time_ms,
        )


def log_chunk_processing_start(chunk_idx: int, chunk_info: Dict[str, Any]) -> None:
    """
    Registra el inicio del procesamiento de un chunk.

    Args:
        chunk_idx: Índice del chunk
        chunk_info: Información del chunk
    """
    logger = get_global_logger().logger
    logger.info(
        "Chunk %03d/%03d • samples=%s • slices=%d",
        chunk_idx,
        chunk_info.get("total_chunks", 0),
        f"{chunk_info.get('chunk_samples', 0):,}",
        chunk_info.get("slices_per_chunk", 0),
    )


def log_chunk_processing_end(chunk_idx: int, results: Dict[str, Any]) -> None:
    """
    Registra el fin del procesamiento de un chunk.
    
    Args:
        chunk_idx: Índice del chunk
        results: Resultados del procesamiento
    """
    logger = get_global_logger().logger
    logger.info(
        "Chunk %03d finished • candidates=%d • runtime=%.2fs",
        chunk_idx,
        results.get("candidates", 0),
        results.get("processing_time", 0.0),
    )


def log_file_processing_summary(file_info: Dict[str, Any]) -> None:
    """
    Registra un resumen del procesamiento del archivo.
    
    Args:
        file_info: Información del archivo procesado
    """
    logger = get_global_logger().logger
    logger.info(
        "File summary • %s • chunks=%d • slices=%s • candidates=%d • runtime=%.2fs",
        file_info.get("filename", "N/A"),
        file_info.get("total_chunks", 0),
        f"{file_info.get('total_slices', 0):,}",
        file_info.get("total_candidates", 0),
        file_info.get("total_time", 0.0),
    )


def log_memory_optimization(optimization_info: Dict[str, Any]) -> None:
    """
    Registra información sobre optimizaciones de memoria.
    
    Args:
        optimization_info: Información de optimización
    """
    logger = get_global_logger().logger
    logger.debug(
        "Memory optimisation • strategy=%s • used=%.2f GB • efficiency=%.1f%%",
        optimization_info.get("strategy", "N/A"),
        optimization_info.get("memory_used_gb", 0.0),
        optimization_info.get("efficiency_percent", 0.0),
    )


def log_slice_configuration(slice_config: Dict[str, Any]) -> None:
    """
    Registra la configuración de slices.
    
    Args:
        slice_config: Configuración de slices
    """
    logger = get_global_logger().logger
    logger.info(
        "Slice configuration • target=%.1f ms • samples=%s • actual=%.1f ms • accuracy=%.1f%%",
        slice_config.get("target_duration_ms", 0.0),
        f"{slice_config.get('samples_per_slice', 0):,}",
        slice_config.get("real_duration_ms", 0.0),
        slice_config.get("precision_percent", 0.0),
    )


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
    logger = get_global_logger().logger
    logger.info(
        "Chunk budget • bytes/sample=%s • usable=%.2f/%.2f GB • nsamp_max=%s → chunk=%s • downsample t=%dx f=%dx",
        f"{budget.get('bytes_per_sample', 0):,}",
        budget.get('usable_bytes', 0) / (1024 ** 3),
        budget.get('available_bytes', 0) / (1024 ** 3),
        f"{budget.get('nsamp_max_aligned', 0):,}",
        f"{budget.get('chunk_samples', 0):,}",
        budget.get('down_time_rate', 1),
        budget.get('down_freq_rate', 1),
    )


def log_slice_plan_summary(chunk_idx: int, plan: Dict[str, Any]) -> None:
    """Registra un resumen del plan de slices para un chunk.

    Muestra número de slices, duración media y algunos ejemplos de límites.
    """
    logger = get_global_logger().logger
    n = plan.get("n_slices", 0)
    avg_ms = plan.get("avg_ms", 0.0)
    delta_ms = plan.get("delta_ms", 0.0)
    slices = plan.get("slices", [])

    if slices:
        lengths = [sl.length for sl in slices]
        min_len = min(lengths)
        max_len = max(lengths)
        median_len = sorted(lengths)[len(lengths) // 2]
    else:
        min_len = max_len = median_len = 0

    logger.info(
        "Slice plan • chunk=%03d • count=%d • avg=%.3f ms (Δ=%.3f ms) • length[min/median/max]=%d/%d/%d",
        chunk_idx,
        n,
        avg_ms,
        delta_ms,
        min_len,
        median_len,
        max_len,
    )

    if slices:
        preview_indices = [idx for idx in (0, 1, len(slices) - 1) if 0 <= idx < len(slices)]
        preview_lines = []
        for idx in preview_indices:
            sl = slices[idx]
            preview_lines.append(
                f"[{idx:04d}] {sl.start_idx}→{sl.end_idx} ({sl.length} samples, {sl.duration_ms:.3f} ms)"
            )
        if preview_lines:
            logger.debug("Slice plan preview • %s", " | ".join(preview_lines))
