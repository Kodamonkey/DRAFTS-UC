"""Slice length calculator for FRB pipeline - dynamically calculates optimal temporal segmentation."""
import logging
from typing import Tuple, Optional
import numpy as np

from .. import config

logger = logging.getLogger(__name__)


def calculate_slice_len_from_duration() -> Tuple[int, float]:
    """
    Calcula SLICE_LEN dinámicamente basado en SLICE_DURATION_MS y metadatos del archivo.
    
    Fórmula inversa: SLICE_LEN = round(SLICE_DURATION_MS / (TIME_RESO × DOWN_TIME_RATE × 1000))
    
    Returns:
        Tuple[int, float]: (slice_len_calculado, duracion_real_ms)
    """
    if config.TIME_RESO <= 0:
        logger.warning("TIME_RESO no está configurado, usando SLICE_LEN_MIN")
        return config.SLICE_LEN_MIN, config.SLICE_DURATION_MS
    
    # 🕐 CORRECCIÓN: SLICE_LEN se calcula para datos YA decimados
    # Por lo tanto, usar TIME_RESO * DOWN_TIME_RATE
    target_duration_s = config.SLICE_DURATION_MS / 1000.0
    calculated_slice_len = round(target_duration_s / (config.TIME_RESO * config.DOWN_TIME_RATE))
    
    # Aplicar límites mín/máx
    slice_len = max(config.SLICE_LEN_MIN, min(config.SLICE_LEN_MAX, calculated_slice_len))
    
    # Calcular duración real obtenida (para datos decimados)
    real_duration_s = slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
    real_duration_ms = real_duration_s * 1000.0
    
    # Actualizar config.SLICE_LEN con el valor calculado
    config.SLICE_LEN = slice_len
    
    logger.info(f"🎯 Duración objetivo: {config.SLICE_DURATION_MS:.1f} ms")
    logger.info(f"📏 SLICE_LEN calculado: {slice_len} muestras")
    logger.info(f"⏱️  Duración real obtenida: {real_duration_ms:.1f} ms")
    
    # 🕐 DEBUG: Verificar cálculo
    logger.info(f"🕐 [DEBUG SLICE_LEN] Cálculo:")
    logger.info(f"   ⏱️  TIME_RESO: {config.TIME_RESO}")
    logger.info(f"   🔽 DOWN_TIME_RATE: {config.DOWN_TIME_RATE}")
    logger.info(f"   📏 SLICE_LEN = {target_duration_s:.6f}s ÷ {config.TIME_RESO} = {calculated_slice_len}")
    logger.info(f"   📊 Para un archivo de 2M muestras decimadas: {2000000 // config.DOWN_TIME_RATE // slice_len} slices")
    
    if abs(real_duration_ms - config.SLICE_DURATION_MS) > 5.0:
        logger.warning(f"⚠️  Diferencia significativa entre objetivo ({config.SLICE_DURATION_MS:.1f} ms) "
                      f"y obtenido ({real_duration_ms:.1f} ms)")
    
    return slice_len, real_duration_ms


def update_slice_len_dynamic():
    """
    Actualiza config.SLICE_LEN basado en SLICE_DURATION_MS.
    Debe llamarse después de cargar metadatos del archivo.
    """
    slice_len, real_duration_ms = calculate_slice_len_from_duration()
    logger.info(f"✅ Slice configurado: {slice_len} muestras = {real_duration_ms:.1f} ms")
    return slice_len, real_duration_ms

