"""
Utilidades para cálculo dinámico de SLICE_LEN basado en duración objetivo.
Nuevo sistema simplificado: SIEMPRE usa SLICE_DURATION_MS como parámetro único.
"""
import logging
from typing import Tuple, Optional
import numpy as np

from . import config

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


def get_slice_duration_info(slice_len: int) -> Tuple[float, str]:
    """
    Convierte SLICE_LEN a información de duración para display.
    
    Args:
        slice_len: Número de muestras
        
    Returns:
        Tuple[float, str]: (duracion_ms, texto_para_display)
    """
    if config.TIME_RESO <= 0:
        return config.SLICE_DURATION_MS, f"{config.SLICE_DURATION_MS:.1f} ms"
    
    duration_s = slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
    duration_ms = duration_s * 1000.0
    
    return duration_ms, f"{duration_ms:.1f} ms"


def update_slice_len_dynamic():
    """
    Actualiza config.SLICE_LEN basado en SLICE_DURATION_MS.
    Debe llamarse después de cargar metadatos del archivo.
    """
    slice_len, real_duration_ms = calculate_slice_len_from_duration()
    logger.info(f"✅ Slice configurado: {slice_len} muestras = {real_duration_ms:.1f} ms")
    return slice_len, real_duration_ms


# ==============================================================================
# FUNCIONES HEREDADAS (mantenidas para compatibilidad pero no se usan)
# ==============================================================================

def calculate_optimal_slice_len(
    time_reso: float,
    down_time_rate: int = 1,
    target_duration_seconds: float = 0.032,
    min_slice_len: int = 16,
    max_slice_len: int = 512,
    prefer_power_of_2: bool = True
) -> Tuple[int, float, str]:
    """
    Calcula el SLICE_LEN óptimo basado en la duración temporal deseada.
    
    Parameters
    ----------
    time_reso : float
        Resolución temporal del archivo en segundos por muestra
    down_time_rate : int
        Factor de reducción temporal aplicado
    target_duration_seconds : float
        Duración deseada por slice en segundos
    min_slice_len : int
        Valor mínimo permitido para SLICE_LEN
    max_slice_len : int
        Valor máximo permitido para SLICE_LEN
    prefer_power_of_2 : bool
        Si True, prefiere valores que son potencias de 2
        
    Returns
    -------
    tuple of (optimal_slice_len, actual_duration, explanation)
        optimal_slice_len : int
            Valor óptimo calculado para SLICE_LEN
        actual_duration : float
            Duración real que tendrá cada slice con este valor
        explanation : str
            Explicación del cálculo realizado
    """
    
    if time_reso <= 0:
        logger.warning("TIME_RESO no válido (%.6f), usando SLICE_LEN por defecto", time_reso)
        return 64, 0.064, "TIME_RESO no válido, usando valor por defecto"
    
    # Calcular SLICE_LEN ideal basado en duración objetivo
    effective_time_reso = time_reso * down_time_rate
    ideal_slice_len = target_duration_seconds / effective_time_reso
    
    logger.info("Cálculo SLICE_LEN dinámico:")
    logger.info("  - Duración objetivo: %.3f s", target_duration_seconds)
    logger.info("  - TIME_RESO efectivo: %.6f s", effective_time_reso)
    logger.info("  - SLICE_LEN ideal: %.1f muestras", ideal_slice_len)
    
    # Redondear a entero más cercano
    slice_len_rounded = int(round(ideal_slice_len))
    
    # Si se prefieren potencias de 2, encontrar la más cercana
    if prefer_power_of_2:
        powers_of_2 = [2**i for i in range(20) if 2**i >= min_slice_len and 2**i <= max_slice_len]
        if powers_of_2:
            # Encontrar la potencia de 2 más cercana
            closest_power = min(powers_of_2, key=lambda x: abs(x - ideal_slice_len))
            slice_len_final = closest_power
            explanation = f"Ajustado a potencia de 2 más cercana: {slice_len_final}"
        else:
            slice_len_final = slice_len_rounded
            explanation = "No hay potencias de 2 en el rango, usando redondeo"
    else:
        slice_len_final = slice_len_rounded
        explanation = "Redondeado al entero más cercano"
    
    # Aplicar límites
    if slice_len_final < min_slice_len:
        slice_len_final = min_slice_len
        explanation += f" (limitado al mínimo: {min_slice_len})"
    elif slice_len_final > max_slice_len:
        slice_len_final = max_slice_len
        explanation += f" (limitado al máximo: {max_slice_len})"
    
    # Calcular duración real
    actual_duration = slice_len_final * effective_time_reso
    
    logger.info("  - SLICE_LEN final: %d muestras", slice_len_final)
    logger.info("  - Duración real: %.3f s", actual_duration)
    logger.info("  - Diferencia: %.1f%% vs objetivo", 
                abs(actual_duration - target_duration_seconds) / target_duration_seconds * 100)
    
    return slice_len_final, actual_duration, explanation

def get_dynamic_slice_len(config_module) -> int:
    """
    Obtiene el SLICE_LEN a usar, ya sea calculado dinámicamente o manual.
    
    Parameters
    ----------
    config_module : module
        Módulo de configuración que contiene los parámetros
        
    Returns
    -------
    int
        Valor de SLICE_LEN a usar
    """
    
    # Si no está habilitado el modo automático, usar valor manual
    if not getattr(config_module, 'SLICE_LEN_AUTO', False):
        manual_value = getattr(config_module, 'SLICE_LEN', 64)
        logger.info("Usando SLICE_LEN manual: %d", manual_value)
        return manual_value
    
    # Obtener parámetros necesarios
    time_reso = getattr(config_module, 'TIME_RESO', 0.0)
    down_time_rate = getattr(config_module, 'DOWN_TIME_RATE', 1)
    
    # Convertir de milisegundos a segundos
    target_duration_ms = getattr(config_module, 'SLICE_DURATION_MS', 64.0)
    target_duration = target_duration_ms / 1000.0  # Convertir ms a segundos
    
    min_slice = getattr(config_module, 'SLICE_LEN_MIN', 16)
    max_slice = getattr(config_module, 'SLICE_LEN_MAX', 512)
    
    # Calcular SLICE_LEN dinámicamente
    optimal_slice_len, actual_duration, explanation = calculate_optimal_slice_len(
        time_reso=time_reso,
        down_time_rate=down_time_rate,
        target_duration_seconds=target_duration,
        min_slice_len=min_slice,
        max_slice_len=max_slice
    )
    
    logger.info("SLICE_LEN dinámico calculado: %d (duración: %.1f ms) - %s", 
                optimal_slice_len, actual_duration * 1000, explanation)
    
    return optimal_slice_len

def suggest_slice_duration_for_signal_type(signal_type: str) -> float:
    """
    Sugiere una duración de slice apropiada para diferentes tipos de señales.
    
    Parameters
    ----------
    signal_type : str
        Tipo de señal: 'short', 'medium', 'long', 'dispersed', 'general'
        
    Returns
    -------
    float
        Duración sugerida en segundos
    """
    
    suggestions = {
        'short': 0.016,      # 16ms - Para pulsos muy cortos
        'medium': 0.032,     # 32ms - Para FRBs típicos (por defecto)
        'long': 0.064,       # 64ms - Para señales largas
        'dispersed': 0.128,  # 128ms - Para señales muy dispersas
        'general': 0.032     # 32ms - Configuración general balanceada
    }
    
    return suggestions.get(signal_type.lower(), 0.032)

def print_slice_len_analysis(config_module, file_length: Optional[int] = None):
    """
    Imprime un análisis completo de la configuración SLICE_LEN.
    
    Parameters
    ----------
    config_module : module
        Módulo de configuración
    file_length : int, optional
        Longitud del archivo en muestras para análisis adicional
    """
    
    print("\n🔬 === ANÁLISIS DE CONFIGURACIÓN SLICE_LEN ===\n")
    
    # Obtener configuración actual
    slice_len_auto = getattr(config_module, 'SLICE_LEN_AUTO', False)
    target_duration = getattr(config_module, 'SLICE_DURATION_SECONDS', 0.032)
    manual_slice_len = getattr(config_module, 'SLICE_LEN', 64)
    time_reso = getattr(config_module, 'TIME_RESO', 0.0)
    down_time_rate = getattr(config_module, 'DOWN_TIME_RATE', 1)
    
    print(f"📋 CONFIGURACIÓN ACTUAL:")
    print(f"   • Modo automático: {'✅ Habilitado' if slice_len_auto else '❌ Deshabilitado'}")
    print(f"   • Duración objetivo: {target_duration:.3f} s ({target_duration*1000:.1f} ms)")
    print(f"   • SLICE_LEN manual: {manual_slice_len}")
    print(f"   • TIME_RESO: {time_reso:.6f} s")
    print(f"   • DOWN_TIME_RATE: {down_time_rate}")
    
    if slice_len_auto and time_reso > 0:
        # Calcular valor dinámico
        dynamic_slice_len = get_dynamic_slice_len(config_module)
        actual_duration = dynamic_slice_len * time_reso * down_time_rate
        
        print(f"\n🎯 CÁLCULO DINÁMICO:")
        print(f"   • SLICE_LEN calculado: {dynamic_slice_len}")
        print(f"   • Duración real: {actual_duration:.3f} s ({actual_duration*1000:.1f} ms)")
        print(f"   • Diferencia vs objetivo: {abs(actual_duration - target_duration)/target_duration*100:.1f}%")
        
        current_slice_len = dynamic_slice_len
    else:
        current_slice_len = manual_slice_len
        if time_reso > 0:
            manual_duration = manual_slice_len * time_reso * down_time_rate
            print(f"\n📐 VALOR MANUAL:")
            print(f"   • Duración con valor manual: {manual_duration:.3f} s ({manual_duration*1000:.1f} ms)")
        
    # Análisis adicional si se proporciona longitud de archivo
    if file_length and file_length > 0:
        n_slices = file_length // down_time_rate // current_slice_len
        total_duration = file_length * time_reso if time_reso > 0 else 0
        
        print(f"\n📊 ANÁLISIS DEL ARCHIVO:")
        print(f"   • Longitud archivo: {file_length} muestras")
        print(f"   • Duración total: {total_duration:.3f} s")
        print(f"   • Número de slices: {n_slices}")
        print(f"   • Resolución por pixel: {current_slice_len/512:.3f} muestras/pixel")
        
        if time_reso > 0:
            temporal_resolution = (current_slice_len/512) * time_reso * down_time_rate
            print(f"   • Resolución temporal: {temporal_resolution*1000:.3f} ms/pixel")
    
    # Sugerencias
    print(f"\n💡 SUGERENCIAS POR TIPO DE SEÑAL:")
    signal_types = ['short', 'medium', 'long', 'dispersed']
    type_names = ['Señales cortas', 'FRBs típicos', 'Señales largas', 'Muy dispersas']
    
    for signal_type, name in zip(signal_types, type_names):
        suggested_duration = suggest_slice_duration_for_signal_type(signal_type)
        print(f"   • {name}: {suggested_duration:.3f} s ({suggested_duration*1000:.0f} ms)")

if __name__ == "__main__":
    # Ejemplo de uso
    from DRAFTS.core import config
    print_slice_len_analysis(config)
