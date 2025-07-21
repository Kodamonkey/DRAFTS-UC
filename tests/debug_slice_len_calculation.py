#!/usr/bin/env python3
"""
Debug del cálculo de SLICE_LEN
"""

import sys
from pathlib import Path
import numpy as np

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRAFTS import config
from DRAFTS.slice_len_utils import calculate_slice_len_from_duration, update_slice_len_dynamic


def debug_slice_len_calculation():
    """Debug del cálculo de SLICE_LEN."""
    
    print("🔍 DEBUG DEL CÁLCULO DE SLICE_LEN")
    print("=" * 60)
    
    # Configurar valores como en tu archivo
    config.SLICE_DURATION_MS = 1000.0
    config.TIME_RESO = 5.46e-05
    config.DOWN_TIME_RATE = 14
    config.SLICE_LEN_MIN = 32
    config.SLICE_LEN_MAX = 2048
    
    print("📊 CONFIGURACIÓN:")
    print(f"   🎯 SLICE_DURATION_MS: {config.SLICE_DURATION_MS} ms")
    print(f"   ⏱️  TIME_RESO: {config.TIME_RESO}")
    print(f"   🔽 DOWN_TIME_RATE: {config.DOWN_TIME_RATE}")
    print(f"   📏 SLICE_LEN_MIN: {config.SLICE_LEN_MIN}")
    print(f"   📏 SLICE_LEN_MAX: {config.SLICE_LEN_MAX}")
    print()
    
    # Cálculo manual
    target_duration_s = config.SLICE_DURATION_MS / 1000.0
    time_reso_decimated = config.TIME_RESO * config.DOWN_TIME_RATE
    calculated_slice_len = round(target_duration_s / time_reso_decimated)
    
    print("📊 CÁLCULO MANUAL:")
    print(f"   ⏱️  target_duration_s: {target_duration_s:.6f} s")
    print(f"   ⏱️  time_reso_decimated: {time_reso_decimated:.6f} s")
    print(f"   📏 calculated_slice_len: {target_duration_s:.6f} ÷ {time_reso_decimated:.6f} = {calculated_slice_len}")
    print()
    
    # Verificar límites
    slice_len_within_limits = max(config.SLICE_LEN_MIN, min(config.SLICE_LEN_MAX, calculated_slice_len))
    print("📊 APLICACIÓN DE LÍMITES:")
    print(f"   📏 Valor calculado: {calculated_slice_len}")
    print(f"   📏 Después de límites: {slice_len_within_limits}")
    print(f"   📊 ¿Está dentro de límites? {config.SLICE_LEN_MIN <= calculated_slice_len <= config.SLICE_LEN_MAX}")
    print()
    
    # Llamar a la función real
    print("📊 LLAMADA A LA FUNCIÓN REAL:")
    original_slice_len = config.SLICE_LEN
    print(f"   📏 SLICE_LEN antes: {original_slice_len}")
    
    slice_len, real_duration_ms = calculate_slice_len_from_duration()
    
    print(f"   📏 SLICE_LEN después: {config.SLICE_LEN}")
    print(f"   📏 Valor retornado: {slice_len}")
    print(f"   ⏱️  Duración real: {real_duration_ms:.1f} ms")
    print()
    
    # Verificar si se actualizó correctamente
    if config.SLICE_LEN == slice_len:
        print("✅ config.SLICE_LEN se actualizó correctamente")
    else:
        print("❌ config.SLICE_LEN NO se actualizó correctamente")
    
    if slice_len == calculated_slice_len:
        print("✅ El valor calculado coincide con el esperado")
    else:
        print(f"❌ El valor calculado ({slice_len}) NO coincide con el esperado ({calculated_slice_len})")
    
    # Calcular slices por chunk
    chunk_size = 2_000_000
    chunk_decimated = chunk_size // config.DOWN_TIME_RATE
    slices_per_chunk = chunk_decimated // slice_len
    
    print(f"\n📊 SLICES POR CHUNK:")
    print(f"   📏 Chunk decimado: {chunk_decimated:,} muestras")
    print(f"   📏 SLICE_LEN: {slice_len} muestras")
    print(f"   📊 Slices por chunk: {chunk_decimated:,} ÷ {slice_len} = {slices_per_chunk}")
    
    return slice_len, slices_per_chunk


def main():
    """Función principal."""
    slice_len, slices_per_chunk = debug_slice_len_calculation()
    
    print(f"\n📋 RESUMEN:")
    print(f"   📏 SLICE_LEN final: {slice_len}")
    print(f"   📊 Slices por chunk: {slices_per_chunk}")
    
    if slices_per_chunk == 109:
        print("✅ ¡PERFECTO! Deberías obtener 109 slices por chunk")
    else:
        print(f"❌ PROBLEMA: Estás obteniendo {slices_per_chunk} slices en lugar de 109")


if __name__ == "__main__":
    main() 