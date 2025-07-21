#!/usr/bin/env python3
"""
Calcular cuántos slices hay con SLICE_DURATION_MS = 1000.0
"""

import sys
from pathlib import Path
import numpy as np

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRAFTS import config


def calcular_slices_1000ms():
    """Calcular slices con SLICE_DURATION_MS = 1000.0"""
    
    print("🔍 CÁLCULO DE SLICES CON SLICE_DURATION_MS = 1000.0")
    print("=" * 60)
    
    # Parámetros del archivo
    chunk_size = 2_000_000
    down_time_rate = 14
    time_reso = 5.46e-05
    slice_duration_ms = 1000.0
    
    print("📊 PARÁMETROS:")
    print(f"   📏 Chunk size: {chunk_size:,}")
    print(f"   🔽 DOWN_TIME_RATE: {down_time_rate}")
    print(f"   ⏱️  TIME_RESO: {time_reso}")
    print(f"   🎯 SLICE_DURATION_MS: {slice_duration_ms} ms")
    print()
    
    # Cálculo de SLICE_LEN
    target_duration_s = slice_duration_ms / 1000.0
    time_reso_decimated = time_reso * down_time_rate
    slice_len = round(target_duration_s / time_reso_decimated)
    
    print("📊 CÁLCULO DE SLICE_LEN:")
    print(f"   ⏱️  TIME_RESO decimado: {time_reso_decimated}")
    print(f"   📏 SLICE_LEN = {target_duration_s:.6f}s ÷ {time_reso_decimated} = {slice_len}")
    print()
    
    # Cálculo de slices por chunk
    chunk_decimated = chunk_size // down_time_rate
    slices_per_chunk = chunk_decimated // slice_len
    
    print("📊 SLICES POR CHUNK:")
    print(f"   📏 Chunk decimado: {chunk_decimated:,} muestras")
    print(f"   📏 SLICE_LEN: {slice_len}")
    print(f"   📊 Slices = {chunk_decimated:,} ÷ {slice_len} = {slices_per_chunk}")
    print()
    
    # Verificar duración real
    duration_ms = slice_len * time_reso_decimated * 1000
    print("📊 VERIFICACIÓN:")
    print(f"   ⏱️  Duración real: {duration_ms:.1f} ms")
    print(f"   🎯 Duración objetivo: {slice_duration_ms:.1f} ms")
    print(f"   📊 Diferencia: {abs(duration_ms - slice_duration_ms):.1f} ms")
    print()
    
    return slice_len, slices_per_chunk


def main():
    """Función principal."""
    slice_len, slices_per_chunk = calcular_slices_1000ms()
    
    print("📋 RESPUESTA:")
    print(f"Con SLICE_DURATION_MS = 1000.0 ms:")
    print(f"   📏 SLICE_LEN = {slice_len} muestras")
    print(f"   📊 Slices por chunk = {slices_per_chunk}")
    print(f"   ⏱️  Duración real = {slice_len * 5.46e-05 * 14 * 1000:.1f} ms")


if __name__ == "__main__":
    main() 