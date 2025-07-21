#!/usr/bin/env python3
"""
Test para verificar que la configuración corregida da ~765 slices por chunk.
"""

import sys
from pathlib import Path
import numpy as np

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRAFTS import config


def test_corrected_slice_len():
    """Test de la configuración corregida."""
    
    print("🧪 TEST: CONFIGURACIÓN CORREGIDA")
    print("=" * 60)
    
    # Parámetros del archivo
    chunk_size = 2_000_000
    down_time_rate = 14
    time_reso = 5.46e-05
    slice_duration_ms = 142.0  # Nueva configuración
    
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
    
    # Comparar con valores anteriores
    print("📊 COMPARACIÓN:")
    print(f"   📏 SLICE_LEN anterior: 2616")
    print(f"   📊 Slices anteriores: 54")
    print(f"   📏 SLICE_LEN corregido: {slice_len}")
    print(f"   📊 Slices corregidos: {slices_per_chunk}")
    print()
    
    # Factor de mejora
    improvement_factor = slices_per_chunk / 54
    additional_slices = slices_per_chunk - 54
    
    print(f"   📊 Factor de mejora: {improvement_factor:.2f}x")
    print(f"   📊 Slices adicionales: {additional_slices}")
    print()
    
    # Verificar si es cercano a 765
    target_slices = 765
    difference = abs(slices_per_chunk - target_slices)
    percentage_diff = difference / target_slices * 100
    
    print("📊 VERIFICACIÓN FINAL:")
    print(f"   📊 Slices obtenidos: {slices_per_chunk}")
    print(f"   📊 Slices objetivo: {target_slices}")
    print(f"   📊 Diferencia: {difference}")
    print(f"   📊 Porcentaje: {percentage_diff:.1f}%")
    print()
    
    if percentage_diff < 5:
        print("✅ ¡CONFIGURACIÓN EXITOSA!")
        print(f"   Los {slices_per_chunk} slices están muy cerca del objetivo de {target_slices}")
    elif percentage_diff < 10:
        print("⚠️  Configuración aceptable")
        print(f"   Los {slices_per_chunk} slices están razonablemente cerca del objetivo")
    else:
        print("❌ Configuración necesita ajuste")
        print(f"   Los {slices_per_chunk} slices están muy lejos del objetivo")


def test_multiple_chunks():
    """Test de múltiples chunks con la configuración corregida."""
    
    print("\n🧪 TEST: MÚLTIPLES CHUNKS")
    print("=" * 60)
    
    # Parámetros
    total_samples = 65_917_985
    chunk_size = 2_000_000
    overlap = 1000
    down_time_rate = 14
    time_reso = 5.46e-05
    slice_duration_ms = 142.0
    
    # Calcular SLICE_LEN
    target_duration_s = slice_duration_ms / 1000.0
    time_reso_decimated = time_reso * down_time_rate
    slice_len = round(target_duration_s / time_reso_decimated)
    
    # Calcular chunks
    effective_chunk_size = chunk_size - overlap
    num_chunks = (total_samples + effective_chunk_size - 1) // effective_chunk_size
    
    print("📊 CONFIGURACIÓN:")
    print(f"   📏 Total samples: {total_samples:,}")
    print(f"   📏 SLICE_LEN: {slice_len}")
    print(f"   📊 Número de chunks: {num_chunks}")
    print()
    
    # Simular procesamiento de chunks
    total_slices = 0
    for chunk_idx in range(min(3, num_chunks)):
        start_sample = chunk_idx * effective_chunk_size
        end_sample = min(start_sample + chunk_size, total_samples)
        
        if chunk_idx > 0:
            start_sample -= overlap
            
        actual_chunk_size = end_sample - start_sample
        chunk_decimated_size = actual_chunk_size // down_time_rate
        slices_in_chunk = chunk_decimated_size // slice_len
        
        total_slices += slices_in_chunk
        
        print(f"Chunk {chunk_idx + 1}:")
        print(f"   📏 Muestras: {start_sample:,} a {end_sample:,}")
        print(f"   📏 Chunk decimado: {chunk_decimated_size:,} muestras")
        print(f"   📊 Slices: {slices_in_chunk}")
        print()
    
    # Verificar total
    total_samples_decimated = total_samples // down_time_rate
    expected_total_slices = total_samples_decimated // slice_len
    
    print(f"📊 VERIFICACIÓN TOTAL:")
    print(f"   📊 Slices calculados: {total_slices} (primeros 3 chunks)")
    print(f"   📊 Slices esperados total: {expected_total_slices:,}")
    print(f"   📊 Slices por chunk promedio: {expected_total_slices / num_chunks:.1f}")
    print()


def main():
    """Función principal."""
    test_corrected_slice_len()
    test_multiple_chunks()


if __name__ == "__main__":
    main() 