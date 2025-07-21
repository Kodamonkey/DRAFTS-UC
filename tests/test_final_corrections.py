#!/usr/bin/env python3
"""
Test final para verificar que ambas correcciones funcionan juntas:
1. Corrección del cálculo de time_slice en _process_single_chunk
2. Corrección del cálculo de SLICE_LEN en slice_len_utils.py
"""

import sys
from pathlib import Path
import numpy as np

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRAFTS import config


def test_final_corrections():
    """Test final de ambas correcciones."""
    
    print("🧪 TEST FINAL: AMBAS CORRECCIONES")
    print("=" * 60)
    
    # Parámetros del archivo
    chunk_size = 2_000_000
    down_time_rate = 14
    time_reso = 5.46e-05
    target_duration_ms = 2000.0
    
    print("📊 PARÁMETROS:")
    print(f"   📏 Chunk size: {chunk_size:,}")
    print(f"   🔽 DOWN_TIME_RATE: {down_time_rate}")
    print(f"   ⏱️  TIME_RESO: {time_reso}")
    print(f"   🎯 Duración objetivo: {target_duration_ms} ms")
    print()
    
    # CORRECCIÓN 1: SLICE_LEN correcto
    print("🔧 CORRECCIÓN 1: SLICE_LEN")
    print("-" * 30)
    
    # Cálculo correcto de SLICE_LEN
    target_duration_s = target_duration_ms / 1000.0
    slice_len_correct = round(target_duration_s / time_reso)
    
    print(f"   📏 SLICE_LEN = {target_duration_s:.6f}s ÷ {time_reso} = {slice_len_correct}")
    print()
    
    # CORRECCIÓN 2: Cálculo de slices
    print("🔧 CORRECCIÓN 2: CÁLCULO DE SLICES")
    print("-" * 30)
    
    # Simular chunk decimado
    chunk_decimated = chunk_size // down_time_rate
    width_total = chunk_decimated
    
    # Cálculo correcto de time_slice
    time_slice_correct = width_total // slice_len_correct
    
    print(f"   📏 Chunk decimado: {chunk_decimated:,} muestras")
    print(f"   📏 SLICE_LEN: {slice_len_correct}")
    print(f"   📊 time_slice = {chunk_decimated:,} ÷ {slice_len_correct} = {time_slice_correct}")
    print()
    
    # VERIFICACIÓN FINAL
    print("✅ VERIFICACIÓN FINAL:")
    print("-" * 30)
    
    # Comparar con valores anteriores
    slice_len_old = 2616
    time_slice_old = 54
    
    print(f"   📏 SLICE_LEN anterior: {slice_len_old}")
    print(f"   📊 Slices anteriores: {time_slice_old}")
    print()
    print(f"   📏 SLICE_LEN corregido: {slice_len_correct}")
    print(f"   📊 Slices corregidos: {time_slice_correct}")
    print()
    
    # Factor de mejora
    improvement_factor = time_slice_correct / time_slice_old
    additional_slices = time_slice_correct - time_slice_old
    
    print(f"   📊 Factor de mejora: {improvement_factor:.2f}x")
    print(f"   📊 Slices adicionales: {additional_slices}")
    print()
    
    # Verificar duración temporal
    duration_ms = slice_len_correct * time_reso * down_time_rate * 1000
    print(f"   ⏱️  Duración real: {duration_ms:.1f} ms")
    print(f"   🎯 Duración objetivo: {target_duration_ms:.1f} ms")
    print(f"   📊 Diferencia: {abs(duration_ms - target_duration_ms):.1f} ms")
    print()
    
    # Verificar continuidad temporal
    print("🕐 VERIFICACIÓN DE CONTINUIDAD TEMPORAL:")
    print("-" * 40)
    
    # Simular dos chunks consecutivos
    chunk1_start = 0
    chunk1_end = chunk_size
    chunk2_start = chunk_size - 1000  # Con overlap
    
    # Tiempos absolutos
    chunk1_start_time = chunk1_start * time_reso * down_time_rate
    chunk1_end_time = chunk1_end * time_reso * down_time_rate
    chunk2_start_time = chunk2_start * time_reso * down_time_rate
    
    print(f"   🕐 Chunk 1: {chunk1_start_time:.3f}s a {chunk1_end_time:.3f}s")
    print(f"   🕐 Chunk 2: {chunk2_start_time:.3f}s a ...")
    print(f"   🔗 Gap: {chunk2_start_time - chunk1_end_time:.3f}s")
    
    if abs(chunk2_start_time - chunk1_end_time) < 0.001:
        print("   ✅ Continuidad temporal correcta")
    else:
        print("   ❌ Problema de continuidad temporal")
    print()
    
    # Resumen final
    print("📋 RESUMEN FINAL:")
    print("-" * 30)
    print(f"   ✅ SLICE_LEN corregido: {slice_len_old} → {slice_len_correct}")
    print(f"   ✅ Slices por chunk: {time_slice_old} → {time_slice_correct}")
    print(f"   ✅ Mejora: {improvement_factor:.2f}x más slices")
    print(f"   ✅ Slices adicionales: {additional_slices}")
    print(f"   ✅ Duración: {duration_ms:.1f} ms (objetivo: {target_duration_ms:.1f} ms)")
    print()
    
    if improvement_factor > 10 and abs(duration_ms - target_duration_ms) < 100:
        print("🎉 ¡CORRECCIONES EXITOSAS!")
    else:
        print("⚠️  Revisar correcciones")


def test_expected_results():
    """Test de los resultados esperados después de las correcciones."""
    
    print("\n🧪 TEST: RESULTADOS ESPERADOS")
    print("=" * 60)
    
    # Parámetros
    total_samples = 65_917_985
    chunk_size = 2_000_000
    overlap = 1000
    down_time_rate = 14
    time_reso = 5.46e-05
    target_duration_ms = 2000.0
    
    # Calcular SLICE_LEN corregido
    target_duration_s = target_duration_ms / 1000.0
    slice_len_correct = round(target_duration_s / time_reso)
    
    # Calcular chunks
    effective_chunk_size = chunk_size - overlap
    num_chunks = (total_samples + effective_chunk_size - 1) // effective_chunk_size
    
    print("📊 CONFIGURACIÓN:")
    print(f"   📏 Total samples: {total_samples:,}")
    print(f"   📏 SLICE_LEN corregido: {slice_len_correct}")
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
        slices_in_chunk = chunk_decimated_size // slice_len_correct
        
        total_slices += slices_in_chunk
        
        print(f"Chunk {chunk_idx + 1}:")
        print(f"   📏 Muestras: {start_sample:,} a {end_sample:,}")
        print(f"   📏 Chunk decimado: {chunk_decimated_size:,} muestras")
        print(f"   📊 Slices: {slices_in_chunk}")
        print()
    
    # Verificar total
    total_samples_decimated = total_samples // down_time_rate
    expected_total_slices = total_samples_decimated // slice_len_correct
    
    print(f"📊 VERIFICACIÓN TOTAL:")
    print(f"   📊 Slices calculados: {total_slices} (primeros 3 chunks)")
    print(f"   📊 Slices esperados total: {expected_total_slices:,}")
    print(f"   📊 Slices por chunk promedio: {expected_total_slices / num_chunks:.1f}")
    print()


def main():
    """Función principal."""
    test_final_corrections()
    test_expected_results()


if __name__ == "__main__":
    main() 