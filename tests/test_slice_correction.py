#!/usr/bin/env python3
"""
Test para verificar que la corrección de slices funciona correctamente.
"""

import sys
from pathlib import Path
import numpy as np

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRAFTS import config


def test_slice_calculation_correction():
    """Test para verificar la corrección del cálculo de slices."""
    
    print("🧪 TEST: CORRECCIÓN DE CÁLCULO DE SLICES")
    print("=" * 60)
    
    # Parámetros del archivo según el output
    chunk_size = 2_000_000
    down_time_rate = 14
    slice_len = 2616
    
    print("📊 PARÁMETROS DE TEST:")
    print(f"   📏 Chunk size: {chunk_size:,}")
    print(f"   🔽 DOWN_TIME_RATE: {down_time_rate}")
    print(f"   📏 SLICE_LEN: {slice_len}")
    print()
    
    # Simular chunk original
    chunk_original = np.zeros((chunk_size, 512))
    print("📊 SIMULACIÓN DE CHUNK:")
    print(f"   📏 Chunk original shape: {chunk_original.shape}")
    
    # Simular decimación
    chunk_decimated = chunk_original[::down_time_rate]
    print(f"   🔽 Chunk decimado shape: {chunk_decimated.shape}")
    
    # Cálculo INCORRECTO (anterior)
    width_total_incorrect = chunk_decimated.shape[0]
    time_slice_incorrect = (width_total_incorrect + slice_len - 1) // slice_len
    
    # Cálculo CORRECTO (nuevo)
    width_total_correct = chunk_decimated.shape[0]
    time_slice_correct = width_total_correct // slice_len
    
    print("📊 COMPARACIÓN DE CÁLCULOS:")
    print(f"   ❌ Cálculo incorrecto: ({width_total_incorrect} + {slice_len} - 1) // {slice_len} = {time_slice_incorrect}")
    print(f"   ✅ Cálculo correcto: {width_total_correct} // {slice_len} = {time_slice_correct}")
    print(f"   📊 Diferencia: {time_slice_incorrect - time_slice_correct} slices")
    print()
    
    # Verificar que el cálculo correcto da el resultado esperado
    expected_slices = chunk_size // slice_len
    print("📊 VERIFICACIÓN:")
    print(f"   📊 Slices esperados (sin decimación): {expected_slices}")
    print(f"   📊 Slices reales (con decimación): {time_slice_correct}")
    print(f"   📊 Factor de reducción: {expected_slices / time_slice_correct:.2f}x")
    print(f"   📊 ¿Coincide con DOWN_TIME_RATE? {abs(expected_slices / time_slice_correct - down_time_rate) < 0.1}")
    print()
    
    # Verificar continuidad temporal
    print("🕐 VERIFICACIÓN DE CONTINUIDAD TEMPORAL:")
    print("-" * 40)
    
    # Simular dos chunks consecutivos
    chunk1_start = 0
    chunk1_end = chunk_size
    chunk2_start = chunk_size - 1000  # Con overlap
    
    # Tiempos absolutos
    time_reso = 5.46e-05
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
    
    # Verificar que los slices cubren todo el chunk
    print("📊 VERIFICACIÓN DE COBERTURA:")
    print("-" * 40)
    
    total_samples_covered = time_slice_correct * slice_len
    chunk_samples = chunk_decimated.shape[0]
    
    print(f"   📏 Muestras en chunk decimado: {chunk_samples:,}")
    print(f"   📏 Muestras cubiertas por slices: {total_samples_covered:,}")
    print(f"   📏 Muestras no cubiertas: {chunk_samples - total_samples_covered:,}")
    print(f"   📊 Cobertura: {total_samples_covered / chunk_samples * 100:.1f}%")
    
    if total_samples_covered >= chunk_samples:
        print("   ✅ Cobertura completa")
    else:
        print("   ❌ Cobertura incompleta")
    print()


def test_multiple_chunks():
    """Test para verificar múltiples chunks."""
    
    print("\n🧪 TEST: MÚLTIPLES CHUNKS")
    print("=" * 60)
    
    # Parámetros
    total_samples = 65_917_985
    chunk_size = 2_000_000
    overlap = 1000
    down_time_rate = 14
    slice_len = 2616
    
    # Calcular chunks
    effective_chunk_size = chunk_size - overlap
    num_chunks = (total_samples + effective_chunk_size - 1) // effective_chunk_size
    
    print(f"📊 CONFIGURACIÓN:")
    print(f"   📏 Total samples: {total_samples:,}")
    print(f"   📏 Effective chunk size: {effective_chunk_size:,}")
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
    print(f"   📊 Factor de reducción: {expected_total_slices / total_slices:.2f}x")
    print()


def main():
    """Función principal."""
    test_slice_calculation_correction()
    test_multiple_chunks()


if __name__ == "__main__":
    main() 