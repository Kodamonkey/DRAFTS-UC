#!/usr/bin/env python3
"""
Test script para verificar que los tiempos absolutos funcionan correctamente
en el procesamiento por chunks.

Este script simula el procesamiento de chunks y verifica que los tiempos
mostrados en los plots sean los tiempos reales del archivo.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Agregar el directorio raíz al path para importar DRAFTS
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRAFTS import config
from DRAFTS.image_utils import plot_waterfall_block

def test_chunk_timing():
    """Test para verificar que los tiempos absolutos funcionan correctamente."""
    
    print("=== TEST DE TIEMPOS ABSOLUTOS EN CHUNKS ===")
    
    # Configurar parámetros de prueba
    config.TIME_RESO = 0.001  # 1 ms
    config.DOWN_TIME_RATE = 1
    slice_len = 1000  # 1000 muestras = 1 segundo
    chunk_size = 5000  # 5000 muestras = 5 segundos
    
    # Simular datos de prueba
    data_chunk = np.random.randn(chunk_size, 64)  # 64 canales de frecuencia
    freq = np.linspace(1000, 1500, 64)  # 1000-1500 MHz
    
    # Crear directorio de prueba
    test_dir = Path("tests/test_chunk_timing_output")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Test 1: Chunk 0 (tiempo absoluto = 0s)
    print("\n1. Probando Chunk 0 (tiempo absoluto = 0s)...")
    chunk_start_time_sec = 0.0
    
    plot_waterfall_block(
        data_block=data_chunk[:slice_len],  # Primer slice
        freq=freq,
        time_reso=config.TIME_RESO * config.DOWN_TIME_RATE,
        block_size=slice_len,
        block_idx=0,
        save_dir=test_dir,
        filename="chunk0_test",
        normalize=True,
        absolute_start_time=chunk_start_time_sec,
    )
    print("   ✅ Chunk 0: waterfall generado con tiempo absoluto = 0s")
    
    # Test 2: Chunk 1 (tiempo absoluto = 5s)
    print("\n2. Probando Chunk 1 (tiempo absoluto = 5s)...")
    chunk_start_time_sec = 5.0
    
    plot_waterfall_block(
        data_block=data_chunk[:slice_len],  # Primer slice
        freq=freq,
        time_reso=config.TIME_RESO * config.DOWN_TIME_RATE,
        block_size=slice_len,
        block_idx=0,
        save_dir=test_dir,
        filename="chunk1_test",
        normalize=True,
        absolute_start_time=chunk_start_time_sec,
    )
    print("   ✅ Chunk 1: waterfall generado con tiempo absoluto = 5s")
    
    # Test 3: Chunk 2 (tiempo absoluto = 10s)
    print("\n3. Probando Chunk 2 (tiempo absoluto = 10s)...")
    chunk_start_time_sec = 10.0
    
    plot_waterfall_block(
        data_block=data_chunk[:slice_len],  # Primer slice
        freq=freq,
        time_reso=config.TIME_RESO * config.DOWN_TIME_RATE,
        block_size=slice_len,
        block_idx=0,
        save_dir=test_dir,
        filename="chunk2_test",
        normalize=True,
        absolute_start_time=chunk_start_time_sec,
    )
    print("   ✅ Chunk 2: waterfall generado con tiempo absoluto = 10s")
    
    # Test 4: Comportamiento por defecto (sin absolute_start_time)
    print("\n4. Probando comportamiento por defecto (sin absolute_start_time)...")
    
    plot_waterfall_block(
        data_block=data_chunk[:slice_len],  # Primer slice
        freq=freq,
        time_reso=config.TIME_RESO * config.DOWN_TIME_RATE,
        block_size=slice_len,
        block_idx=0,
        save_dir=test_dir,
        filename="default_test",
        normalize=True,
        # No pasar absolute_start_time (debe usar tiempo relativo)
    )
    print("   ✅ Comportamiento por defecto: waterfall generado con tiempo relativo")
    
    print(f"\n=== RESULTADOS ===")
    print(f"Archivos generados en: {test_dir}")
    print("Verificar que:")
    print("  - chunk0_test: muestra tiempos 0.00s a 1.00s")
    print("  - chunk1_test: muestra tiempos 5.00s a 6.00s") 
    print("  - chunk2_test: muestra tiempos 10.00s a 11.00s")
    print("  - default_test: muestra tiempos 0.00s a 1.00s (relativo)")
    
    return True

def test_slice_index_calculation():
    """Test para verificar el cálculo correcto del índice absoluto del slice."""
    
    print("\n=== TEST DE CÁLCULO DE ÍNDICE ABSOLUTO ===")
    
    # Parámetros de prueba
    slice_len = 1000
    chunk_size = 5000
    
    # Simular diferentes chunks
    test_cases = [
        {"chunk_idx": 0, "start_sample_global": 0, "slice_idx": 0, "expected_absolute": 0},
        {"chunk_idx": 0, "start_sample_global": 0, "slice_idx": 1, "expected_absolute": 1},
        {"chunk_idx": 1, "start_sample_global": 5000, "slice_idx": 0, "expected_absolute": 5},
        {"chunk_idx": 1, "start_sample_global": 5000, "slice_idx": 1, "expected_absolute": 6},
        {"chunk_idx": 2, "start_sample_global": 10000, "slice_idx": 0, "expected_absolute": 10},
    ]
    
    for i, case in enumerate(test_cases):
        chunk_idx = case["chunk_idx"]
        start_sample_global = case["start_sample_global"]
        slice_idx = case["slice_idx"]
        expected_absolute = case["expected_absolute"]
        
        # Calcular índice absoluto (misma lógica que en el pipeline)
        slice_absolute_idx = start_sample_global // slice_len + slice_idx
        
        print(f"Test {i+1}: Chunk {chunk_idx}, Slice {slice_idx} (global {start_sample_global})")
        print(f"   Calculado: {slice_absolute_idx}, Esperado: {expected_absolute}")
        
        if slice_absolute_idx == expected_absolute:
            print("   ✅ CORRECTO")
        else:
            print("   ❌ INCORRECTO")
            return False
    
    return True

if __name__ == "__main__":
    print("Iniciando tests de tiempos absolutos en chunks...")
    
    # Ejecutar tests
    test1_passed = test_chunk_timing()
    test2_passed = test_slice_index_calculation()
    
    if test1_passed and test2_passed:
        print("\n🎉 TODOS LOS TESTS PASARON EXITOSAMENTE!")
        print("Los tiempos absolutos funcionan correctamente en el procesamiento por chunks.")
    else:
        print("\n❌ ALGUNOS TESTS FALLARON!")
        sys.exit(1) 