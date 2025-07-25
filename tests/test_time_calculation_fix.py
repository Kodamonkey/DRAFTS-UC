#!/usr/bin/env python3
"""
Script de prueba para verificar la corrección del cálculo de tiempo absoluto en los plots.

Este script simula el problema que se estaba produciendo donde los tiempos absolutos
se estaban calculando incorrectamente, causando saltos de 2 en 2 en lugar de continuidad.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Agregar el directorio raíz al path para importar DRAFTS
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from DRAFTS.visualization.image_utils import plot_waterfall_block

def test_time_calculation_fix():
    """Prueba la corrección del cálculo de tiempo absoluto."""
    
    print("🧪 PRUEBA: Corrección del cálculo de tiempo absoluto")
    print("=" * 60)
    
    # Simular datos de prueba
    n_freq = 100
    n_time = 64  # SLICE_LEN típico
    freq = np.linspace(1200, 1500, n_freq)
    time_reso = 0.0001  # 0.1 ms
    block_size = n_time
    
    # Crear datos simulados
    data_block = np.random.normal(1.0, 0.2, (n_time, n_freq))
    
    # Crear directorio de prueba
    test_dir = Path("test_time_fix")
    test_dir.mkdir(exist_ok=True)
    
    print(f"📊 Parámetros de prueba:")
    print(f"   - Frecuencias: {freq[0]:.1f} - {freq[-1]:.1f} MHz")
    print(f"   - Tiempo por slice: {n_time * time_reso:.4f} s")
    print(f"   - Resolución temporal: {time_reso:.6f} s")
    print()
    
    # Probar con diferentes tiempos absolutos
    test_cases = [
        (0.0, "slice_0"),
        (0.0064, "slice_1"),  # 64 * 0.0001
        (0.0128, "slice_2"),  # 128 * 0.0001
        (0.0192, "slice_3"),  # 192 * 0.0001
    ]
    
    print("🕐 Generando plots con tiempos absolutos corregidos:")
    
    for absolute_start_time, slice_name in test_cases:
        print(f"   - {slice_name}: tiempo absoluto = {absolute_start_time:.4f}s")
        
        # Llamar a la función corregida
        plot_waterfall_block(
            data_block=data_block,
            freq=freq,
            time_reso=time_reso,
            block_size=block_size,
            block_idx=0,  # Siempre 0 porque absolute_start_time ya incluye el offset
            save_dir=test_dir,
            filename=f"test_{slice_name}",
            normalize=True,
            absolute_start_time=absolute_start_time
        )
    
    print()
    print("✅ Prueba completada. Verificar los archivos generados en:")
    print(f"   {test_dir.absolute()}")
    print()
    print("📋 Verificación manual:")
    print("   - Los plots deberían mostrar tiempos continuos:")
    print("   - slice_0: 0.0000s - 0.0064s")
    print("   - slice_1: 0.0064s - 0.0128s")
    print("   - slice_2: 0.0128s - 0.0192s")
    print("   - slice_3: 0.0192s - 0.0256s")
    print()
    print("🔍 Si los tiempos muestran saltos de 2 en 2, la corrección no funcionó.")
    print("   Si muestran continuidad, la corrección fue exitosa.")

def simulate_old_bug():
    """Simula el comportamiento del bug anterior para comparación."""
    
    print("🐛 SIMULACIÓN: Comportamiento del bug anterior")
    print("=" * 50)
    
    # Simular el cálculo incorrecto que se hacía antes
    time_reso = 0.0001
    block_size = 64
    
    print("❌ Cálculo INCORRECTO (antes de la corrección):")
    print("   time_start = absolute_start_time + block_idx * block_size * time_reso")
    print()
    
    for j in range(4):
        absolute_start_time = j * block_size * time_reso  # Tiempo de inicio del slice
        block_idx = 0  # Índice del slice dentro del chunk
        
        # Cálculo incorrecto (el bug)
        time_start_incorrect = absolute_start_time + block_idx * block_size * time_reso
        
        print(f"   Slice {j}:")
        print(f"     absolute_start_time = {absolute_start_time:.4f}s")
        print(f"     block_idx = {block_idx}")
        print(f"     time_start_incorrect = {absolute_start_time:.4f} + {block_idx} * {block_size} * {time_reso:.6f}")
        print(f"     time_start_incorrect = {time_start_incorrect:.4f}s")
        print()
    
    print("✅ Cálculo CORRECTO (después de la corrección):")
    print("   time_start = absolute_start_time")
    print()
    
    for j in range(4):
        absolute_start_time = j * block_size * time_reso
        
        # Cálculo correcto
        time_start_correct = absolute_start_time
        
        print(f"   Slice {j}:")
        print(f"     absolute_start_time = {absolute_start_time:.4f}s")
        print(f"     time_start_correct = {time_start_correct:.4f}s")
        print()

if __name__ == "__main__":
    print("🔧 PRUEBA DE CORRECCIÓN DE TIEMPO ABSOLUTO")
    print("=" * 60)
    print()
    
    simulate_old_bug()
    print()
    test_time_calculation_fix() 