#!/usr/bin/env python3
"""
Test para debuggear el problema de timing entre chunks.
Simula el cálculo de tiempo absoluto para identificar el problema.
"""

import sys
from pathlib import Path
import numpy as np

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRAFTS import config


def simulate_chunk_timing():
    """Simular el cálculo de timing entre chunks para identificar el problema."""
    
    print("🔍 DEBUG: Problema de Timing Entre Chunks")
    print("=" * 60)
    
    # Parámetros típicos
    total_samples = 65_917_985  # Archivo de 1 hora
    chunk_size = 2_000_000
    overlap = 1000
    
    # Configuración temporal
    config.TIME_RESO = 0.000001  # 1 microsegundo
    config.DOWN_TIME_RATE = 14
    
    print(f"📊 Parámetros de simulación:")
    print(f"   📏 Total samples: {total_samples:,}")
    print(f"   📏 Chunk size: {chunk_size:,}")
    print(f"   📏 Overlap: {overlap}")
    print(f"   ⏱️  TIME_RESO: {config.TIME_RESO}")
    print(f"   🔽 DOWN_TIME_RATE: {config.DOWN_TIME_RATE}")
    print()
    
    # Calcular número de chunks
    effective_chunk_size = chunk_size - overlap
    num_chunks = (total_samples + effective_chunk_size - 1) // effective_chunk_size
    
    print(f"📊 Cálculo de chunks:")
    print(f"   📏 Effective chunk size: {effective_chunk_size:,}")
    print(f"   📊 Número de chunks: {num_chunks}")
    print()
    
    # Simular el cálculo actual (problemático)
    print("❌ CÁLCULO ACTUAL (PROBLEMÁTICO):")
    print("-" * 40)
    
    for chunk_idx in range(min(3, num_chunks)):  # Solo primeros 3 chunks
        # Cálculo actual del código
        start_sample = chunk_idx * effective_chunk_size
        end_sample = min(start_sample + chunk_size, total_samples)
        
        if chunk_idx > 0:
            start_sample -= overlap  # Agregar overlap con chunk anterior
            
        actual_chunk_size = end_sample - start_sample
        
        # Primera vez que se calcula el tiempo (líneas 1099-1101)
        chunk_start_time_sec_1 = start_sample * config.TIME_RESO * config.DOWN_TIME_RATE
        chunk_end_time_sec_1 = end_sample * config.TIME_RESO * config.DOWN_TIME_RATE
        
        # Segunda vez que se calcula el tiempo (líneas 1115-1117) - DUPLICACIÓN
        chunk_start_time_sec_2 = start_sample * config.TIME_RESO * config.DOWN_TIME_RATE
        chunk_end_time_sec_2 = end_sample * config.TIME_RESO * config.DOWN_TIME_RATE
        
        print(f"Chunk {chunk_idx + 1}:")
        print(f"   📏 Muestras: {start_sample:,} a {end_sample:,} ({actual_chunk_size:,})")
        print(f"   🕐 Tiempo 1: {chunk_start_time_sec_1:.3f}s a {chunk_end_time_sec_1:.3f}s")
        print(f"   🕐 Tiempo 2: {chunk_start_time_sec_2:.3f}s a {chunk_end_time_sec_2:.3f}s")
        print(f"   ⏱️  Duración: {chunk_end_time_sec_1 - chunk_start_time_sec_1:.3f}s")
        print()
    
    # Simular el cálculo corregido
    print("✅ CÁLCULO CORREGIDO:")
    print("-" * 40)
    
    for chunk_idx in range(min(3, num_chunks)):
        # Cálculo corregido
        start_sample = chunk_idx * effective_chunk_size
        end_sample = min(start_sample + chunk_size, total_samples)
        
        if chunk_idx > 0:
            start_sample -= overlap
            
        actual_chunk_size = end_sample - start_sample
        
        # Solo una vez el cálculo de tiempo
        chunk_start_time_sec = start_sample * config.TIME_RESO * config.DOWN_TIME_RATE
        chunk_end_time_sec = end_sample * config.TIME_RESO * config.DOWN_TIME_RATE
        
        print(f"Chunk {chunk_idx + 1}:")
        print(f"   📏 Muestras: {start_sample:,} a {end_sample:,} ({actual_chunk_size:,})")
        print(f"   🕐 Tiempo: {chunk_start_time_sec:.3f}s a {chunk_end_time_sec:.3f}s")
        print(f"   ⏱️  Duración: {chunk_end_time_sec - chunk_start_time_sec:.3f}s")
        
        if chunk_idx > 0:
            # Calcular gap con chunk anterior
            prev_end = (chunk_idx - 1) * effective_chunk_size + chunk_size
            if chunk_idx > 0:
                prev_end -= overlap
            prev_end = min(prev_end, total_samples)
            prev_end_time = prev_end * config.TIME_RESO * config.DOWN_TIME_RATE
            gap = chunk_start_time_sec - prev_end_time
            print(f"   🔗 Gap con chunk anterior: {gap:.3f}s")
        print()
    
    # Verificar continuidad
    print("🔍 VERIFICACIÓN DE CONTINUIDAD:")
    print("-" * 40)
    
    for chunk_idx in range(min(3, num_chunks)):
        start_sample = chunk_idx * effective_chunk_size
        end_sample = min(start_sample + chunk_size, total_samples)
        
        if chunk_idx > 0:
            start_sample -= overlap
            
        chunk_start_time_sec = start_sample * config.TIME_RESO * config.DOWN_TIME_RATE
        chunk_end_time_sec = end_sample * config.TIME_RESO * config.DOWN_TIME_RATE
        
        if chunk_idx > 0:
            # Calcular dónde debería terminar el chunk anterior
            prev_start = (chunk_idx - 1) * effective_chunk_size
            prev_end = min(prev_start + chunk_size, total_samples)
            if chunk_idx - 1 > 0:
                prev_start -= overlap
            prev_end_time = prev_end * config.TIME_RESO * config.DOWN_TIME_RATE
            
            # Verificar continuidad
            expected_start = prev_end_time
            actual_start = chunk_start_time_sec
            gap = actual_start - expected_start
            
            print(f"Chunk {chunk_idx}:")
            print(f"   🕐 Chunk {chunk_idx-1} debería terminar en: {prev_end_time:.3f}s")
            print(f"   🕐 Chunk {chunk_idx} empieza en: {actual_start:.3f}s")
            print(f"   🔗 Gap: {gap:.3f}s")
            
            if abs(gap) > 0.001:  # Más de 1ms de gap
                print(f"   ❌ PROBLEMA: Gap de {gap:.3f}s detectado!")
            else:
                print(f"   ✅ OK: Continuidad correcta")
        print()


def main():
    """Función principal."""
    simulate_chunk_timing()


if __name__ == "__main__":
    main() 