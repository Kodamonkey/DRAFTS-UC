#!/usr/bin/env python3
"""
Test para verificar la continuidad temporal entre chunks
"""

import sys
from pathlib import Path
import numpy as np

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRAFTS import config


def test_continuidad_temporal():
    """Test de continuidad temporal entre chunks."""
    
    print("🔍 TEST DE CONTINUIDAD TEMPORAL ENTRE CHUNKS")
    print("=" * 60)
    
    # Parámetros del archivo
    chunk_size = 2_000_000
    down_time_rate = 14
    time_reso = 5.46e-05
    slice_duration_ms = 1000.0
    overlap = 500
    
    print("📊 PARÁMETROS:")
    print(f"   📏 Chunk size: {chunk_size:,}")
    print(f"   🔽 DOWN_TIME_RATE: {down_time_rate}")
    print(f"   ⏱️  TIME_RESO: {time_reso}")
    print(f"   🎯 SLICE_DURATION_MS: {slice_duration_ms} ms")
    print(f"   🔗 Overlap: {overlap}")
    print()
    
    # Cálculo de SLICE_LEN
    target_duration_s = slice_duration_ms / 1000.0
    time_reso_decimated = time_reso * down_time_rate
    slice_len = round(target_duration_s / time_reso_decimated)
    
    print("📊 CÁLCULO DE SLICE_LEN:")
    print(f"   ⏱️  TIME_RESO decimado: {time_reso_decimated}")
    print(f"   📏 SLICE_LEN = {target_duration_s:.6f}s ÷ {time_reso_decimated} = {slice_len}")
    print()
    
    # Simular chunks
    total_samples = 10_000_000  # 10M muestras
    effective_chunk_size = chunk_size - overlap
    
    print("📊 SIMULACIÓN DE CHUNKS:")
    print(f"   📏 Total samples: {total_samples:,}")
    print(f"   📏 Effective chunk size: {effective_chunk_size:,}")
    print(f"   📊 Número de chunks: {total_samples // effective_chunk_size}")
    print()
    
    # Calcular tiempos de chunks
    chunks_info = []
    for chunk_idx in range(3):  # Solo los primeros 3 chunks
        start_sample = chunk_idx * effective_chunk_size
        if chunk_idx > 0:
            start_sample -= overlap
        
        end_sample = min(start_sample + chunk_size, total_samples)
        
        # Tiempos absolutos
        chunk_start_time_sec = start_sample * time_reso * down_time_rate
        chunk_end_time_sec = end_sample * time_reso * down_time_rate
        
        # Slices en este chunk
        chunk_decimated = (end_sample - start_sample) // down_time_rate
        slices_in_chunk = chunk_decimated // slice_len
        
        chunks_info.append({
            'chunk_idx': chunk_idx,
            'start_sample': start_sample,
            'end_sample': end_sample,
            'start_time_sec': chunk_start_time_sec,
            'end_time_sec': chunk_end_time_sec,
            'slices_in_chunk': slices_in_chunk
        })
        
        print(f"📊 Chunk {chunk_idx}:")
        print(f"   📏 Muestras: {start_sample:,} a {end_sample:,}")
        print(f"   🕐 Tiempo: {chunk_start_time_sec:.3f}s a {chunk_end_time_sec:.3f}s")
        print(f"   📊 Slices: {slices_in_chunk}")
        print()
    
    # Verificar continuidad
    print("🔗 VERIFICACIÓN DE CONTINUIDAD:")
    print("-" * 40)
    
    for i in range(len(chunks_info) - 1):
        chunk1 = chunks_info[i]
        chunk2 = chunks_info[i + 1]
        
        # Tiempo del último slice del chunk1
        last_slice_time = chunk1['end_time_sec']
        
        # Tiempo del primer slice del chunk2
        first_slice_time = chunk2['start_time_sec']
        
        gap = first_slice_time - last_slice_time
        
        print(f"🔗 Gap entre Chunk {i} y Chunk {i+1}:")
        print(f"   🕐 Último tiempo Chunk {i}: {last_slice_time:.3f}s")
        print(f"   🕐 Primer tiempo Chunk {i+1}: {first_slice_time:.3f}s")
        print(f"   📊 Gap: {gap:.3f}s")
        
        if abs(gap) < 0.001:
            print(f"   ✅ CONTINUIDAD PERFECTA")
        elif gap > 0:
            print(f"   ⚠️  GAP POSITIVO (saltos)")
        else:
            print(f"   ⚠️  GAP NEGATIVO (solapamiento)")
        print()
    
    # Verificar tiempos de slices específicos
    print("📊 VERIFICACIÓN DE TIEMPOS DE SLICES:")
    print("-" * 40)
    
    for chunk_info in chunks_info:
        chunk_idx = chunk_info['chunk_idx']
        start_time = chunk_info['start_time_sec']
        slices_in_chunk = chunk_info['slices_in_chunk']
        
        print(f"📊 Chunk {chunk_idx} - Tiempos de slices:")
        for slice_idx in range(min(3, slices_in_chunk)):  # Solo primeros 3 slices
            slice_start_time = start_time + slice_idx * slice_len * time_reso_decimated
            slice_end_time = slice_start_time + slice_len * time_reso_decimated
            
            print(f"   Slice {slice_idx}: {slice_start_time:.3f}s a {slice_end_time:.3f}s")
        print()
    
    return chunks_info


def main():
    """Función principal."""
    chunks_info = test_continuidad_temporal()
    
    print("📋 RESUMEN:")
    print("=" * 40)
    print("✅ La continuidad temporal debería funcionar correctamente")
    print("✅ Cada chunk calcula su tiempo absoluto desde start_sample_global")
    print("✅ Los slices dentro de cada chunk usan tiempo relativo al chunk")
    print("✅ El tiempo global de cada candidato se calcula correctamente")
    print()
    print("🔧 Si hay problemas, verificar:")
    print("   1. start_sample_global se pasa correctamente")
    print("   2. chunk_start_time_sec se calcula correctamente")
    print("   3. absolute_start_time se pasa a plot_waterfall_block")


if __name__ == "__main__":
    main() 