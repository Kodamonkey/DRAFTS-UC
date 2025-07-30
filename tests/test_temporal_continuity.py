#!/usr/bin/env python3
"""
Script de prueba para verificar la continuidad temporal entre chunks.
"""

import os
import sys
import time
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRAFTS import config
from DRAFTS.filterbank_io import stream_fil


def test_temporal_continuity(file_path: str, chunk_samples: int = 1_048_576):
    """Prueba la continuidad temporal entre chunks."""
    
    print(f"🕐 Probando continuidad temporal con chunking")
    print(f"📁 Archivo: {file_path}")
    print(f"📊 Chunk size: {chunk_samples:,} muestras")
    
    chunk_count = 0
    total_samples = 0
    temporal_info = []
    
    try:
        start_time = time.time()
        
        for block, metadata in stream_fil(file_path, chunk_samples):
            chunk_count += 1
            total_samples += metadata["actual_chunk_size"]
            
            # Calcular tiempos absolutos
            chunk_start_time_sec = metadata["start_sample"] * config.TIME_RESO
            chunk_duration_sec = metadata["actual_chunk_size"] * config.TIME_RESO
            chunk_end_time_sec = chunk_start_time_sec + chunk_duration_sec
            
            temporal_info.append({
                "chunk_idx": chunk_count - 1,
                "start_sample": metadata["start_sample"],
                "end_sample": metadata["end_sample"],
                "start_time_sec": chunk_start_time_sec,
                "end_time_sec": chunk_end_time_sec,
                "duration_sec": chunk_duration_sec
            })
            
            print(f"🧩 Chunk {chunk_count-1:03d}: "
                  f"{metadata['actual_chunk_size']:,} muestras, "
                  f"tiempo {chunk_start_time_sec:.2f}s - {chunk_end_time_sec:.2f}s "
                  f"(duración: {chunk_duration_sec:.2f}s)")
        
        end_time = time.time()
        runtime = end_time - start_time
        
        print(f"\n✅ Prueba completada:")
        print(f"   📊 Chunks procesados: {chunk_count}")
        print(f"   📊 Muestras totales: {total_samples:,}")
        print(f"   ⏱️  Tiempo total: {runtime:.2f} s")
        
        # Verificar continuidad temporal
        print(f"\n🕐 ANÁLISIS DE CONTINUIDAD TEMPORAL:")
        print(f"{'='*80}")
        
        for i, info in enumerate(temporal_info):
            print(f"Chunk {info['chunk_idx']:03d}: "
                  f"Tiempo {info['start_time_sec']:.2f}s - {info['end_time_sec']:.2f}s "
                  f"(duración: {info['duration_sec']:.2f}s)")
            
            # Verificar continuidad con el siguiente chunk
            if i < len(temporal_info) - 1:
                next_info = temporal_info[i + 1]
                gap = next_info['start_time_sec'] - info['end_time_sec']
                
                if abs(gap) < 1e-6:  # Tolerancia para errores de punto flotante
                    print(f"   ✅ Continuidad perfecta con siguiente chunk")
                else:
                    print(f"   ⚠️  Gap detectado: {gap:.6f}s")
        
        # Verificar cobertura total
        total_duration = temporal_info[-1]['end_time_sec'] - temporal_info[0]['start_time_sec']
        expected_duration = total_samples * config.TIME_RESO
        
        print(f"\n📊 RESUMEN TEMPORAL:")
        print(f"   🕐 Tiempo total cubierto: {total_duration:.2f}s")
        print(f"   🕐 Tiempo esperado: {expected_duration:.2f}s")
        print(f"   🕐 Diferencia: {abs(total_duration - expected_duration):.6f}s")
        
        if abs(total_duration - expected_duration) < 1e-6:
            print(f"   ✅ Cobertura temporal completa")
            return True
        else:
            print(f"   ❌ Error en cobertura temporal")
            return False
        
    except Exception as e:
        print(f"❌ ERROR durante la prueba: {e}")
        return False


def main():
    """Función principal."""
    
    # Buscar archivos .fil en el directorio de datos
    data_dir = Path("./Data")
    fil_files = list(data_dir.glob("*.fil"))
    
    if not fil_files:
        print("❌ No se encontraron archivos .fil en ./Data/")
        print("   Coloque un archivo .fil en ./Data/ para probar")
        return
    
    # Usar el primer archivo encontrado
    test_file = str(fil_files[0])
    print(f"🔍 Archivo de prueba: {test_file}")
    
    # Probar con chunk pequeño para verificar continuidad
    chunk_size = 1_048_576  # 1M muestras
    
    print(f"\n{'='*80}")
    print(f"🕐 PRUEBA DE CONTINUIDAD TEMPORAL")
    print(f"{'='*80}")
    
    success = test_temporal_continuity(test_file, chunk_size)
    
    if success:
        print(f"\n🎉 Continuidad temporal verificada correctamente")
    else:
        print(f"\n⚠️  Problemas detectados en continuidad temporal")
    
    return success


if __name__ == "__main__":
    main() 
