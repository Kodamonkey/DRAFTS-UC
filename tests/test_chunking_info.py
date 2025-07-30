#!/usr/bin/env python3
"""
Script de prueba para verificar la información de chunks y continuidad temporal.
"""

import os
import sys
import time
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRAFTS import config
from DRAFTS.filterbank_io import stream_fil


def test_chunking_info(file_path: str, chunk_samples: int = 1_048_576):
    """Prueba la información de chunks y continuidad temporal."""
    
    print(f"🧩 Probando información de chunking")
    print(f"📁 Archivo: {file_path}")
    print(f"📊 Chunk size: {chunk_samples:,} muestras")
    print(f"{'='*80}")
    
    # Primera pasada: analizar estructura
    print(f"📊 ANÁLISIS DE ESTRUCTURA DEL ARCHIVO:")
    chunk_count = 0
    total_samples = 0
    total_duration_sec = 0.0
    chunk_info = []
    
    for _, metadata in stream_fil(file_path, chunk_samples):
        chunk_count += 1
        total_samples += metadata["actual_chunk_size"]
        chunk_duration_sec = metadata["actual_chunk_size"] * config.TIME_RESO
        total_duration_sec += chunk_duration_sec
        
        chunk_info.append({
            "chunk_idx": chunk_count - 1,
            "start_sample": metadata["start_sample"],
            "end_sample": metadata["end_sample"],
            "actual_size": metadata["actual_chunk_size"],
            "start_time_sec": metadata["start_sample"] * config.TIME_RESO,
            "end_time_sec": (metadata["start_sample"] + metadata["actual_chunk_size"]) * config.TIME_RESO,
            "duration_sec": chunk_duration_sec
        })
    
    print(f"   🧩 Total de chunks: {chunk_count}")
    print(f"   📊 Muestras totales: {total_samples:,}")
    print(f"   🕐 Duración total: {total_duration_sec:.2f} segundos ({total_duration_sec/60:.1f} minutos)")
    print(f"   📦 Tamaño de chunk: {chunk_samples:,} muestras ({chunk_samples * config.TIME_RESO:.2f}s)")
    
    # Verificar continuidad temporal
    print(f"\n🕐 VERIFICACIÓN DE CONTINUIDAD TEMPORAL:")
    print(f"{'='*80}")
    
    for i, info in enumerate(chunk_info):
        print(f"Chunk {info['chunk_idx']:03d}: "
              f"Tiempo {info['start_time_sec']:.2f}s - {info['end_time_sec']:.2f}s "
              f"(duración: {info['duration_sec']:.2f}s)")
        
        # Verificar continuidad con el siguiente chunk
        if i < len(chunk_info) - 1:
            next_info = chunk_info[i + 1]
            gap = next_info['start_time_sec'] - info['end_time_sec']
            
            if abs(gap) < 1e-6:  # Tolerancia para errores de punto flotante
                print(f"   ✅ Continuidad perfecta con siguiente chunk")
            else:
                print(f"   ⚠️  Gap detectado: {gap:.6f}s")
    
    # Verificar cobertura total
    print(f"\n📊 RESUMEN FINAL:")
    print(f"{'='*80}")
    
    actual_duration = chunk_info[-1]['end_time_sec'] - chunk_info[0]['start_time_sec']
    expected_duration = total_samples * config.TIME_RESO
    
    print(f"   🕐 Tiempo total cubierto: {actual_duration:.2f}s")
    print(f"   🕐 Tiempo esperado: {expected_duration:.2f}s")
    print(f"   🕐 Diferencia: {abs(actual_duration - expected_duration):.6f}s")
    
    if abs(actual_duration - expected_duration) < 1e-6:
        print(f"   ✅ Cobertura temporal completa")
        continuity_ok = True
    else:
        print(f"   ❌ Error en cobertura temporal")
        continuity_ok = False
    
    # Verificar que el último chunk no exceda el archivo
    last_chunk = chunk_info[-1]
    if last_chunk['end_sample'] <= total_samples:
        print(f"   ✅ Último chunk dentro de límites del archivo")
        bounds_ok = True
    else:
        print(f"   ❌ Último chunk excede límites del archivo")
        bounds_ok = False
    
    return continuity_ok and bounds_ok


def main():
    """Función principal."""
    
    # Buscar archivos .fil en el directorio de datos
    data_dir = Path("./Data")
    fil_files = list(data_dir.glob("*.fil"))
    
    if not fil_files:
        print("❌ No se encontraron archivos .fil en ./Data/")
        print("   Coloque un archivo .fil en ./Data/ para probar")
        return False
    
    # Usar el primer archivo encontrado
    test_file = str(fil_files[0])
    print(f"🔍 Archivo de prueba: {test_file}")
    
    # Probar con diferentes tamaños de chunk
    chunk_sizes = [524_288, 1_048_576, 2_097_152]  # 0.5M, 1M, 2M muestras
    
    all_tests_passed = True
    
    for chunk_size in chunk_sizes:
        print(f"\n{'='*80}")
        print(f"🧩 PRUEBA CON CHUNK SIZE: {chunk_size:,} muestras")
        print(f"{'='*80}")
        
        success = test_chunking_info(test_file, chunk_size)
        
        if success:
            print(f"✅ Prueba exitosa con chunk size {chunk_size:,}")
        else:
            print(f"❌ Prueba fallida con chunk size {chunk_size:,}")
            all_tests_passed = False
    
    print(f"\n{'='*80}")
    if all_tests_passed:
        print(f"🎉 TODAS LAS PRUEBAS EXITOSAS")
    else:
        print(f"⚠️  ALGUNAS PRUEBAS FALLARON")
    print(f"{'='*80}")
    
    return all_tests_passed


if __name__ == "__main__":
    main() 
