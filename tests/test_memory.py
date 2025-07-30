#!/usr/bin/env python3
"""
Script de prueba para verificar el uso de memoria con el sistema de chunking.
"""

import os
import sys
import psutil
import time
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRAFTS import config
from DRAFTS.filterbank_io import stream_fil


def get_memory_usage():
    """Obtiene el uso de memoria actual en GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)


def test_chunking_memory(file_path: str, chunk_samples: int = 2_097_152):
    """Prueba el uso de memoria con chunking."""
    
    print(f"🧪 Probando chunking con {chunk_samples:,} muestras por bloque")
    print(f"📁 Archivo: {file_path}")
    
    # Memoria inicial
    mem_initial = get_memory_usage()
    print(f"💾 Memoria inicial: {mem_initial:.2f} GB")
    
    chunk_count = 0
    total_samples = 0
    mem_peak = mem_initial
    
    try:
        start_time = time.time()
        
        for block, metadata in stream_fil(file_path, chunk_samples):
            chunk_count += 1
            total_samples += metadata["actual_chunk_size"]
            
            # Verificar memoria
            mem_current = get_memory_usage()
            mem_peak = max(mem_peak, mem_current)
            
            print(f"🧩 Chunk {chunk_count:03d}: "
                  f"{metadata['actual_chunk_size']:,} muestras, "
                  f"memoria: {mem_current:.2f} GB")
            
            # Simular procesamiento
            time.sleep(0.1)  # Simular trabajo
        
        end_time = time.time()
        runtime = end_time - start_time
        
        # Memoria final
        mem_final = get_memory_usage()
        
        print(f"\n✅ Prueba completada:")
        print(f"   📊 Chunks procesados: {chunk_count}")
        print(f"   📊 Muestras totales: {total_samples:,}")
        print(f"   ⏱️  Tiempo total: {runtime:.2f} s")
        print(f"   💾 Memoria inicial: {mem_initial:.2f} GB")
        print(f"   💾 Memoria pico: {mem_peak:.2f} GB")
        print(f"   💾 Memoria final: {mem_final:.2f} GB")
        print(f"   💾 Incremento pico: {mem_peak - mem_initial:.2f} GB")
        
        # Verificar criterios de aceptación
        success = True
        if mem_peak > 1.5:
            print(f"❌ ERROR: Memoria pico ({mem_peak:.2f} GB) excede 1.5 GB")
            success = False
        else:
            print(f"✅ Memoria pico ({mem_peak:.2f} GB) dentro del límite")
        
        if total_samples < 66_000_000:
            print(f"❌ ERROR: Solo se procesaron {total_samples:,} muestras de 66M")
            success = False
        else:
            print(f"✅ Se procesaron {total_samples:,} muestras (≥66M)")
        
        return success
        
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
    
    # Probar diferentes tamaños de chunk
    chunk_sizes = [
        1_048_576,    # 1M muestras
        2_097_152,    # 2M muestras (default)
        4_194_304,    # 4M muestras
    ]
    
    results = {}
    
    for chunk_size in chunk_sizes:
        print(f"\n{'='*60}")
        print(f"🧪 PRUEBA CON CHUNK_SIZE = {chunk_size:,}")
        print(f"{'='*60}")
        
        success = test_chunking_memory(test_file, chunk_size)
        results[chunk_size] = success
        
        # Pausa entre pruebas
        time.sleep(2)
    
    # Resumen final
    print(f"\n{'='*60}")
    print(f"📋 RESUMEN DE PRUEBAS")
    print(f"{'='*60}")
    
    for chunk_size, success in results.items():
        status = "✅ PASÓ" if success else "❌ FALLÓ"
        print(f"   {chunk_size:,} muestras/chunk: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print(f"\n🎉 TODAS LAS PRUEBAS PASARON")
    else:
        print(f"\n⚠️  ALGUNAS PRUEBAS FALLARON")
    
    return all_passed


if __name__ == "__main__":
    main() 
