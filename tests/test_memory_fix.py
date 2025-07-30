#!/usr/bin/env python3
"""
Script de prueba rápida para verificar que el problema de memoria se ha resuelto.
"""

import os
import sys
import time
import psutil
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRAFTS import config
from DRAFTS.filterbank_io import stream_fil


def get_memory_usage():
    """Obtener uso de memoria actual."""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB


def test_memory_efficiency(file_path: str, chunk_samples: int = 1_048_576):
    """Prueba la eficiencia de memoria del chunking."""
    
    print(f"🧪 Probando eficiencia de memoria")
    print(f"📁 Archivo: {file_path}")
    print(f"📊 Chunk size: {chunk_samples:,} muestras")
    print(f"{'='*80}")
    
    # Memoria inicial
    initial_memory = get_memory_usage()
    print(f"📊 Memoria inicial: {initial_memory:.1f} MB")
    
    chunk_count = 0
    total_samples = 0
    max_memory = initial_memory
    
    try:
        start_time = time.time()
        
        # Simular el procesamiento (solo lectura, sin procesamiento real)
        for block, metadata in stream_fil(file_path, chunk_samples):
            chunk_count += 1
            total_samples += metadata["actual_chunk_size"]
            
            # Verificar memoria después de cada chunk
            current_memory = get_memory_usage()
            max_memory = max(max_memory, current_memory)
            
            if chunk_count <= 3 or chunk_count % 5 == 0:  # Mostrar primeros 3 y cada 5
                print(f"🧩 Chunk {chunk_count:03d}: "
                      f"{metadata['actual_chunk_size']:,} muestras, "
                      f"memoria: {current_memory:.1f} MB")
            
            # Simular limpieza de memoria
            del block
            if hasattr(sys, 'getrefcount'):
                import gc
                gc.collect()
        
        end_time = time.time()
        runtime = end_time - start_time
        
        # Memoria final
        final_memory = get_memory_usage()
        
        print(f"\n📊 RESULTADOS:")
        print(f"   🧩 Chunks procesados: {chunk_count}")
        print(f"   📊 Muestras totales: {total_samples:,}")
        print(f"   ⏱️  Tiempo total: {runtime:.2f} s")
        print(f"   📊 Memoria inicial: {initial_memory:.1f} MB")
        print(f"   📊 Memoria máxima: {max_memory:.1f} MB")
        print(f"   📊 Memoria final: {final_memory:.1f} MB")
        print(f"   📊 Incremento máximo: {max_memory - initial_memory:.1f} MB")
        
        # Verificar que el uso de memoria sea razonable
        memory_increase = max_memory - initial_memory
        if memory_increase < 500:  # Menos de 500MB de incremento
            print(f"   ✅ Uso de memoria eficiente")
            return True
        else:
            print(f"   ⚠️  Uso de memoria alto: {memory_increase:.1f} MB")
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
        return False
    
    # Usar el primer archivo encontrado
    test_file = str(fil_files[0])
    print(f"🔍 Archivo de prueba: {test_file}")
    
    # Probar con chunk pequeño para verificar eficiencia
    chunk_size = 1_048_576  # 1M muestras
    
    print(f"\n{'='*80}")
    print(f"🧪 PRUEBA DE EFICIENCIA DE MEMORIA")
    print(f"{'='*80}")
    
    success = test_memory_efficiency(test_file, chunk_size)
    
    if success:
        print(f"\n🎉 Eficiencia de memoria verificada")
        print(f"   El pipeline debería funcionar sin problemas de OOM")
    else:
        print(f"\n⚠️  Posibles problemas de memoria detectados")
        print(f"   Considera reducir el tamaño de chunk")
    
    return success


if __name__ == "__main__":
    main() 
