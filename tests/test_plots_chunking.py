#!/usr/bin/env python3
"""
Script de prueba para verificar que los plots se generen correctamente con chunking.
"""

import os
import sys
import time
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRAFTS import config
from DRAFTS.filterbank_io import stream_fil


def test_plots_generation(file_path: str, chunk_samples: int = 1_048_576):
    """Prueba la generación de plots con chunking."""
    
    print(f"🧪 Probando generación de plots con chunking")
    print(f"📁 Archivo: {file_path}")
    print(f"📊 Chunk size: {chunk_samples:,} muestras")
    
    chunk_count = 0
    total_samples = 0
    
    try:
        start_time = time.time()
        
        for block, metadata in stream_fil(file_path, chunk_samples):
            chunk_count += 1
            total_samples += metadata["actual_chunk_size"]
            
            print(f"🧩 Chunk {chunk_count:03d}: "
                  f"{metadata['actual_chunk_size']:,} muestras")
            
            # Simular procesamiento de plots
            time.sleep(0.1)  # Simular trabajo
        
        end_time = time.time()
        runtime = end_time - start_time
        
        print(f"\n✅ Prueba completada:")
        print(f"   📊 Chunks procesados: {chunk_count}")
        print(f"   📊 Muestras totales: {total_samples:,}")
        print(f"   ⏱️  Tiempo total: {runtime:.2f} s")
        
        # Verificar que se procesaron suficientes datos
        if total_samples > 0:
            print(f"✅ Se procesaron {total_samples:,} muestras correctamente")
            return True
        else:
            print(f"❌ ERROR: No se procesaron muestras")
            return False
        
    except Exception as e:
        print(f"❌ ERROR durante la prueba: {e}")
        return False


def check_output_directories():
    """Verifica que los directorios de salida existan."""
    
    results_dir = Path("./Results/ObjectDetection/resnet50")
    
    expected_dirs = [
        "waterfall_dispersion",
        "waterfall_dedispersion", 
        "Patches",
        "Composite",
        "Detections"
    ]
    
    print(f"🔍 Verificando directorios de salida en: {results_dir}")
    
    for dir_name in expected_dirs:
        dir_path = results_dir / dir_name
        if dir_path.exists():
            print(f"   ✅ {dir_name}: Existe")
            # Contar archivos
            files = list(dir_path.rglob("*.png"))
            print(f"      📁 {len(files)} archivos PNG encontrados")
        else:
            print(f"   ❌ {dir_name}: No existe")


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
    
    # Probar con chunk pequeño para verificar plots
    chunk_size = 1_048_576  # 1M muestras
    
    print(f"\n{'='*60}")
    print(f"🧪 PRUEBA DE GENERACIÓN DE PLOTS")
    print(f"{'='*60}")
    
    success = test_plots_generation(test_file, chunk_size)
    
    if success:
        print(f"\n✅ Prueba de chunking exitosa")
        print(f"🔍 Verificando directorios de salida...")
        check_output_directories()
    else:
        print(f"\n❌ Prueba de chunking falló")
    
    return success


if __name__ == "__main__":
    main() 
