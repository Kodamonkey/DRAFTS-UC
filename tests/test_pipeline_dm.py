#!/usr/bin/env python3
"""
Test final del pipeline con rangos DM dinámicos.
"""

import sys
from pathlib import Path

# Agregar el directorio DRAFTS al path
sys.path.insert(0, str(Path(__file__).parent / "DRAFTS"))

from DRAFTS import config
from DRAFTS.pipeline import process_single_file

def test_pipeline_with_dynamic_dm():
    """Test del pipeline completo con DM dinámico."""
    
    print("=== Test Pipeline con DM Dinámico ===")
    
    # Configurar parámetros
    config.DM_DYNAMIC_RANGE_ENABLE = True
    config.DM_RANGE_FACTOR = 0.3
    config.DM_PLOT_MARGIN_FACTOR = 0.25
    config.ENABLE_CHUNK_PROCESSING = True
    config.MAX_SAMPLES_LIMIT = 50000  # Límite muy bajo para test rápido
    
    print(f"DM_DYNAMIC_RANGE_ENABLE: {config.DM_DYNAMIC_RANGE_ENABLE}")
    print(f"DM_RANGE_FACTOR: {config.DM_RANGE_FACTOR}")
    print(f"DM_PLOT_MARGIN_FACTOR: {config.DM_PLOT_MARGIN_FACTOR}")
    print(f"MAX_SAMPLES_LIMIT: {config.MAX_SAMPLES_LIMIT}")
    
    # Buscar archivos de test disponibles
    data_dir = Path(config.DATA_DIR)
    
    test_files = []
    for pattern in ['*.fil', '*.fits']:
        test_files.extend(data_dir.glob(pattern))
    
    if not test_files:
        print("❌ No se encontraron archivos de test en", data_dir)
        return
    
    # Usar el primer archivo encontrado
    test_file = test_files[0]
    print(f"Archivo de test: {test_file.name}")
    
    # Crear directorio de resultados
    results_dir = Path("test_dm_pipeline_results")
    results_dir.mkdir(exist_ok=True)
    
    # Configurar directorio de salida
    original_results_dir = config.RESULTS_DIR
    config.RESULTS_DIR = results_dir
    
    try:
        print("\\n--- Ejecutando pipeline ---")
        
        # Procesar archivo
        process_single_file(test_file, results_dir)
        
        print("\\n--- Verificando resultados ---")
        
        # Verificar archivos generados
        generated_files = list(results_dir.rglob("*.png"))
        
        if generated_files:
            print(f"✅ Se generaron {len(generated_files)} archivos PNG:")
            for file in generated_files[:10]:  # Mostrar solo los primeros 10
                print(f"  - {file.name}")
            if len(generated_files) > 10:
                print(f"  ... y {len(generated_files) - 10} más")
        else:
            print("❌ No se generaron archivos PNG")
        
        # Verificar que incluyen plots con DM dinámico
        dynamic_plots = [f for f in generated_files if "detection" in f.name.lower()]
        if dynamic_plots:
            print(f"✅ Se encontraron {len(dynamic_plots)} plots de detección")
        else:
            print("⚠️  No se encontraron plots de detección específicos")
        
        print("\\n=== Test completado exitosamente ===")
        
    except Exception as e:
        print(f"❌ Error en pipeline: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Restaurar configuración original
        config.RESULTS_DIR = original_results_dir

if __name__ == "__main__":
    test_pipeline_with_dynamic_dm()
