#!/usr/bin/env python3
"""
Test script para verificar que las carpetas de plots solo se crean cuando hay candidatos.
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path para importar DRAFTS
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from drafts import config
from drafts.pipeline import run_pipeline
from drafts.preprocessing.slice_len_calculator import get_processing_parameters
from drafts.logging.logging_config import setup_logging

def test_optimized_folders():
    """Prueba que las carpetas solo se crean cuando hay candidatos."""
    
    # Configurar logging usando configuraciones del sistema
    try:
        from drafts.config import LOG_LEVEL, LOG_COLORS
        logger = setup_logging(level=LOG_LEVEL, use_colors=LOG_COLORS)
    except ImportError:
        logger = setup_logging(level="INFO")
    
    print("🧪 PRUEBA: Optimización de Carpetas")
    print("=" * 50)
    
    # Configurar parámetros de prueba
    config.SLICE_DURATION_MS = 10.0  # 10ms por slice
    config.DATA_DIR = Path("./Data")
    
    # Verificar que existe al menos un archivo de datos
    data_files = list(config.DATA_DIR.glob("*.fil")) + list(config.DATA_DIR.glob("*.fits"))
    if not data_files:
        print("❌ No se encontraron archivos de datos en ./Data/")
        print("   Crear archivos .fil o .fits para la prueba")
        return False
    
    test_file = data_files[0]
    print(f"📁 Archivo de prueba: {test_file.name}")
    
    # Calcular parámetros automáticos
    try:
        params = get_processing_parameters()
        print(f"⚙️  Parámetros calculados:")
        print(f"   - Slice len: {params['slice_len']:,} muestras")
        print(f"   - Chunk samples: {params['chunk_samples']:,} muestras")
        print(f"   - Chunk duration: {params['chunk_duration_sec']:.1f}s")
        print(f"   - Slices per chunk: {params['slices_per_chunk']}")
    except Exception as e:
        print(f"❌ Error calculando parámetros: {e}")
        return False
    
    # Ejecutar pipeline con chunk_samples=0 (automático)
    print(f"\n🚀 Ejecutando pipeline con parámetros automáticos...")
    
    try:
        run_pipeline(chunk_samples=0)
        print("✅ Pipeline completado exitosamente")
        
        # Verificar estructura de carpetas
        results_dir = Path("./Results/ObjectDetection")
        if not results_dir.exists():
            print("❌ No se encontró el directorio de resultados")
            return False
        
        print(f"\n📊 ANÁLISIS DE CARPETAS:")
        print("=" * 30)
        
        # Verificar carpetas principales
        main_folders = ["Composite", "Detections", "Patches", "waterfall_dispersion", "waterfall_dedispersion"]
        
        for folder_name in main_folders:
            folder_path = results_dir / folder_name
            if folder_path.exists():
                # Contar subcarpetas
                subfolders = [f for f in folder_path.iterdir() if f.is_dir()]
                print(f"📁 {folder_name}: {len(subfolders)} subcarpetas")
                
                # Verificar si hay carpetas vacías
                empty_chunks = 0
                total_files = 0
                
                for subfolder in subfolders:
                    files = list(subfolder.glob("*.png"))
                    total_files += len(files)
                    if len(files) == 0:
                        empty_chunks += 1
                
                if empty_chunks > 0:
                    print(f"   ⚠️  {empty_chunks} carpetas de chunk vacías")
                else:
                    print(f"   ✅ Todas las carpetas tienen contenido")
                
                print(f"   📄 Total de archivos: {total_files}")
            else:
                print(f"📁 {folder_name}: No existe (correcto si no hay candidatos)")
        
        print(f"\n✅ PRUEBA COMPLETADA")
        return True
        
    except Exception as e:
        print(f"❌ Error ejecutando pipeline: {e}")
        return False

def test_empty_chunks_optimization():
    """Prueba específica para verificar que no se crean carpetas vacías."""
    
    print(f"\n🔍 PRUEBA ESPECÍFICA: Optimización de Carpetas Vacías")
    print("=" * 60)
    
    results_dir = Path("./Results/ObjectDetection")
    if not results_dir.exists():
        print("❌ No se encontró el directorio de resultados")
        return False
    
    # Verificar cada tipo de carpeta
    folder_types = {
        "Composite": "composite plots",
        "Detections": "detection plots", 
        "Patches": "patch plots",
        "waterfall_dispersion": "waterfall dispersion",
        "waterfall_dedispersion": "waterfall dedispersion"
    }
    
    optimization_working = True
    
    for folder_name, description in folder_types.items():
        folder_path = results_dir / folder_name
        if not folder_path.exists():
            print(f"✅ {folder_name}: No existe (correcto)")
            continue
        
        # Verificar subcarpetas
        subfolders = [f for f in folder_path.iterdir() if f.is_dir()]
        empty_chunks = 0
        
        for subfolder in subfolders:
            files = list(subfolder.glob("*.png"))
            if len(files) == 0:
                empty_chunks += 1
                print(f"   ⚠️  Carpeta vacía encontrada: {subfolder.name}")
        
        if empty_chunks == 0:
            print(f"✅ {folder_name}: Todas las carpetas tienen {description}")
        else:
            print(f"❌ {folder_name}: {empty_chunks} carpetas vacías de {description}")
            optimization_working = False
    
    if optimization_working:
        print(f"\n🎉 ¡OPTIMIZACIÓN FUNCIONANDO CORRECTAMENTE!")
        print("   Las carpetas solo se crean cuando hay contenido para guardar")
    else:
        print(f"\n⚠️  OPTIMIZACIÓN INCOMPLETA")
        print("   Algunas carpetas vacías fueron creadas")
    
    return optimization_working

if __name__ == "__main__":
    print("🧪 INICIANDO PRUEBAS DE OPTIMIZACIÓN DE CARPETAS")
    print("=" * 60)
    
    # Ejecutar pruebas
    success1 = test_optimized_folders()
    success2 = test_empty_chunks_optimization()
    
    if success1 and success2:
        print(f"\n🎉 TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
        print("   La optimización de carpetas está funcionando correctamente")
        sys.exit(0)
    else:
        print(f"\n❌ ALGUNAS PRUEBAS FALLARON")
        print("   Revisar la implementación de la optimización")
        sys.exit(1) 