#!/usr/bin/env python3
"""
Script de prueba para verificar la estructura de directorios.
"""

import os
from pathlib import Path

def test_directory_structure():
    """Verificar que la estructura de directorios se crea correctamente."""
    
    # Simular directorio de resultados
    results_dir = Path("./tests/test_directory_output")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Simular archivos procesados
    test_files = [
        "FRB20201124_0009.fits",
        "B0355+54_FB_20220918.fits"
    ]
    
    print("🧪 Probando estructura de directorios...")
    print(f"📁 Directorio base: {results_dir}")
    
    for filename in test_files:
        file_stem = Path(filename).stem
        print(f"\n📄 Procesando: {filename}")
        
        # Crear estructura de directorios como lo haría el pipeline
        file_folder_name = file_stem
        chunk_folder_name = "chunk000"  # Para archivos no chunked
        
        # Directorios principales
        composite_dir = results_dir / "Composite" / file_folder_name / chunk_folder_name
        detections_dir = results_dir / "Detections" / file_folder_name / chunk_folder_name
        patches_dir = results_dir / "Patches" / file_folder_name / chunk_folder_name
        waterfall_dispersion_dir = results_dir / "waterfall_dispersion" / file_folder_name / chunk_folder_name
        waterfall_dedispersion_dir = results_dir / "waterfall_dedispersion" / file_folder_name / chunk_folder_name
        
        # Crear directorios
        composite_dir.mkdir(parents=True, exist_ok=True)
        detections_dir.mkdir(parents=True, exist_ok=True)
        patches_dir.mkdir(parents=True, exist_ok=True)
        waterfall_dispersion_dir.mkdir(parents=True, exist_ok=True)
        waterfall_dedispersion_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  ✅ Composite: {composite_dir}")
        print(f"  ✅ Detections: {detections_dir}")
        print(f"  ✅ Patches: {patches_dir}")
        print(f"  ✅ Waterfall dispersion: {waterfall_dispersion_dir}")
        print(f"  ✅ Waterfall dedispersion: {waterfall_dedispersion_dir}")
        
        # Crear archivos de prueba
        test_files_to_create = [
            (composite_dir / f"{file_stem}_slice000.png", "Composite plot"),
            (detections_dir / f"{file_stem}_slice000.png", "Detection plot"),
            (patches_dir / f"{file_stem}_slice000.png", "Patch plot"),
            (waterfall_dispersion_dir / f"{file_stem}_slice000.png", "Waterfall dispersion"),
            (waterfall_dedispersion_dir / f"{file_stem}_slice000.png", "Waterfall dedispersion")
        ]
        
        for file_path, description in test_files_to_create:
            file_path.write_text(f"Test {description} for {filename}")
            print(f"    📄 Creado: {file_path.name}")
    
    # Mostrar estructura final
    print(f"\n📋 Estructura final de directorios:")
    for root, dirs, files in os.walk(results_dir):
        level = root.replace(str(results_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")
    
    print(f"\n✅ Prueba completada. Revisa la estructura en: {results_dir}")

if __name__ == "__main__":
    test_directory_structure()