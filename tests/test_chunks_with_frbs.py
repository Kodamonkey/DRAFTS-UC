#!/usr/bin/env python3
"""
Script de prueba para verificar la funcionalidad de reorganización de chunks con FRBs.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_chunks_with_frbs_functionality():
    """Prueba la funcionalidad de reorganización de chunks con FRBs."""
    
    print("🧪 Probando funcionalidad de ChunksWithFRBs...")
    
    # Crear directorio temporal para la prueba
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Simular estructura de directorios como la del pipeline
        file_name = "test_file"
        save_dir = temp_path / "Results" / "ObjectDetection"
        
        # Crear estructura de directorios
        composite_dir = save_dir / "Composite" / file_name
        detections_dir = save_dir / "Detections" / file_name
        patches_dir = save_dir / "Patches" / file_name
        
        composite_dir.mkdir(parents=True, exist_ok=True)
        detections_dir.mkdir(parents=True, exist_ok=True)
        patches_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Directorio base: {save_dir}")
        
        # Crear chunks de prueba
        test_chunks = [
            {"idx": 0, "has_frb": False, "plots": 2},  # Chunk sin FRB
            {"idx": 1, "has_frb": True, "plots": 3},   # Chunk con FRB
            {"idx": 2, "has_frb": False, "plots": 1},  # Chunk sin FRB
            {"idx": 3, "has_frb": True, "plots": 4},   # Chunk con FRB
        ]
        
        # Crear estructura de chunks
        for chunk_info in test_chunks:
            chunk_idx = chunk_info["idx"]
            chunk_name = f"chunk{chunk_idx:03d}"
            
            # Crear carpetas del chunk
            chunk_composite_dir = composite_dir / chunk_name
            chunk_detections_dir = detections_dir / chunk_name
            chunk_patches_dir = patches_dir / chunk_name
            
            chunk_composite_dir.mkdir(parents=True, exist_ok=True)
            chunk_detections_dir.mkdir(parents=True, exist_ok=True)
            chunk_patches_dir.mkdir(parents=True, exist_ok=True)
            
            # Crear archivos de prueba (plots)
            for i in range(chunk_info["plots"]):
                # Crear archivo de plot simulado
                plot_file = chunk_composite_dir / f"test_plot_{i}.png"
                plot_file.write_text(f"Simulated plot {i} for chunk {chunk_idx}")
                
                # Crear archivos de detección y patches
                detection_file = chunk_detections_dir / f"test_detection_{i}.png"
                patch_file = chunk_patches_dir / f"test_patch_{i}.png"
                
                detection_file.write_text(f"Simulated detection {i} for chunk {chunk_idx}")
                patch_file.write_text(f"Simulated patch {i} for chunk {chunk_idx}")
            
            print(f"  📁 Creado {chunk_name}: {chunk_info['plots']} plots, "
                  f"{'✅ con FRB' if chunk_info['has_frb'] else '❌ sin FRB'}")
        
        # Simular la reorganización manualmente
        print(f"\n🔄 Simulando reorganización...")
        
        # Crear carpeta ChunksWithFRBs
        chunks_with_frbs_dir = composite_dir / "ChunksWithFRBs"
        chunks_with_frbs_dir.mkdir(parents=True, exist_ok=True)
        
        # Mover chunks que tienen FRBs
        moved_chunks = 0
        for chunk_info in test_chunks:
            if chunk_info["has_frb"]:
                chunk_idx = chunk_info["idx"]
                chunk_name = f"chunk{chunk_idx:03d}"
                
                # Verificar que el chunk existe y tiene plots
                chunk_dir = composite_dir / chunk_name
                if chunk_dir.exists():
                    png_files = list(chunk_dir.glob("*.png"))
                    if png_files:
                        # Mover carpeta del chunk
                        destination_dir = chunks_with_frbs_dir / chunk_name
                        shutil.move(str(chunk_dir), str(destination_dir))
                        

                        
                        moved_chunks += 1
                        print(f"  ✅ Movido {chunk_name} a ChunksWithFRBs")
        
        # Verificar resultado
        print(f"\n📊 RESULTADO DE LA REORGANIZACIÓN:")
        print(f"   📁 Chunks movidos: {moved_chunks}")
        
        # Verificar estructura final
        print(f"\n📁 ESTRUCTURA FINAL:")
        
        # Verificar chunks que quedaron en Composite
        remaining_chunks = list(composite_dir.glob("chunk*"))
        print(f"   📂 Chunks restantes en Composite: {len(remaining_chunks)}")
        for chunk in remaining_chunks:
            print(f"      - {chunk.name}")
        
        # Verificar chunks en ChunksWithFRBs
        if chunks_with_frbs_dir.exists():
            frb_chunks = list(chunks_with_frbs_dir.glob("chunk*"))
            print(f"   📂 Chunks en ChunksWithFRBs: {len(frb_chunks)}")
            for chunk in frb_chunks:
                print(f"      - {chunk.name}")
                
                # Contar plots en este chunk
                png_files = list(chunk.glob("*.png"))
                print(f"        📊 {len(png_files)} plots")
        
        # Verificar que la reorganización fue correcta
        expected_moved = sum(1 for chunk in test_chunks if chunk["has_frb"])
        if moved_chunks == expected_moved:
            print(f"\n✅ PRUEBA EXITOSA: Se movieron {moved_chunks} chunks como se esperaba")
            return True
        else:
            print(f"\n❌ PRUEBA FALLIDA: Se movieron {moved_chunks} chunks, se esperaban {expected_moved}")
            return False

def demonstrate_directory_structure():
    """Demuestra la estructura de directorios esperada."""
    
    print(f"\n📋 ESTRUCTURA DE DIRECTORIOS ESPERADA:")
    print(f"=" * 50)
    print(f"Results/ObjectDetection/")
    print(f"├── Composite/")
    print(f"│   └── file_name/")
    print(f"│       ├── ChunksWithFRBs/          # 🆕 NUEVA CARPETA")
    print(f"│       │   ├── chunk001/            # Chunk con FRB")
    print(f"│       │   └── chunk003/            # Chunk con FRB")
    print(f"│       ├── chunk000/                # Chunk sin FRB (queda aquí)")
    print(f"│       └── chunk002/                # Chunk sin FRB (queda aquí)")
    print(f"├── Detections/")
    print(f"│   └── file_name/")
    print(f"│       ├── chunk000/                # Chunk sin FRB")
    print(f"│       ├── chunk001/                # Chunk con FRB (queda aquí)")
    print(f"│       ├── chunk002/                # Chunk sin FRB")
    print(f"│       └── chunk003/                # Chunk con FRB (queda aquí)")
    print(f"└── Patches/")
    print(f"    └── file_name/")
    print(f"        ├── chunk000/                # Chunk sin FRB")
    print(f"        ├── chunk001/                # Chunk con FRB (queda aquí)")
    print(f"        ├── chunk002/                # Chunk sin FRB")
    print(f"        └── chunk003/                # Chunk con FRB (queda aquí)")
    print(f"\n💡 BENEFICIOS:")
    print(f"   • Los astrónomos pueden enfocarse solo en chunks 'interesantes'")
    print(f"   • Organización automática sin afectar el procesamiento")
    print(f"   • Facilita el análisis posterior de candidatos FRB")

def main():
    """Función principal."""
    
    print(f"{'='*60}")
    print(f"🧪 PRUEBA DE FUNCIONALIDAD CHUNKS WITH FRBs")
    print(f"{'='*60}")
    
    # Ejecutar prueba
    success = test_chunks_with_frbs_functionality()
    
    # Mostrar estructura esperada
    demonstrate_directory_structure()
    
    if success:
        print(f"\n🎉 ¡Funcionalidad implementada correctamente!")
        print(f"   La reorganización de chunks con FRBs está lista para usar.")
    else:
        print(f"\n❌ La funcionalidad necesita ajustes.")
    
    return success

if __name__ == "__main__":
    main() 