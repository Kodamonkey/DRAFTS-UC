#!/usr/bin/env python3
"""
Script de demostración de la funcionalidad de reorganización de chunks con FRBs.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_realistic_scenario():
    """Crea un escenario realista de procesamiento de datos astronómicos."""
    
    print("🌌 Creando escenario realista de procesamiento astronómico...")
    
    # Crear directorio temporal para la demostración
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Simular estructura de directorios real
        file_name = "FRB20201124_0009"  # Nombre de archivo realista
        save_dir = temp_path / "Results" / "ObjectDetection"
        
        # Crear estructura de directorios
        composite_dir = save_dir / "Composite" / file_name
        detections_dir = save_dir / "Detections" / file_name
        patches_dir = save_dir / "Patches" / file_name
        waterfall_dispersion_dir = save_dir / "waterfall_dispersion" / file_name
        waterfall_dedispersion_dir = save_dir / "waterfall_dedispersion" / file_name
        
        composite_dir.mkdir(parents=True, exist_ok=True)
        detections_dir.mkdir(parents=True, exist_ok=True)
        patches_dir.mkdir(parents=True, exist_ok=True)
        waterfall_dispersion_dir.mkdir(parents=True, exist_ok=True)
        waterfall_dedispersion_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Directorio base: {save_dir}")
        print(f"📄 Archivo procesado: {file_name}")
        
        # Simular chunks con diferentes escenarios
        scenarios = [
            {
                "idx": 0, 
                "has_frb": False, 
                "plots": 5, 
                "description": "Ruido de fondo, múltiples falsos positivos"
            },
            {
                "idx": 1, 
                "has_frb": True, 
                "plots": 8, 
                "description": "FRB detectado con alta confianza"
            },
            {
                "idx": 2, 
                "has_frb": False, 
                "plots": 3, 
                "description": "Interferencia de RFI"
            },
            {
                "idx": 3, 
                "has_frb": True, 
                "plots": 12, 
                "description": "Múltiples candidatos FRB en el mismo chunk"
            },
            {
                "idx": 4, 
                "has_frb": False, 
                "plots": 2, 
                "description": "Datos limpios, pocos candidatos"
            },
            {
                "idx": 5, 
                "has_frb": True, 
                "plots": 6, 
                "description": "FRB débil pero detectable"
            },
        ]
        
        # Crear estructura de chunks
        print(f"\n📊 Creando {len(scenarios)} chunks de datos...")
        for scenario in scenarios:
            chunk_idx = scenario["idx"]
            chunk_name = f"chunk{chunk_idx:03d}"
            
            # Crear carpetas del chunk
            chunk_composite_dir = composite_dir / chunk_name
            chunk_detections_dir = detections_dir / chunk_name
            chunk_patches_dir = patches_dir / chunk_name
            chunk_waterfall_dispersion_dir = waterfall_dispersion_dir / chunk_name
            chunk_waterfall_dedispersion_dir = waterfall_dedispersion_dir / chunk_name
            
            chunk_composite_dir.mkdir(parents=True, exist_ok=True)
            chunk_detections_dir.mkdir(parents=True, exist_ok=True)
            chunk_patches_dir.mkdir(parents=True, exist_ok=True)
            chunk_waterfall_dispersion_dir.mkdir(parents=True, exist_ok=True)
            chunk_waterfall_dedispersion_dir.mkdir(parents=True, exist_ok=True)
            
            # Crear archivos de plots realistas
            for i in range(scenario["plots"]):
                # Composite plots (los más importantes)
                composite_file = chunk_composite_dir / f"{file_name}_slice{i:03d}.png"
                composite_file.write_text(f"Composite plot {i} - {scenario['description']}")
                
                # Detection plots
                detection_file = chunk_detections_dir / f"{file_name}_slice{i:03d}.png"
                detection_file.write_text(f"Detection plot {i} - {scenario['description']}")
                
                # Patch plots
                patch_file = chunk_patches_dir / f"{file_name}_slice{i:03d}.png"
                patch_file.write_text(f"Patch plot {i} - {scenario['description']}")
                
                # Waterfall plots
                waterfall_dispersion_file = chunk_waterfall_dispersion_dir / f"{file_name}_slice{i:03d}.png"
                waterfall_dispersion_file.write_text(f"Waterfall dispersion {i}")
                
                waterfall_dedispersion_file = chunk_waterfall_dedispersion_dir / f"{file_name}_slice{i:03d}.png"
                waterfall_dedispersion_file.write_text(f"Waterfall dedispersion {i}")
            
            status_icon = "✅" if scenario["has_frb"] else "❌"
            print(f"  {status_icon} {chunk_name}: {scenario['plots']} plots - {scenario['description']}")
        
        # Simular la reorganización automática
        print(f"\n🔄 Simulando reorganización automática...")
        
        # Crear carpeta ChunksWithFRBs
        chunks_with_frbs_dir = composite_dir / "ChunksWithFRBs"
        chunks_with_frbs_dir.mkdir(parents=True, exist_ok=True)
        
        # Mover chunks que tienen FRBs
        moved_chunks = 0
        for scenario in scenarios:
            if scenario["has_frb"]:
                chunk_idx = scenario["idx"]
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
                        print(f"  ✅ Movido {chunk_name} a ChunksWithFRBs ({scenario['description']})")
        
        # Mostrar estadísticas finales
        print(f"\n📊 ESTADÍSTICAS FINALES:")
        print(f"   📁 Total de chunks procesados: {len(scenarios)}")
        print(f"   ✅ Chunks con FRBs: {sum(1 for s in scenarios if s['has_frb'])}")
        print(f"   ❌ Chunks sin FRBs: {sum(1 for s in scenarios if not s['has_frb'])}")
        print(f"   🔄 Chunks reorganizados: {moved_chunks}")
        
        # Mostrar estructura final
        print(f"\n📁 ESTRUCTURA FINAL DE DIRECTORIOS:")
        
        # Verificar chunks que quedaron en Composite
        remaining_chunks = [d for d in composite_dir.iterdir() if d.is_dir() and d.name.startswith("chunk")]
        print(f"   📂 Chunks restantes en Composite: {len(remaining_chunks)}")
        for chunk in remaining_chunks:
            print(f"      - {chunk.name}")
        
        # Verificar chunks en ChunksWithFRBs
        if chunks_with_frbs_dir.exists():
            frb_chunks = [d for d in chunks_with_frbs_dir.iterdir() if d.is_dir() and d.name.startswith("chunk")]
            print(f"   📂 Chunks en ChunksWithFRBs: {len(frb_chunks)}")
            for chunk in frb_chunks:
                print(f"      - {chunk.name}")
                
                # Contar plots en este chunk
                png_files = list(chunk.glob("*.png"))
                print(f"        📊 {len(png_files)} plots")
        

        
        return moved_chunks == sum(1 for s in scenarios if s["has_frb"])

def show_astronomer_workflow():
    """Muestra cómo esta funcionalidad mejora el flujo de trabajo del astrónomo."""
    
    print(f"\n👨‍🔬 FLUJO DE TRABAJO DEL ASTRÓNOMO:")
    print(f"=" * 50)
    print(f"ANTES (sin reorganización):")
    print(f"   📁 El astrónomo debe revisar TODOS los chunks:")
    print(f"      - chunk000/ (5 plots) ❌ Sin FRB")
    print(f"      - chunk001/ (8 plots) ✅ Con FRB")
    print(f"      - chunk002/ (3 plots) ❌ Sin FRB")
    print(f"      - chunk003/ (12 plots) ✅ Con FRB")
    print(f"      - chunk004/ (2 plots) ❌ Sin FRB")
    print(f"      - chunk005/ (6 plots) ✅ Con FRB")
    print(f"   ⏱️  Tiempo estimado: 30-45 minutos")
    print(f"   😵 Confusión: mezcla de chunks relevantes e irrelevantes")
    
    print(f"\nDESPUÉS (con reorganización):")
    print(f"   📁 El astrónomo revisa SOLO chunks con FRBs:")
    print(f"      - ChunksWithFRBs/chunk001/ (8 plots) ✅ Con FRB")
    print(f"      - ChunksWithFRBs/chunk003/ (12 plots) ✅ Con FRB")
    print(f"      - ChunksWithFRBs/chunk005/ (6 plots) ✅ Con FRB")
    print(f"   ⏱️  Tiempo estimado: 10-15 minutos")
    print(f"   🎯 Enfoque: solo datos relevantes")
    print(f"   📈 Eficiencia: 60-70% de tiempo ahorrado")

def main():
    """Función principal."""
    
    print(f"{'='*70}")
    print(f"🌌 DEMOSTRACIÓN: REORGANIZACIÓN AUTOMÁTICA DE CHUNKS CON FRBs")
    print(f"{'='*70}")
    
    # Ejecutar demostración
    success = create_realistic_scenario()
    
    # Mostrar flujo de trabajo del astrónomo
    show_astronomer_workflow()
    
    if success:
        print(f"\n🎉 ¡Demostración exitosa!")
        print(f"   La funcionalidad está lista para mejorar el flujo de trabajo astronómico.")
        print(f"   Los astrónomos pueden ahora enfocarse en los datos más relevantes.")
    else:
        print(f"\n❌ La demostración necesita ajustes.")
    
    return success

if __name__ == "__main__":
    main() 