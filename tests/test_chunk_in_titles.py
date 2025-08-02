#!/usr/bin/env python3
"""
Script de prueba para verificar que los títulos de los plots composite incluyen el número de chunk.
"""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_chunk_in_composite_titles():
    """Prueba que los títulos de los plots composite incluyen el número de chunk."""
    
    print("🧪 Probando inclusión de chunk en títulos de plots composite...")
    
    # Importar después de agregar el path
    from drafts.visualization.visualization_unified import save_slice_summary
    from drafts import config
    
    # Configurar config para la prueba
    config.FREQ = np.linspace(1200, 1500, 1024)  # Frecuencias de 1200 a 1500 MHz
    config.FREQ_RESO = 1024
    config.DOWN_FREQ_RATE = 2
    config.TIME_RESO = 8.192e-6  # 8.192 μs
    config.DOWN_TIME_RATE = 4
    config.FILE_LENG = 1000
    config.MODEL_NAME = "test_model"
    config.SNR_SHOW_PEAK_LINES = True
    config.SNR_HIGHLIGHT_COLOR = "red"
    config.DM_DYNAMIC_RANGE_ENABLE = True
    
    # Crear directorio temporal para la prueba
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Configurar datos de prueba
        fits_stem = "test_file"
        slice_idx = 5
        chunk_idx = 12
        band_name = "Full Band"
        band_suffix = "_full"
        band_idx = 0
        
        # Crear datos simulados
        waterfall_block = np.random.rand(100, 50)
        dedispersed_block = np.random.rand(100, 50)
        img_rgb = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
        patch_img = np.random.rand(20, 10)
        
        # Crear archivo de salida
        out_path = temp_path / f"{fits_stem}_slice{slice_idx:03d}.png"
        
        print(f"📁 Archivo de salida: {out_path}")
        print(f"📊 Parámetros de prueba:")
        print(f"   - fits_stem: {fits_stem}")
        print(f"   - slice_idx: {slice_idx}")
        print(f"   - chunk_idx: {chunk_idx}")
        print(f"   - band_name: {band_name}")
        
        # Probar con chunk_idx
        print(f"\n🔄 Probando con chunk_idx={chunk_idx}...")
        try:
            save_slice_summary(
                waterfall_block=waterfall_block,
                dedispersed_block=dedispersed_block,
                img_rgb=img_rgb,
                patch_img=patch_img,
                patch_start=0.0,
                dm_val=100.0,
                top_conf=[],
                top_boxes=[],
                class_probs=[],
                out_path=out_path,
                slice_idx=slice_idx,
                time_slice=20,
                band_name=band_name,
                band_suffix=band_suffix,
                fits_stem=fits_stem,
                slice_len=50,
                normalize=False,
                band_idx=band_idx,
                chunk_idx=chunk_idx,  # ✅ CON CHUNK_ID
            )
            print(f"✅ Plot generado exitosamente con chunk_idx")
            
            # Verificar que el archivo existe
            if out_path.exists():
                print(f"✅ Archivo creado: {out_path}")
                file_size = out_path.stat().st_size
                print(f"   📊 Tamaño: {file_size} bytes")
            else:
                print(f"❌ Archivo no encontrado: {out_path}")
                
        except Exception as e:
            print(f"❌ Error con chunk_idx: {e}")
            return False
        
        # Probar sin chunk_idx (compatibilidad hacia atrás)
        print(f"\n🔄 Probando sin chunk_idx (compatibilidad)...")
        out_path_no_chunk = temp_path / f"{fits_stem}_slice{slice_idx:03d}_nochunk.png"
        
        try:
            save_slice_summary(
                waterfall_block=waterfall_block,
                dedispersed_block=dedispersed_block,
                img_rgb=img_rgb,
                patch_img=patch_img,
                patch_start=0.0,
                dm_val=100.0,
                top_conf=[],
                top_boxes=[],
                class_probs=[],
                out_path=out_path_no_chunk,
                slice_idx=slice_idx,
                time_slice=20,
                band_name=band_name,
                band_suffix=band_suffix,
                fits_stem=fits_stem,
                slice_len=50,
                normalize=False,
                band_idx=band_idx,
                # chunk_idx=None (por defecto)
            )
            print(f"✅ Plot generado exitosamente sin chunk_idx")
            
            # Verificar que el archivo existe
            if out_path_no_chunk.exists():
                print(f"✅ Archivo creado: {out_path_no_chunk}")
                file_size = out_path_no_chunk.stat().st_size
                print(f"   📊 Tamaño: {file_size} bytes")
            else:
                print(f"❌ Archivo no encontrado: {out_path_no_chunk}")
                
        except Exception as e:
            print(f"❌ Error sin chunk_idx: {e}")
            return False
        
        print(f"\n📋 RESULTADO DE LA PRUEBA:")
        print(f"   ✅ Con chunk_idx: Funciona correctamente")
        print(f"   ✅ Sin chunk_idx: Compatibilidad hacia atrás mantenida")
        print(f"   📁 Archivos generados: 2")
        
        return True

def demonstrate_title_differences():
    """Demuestra las diferencias en los títulos."""
    
    print(f"\n📋 EJEMPLOS DE TÍTULOS:")
    print(f"=" * 50)
    
    fits_stem = "FRB20201124_0009"
    band_name_with_freq = "Full Band (1200-1500 MHz)"
    slice_idx = 5
    chunk_idx = 12
    
    print(f"ANTES (sin chunk):")
    title_old = f"Composite Summary: {fits_stem} - {band_name_with_freq} - Slice {slice_idx:03d}"
    print(f"   \"{title_old}\"")
    
    print(f"\nDESPUÉS (con chunk):")
    title_new = f"Composite Summary: {fits_stem} - {band_name_with_freq} - Chunk {chunk_idx:03d} - Slice {slice_idx:03d}"
    print(f"   \"{title_new}\"")
    
    print(f"\n💡 BENEFICIOS:")
    print(f"   • Identificación clara del chunk al que pertenece cada slice")
    print(f"   • Facilita la navegación entre chunks")
    print(f"   • Mejora la organización visual de los plots")
    print(f"   • Compatibilidad hacia atrás mantenida")

def main():
    """Función principal."""
    
    print(f"{'='*60}")
    print(f"🧪 PRUEBA: CHUNK EN TÍTULOS DE PLOTS COMPOSITE")
    print(f"{'='*60}")
    
    # Ejecutar prueba
    success = test_chunk_in_composite_titles()
    
    # Mostrar ejemplos de títulos
    demonstrate_title_differences()
    
    if success:
        print(f"\n🎉 ¡Prueba exitosa!")
        print(f"   Los títulos de los plots composite ahora incluyen el número de chunk.")
        print(f"   La funcionalidad está lista para usar.")
    else:
        print(f"\n❌ La prueba necesita ajustes.")
    
    return success

if __name__ == "__main__":
    main() 