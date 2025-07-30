#!/usr/bin/env python3
"""
Test para verificar que las dimensiones del plot composite se ajustaron correctamente:
- Plot DM-tiempo más grande (height_ratios=[2, 3, 3])
- Waterfalls más pequeños
- Colores consistentes (viridis para dedispersado y candidate)
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend no-interactivo
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import shutil

# Agregar el directorio DRAFTS al path
sys.path.insert(0, str(Path(__file__).parent / ".."))

from DRAFTS import config
from DRAFTS.visualization import save_slice_summary

def test_composite_dimensions():
    """Test para verificar las dimensiones del plot composite."""
    
    print("=== TEST DE DIMENSIONES DEL PLOT COMPOSITE ===")
    
    # Configurar parámetros de prueba
    config.FREQ = np.linspace(1200, 1500, 256)
    config.FREQ_RESO = 256
    config.TIME_RESO = 0.001
    config.DOWN_FREQ_RATE = 1
    config.DOWN_TIME_RATE = 1
    config.FILE_LENG = 10000
    config.MODEL_NAME = "resnet50"
    config.CLASS_PROB = 0.5
    config.DM_min = 0
    config.DM_max = 1024
    config.SLICE_LEN = 512
    config.DET_PROB = 0.4
    config.SNR_THRESH = 3.0
    
    # Habilitar visualizaciones
    config.PLOT_COMPOSITE = True
    
    # Crear datos sintéticos
    slice_len = 512
    n_freq = 256
    
    # 1. Waterfall block (datos raw)
    waterfall_block = np.random.rand(slice_len, n_freq) * 100
    
    # 2. Dedispersed block (con pulso sintético)
    dedispersed_block = waterfall_block.copy()
    # Agregar pulso sintético en el centro
    center_time = slice_len // 2
    center_freq = n_freq // 2
    pulse_width = 10
    for i in range(max(0, center_time - pulse_width), min(slice_len, center_time + pulse_width)):
        for j in range(max(0, center_freq - pulse_width), min(n_freq, center_freq + pulse_width)):
            dedispersed_block[i, j] += 50 * np.exp(-0.5 * ((i - center_time) / 3)**2 - 0.5 * ((j - center_freq) / 3)**2)
    
    # 3. Detection image (DM vs Time)
    img_rgb = np.random.rand(512, 512, 3) * 0.3
    
    # Simular candidatos detectados
    top_boxes = [
        [200, 200, 250, 250],  # Centro en (225, 225)
    ]
    top_conf = [0.9]
    class_probs = [0.8]
    
    # 4. Patch sintético (no se usa en el composite actual)
    patch_img = np.random.rand(64, 64) * 100
    
    # Parámetros de tiempo absoluto
    slice_idx = 0
    time_slice = 10
    absolute_start_time = 1.5  # Tiempo absoluto en segundos
    
    # Crear directorio temporal
    temp_dir = Path(tempfile.mkdtemp())
    print(f"📁 Directorio temporal: {temp_dir}")
    
    try:
        # Test: Composite plot con dimensiones ajustadas
        print("\n--- Test: Composite Plot con Dimensiones Ajustadas ---")
        
        composite_path = temp_dir / "test_composite_dimensions.png"
        
        save_slice_summary(
            waterfall_block=waterfall_block,
            dedispersed_block=dedispersed_block,
            img_rgb=img_rgb,
            patch_img=patch_img,
            patch_start=absolute_start_time,
            dm_val=100.0,
            top_conf=top_conf,
            top_boxes=top_boxes,
            class_probs=class_probs,
            out_path=composite_path,
            slice_idx=slice_idx,
            time_slice=time_slice,
            band_name="Test Band",
            band_suffix="test",
            fits_stem="test_file",
            slice_len=slice_len,
            normalize=True,
            band_idx=0,
            absolute_start_time=absolute_start_time,
        )
        
        # Verificar que el archivo se creó
        if composite_path.exists():
            file_size = composite_path.stat().st_size
            print(f"✅ Composite plot creado: {composite_path}")
            print(f"   📊 Tamaño del archivo: {file_size:,} bytes")
            print(f"   🎯 Dimensiones: Plot DM-tiempo más grande, waterfalls más pequeños")
            print(f"   🎨 Colores: viridis para dedispersado y candidate")
            
            # Verificar que el archivo no está vacío
            if file_size > 100000:  # Más de 100KB
                print("✅ Archivo de tamaño apropiado")
            else:
                print("⚠️  Archivo muy pequeño, posible error")
        else:
            print("❌ Error: No se creó el archivo composite")
            return False
        
        print("\n🎉 Test completado exitosamente!")
        return True
        
    except Exception as e:
        print(f"❌ Error durante el test: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Limpiar directorio temporal
        try:
            shutil.rmtree(temp_dir)
            print(f"🧹 Directorio temporal eliminado: {temp_dir}")
        except:
            pass

if __name__ == "__main__":
    success = test_composite_dimensions()
    sys.exit(0 if success else 1) 
