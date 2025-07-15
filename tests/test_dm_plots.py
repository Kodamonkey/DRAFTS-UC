#!/usr/bin/env python3
"""
Test específico para verificar que los plots de detección y composite 
usen correctamente el rango DM dinámico.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Agregar el directorio DRAFTS al path
sys.path.insert(0, str(Path(__file__).parent / "DRAFTS"))

from DRAFTS import config
from DRAFTS.image_utils import save_detection_plot
from DRAFTS.visualization import save_slice_summary

def create_test_plots():
    """Crear plots de prueba con rangos DM dinámicos."""
    
    print("=== Test de Plots con Rango DM Dinámico ===")
    
    # Configurar parámetros de prueba
    config.FREQ = np.linspace(1200, 1500, 256)
    config.FREQ_RESO = 256
    config.TIME_RESO = 0.001
    config.DOWN_FREQ_RATE = 1
    config.DOWN_TIME_RATE = 1
    config.FILE_LENG = 10000
    config.MODEL_NAME = "resnet50"
    config.CLASS_PROB = 0.5
    
    # Crear imagen sintética de detección (512x512)
    img_rgb = np.random.rand(512, 512, 3) * 0.3
    
    # Simular candidatos detectados
    # Candidato 1: DM bajo (~200 pc cm⁻³) 
    top_boxes = [
        [200, 200, 250, 250],  # Centro en (225, 225) -> DM ~200
        [100, 150, 140, 190],  # Centro en (120, 170) -> DM ~150
    ]
    
    top_conf = [0.9, 0.7]
    class_probs = [0.8, 0.6]
    
    slice_len = 32
    slice_idx = 0
    time_slice = 10
    
    # Test 1: Plot de detección con DM dinámico
    print("\n--- Test 1: Plot de Detección ---")
    out_path = Path("test_dm_dynamic_detection.png")
    
    try:
        save_detection_plot(
            img_rgb=img_rgb,
            top_conf=top_conf,
            top_boxes=top_boxes,
            class_probs=class_probs,
            out_img_path=out_path,
            slice_idx=slice_idx,
            time_slice=time_slice,
            band_name="Full Band",
            band_suffix="fullband",
            det_prob=config.DET_PROB,
            fits_stem="test_frb",
            slice_len=slice_len
        )
        print(f"✅ Plot de detección generado: {out_path}")
        
    except Exception as e:
        print(f"❌ Error generando plot de detección: {e}")
    
    # Test 2: Plot composite con DM dinámico
    print("\n--- Test 2: Plot Composite ---")
    
    # Crear datos sintéticos para waterfall y dedispersión
    waterfall_block = np.random.rand(1000, 256) * 100
    dedispersed_block = np.random.rand(1000, 256) * 150
    
    out_path_composite = Path("test_dm_dynamic_composite.png")
    
    try:
        save_slice_summary(
            waterfall_block=waterfall_block,
            dedispersed_block=dedispersed_block,
            img_rgb=img_rgb,
            patch_img=img_rgb[:64, :64],  # Patch sintético
            patch_start=0.0,
            dm_val=175.0,  # DM de dedispersión
            top_conf=top_conf,
            top_boxes=top_boxes,
            class_probs=class_probs,
            out_path=out_path_composite,
            slice_idx=slice_idx,
            time_slice=time_slice,
            band_name="Full Band",
            band_suffix="fullband",
            fits_stem="test_frb",
            slice_len=slice_len,
            normalize=True
        )
        print(f"✅ Plot composite generado: {out_path_composite}")
        
    except Exception as e:
        print(f"❌ Error generando plot composite: {e}")
    
    # Test 3: Comparar con rango fijo (deshabilitar DM dinámico)
    print("\n--- Test 3: Comparación con Rango Fijo ---")
    
    # Deshabilitar DM dinámico temporalmente
    original_enable = config.DM_DYNAMIC_RANGE_ENABLE
    config.DM_DYNAMIC_RANGE_ENABLE = False
    
    out_path_fixed = Path("test_dm_fixed_detection.png")
    
    try:
        save_detection_plot(
            img_rgb=img_rgb,
            top_conf=top_conf,
            top_boxes=top_boxes,
            class_probs=class_probs,
            out_img_path=out_path_fixed,
            slice_idx=slice_idx,
            time_slice=time_slice,
            band_name="Full Band",
            band_suffix="fullband",
            det_prob=config.DET_PROB,
            fits_stem="test_frb_fixed",
            slice_len=slice_len
        )
        print(f"✅ Plot con rango fijo generado: {out_path_fixed}")
        
    except Exception as e:
        print(f"❌ Error generando plot con rango fijo: {e}")
    finally:
        # Restaurar configuración original
        config.DM_DYNAMIC_RANGE_ENABLE = original_enable
    
    print(f"\n=== Resumen ===")
    print(f"DM_DYNAMIC_RANGE_ENABLE: {config.DM_DYNAMIC_RANGE_ENABLE}")
    print(f"DM_RANGE_FACTOR: {config.DM_RANGE_FACTOR}")
    print(f"DM_PLOT_MARGIN_FACTOR: {config.DM_PLOT_MARGIN_FACTOR}")
    print(f"Archivos generados:")
    
    for path in [out_path, out_path_composite, out_path_fixed]:
        if path.exists():
            print(f"  ✅ {path}")
        else:
            print(f"  ❌ {path} (no generado)")

if __name__ == "__main__":
    create_test_plots()
