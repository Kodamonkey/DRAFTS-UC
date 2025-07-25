#!/usr/bin/env python3
"""
Test para verificar consistencia del cálculo de DM en el pipeline real.
Simula exactamente lo que pasa en CSV vs plots.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from DRAFTS.detection.astro_conversions import pixel_to_physical
from DRAFTS.core import config

def simulate_pipeline_dm_calculation():
    """Simula exactamente el cálculo de DM en el pipeline."""
    
    print("🧪 === SIMULACIÓN DE CÁLCULO DM EN PIPELINE ===")
    
    # Configurar parámetros como en el pipeline real
    config.DM_min = 0
    config.DM_max = 1024
    config.TIME_RESO = 0.001
    config.DOWN_TIME_RATE = 1
    slice_len = 512
    
    # Simular bounding boxes de detecciones (como vendrían del modelo)
    # Formato: (x1, y1, x2, y2)
    test_boxes = [
        (100, 150, 200, 250),  # Box 1
        (300, 200, 400, 300),  # Box 2
        (250, 100, 350, 200),  # Box 3
        (150, 300, 250, 400),  # Box 4
    ]
    
    print(f"📊 Configuración del pipeline:")
    print(f"   DM_min: {config.DM_min}")
    print(f"   DM_max: {config.DM_max}")
    print(f"   slice_len: {slice_len}")
    print()
    
    print("🔍 Cálculo de DM para cada candidato:")
    print("   Box\t\tCenter\t\tDM (CSV)\tDM (Plot)\t¿Consistente?")
    print("   " + "-" * 80)
    
    for i, box in enumerate(test_boxes):
        x1, y1, x2, y2 = box
        
        # Cálculo como en CSV (pipeline_utils.py línea 74)
        center_x_csv = (x1 + x2) / 2
        center_y_csv = (y1 + y2) / 2
        dm_csv, t_sec_csv, t_sample_csv = pixel_to_physical(center_x_csv, center_y_csv, slice_len)
        
        # Cálculo como en Plot (visualization.py línea 400)
        center_x_plot = (x1 + x2) / 2
        center_y_plot = (y1 + y2) / 2
        dm_plot, t_sec_plot, t_sample_plot = pixel_to_physical(center_x_plot, center_y_plot, slice_len)
        
        # Verificar consistencia
        dm_consistent = abs(dm_csv - dm_plot) < 0.01
        t_consistent = abs(t_sec_csv - t_sec_plot) < 0.001
        
        status = "✅ SÍ" if dm_consistent and t_consistent else "❌ NO"
        
        print(f"   {i+1}: {box}\t({center_x_csv:.1f},{center_y_csv:.1f})\t{dm_csv:.2f}\t\t{dm_plot:.2f}\t\t{status}")
        
        if not dm_consistent:
            print(f"      ⚠️  Diferencia DM: {abs(dm_csv - dm_plot):.6f}")
        if not t_consistent:
            print(f"      ⚠️  Diferencia tiempo: {abs(t_sec_csv - t_sec_plot):.6f}")
    
    print()
    print("🎯 ANÁLISIS:")
    print("   Si ves '❌ NO' arriba, hay un problema real en el código.")
    print("   Si ves '✅ SÍ' para todos, el problema está en otro lugar.")
    print()
    print("🔍 Posibles causas de inconsistencia real:")
    print("   1. Diferentes valores de slice_len entre CSV y plots")
    print("   2. Diferentes configuraciones de DM_min/DM_max")
    print("   3. Problema de redondeo en la visualización")
    print("   4. Error en la lectura de datos del CSV")

def test_different_slice_lengths():
    """Test con diferentes valores de slice_len."""
    
    print("\n🧪 === TEST CON DIFERENTES SLICE_LEN ===")
    
    config.DM_min = 0
    config.DM_max = 1024
    config.TIME_RESO = 0.001
    config.DOWN_TIME_RATE = 1
    
    box = (100, 150, 200, 250)
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    
    slice_lengths = [256, 512, 1024, 2048]
    
    print("   slice_len\tDM\t\tTiempo(s)")
    print("   " + "-" * 30)
    
    for slice_len in slice_lengths:
        dm_val, t_sec, t_sample = pixel_to_physical(center_x, center_y, slice_len)
        print(f"   {slice_len}\t\t{dm_val:.2f}\t\t{t_sec:.3f}")
    
    print()
    print("💡 NOTA: El DM NO debería cambiar con slice_len, solo el tiempo.")

if __name__ == "__main__":
    simulate_pipeline_dm_calculation()
    test_different_slice_lengths() 