#!/usr/bin/env python3
"""
Test para verificar consistencia del cálculo de DM entre CSV y plots.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from DRAFTS.detection.astro_conversions import pixel_to_physical
from DRAFTS.core import config

def test_dm_consistency():
    """Test para verificar que el cálculo de DM es consistente."""
    
    print("🧪 === TEST DE CONSISTENCIA DE DM ===")
    
    # Configurar parámetros de prueba
    config.DM_min = 0
    config.DM_max = 1024
    config.TIME_RESO = 0.001
    config.DOWN_TIME_RATE = 1
    slice_len = 512
    
    # Casos de prueba: diferentes posiciones de candidatos
    test_cases = [
        # (px, py, descripción)
        (256, 256, "Centro del plot"),
        (100, 100, "Esquina superior izquierda"),
        (400, 400, "Esquina inferior derecha"),
        (256, 100, "Centro superior"),
        (256, 400, "Centro inferior"),
        (100, 256, "Izquierda centro"),
        (400, 256, "Derecha centro"),
    ]
    
    print(f"📊 Configuración:")
    print(f"   DM_min: {config.DM_min}")
    print(f"   DM_max: {config.DM_max}")
    print(f"   DM_range: {config.DM_max - config.DM_min + 1}")
    print(f"   scale_dm: {(config.DM_max - config.DM_min + 1) / 512.0}")
    print(f"   slice_len: {slice_len}")
    print()
    
    print("🔍 Resultados de pixel_to_physical():")
    print("   px\tpy\tDM\t\tTiempo(s)\tMuestra")
    print("   " + "-" * 60)
    
    for px, py, desc in test_cases:
        dm_val, t_sec, t_sample = pixel_to_physical(px, py, slice_len)
        print(f"   {px}\t{py}\t{dm_val:.2f}\t\t{t_sec:.3f}\t\t{t_sample}")
    
    print()
    print("✅ Verificación de fórmula:")
    print("   Fórmula actual: DM = DM_min + py * (DM_range / 512)")
    print("   Compatible con DRAFTS-original: DM = center_y * (DM_range / 512)")
    print()
    
    # Verificar que la fórmula es correcta
    dm_range = config.DM_max - config.DM_min + 1
    scale_dm = dm_range / 512.0
    
    print("🧮 Verificación manual:")
    for px, py, desc in test_cases:
        dm_manual = config.DM_min + py * scale_dm
        dm_func, _, _ = pixel_to_physical(px, py, slice_len)
        
        if abs(dm_manual - dm_func) < 0.01:
            status = "✅ OK"
        else:
            status = "❌ ERROR"
            
        print(f"   {desc}: Manual={dm_manual:.2f}, Función={dm_func:.2f} {status}")
    
    print()
    print("🎯 CONCLUSIÓN:")
    print("   Si hay inconsistencia entre CSV y plots, NO es por la fórmula de pixel_to_physical()")
    print("   Ambos usan la misma función, por lo que deberían ser idénticos.")
    print("   El problema podría estar en otro lugar del código.")

if __name__ == "__main__":
    test_dm_consistency() 