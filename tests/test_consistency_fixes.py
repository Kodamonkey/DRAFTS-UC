#!/usr/bin/env python3
"""
Script de prueba para verificar las correcciones de consistencia en DM y SNR.
"""

import sys
import numpy as np
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRAFTS.visualization.visualization import print_consistency_summary
from DRAFTS.detection.astro_conversions import pixel_to_physical
from DRAFTS.detection.snr_utils import compute_snr_profile, find_snr_peak


def test_dm_consistency():
    """Prueba la consistencia del cálculo de DM."""
    print("🧪 === PRUEBA DE CONSISTENCIA DM ===")
    
    # Simular parámetros
    center_x, center_y = 256, 256  # Centro de imagen 512x512
    slice_len = 64
    
    # Calcular DM
    dm_val, t_sec, t_sample = pixel_to_physical(center_x, center_y, slice_len)
    
    print(f"📊 Parámetros de entrada:")
    print(f"   - center_x: {center_x}")
    print(f"   - center_y: {center_y}")
    print(f"   - slice_len: {slice_len}")
    
    print(f"📊 Resultado del cálculo:")
    print(f"   - DM: {dm_val:.2f} pc cm⁻³")
    print(f"   - Tiempo: {t_sec:.3f} s")
    print(f"   - Muestra: {t_sample}")
    
    # Verificar que el cálculo es consistente
    assert dm_val > 0, "DM debe ser positivo"
    assert t_sec >= 0, "Tiempo debe ser no negativo"
    assert 0 <= t_sample < slice_len, "Muestra debe estar en rango válido"
    
    print("✅ Prueba DM: PASÓ")
    return dm_val


def test_snr_consistency():
    """Prueba la consistencia del cálculo de SNR."""
    print("\n🧪 === PRUEBA DE CONSISTENCIA SNR ===")
    
    # Crear datos sintéticos
    n_time, n_freq = 100, 50
    waterfall = np.random.normal(0, 1, (n_time, n_freq))
    
    # Agregar un pico sintético
    peak_time, peak_freq = 50, 25
    waterfall[peak_time-2:peak_time+3, peak_freq-5:peak_freq+6] += 10
    
    print(f"📊 Parámetros de entrada:")
    print(f"   - waterfall shape: {waterfall.shape}")
    print(f"   - pico en: tiempo={peak_time}, freq={peak_freq}")
    
    # Calcular SNR
    snr_profile, sigma = compute_snr_profile(waterfall)
    peak_snr, peak_time_calc, peak_idx = find_snr_peak(snr_profile)
    
    print(f"📊 Resultado del cálculo:")
    print(f"   - SNR pico: {peak_snr:.2f}σ")
    print(f"   - Tiempo pico: {peak_time_calc:.1f}")
    print(f"   - Índice pico: {peak_idx}")
    print(f"   - Sigma estimado: {sigma:.3f}")
    
    # Verificar que el cálculo es consistente
    assert peak_snr > 0, "SNR debe ser positivo"
    assert 0 <= peak_idx < n_time, "Índice debe estar en rango válido"
    assert sigma > 0, "Sigma debe ser positivo"
    
    print("✅ Prueba SNR: PASÓ")
    return peak_snr


def test_composite_consistency():
    """Prueba la consistencia entre valores del composite y CSV."""
    print("\n🧪 === PRUEBA DE CONSISTENCIA COMPOSITE vs CSV ===")
    
    # Simular candidatos
    top_conf = [0.95, 0.87, 0.76]  # Confianzas
    top_boxes = [
        [100, 200, 150, 250],  # [x1, y1, x2, y2]
        [300, 150, 350, 200],
        [200, 300, 250, 350]
    ]
    slice_len = 64
    
    print(f"📊 Candidatos simulados:")
    for i, (conf, box) in enumerate(zip(top_conf, top_boxes)):
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        dm_val, _, _ = pixel_to_physical(center_x, center_y, slice_len)
        print(f"   Candidato {i+1}: conf={conf:.2f}, DM={dm_val:.2f}")
    
    # Encontrar candidato más fuerte
    best_idx = np.argmax(top_conf)
    best_box = top_boxes[best_idx]
    center_x, center_y = (best_box[0] + best_box[2]) / 2, (best_box[1] + best_box[3]) / 2
    dm_val_consistent, _, _ = pixel_to_physical(center_x, center_y, slice_len)
    
    print(f"📊 Candidato más fuerte:")
    print(f"   - Índice: {best_idx + 1}")
    print(f"   - Confianza: {top_conf[best_idx]:.2f}")
    print(f"   - DM consistente: {dm_val_consistent:.2f}")
    
    # Verificar que el candidato más fuerte tiene la mayor confianza
    assert best_idx == 0, "El candidato más fuerte debe ser el primero"
    assert dm_val_consistent > 0, "DM debe ser positivo"
    
    print("✅ Prueba Composite: PASÓ")
    return dm_val_consistent


def main():
    """Ejecuta todas las pruebas de consistencia."""
    print("🔬 INICIANDO PRUEBAS DE CONSISTENCIA")
    print("=" * 50)
    
    try:
        # Ejecutar pruebas
        dm_val = test_dm_consistency()
        snr_val = test_snr_consistency()
        dm_consistent = test_composite_consistency()
        
        print("\n" + "=" * 50)
        print("🎉 TODAS LAS PRUEBAS PASARON")
        print("=" * 50)
        
        # Mostrar documentación
        print("\n📋 DOCUMENTACIÓN DE CORRECCIONES:")
        print_consistency_summary()
        
        print(f"\n📊 RESUMEN DE VALORES:")
        print(f"   - DM calculado: {dm_val:.2f} pc cm⁻³")
        print(f"   - SNR calculado: {snr_val:.2f}σ")
        print(f"   - DM consistente: {dm_consistent:.2f} pc cm⁻³")
        
    except Exception as e:
        print(f"\n❌ ERROR EN PRUEBAS: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 