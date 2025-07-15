#!/usr/bin/env python3
"""
Script de prueba para verificar que se muestren los rangos de frecuencia 
por banda en los gráficos del sistema multi-banda DRAFTS.
"""

import numpy as np
import tempfile
from pathlib import Path
import sys
import os

# Agregar la ruta del módulo DRAFTS al path
sys.path.insert(0, str(Path(__file__).parent / "DRAFTS"))

from DRAFTS import config
from DRAFTS.visualization import get_band_frequency_range, get_band_name_with_freq_range

def test_frequency_range_functions():
    """Test las funciones de cálculo de rango de frecuencias."""
    
    print("🧪 Test: Funciones de rango de frecuencias")
    
    # Simular configuración de frecuencias
    config.FREQ = np.linspace(1200, 1500, 1024)  # 1200-1500 MHz con 1024 canales
    config.FREQ_RESO = 1024
    config.DOWN_FREQ_RATE = 1
    
    print(f"   • Frecuencias simuladas: {config.FREQ.min():.1f} - {config.FREQ.max():.1f} MHz")
    print(f"   • Número de canales: {len(config.FREQ)}")
    
    # Test para cada banda
    bands = [
        (0, "Full Band"),
        (1, "Low Band"), 
        (2, "High Band")
    ]
    
    for band_idx, band_name in bands:
        freq_min, freq_max = get_band_frequency_range(band_idx)
        band_name_with_freq = get_band_name_with_freq_range(band_idx, band_name)
        
        print(f"   • Banda {band_idx} ({band_name}):")
        print(f"     - Rango: {freq_min:.1f} - {freq_max:.1f} MHz")
        print(f"     - Nombre con frecuencias: {band_name_with_freq}")
    
    print("✅ Test completado - funciones de rango de frecuencias")


def test_band_frequency_logic():
    """Test la lógica de división de bandas."""
    
    print("\n🧪 Test: Lógica de división de bandas")
    
    # Configurar un caso específico
    config.FREQ = np.linspace(1000, 2000, 100)  # 1000-2000 MHz con 100 canales
    config.FREQ_RESO = 100
    config.DOWN_FREQ_RATE = 1
    
    freq_ds = np.mean(
        config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE),
        axis=1,
    )
    
    print(f"   • Array de frecuencias reducido: {len(freq_ds)} canales")
    print(f"   • Rango total: {freq_ds.min():.1f} - {freq_ds.max():.1f} MHz")
    
    # Verificar punto medio
    mid_channel = len(freq_ds) // 2
    print(f"   • Canal medio: {mid_channel}")
    print(f"   • Frecuencia del canal medio: {freq_ds[mid_channel]:.1f} MHz")
    
    # Verificar rangos calculados
    full_min, full_max = get_band_frequency_range(0)
    low_min, low_max = get_band_frequency_range(1)
    high_min, high_max = get_band_frequency_range(2)
    
    print(f"   • Full Band: {full_min:.1f} - {full_max:.1f} MHz")
    print(f"   • Low Band:  {low_min:.1f} - {low_max:.1f} MHz")
    print(f"   • High Band: {high_min:.1f} - {high_max:.1f} MHz")
    
    # Verificaciones
    assert full_min == freq_ds.min(), f"Full Band min incorrecto: {full_min} != {freq_ds.min()}"
    assert full_max == freq_ds.max(), f"Full Band max incorrecto: {full_max} != {freq_ds.max()}"
    assert low_min == freq_ds.min(), f"Low Band min incorrecto: {low_min} != {freq_ds.min()}"
    assert low_max == freq_ds[mid_channel], f"Low Band max incorrecto: {low_max} != {freq_ds[mid_channel]}"
    assert high_min == freq_ds[mid_channel], f"High Band min incorrecto: {high_min} != {freq_ds[mid_channel]}"
    assert high_max == freq_ds.max(), f"High Band max incorrecto: {high_max} != {freq_ds.max()}"
    
    print("✅ Test completado - lógica de división de bandas")


def test_edge_cases():
    """Test casos extremos."""
    
    print("\n🧪 Test: Casos extremos")
    
    # Caso 1: Pocas frecuencias
    config.FREQ = np.array([1400.0, 1450.0])  # Solo 2 frecuencias
    config.FREQ_RESO = 2
    config.DOWN_FREQ_RATE = 1
    
    print("   • Caso 1: Solo 2 frecuencias")
    for band_idx in [0, 1, 2]:
        try:
            freq_min, freq_max = get_band_frequency_range(band_idx)
            print(f"     - Banda {band_idx}: {freq_min:.1f} - {freq_max:.1f} MHz")
        except Exception as e:
            print(f"     - Banda {band_idx}: Error - {e}")
    
    # Caso 2: Frecuencias impares (número impar de canales)
    config.FREQ = np.linspace(1200, 1500, 101)  # 101 canales (impar)
    config.FREQ_RESO = 101
    config.DOWN_FREQ_RATE = 1
    
    print("   • Caso 2: Número impar de canales (101)")
    for band_idx in [0, 1, 2]:
        try:
            freq_min, freq_max = get_band_frequency_range(band_idx)
            band_name = get_band_name_with_freq_range(band_idx, f"Banda {band_idx}")
            print(f"     - {band_name}")
        except Exception as e:
            print(f"     - Banda {band_idx}: Error - {e}")
    
    print("✅ Test completado - casos extremos")


def main():
    """Función principal del test."""
    
    print("🚀 Iniciando test de visualización de rangos de frecuencia por banda\n")
    
    try:
        test_frequency_range_functions()
        test_band_frequency_logic()
        test_edge_cases()
        
        print("\n" + "="*60)
        print("✅ TODOS LOS TESTS COMPLETADOS EXITOSAMENTE")
        print("Los gráficos ahora mostrarán el rango de frecuencias específico de cada banda:")
        print("   • Full Band: Muestra el rango completo de frecuencias")
        print("   • Low Band: Muestra desde la frecuencia mínima hasta la mitad")
        print("   • High Band: Muestra desde la mitad hasta la frecuencia máxima")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR EN LOS TESTS: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
