#!/usr/bin/env python3
"""
Demostración visual de cómo se muestran los rangos de frecuencia 
por banda en los gráficos del sistema DRAFTS.
"""

import numpy as np
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
import sys

# Agregar la ruta del módulo DRAFTS al path
sys.path.insert(0, str(Path(__file__).parent / "DRAFTS"))

from DRAFTS import config
from DRAFTS.visualization import get_band_frequency_range, get_band_name_with_freq_range

def demo_frequency_ranges():
    """Demonstrar los rangos de frecuencia para diferentes bandas."""
    
    print("🎯 Demostración: Rangos de frecuencia por banda\n")
    
    # Simular configuración típica de un telescopio (ej: FAST)
    config.FREQ = np.linspace(1050, 1450, 400)  # 1050-1450 MHz con 400 canales
    config.FREQ_RESO = 400
    config.DOWN_FREQ_RATE = 2  # Factor de reducción típico
    
    print(f"📡 Configuración simulada (tipo FAST):")
    print(f"   • Rango original: {config.FREQ.min():.1f} - {config.FREQ.max():.1f} MHz")
    print(f"   • Canales originales: {config.FREQ_RESO}")
    print(f"   • Factor de reducción: {config.DOWN_FREQ_RATE}")
    
    # Calcular frecuencias reducidas (como se hace en el pipeline)
    freq_ds = np.mean(
        config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE),
        axis=1,
    )
    print(f"   • Canales después de reducción: {len(freq_ds)}")
    print(f"   • Rango después de reducción: {freq_ds.min():.1f} - {freq_ds.max():.1f} MHz")
    
    print(f"\n📊 División en bandas:")
    
    # Información detallada para cada banda
    bands_info = [
        (0, "Full Band", "Toda la banda de observación"),
        (1, "Low Band", "Mitad inferior del espectro"),
        (2, "High Band", "Mitad superior del espectro")
    ]
    
    for band_idx, band_name, description in bands_info:
        freq_min, freq_max = get_band_frequency_range(band_idx)
        band_name_with_freq = get_band_name_with_freq_range(band_idx, band_name)
        bandwidth = freq_max - freq_min
        
        print(f"   • {band_name_with_freq}")
        print(f"     - Descripción: {description}")
        print(f"     - Ancho de banda: {bandwidth:.1f} MHz")
        
        if band_idx == 1:  # Low Band
            mid_channel = len(freq_ds) // 2
            print(f"     - Canales: 0 a {mid_channel} (de {len(freq_ds)} total)")
        elif band_idx == 2:  # High Band
            mid_channel = len(freq_ds) // 2
            print(f"     - Canales: {mid_channel} a {len(freq_ds)} (de {len(freq_ds)} total)")
        else:  # Full Band
            print(f"     - Canales: 0 a {len(freq_ds)} (todos)")
        print()


def demo_different_telescopes():
    """Demostrar cómo se ven diferentes configuraciones de telescopios."""
    
    print("🔭 Ejemplos para diferentes telescopios:\n")
    
    telescope_configs = [
        {
            "name": "FAST (China)",
            "freq_range": (1050, 1450),  # MHz
            "channels": 4096,
            "down_rate": 4
        },
        {
            "name": "Arecibo (histórico)", 
            "freq_range": (1200, 1500),  # MHz
            "channels": 2048,
            "down_rate": 2
        },
        {
            "name": "Green Bank Telescope",
            "freq_range": (1100, 1800),  # MHz  
            "channels": 1024,
            "down_rate": 1
        }
    ]
    
    for telescope in telescope_configs:
        print(f"📡 {telescope['name']}:")
        
        # Configurar para este telescopio
        config.FREQ = np.linspace(telescope['freq_range'][0], telescope['freq_range'][1], telescope['channels'])
        config.FREQ_RESO = telescope['channels']
        config.DOWN_FREQ_RATE = telescope['down_rate']
        
        print(f"   • Rango: {config.FREQ.min():.0f} - {config.FREQ.max():.0f} MHz")
        print(f"   • Canales: {config.FREQ_RESO} → {config.FREQ_RESO // config.DOWN_FREQ_RATE}")
        
        # Mostrar bandas para este telescopio
        for band_idx, band_name in [(0, "Full"), (1, "Low"), (2, "High")]:
            band_name_with_freq = get_band_name_with_freq_range(band_idx, f"{band_name} Band")
            print(f"     - {band_name_with_freq}")
        print()


def demo_plot_titles():
    """Mostrar ejemplos de títulos de gráficos con los nuevos rangos."""
    
    print("📋 Ejemplos de títulos en los gráficos:\n")
    
    # Configuración ejemplo
    config.FREQ = np.linspace(1200, 1500, 1024)
    config.FREQ_RESO = 1024
    config.DOWN_FREQ_RATE = 2
    config.TIME_RESO = 8.192e-6  # 8.192 μs típico
    config.DOWN_TIME_RATE = 4
    
    fits_stem = "FRB20180301_0001"
    slice_idx = 5
    time_slice = 20
    
    print("🖼️  Títulos en gráficos de detección:")
    
    for band_idx, band_name in [(0, "Full Band"), (1, "Low Band"), (2, "High Band")]:
        band_name_with_freq = get_band_name_with_freq_range(band_idx, band_name)
        
        # Simular título como aparecería en save_detection_plot
        freq_min, freq_max = get_band_frequency_range(band_idx)
        freq_range = f"{freq_min:.0f}–{freq_max:.0f} MHz"
        time_resolution = config.TIME_RESO * config.DOWN_TIME_RATE * 1e6
        
        title = (
            f"{fits_stem} - {band_name_with_freq}\n"
            f"Slice {slice_idx + 1}/{time_slice} | "
            f"Time Resolution: {time_resolution:.1f} μs | "
            f"DM Range: 50–180 (auto) pc cm⁻³"
        )
        
        print(f"   • {band_name}:")
        print(f"     \"{title}\"")
        print()
    
    print("🖼️  Títulos en gráficos compuestos (Composite Summary):")
    
    for band_idx, band_name in [(0, "Full Band"), (1, "Low Band"), (2, "High Band")]:
        band_name_with_freq = get_band_name_with_freq_range(band_idx, band_name)
        
        title = f"Composite Summary: {fits_stem} - {band_name_with_freq} - Slice {slice_idx + 1}"
        
        print(f"   • {band_name}:")
        print(f"     \"{title}\"")
        print()
    
    print("🖼️  Títulos en gráficos de patches:")
    
    for band_idx, band_name in [(0, "Full Band"), (1, "Low Band"), (2, "High Band")]:
        band_name_with_freq = get_band_name_with_freq_range(band_idx, band_name)
        
        title = f"Candidate Patch - {band_name_with_freq}"
        
        print(f"   • {band_name}:")
        print(f"     \"{title}\"")
        print()


def main():
    """Función principal de la demostración."""
    
    print("🎨 DEMOSTRACIÓN: Visualización de rangos de frecuencia por banda")
    print("=" * 70)
    print()
    
    demo_frequency_ranges()
    print("\n" + "-" * 70)
    demo_different_telescopes()
    print("\n" + "-" * 70)
    demo_plot_titles()
    
    print("=" * 70)
    print("✅ IMPLEMENTACIÓN COMPLETADA")
    print()
    print("📋 Resumen de cambios implementados:")
    print("   • Los gráficos ahora muestran el rango específico de frecuencias de cada banda")
    print("   • Full Band: muestra el rango completo (ej: 1200-1500 MHz)")
    print("   • Low Band: muestra la mitad inferior (ej: 1200-1350 MHz)")
    print("   • High Band: muestra la mitad superior (ej: 1350-1500 MHz)")
    print("   • Los rangos se calculan dinámicamente según los metadatos del archivo")
    print("   • Compatible con cualquier configuración de telescopio")
    print()
    print("🔧 Archivos modificados:")
    print("   • DRAFTS/visualization.py - Nuevas funciones de cálculo de rangos")
    print("   • DRAFTS/image_utils.py - Títulos actualizados con rangos específicos")
    print("   • DRAFTS/pipeline.py - Integración del band_idx en las llamadas")
    print()
    print("🎯 Los gráficos generados ahora incluirán información clara sobre")
    print("   qué frecuencias específicas está procesando cada banda.")
    print("=" * 70)


if __name__ == "__main__":
    main()
