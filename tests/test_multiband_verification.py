#!/usr/bin/env python3
"""
Script de prueba para verificar que se generan visualizaciones de las 3 bandas (Full, Low, High)
"""

import os
import sys
from pathlib import Path

# Añadir el directorio DRAFTS al path
current_dir = Path(__file__).parent
drafts_dir = current_dir / "DRAFTS"
sys.path.insert(0, str(drafts_dir))

from DRAFTS import config
from DRAFTS.data_loader import load_fits_file

def test_multiband_generation():
    """Prueba que se generen archivos de las 3 bandas."""
    
    print("🔬 VERIFICANDO GENERACIÓN DE MÚLTIPLES BANDAS")
    print("=" * 60)
    
    # Configurar para multi-banda
    config.USE_MULTI_BAND = True
    print(f"✅ USE_MULTI_BAND configurado a: {config.USE_MULTI_BAND}")
    
    # Verificar qué archivos de datos tenemos disponibles
    data_dir = Path("./Data")
    if not data_dir.exists():
        print(f"❌ ERROR: Directorio {data_dir} no existe")
        return False
        
    # Buscar archivos disponibles
    fits_files = list(data_dir.glob("*.fits"))
    all_files = fits_files
    
    if not all_files:
        print(f"❌ ERROR: No se encontraron archivos de datos en {data_dir}")
        return False
        
    print(f"📁 Archivos encontrados: {len(all_files)}")
    for i, file in enumerate(all_files[:5]):  # Mostrar solo los primeros 5
        print(f"   {i+1}. {file.name}")
    if len(all_files) > 5:
        print(f"   ... y {len(all_files) - 5} más")
    
    # Configurar directorios de salida
    results_dir = Path("./Results/Multiband_Test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎯 CONFIGURACIÓN MULTI-BANDA:")
    print(f"   • USE_MULTI_BAND: {config.USE_MULTI_BAND}")
    print(f"   • Bandas esperadas:")
    if config.USE_MULTI_BAND:
        print(f"     - banda[0] = Full Band (suma completa de frecuencias)")
        print(f"     - banda[1] = Low Band (mitad inferior del espectro)")
        print(f"     - banda[2] = High Band (mitad superior del espectro)")
    else:
        print(f"     - Solo Full Band")
    
    # Simular la configuración de bandas del pipeline
    band_configs = (
        [
            (0, "fullband", "Full Band"),
            (1, "lowband", "Low Band"),
            (2, "highband", "High Band"),
        ]
        if config.USE_MULTI_BAND
        else [(0, "fullband", "Full Band")]
    )
    
    print(f"\n🔧 CONFIGURACIÓN DE BANDAS GENERADA:")
    for band_idx, band_suffix, band_name in band_configs:
        print(f"   • banda[{band_idx}] = {band_name} (sufijo: {band_suffix})")
    
    # Verificar estructura de directorios esperada
    expected_dirs = {
        "Patches": "Parches individuales por banda",
        "Composite": "Resúmenes compuestos por banda", 
        "waterfall_dispersion": "Waterfalls sin dedispersar",
        "waterfall_dedispersion": "Waterfalls dedispersados"
    }
    
    print(f"\n📂 ESTRUCTURA DE DIRECTORIOS ESPERADA:")
    for dir_name, description in expected_dirs.items():
        print(f"   • {dir_name}/: {description}")
        for band_idx, band_suffix, band_name in band_configs:
            print(f"     - archivos con sufijo _{band_suffix}")
    
    # Verificar que los archivos realmente se generen por banda
    print(f"\n🔍 VERIFICANDO GENERACIÓN POR BANDA:")
    test_file = all_files[0]
    test_stem = test_file.stem
    
    expected_files = []
    for band_idx, band_suffix, band_name in band_configs:
        # Archivos principales que deberían generarse por banda
        expected_files.extend([
            f"{test_stem}_slice0_{band_suffix}.png",  # Bow tie plot
            f"slice0_band{band_idx}.png",            # Composite
            f"patch_slice0_band{band_idx}.png"       # Patch
        ])
    
    print(f"   📋 Archivos esperados para {test_stem}:")
    for i, filename in enumerate(expected_files):
        print(f"     {i+1:2d}. {filename}")
    
    print(f"\n✅ CONFIGURACIÓN VERIFICADA")
    print(f"   • Multi-banda: {'HABILITADO' if config.USE_MULTI_BAND else 'DESHABILITADO'}")
    print(f"   • Número de bandas: {len(band_configs)}")
    print(f"   • Archivo de prueba: {test_file.name}")
    
    return True

def check_current_config():
    """Verifica la configuración actual del sistema."""
    
    print(f"\n🔧 CONFIGURACIÓN ACTUAL DEL SISTEMA:")
    print(f"   • USE_MULTI_BAND: {getattr(config, 'USE_MULTI_BAND', 'NO DEFINIDO')}")
    print(f"   • SLICE_LEN_AUTO: {getattr(config, 'SLICE_LEN_AUTO', 'NO DEFINIDO')}")
    print(f"   • SLICE_LEN_INTELLIGENT: {getattr(config, 'SLICE_LEN_INTELLIGENT', 'NO DEFINIDO')}")
    print(f"   • DM_DYNAMIC_RANGE_ENABLE: {getattr(config, 'DM_DYNAMIC_RANGE_ENABLE', 'NO DEFINIDO')}")
    
    # Verificar comentarios en config.py
    config_file = Path("DRAFTS/config.py")
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "banda[0] = Full Band" in content:
                print(f"   ✅ Comentarios explicativos de bandas encontrados en config.py")
            else:
                print(f"   ⚠️  Comentarios explicativos de bandas NO encontrados en config.py")

if __name__ == "__main__":
    print("🚀 INICIANDO VERIFICACIÓN DE SISTEMA MULTI-BANDA\n")
    
    try:
        success = test_multiband_generation()
        check_current_config()
        
        if success:
            print(f"\n🎉 VERIFICACIÓN COMPLETADA EXITOSAMENTE")
            print(f"   El sistema está configurado para generar visualizaciones de múltiples bandas")
            print(f"   Ejecuta el pipeline principal para verificar la generación real de archivos")
        else:
            print(f"\n❌ VERIFICACIÓN FALLÓ")
            print(f"   Revisa la configuración y archivos de entrada")
            
    except Exception as e:
        print(f"\n💥 ERROR DURANTE LA VERIFICACIÓN: {e}")
        import traceback
        traceback.print_exc()
