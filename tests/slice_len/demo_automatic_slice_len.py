#!/usr/bin/env python3
"""
Demostración del sistema automático de SLICE_LEN basado en metadatos.

Este script muestra cómo el sistema analiza automáticamente las características
del archivo y calcula el SLICE_LEN óptimo sin intervención del usuario.
"""

import sys
from pathlib import Path
import numpy as np

# Configurar path
sys.path.append(str(Path(__file__).parent))

from DRAFTS.auto_slice_len import SliceLenOptimizer

def demo_automatic_slice_len():
    """Demuestra el cálculo automático de SLICE_LEN."""
    
    print("🚀 === SISTEMA AUTOMÁTICO DE SLICE_LEN ===\n")
    print("Este sistema analiza automáticamente los metadatos del archivo")
    print("y calcula el SLICE_LEN óptimo para máxima resolución temporal.\n")
    
    # Crear optimizador
    optimizer = SliceLenOptimizer()
    
    # Casos de prueba con diferentes tipos de archivos
    test_cases = [
        {
            'name': 'FRB típico alta resolución',
            'time_reso': 0.001,      # 1ms
            'freq_reso': 1.0,        # 1 MHz
            'file_length': 2048,     # 2048 muestras
            'freq_range': (1200.0, 1500.0),  # 300 MHz de ancho de banda
            'dm_max': 1024,
            'filename': 'FRB20121102_highres.fil'
        },
        {
            'name': 'Pulsar rápido',
            'time_reso': 0.0005,     # 0.5ms
            'freq_reso': 0.5,        # 0.5 MHz
            'file_length': 4096,     # 4096 muestras
            'freq_range': (400.0, 800.0),    # 400 MHz
            'dm_max': 500,
            'filename': 'PSR_J1234+5678.fits'
        },
        {
            'name': 'FRB disperso banda L',
            'time_reso': 0.002,      # 2ms
            'freq_reso': 2.0,        # 2 MHz
            'file_length': 1024,     # 1024 muestras
            'freq_range': (1000.0, 2000.0),  # 1000 MHz
            'dm_max': 2000,
            'filename': 'FRB_dispersed_Lband.fil'
        },
        {
            'name': 'Observación larga banda S',
            'time_reso': 0.01,       # 10ms
            'freq_reso': 5.0,        # 5 MHz
            'file_length': 10000,    # 10000 muestras
            'freq_range': (2000.0, 4000.0),  # 2000 MHz
            'dm_max': 1500,
            'filename': 'long_observation_Sband.fits'
        },
        {
            'name': 'Archivo filterbank estándar',
            'time_reso': 0.000512,   # ~0.5ms típico
            'freq_reso': 1.0,
            'file_length': 8192,
            'freq_range': (1100.0, 1700.0),
            'dm_max': 1024,
            'filename': '3100_0001_00_8bit.fil'
        }
    ]
    
    print("📊 ANÁLISIS AUTOMÁTICO PARA DIFERENTES TIPOS DE ARCHIVOS:\n")
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"🔸 CASO {i}: {case['name']}")
        print(f"   Archivo: {case['filename']}")
        print(f"   Resolución temporal: {case['time_reso']*1000:.3f} ms")
        print(f"   Duración total: {case['file_length'] * case['time_reso']:.3f} s")
        print(f"   Ancho de banda: {case['freq_range'][1] - case['freq_range'][0]:.0f} MHz")
        print("   " + "-" * 50)
        
        # Calcular SLICE_LEN automático
        optimal_slice_len, analysis = optimizer.get_automatic_slice_len(
            time_reso=case['time_reso'],
            freq_reso=case['freq_reso'],
            file_length=case['file_length'],
            freq_range=case['freq_range'],
            dm_max=case['dm_max'],
            filename=case['filename']
        )
        
        opt_details = analysis['optimization']
        
        print(f"   🎯 SLICE_LEN óptimo: {optimal_slice_len}")
        print(f"   ⏰ Duración por slice: {opt_details['real_duration']*1000:.1f} ms")
        print(f"   🔢 Número de slices: {opt_details['n_slices']}")
        print(f"   📈 Tipo de archivo detectado: {analysis['content']['file_type']}")
        print(f"   🔍 Resolución temporal: {analysis['resolution']['resolution_score']}")
        
        # Factores aplicados
        factors = opt_details['factors']
        print(f"   📊 Factores de ajuste:")
        print(f"      • Dispersión: {factors['disp_factor']:.3f}")
        print(f"      • Resolución: {factors['resolution_factor']:.3f}")
        print(f"      • Contexto: {factors['context_factor']:.3f}")
        print(f"      • Eficiencia: {factors['efficiency_factor']:.3f}")
        
        results.append({
            'name': case['name'],
            'filename': case['filename'],
            'slice_len': optimal_slice_len,
            'duration_ms': opt_details['real_duration'] * 1000,
            'n_slices': opt_details['n_slices']
        })
        
        print()
    
    # Resumen comparativo
    print("📋 === RESUMEN COMPARATIVO ===\n")
    print("Archivo                          | SLICE_LEN | Duración | N_Slices | Características")
    print("-" * 95)
    
    for result in results:
        filename_short = result['filename'][:25] + "..." if len(result['filename']) > 28 else result['filename']
        print(f"{filename_short:32s} | {result['slice_len']:8d} | {result['duration_ms']:7.1f}ms | "
              f"{result['n_slices']:7d} | Automático")
    
    print("-" * 95)
    
    print(f"\n💡 BENEFICIOS DEL SISTEMA AUTOMÁTICO:")
    print(f"   ✅ Sin configuración manual necesaria")
    print(f"   ✅ Optimización específica por archivo")
    print(f"   ✅ Considera dispersión, resolución y contenido")
    print(f"   ✅ Se adapta automáticamente a cualquier tipo de datos")
    print(f"   ✅ Maximiza resolución temporal sin perder contexto")

def demo_comparison_with_manual():
    """Compara sistema automático vs configuración manual."""
    
    print(f"\n🔄 === COMPARACIÓN: AUTOMÁTICO vs MANUAL ===\n")
    
    # Archivo de ejemplo
    optimizer = SliceLenOptimizer()
    
    test_file = {
        'time_reso': 0.001,
        'freq_reso': 1.0,
        'file_length': 2048,
        'freq_range': (1200.0, 1500.0),
        'dm_max': 1024,
        'filename': 'example_FRB.fil'
    }
    
    # Calcular automáticamente
    auto_slice_len, auto_analysis = optimizer.get_automatic_slice_len(**test_file)
    auto_duration = auto_slice_len * test_file['time_reso']
    auto_slices = test_file['file_length'] // auto_slice_len
    
    # Valores manuales típicos
    manual_values = [32, 64, 128, 256]
    
    print("Método                  | SLICE_LEN | Duración | N_Slices | Resolución/pixel | Comentario")
    print("-" * 90)
    
    # Mostrar automático
    auto_res_per_pixel = auto_duration / 512 * 1000  # ms por pixel
    print(f"🚀 Automático            | {auto_slice_len:8d} | {auto_duration*1000:7.1f}ms | "
          f"{auto_slices:7d} | {auto_res_per_pixel:11.3f}ms | Óptimo calculado")
    
    # Mostrar manuales
    for manual_val in manual_values:
        manual_duration = manual_val * test_file['time_reso']
        manual_slices = test_file['file_length'] // manual_val
        manual_res_per_pixel = manual_duration / 512 * 1000
        
        if manual_val == auto_slice_len:
            comment = "✅ Igual al automático"
        elif abs(manual_val - auto_slice_len) <= 16:
            comment = "🟡 Cercano al óptimo"
        elif manual_val < auto_slice_len:
            comment = "🔍 Más resolución"
        else:
            comment = "🌊 Más contexto"
        
        print(f"📐 Manual (SLICE_LEN={manual_val:3d}) | {manual_val:8d} | {manual_duration*1000:7.1f}ms | "
              f"{manual_slices:7d} | {manual_res_per_pixel:11.3f}ms | {comment}")
    
    print("-" * 90)
    
    print(f"\n🎯 ANÁLISIS:")
    print(f"   • El sistema automático eligió SLICE_LEN = {auto_slice_len}")
    print(f"   • Basado en análisis de dispersión, resolución y tipo de archivo")
    print(f"   • Optimiza balance entre resolución temporal y contexto")
    print(f"   • No requiere intervención manual del usuario")

def demo_config_usage():
    """Muestra cómo usar el sistema en la configuración."""
    
    print(f"\n⚙️  === USO EN CONFIGURACIÓN ===\n")
    
    print("Para habilitar el sistema automático en tu pipeline:")
    print()
    print("📝 En config.py:")
    print("```python")
    print("# Habilitar sistema automático inteligente")
    print("SLICE_LEN_INTELLIGENT: bool = True")
    print("SLICE_LEN_OVERRIDE_MANUAL: bool = True")
    print()
    print("# Configuración de fallback (si el automático falla)")
    print("SLICE_LEN_AUTO: bool = True")
    print("SLICE_DURATION_SECONDS: float = 0.032")
    print()
    print("# Valor manual de emergencia")
    print("SLICE_LEN: int = 32")
    print("```")
    print()
    print("🚀 ¡Eso es todo! El sistema trabajará automáticamente.")
    print()
    print("📊 Jerarquía de decisión:")
    print("   1. 🎯 Sistema Inteligente (analiza metadatos del archivo)")
    print("   2. ⚙️  Sistema Dinámico (basado en SLICE_DURATION_SECONDS)")
    print("   3. 📐 Valor Manual (SLICE_LEN fijo)")
    print()
    print("💡 El sistema automáticamente:")
    print("   ✅ Lee metadatos del archivo (.fil, .fits)")
    print("   ✅ Analiza resolución temporal, frecuencias, dispersión")
    print("   ✅ Detecta tipo de señales esperadas")
    print("   ✅ Calcula SLICE_LEN óptimo para cada archivo")
    print("   ✅ Se adapta automáticamente sin configuración manual")

def main():
    """Función principal de demostración."""
    
    print("🌟 SISTEMA AUTOMÁTICO DE SLICE_LEN BASADO EN METADATOS")
    print("=" * 60)
    print("¡Calcula automáticamente el SLICE_LEN óptimo para cada archivo!")
    print("Sin configuración manual. Sin cálculos. ¡Completamente automático!")
    print()
    
    demo_automatic_slice_len()
    demo_comparison_with_manual()
    demo_config_usage()
    
    print(f"\n🎉 ¡DISFRUTA DEL SISTEMA COMPLETAMENTE AUTOMÁTICO!")
    print(f"   Tu pipeline ahora se optimiza automáticamente para cada archivo.")

if __name__ == "__main__":
    main()
