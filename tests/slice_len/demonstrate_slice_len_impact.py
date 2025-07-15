#!/usr/bin/env python3
"""
Demostración práctica del impacto de SLICE_LEN en la conversión pixel-to-physical.

Este script muestra exactamente cómo SLICE_LEN afecta:
1. La resolución temporal por pixel
2. La conversión de coordenadas pixel a físicas
3. La precisión de localización temporal
4. El número de slices generados
"""

import sys
from pathlib import Path
import numpy as np

# Configurar path
sys.path.append(str(Path(__file__).parent))

from DRAFTS import config
from DRAFTS.astro_conversions import pixel_to_physical
from DRAFTS.pipeline import _slice_parameters

def demonstrate_pixel_conversion():
    """Demostrar conversión pixel-to-physical con diferentes SLICE_LEN."""
    
    print("🔬 === DEMOSTRACIÓN PRÁCTICA: SLICE_LEN Y CONVERSIÓN PIXEL-TO-PHYSICAL ===\n")
    
    # Parámetros de ejemplo (similares a tus datos reales)
    file_length = 2048      # muestras totales (ejemplo)
    time_reso = 0.000512    # resolución temporal típica para .fil (512 µs)
    dm_min = config.DM_min
    dm_max = config.DM_max
    
    print(f"📊 PARÁMETROS DEL ARCHIVO:")
    print(f"   • Longitud total: {file_length} muestras")
    print(f"   • Resolución temporal: {time_reso*1000:.3f} ms/muestra")
    print(f"   • Duración total: {file_length * time_reso:.3f} segundos")
    print(f"   • Rango DM: {dm_min} - {dm_max} pc cm⁻³")
    
    # Diferentes valores de SLICE_LEN para comparar
    slice_len_values = [32, 64, 128, 256]
    
    print(f"\n🎯 ANÁLISIS DETALLADO POR SLICE_LEN:")
    print("="*100)
    
    for slice_len in slice_len_values:
        print(f"\n🔸 SLICE_LEN = {slice_len}")
        print("-" * 50)
        
        # Cálculo del número de slices
        adjusted_slice_len, num_slices = _slice_parameters(file_length, slice_len)
        slice_duration = adjusted_slice_len * time_reso
        
        print(f"   📏 División temporal:")
        print(f"      • Slices generados: {num_slices}")
        print(f"      • Muestras por slice: {adjusted_slice_len}")
        print(f"      • Duración por slice: {slice_duration*1000:.3f} ms")
        
        # Cálculo de escalas para conversión
        dm_range = dm_max - dm_min + 1
        scale_dm = dm_range / 512.0
        scale_time = adjusted_slice_len / 512.0
        pixel_time_resolution = slice_duration / 512.0
        
        print(f"   🎯 Resolución y escalas:")
        print(f"      • Escala temporal: {scale_time:.3f} muestras/pixel")
        print(f"      • Escala DM: {scale_dm:.3f} pc cm⁻³/pixel")
        print(f"      • Resolución temporal/pixel: {pixel_time_resolution*1000:.6f} ms")
        
        # Ejemplos de conversión pixel-to-physical
        print(f"   📍 Ejemplos de conversión (pixel → físico):")
        
        test_pixels = [
            (128, 256, "Esquina inferior izquierda"),
            (256, 256, "Centro de la imagen"),
            (384, 256, "Esquina inferior derecha"),
            (256, 128, "Centro-arriba (DM bajo)"),
            (256, 384, "Centro-abajo (DM alto)")
        ]
        
        for px, py, description in test_pixels:
            dm_val, t_seconds, t_sample = pixel_to_physical(px, py, adjusted_slice_len)
            t_ms = t_seconds * 1000
            
            print(f"      • Pixel ({px:3d},{py:3d}) → DM={dm_val:6.1f}, t={t_ms:7.3f}ms ({description})")
        
        # Análisis de precisión
        total_coverage = num_slices * slice_duration
        temporal_precision = pixel_time_resolution
        
        print(f"   📊 Métricas de precisión:")
        print(f"      • Cobertura temporal total: {total_coverage*1000:.1f} ms")
        print(f"      • Precisión temporal mínima: {temporal_precision*1000:.6f} ms")
        print(f"      • Relación precisión/duración: 1:{slice_duration/temporal_precision:.0f}")
        
        # Evaluación para detección de FRBs
        print(f"   🎯 Evaluación para FRBs:")
        frb_durations = [1, 5, 10, 50, 100]  # ms
        
        for frb_dur in frb_durations:
            frb_dur_sec = frb_dur / 1000.0
            pixels_covered = frb_dur_sec / pixel_time_resolution
            
            if pixels_covered < 1:
                resolution_quality = "❌ Sub-pixel (puede perderse)"
            elif pixels_covered < 3:
                resolution_quality = "⚠️  Baja (1-2 pixels)"
            elif pixels_covered < 10:
                resolution_quality = "✅ Buena (3-9 pixels)"
            else:
                resolution_quality = "🎯 Excelente (>10 pixels)"
            
            print(f"      • FRB {frb_dur:3d}ms → {pixels_covered:5.1f} pixels: {resolution_quality}")

def analyze_slice_distribution():
    """Analizar cómo SLICE_LEN afecta la distribución de slices."""
    
    print(f"\n" + "="*100)
    print("📊 ANÁLISIS: DISTRIBUCIÓN DE SLICES Y OPORTUNIDADES DE DETECCIÓN")
    print("="*100)
    
    # Simular archivo real
    file_length = 2048  # muestras
    time_reso = 0.000512  # 512 µs
    
    slice_len_values = [16, 32, 64, 128, 256, 512]
    
    print(f"📁 Archivo simulado: {file_length} muestras ({file_length * time_reso:.3f}s)")
    print()
    print("SLICE_LEN | N_Slices | Dur/Slice(ms) | Resolución(µs) | Oportunidades | Contexto")
    print("-" * 85)
    
    for slice_len in slice_len_values:
        adjusted_slice_len, num_slices = _slice_parameters(file_length, slice_len)
        slice_duration_ms = adjusted_slice_len * time_reso * 1000
        pixel_resolution_us = (slice_duration_ms / 512) * 1000
        
        # Evaluar contexto y oportunidades
        if num_slices > 64:
            opportunities = "🔥 Muy altas"
            context = "❌ Muy limitado"
        elif num_slices > 32:
            opportunities = "✅ Altas"
            context = "⚠️  Limitado"
        elif num_slices > 16:
            opportunities = "✅ Buenas"
            context = "✅ Bueno"
        elif num_slices > 8:
            opportunities = "⚠️  Moderadas"
            context = "✅ Muy bueno"
        else:
            opportunities = "❌ Bajas"
            context = "🎯 Excelente"
        
        print(f"{slice_len:8d} | {num_slices:8d} | {slice_duration_ms:10.1f} | "
              f"{pixel_resolution_us:9.1f} | {opportunities:12s} | {context}")

def recommend_slice_len():
    """Proporcionar recomendaciones específicas basadas en análisis."""
    
    print(f"\n" + "="*100)
    print("💡 RECOMENDACIONES ESPECÍFICAS PARA TU PIPELINE")
    print("="*100)
    
    current_slice_len = config.SLICE_LEN
    print(f"🎯 Configuración actual: SLICE_LEN = {current_slice_len}")
    
    print(f"\n📋 ESCENARIOS DE USO:")
    
    scenarios = [
        {
            'name': 'Pulsos muy cortos (< 5ms)',
            'recommended': 32,
            'reason': 'Necesita alta resolución temporal',
            'trade_offs': 'Muchos slices, más tiempo de procesamiento'
        },
        {
            'name': 'FRBs típicos (5-50ms)',
            'recommended': 64,
            'reason': 'Balance óptimo resolución/contexto',
            'trade_offs': 'Configuración estándar equilibrada'
        },
        {
            'name': 'Señales dispersas largas (> 50ms)',
            'recommended': 128,
            'reason': 'Mejor contexto temporal',
            'trade_offs': 'Menos resolución, menos slices'
        },
        {
            'name': 'Análisis exploratorio',
            'recommended': 64,
            'reason': 'Versatilidad para diferentes señales',
            'trade_offs': 'Compromiso para búsqueda general'
        }
    ]
    
    for scenario in scenarios:
        current_marker = " 👈 ACTUAL" if scenario['recommended'] == current_slice_len else ""
        print(f"\n   🔸 {scenario['name']}:")
        print(f"      • Recomendado: SLICE_LEN = {scenario['recommended']}{current_marker}")
        print(f"      • Razón: {scenario['reason']}")
        print(f"      • Trade-offs: {scenario['trade_offs']}")
    
    print(f"\n🚀 ESTRATEGIA DE OPTIMIZACIÓN:")
    print(f"   1. Comenzar con tu valor actual ({current_slice_len})")
    print(f"   2. Si necesitas más resolución → probar {current_slice_len//2}")
    print(f"   3. Si necesitas más contexto → probar {current_slice_len*2}")
    print(f"   4. Comparar número y calidad de detecciones")
    print(f"   5. Ajustar según resultados específicos")
    
    print(f"\n⚙️  COMANDO PARA EXPERIMENTAR:")
    print(f"   python experiment_slice_len.py")

def main():
    """Función principal."""
    demonstrate_pixel_conversion()
    analyze_slice_distribution()
    recommend_slice_len()
    
    print(f"\n✅ Análisis completado")
    print(f"💡 Próximo paso: Ejecutar experiment_slice_len.py para probar diferentes valores")

if __name__ == "__main__":
    main()
