#!/usr/bin/env python3
"""
Demostración de la funcionalidad dinámica de SLICE_LEN.

Este script muestra cómo usar la nueva configuración basada en duración temporal
para SLICE_LEN, que es más intuitiva para el usuario.
"""

import sys
from pathlib import Path

# Configurar path
sys.path.append(str(Path(__file__).parent))

from DRAFTS import config
from DRAFTS.slice_len_utils import (
    calculate_optimal_slice_len,
    get_dynamic_slice_len,
    suggest_slice_duration_for_signal_type,
    print_slice_len_analysis
)

def demonstrate_dynamic_slice_len():
    """Demuestra la funcionalidad dinámica de SLICE_LEN."""
    
    print("🚀 === DEMOSTRACIÓN SLICE_LEN DINÁMICO ===\n")
    
    # Simular diferentes parámetros de archivo
    example_files = [
        {"name": "Archivo típico filterbank", "time_reso": 0.001, "down_time_rate": 1},
        {"name": "Archivo alta resolución", "time_reso": 0.0005, "down_time_rate": 1},
        {"name": "Archivo decimado", "time_reso": 0.002, "down_time_rate": 2},
    ]
    
    # Diferentes duraciones objetivo
    target_durations = [0.016, 0.032, 0.064, 0.128]  # 16ms, 32ms, 64ms, 128ms
    
    print("📊 CÁLCULO AUTOMÁTICO DE SLICE_LEN PARA DIFERENTES CASOS:\n")
    
    for file_info in example_files:
        print(f"🔸 {file_info['name']}:")
        print(f"   TIME_RESO: {file_info['time_reso']:.6f} s")
        print(f"   DOWN_TIME_RATE: {file_info['down_time_rate']}")
        print("   " + "-" * 50)
        
        for target_duration in target_durations:
            slice_len, actual_duration, explanation = calculate_optimal_slice_len(
                time_reso=file_info['time_reso'],
                down_time_rate=file_info['down_time_rate'],
                target_duration_seconds=target_duration
            )
            
            print(f"   • Objetivo: {target_duration*1000:4.0f}ms → SLICE_LEN: {slice_len:3d} → "
                  f"Real: {actual_duration*1000:5.1f}ms ({explanation})")
        
        print()

def demonstrate_signal_type_suggestions():
    """Demuestra las sugerencias por tipo de señal."""
    
    print("🎯 === SUGERENCIAS POR TIPO DE SEÑAL ===\n")
    
    signal_types = {
        'short': 'Pulsos muy cortos (< 20ms)',
        'medium': 'FRBs típicos (20-100ms)',
        'long': 'Señales largas (100-500ms)',
        'dispersed': 'Señales muy dispersas (> 500ms)',
        'general': 'Configuración general balanceada'
    }
    
    print("💡 CONFIGURACIONES RECOMENDADAS:")
    print("   Tipo de Señal                    | Duración | Descripción")
    print("   " + "-" * 65)
    
    for signal_type, description in signal_types.items():
        suggested_duration = suggest_slice_duration_for_signal_type(signal_type)
        print(f"   {description:32s} | {suggested_duration*1000:6.0f}ms | "
              f"SLICE_DURATION_SECONDS = {suggested_duration:.3f}")
    
    print("\n📝 CÓMO USAR:")
    print("   1. Identifica el tipo de señal que buscas")
    print("   2. Modifica SLICE_DURATION_SECONDS en config.py")
    print("   3. Asegúrate de que SLICE_LEN_AUTO = True")
    print("   4. Ejecuta el pipeline normalmente")

def demonstrate_configuration_examples():
    """Demuestra ejemplos de configuración."""
    
    print("\n⚙️  === EJEMPLOS DE CONFIGURACIÓN ===\n")
    
    examples = [
        {
            "name": "Búsqueda de pulsos cortos",
            "config": {
                "SLICE_DURATION_SECONDS": 0.016,
                "SLICE_LEN_AUTO": True,
                "SLICE_LEN_MIN": 8,
                "SLICE_LEN_MAX": 64
            },
            "description": "Ideal para detectar pulsos muy cortos y estrechos"
        },
        {
            "name": "FRBs típicos (recomendado)",
            "config": {
                "SLICE_DURATION_SECONDS": 0.032,
                "SLICE_LEN_AUTO": True,
                "SLICE_LEN_MIN": 16,
                "SLICE_LEN_MAX": 512
            },
            "description": "Configuración balanceada para la mayoría de FRBs"
        },
        {
            "name": "Señales largas y dispersas",
            "config": {
                "SLICE_DURATION_SECONDS": 0.128,
                "SLICE_LEN_AUTO": True,
                "SLICE_LEN_MIN": 32,
                "SLICE_LEN_MAX": 1024
            },
            "description": "Para señales con alta dispersión temporal"
        },
        {
            "name": "Configuración manual (tradicional)",
            "config": {
                "SLICE_LEN_AUTO": False,
                "SLICE_LEN": 64
            },
            "description": "Usa valor fijo como antes"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"🔹 EJEMPLO {i}: {example['name']}")
        print(f"   Descripción: {example['description']}")
        print(f"   Configuración en config.py:")
        
        for key, value in example['config'].items():
            if isinstance(value, bool):
                print(f"   {key}: bool = {value}")
            elif isinstance(value, int):
                print(f"   {key}: int = {value}")
            elif isinstance(value, float):
                print(f"   {key}: float = {value}")
        
        print()

def demonstrate_current_config():
    """Demuestra la configuración actual."""
    
    print("📋 === CONFIGURACIÓN ACTUAL ===\n")
    
    # Análisis de la configuración actual
    print_slice_len_analysis(config)
    
    # Probar con diferentes longitudes de archivo
    print("\n🔍 PRUEBA CON DIFERENTES ARCHIVOS:")
    
    test_file_lengths = [512, 1024, 2048, 4096]
    
    for file_length in test_file_lengths:
        if hasattr(config, 'SLICE_LEN_AUTO') and config.SLICE_LEN_AUTO:
            slice_len = get_dynamic_slice_len(config)
        else:
            slice_len = config.SLICE_LEN
        
        n_slices = file_length // getattr(config, 'DOWN_TIME_RATE', 1) // slice_len
        
        print(f"   • Archivo {file_length} muestras → {n_slices} slices de {slice_len} muestras")

def main():
    """Función principal de demostración."""
    
    print("🌟 SISTEMA DINÁMICO DE SLICE_LEN")
    print("=" * 50)
    print("Nueva funcionalidad que permite especificar SLICE_LEN")
    print("en términos de duración temporal (segundos) en lugar de muestras.")
    print("¡Mucho más intuitivo para el usuario!")
    print()
    
    # Demostraciones
    demonstrate_dynamic_slice_len()
    demonstrate_signal_type_suggestions()
    demonstrate_configuration_examples()
    demonstrate_current_config()
    
    print("\n✅ PRÓXIMOS PASOS:")
    print("   1. Modifica SLICE_DURATION_SECONDS en config.py según tu tipo de señal")
    print("   2. Asegúrate de que SLICE_LEN_AUTO = True")
    print("   3. Ejecuta el pipeline con: python main.py")
    print("   4. El sistema calculará automáticamente el SLICE_LEN óptimo")
    print("\n🎯 ¡Disfruta de la nueva funcionalidad más intuitiva!")

if __name__ == "__main__":
    main()
