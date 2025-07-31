#!/usr/bin/env python3
"""
Script de prueba para demostrar el cálculo automático de parámetros de procesamiento
basado únicamente en SLICE_DURATION_MS.

Este script simula diferentes configuraciones y muestra cómo el sistema calcula
automáticamente todos los demás parámetros.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path

# Simular diferentes configuraciones de archivo
def simulate_file_configs():
    """Simula diferentes configuraciones de archivo para probar el cálculo automático."""
    
    # Configuraciones típicas de archivos de radioastronomía
    configs = [
        {
            'name': 'Archivo de alta resolución temporal',
            'TIME_RESO': 0.0001,  # 0.1 ms
            'FREQ_RESO': 1024,
            'FILE_LENG': 10_000_000,
            'DOWN_TIME_RATE': 1,
            'DOWN_FREQ_RATE': 1
        },
        {
            'name': 'Archivo de resolución media',
            'TIME_RESO': 0.001,   # 1 ms
            'FREQ_RESO': 512,
            'FILE_LENG': 5_000_000,
            'DOWN_TIME_RATE': 1,
            'DOWN_FREQ_RATE': 1
        },
        {
            'name': 'Archivo de baja resolución temporal',
            'TIME_RESO': 0.01,    # 10 ms
            'FREQ_RESO': 256,
            'FILE_LENG': 1_000_000,
            'DOWN_TIME_RATE': 1,
            'DOWN_FREQ_RATE': 1
        },
        {
            'name': 'Archivo con downsampling',
            'TIME_RESO': 0.0001,  # 0.1 ms original
            'FREQ_RESO': 1024,
            'FILE_LENG': 20_000_000,
            'DOWN_TIME_RATE': 4,  # Decimado temporal
            'DOWN_FREQ_RATE': 2   # Decimado frecuencial
        }
    ]
    
    return configs

def test_slice_durations():
    """Prueba diferentes duraciones de slice."""
    return [32.0, 64.0, 128.0, 256.0, 512.0]  # ms

def run_automatic_parameter_test():
    """Ejecuta la prueba del sistema automático de parámetros."""
    
    print("🧪 PRUEBA DEL SISTEMA AUTOMÁTICO DE PARÁMETROS")
    print("=" * 60)
    print("Este script demuestra cómo el sistema calcula automáticamente")
    print("todos los parámetros basándose únicamente en SLICE_DURATION_MS")
    print()
    
    # Importar el módulo de cálculo
    from drafts.preprocessing.slice_len_calculator import (
        calculate_slice_len_from_duration,
        calculate_optimal_chunk_size,
        get_processing_parameters,
        validate_processing_parameters
    )
    
    # Simular configuraciones
    file_configs = simulate_file_configs()
    slice_durations = test_slice_durations()
    
    print("📊 RESULTADOS DE LA PRUEBA")
    print("-" * 60)
    
    for file_config in file_configs:
        print(f"\n🔧 {file_config['name']}")
        print(f"   TIME_RESO: {file_config['TIME_RESO']}s")
        print(f"   FREQ_RESO: {file_config['FREQ_RESO']} canales")
        print(f"   FILE_LENG: {file_config['FILE_LENG']:,} muestras")
        print(f"   DOWN_TIME_RATE: {file_config['DOWN_TIME_RATE']}")
        print(f"   DOWN_FREQ_RATE: {file_config['DOWN_FREQ_RATE']}")
        
        # Simular la configuración global
        import drafts.config as config
        config.TIME_RESO = file_config['TIME_RESO']
        config.FREQ_RESO = file_config['FREQ_RESO']
        config.FILE_LENG = file_config['FILE_LENG']
        config.DOWN_TIME_RATE = file_config['DOWN_TIME_RATE']
        config.DOWN_FREQ_RATE = file_config['DOWN_FREQ_RATE']
        
        for slice_duration_ms in slice_durations:
            config.SLICE_DURATION_MS = slice_duration_ms
            
            try:
                # Calcular parámetros automáticamente
                params = get_processing_parameters()
                
                if validate_processing_parameters(params):
                    print(f"   ✅ SLICE_DURATION_MS={slice_duration_ms}ms:")
                    print(f"      • Slice: {params['slice_len']} muestras ({params['slice_duration_ms']:.1f}ms)")
                    print(f"      • Chunk: {params['chunk_samples']:,} muestras ({params['chunk_duration_sec']:.1f}s)")
                    print(f"      • Slices/chunk: {params['slices_per_chunk']}")
                    print(f"      • Total: {params['total_chunks']} chunks, {params['total_slices']} slices")
                else:
                    print(f"   ❌ SLICE_DURATION_MS={slice_duration_ms}ms: Parámetros inválidos")
                    
            except Exception as e:
                print(f"   ❌ SLICE_DURATION_MS={slice_duration_ms}ms: Error - {e}")
    
    print("\n" + "=" * 60)
    print("✅ PRUEBA COMPLETADA")
    print("\n💡 CONCLUSIONES:")
    print("• El sistema calcula automáticamente slice_len basado en SLICE_DURATION_MS")
    print("• El chunk_size se optimiza considerando memoria y eficiencia")
    print("• Todos los parámetros se validan automáticamente")
    print("• El usuario solo necesita configurar SLICE_DURATION_MS")

if __name__ == "__main__":
    run_automatic_parameter_test() 