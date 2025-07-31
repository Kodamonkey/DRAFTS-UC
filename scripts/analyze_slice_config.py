#!/usr/bin/env python3
"""
Script para analizar la configuración de slices y chunks
=======================================================

Este script te permite:
1. Calcular cuántos slices deberías tener por chunk según SLICE_DURATION_MS
2. Mostrar información detallada del archivo
3. Analizar la configuración de chunks
4. Verificar la continuidad temporal
5. Proporcionar recomendaciones de configuración

Uso: python scripts/analyze_slice_config.py <archivo.fits|archivo.fil>
"""

import sys
from pathlib import Path
import numpy as np

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRAFTS.core import config
from DRAFTS.preprocessing.slice_len_utils import update_slice_len_dynamic, calculate_slice_len_from_duration
from DRAFTS.data_loader import create_data_loader


def analyze_slice_configuration(file_path: str = None):
    """
    Analizar la configuración de slices y chunks.
    
    Args:
        file_path: Ruta al archivo de datos (opcional)
    """
    
    print("🔍 ANÁLISIS DE CONFIGURACIÓN DE SLICES Y CHUNKS")
    print("=" * 70)
    
    # 1. CONFIGURACIÓN ACTUAL
    print("\n📊 1. CONFIGURACIÓN ACTUAL:")
    print(f"   🎯 SLICE_DURATION_MS: {config.SLICE_DURATION_MS} ms")
    print(f"   📏 SLICE_LEN actual: {config.SLICE_LEN}")
    print(f"   ⏱️  TIME_RESO: {config.TIME_RESO}")
    print(f"   🔽 DOWN_TIME_RATE: {config.DOWN_TIME_RATE}")
    print(f"   🔽 DOWN_FREQ_RATE: {config.DOWN_FREQ_RATE}")
    print(f"   📏 SLICE_LEN_MIN: {config.SLICE_LEN_MIN}")
    print(f"   📏 SLICE_LEN_MAX: {config.SLICE_LEN_MAX}")
    
    # 2. CÁLCULO DINÁMICO DE SLICE_LEN
    print("\n🧮 2. CÁLCULO DINÁMICO DE SLICE_LEN:")
    
    # Calcular SLICE_LEN esperado
    slice_len_expected, duration_ms = calculate_slice_len_from_duration()
    print(f"   📏 SLICE_LEN esperado: {slice_len_expected}")
    
    # Verificar si coincide con el actual
    if config.SLICE_LEN == slice_len_expected:
        print(f"   ✅ SLICE_LEN actual coincide con el esperado")
    else:
        print(f"   ❌ SLICE_LEN actual ({config.SLICE_LEN}) NO coincide con el esperado ({slice_len_expected})")
        print(f"   💡 Ejecuta update_slice_len_dynamic() para corregir")
    
    # 3. ANÁLISIS DE ARCHIVO (si se proporciona)
    if file_path:
        print(f"\n📁 3. ANÁLISIS DEL ARCHIVO: {file_path}")
        
        try:
            # Crear DataLoader
            data_loader = create_data_loader(Path(file_path))
            metadata = data_loader.load_metadata()
            
            total_samples = metadata.get('nsamples', 0)
            total_time_seconds = total_samples * config.TIME_RESO * config.DOWN_TIME_RATE
            
            print(f"   📏 Muestras totales: {total_samples:,}")
            print(f"   ⏱️  Tiempo total: {total_time_seconds:.1f} s")
            print(f"   📊 Canales: {metadata.get('nchans', 'N/A')}")
            print(f"   🔢 Bits: {metadata.get('nbits', 'N/A')}")
            
            # 4. CONFIGURACIÓN DE CHUNKS
            print(f"\n📦 4. CONFIGURACIÓN DE CHUNKS:")
            
            # Usar SLICE_LEN esperado para cálculos
            slice_len_for_calc = slice_len_expected if slice_len_expected else config.SLICE_LEN
            
            # Configuración típica de chunks
            chunk_size = 2_000_000  # Tamaño típico
            num_chunks = (total_samples + chunk_size - 1) // chunk_size
            chunk_duration = chunk_size * config.TIME_RESO * config.DOWN_TIME_RATE
            slices_per_chunk = chunk_size // slice_len_for_calc
            
            print(f"   📏 Tamaño de chunk: {chunk_size:,}")
            print(f"   📊 Número de chunks: {num_chunks}")
            print(f"   ⏱️  Duración por chunk: {chunk_duration:.1f} s")
            print(f"   📊 Slices por chunk: {slices_per_chunk}")
            
            # 5. VERIFICACIÓN DE CONTINUIDAD TEMPORAL
            print(f"\n🔄 5. VERIFICACIÓN DE CONTINUIDAD TEMPORAL:")
            
            for chunk_idx in range(min(3, num_chunks)):
                start_sample = chunk_idx * chunk_size
                end_sample = min(start_sample + chunk_size, total_samples)
                
                start_time = start_sample * config.TIME_RESO * config.DOWN_TIME_RATE
                end_time = end_sample * config.TIME_RESO * config.DOWN_TIME_RATE
                
                print(f"   📦 Chunk {chunk_idx}:")
                print(f"      📏 Muestras: {start_sample:,} a {end_sample:,}")
                print(f"      ⏱️  Tiempo: {start_time:.1f}s a {end_time:.1f}s")
                print(f"      📊 Slices: {slices_per_chunk}")
                
                # Verificar continuidad con chunk anterior
                if chunk_idx > 0:
                    prev_end_time = (chunk_idx - 1) * chunk_size * config.TIME_RESO * config.DOWN_TIME_RATE
                    gap = start_time - prev_end_time
                    print(f"      🔗 Gap con anterior: {gap:.3f}s")
                    
                    if abs(gap) < 0.001:
                        print(f"      ✅ Continuidad perfecta")
                    else:
                        print(f"      ⚠️  Gap detectado: {gap:.3f}s")
            
            # 6. VERIFICACIÓN DE SLICES
            print(f"\n📊 6. VERIFICACIÓN DE SLICES:")
            
            total_slices_expected = total_samples // slice_len_for_calc
            slice_duration_actual = slice_len_for_calc * config.TIME_RESO * config.DOWN_TIME_RATE * 1000  # ms
            
            print(f"   📊 Slices totales esperados: {total_slices_expected}")
            print(f"   📊 Slices por chunk: {slices_per_chunk}")
            print(f"   ⏱️  Duración real por slice: {slice_duration_actual:.1f} ms")
            
            if abs(slice_duration_actual - config.SLICE_DURATION_MS) < 1.0:
                print(f"   ✅ Duración de slices correcta")
            else:
                print(f"   ❌ Duración de slices incorrecta")
            
            # 7. RECOMENDACIONES
            print(f"\n💡 7. RECOMENDACIONES:")
            
            if config.SLICE_LEN != slice_len_expected:
                print(f"   🔧 Ejecutar update_slice_len_dynamic() para corregir SLICE_LEN")
            
            if slices_per_chunk < 10:
                print(f"   ⚠️  Pocos slices por chunk ({slices_per_chunk}). Considera reducir chunk_size")
            
            if slices_per_chunk > 1000:
                print(f"   ⚠️  Muchos slices por chunk ({slices_per_chunk}). Considera aumentar chunk_size")
            
            print(f"   ✅ Usar Pipeline Chunked V2 para continuidad temporal perfecta")
            
        except Exception as e:
            print(f"   ❌ Error al analizar archivo: {e}")
    
    # 8. RESUMEN FINAL
    print(f"\n🎉 RESUMEN FINAL:")
    print(f"   🎯 SLICE_DURATION_MS: {config.SLICE_DURATION_MS} ms")
    print(f"   📏 SLICE_LEN esperado: {slice_len_expected}")
    print(f"   📏 SLICE_LEN actual: {config.SLICE_LEN}")
    
    if file_path:
        print(f"   📊 Slices por chunk: {slices_per_chunk if 'slices_per_chunk' in locals() else 'N/A'}")
        print(f"   ⏱️  Tiempo total: {total_time_seconds:.1f}s" if 'total_time_seconds' in locals() else "   ⏱️  Tiempo total: N/A")
    
    print(f"   🚀 Estado: {'✅ LISTO' if config.SLICE_LEN == slice_len_expected else '❌ REQUIERE CORRECCIÓN'}")


def main():
    """Función principal."""
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        analyze_slice_configuration(file_path)
    else:
        print("📋 Uso: python scripts/analyze_slice_config.py <archivo.fits|archivo.fil>")
        print("📋 Ejemplo: python scripts/analyze_slice_config.py Data/3098_0001_00_8bit.fil")
        print("\n" + "=" * 70)
        analyze_slice_configuration()


if __name__ == "__main__":
    main() 
