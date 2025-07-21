#!/usr/bin/env python3
"""
Análisis específico de la discrepancia de slices:
Esperado: 765 slices por chunk
Real: 140 slices por chunk
"""

import sys
from pathlib import Path
import numpy as np

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRAFTS import config


def analyze_slice_calculation():
    """Analizar el cálculo de slices paso a paso."""
    
    print("🔍 ANÁLISIS DE DISCREPANCIA DE SLICES")
    print("=" * 60)
    
    # Parámetros del archivo según el output
    total_samples = 65_917_985
    down_time_rate = 14
    slice_len = 2616
    chunk_size = 2_000_000
    overlap = 1000
    
    print("📊 PARÁMETROS DEL ARCHIVO:")
    print(f"   📏 Total samples: {total_samples:,}")
    print(f"   🔽 DOWN_TIME_RATE: {down_time_rate}")
    print(f"   📏 SLICE_LEN: {slice_len}")
    print(f"   📏 Chunk size: {chunk_size:,}")
    print(f"   📏 Overlap: {overlap}")
    print()
    
    # Cálculo 1: Muestras después de decimación
    samples_after_decimation = total_samples // down_time_rate
    print("📊 CÁLCULO 1: MUESTRAS DESPUÉS DE DECIMACIÓN:")
    print(f"   📏 Muestras originales: {total_samples:,}")
    print(f"   🔽 Factor decimación: {down_time_rate}")
    print(f"   📏 Muestras después decimación: {samples_after_decimation:,}")
    print()
    
    # Cálculo 2: Slices totales en el archivo
    total_slices = samples_after_decimation // slice_len
    print("📊 CÁLCULO 2: SLICES TOTALES EN EL ARCHIVO:")
    print(f"   📏 Muestras decimadas: {samples_after_decimation:,}")
    print(f"   📏 SLICE_LEN: {slice_len}")
    print(f"   📊 Slices totales: {total_slices:,}")
    print()
    
    # Cálculo 3: Chunks y slices por chunk
    effective_chunk_size = chunk_size - overlap
    num_chunks = (total_samples + effective_chunk_size - 1) // effective_chunk_size
    
    print("📊 CÁLCULO 3: CHUNKS Y SLICES POR CHUNK:")
    print(f"   📏 Effective chunk size: {effective_chunk_size:,}")
    print(f"   📊 Número de chunks: {num_chunks}")
    print()
    
    # Cálculo 4: Slices por chunk (ESPERADO)
    slices_per_chunk_expected = chunk_size // slice_len
    print("📊 CÁLCULO 4: SLICES POR CHUNK (ESPERADO):")
    print(f"   📏 Chunk size: {chunk_size:,}")
    print(f"   📏 SLICE_LEN: {slice_len}")
    print(f"   📊 Slices por chunk esperado: {slices_per_chunk_expected}")
    print()
    
    # Cálculo 5: Slices por chunk (REAL - después de decimación)
    chunk_size_decimated = chunk_size // down_time_rate
    slices_per_chunk_real = chunk_size_decimated // slice_len
    print("📊 CÁLCULO 5: SLICES POR CHUNK (REAL - DESPUÉS DECIMACIÓN):")
    print(f"   📏 Chunk size original: {chunk_size:,}")
    print(f"   🔽 Chunk size decimado: {chunk_size_decimated:,}")
    print(f"   📏 SLICE_LEN: {slice_len}")
    print(f"   📊 Slices por chunk real: {slices_per_chunk_real}")
    print()
    
    # Cálculo 6: Verificar el problema en el código
    print("🔍 ANÁLISIS DEL PROBLEMA EN EL CÓDIGO:")
    print("-" * 40)
    
    # Simular el cálculo actual en _process_single_chunk
    print("❌ CÁLCULO ACTUAL EN _process_single_chunk:")
    print("   data_chunk = downsample_data(data_chunk)  # Decimación")
    print("   width_total = data_chunk.shape[0]  # Ya decimado")
    print("   time_slice = (width_total + slice_len - 1) // slice_len")
    print()
    
    # Simular con datos reales
    chunk_data_original = np.zeros((chunk_size, 512))  # Simular chunk original
    print(f"   📏 Chunk original shape: {chunk_data_original.shape}")
    
    # Simular decimación
    chunk_data_decimated = chunk_data_original[::down_time_rate]
    print(f"   🔽 Chunk decimado shape: {chunk_data_decimated.shape}")
    
    # Cálculo actual
    width_total = chunk_data_decimated.shape[0]
    time_slice_current = (width_total + slice_len - 1) // slice_len
    print(f"   📊 Slices calculados (actual): {time_slice_current}")
    print()
    
    # Cálculo correcto
    print("✅ CÁLCULO CORRECTO:")
    print("   width_total = data_chunk.shape[0]  # Ya decimado")
    print("   time_slice = width_total // slice_len  # Sin +1")
    print()
    
    time_slice_correct = width_total // slice_len
    print(f"   📊 Slices calculados (correcto): {time_slice_correct}")
    print()
    
    # Verificar si el problema está en el +1
    print("🔍 VERIFICACIÓN DEL PROBLEMA:")
    print(f"   📊 Con +1: {time_slice_current}")
    print(f"   📊 Sin +1: {time_slice_correct}")
    print(f"   📊 Diferencia: {time_slice_current - time_slice_correct}")
    print()
    
    # Verificar si el problema está en el SLICE_LEN
    print("🔍 VERIFICACIÓN DE SLICE_LEN:")
    print(f"   📏 SLICE_LEN actual: {slice_len}")
    print(f"   📏 Chunk decimado: {chunk_size_decimated:,}")
    print(f"   📊 División exacta: {chunk_size_decimated % slice_len == 0}")
    print(f"   📊 Resto: {chunk_size_decimated % slice_len}")
    print()
    
    # Calcular SLICE_LEN óptimo
    optimal_slice_len = chunk_size_decimated // 765  # Para obtener 765 slices
    print("🔍 SLICE_LEN ÓPTIMO PARA 765 SLICES:")
    print(f"   📊 Slices deseados: 765")
    print(f"   📏 Chunk decimado: {chunk_size_decimated:,}")
    print(f"   📏 SLICE_LEN óptimo: {optimal_slice_len}")
    print(f"   📊 Verificación: {chunk_size_decimated // optimal_slice_len} slices")
    print()


def analyze_config_issue():
    """Analizar si el problema está en la configuración."""
    
    print("\n🔍 ANÁLISIS DE CONFIGURACIÓN:")
    print("=" * 60)
    
    # Verificar si SLICE_LEN se está calculando correctamente
    print("📊 CÁLCULO DE SLICE_LEN:")
    
    # Parámetros del archivo
    time_reso = 5.46e-05  # Según el output
    down_time_rate = 14
    slice_duration_ms = 2000.0  # Configuración actual
    
    # Cálculo esperado de SLICE_LEN
    time_reso_decimated = time_reso * down_time_rate
    slice_len_expected = round(slice_duration_ms / (time_reso_decimated * 1000))
    
    print(f"   ⏱️  TIME_RESO: {time_reso}")
    print(f"   🔽 DOWN_TIME_RATE: {down_time_rate}")
    print(f"   ⏱️  TIME_RESO decimado: {time_reso_decimated}")
    print(f"   🎯 SLICE_DURATION_MS: {slice_duration_ms}")
    print(f"   📏 SLICE_LEN esperado: {slice_len_expected}")
    print()
    
    # Verificar si el problema está en el cálculo dinámico
    print("🔍 PROBLEMA POTENCIAL:")
    print("   El SLICE_LEN se calcula dinámicamente basado en SLICE_DURATION_MS")
    print("   Pero el chunk ya está decimado, entonces:")
    print("   - SLICE_LEN se calcula para datos decimados")
    print("   - Pero se aplica a datos que YA están decimados")
    print("   - Esto puede causar doble decimación en el cálculo")
    print()


def suggest_fixes():
    """Sugerir correcciones para el problema."""
    
    print("\n💡 CORRECCIONES SUGERIDAS:")
    print("=" * 60)
    
    print("1️⃣ CORRECCIÓN EN _process_single_chunk:")
    print("   - Cambiar: time_slice = (width_total + slice_len - 1) // slice_len")
    print("   - Por: time_slice = width_total // slice_len")
    print("   - Razón: Los datos ya están decimados, no necesitamos el +1")
    print()
    
    print("2️⃣ VERIFICAR CÁLCULO DE SLICE_LEN:")
    print("   - Confirmar que SLICE_LEN se calcula correctamente")
    print("   - Verificar que no hay doble decimación")
    print("   - Asegurar que coincide con SLICE_DURATION_MS")
    print()
    
    print("3️⃣ DEBUG DETALLADO:")
    print("   - Agregar logs en _process_single_chunk")
    print("   - Verificar shape de data_chunk antes y después de decimación")
    print("   - Confirmar que width_total es correcto")
    print()


def main():
    """Función principal."""
    analyze_slice_calculation()
    analyze_config_issue()
    suggest_fixes()


if __name__ == "__main__":
    main() 