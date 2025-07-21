#!/usr/bin/env python3
"""
Test para encontrar el SLICE_LEN óptimo que dé ~765 slices por chunk.
"""

import sys
from pathlib import Path
import numpy as np

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRAFTS import config


def find_optimal_slice_len():
    """Encontrar el SLICE_LEN óptimo para obtener ~765 slices por chunk."""
    
    print("🔍 BÚSQUEDA DE SLICE_LEN ÓPTIMO")
    print("=" * 60)
    
    # Parámetros del archivo
    chunk_size = 2_000_000
    down_time_rate = 14
    target_slices = 765
    
    print("📊 PARÁMETROS:")
    print(f"   📏 Chunk size: {chunk_size:,}")
    print(f"   🔽 DOWN_TIME_RATE: {down_time_rate}")
    print(f"   📊 Slices objetivo: {target_slices}")
    print()
    
    # Calcular chunk decimado
    chunk_decimated = chunk_size // down_time_rate
    print(f"📊 CHUNK DECIMADO:")
    print(f"   📏 Chunk original: {chunk_size:,} muestras")
    print(f"   🔽 Chunk decimado: {chunk_decimated:,} muestras")
    print()
    
    # Calcular SLICE_LEN óptimo
    optimal_slice_len = chunk_decimated // target_slices
    print(f"📊 SLICE_LEN ÓPTIMO:")
    print(f"   📏 SLICE_LEN = {chunk_decimated:,} ÷ {target_slices} = {optimal_slice_len}")
    print()
    
    # Verificar
    actual_slices = chunk_decimated // optimal_slice_len
    print(f"📊 VERIFICACIÓN:")
    print(f"   📊 Slices obtenidos: {actual_slices}")
    print(f"   📊 Diferencia: {actual_slices - target_slices}")
    print()
    
    # Calcular duración temporal
    time_reso = 5.46e-05
    time_reso_decimated = time_reso * down_time_rate
    duration_ms = optimal_slice_len * time_reso_decimated * 1000
    
    print(f"📊 DURACIÓN TEMPORAL:")
    print(f"   ⏱️  TIME_RESO: {time_reso}")
    print(f"   🔽 TIME_RESO decimado: {time_reso_decimated}")
    print(f"   📏 SLICE_LEN: {optimal_slice_len}")
    print(f"   ⏱️  Duración: {duration_ms:.1f} ms")
    print()
    
    # Comparar con configuración actual
    current_slice_len = 2616
    current_slices = chunk_decimated // current_slice_len
    current_duration_ms = current_slice_len * time_reso_decimated * 1000
    
    print(f"📊 COMPARACIÓN:")
    print(f"   📏 SLICE_LEN actual: {current_slice_len}")
    print(f"   📊 Slices actuales: {current_slices}")
    print(f"   ⏱️  Duración actual: {current_duration_ms:.1f} ms")
    print()
    print(f"   📏 SLICE_LEN óptimo: {optimal_slice_len}")
    print(f"   📊 Slices óptimos: {actual_slices}")
    print(f"   ⏱️  Duración óptima: {duration_ms:.1f} ms")
    print()
    
    # Factor de mejora
    improvement_factor = current_slices / actual_slices
    print(f"📊 MEJORA:")
    print(f"   📊 Factor de mejora: {improvement_factor:.2f}x más slices")
    print(f"   📊 Slices adicionales: {actual_slices - current_slices}")
    print()
    
    return optimal_slice_len, duration_ms


def test_slice_len_calculation():
    """Test del cálculo de SLICE_LEN basado en duración."""
    
    print("🧪 TEST: CÁLCULO DE SLICE_LEN POR DURACIÓN")
    print("=" * 60)
    
    # Parámetros
    time_reso = 5.46e-05
    down_time_rate = 14
    target_duration_ms = 2000.0  # Configuración actual
    
    print("📊 PARÁMETROS:")
    print(f"   ⏱️  TIME_RESO: {time_reso}")
    print(f"   🔽 DOWN_TIME_RATE: {down_time_rate}")
    print(f"   🎯 Duración objetivo: {target_duration_ms} ms")
    print()
    
    # Cálculo actual (incorrecto)
    time_reso_decimated = time_reso * down_time_rate
    slice_len_current = round(target_duration_ms / (time_reso_decimated * 1000))
    
    print("❌ CÁLCULO ACTUAL (INCORRECTO):")
    print(f"   ⏱️  TIME_RESO decimado: {time_reso_decimated}")
    print(f"   📏 SLICE_LEN = {target_duration_ms} ÷ ({time_reso_decimated} × 1000) = {slice_len_current}")
    print()
    
    # Cálculo correcto (para datos NO decimados)
    slice_len_correct = round(target_duration_ms / (time_reso * 1000))
    
    print("✅ CÁLCULO CORRECTO (PARA DATOS NO DECIMADOS):")
    print(f"   ⏱️  TIME_RESO original: {time_reso}")
    print(f"   📏 SLICE_LEN = {target_duration_ms} ÷ ({time_reso} × 1000) = {slice_len_correct}")
    print()
    
    # Verificar resultados
    chunk_size = 2_000_000
    chunk_decimated = chunk_size // down_time_rate
    
    slices_current = chunk_decimated // slice_len_current
    slices_correct = chunk_decimated // slice_len_correct
    
    print("📊 VERIFICACIÓN:")
    print(f"   📏 SLICE_LEN actual: {slice_len_current}")
    print(f"   📊 Slices obtenidos: {slices_current}")
    print(f"   📏 SLICE_LEN correcto: {slice_len_correct}")
    print(f"   📊 Slices correctos: {slices_correct}")
    print()
    
    # Recomendación
    print("💡 RECOMENDACIÓN:")
    print("   El SLICE_LEN debe calcularse para datos NO decimados")
    print("   Luego se aplica a datos que YA están decimados")
    print("   Esto da el número correcto de slices")
    print()


def main():
    """Función principal."""
    find_optimal_slice_len()
    print("\n" + "="*60 + "\n")
    test_slice_len_calculation()


if __name__ == "__main__":
    main() 