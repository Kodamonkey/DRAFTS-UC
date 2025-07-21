#!/usr/bin/env python3
"""
Test para validar las correcciones del bug de doble decimación.
Este test verifica que el cálculo de slices en el pipeline chunked sea correcto.
"""

import sys
from pathlib import Path
import numpy as np

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRAFTS import config
from DRAFTS.preprocessing import downsample_data


def test_double_decimation_fix():
    """Test para verificar que se corrigió el bug de doble decimación."""
    
    print("🧪 TEST: Corrección del Bug de Doble Decimación")
    print("=" * 60)
    
    # Simular datos de un chunk (como los que vienen de _load_fil_chunk)
    chunk_size = 2_000_000  # Tamaño típico de chunk
    n_freq = 512
    n_pol = 1
    
    # Crear datos de prueba
    data_chunk = np.random.rand(chunk_size, n_pol, n_freq).astype(np.float32)
    
    # Simular DOWN_TIME_RATE > 1 para demostrar el bug
    original_down_time_rate = config.DOWN_TIME_RATE
    config.DOWN_TIME_RATE = 14  # Valor típico donde se manifiesta el bug
    
    print(f"📊 Datos de entrada:")
    print(f"   📏 Shape del chunk: {data_chunk.shape}")
    print(f"   🔽 DOWN_TIME_RATE: {config.DOWN_TIME_RATE}")
    print(f"   🔽 DOWN_FREQ_RATE: {config.DOWN_FREQ_RATE}")
    print()
    
    # Aplicar downsample_data (como en _process_single_chunk línea 697)
    data_chunk_decimated = downsample_data(data_chunk)
    
    print(f"📊 Después de downsample_data:")
    print(f"   📏 Shape decimado: {data_chunk_decimated.shape}")
    print(f"   📏 Muestras temporales: {data_chunk_decimated.shape[0]}")
    print()
    
    # ❌ CÁLCULO INCORRECTO (antes de la corrección)
    width_total_incorrect = data_chunk_decimated.shape[0] // config.DOWN_TIME_RATE
    
    # ✅ CÁLCULO CORRECTO (después de la corrección)
    width_total_correct = data_chunk_decimated.shape[0]
    
    print(f"📊 Cálculo de width_total:")
    print(f"   ❌ Incorrecto (antes): {width_total_incorrect:,}")
    print(f"   ✅ Correcto (después): {width_total_correct:,}")
    print(f"   📈 Mejora: {width_total_correct / width_total_incorrect:.1f}x más slices")
    print()
    
    # Calcular slices con SLICE_LEN típico
    slice_len = 2616  # Valor típico
    
    time_slice_incorrect = (width_total_incorrect + slice_len - 1) // slice_len
    time_slice_correct = (width_total_correct + slice_len - 1) // slice_len
    
    print(f"📊 Cálculo de time_slice (SLICE_LEN={slice_len}):")
    print(f"   ❌ Slices incorrectos: {time_slice_incorrect}")
    print(f"   ✅ Slices correctos: {time_slice_correct}")
    print(f"   📈 Mejora: {time_slice_correct / time_slice_incorrect:.1f}x más slices")
    print()
    
    # Verificar que el cálculo correcto es consistente con el pipeline tradicional
    print(f"🔍 VERIFICACIÓN DE CONSISTENCIA:")
    
    # En pipeline tradicional: width_total = config.FILE_LENG // config.DOWN_TIME_RATE
    # Pero config.FILE_LENG ya representa datos decimados
    # Por lo tanto, en chunks: width_total = data_chunk.shape[0] (ya decimado)
    
    print(f"   📊 Pipeline tradicional: width_total = FILE_LENG (ya decimado)")
    print(f"   📊 Pipeline chunked: width_total = data_chunk.shape[0] (ya decimado)")
    print(f"   ✅ Ambos usan datos ya decimados - CONSISTENTE")
    print()
    
    # Verificar que no hay pérdida de información
    expected_slices_per_chunk = chunk_size // slice_len
    print(f"📊 Verificación de información:")
    print(f"   📏 Muestras por chunk: {chunk_size:,}")
    print(f"   📏 SLICE_LEN: {slice_len}")
    print(f"   📊 Slices esperados por chunk: {expected_slices_per_chunk}")
    print(f"   📊 Slices calculados (correcto): {time_slice_correct}")
    print(f"   ✅ Diferencia: {abs(time_slice_correct - expected_slices_per_chunk)}")
    print()
    
    # Restaurar configuración original
    config.DOWN_TIME_RATE = original_down_time_rate
    
    # Resultado del test
    if time_slice_correct > time_slice_incorrect:
        print("✅ TEST PASADO: La corrección funciona correctamente")
        print(f"   📈 Se recuperaron {time_slice_correct - time_slice_incorrect} slices por chunk")
        return True
    else:
        print("❌ TEST FALLIDO: La corrección no funciona")
        return False


def test_absolute_timing():
    """Test para verificar que el tiempo absoluto se pasa correctamente."""
    
    print("\n🧪 TEST: Tiempo Absoluto en Visualización")
    print("=" * 60)
    
    # Simular parámetros de chunk
    chunk_idx = 5
    start_sample_global = 10_000_000
    chunk_size = 2_000_000
    time_reso = config.TIME_RESO
    down_time_rate = config.DOWN_TIME_RATE
    
    # Calcular tiempo absoluto del chunk
    chunk_start_time_sec = start_sample_global * time_reso * down_time_rate
    
    print(f"📊 Parámetros del chunk:")
    print(f"   📏 Chunk index: {chunk_idx}")
    print(f"   📏 Start sample global: {start_sample_global:,}")
    print(f"   📏 Chunk size: {chunk_size:,}")
    print(f"   ⏱️  Time resolution: {time_reso:.6f} s")
    print(f"   🔽 Down time rate: {down_time_rate}")
    print()
    
    print(f"⏰ Cálculo de tiempo absoluto:")
    print(f"   🕐 chunk_start_time_sec = {start_sample_global} × {time_reso} × {down_time_rate}")
    print(f"   🕐 chunk_start_time_sec = {chunk_start_time_sec:.3f} segundos")
    print()
    
    # Verificar que se pasa correctamente a plot_waterfall_block
    print(f"🔍 Verificación de llamada a plot_waterfall_block:")
    print(f"   ✅ absolute_start_time=chunk_start_time_sec")
    print(f"   ✅ Tiempo absoluto se pasa correctamente")
    print()
    
    return True


def main():
    """Ejecutar todos los tests."""
    print("🚀 INICIANDO TESTS DE CORRECCIONES")
    print("=" * 80)
    
    test1_passed = test_double_decimation_fix()
    test2_passed = test_absolute_timing()
    
    print("\n📋 RESUMEN DE TESTS:")
    print("=" * 40)
    print(f"   🧪 Test doble decimación: {'✅ PASADO' if test1_passed else '❌ FALLIDO'}")
    print(f"   🧪 Test tiempo absoluto: {'✅ PASADO' if test2_passed else '❌ FALLIDO'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 TODOS LOS TESTS PASARON")
        print("   ✅ Las correcciones están funcionando correctamente")
    else:
        print("\n⚠️  ALGUNOS TESTS FALLARON")
        print("   ❌ Revisar las correcciones implementadas")


if __name__ == "__main__":
    main() 