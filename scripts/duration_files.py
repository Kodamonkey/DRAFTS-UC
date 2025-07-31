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
from DRAFTS.io.io import get_obparams
from DRAFTS.io.filterbank_io import get_obparams_fil


def analyze_slice_configuration(file_path: str) -> None:
    """Analizar la configuración de slices y chunks para un archivo."""
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ Error: El archivo {file_path} no existe")
        return
    
    print(f" ANÁLISIS DE CONFIGURACIÓN DE SLICES Y CHUNKS")
    print("=" * 80)
    print(f"📁 Archivo: {file_path.name}")
    print("=" * 80)
    
    try:
        # Cargar parámetros según el tipo de archivo
        if file_path.suffix.lower() == ".fits":
            get_obparams(str(file_path))
        elif file_path.suffix.lower() == ".fil":
            get_obparams_fil(str(file_path))
        else:
            print(f"❌ Error: Formato de archivo no soportado: {file_path.suffix}")
            return
        
        # =============================================================================
        # 1. INFORMACIÓN BÁSICA DEL ARCHIVO
        # =============================================================================
        
        print("\n INFORMACIÓN BÁSICA DEL ARCHIVO:")
        print("-" * 50)
        
        total_samples = config.FILE_LENG
        time_resolution = config.TIME_RESO
        down_time_rate = config.DOWN_TIME_RATE
        down_freq_rate = config.DOWN_FREQ_RATE
        
        # Duración sin decimación
        duration_raw_sec = total_samples * time_resolution
        duration_raw_min = duration_raw_sec / 60
        duration_raw_hour = duration_raw_min / 60
        
        # Duración después de decimación
        duration_decimated_sec = total_samples * time_resolution * down_time_rate
        duration_decimated_min = duration_decimated_sec / 60
        duration_decimated_hour = duration_decimated_min / 60
        
        print(f"   📏 Muestras totales: {total_samples:,}")
        print(f"   ⏱️  Resolución temporal: {time_resolution:.6f} segundos")
        print(f"   🔽 Factor de decimación temporal: {down_time_rate}")
        print(f"   🔽 Factor de decimación frecuencial: {down_freq_rate}")
        print()
        
        print("⏰ DURACIÓN CALCULADA:")
        print(f"    Sin decimación:")
        print(f"      {duration_raw_sec:.3f} segundos ({duration_raw_min:.2f} min, {duration_raw_hour:.2f} h)")
        print(f"    Con decimación (usado en pipeline):")
        print(f"      {duration_decimated_sec:.3f} segundos ({duration_decimated_min:.2f} min, {duration_decimated_hour:.2f} h)")
        
        # =============================================================================
        # 2. CONFIGURACIÓN ACTUAL DE SLICES
        # =============================================================================
        
        print("\n🎯 CONFIGURACIÓN ACTUAL DE SLICES:")
        print("-" * 50)
        
        slice_duration_ms = config.SLICE_DURATION_MS
        slice_len = config.SLICE_LEN
        
        print(f"    SLICE_DURATION_MS: {slice_duration_ms:.1f} ms")
        print(f"   📏 SLICE_LEN: {slice_len} muestras")
        
        # Verificar si SLICE_LEN coincide con SLICE_DURATION_MS
        target_duration_s = slice_duration_ms / 1000.0
        time_reso_decimated = time_resolution * down_time_rate
        expected_slice_len = round(target_duration_s / time_reso_decimated)
        
        print(f"    SLICE_LEN esperado: {expected_slice_len} muestras")
        print(f"   ⏱️  Duración real del slice: {slice_len * time_reso_decimated * 1000:.1f} ms")
        print(f"    Diferencia: {abs(slice_len * time_reso_decimated * 1000 - slice_duration_ms):.1f} ms")
        
        if abs(slice_len * time_reso_decimated * 1000 - slice_duration_ms) < 1.0:
            print("   ✅ CONFIGURACIÓN CORRECTA")
        else:
            print("   ⚠️  CONFIGURACIÓN INCORRECTA - SLICE_LEN no coincide con SLICE_DURATION_MS")
        
        # =============================================================================
        # 3. ANÁLISIS DE CHUNKS
        # =============================================================================
        
        print("\n ANÁLISIS DE CHUNKS:")
        print("-" * 50)
        
        chunk_size = config.MAX_SAMPLES_LIMIT
        overlap = getattr(config, 'CHUNK_OVERLAP_SAMPLES', 500)
        effective_chunk_size = chunk_size - overlap
        
        print(f"    Tamaño de chunk: {chunk_size:,} muestras")
        print(f"    Overlap: {overlap} muestras")
        print(f"    Tamaño efectivo: {effective_chunk_size:,} muestras")
        
        # Calcular número de chunks
        num_chunks = (total_samples + effective_chunk_size - 1) // effective_chunk_size
        
        print(f"   📊 Número de chunks: {num_chunks}")
        print(f"   ⏱️  Duración por chunk: {chunk_size * time_reso_decimated:.1f} segundos")
        
        # =============================================================================
        # 4. CÁLCULO DE SLICES POR CHUNK
        # =============================================================================
        
        print("\n📊 CÁLCULO DE SLICES POR CHUNK:")
        print("-" * 50)
        
        # Chunk decimado
        chunk_decimated = chunk_size // down_time_rate
        slices_per_chunk = chunk_decimated // slice_len
        
        print(f"   📏 Chunk original: {chunk_size:,} muestras")
        print(f"   📏 Chunk decimado: {chunk_decimated:,} muestras")
        print(f"   📏 SLICE_LEN: {slice_len} muestras")
        print(f"   📊 Slices por chunk: {chunk_decimated} ÷ {slice_len} = {slices_per_chunk}")
        
        # Verificar si hay slices perdidos
        total_slices_in_chunk = chunk_decimated // slice_len
        remaining_samples = chunk_decimated % slice_len
        
        if remaining_samples > 0:
            print(f"   ⚠️  Muestras perdidas por chunk: {remaining_samples}")
            print(f"   ⚠️  Tiempo perdido por chunk: {remaining_samples * time_reso_decimated * 1000:.1f} ms")
        else:
            print("   ✅ No hay muestras perdidas")
        
        # =============================================================================
        # 5. ANÁLISIS DE CONTINUIDAD TEMPORAL
        # =============================================================================
        
        print("\n ANÁLISIS DE CONTINUIDAD TEMPORAL:")
        print("-" * 50)
        
        # Simular primeros 3 chunks
        for chunk_idx in range(min(3, num_chunks)):
            start_sample = chunk_idx * effective_chunk_size
            if chunk_idx > 0:
                start_sample -= overlap
            
            end_sample = min(start_sample + chunk_size, total_samples)
            
            # Tiempos absolutos
            chunk_start_time_sec = start_sample * time_reso_decimated
            chunk_end_time_sec = end_sample * time_reso_decimated
            
            # Slices en este chunk
            chunk_decimated_size = (end_sample - start_sample) // down_time_rate
            slices_in_chunk = chunk_decimated_size // slice_len
            
            print(f"    Chunk {chunk_idx}:")
            print(f"      📏 Muestras: {start_sample:,} a {end_sample:,}")
            print(f"      🕐 Tiempo: {chunk_start_time_sec:.3f}s a {chunk_end_time_sec:.3f}s")
            print(f"       Slices: {slices_in_chunk}")
            
            if chunk_idx > 0:
                prev_chunk = chunk_idx - 1
                prev_start = prev_chunk * effective_chunk_size
                if prev_chunk > 0:
                    prev_start -= overlap
                prev_end = min(prev_start + chunk_size, total_samples)
                prev_end_time = prev_end * time_reso_decimated
                gap = chunk_start_time_sec - prev_end_time
                print(f"      🔗 Gap con anterior: {gap:.3f}s")
        
        # =============================================================================
        # 6. ESTIMACIÓN TOTAL DE SLICES
        # =============================================================================
        
        print("\n📈 ESTIMACIÓN TOTAL DE SLICES:")
        print("-" * 50)
        
        # Slices totales esperados
        total_slices_expected = (total_samples // down_time_rate) // slice_len
        total_slices_actual = (total_samples // down_time_rate + slice_len - 1) // slice_len
        
        print(f"   📊 Slices totales esperados: {total_slices_expected:,}")
        print(f"   📊 Slices totales (con redondeo): {total_slices_actual:,}")
        print(f"    Slices por hora: {total_slices_actual / duration_decimated_hour:.0f}")
        print(f"    Slices por minuto: {total_slices_actual / duration_decimated_min:.1f}")
        
        # =============================================================================
        # 7. INFORMACIÓN DE FRECUENCIA
        # =============================================================================
        
        print("\n📡 INFORMACIÓN DE FRECUENCIA:")
        print("-" * 50)
        
        if hasattr(config, 'FREQ') and config.FREQ is not None:
            freq_min = config.FREQ.min()
            freq_max = config.FREQ.max()
            bandwidth = freq_max - freq_min
            
            print(f"    Frecuencia mínima: {freq_min:.1f} MHz")
            print(f"    Frecuencia máxima: {freq_max:.1f} MHz")
            print(f"   📡 Ancho de banda: {bandwidth:.1f} MHz")
            print(f"    Canales originales: {config.FREQ_RESO}")
            print(f"    Canales después de decimación: {config.FREQ_RESO // down_freq_rate}")
        else:
            print("   ⚠️  Información de frecuencia no disponible")
        
        # =============================================================================
        # 8. RECOMENDACIONES
        # =============================================================================
        
        print("\n💡 RECOMENDACIONES:")
        print("-" * 50)
        
        # Verificar configuración
        if abs(slice_len * time_reso_decimated * 1000 - slice_duration_ms) > 1.0:
            print("   🔧 Ajustar SLICE_DURATION_MS para que coincida con SLICE_LEN")
            print(f"      SLICE_DURATION_MS sugerido: {slice_len * time_reso_decimated * 1000:.1f}")
        
        if remaining_samples > 0:
            print("    Considerar ajustar SLICE_LEN para evitar pérdida de datos")
        
        if slices_per_chunk < 10:
            print("   🔧 Slices por chunk muy bajos - considerar reducir SLICE_DURATION_MS")
        elif slices_per_chunk > 1000:
            print("   🔧 Slices por chunk muy altos - considerar aumentar SLICE_DURATION_MS")
        
        # Memoria estimada
        memory_per_chunk_gb = (chunk_size * config.FREQ_RESO // down_freq_rate * 4) / (1024**3)
        print(f"   💾 Memoria estimada por chunk: {memory_per_chunk_gb:.2f} GB")
        
        if memory_per_chunk_gb > 4.0:
            print("   ⚠️  Memoria por chunk muy alta - considerar reducir MAX_SAMPLES_LIMIT")
        
        # =============================================================================
        # 9. RESUMEN FINAL
        # =============================================================================
        
        print("\n📋 RESUMEN FINAL:")
        print("-" * 50)
        print(f"    SLICE_DURATION_MS: {slice_duration_ms:.1f} ms")
        print(f"   📏 SLICE_LEN: {slice_len} muestras")
        print(f"   📊 Slices por chunk: {slices_per_chunk}")
        print(f"   📦 Número de chunks: {num_chunks}")
        print(f"   ⏱️  Duración total: {duration_decimated_hour:.2f} horas")
        print(f"   📊 Slices totales: {total_slices_actual:,}")
        
        if abs(slice_len * time_reso_decimated * 1000 - slice_duration_ms) < 1.0 and remaining_samples == 0:
            print("   ✅ CONFIGURACIÓN ÓPTIMA")
        else:
            print("   ⚠️  CONFIGURACIÓN REQUIERE AJUSTES")
        
    except Exception as e:
        print(f"❌ Error al analizar el archivo: {e}")
        import traceback
        traceback.print_exc()
        return


def main():
    """Función principal."""
    if len(sys.argv) != 2:
        print(" ANÁLISIS DE CONFIGURACIÓN DE SLICES Y CHUNKS")
        print("=" * 60)
        print("Uso: python scripts/analyze_slice_config.py <archivo.fits|archivo.fil>")
        print()
        print("Ejemplos:")
        print("  python scripts/analyze_slice_config.py Data/observacion.fits")
        print("  python scripts/analyze_slice_config.py Data/observacion.fil")
        print()
        print("Este script te mostrará:")
        print("   Información detallada del archivo")
        print("🎯 Configuración actual de slices")
        print(" Análisis de chunks")
        print("📊 Cálculo de slices por chunk")
        print("🔗 Continuidad temporal")
        print(" Recomendaciones de configuración")
        return
    
    file_path = sys.argv[1]
    analyze_slice_configuration(file_path)


if __name__ == "__main__":
    main()
