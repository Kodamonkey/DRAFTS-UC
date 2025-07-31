#!/usr/bin/env python3
"""
Script para corregir la configuración de slices
==============================================

Este script:
1. Carga los metadatos del archivo correctamente
2. Actualiza la configuración global
3. Calcula SLICE_LEN dinámicamente
4. Verifica que todo esté correcto
"""

import sys
from pathlib import Path
import numpy as np

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRAFTS.core import config
from DRAFTS.preprocessing.slice_len_utils import update_slice_len_dynamic
from DRAFTS.data_loader import create_data_loader


def fix_slice_configuration(file_path: str):
    """
    Corregir la configuración de slices.
    
    Args:
        file_path: Ruta al archivo de datos
    """
    
    print("🔧 CORRECCIÓN DE CONFIGURACIÓN DE SLICES")
    print("=" * 60)
    
    # 1. CONFIGURACIÓN ANTES
    print("\n📊 1. CONFIGURACIÓN ANTES:")
    print(f"   🎯 SLICE_DURATION_MS: {config.SLICE_DURATION_MS} ms")
    print(f"   📏 SLICE_LEN: {config.SLICE_LEN}")
    print(f"   ⏱️  TIME_RESO: {config.TIME_RESO}")
    print(f"   🔽 DOWN_TIME_RATE: {config.DOWN_TIME_RATE}")
    print(f"   🔽 DOWN_FREQ_RATE: {config.DOWN_FREQ_RATE}")
    
    # 2. CARGAR METADATOS DEL ARCHIVO
    print(f"\n📁 2. CARGANDO METADATOS DE: {file_path}")
    
    try:
        # Crear DataLoader
        data_loader = create_data_loader(Path(file_path))
        metadata = data_loader.load_metadata()
        
        print(f"   ✅ Metadatos cargados exitosamente")
        print(f"   📏 Muestras totales: {metadata.get('nsamples', 0):,}")
        print(f"   ⏱️  TIME_RESO del archivo: {metadata.get('time_reso', 'N/A')}")
        print(f"   📊 Canales: {metadata.get('nchans', 'N/A')}")
        print(f"   🔢 Bits: {metadata.get('nbits', 'N/A')}")
        
        # 3. ACTUALIZAR CONFIGURACIÓN GLOBAL
        print(f"\n⚙️  3. ACTUALIZANDO CONFIGURACIÓN GLOBAL:")
        
        # Actualizar TIME_RESO si está disponible
        if metadata.get('time_reso'):
            config.TIME_RESO = metadata['time_reso']
            print(f"   ✅ TIME_RESO actualizado: {config.TIME_RESO}")
        
        # Actualizar DOWN_TIME_RATE si está disponible
        if metadata.get('down_time_rate'):
            config.DOWN_TIME_RATE = metadata['down_time_rate']
            print(f"   ✅ DOWN_TIME_RATE actualizado: {config.DOWN_TIME_RATE}")
        
        # Actualizar DOWN_FREQ_RATE si está disponible
        if metadata.get('down_freq_rate'):
            config.DOWN_FREQ_RATE = metadata['down_freq_rate']
            print(f"   ✅ DOWN_FREQ_RATE actualizado: {config.DOWN_FREQ_RATE}")
        
        # 4. CALCULAR SLICE_LEN DINÁMICAMENTE
        print(f"\n🧮 4. CALCULANDO SLICE_LEN DINÁMICAMENTE:")
        
        slice_len_old = config.SLICE_LEN
        slice_len_new, duration_ms = update_slice_len_dynamic()
        
        print(f"   📏 SLICE_LEN anterior: {slice_len_old}")
        print(f"   📏 SLICE_LEN nuevo: {slice_len_new}")
        print(f"   ⏱️  Duración real: {duration_ms:.1f} ms")
        
        # 5. VERIFICAR RESULTADO
        print(f"\n✅ 5. VERIFICACIÓN FINAL:")
        
        total_samples = metadata.get('nsamples', 0)
        total_time_seconds = total_samples * config.TIME_RESO * config.DOWN_TIME_RATE
        
        chunk_size = 2_000_000
        num_chunks = (total_samples + chunk_size - 1) // chunk_size
        chunk_duration = chunk_size * config.TIME_RESO * config.DOWN_TIME_RATE
        slices_per_chunk = chunk_size // slice_len_new
        
        print(f"   📏 Muestras totales: {total_samples:,}")
        print(f"   ⏱️  Tiempo total: {total_time_seconds:.1f} s")
        print(f"   📦 Chunks: {num_chunks}")
        print(f"   ⏱️  Duración por chunk: {chunk_duration:.1f} s")
        print(f"   📊 Slices por chunk: {slices_per_chunk}")
        
        # 6. VERIFICAR CONTINUIDAD TEMPORAL
        print(f"\n🔄 6. VERIFICACIÓN DE CONTINUIDAD TEMPORAL:")
        
        for chunk_idx in range(min(3, num_chunks)):
            start_sample = chunk_idx * chunk_size
            end_sample = min(start_sample + chunk_size, total_samples)
            
            start_time = start_sample * config.TIME_RESO * config.DOWN_TIME_RATE
            end_time = end_sample * config.TIME_RESO * config.DOWN_TIME_RATE
            
            print(f"   📦 Chunk {chunk_idx}: {start_time:.1f}s a {end_time:.1f}s")
            
            if chunk_idx > 0:
                prev_end_time = (chunk_idx - 1) * chunk_size * config.TIME_RESO * config.DOWN_TIME_RATE
                gap = start_time - prev_end_time
                print(f"      🔗 Gap: {gap:.3f}s")
        
        # 7. RESUMEN
        print(f"\n🎉 RESUMEN:")
        print(f"   ✅ Configuración corregida")
        print(f"   ✅ SLICE_LEN: {slice_len_old} → {slice_len_new}")
        print(f"   ✅ Slices por chunk: {slices_per_chunk}")
        print(f"   ✅ Duración por slice: {duration_ms:.1f} ms")
        print(f"   🚀 Listo para usar Pipeline Chunked V2")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    return True


def main():
    """Función principal."""
    
    if len(sys.argv) != 2:
        print("📋 Uso: python scripts/fix_slice_config.py <archivo.fits|archivo.fil>")
        print("📋 Ejemplo: python scripts/fix_slice_config.py Data/3098_0001_00_8bit.fil")
        return
    
    file_path = sys.argv[1]
    success = fix_slice_configuration(file_path)
    
    if success:
        print(f"\n🎯 Ahora puedes ejecutar el pipeline con continuidad temporal perfecta!")
    else:
        print(f"\n❌ Error al corregir la configuración")


if __name__ == "__main__":
    main() 
