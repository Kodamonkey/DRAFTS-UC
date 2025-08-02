"""
Demostración del nuevo sistema de logging detallado
==================================================

Este script simula diferentes escenarios de procesamiento para mostrar
el nuevo sistema de logging informativo que explica exactamente qué está
pasando cuando se hace un ajuste o se salta un slice.
"""

import numpy as np
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tests/demo_logging_detallado.log')
    ]
)

def simular_procesamiento_chunk():
    """Simula el procesamiento de un chunk con diferentes escenarios."""
    
    # Parámetros simulados
    chunk_idx = 0
    slice_len = 512
    block_shape = 12795  # Como en tu caso
    chunk_start_time_sec = 0.0
    config_time_reso = 0.001
    config_down_time_rate = 8
    
    print("=" * 80)
    print("DEMOSTRACIÓN DEL NUEVO SISTEMA DE LOGGING DETALLADO")
    print("=" * 80)
    
    # Escenario 1: Slice normal (sin ajuste)
    print("\n📊 ESCENARIO 1: Slice normal (sin ajuste)")
    print("-" * 50)
    
    j = 0  # Primer slice
    start_idx = slice_len * j
    end_idx = slice_len * (j + 1)
    
    slice_info = {
        'slice_idx': j,
        'slice_len': slice_len,
        'start_idx': start_idx,
        'end_idx_calculado': end_idx,
        'block_shape': block_shape,
        'chunk_idx': chunk_idx,
        'tiempo_absoluto_inicio': chunk_start_time_sec + (j * slice_len * config_time_reso * config_down_time_rate),
        'duracion_slice_esperada_ms': slice_len * config_time_reso * config_down_time_rate * 1000
    }
    
    print(f"📊 PROCESANDO SLICE {j} (chunk {chunk_idx}):")
    print(f"   • Rango de muestras: [{start_idx} → {end_idx}] ({end_idx - start_idx} muestras)")
    print(f"   • Tiempo absoluto: {slice_info['tiempo_absoluto_inicio']:.3f}s")
    print(f"   • Duración esperada: {slice_info['duracion_slice_esperada_ms']:.1f}ms")
    print(f"   • Shape del slice: (DM_range, freq_channels, {end_idx - start_idx})")
    print(f"   • Datos del waterfall: ({end_idx - start_idx}, freq_channels)")
    
    # Escenario 2: Slice con ajuste (como en tu caso)
    print("\n🔧 ESCENARIO 2: Slice con ajuste (como en tu caso)")
    print("-" * 50)
    
    j = 18  # Slice 18
    start_idx = slice_len * j
    end_idx = slice_len * (j + 1)
    
    # Simular el caso donde end_idx > block_shape
    end_idx_original = end_idx
    end_idx = block_shape  # Ajuste
    
    muestras_esperadas = end_idx_original - start_idx
    muestras_disponibles = block_shape - start_idx
    porcentaje_ajuste = ((end_idx_original - block_shape) / (end_idx_original - start_idx)) * 100
    
    print(f"🔧 AJUSTANDO SLICE {j} (chunk {chunk_idx}):")
    print(f"   • end_idx calculado ({end_idx_original}) > block.shape[0] ({block_shape})")
    print(f"   • Muestras esperadas: {muestras_esperadas}")
    print(f"   • Muestras disponibles: {muestras_disponibles}")
    print(f"   • Ajuste necesario: {end_idx_original - block_shape} muestras ({porcentaje_ajuste:.1f}%)")
    print(f"   • Tiempo absoluto: {chunk_start_time_sec + (j * slice_len * config_time_reso * config_down_time_rate):.3f}s")
    print(f"   • Duración esperada: {slice_len * config_time_reso * config_down_time_rate * 1000:.1f}ms")
    print(f"   • Razón: Último slice del chunk con datos residuales")
    
    # Después del ajuste
    slice_tiempo_real_ms = (end_idx - start_idx) * config_time_reso * config_down_time_rate * 1000
    print(f"\n📊 PROCESANDO SLICE {j} (chunk {chunk_idx}) DESPUÉS DEL AJUSTE:")
    print(f"   • Rango de muestras: [{start_idx} → {end_idx}] ({end_idx - start_idx} muestras)")
    print(f"   • Tiempo absoluto: {chunk_start_time_sec + (j * slice_len * config_time_reso * config_down_time_rate):.3f}s")
    print(f"   • Duración real: {slice_tiempo_real_ms:.1f}ms")
    print(f"   • Shape del slice: (DM_range, freq_channels, {end_idx - start_idx})")
    print(f"   • Datos del waterfall: ({end_idx - start_idx}, freq_channels)")
    
    # Escenario 3: Slice muy pequeño (se salta)
    print("\n🚫 ESCENARIO 3: Slice muy pequeño (se salta)")
    print("-" * 50)
    
    j = 25  # Slice que sería muy pequeño
    start_idx = slice_len * j
    end_idx_original = slice_len * (j + 1)
    end_idx = block_shape  # Ajuste
    
    if end_idx - start_idx < slice_len // 2:
        print(f"🚫 SALTANDO SLICE {j} (chunk {chunk_idx}) - MUY PEQUEÑO:")
        print(f"   • Tamaño después del ajuste: {end_idx - start_idx} muestras")
        print(f"   • Tamaño mínimo requerido: {slice_len // 2} muestras")
        print(f"   • Porcentaje del tamaño esperado: {((end_idx - start_idx) / slice_len) * 100:.1f}%")
        print(f"   • Tiempo absoluto: {chunk_start_time_sec + (j * slice_len * config_time_reso * config_down_time_rate):.3f}s")
        print(f"   • Razón: Slice demasiado pequeño para procesamiento efectivo")
    
    # Escenario 4: Slice fuera de límites
    print("\n🚫 ESCENARIO 4: Slice fuera de límites")
    print("-" * 50)
    
    j = 30  # Slice que está fuera de límites
    start_idx = slice_len * j
    
    if start_idx >= block_shape:
        print(f"🚫 SALTANDO SLICE {j} (chunk {chunk_idx}):")
        print(f"   • start_idx ({start_idx}) >= block.shape[0] ({block_shape})")
        print(f"   • Slice fuera de límites - no hay datos que procesar")
        print(f"   • Tiempo absoluto: {chunk_start_time_sec + (j * slice_len * config_time_reso * config_down_time_rate):.3f}s")
        print(f"   • Duración esperada: {slice_len * config_time_reso * config_down_time_rate * 1000:.1f}ms")
        print(f"   • Datos disponibles en bloque: {block_shape} muestras")
        print(f"   • Razón: El slice empieza después del final del bloque")

def simular_logging_chunk():
    """Simula el logging informativo de un chunk completo."""
    
    print("\n🔄 SIMULACIÓN DE LOGGING DE CHUNK COMPLETO")
    print("=" * 80)
    
    # Metadatos simulados
    metadata = {
        'actual_chunk_size': 12795,
        'total_samples': 1000000,
        'start_sample': 0,
        'end_sample': 12795
    }
    
    chunk_idx = 0
    chunk_start_time_sec = metadata["start_sample"] * 0.001
    chunk_duration_sec = metadata["actual_chunk_size"] * 0.001
    
    print(f"🔄 INICIANDO CHUNK {chunk_idx:03d}:")
    print(f"   • Muestras en chunk: {metadata['actual_chunk_size']:,} / {metadata['total_samples']:,} totales")
    print(f"   • Rango de muestras: [{metadata['start_sample']:,} → {metadata['end_sample']:,}]")
    print(f"   • Tiempo absoluto: {chunk_start_time_sec:.3f}s → {chunk_start_time_sec + chunk_duration_sec:.3f}s")
    print(f"   • Duración del chunk: {chunk_duration_sec:.2f}s")
    print(f"   • Progreso del archivo: {(metadata['start_sample'] / metadata['total_samples']) * 100:.1f}%")

def main():
    """Ejecutar todas las simulaciones."""
    print("Iniciando demostración del nuevo sistema de logging...")
    
    # Crear directorio de logs si no existe
    Path("tests").mkdir(exist_ok=True)
    
    # Simular diferentes escenarios
    simular_procesamiento_chunk()
    simular_logging_chunk()
    
    print("\n" + "=" * 80)
    print("DEMOSTRACIÓN COMPLETADA")
    print("=" * 80)
    print("Revisa el archivo 'tests/demo_logging_detallado.log' para ver los logs detallados.")
    print("\nBENEFICIOS DEL NUEVO SISTEMA:")
    print("✅ Muestra exactamente qué datos se están procesando")
    print("✅ Explica por qué se hace cada ajuste o salto")
    print("✅ Proporciona contexto temporal completo")
    print("✅ Incluye métricas de rendimiento")
    print("✅ Facilita el debugging y monitoreo")

if __name__ == "__main__":
    main() 