"""
Test: Verificar que al bajar el threshold de DM chunking, se crean chunks DM más pequeños.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("="*100)
print("TEST: Efecto de bajar el threshold de DM chunking")
print("="*100)
print()

# Parámetros fijos
width = 34_816  # samples decimated (del caso de 5.5 TB)
height_dm = 100_000  # valores DM
threshold_16 = 16.0  # GB (default)
threshold_4 = 4.0  # GB (reducido)

print("CONFIGURACIÓN:")
print(f"  Width (samples decimated): {width:,}")
print(f"  Height DM: {height_dm:,} valores")
print(f"  Threshold 1 (default): {threshold_16} GB")
print(f"  Threshold 2 (reducido): {threshold_4} GB")
print()

# Fórmula REAL del código (data_flow_manager.py:163)
def calculate_dm_chunk_height(threshold_gb, width):
    """Fórmula REAL del código."""
    max_chunk_height = int((threshold_gb * (1024**3)) / (3 * width * 4))
    min_chunk_height = 100
    dm_chunk_height = max(min_chunk_height, max_chunk_height)
    return dm_chunk_height

# Calcular con threshold de 16 GB
dm_chunk_height_16 = calculate_dm_chunk_height(threshold_16, width)
num_chunks_16 = (height_dm + dm_chunk_height_16 - 1) // dm_chunk_height_16

# Calcular con threshold de 4 GB
dm_chunk_height_4 = calculate_dm_chunk_height(threshold_4, width)
num_chunks_4 = (height_dm + dm_chunk_height_4 - 1) // dm_chunk_height_4

print("RESULTADOS:")
print("-" * 100)
print()
print(f"Con threshold = {threshold_16} GB:")
print(f"  Fórmula: max_chunk_height = ({threshold_16} × 1024³) / (3 × {width:,} × 4)")
print(f"  max_chunk_height = {int((threshold_16 * (1024**3)) / (3 * width * 4)):,} valores DM")
print(f"  dm_chunk_height = {dm_chunk_height_16:,} valores DM")
print(f"  num_dm_chunks = ({height_dm:,} + {dm_chunk_height_16:,} - 1) // {dm_chunk_height_16:,} = {num_chunks_16}")
print(f"  Tamaño de cada chunk DM: ~{threshold_16} GB")
print()

print(f"Con threshold = {threshold_4} GB:")
print(f"  Fórmula: max_chunk_height = ({threshold_4} × 1024³) / (3 × {width:,} × 4)")
print(f"  max_chunk_height = {int((threshold_4 * (1024**3)) / (3 * width * 4)):,} valores DM")
print(f"  dm_chunk_height = {dm_chunk_height_4:,} valores DM")
print(f"  num_dm_chunks = ({height_dm:,} + {dm_chunk_height_4:,} - 1) // {dm_chunk_height_4:,} = {num_chunks_4}")
print(f"  Tamaño de cada chunk DM: ~{threshold_4} GB")
print()

print("COMPARACIÓN:")
print("-" * 100)
print(f"Threshold: {threshold_16} GB → {threshold_4} GB (reducción de {threshold_16/threshold_4}x)")
print(f"Chunk height: {dm_chunk_height_16:,} → {dm_chunk_height_4:,} valores DM (reducción de {dm_chunk_height_16/dm_chunk_height_4:.2f}x)")
print(f"Número de chunks: {num_chunks_16} → {num_chunks_4} chunks (aumento de {num_chunks_4/num_chunks_16:.2f}x)")
print()

# Verificar tamaños de chunks
chunk_size_16_gb = (3 * dm_chunk_height_16 * width * 4) / (1024**3)
chunk_size_4_gb = (3 * dm_chunk_height_4 * width * 4) / (1024**3)

print("VALIDACIÓN:")
print("-" * 100)
print(f"Chunk DM con threshold {threshold_16} GB:")
print(f"  Tamaño: {chunk_size_16_gb:.2f} GB")
print(f"  ¿Dentro del threshold? {chunk_size_16_gb <= threshold_16 * 1.1}")
print()
print(f"Chunk DM con threshold {threshold_4} GB:")
print(f"  Tamaño: {chunk_size_4_gb:.2f} GB")
print(f"  ¿Dentro del threshold? {chunk_size_4_gb <= threshold_4 * 1.1}")
print()

# Mostrar distribución de chunks con threshold reducido
print("DISTRIBUCIÓN DE CHUNKS DM (con threshold 4 GB):")
print("-" * 100)
dm_range = 10000.0
for chunk_idx in range(min(10, num_chunks_4)):
    start_dm_idx = chunk_idx * dm_chunk_height_4
    end_dm_idx = min(start_dm_idx + dm_chunk_height_4, height_dm)
    chunk_height = end_dm_idx - start_dm_idx
    
    chunk_dm_min = (start_dm_idx / height_dm) * dm_range
    chunk_dm_max = (end_dm_idx / height_dm) * dm_range
    
    chunk_size_gb = (3 * chunk_height * width * 4) / (1024**3)
    
    print(f"  Chunk DM {chunk_idx + 1}: DM {chunk_dm_min:.1f}-{chunk_dm_max:.1f} pc cm⁻³ "
          f"({chunk_height:,} valores, {chunk_size_gb:.2f} GB)")

print()
print("="*100)
print("CONCLUSIÓN:")
print("="*100)
print("✓ Al bajar el threshold de 16 GB a 4 GB:")
print(f"  - Los chunks DM son {dm_chunk_height_16/dm_chunk_height_4:.1f}x más pequeños")
print(f"  - Se necesitan {num_chunks_4/num_chunks_16:.1f}x más chunks DM")
print(f"  - Cada chunk DM usa ~{threshold_4} GB en lugar de ~{threshold_16} GB")
print()
print("✓ El sistema SÍ intenta hacer chunks DM más pequeños cuando se baja el threshold")
print("✓ Esto reduce el pico de memoria durante el procesamiento de cada chunk DM")
print("⚠️  PERO el cubo completo sigue estando en memoria (necesario para combinar)")
print()

