#!/usr/bin/env python3
"""
Ejemplo práctico de uso del sistema automático de parámetros.

Este script demuestra cómo usar el pipeline con cálculo automático de parámetros
basado únicamente en SLICE_DURATION_MS.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def ejemplo_uso_basico():
    """Ejemplo básico de uso del sistema automático."""
    
    print("🚀 EJEMPLO DE USO DEL SISTEMA AUTOMÁTICO")
    print("=" * 50)
    print()
    
    print("1️⃣ CONFIGURACIÓN MÍNIMA EN user_config.py:")
    print("""
# Solo necesitas configurar esto:
SLICE_DURATION_MS: float = 64.0  # Duración deseada de cada slice en ms

# El resto se calcula automáticamente:
# - SLICE_LEN (muestras por slice)
# - chunk_samples (tamaño óptimo de chunk)
# - slices_per_chunk
# - etc.
""")
    
    print("2️⃣ EJECUCIÓN DEL PIPELINE:")
    print("""
# Modo automático (recomendado):
python main.py

# O explícitamente:
from drafts.pipeline import run_pipeline
run_pipeline(chunk_samples=0)  # 0 = cálculo automático
""")
    
    print("3️⃣ EL SISTEMA CALCULA AUTOMÁTICAMENTE:")
    print("""
✅ SLICE_LEN basado en SLICE_DURATION_MS y TIME_RESO
✅ chunk_samples optimizado para memoria y rendimiento
✅ slices_per_chunk para evitar fragmentación
✅ Validación automática de todos los parámetros
✅ Logs informativos del proceso
""")

def ejemplo_configuraciones_tipicas():
    """Muestra configuraciones típicas para diferentes casos de uso."""
    
    print("\n📋 CONFIGURACIONES TÍPICAS")
    print("=" * 50)
    
    configs = [
        {
            'caso': 'Detección de FRB rápidos',
            'SLICE_DURATION_MS': 32.0,
            'descripcion': 'Slices cortos para capturar pulsos muy rápidos'
        },
        {
            'caso': 'Detección general de FRB',
            'SLICE_DURATION_MS': 64.0,
            'descripcion': 'Balance entre sensibilidad y velocidad'
        },
        {
            'caso': 'Análisis de pulsos largos',
            'SLICE_DURATION_MS': 128.0,
            'descripcion': 'Slices más largos para pulsos extendidos'
        },
        {
            'caso': 'Búsqueda de señales débiles',
            'SLICE_DURATION_MS': 256.0,
            'descripcion': 'Mayor integración temporal para mejor SNR'
        }
    ]
    
    for config in configs:
        print(f"\n🔧 {config['caso']}")
        print(f"   SLICE_DURATION_MS = {config['SLICE_DURATION_MS']} ms")
        print(f"   {config['descripcion']}")

def ejemplo_optimizacion_memoria():
    """Muestra cómo el sistema optimiza la memoria automáticamente."""
    
    print("\n💾 OPTIMIZACIÓN AUTOMÁTICA DE MEMORIA")
    print("=" * 50)
    
    print("""
El sistema considera automáticamente:

🖥️  Memoria disponible del sistema
📊 Tamaño del archivo de datos
⚡ Resolución temporal y frecuencial
🎯 Factor de decimado aplicado
📈 Número de canales de frecuencia

Y calcula el chunk_size óptimo que:
✅ No exceda el 25% de memoria disponible
✅ Contenga entre 50-1000 slices por chunk
✅ Sea múltiplo del slice_len
✅ Optimice el rendimiento de I/O
""")

def ejemplo_logs_salida():
    """Muestra ejemplos de los logs que genera el sistema."""
    
    print("\n📝 EJEMPLO DE LOGS DEL SISTEMA")
    print("=" * 50)
    
    print("""
Cuando ejecutes el pipeline, verás logs como:

✅ Parámetros calculados automáticamente:
   • Slice: 512 muestras (64.0 ms)
   • Chunk: 128,000 muestras (45.2s)
   • Slices por chunk: 250
   • Total estimado: 78 chunks, 19,531 slices

🔧 Archivo de alta resolución temporal
   TIME_RESO: 0.0001s
   FREQ_RESO: 1024 canales
   FILE_LENG: 10,000,000 muestras
   DOWN_TIME_RATE: 1
   DOWN_FREQ_RATE: 1

✅ SLICE_DURATION_MS=64.0ms:
   • Slice: 640 muestras (64.0ms)
   • Chunk: 160,000 muestras (16.0s)
   • Slices/chunk: 250
   • Total: 63 chunks, 15,625 slices
""")

def main():
    """Función principal del ejemplo."""
    
    ejemplo_uso_basico()
    ejemplo_configuraciones_tipicas()
    ejemplo_optimizacion_memoria()
    ejemplo_logs_salida()
    
    print("\n" + "=" * 50)
    print("🎉 ¡LISTO! Tu pipeline ahora es mucho más simple de usar.")
    print("\n💡 VENTAJAS DEL NUEVO SISTEMA:")
    print("• Solo configuras SLICE_DURATION_MS")
    print("• Optimización automática de memoria")
    print("• Cálculo inteligente de chunk_size")
    print("• Validación automática de parámetros")
    print("• Logs informativos detallados")
    print("• Compatible con archivos de diferentes tamaños")

if __name__ == "__main__":
    main() 