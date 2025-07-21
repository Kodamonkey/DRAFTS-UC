#!/usr/bin/env python3
"""
Test para simular exactamente el problema reportado por el usuario:
Chunk 1 termina en 111.54s y Chunk 2 parte en 1528.41s (gap de ~1417s)
"""

import sys
from pathlib import Path
import numpy as np

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRAFTS import config


def analyze_reported_issue():
    """Analizar el problema específico reportado por el usuario."""
    
    print("🔍 ANÁLISIS DEL PROBLEMA REPORTADO")
    print("=" * 60)
    print("Usuario reporta: Chunk 1 termina en 111.54s, Chunk 2 parte en 1528.41s")
    print("Gap detectado: ~1417 segundos")
    print()
    
    # Intentar recrear el problema con diferentes configuraciones
    scenarios = [
        {
            "name": "Configuración típica",
            "TIME_RESO": 0.000001,
            "DOWN_TIME_RATE": 14,
            "chunk_size": 2_000_000,
            "overlap": 1000
        },
        {
            "name": "Configuración alta resolución",
            "TIME_RESO": 0.0000001,
            "DOWN_TIME_RATE": 1,
            "chunk_size": 2_000_000,
            "overlap": 1000
        },
        {
            "name": "Configuración con decimación alta",
            "TIME_RESO": 0.000001,
            "DOWN_TIME_RATE": 100,
            "chunk_size": 2_000_000,
            "overlap": 1000
        },
        {
            "name": "Configuración con chunk grande",
            "TIME_RESO": 0.000001,
            "DOWN_TIME_RATE": 14,
            "chunk_size": 10_000_000,
            "overlap": 1000
        }
    ]
    
    for scenario in scenarios:
        print(f"📊 ESCENARIO: {scenario['name']}")
        print("-" * 40)
        
        # Configurar parámetros
        config.TIME_RESO = scenario["TIME_RESO"]
        config.DOWN_TIME_RATE = scenario["DOWN_TIME_RATE"]
        chunk_size = scenario["chunk_size"]
        overlap = scenario["overlap"]
        
        # Calcular chunks
        effective_chunk_size = chunk_size - overlap
        start_sample_1 = 0
        end_sample_1 = chunk_size
        start_sample_2 = effective_chunk_size
        end_sample_2 = start_sample_2 + chunk_size
        
        # Calcular tiempos
        time_1_start = start_sample_1 * config.TIME_RESO * config.DOWN_TIME_RATE
        time_1_end = end_sample_1 * config.TIME_RESO * config.DOWN_TIME_RATE
        time_2_start = start_sample_2 * config.TIME_RESO * config.DOWN_TIME_RATE
        time_2_end = end_sample_2 * config.TIME_RESO * config.DOWN_TIME_RATE
        
        gap = time_2_start - time_1_end
        
        print(f"   ⏱️  TIME_RESO: {config.TIME_RESO}")
        print(f"   🔽 DOWN_TIME_RATE: {config.DOWN_TIME_RATE}")
        print(f"   📏 Chunk size: {chunk_size:,}")
        print(f"   📏 Effective chunk size: {effective_chunk_size:,}")
        print()
        print(f"   Chunk 1: {start_sample_1:,} a {end_sample_1:,}")
        print(f"   Chunk 2: {start_sample_2:,} a {end_sample_2:,}")
        print()
        print(f"   🕐 Chunk 1: {time_1_start:.3f}s a {time_1_end:.3f}s")
        print(f"   🕐 Chunk 2: {time_2_start:.3f}s a {time_2_end:.3f}s")
        print(f"   🔗 Gap: {gap:.3f}s")
        
        # Verificar si coincide con el problema reportado
        if abs(time_1_end - 111.54) < 1.0 and abs(time_2_start - 1528.41) < 1.0:
            print(f"   🎯 ¡COINCIDENCIA ENCONTRADA!")
            print(f"   ✅ Este escenario reproduce el problema reportado")
        elif abs(gap - 1417) < 10:
            print(f"   🎯 ¡GAP SIMILAR ENCONTRADO!")
            print(f"   ✅ Gap de {gap:.3f}s similar al reportado ({1417}s)")
        else:
            print(f"   ❌ No coincide con el problema reportado")
        print()


def analyze_configuration_issues():
    """Analizar posibles problemas de configuración."""
    
    print("\n🔍 ANÁLISIS DE POSIBLES PROBLEMAS DE CONFIGURACIÓN")
    print("=" * 60)
    
    # Problema 1: Configuración incorrecta de TIME_RESO
    print("1️⃣ PROBLEMA: TIME_RESO incorrecto")
    print("-" * 30)
    print("Si TIME_RESO está mal configurado, los tiempos serán incorrectos.")
    print("Ejemplo: TIME_RESO = 0.001 en lugar de 0.000001")
    print()
    
    # Problema 2: DOWN_TIME_RATE incorrecto
    print("2️⃣ PROBLEMA: DOWN_TIME_RATE incorrecto")
    print("-" * 30)
    print("Si DOWN_TIME_RATE no coincide con la decimación real, habrá saltos.")
    print("Ejemplo: DOWN_TIME_RATE = 1 pero datos están decimados por 14")
    print()
    
    # Problema 3: Cálculo incorrecto de start_sample_global
    print("3️⃣ PROBLEMA: start_sample_global incorrecto")
    print("-" * 30)
    print("Si start_sample_global no se calcula correctamente, los tiempos serán erróneos.")
    print("Esto puede pasar si hay confusión entre muestras originales y decimadas.")
    print()
    
    # Problema 4: Problema en _load_fil_chunk
    print("4️⃣ PROBLEMA: _load_fil_chunk incorrecto")
    print("-" * 30)
    print("Si _load_fil_chunk no lee las muestras correctas, los tiempos serán erróneos.")
    print("Esto puede pasar si el offset en el archivo no se calcula correctamente.")
    print()


def suggest_solutions():
    """Sugerir soluciones para el problema."""
    
    print("\n💡 SOLUCIONES SUGERIDAS")
    print("=" * 60)
    
    print("1️⃣ VERIFICAR CONFIGURACIÓN:")
    print("   - Revisar TIME_RESO en el archivo de datos")
    print("   - Verificar DOWN_TIME_RATE vs decimación real")
    print("   - Confirmar que los parámetros coinciden con los datos")
    print()
    
    print("2️⃣ AGREGAR DEBUG TEMPORAL:")
    print("   - Agregar logs detallados de timing en _process_single_chunk")
    print("   - Verificar que chunk_start_time_sec se calcula correctamente")
    print("   - Confirmar que se pasa correctamente a las funciones de visualización")
    print()
    
    print("3️⃣ VERIFICAR _load_fil_chunk:")
    print("   - Confirmar que lee las muestras correctas del archivo")
    print("   - Verificar el cálculo del offset en el archivo")
    print("   - Asegurar que no hay pérdida de muestras")
    print()
    
    print("4️⃣ COMPARAR CON PIPELINE TRADICIONAL:")
    print("   - Ejecutar el mismo archivo con pipeline tradicional")
    print("   - Comparar los tiempos generados")
    print("   - Identificar dónde divergen los resultados")
    print()


def main():
    """Función principal."""
    analyze_reported_issue()
    analyze_configuration_issues()
    suggest_solutions()


if __name__ == "__main__":
    main() 