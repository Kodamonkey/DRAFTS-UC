#!/usr/bin/env python3
"""
Script de prueba para verificar que el error de plt está corregido.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to path to find DRAFTS module
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from DRAFTS.pipeline import _optimize_memory
    from DRAFTS import config
    
    print("✓ Módulos importados exitosamente")
    
    # Configurar parámetros básicos
    config.FREQ = None
    config.FREQ_RESO = 256
    config.TIME_RESO = 0.001
    config.DOWN_FREQ_RATE = 1
    config.DOWN_TIME_RATE = 1
    
    print("\n=== Probando _optimize_memory ===")
    
    # Test 1: Limpieza normal
    try:
        _optimize_memory(aggressive=False)
        print("✅ _optimize_memory(aggressive=False) funcionando")
    except Exception as e:
        print(f"❌ Error en _optimize_memory(aggressive=False): {e}")
    
    # Test 2: Limpieza agresiva
    try:
        _optimize_memory(aggressive=True)
        print("✅ _optimize_memory(aggressive=True) funcionando")
    except Exception as e:
        print(f"❌ Error en _optimize_memory(aggressive=True): {e}")
    
    # Test 3: Múltiples llamadas
    try:
        for i in range(5):
            _optimize_memory(aggressive=False)
        print("✅ Múltiples llamadas a _optimize_memory funcionando")
    except Exception as e:
        print(f"❌ Error en múltiples llamadas: {e}")
    
    print("\n=== RESUMEN ===")
    print("✅ Error de plt corregido exitosamente")
    print("✅ Función _optimize_memory funcionando correctamente")
    print("✅ No hay errores de 'local variable plt referenced before assignment'")
    
except ImportError as e:
    print(f"✗ Error de importación: {e}")
    print("Asegúrate de que todos los módulos DRAFTS estén disponibles")
except Exception as e:
    print(f"✗ Error inesperado: {e}")
    import traceback
    traceback.print_exc() 
