#!/usr/bin/env python3
"""
Prueba simple de la funcionalidad dinámica de SLICE_LEN.
"""

import sys
from pathlib import Path

# Configurar path
sys.path.append(str(Path(__file__).parent))

def test_dynamic_slice_len():
    """Prueba la funcionalidad dinámica."""
    
    print("🧪 === PRUEBA SLICE_LEN DINÁMICO ===\n")
    
    # Simular configuración
    class MockConfig:
        SLICE_LEN_AUTO = True
        SLICE_DURATION_SECONDS = 0.032  # 32ms
        SLICE_LEN_MIN = 16
        SLICE_LEN_MAX = 512
        TIME_RESO = 0.001  # 1ms
        DOWN_TIME_RATE = 1
        SLICE_LEN = 64  # Valor manual de respaldo
    
    config = MockConfig()
    
    # Importar función
    try:
        from DRAFTS.slice_len_utils import calculate_optimal_slice_len, get_dynamic_slice_len
        
        print("✅ Módulo importado correctamente")
        
        # Prueba 1: Cálculo manual
        print("\n🔸 PRUEBA 1: Cálculo manual")
        slice_len, duration, explanation = calculate_optimal_slice_len(
            time_reso=0.001,
            down_time_rate=1,
            target_duration_seconds=0.032,
            min_slice_len=16,
            max_slice_len=512
        )
        print(f"   Objetivo: 32ms → SLICE_LEN: {slice_len} → Real: {duration*1000:.1f}ms")
        print(f"   Explicación: {explanation}")
        
        # Prueba 2: Usando configuración
        print("\n🔸 PRUEBA 2: Usando configuración dinámica")
        dynamic_slice_len = get_dynamic_slice_len(config)
        print(f"   SLICE_LEN dinámico: {dynamic_slice_len}")
        
        # Prueba 3: Diferentes duraciones
        print("\n🔸 PRUEBA 3: Diferentes duraciones objetivo")
        test_durations = [0.016, 0.032, 0.064, 0.128]  # 16, 32, 64, 128 ms
        
        for target_dur in test_durations:
            slice_len, actual_dur, _ = calculate_optimal_slice_len(
                time_reso=0.001,
                target_duration_seconds=target_dur
            )
            print(f"   {target_dur*1000:3.0f}ms → SLICE_LEN: {slice_len:3d} → {actual_dur*1000:5.1f}ms")
        
        print("\n✅ Todas las pruebas completadas exitosamente")
        
    except ImportError as e:
        print(f"❌ Error importando módulo: {e}")
    except Exception as e:
        print(f"❌ Error en prueba: {e}")

if __name__ == "__main__":
    test_dynamic_slice_len()
