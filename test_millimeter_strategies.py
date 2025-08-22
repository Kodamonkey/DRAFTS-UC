#!/usr/bin/env python3
"""
Script de prueba para las estrategias de detección en régimen milimétrico (ALMA Band 3)
====================================================================================

Este script prueba la implementación de las estrategias E1 (Expandir DM) y E2 (Pescar en DM≈0)
para mejorar la detección de bursts cuando la dispersión es mínima.

Autor: DRAFTS-MB Team
Fecha: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Añadir el directorio drafts al path
sys.path.insert(0, str(Path(__file__).parent / "drafts"))

from drafts.preprocessing.dm_planner import build_dm_grids, validate_dm_grids
from drafts.validators.dm_validator import DMValidator, ValidationResult
from drafts.output.candidate_manager import Candidate


def test_dm_planner():
    """Prueba el planificador de DM."""
    print("🧪 Probando DM Planner...")
    
    # Parámetros de observación simulados (ALMA Band 3)
    obparams = {
        'freq_low': 84.0,      # MHz (ALMA Band 3)
        'freq_high': 116.0,    # MHz
        'time_resolution': 0.001  # segundos
    }
    
    try:
        # Construir grids de DM
        grid_expand, grid_fish, meta_expand, meta_fish = build_dm_grids(obparams)
        
        print(f"✅ Grid expandido (E1): {len(grid_expand)} DMs, "
              f"rango [{grid_expand.min():.1f}, {grid_expand.max():.1f}] pc cm⁻³")
        print(f"✅ Grid fish (E2): {len(grid_fish)} DMs, "
              f"rango [{grid_fish.min():.1f}, {grid_fish.max():.1f}] pc cm⁻³")
        
        # Validar grids
        validate_dm_grids(grid_expand, grid_fish, meta_expand, meta_fish)
        print("✅ Validación de grids exitosa")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en DM Planner: {e}")
        return False


def test_dm_validator():
    """Prueba el validador DM-aware."""
    print("\n🧪 Probando DM Validator...")
    
    try:
        # Crear validador
        validator = DMValidator()
        print("✅ DMValidator creado exitosamente")
        
        # Simular datos de candidato
        candidate = {
            'id': 'test_candidate_001',
            't0': 1.5,
            'window_start': 100,
            'window_end': 200
        }
        
        # Simular datos de bloque (tiempo, freq)
        np.random.seed(42)  # Para reproducibilidad
        data = np.random.randn(300, 64).astype(np.float32)
        
        # Añadir un "burst" sintético en la ventana del candidato
        burst_window = slice(100, 200)
        data[burst_window, 32] += 5.0  # Pico en el canal central
        
        # Valores de frecuencia simulados
        freq_values = np.linspace(84, 116, 64)  # MHz
        
        # Validar candidato
        result = validator.validate_candidate(
            candidate, data, freq_values, 0.001
        )
        
        print(f"✅ Validación completada: {result.passed}")
        if result.passed:
            print(f"   DM* = {result.dm_star:.1f} ± {result.dm_star_err:.1f} pc cm⁻³")
            print(f"   ΔSNR = {result.delta_snr:.2f}")
            print(f"   Acuerdo sub-bandas = {result.subband_agreement:.1f}%")
        else:
            print(f"   Razón del fallo: {result.reason}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en DM Validator: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_candidate_extensions():
    """Prueba las extensiones de la clase Candidate."""
    print("\n🧪 Probando extensiones de Candidate...")
    
    try:
        # Crear candidato con nuevos campos
        candidate = Candidate(
            file="test_file.fits",
            chunk_id=0,
            slice_id=1,
            band_id=0,
            prob=0.85,
            dm=25.5,
            t_sec=1.5,
            t_sample=1500,
            box=(100, 25, 200, 35),
            snr=8.2,
            class_prob=0.92,
            is_burst=True,
            patch_file="test_patch.png",
            # Nuevos campos DM-aware
            dm_star=28.3,
            dm_star_err=1.2,
            snr_dm0=5.1,
            snr_dmstar=8.2,
            delta_snr=3.1,
            subband_agreement=85.5,
            validation_passed=True,
            validation_reason=None,
            strategy="E2_fish"
        )
        
        print("✅ Candidate creado con extensiones DM-aware")
        
        # Probar métodos nuevos
        priority_score = candidate.calculate_priority_score()
        validation_summary = candidate.get_validation_summary()
        
        print(f"   Score de prioridad: {priority_score:.3f}")
        print(f"   Resumen de validación: {validation_summary}")
        
        # Probar serialización a CSV
        csv_row = candidate.to_row()
        print(f"   Fila CSV generada: {len(csv_row)} campos")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en extensiones de Candidate: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Prueba la integración de todos los componentes."""
    print("\n🧪 Probando integración completa...")
    
    try:
        # Simular flujo completo
        print("1. Construyendo grids de DM...")
        obparams = {'freq_low': 84.0, 'freq_high': 116.0, 'time_resolution': 0.001}
        grid_expand, grid_fish, meta_expand, meta_fish = build_dm_grids(obparams)
        
        print("2. Inicializando validador...")
        validator = DMValidator()
        
        print("3. Simulando detección y validación...")
        # Simular candidato detectado con E2
        candidate = {
            'id': 'integration_test_001',
            't0': 2.0,
            'window_start': 150,
            'window_end': 250
        }
        
        # Simular datos con burst
        data = np.random.randn(400, 64).astype(np.float32)
        burst_window = slice(150, 250)
        data[burst_window, 32] += 6.0
        
        freq_values = np.linspace(84, 116, 64)
        
        # Validar candidato
        result = validator.validate_candidate(candidate, data, freq_values, 0.001)
        
        print("4. Creando candidato final...")
        final_candidate = Candidate(
            file="integration_test.fits",
            chunk_id=0,
            slice_id=2,
            band_id=0,
            prob=0.78,
            dm=result.dm_star,
            t_sec=candidate['t0'],
            t_sample=int(candidate['t0'] / 0.001),
            box=(150, 30, 250, 40),
            snr=result.snr_dmstar,
            class_prob=None,
            is_burst=None,
            patch_file=None,
            dm_star=result.dm_star,
            dm_star_err=result.dm_star_err,
            snr_dm0=result.snr_dm0,
            snr_dmstar=result.snr_dmstar,
            delta_snr=result.delta_snr,
            subband_agreement=result.subband_agreement,
            validation_passed=result.passed,
            validation_reason=result.reason,
            strategy="E2_fish"
        )
        
        print("✅ Integración completa exitosa!")
        print(f"   Candidato final: {final_candidate.get_validation_summary()}")
        print(f"   Score de prioridad: {final_candidate.calculate_priority_score():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en integración: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Función principal de prueba."""
    print("🚀 Iniciando pruebas de estrategias milimétricas...\n")
    
    tests = [
        ("DM Planner", test_dm_planner),
        ("DM Validator", test_dm_validator),
        ("Candidate Extensions", test_candidate_extensions),
        ("Integration", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error inesperado en {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen de resultados
    print("\n" + "="*60)
    print("📊 RESUMEN DE PRUEBAS")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print("-"*60)
    print(f"Total: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron! Las estrategias milimétricas están listas.")
    else:
        print("⚠️  Algunas pruebas fallaron. Revisar errores antes de usar en producción.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
