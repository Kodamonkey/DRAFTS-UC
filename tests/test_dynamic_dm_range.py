#!/usr/bin/env python3
"""
Script de prueba para verificar el funcionamiento del rango DM dinámico 
en plots de detección y composite.
"""

import sys
import numpy as np
from pathlib import Path

# Agregar el directorio DRAFTS al path
sys.path.insert(0, str(Path(__file__).parent / "DRAFTS"))

from DRAFTS import config
from DRAFTS.image_utils import _calculate_dynamic_dm_range
from DRAFTS.dynamic_dm_range import get_dynamic_dm_range_for_candidate

def test_dynamic_dm_range():
    """Test del cálculo de rango DM dinámico."""
    
    print("=== Test de Rango DM Dinámico ===")
    print(f"DM_DYNAMIC_RANGE_ENABLE: {config.DM_DYNAMIC_RANGE_ENABLE}")
    print(f"DM_RANGE_FACTOR: {config.DM_RANGE_FACTOR}")
    print(f"DM_RANGE_MIN_WIDTH: {config.DM_RANGE_MIN_WIDTH}")
    print(f"DM_RANGE_MAX_WIDTH: {config.DM_RANGE_MAX_WIDTH}")
    print(f"DM_PLOT_MARGIN_FACTOR: {config.DM_PLOT_MARGIN_FACTOR}")
    
    # Simular candidatos detectados con diferentes DMs
    test_cases = [
        {
            "name": "Candidato DM bajo",
            "dm_optimal": 50.0,
            "confidence": 0.9,
        },
        {
            "name": "Candidato DM medio",
            "dm_optimal": 200.0,
            "confidence": 0.8,
        },
        {
            "name": "Candidato DM alto",
            "dm_optimal": 500.0,
            "confidence": 0.95,
        },
        {
            "name": "Candidato DM muy alto",
            "dm_optimal": 800.0,
            "confidence": 0.7,
        }
    ]
    
    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        print(f"DM óptimo: {case['dm_optimal']:.1f} pc cm⁻³")
        print(f"Confianza: {case['confidence']:.2f}")
        
        try:
            # Test usando la función directa
            dm_min, dm_max = get_dynamic_dm_range_for_candidate(
                dm_optimal=case['dm_optimal'],
                config_module=config,
                visualization_type='detailed',
                confidence=case['confidence']
            )
            
            range_width = dm_max - dm_min
            center_dm = (dm_min + dm_max) / 2
            candidate_position = (case['dm_optimal'] - dm_min) / range_width
            
            print(f"Rango calculado: {dm_min:.1f} - {dm_max:.1f} pc cm⁻³")
            print(f"Ancho del rango: {range_width:.1f} pc cm⁻³")
            print(f"Centro del rango: {center_dm:.1f} pc cm⁻³")
            print(f"Posición del candidato: {candidate_position:.1%} del rango")
            
            # Verificar que el candidato no esté en los bordes
            if candidate_position < 0.1:
                print("⚠️  ADVERTENCIA: Candidato muy cerca del borde inferior")
            elif candidate_position > 0.9:
                print("⚠️  ADVERTENCIA: Candidato muy cerca del borde superior") 
            else:
                print("✅ Candidato bien centrado en el rango")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test con múltiples candidatos
    print(f"\n--- Test con múltiples candidatos ---")
    
    # Simular múltiples bounding boxes
    top_boxes = [
        [100, 150, 120, 170],  # DM ~200
        [200, 100, 220, 120],  # DM ~140  
        [150, 200, 170, 220],  # DM ~270
    ]
    
    top_conf = [0.9, 0.7, 0.85]
    slice_len = 32
    
    try:
        dm_min, dm_max = _calculate_dynamic_dm_range(
            top_boxes=top_boxes,
            slice_len=slice_len,
            fallback_dm_min=config.DM_min,
            fallback_dm_max=config.DM_max,
            confidence_scores=top_conf
        )
        
        print(f"Candidatos múltiples:")
        print(f"Rango calculado: {dm_min:.1f} - {dm_max:.1f} pc cm⁻³")
        print(f"Ancho del rango: {dm_max - dm_min:.1f} pc cm⁻³")
        print("✅ Cálculo exitoso para múltiples candidatos")
        
    except Exception as e:
        print(f"❌ Error con múltiples candidatos: {e}")

if __name__ == "__main__":
    test_dynamic_dm_range()
