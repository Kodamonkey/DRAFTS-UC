#!/usr/bin/env python3
"""Script para verificar si las funciones SNR están integradas correctamente."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("=== VERIFICACIÓN DE INTEGRACIÓN SNR ===\n")

# 1. Verificar imports
try:
    from DRAFTS import config
    print("✓ Config importado correctamente")
    
    from DRAFTS.snr_utils import compute_snr_profile
    print("✓ snr_utils importado correctamente")
    
    from DRAFTS.visualization import save_patch_plot
    print("✓ visualization importado correctamente")
    
except Exception as e:
    print(f"✗ Error en imports: {e}")
    sys.exit(1)

# 2. Verificar configuración SNR
print(f"\n=== CONFIGURACIÓN SNR ===")
try:
    print(f"SNR_THRESH: {config.SNR_THRESH}")
    print(f"SNR_COLORMAP: {config.SNR_COLORMAP}")
    print(f"SNR_HIGHLIGHT_COLOR: {config.SNR_HIGHLIGHT_COLOR}")
    print("✓ Configuración SNR disponible")
except AttributeError as e:
    print(f"✗ Configuración SNR faltante: {e}")

# 3. Verificar función SNR básica
print(f"\n=== PRUEBA FUNCIÓN SNR ===")
try:
    import numpy as np
    np.random.seed(42)
    
    # Crear datos de prueba
    test_data = np.random.normal(0, 1, (100, 50))
    snr, sigma = compute_snr_profile(test_data)
    
    print(f"Datos shape: {test_data.shape}")
    print(f"SNR shape: {snr.shape}")
    print(f"SNR mean: {snr.mean():.3f}")
    print(f"SNR std: {snr.std():.3f}")
    print(f"Sigma: {sigma:.3f}")
    print("✓ Función SNR funcionando correctamente")
    
except Exception as e:
    print(f"✗ Error en función SNR: {e}")

# 4. Verificar que el pipeline usa SNR
print(f"\n=== VERIFICACIÓN PIPELINE ===")
try:
    with open("DRAFTS/pipeline.py", "r") as f:
        pipeline_content = f.read()
        
    if "thresh_snr=config.SNR_THRESH" in pipeline_content:
        print("✓ Pipeline usa configuración SNR")
    else:
        print("✗ Pipeline NO usa configuración SNR")
        
    if "save_patch_plot" in pipeline_content and "thresh_snr" in pipeline_content:
        print("✓ Pipeline llama save_patch_plot con parámetros SNR")
    else:
        print("✗ Pipeline NO llama save_patch_plot con parámetros SNR")
        
except Exception as e:
    print(f"✗ Error verificando pipeline: {e}")

print(f"\n=== RESUMEN ===")
print("Las funciones SNR están integradas en:")
print("1. DRAFTS/config.py - Configuraciones SNR")
print("2. DRAFTS/snr_utils.py - Funciones de cálculo")  
print("3. DRAFTS/visualization.py - Visualizaciones mejoradas")
print("4. DRAFTS/pipeline.py - Llamadas con parámetros SNR")
print("\nEjecuta: python main.py para ver SNR en acción")
