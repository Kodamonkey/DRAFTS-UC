"""
Configuración Completa Auto-Generada
===================================

Este módulo toma la configuración simple del usuario y la expande
automáticamente a todos los parámetros necesarios para el pipeline.

El astrónomo solo necesita modificar config_simple.py
"""

from pathlib import Path
import numpy as np

# Importar configuración simple del usuario
try:
    from .config_simple import *
except ImportError:
    # Si no se encuentra, usar valores por defecto
    from pathlib import Path
    DATA_DIR = Path("./Data")
    RESULTS_DIR = Path("./Results")
    FRB_TARGETS = ["B0355+54"]
    DM_min = 0
    DM_max = 1024
    DET_PROB = 0.1

# =============================================================================
# EXPANSIÓN AUTOMÁTICA DE CONFIGURACIÓN
# =============================================================================

def get_complete_config():
    """
    Retorna la configuración completa basada en los parámetros simples del usuario.
    Esta función se llama automáticamente desde el pipeline principal.
    """
    
    config = {
        # --- Configuración del usuario (de config_simple.py) ---
        'DATA_DIR': DATA_DIR,
        'RESULTS_DIR': RESULTS_DIR,
        'FRB_TARGETS': FRB_TARGETS,
        'DM_min': DM_min,
        'DM_max': DM_max,
        'DET_PROB': DET_PROB,
        
        # --- Configuración de slice temporal ---
        'SLICE_DURATION_SECONDS': 0.032,
        'SLICE_LEN_AUTO': True,
        'SLICE_LEN_INTELLIGENT': True,
        'SLICE_LEN_OVERRIDE_MANUAL': True,
        'SLICE_LEN_MIN': 16,
        'SLICE_LEN_MAX': 512,
        'SLICE_LEN': 32,
        
        # --- Configuración de visualización DM dinámico ---
        'DM_DYNAMIC_RANGE_ENABLE': True,
        'DM_RANGE_FACTOR': 0.3,
        'DM_PLOT_MARGIN_FACTOR': 0.25,
        'DM_RANGE_MIN_WIDTH': 80.0,
        'DM_RANGE_MAX_WIDTH': 300.0,
        'DM_PLOT_MIN_RANGE': 120.0,
        'DM_PLOT_MAX_RANGE': 400.0,
        'DM_PLOT_DEFAULT_RANGE': 250.0,
        'DM_RANGE_ADAPTIVE': True,
        'DM_RANGE_DEFAULT_VISUALIZATION': "detailed",
        
        # --- Configuración de modelos ---
        'MODEL_NAME': "resnet50",
        'MODEL_PATH': Path(f"./models/cent_resnet50.pth"),
        'CLASS_MODEL_NAME': "resnet18",
        'CLASS_MODEL_PATH': Path(f"./models/class_resnet18.pth"),
        'CLASS_PROB': 0.5,
        
        # --- Configuración de procesamiento ---
        'USE_MULTI_BAND': True,
        'ENABLE_CHUNK_PROCESSING': True,
        'MAX_SAMPLES_LIMIT': 2000000,
        'CHUNK_OVERLAP_SAMPLES': 1000,
        
        # --- Configuración de SNR ---
        'SNR_THRESH': 3.0,
        'SNR_OFF_REGIONS': [(-250, -150), (-100, -50), (50, 100), (150, 250)],
        'SNR_COLORMAP': "viridis",
        'SNR_HIGHLIGHT_COLOR': "red",
        
        # --- Configuración de RFI (valores conservadores) ---
        'RFI_FREQ_SIGMA_THRESH': 5.0,
        'RFI_TIME_SIGMA_THRESH': 5.0,
        'RFI_ZERO_DM_SIGMA_THRESH': 4.0,
        'RFI_IMPULSE_SIGMA_THRESH': 6.0,
        'RFI_POLARIZATION_THRESH': 0.8,
        'RFI_CHANNEL_DETECTION_METHOD': "mad",
        'RFI_TIME_DETECTION_METHOD': "mad",
        'RFI_ENABLE_ALL_FILTERS': False,
        'RFI_INTERPOLATE_MASKED': False,
        'RFI_SAVE_DIAGNOSTICS': False,
        
        # --- Metadatos (se configuran en runtime) ---
        'FREQ': None,
        'FREQ_RESO': 0,
        'TIME_RESO': 0.0,
        'FILE_LENG': 0,
        'DOWN_FREQ_RATE': 1,
        'DOWN_TIME_RATE': 1,
        'DATA_NEEDS_REVERSAL': False,
    }
    
    # Configuración del dispositivo de cómputo
    try:
        import torch
        config['DEVICE'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        config['DEVICE'] = "cpu"
    
    return config

# =============================================================================
# INTERFAZ PARA BACKWARD COMPATIBILITY
# =============================================================================

# Para mantener compatibilidad con el código existente,
# exportamos todas las variables como si fueran del módulo original
_config = get_complete_config()
for key, value in _config.items():
    globals()[key] = value

# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def print_user_config():
    """Muestra solo la configuración que el usuario puede modificar."""
    print("=== CONFIGURACIÓN ACTUAL DEL USUARIO ===")
    print(f"Directorio de datos: {DATA_DIR}")
    print(f"Directorio de resultados: {RESULTS_DIR}")
    print(f"Targets FRB: {FRB_TARGETS}")
    print(f"Rango DM: {DM_min} - {DM_max} pc cm⁻³")
    print(f"Probabilidad mínima detección: {DET_PROB}")
    print("=========================================")
    print("Para modificar estos valores, edite config_simple.py")

def print_auto_config():
    """Muestra la configuración automática expandida."""
    print("=== CONFIGURACIÓN AUTOMÁTICA ===")
    config = get_complete_config()
    for key, value in sorted(config.items()):
        if key not in ['DATA_DIR', 'RESULTS_DIR', 'FRB_TARGETS', 'DM_min', 'DM_max', 'DET_PROB']:
            print(f"{key}: {value}")
    print("===============================")

if __name__ == "__main__":
    print("Configuración de Pipeline FRB")
    print_user_config()
    print()
    print_auto_config()
