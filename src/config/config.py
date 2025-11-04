# This module defines runtime configuration defaults for the pipeline.

"""
Configuración del Sistema para el Pipeline de Detección de FRB
==============================================================

Este archivo contiene las configuraciones del sistema que se configuran automáticamente
o que son específicas del funcionamiento interno del pipeline.

IMPORTANTE: 
- NO modifique este archivo directamente
- Para configurar parámetros del usuario, use argumentos de línea de comandos o modifique user_config.py
- Este archivo mantiene compatibilidad con el código existente
"""

from __future__ import annotations
from pathlib import Path

# Variable global para almacenar configuración inyectada
_injected_config: dict = {}
_config_injected: bool = False

                                      
try:
    from .user_config import (
        DATA_DIR,
        DEBUG_FREQUENCY_ORDER,
        DM_max,
        DM_min,
        DOWN_FREQ_RATE,
        DOWN_TIME_RATE,
        DET_PROB,
        CLASS_PROB,
        FORCE_PLOTS,
        FRB_TARGETS,
        RESULTS_DIR,
        SLICE_DURATION_MS,
        SNR_THRESH,
        USE_MULTI_BAND,
        SAVE_ONLY_BURST,
        AUTO_HIGH_FREQ_PIPELINE,
        HIGH_FREQ_THRESHOLD_MHZ,
        POLARIZATION_MODE,
        POLARIZATION_INDEX,
    )
except ImportError:
    try:
        from user_config import (
            DATA_DIR,
            DEBUG_FREQUENCY_ORDER,
            DM_max,
            DM_min,
            DOWN_FREQ_RATE,
            DOWN_TIME_RATE,
            DET_PROB,
            CLASS_PROB,
            FORCE_PLOTS,
            FRB_TARGETS,
            RESULTS_DIR,
            SLICE_DURATION_MS,
            SNR_THRESH,
            USE_MULTI_BAND,
            SAVE_ONLY_BURST,
            AUTO_HIGH_FREQ_PIPELINE,
            HIGH_FREQ_THRESHOLD_MHZ,
            POLARIZATION_MODE,
            POLARIZATION_INDEX,
        )
    except:
        # Valores por defecto si no se puede importar user_config
        DATA_DIR = Path("./Data/raw/")
        RESULTS_DIR = Path("./Tests-Pulse-big-new")
        FRB_TARGETS = ["2017-04-03-08_55_22_153_0006_t23.444"]
        SLICE_DURATION_MS = 300.0
        DOWN_FREQ_RATE = 1
        DOWN_TIME_RATE = 8
        DM_min = 0
        DM_max = 1024
        DET_PROB = 0.3
        CLASS_PROB = 0.5
        SNR_THRESH = 5.0
        USE_MULTI_BAND = False
        SAVE_ONLY_BURST = True
        AUTO_HIGH_FREQ_PIPELINE = True
        HIGH_FREQ_THRESHOLD_MHZ = 8000.0
        POLARIZATION_MODE = "intensity"
        POLARIZATION_INDEX = 0
        DEBUG_FREQUENCY_ORDER = False
        FORCE_PLOTS = False


def inject_config(config_dict: dict):
    """
    Inyecta configuración desde argumentos de línea de comandos.
    Sobrescribe las variables del módulo.
    
    Args:
        config_dict: Diccionario con valores de configuración a inyectar
    """
    import sys
    global _config_injected
    
    # Importar el módulo actual (config)
    current_module = sys.modules[__name__]
    
    # Aplicar los valores inyectados a las variables del módulo
    for key, value in config_dict.items():
        setattr(current_module, key, value)
    
    _config_injected = True

                     
import numpy as np

                              
try:
    import torch
except ImportError:                                                    
    torch = None
              
MODEL_NAME = "resnet18"                                                           
MODEL_PATH = Path(__file__).parent.parent / "models" / f"cent_{MODEL_NAME}.pth"                            
                                
CLASS_MODEL_NAME = "resnet18"                                                         
CLASS_MODEL_PATH = Path(__file__).parent.parent / "models" / f"class_{CLASS_MODEL_NAME}.pth"                  
                   
if torch is not None:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = "cpu"
                                                   
FREQ: np.ndarray | None = None                                                         
FREQ_RESO: int = 0                                                              
TIME_RESO: float = 0.0                                                      
FILE_LENG: int = 0                                                           
                                                         
SLICE_LEN: int = 512                                                                                                          
                      
DATA_NEEDS_REVERSAL: bool = False                                                       
                                        
MAX_SAMPLES_LIMIT: int = 10_000_000                                                                      
                                             
DM_RANGE_ADAPTIVE: bool = False                                                                  
DM_RANGE_MIN_WIDTH: float = 80.0                                                  
DM_RANGE_MAX_WIDTH: float = 300.0                                                 
                                         
DM_PLOT_MARGIN_FACTOR: float = 0.25                                                    
DM_PLOT_MIN_RANGE: float = 120.0                                              
DM_PLOT_MAX_RANGE: float = 400.0                                              
DM_PLOT_DEFAULT_RANGE: float = 250.0                                          
DM_RANGE_DEFAULT_VISUALIZATION: str = "detailed"                                     
                               
SNR_OFF_REGIONS = [(-250, -150), (-100, -50), (50, 100), (150, 250)]                       
SNR_HIGHLIGHT_COLOR = "red"                                                  
                                         
SLICE_LEN_MIN: int = 32                                                                                           
SLICE_LEN_MAX: int = 2048                                                                                         
                                                         
USE_PLANNED_CHUNKING: bool = True                                                                       
                            
MAX_CHUNK_BYTES: int | None = None                                                                                            
MAX_RAM_FRACTION: float = 0.25                                                                                
OVERHEAD_FACTOR: float = 1.3                                                                  
                                  
TIME_TOL_MS: float = 0.1                                                                                   
MAX_SLICE_COUNT: int = 5000                                                      
                                                              
DM_DYNAMIC_RANGE_ENABLE: bool = False                                                         
DM_RANGE_FACTOR: float = 0.3                                                                 
                                     
SNR_SHOW_PEAK_LINES: bool = False                                                              
SNR_COLORMAP = "viridis"                                                     
                                      
LOG_LEVEL: str = "INFO"                                                                    
LOG_COLORS: bool = True                                                 
LOG_FILE: bool = False                                               
GPU_VERBOSE: bool = False                                                       
SHOW_PROGRESS: bool = True                                              
                                                                               
PREWHITEN_BEFORE_DM: bool = True

                                                                                   
SHADE_INVALID_TAIL: bool = True


def get_band_configs():
    """Retorna la configuración de bandas según USE_MULTI_BAND"""
    return [
        (0, "fullband", "Full Band"),
        (1, "lowband", "Low Band"),
        (2, "highband", "High Band"),
    ] if USE_MULTI_BAND else [(0, "fullband", "Full Band")]


                                                                               
                                                       
                                                                               

def validate_configuration():
    """
    Valida la configuración del sistema y genera mensajes de error informativos.
    
    Esta función verifica que todos los parámetros críticos estén configurados
    correctamente antes de ejecutar el pipeline.
    """
    errors = []
    
                                      
    if FREQ_RESO <= 0:
        errors.append(
            f"FREQ_RESO={FREQ_RESO} es inválido\n"
            f"  → FREQ_RESO debe ser > 0 para procesar datos de frecuencia\n"
            f"  → Este valor se extrae del header del archivo FITS/FIL\n"
            f"  → Recomendación: Verificar que el archivo de datos sea válido"
        )
    
    if TIME_RESO <= 0:
        errors.append(
            f"TIME_RESO={TIME_RESO} es inválido\n"
            f"  → TIME_RESO debe ser > 0 para procesar datos temporales\n"
            f"  → Este valor se extrae del header del archivo FITS/FIL\n"
            f"  → Recomendación: Verificar que el archivo de datos sea válido"
        )
    
    if FILE_LENG <= 0:
        errors.append(
            f"FILE_LENG={FILE_LENG} es inválido\n"
            f"  → FILE_LENG debe ser > 0 para procesar datos\n"
            f"  → Este valor indica el número total de muestras temporales\n"
            f"  → Recomendación: Verificar que el archivo de datos sea válido"
        )
    
                                    
    if SLICE_LEN < SLICE_LEN_MIN or SLICE_LEN > SLICE_LEN_MAX:
        errors.append(
            f"SLICE_LEN={SLICE_LEN} está fuera del rango válido [{SLICE_LEN_MIN}, {SLICE_LEN_MAX}]\n"
            f"  → SLICE_LEN debe estar entre {SLICE_LEN_MIN} y {SLICE_LEN_MAX} muestras\n"
            f"  → Este valor se calcula automáticamente desde SLICE_DURATION_MS\n"
            f"  → Recomendación: Ajustar SLICE_DURATION_MS en user_config.py"
        )
    
                       
    if DM_RANGE_MIN_WIDTH <= 0 or DM_RANGE_MAX_WIDTH <= 0:
        errors.append(
            f"Rangos DM inválidos: MIN={DM_RANGE_MIN_WIDTH}, MAX={DM_RANGE_MAX_WIDTH}\n"
            f"  → Ambos valores deben ser > 0\n"
            f"  → Estos valores definen los límites del rango DM dinámico\n"
            f"  → Recomendación: Verificar configuración de rangos DM"
        )
    
    if DM_RANGE_MIN_WIDTH >= DM_RANGE_MAX_WIDTH:
        errors.append(
            f"Rangos DM inconsistentes: MIN={DM_RANGE_MIN_WIDTH} >= MAX={DM_RANGE_MAX_WIDTH}\n"
            f"  → DM_RANGE_MIN_WIDTH debe ser < DM_RANGE_MAX_WIDTH\n"
            f"  → Recomendación: Ajustar los valores de rango DM"
        )
    
                                      
    if not MODEL_PATH.exists():
        errors.append(
            f"Modelo de detección no encontrado: {MODEL_PATH}\n"
            f"  → El archivo del modelo no existe en la ruta especificada\n"
            f"  → Verificar que el modelo esté entrenado y guardado\n"
            f"  → Recomendación: Entrenar el modelo o verificar la ruta"
        )
    
    if not CLASS_MODEL_PATH.exists():
        errors.append(
            f"Modelo de clasificación no encontrado: {CLASS_MODEL_PATH}\n"
            f"  → El archivo del modelo no existe en la ruta especificada\n"
            f"  → Verificar que el modelo esté entrenado y guardado\n"
            f"  → Recomendación: Entrenar el modelo o verificar la ruta"
        )
    
                                          
    if torch is not None and torch.cuda.is_available():
        try:
                                                       
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
        except Exception as e:
            errors.append(
                f"Error con CUDA: {e}\n"
                f"  → CUDA está disponible pero no funciona correctamente\n"
                f"  → Verificar drivers de NVIDIA y instalación de PyTorch\n"
                f"  → Recomendación: Reinstalar PyTorch con soporte CUDA o usar CPU"
            )
    
                                                            
    if errors:
        error_message = "Configuración del sistema inválida:\n\n"
        for i, error in enumerate(errors, 1):
            error_message += f"{i}. {error}\n\n"
        error_message += "Corrija estos errores antes de ejecutar el pipeline."
        
        raise ValueError(error_message)
    
    return True


def check_model_files():
    """
    Verifica que los archivos de modelo existan y sean accesibles.
    
    Returns:
        dict: Diccionario con el estado de cada modelo
    """
    model_status = {}
    
                                   
    if MODEL_PATH.exists():
        try:
                                                                           
            if torch is not None:
                state = torch.load(MODEL_PATH, map_location='cpu')
                model_status['detection'] = {
                    'exists': True,
                    'size_mb': MODEL_PATH.stat().st_size / (1024 * 1024),
                    'state_dict_keys': len(state.keys()) if isinstance(state, dict) else 0
                }
            else:
                model_status['detection'] = {
                    'exists': True,
                    'size_mb': MODEL_PATH.stat().st_size / (1024 * 1024),
                    'note': 'PyTorch no disponible para verificación completa'
                }
        except Exception as e:
            model_status['detection'] = {
                'exists': True,
                'error': f"Modelo corrupto: {e}",
                'recommendation': 'Reentrenar el modelo'
            }
    else:
        model_status['detection'] = {
            'exists': False,
            'error': f"Archivo no encontrado: {MODEL_PATH}",
            'recommendation': 'Entrenar el modelo o verificar la ruta'
        }
    
                                       
    if CLASS_MODEL_PATH.exists():
        try:
            if torch is not None:
                state = torch.load(CLASS_MODEL_PATH, map_location='cpu')
                model_status['classification'] = {
                    'exists': True,
                    'size_mb': CLASS_MODEL_PATH.stat().st_size / (1024 * 1024),
                    'state_dict_keys': len(state.keys()) if isinstance(state, dict) else 0
                }
            else:
                model_status['classification'] = {
                    'exists': True,
                    'size_mb': CLASS_MODEL_PATH.stat().st_size / (1024 * 1024),
                    'note': 'PyTorch no disponible para verificación completa'
                }
        except Exception as e:
            model_status['classification'] = {
                'exists': True,
                'error': f"Modelo corrupto: {e}",
                'recommendation': 'Reentrenar el modelo'
            }
    else:
        model_status['classification'] = {
            'exists': False,
            'error': f"Archivo no encontrado: {CLASS_MODEL_PATH}",
            'recommendation': 'Entrenar el modelo o verificar la ruta'
        }
    
    return model_status
