from pathlib import Path

# =============================================================================
# CONFIGURACIÓN DE DATOS Y ARCHIVOS
# =============================================================================

# Directorios de entrada y salida
DATA_DIR = Path("./Data/raw")                        # Directorio con archivos de entrada (.fits, .fil)
RESULTS_DIR = Path("./Results/ObjectDetection")      # Directorio para guardar resultados

# Lista de archivos a procesar
FRB_TARGETS = [
   "2017-04-03-08-16-13_142_0003_t39.977"
]

# =============================================================================
# CONFIGURACIÓN DE ANÁLISIS TEMPORAL
# =============================================================================

# Duración de cada slice temporal (milisegundos)
SLICE_DURATION_MS: float = 300.0

# =============================================================================
# CONFIGURACIÓN DE DOWNSAMPLING
# =============================================================================

# Factores de reducción para optimizar el procesamiento
DOWN_FREQ_RATE: int = 1                      # Factor de reducción en frecuencia (1 = sin reducción)
DOWN_TIME_RATE: int = 45                     # Factor de reducción en tiempo (1 = sin reducción)


# =============================================================================
# CONFIGURACIÓN DE DISPERSIÓN (DM)
# =============================================================================

# Rango de Dispersion Measure para búsqueda
DM_min: int = 0                             # DM mínimo en pc cm⁻³
DM_max: int = 1024                          # DM máximo en pc cm⁻³

# =============================================================================
# UMBRALES DE DETECCIÓN
# =============================================================================

# Probabilidades mínimas para detección y clasificación
DET_PROB: float = 0.1                       # Probabilidad mínima para considerar una detección válida
CLASS_PROB: float = 0.2                     # Probabilidad mínima para clasificar como burst

# Umbral de SNR para resaltar en visualizaciones
SNR_THRESH: float = 3.0                     # Umbral de SNR para resaltar en visualizaciones

# =============================================================================
# CONFIGURACIÓN DE ANÁLISIS MULTI-BANDA
# =============================================================================

# Análisis multi-banda (Full/Low/High)
USE_MULTI_BAND: bool = False                # True = usar análisis multi-banda, False = solo banda completa

# =============================================================================
# CONFIGURACIÓN DE ESTRATEGIAS PARA DETECCIÓN DE BURSTS EN RÉGIMEN MILIMÉTRICO (ALMA Band 3)
# =============================================================================

# Estrategia E1: Expandir rango y paso de DM para "abrir" el bow-tie
STRATEGY_DM_EXPAND: dict = {
    'enabled': True,                        # True = habilitar estrategia E1
    'dm_max': 2000,                         # DM máximo expandido (ajustar según dataset)
    'smear_frac': 0.25,                     # Δt_residual ≤ 0.25 · W (ancho temporal)
    'min_dm_sigmas': 3.0,                   # Exigir centro DM* > 0 con ≥ 3σ
}

# Estrategia E2: Pescar en DM≈0 con umbral laxo y validación
STRATEGY_FISH_NEAR_ZERO: dict = {
    'enabled': True,                         # True = habilitar estrategia E2
    'dm_fish_max': 75,                      # Barrido muy bajo solo para "pescar"
    'fish_thresh': 0.3,                     # Umbral CenterNet más laxo para E2
    'refine': {
        'dm_local_max': 300,                # Micro-rejilla de validación local
        'ddm_local': 1,                     # Paso DM para micro-rejilla
        'min_delta_snr': 5.0,               # SNR(DM*) - SNR(0) mínima
        'min_dm_star': 3,                   # DM* mínimo aceptable
        'subband_consistency_pc': 50,       # Tolerancia relativa entre sub-bandas (%)
    }
}

# =============================================================================
# CONFIGURACIÓN DE LOGGING Y DEBUG
# =============================================================================

# Debug de frecuencias y archivos
DEBUG_FREQUENCY_ORDER: bool = False        # True = mostrar información detallada de frecuencias y archivos
                                           # False = modo silencioso (recomendado para procesamiento en lote)

# Forzar generación de plots incluso sin candidatos (modo debug)
FORCE_PLOTS: bool = True                  # True = siempre generar plots para inspección