from pathlib import Path

# =============================================================================
# CONFIGURACIÓN DE DATOS Y ARCHIVOS
# =============================================================================

# Directorios de entrada y salida
DATA_DIR = Path("./Data/raw")                        # Directorio con archivos de entrada (.fits, .fil)
RESULTS_DIR = Path("./Results")                      # Directorio para guardar resultados

# Lista de archivos a procesar
FRB_TARGETS = [
   "2017-04-03-12_56_05_230_0003_t36.548",
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
DOWN_TIME_RATE: int = 32                     # Factor de reducción en tiempo (1 = sin reducción)

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
DET_PROB: float = 0.05                       # Probabilidad mínima para considerar una detección válida
CLASS_PROB: float = 0.6                     # Probabilidad mínima para clasificar como burst

# Umbral de SNR para resaltar en visualizaciones
SNR_THRESH: float = 3.0                     # Umbral de SNR para resaltar en visualizaciones

# =============================================================================
# CONFIGURACIÓN DE ANÁLISIS MULTI-BANDA
# =============================================================================

# Análisis multi-banda (Full/Low/High)
USE_MULTI_BAND: bool = False                # True = usar análisis multi-banda, False = solo banda completa

# =============================================================================
# CONFIGURACIÓN DE POLARIZACIÓN (ENTRADA PSRFITS)
# =============================================================================

# Modo de polarización para PSRFITS con POL_TYPE=IQUV y npol>=4
# Opciones: "intensity" (I), "linear" (sqrt(Q^2+U^2)), "circular" (abs(V)),
#           "pol0", "pol1", "pol2", "pol3" para seleccionar un índice específico
POLARIZATION_MODE: str = "intensity"

# Índice por defecto si no hay IQUV (e.g., AABB, dos pols)
POLARIZATION_INDEX: int = 0

# =============================================================================
# CONFIGURACIÓN DE LOGGING Y DEBUG
# =============================================================================

# Debug de frecuencias y archivos
DEBUG_FREQUENCY_ORDER: bool = False        # True = mostrar información detallada de frecuencias y archivos
                                           # False = modo silencioso (recomendado para procesamiento en lote)

# Forzar generación de plots incluso sin candidatos (modo debug)
FORCE_PLOTS: bool = True                  # True = siempre generar plots para inspección

# =============================================================================
# CONFIGURACIÓN DE FILTRADO DE CANDIDATOS
# =============================================================================

# Solo guardar y mostrar candidatos clasificados como BURST
SAVE_ONLY_BURST: bool = False             # True = solo guardar candidatos BURST, False = guardar todos los candidatos
