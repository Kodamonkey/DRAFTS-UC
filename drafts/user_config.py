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
SLICE_DURATION_MS: float = 100.0 # Testear con 1034

# =============================================================================
# CONFIGURACIÓN DE DOWNSAMPLING
# =============================================================================

# Factores de reducción para optimizar el procesamiento
DOWN_FREQ_RATE: int = 1                      # Factor de reducción en frecuencia (1 = sin reducción)
DOWN_TIME_RATE: int = 24                     # Factor de reducción en tiempo (1 = sin reducción)


# =============================================================================
# CONFIGURACIÓN DE DISPERSIÓN (DM)
# =============================================================================

# Rango de Dispersion Measure para búsqueda
DM_min: int = 1700                             # DM mínimo en pc cm⁻³
DM_max: int = 1800                          # DM máximo en pc cm⁻³

# =============================================================================
# UMBRALES DE DETECCIÓN
# =============================================================================

# Probabilidades mínimas para detección y clasificación
DET_PROB: float = 0.1                       # Probabilidad mínima para considerar una detección válida
CLASS_PROB: float = 0.5                     # Probabilidad mínima para clasificar como burst

# Umbral de SNR para resaltar en visualizaciones
SNR_THRESH: float = 3.0                     # Umbral de SNR para resaltar en visualizaciones

# =============================================================================
# CONFIGURACIÓN DE ANÁLISIS MULTI-BANDA
# =============================================================================

# Análisis multi-banda (Full/Low/High)
USE_MULTI_BAND: bool = False                # True = usar análisis multi-banda, False = solo banda completa

# =============================================================================
# CONFIGURACIÓN DE LOGGING Y DEBUG
# =============================================================================

# Debug de frecuencias y archivos
DEBUG_FREQUENCY_ORDER: bool = False        # True = mostrar información detallada de frecuencias y archivos
                                           # False = modo silencioso (recomendado para procesamiento en lote)

# Forzar generación de plots incluso sin candidatos (modo debug)
FORCE_PLOTS: bool = True                  # True = siempre generar plots para inspección