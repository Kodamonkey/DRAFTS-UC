from pathlib import Path

# =============================================================================
# CONFIGURACIÓN DE DATOS Y ARCHIVOS
# =============================================================================

# Directorios de entrada y salida
DATA_DIR = Path("./Data/raw")                        # Directorio con archivos de entrada (.fits, .fil)
RESULTS_DIR = Path("./Results/ObjectDetection")      # Directorio para guardar resultados

# Lista de archivos a procesar
FRB_TARGETS = [
   "3100_0001_00_8bit"
]

# =============================================================================
# CONFIGURACIÓN DE ANÁLISIS TEMPORAL
# =============================================================================

# Duración de cada slice temporal (milisegundos)
SLICE_DURATION_MS: float = 500.0 # Testear con 1034

# =============================================================================
# CONFIGURACIÓN DE DOWNSAMPLING
# =============================================================================

# Factores de reducción para optimizar el procesamiento
DOWN_FREQ_RATE: int = 1                      # Factor de reducción en frecuencia (1 = sin reducción)
DOWN_TIME_RATE: int = 14                     # Factor de reducción en tiempo (1 = sin reducción)


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
DET_PROB: float = 0.3                       # Probabilidad mínima para considerar una detección válida
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

# Configuración de logging
LOG_LEVEL: str = "INFO"                     # Nivel de logging: DEBUG, INFO, WARNING, ERROR
LOG_COLORS: bool = True                     # Usar colores en la consola
LOG_FILE: bool = False                      # Guardar logs en archivo
GPU_VERBOSE: bool = False                   # Mostrar mensajes detallados de GPU
SHOW_PROGRESS: bool = True                  # Mostrar barras de progreso

# =============================================================================
# CONFIGURACIÓN DE CHUNKING Y SLICING AVANZADO
# =============================================================================

# Nuevo sistema de chunking contiguo y slicing ajustado a divisores
USE_PLANNED_CHUNKING: bool = True           # Usar planificador para tamaños de chunk y slices uniformes

# Límites de memoria para chunking (elige uno)
MAX_CHUNK_BYTES: int | None = None          # Límite duro en bytes para un chunk (recomendado); None para usar fracción de RAM
MAX_RAM_FRACTION: float = 0.25              # Fracción de RAM disponible a utilizar si MAX_CHUNK_BYTES es None
OVERHEAD_FACTOR: float = 1.3                # Factor de overhead por copias/buffers temporales

# Tolerancias y límites de slicing
TIME_TOL_MS: float = 0.1                    # Tolerancia temporal al ajustar la duración objetivo del slice
MAX_SLICE_COUNT: int = 5000                 # Límite superior de slices por chunk

# =============================================================================
# FLAGS AVANZADOS PARA MITIGAR ARTEFACTOS Y MEJORAR ESTABILIDAD
# =============================================================================

# Pre-whitening por canal antes de construir el cubo DM–tiempo (z-score por canal)
PREWHITEN_BEFORE_DM: bool = True

# Sombrear visualmente la cola inválida en los waterfalls cuando no hay solapamiento
SHADE_INVALID_TAIL: bool = True