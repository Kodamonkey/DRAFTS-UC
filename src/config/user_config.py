from pathlib import Path

# =============================================================================
# CONFIGURACIÓN DE DATOS Y ARCHIVOS
# =============================================================================

# Directorios de entrada y salida
DATA_DIR = Path("./Data/raw")                        # Directorio con archivos de entrada (.fits, .fil)
RESULTS_DIR = Path("./Results-test-refactor")                      # Directorio para guardar resultados

# Lista de archivos a procesar
FRB_TARGETS = [
   "3096_0001_00_8bit"
]

# =============================================================================
# CONFIGURACIÓN DE ANÁLISIS TEMPORAL
# =============================================================================

# Duración de cada slice temporal (milisegundos)
SLICE_DURATION_MS: float = 500.0

# =============================================================================
# CONFIGURACIÓN DE DOWNSAMPLING
# =============================================================================

# Factores de reducción para optimizar el procesamiento
DOWN_FREQ_RATE: int = 1                      # Factor de reducción en frecuencia (1 = sin reducción)
DOWN_TIME_RATE: int = 12                     # Factor de reducción en tiempo (1 = sin reducción)


# =============================================================================
# CONFIGURACIÓN DE DISPERSIÓN (DM)
# =============================================================================

# Rango de Dispersion Measure para búsqueda
DM_min: int = 0                             # DM mínimo en pc cm⁻³
DM_max: int = 3000                          # DM máximo en pc cm⁻³

# =============================================================================
# UMBRALES DE DETECCIÓN
# =============================================================================

# Probabilidades mínimas para detección y clasificación
DET_PROB: float = 0.3                       # Probabilidad mínima para considerar una detección válida

CLASS_PROB: float = 0.5                     # Probabilidad mínima para clasificar como burst

# Umbral de SNR para resaltar en visualizaciones
SNR_THRESH: float = 4.0                     # Umbral de SNR para resaltar en visualizaciones

# =============================================================================
# CONFIGURACIÓN DE ANÁLISIS MULTI-BANDA
# =============================================================================

# Análisis multi-banda (Full/Low/High)
USE_MULTI_BAND: bool = False                # True = usar análisis multi-banda, False = solo banda completa

# =============================================================================

# CONFIGURACIÓN DEL PIPELINE DE ALTA FRECUENCIA
# =============================================================================

# Controla si el pipeline de alta frecuencia se activa automáticamente
# según la frecuencia central del archivo (por defecto, ≥ 8000 MHz)
AUTO_HIGH_FREQ_PIPELINE: bool = True

# Umbral de frecuencia central (en MHz) para considerar "alta frecuencia"
HIGH_FREQ_THRESHOLD_MHZ: float = 8000.0

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
DEBUG_FREQUENCY_ORDER: bool = True        # True = mostrar información detallada de frecuencias y archivos
                                           # False = modo silencioso (recomendado para procesamiento en lote)

# Forzar generación de plots incluso sin candidatos (modo debug)
FORCE_PLOTS: bool = False                  # True = siempre generar plots para inspección

# =============================================================================
# CONFIGURACIÓN DE FILTRADO DE CANDIDATOS
# =============================================================================

# Solo guardar y mostrar candidatos clasificados como BURST
SAVE_ONLY_BURST: bool = True             # True = solo guardar candidatos BURST, False = guardar todos los candidatos

