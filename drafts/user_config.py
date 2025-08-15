from pathlib import Path

# =============================================================================
# CONFIGURACIÓN DE DATOS Y ARCHIVOS
# =============================================================================

# Directorios de entrada y salida
DATA_DIR = Path("./Data/raw")                        # Directorio con archivos de entrada (.fits, .fil)
RESULTS_DIR = Path("./Results/ObjectDetection")      # Directorio para guardar resultados

# Lista de archivos a procesar
FRB_TARGETS = [
   "FRB20201124_0009"
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
DOWN_TIME_RATE: int = 8                      # Factor de reducción en tiempo (1 = sin reducción)

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
CLASS_PROB: float = 0.5                    # Probabilidad mínima para clasificar como burst

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
FORCE_PLOTS: bool = False                  # True = siempre generar plots para inspección

# =============================================================================
# CONFIGURACIÓN AVANZADA DE SNR Y TIME SERIES
# =============================================================================

# Mejoras de SNR basadas en métodos robustos de detección
ENHANCED_SNR_CALCULATION: bool = True      # True = usar métodos mejorados de SNR, False = métodos originales

# Configuración de matched filtering para SNR
SNR_MATCHED_FILTERING: bool = True         # True = aplicar matched filtering con múltiples anchos de boxcar
SNR_BOXCAR_WIDTHS: list = [1, 2, 3, 4, 6, 9, 14, 20, 30]  # Anchos óptimos para detección FRB

# Configuración de estimación de ruido robusta
SNR_ROBUST_NOISE_ESTIMATION: bool = True   # True = usar estimación robusta de sigma (recorte central 95%)
SNR_NOISE_CORRECTION_FACTOR: float = 1.148 # Factor de corrección para estimación robusta

# Configuración de detección de picos mejorada
SNR_ENHANCED_PEAK_DETECTION: bool = True   # True = usar interpolación cuadrática para precisión temporal
SNR_MIN_THRESHOLD: float = 3.0             # Umbral mínimo de SNR para considerar picos válidos

# Configuración de time series en waterfalls
WATERFALL_SHOW_TIME_SERIES: bool = True    # True = mostrar time series integrada en waterfalls
WATERFALL_SHOW_SNR_PROFILE: bool = True    # True = mostrar perfil SNR en waterfalls
WATERFALL_TIME_SERIES_ALPHA: float = 0.7   # Transparencia de la time series (0.0-1.0)
WATERFALL_SNR_PROFILE_ALPHA: float = 0.8  # Transparencia del perfil SNR (0.0-1.0)

# Configuración de visualización de picos SNR
SNR_SHOW_PEAK_MARKERS: bool = True         # True = mostrar marcadores de picos SNR en plots
SNR_SHOW_PEAK_VALUES: bool = True          # True = mostrar valores de picos SNR en plots
SNR_PEAK_MARKER_SIZE: int = 5              # Tamaño de marcadores de picos
SNR_PEAK_LINE_COLOR: str = "red"           # Color de líneas de picos SNR

# =============================================================================
# CONFIGURACIÓN AVANZADA DE SNR Y TIME SERIES
# =============================================================================

# Mejoras de SNR basadas en métodos robustos de detección
ENHANCED_SNR_CALCULATION: bool = True      # True = usar métodos mejorados de SNR, False = métodos originales

# Configuración de matched filtering para SNR
SNR_MATCHED_FILTERING: bool = True         # True = aplicar matched filtering con múltiples anchos de boxcar
SNR_BOXCAR_WIDTHS: list = [1, 2, 3, 4, 6, 9, 14, 20, 30]  # Anchos óptimos para detección FRB

# Configuración de estimación de ruido robusta
SNR_ROBUST_NOISE_ESTIMATION: bool = True   # True = usar estimación robusta de sigma (recorte central 95%)
SNR_NOISE_CORRECTION_FACTOR: float = 1.148 # Factor de corrección para estimación robusta

# Configuración de detección de picos mejorada
SNR_ENHANCED_PEAK_DETECTION: bool = True   # True = usar interpolación cuadrática para precisión temporal
SNR_MIN_THRESHOLD: float = 3.0             # Umbral mínimo de SNR para considerar picos válidos

# Configuración de time series en waterfalls
WATERFALL_SHOW_TIME_SERIES: bool = True    # True = mostrar time series integrada en waterfalls
WATERFALL_SHOW_SNR_PROFILE: bool = True    # True = mostrar perfil SNR en waterfalls
WATERFALL_TIME_SERIES_ALPHA: float = 0.7   # Transparencia de la time series (0.0-1.0)
WATERFALL_SNR_PROFILE_ALPHA: float = 0.8  # Transparencia del perfil SNR (0.0-1.0)

# Configuración de visualización de picos SNR
SNR_SHOW_PEAK_MARKERS: bool = True         # True = mostrar marcadores de picos SNR en plots
SNR_SHOW_PEAK_VALUES: bool = True          # True = mostrar valores de picos SNR en plots
SNR_PEAK_MARKER_SIZE: int = 5              # Tamaño de marcadores de picos
SNR_PEAK_LINE_COLOR: str = "red"           # Color de líneas de picos SNR

