from pathlib import Path

# =============================================================================
# CONFIGURACIÓN DE DATOS Y ARCHIVOS
# =============================================================================

# Directorios de entrada y salida
DATA_DIR = Path("./Data/raw")                        # Directorio con archivos de entrada (.fits, .fil)
RESULTS_DIR = Path("./Results/ObjectDetection")      # Directorio para guardar resultados

# Lista de archivos a procesar
FRB_TARGETS = [
    "FRB20201124_0009",
    "B0355+54_FB_20220918",    
]

# =============================================================================
# CONFIGURACIÓN DE ANÁLISIS TEMPORAL
# =============================================================================

# Duración de cada slice temporal (milisegundos)
# El sistema calcula automáticamente SLICE_LEN según los metadatos del archivo
SLICE_DURATION_MS: float = 1000.0

# Valores recomendados:
# - 16.0 ms: Análisis muy rápido (puede perder detalle)
# - 32.0 ms: Análisis rápido (buen balance)
# - 64.0 ms: Análisis estándar (recomendado)
# - 128.0 ms: Análisis detallado (más lento)
# - 256.0 ms: Análisis ultra-detallado (muy lento)

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
# CONFIGURACIÓN DE DEBUG Y DIAGNÓSTICO
# =============================================================================

# Debug de frecuencias y archivos
DEBUG_FREQUENCY_ORDER: bool = False         # True = mostrar información detallada de frecuencias y archivos
                                           # False = modo silencioso (recomendado para procesamiento en lote)
