"""
ARCHIVO DE CONFIGURACIÓN RÁPIDA - COPIA Y MODIFICA
================================================

Este archivo contiene configuraciones predefinidas para casos comunes.
Copia la sección que necesites y modifica los parámetros según tu caso.

INSTRUCCIONES:
1. Copia la configuración que más se parezca a tu caso
2. Modifica los parámetros según tus necesidades
3. Ejecuta el código para aplicar la configuración
"""

# =============================================================================
# CONFIGURACIÓN RÁPIDA - SURVEY DE FRB ESTÁNDAR
# =============================================================================

def configurar_survey_frb_estandar():
    """
    Configuración balanceada para survey general de FRB.
    Buena relación sensibilidad/velocidad/calidad.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "DRAFTS"))
    from DRAFTS import config
    
    # === PARÁMETROS PRINCIPALES ===
    config.DATA_DIR = Path("./Data")                    # Carpeta con tus archivos .fil o .fits
    config.RESULTS_DIR = Path("./Results/Survey_FRB")   # Carpeta para resultados
    config.FRB_TARGETS = ["B0355+54"]                   # Lista de archivos a procesar
    
    # === DETECCIÓN ===
    config.DET_PROB = 0.1          # Probabilidad mínima de detección (0.05-0.2)
    config.SNR_THRESH = 3.0        # Umbral SNR para resaltar (2.0-5.0)
    config.DM_min = 0              # DM mínimo en pc cm⁻³
    config.DM_max = 1200           # DM máximo en pc cm⁻³
    
    # === VISUALIZACIÓN OPTIMIZADA ===
    config.DM_DYNAMIC_RANGE_ENABLE = True    # Plots centrados en candidatos
    config.DM_RANGE_FACTOR = 0.3             # Margen alrededor del candidato (0.2-0.4)
    config.DM_PLOT_MARGIN_FACTOR = 0.25      # Margen extra para no pegar arriba
    
    # === PROCESAMIENTO ===
    config.USE_MULTI_BAND = True             # Análisis en 3 bandas (Full/Low/High)
    config.ENABLE_CHUNK_PROCESSING = True    # Para archivos grandes
    config.MAX_SAMPLES_LIMIT = 1500000       # Tamaño de chunk (ajustar según RAM)
    
    # === LIMPIEZA RFI ===
    config.RFI_ENABLE_ALL_FILTERS = True     # Activar limpieza completa
    config.RFI_FREQ_SIGMA_THRESH = 4.0       # Umbral para canales malos
    config.RFI_TIME_SIGMA_THRESH = 4.0       # Umbral para tiempos malos
    
    print("✅ Configuración 'Survey FRB Estándar' aplicada")
    return config

# =============================================================================
# CONFIGURACIÓN RÁPIDA - BÚSQUEDA RÁPIDA
# =============================================================================

def configurar_busqueda_rapida():
    """
    Configuración para procesamiento rápido.
    Menos sensible pero más veloz.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "DRAFTS"))
    from DRAFTS import config
    
    # === PARÁMETROS PRINCIPALES ===
    config.DATA_DIR = Path("./Data")
    config.RESULTS_DIR = Path("./Results/Busqueda_Rapida")
    
    # === DETECCIÓN RÁPIDA ===
    config.DET_PROB = 0.15         # Menos sensible pero más rápido
    config.SNR_THRESH = 4.0        # Umbral más alto
    config.DM_min = 50             # Rango más estrecho si conoces la fuente
    config.DM_max = 800
    
    # === VISUALIZACIÓN ===
    config.DM_DYNAMIC_RANGE_ENABLE = True
    config.DM_RANGE_FACTOR = 0.4   # Rango más amplio para exploración
    
    # === PROCESAMIENTO RÁPIDO ===
    config.USE_MULTI_BAND = False  # Solo banda completa
    config.MAX_SAMPLES_LIMIT = 1000000     # Chunks más pequeños
    config.SLICE_DURATION_SECONDS = 0.064  # Resolución temporal menor
    
    # === RFI BÁSICO ===
    config.RFI_ENABLE_ALL_FILTERS = False
    config.RFI_FREQ_SIGMA_THRESH = 5.0     # Menos agresivo
    
    print("✅ Configuración 'Búsqueda Rápida' aplicada")
    return config

# =============================================================================
# CONFIGURACIÓN RÁPIDA - ALTA SENSIBILIDAD
# =============================================================================

def configurar_alta_sensibilidad():
    """
    Configuración para máxima sensibilidad.
    Detecta eventos débiles pero es más lento.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "DRAFTS"))
    from DRAFTS import config
    
    # === PARÁMETROS PRINCIPALES ===
    config.DATA_DIR = Path("./Data")
    config.RESULTS_DIR = Path("./Results/Alta_Sensibilidad")
    
    # === DETECCIÓN SENSIBLE ===
    config.DET_PROB = 0.05         # Muy sensible
    config.SNR_THRESH = 2.5        # Umbral bajo
    config.DM_min = 0
    config.DM_max = 2000           # Rango extendido
    
    # === VISUALIZACIÓN DETALLADA ===
    config.DM_DYNAMIC_RANGE_ENABLE = True
    config.DM_RANGE_FACTOR = 0.2   # Rango más estrecho para mejor resolución
    config.DM_PLOT_MARGIN_FACTOR = 0.3
    
    # === PROCESAMIENTO DETALLADO ===
    config.USE_MULTI_BAND = True
    config.MAX_SAMPLES_LIMIT = 800000      # Chunks más pequeños
    config.SLICE_DURATION_SECONDS = 0.016  # Alta resolución temporal
    
    # === RFI COMPLETO ===
    config.RFI_ENABLE_ALL_FILTERS = True
    config.RFI_INTERPOLATE_MASKED = True
    config.RFI_SAVE_DIAGNOSTICS = True
    config.RFI_FREQ_SIGMA_THRESH = 3.0     # Más agresivo
    
    print("✅ Configuración 'Alta Sensibilidad' aplicada")
    return config

# =============================================================================
# CONFIGURACIÓN RÁPIDA - AMBIENTE CON MUCHO RFI
# =============================================================================

def configurar_ambiente_rfi():
    """
    Configuración para ambientes con mucha interferencia.
    Prioriza limpieza sobre sensibilidad.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "DRAFTS"))
    from DRAFTS import config
    
    # === PARÁMETROS PRINCIPALES ===
    config.DATA_DIR = Path("./Data")
    config.RESULTS_DIR = Path("./Results/Ambiente_RFI")
    
    # === DETECCIÓN ROBUSTA ===
    config.DET_PROB = 0.2          # Menos sensible para evitar RFI
    config.SNR_THRESH = 5.0        # Umbral alto
    
    # === VISUALIZACIÓN ===
    config.DM_DYNAMIC_RANGE_ENABLE = True
    config.DM_RANGE_FACTOR = 0.25
    
    # === PROCESAMIENTO ===
    config.USE_MULTI_BAND = True
    config.MAX_SAMPLES_LIMIT = 1200000
    
    # === RFI AGRESIVO ===
    config.RFI_ENABLE_ALL_FILTERS = True
    config.RFI_FREQ_SIGMA_THRESH = 3.0     # Muy agresivo
    config.RFI_TIME_SIGMA_THRESH = 3.0     # Muy agresivo
    config.RFI_ZERO_DM_SIGMA_THRESH = 3.0  # Filtro Zero-DM agresivo
    config.RFI_IMPULSE_SIGMA_THRESH = 4.0  # Filtro de impulsos
    config.RFI_INTERPOLATE_MASKED = True   # Interpolar datos malos
    config.RFI_SAVE_DIAGNOSTICS = True     # Guardar diagnósticos
    
    print("✅ Configuración 'Ambiente RFI' aplicada")
    return config

# =============================================================================
# CONFIGURACIÓN RÁPIDA - PULSAR CONOCIDO
# =============================================================================

def configurar_pulsar_conocido(dm_esperado=None):
    """
    Configuración para estudio de pulsar conocido.
    
    Parameters
    ----------
    dm_esperado : float, optional
        DM esperado del pulsar en pc cm⁻³
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "DRAFTS"))
    from DRAFTS import config
    
    # === PARÁMETROS PRINCIPALES ===
    config.DATA_DIR = Path("./Data")
    config.RESULTS_DIR = Path("./Results/Pulsar_Conocido")
    
    # === DETECCIÓN ESPECÍFICA ===
    config.DET_PROB = 0.05         # Alta sensibilidad
    config.SNR_THRESH = 2.0        # Umbral muy bajo
    
    # Configurar DM si se conoce
    if dm_esperado:
        margen_dm = max(20, dm_esperado * 0.2)  # 20% de margen
        config.DM_min = max(0, int(dm_esperado - margen_dm))
        config.DM_max = int(dm_esperado + margen_dm)
        print(f"   - DM configurado: {config.DM_min}-{config.DM_max} pc cm⁻³ (centrado en {dm_esperado})")
    
    # === VISUALIZACIÓN DETALLADA ===
    config.DM_DYNAMIC_RANGE_ENABLE = True
    config.DM_RANGE_FACTOR = 0.15  # Rango muy estrecho
    config.DM_PLOT_MARGIN_FACTOR = 0.2
    
    # === PROCESAMIENTO DETALLADO ===
    config.USE_MULTI_BAND = True
    config.MAX_SAMPLES_LIMIT = 600000      # Chunks pequeños
    config.SLICE_DURATION_SECONDS = 0.016  # Muy alta resolución
    
    # === RFI COMPLETO ===
    config.RFI_ENABLE_ALL_FILTERS = True
    config.RFI_INTERPOLATE_MASKED = True
    config.RFI_SAVE_DIAGNOSTICS = True
    
    print("✅ Configuración 'Pulsar Conocido' aplicada")
    return config

# =============================================================================
# CÓMO USAR ESTAS CONFIGURACIONES
# =============================================================================

def mostrar_instrucciones():
    """Muestra instrucciones de uso."""
    print("""
=== CÓMO USAR ESTAS CONFIGURACIONES ===

1. ELEGIR CONFIGURACIÓN:
   - Survey FRB estándar: configurar_survey_frb_estandar()
   - Búsqueda rápida: configurar_busqueda_rapida()
   - Alta sensibilidad: configurar_alta_sensibilidad()
   - Ambiente con RFI: configurar_ambiente_rfi()
   - Pulsar conocido: configurar_pulsar_conocido(dm_esperado=123.4)

2. APLICAR CONFIGURACIÓN:
   
   # Ejemplo:
   config = configurar_survey_frb_estandar()
   
   # Modificar parámetros adicionales si necesitas:
   config.DM_max = 1500  # Cambiar DM máximo
   config.SNR_THRESH = 2.5  # Cambiar umbral SNR
   
   # Ejecutar pipeline:
   from DRAFTS.pipeline import run_pipeline
   run_pipeline()

3. VERIFICAR RESULTADOS:
   - Los plots aparecerán en la carpeta RESULTS_DIR
   - Los candidatos estarán centrados en los plots DM vs Time
   - Revisa los logs para información de procesamiento

4. AJUSTAR SI ES NECESARIO:
   - Muchos falsos positivos → Aumentar DET_PROB o SNR_THRESH
   - Pocas detecciones → Reducir DET_PROB o SNR_THRESH
   - Candidatos pegados arriba → Aumentar DM_RANGE_FACTOR
   - Procesamiento lento → Reducir MAX_SAMPLES_LIMIT
   
¡Buena suerte con tu búsqueda de FRB! 🔭
""")

if __name__ == "__main__":
    mostrar_instrucciones()
