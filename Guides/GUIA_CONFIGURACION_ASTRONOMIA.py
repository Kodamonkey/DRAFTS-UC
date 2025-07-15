"""
GUÍA DE CONFIGURACIÓN PARA USUARIOS DE ASTRONOMÍA
================================================

Este archivo muestra ejemplos de configuración del pipeline de FRB para
diferentes casos de uso astronómicos comunes.

ESTRUCTURA DE CONFIGURACIÓN:
El archivo config.py está organizado en secciones lógicas:

1. CONFIGURACIÓN PRINCIPAL - Parámetros que típicamente se modifican
2. CONFIGURACIÓN DE SLICE TEMPORAL - Resolución temporal
3. CONFIGURACIÓN DE VISUALIZACIÓN DM DINÁMICO - Plots optimizados
4. CONFIGURACIÓN DE MODELOS ML - Redes neuronales
5. CONFIGURACIÓN DE MITIGACIÓN DE RFI - Filtros de interferencia
6. PARÁMETROS DE OBSERVACIÓN - Configurados automáticamente
7. CONFIGURACIÓN DE SNR Y VISUALIZACIÓN - Análisis de señal
8. CONFIGURACIÓN DE CHUNKING - Procesamiento de archivos grandes
9. CONFIGURACIÓN DEL SISTEMA - Configuración técnica
"""

# =============================================================================
# CASOS DE USO COMUNES
# =============================================================================

# CASO 1: Detección de FRB en tiempo real con alta sensibilidad
CASE_1_HIGH_SENSITIVITY = {
    "description": "Detección en tiempo real con máxima sensibilidad",
    "parameters": {
        "DET_PROB": 0.05,          # Más sensible a detecciones débiles
        "SNR_THRESH": 2.5,         # Umbral SNR más bajo
        "DM_RANGE_FACTOR": 0.2,    # Rango DM más estrecho para mejor resolución
        "MAX_SAMPLES_LIMIT": 500000,  # Chunks más pequeños para procesamiento rápido
        "USE_MULTI_BAND": True,    # Usar análisis multi-banda
        "RFI_ENABLE_ALL_FILTERS": True,  # Máxima limpieza RFI
    },
    "recommended_for": "Surveys de FRB, búsqueda de eventos débiles"
}

# CASO 2: Procesamiento de archivos históricos con análisis detallado
CASE_2_ARCHIVE_PROCESSING = {
    "description": "Procesamiento detallado de archivos históricos",
    "parameters": {
        "DET_PROB": 0.1,           # Balanced para evitar falsos positivos
        "SNR_THRESH": 3.0,         # Umbral estándar
        "DM_RANGE_FACTOR": 0.3,    # Rango más amplio para exploración
        "MAX_SAMPLES_LIMIT": 2000000,  # Chunks grandes para máxima eficiencia
        "USE_MULTI_BAND": True,    # Análisis completo multi-banda
        "RFI_SAVE_DIAGNOSTICS": True,  # Guardar diagnósticos RFI
        "SLICE_DURATION_SECONDS": 0.032,  # Resolución temporal estándar
    },
    "recommended_for": "Análisis retrospectivo, estudios de población"
}

# CASO 3: Búsqueda exploratoria rápida
CASE_3_QUICK_SURVEY = {
    "description": "Búsqueda rápida para identificar candidatos prometedores",
    "parameters": {
        "DET_PROB": 0.15,          # Menos sensible pero más rápido
        "SNR_THRESH": 4.0,         # Umbral más alto
        "DM_RANGE_FACTOR": 0.4,    # Rango amplio para exploración
        "MAX_SAMPLES_LIMIT": 1000000,  # Chunks medianos
        "USE_MULTI_BAND": False,   # Solo banda completa
        "RFI_ENABLE_ALL_FILTERS": False,  # RFI básico
        "SLICE_DURATION_SECONDS": 0.064,  # Resolución temporal menor
    },
    "recommended_for": "Surveys iniciales, identificación rápida de candidatos"
}

# CASO 4: Análisis de pulsar conocido
CASE_4_PULSAR_ANALYSIS = {
    "description": "Análisis detallado de pulsar conocido",
    "parameters": {
        "DET_PROB": 0.05,          # Alta sensibilidad
        "SNR_THRESH": 2.0,         # Umbral muy bajo
        "DM_min": 20,              # DM mínimo conocido del pulsar
        "DM_max": 80,              # DM máximo esperado
        "DM_RANGE_FACTOR": 0.15,   # Rango muy estrecho
        "SLICE_DURATION_SECONDS": 0.016,  # Alta resolución temporal
        "USE_MULTI_BAND": True,    # Análisis completo
        "RFI_ENABLE_ALL_FILTERS": True,  # Limpieza completa
    },
    "recommended_for": "Estudios de pulsars, análisis de perfil de pulso"
}

# CASO 5: Procesamiento de datos con mucho RFI
CASE_5_HIGH_RFI = {
    "description": "Procesamiento en ambiente con mucha interferencia",
    "parameters": {
        "DET_PROB": 0.2,           # Menos sensible para evitar RFI
        "SNR_THRESH": 5.0,         # Umbral alto
        "RFI_ENABLE_ALL_FILTERS": True,  # Todos los filtros RFI
        "RFI_FREQ_SIGMA_THRESH": 3.0,    # Más agresivo en frecuencia
        "RFI_TIME_SIGMA_THRESH": 3.0,    # Más agresivo en tiempo
        "RFI_INTERPOLATE_MASKED": True,  # Interpolar datos enmascarados
        "RFI_SAVE_DIAGNOSTICS": True,   # Guardar diagnósticos
        "DM_RANGE_FACTOR": 0.25,   # Rango intermedio
    },
    "recommended_for": "Observaciones urbanas, datos con mucha RFI"
}

# =============================================================================
# CONFIGURACIÓN POR TIPO DE TELESCOPIO
# =============================================================================

# Effelsberg 100m - Configuración típica
EFFELSBERG_CONFIG = {
    "freq_range": "1200-1500 MHz",  # Banda L
    "time_resolution": "40-100 μs",
    "typical_dm_range": "0-1000 pc cm⁻³",
    "recommended_settings": {
        "SLICE_DURATION_SECONDS": 0.032,
        "DM_RANGE_FACTOR": 0.3,
        "SNR_THRESH": 3.0,
        "MAX_SAMPLES_LIMIT": 2000000,
    }
}

# Arecibo-like - Configuración para telescopios grandes
LARGE_TELESCOPE_CONFIG = {
    "freq_range": "1300-1700 MHz",
    "time_resolution": "10-50 μs",
    "typical_dm_range": "0-2000 pc cm⁻³",
    "recommended_settings": {
        "SLICE_DURATION_SECONDS": 0.016,  # Mayor resolución temporal
        "DM_RANGE_FACTOR": 0.2,           # Rango más estrecho
        "SNR_THRESH": 2.5,                # Más sensible
        "MAX_SAMPLES_LIMIT": 1000000,     # Chunks más pequeños
    }
}

# =============================================================================
# EJEMPLOS DE CONFIGURACIÓN PASO A PASO
# =============================================================================

def configure_for_frb_survey():
    """
    Configuración típica para survey de FRB.
    
    Esta configuración está optimizada para detectar FRBs en un survey
    general, balanceando sensibilidad y velocidad de procesamiento.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "DRAFTS"))
    
    from DRAFTS import config
    
    # Configurar rutas
    config.DATA_DIR = Path("./Data")
    config.RESULTS_DIR = Path("./Results/FRB_Survey")
    
    # Configurar detección
    config.DET_PROB = 0.1           # Balanced
    config.SNR_THRESH = 3.0         # Estándar
    config.DM_min = 0               # Rango completo
    config.DM_max = 1500            # Expandido para FRBs lejanos
    
    # Configurar visualización
    config.DM_DYNAMIC_RANGE_ENABLE = True
    config.DM_RANGE_FACTOR = 0.3
    config.DM_PLOT_MARGIN_FACTOR = 0.25
    
    # Configurar procesamiento
    config.USE_MULTI_BAND = True
    config.MAX_SAMPLES_LIMIT = 1500000
    config.ENABLE_CHUNK_PROCESSING = True
    
    # Configurar RFI
    config.RFI_ENABLE_ALL_FILTERS = True
    config.RFI_SAVE_DIAGNOSTICS = True
    
    print("✅ Configuración para FRB survey aplicada")
    print(f"   - Rango DM: {config.DM_min}-{config.DM_max} pc cm⁻³")
    print(f"   - Umbral detección: {config.DET_PROB}")
    print(f"   - Umbral SNR: {config.SNR_THRESH}")
    print(f"   - Multi-banda: {config.USE_MULTI_BAND}")

def configure_for_pulsar_study():
    """
    Configuración para estudio detallado de pulsar.
    
    Esta configuración maximiza la sensibilidad y resolución temporal
    para estudios detallados de pulsars conocidos.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "DRAFTS"))
    
    from DRAFTS import config
    
    # Configurar para alta sensibilidad
    config.DET_PROB = 0.05          # Muy sensible
    config.SNR_THRESH = 2.0         # Umbral bajo
    config.SLICE_DURATION_SECONDS = 0.016  # Alta resolución
    
    # Configurar visualización detallada
    config.DM_RANGE_FACTOR = 0.15   # Rango estrecho
    config.DM_PLOT_MARGIN_FACTOR = 0.2
    
    # Configurar procesamiento detallado
    config.USE_MULTI_BAND = True
    config.MAX_SAMPLES_LIMIT = 800000  # Chunks más pequeños
    
    # Configurar RFI completo
    config.RFI_ENABLE_ALL_FILTERS = True
    config.RFI_INTERPOLATE_MASKED = True
    config.RFI_SAVE_DIAGNOSTICS = True
    
    print("✅ Configuración para estudio de pulsar aplicada")
    print(f"   - Resolución temporal: {config.SLICE_DURATION_SECONDS*1000:.1f} ms")
    print(f"   - Sensibilidad: {config.DET_PROB}")
    print(f"   - Factor rango DM: {config.DM_RANGE_FACTOR}")

# =============================================================================
# GUÍA DE TROUBLESHOOTING
# =============================================================================

TROUBLESHOOTING_GUIDE = """
PROBLEMAS COMUNES Y SOLUCIONES:

1. MUCHOS FALSOS POSITIVOS:
   - Aumentar DET_PROB (ej: 0.1 → 0.15)
   - Aumentar SNR_THRESH (ej: 3.0 → 4.0)
   - Habilitar RFI_ENABLE_ALL_FILTERS = True

2. POCAS DETECCIONES:
   - Reducir DET_PROB (ej: 0.1 → 0.05)
   - Reducir SNR_THRESH (ej: 3.0 → 2.5)
   - Verificar rango DM (DM_min, DM_max)

3. CANDIDATOS "PEGADOS" EN PLOTS:
   - Aumentar DM_RANGE_FACTOR (ej: 0.2 → 0.3)
   - Aumentar DM_PLOT_MARGIN_FACTOR (ej: 0.2 → 0.3)
   - Verificar DM_DYNAMIC_RANGE_ENABLE = True

4. PROCESAMIENTO LENTO:
   - Reducir MAX_SAMPLES_LIMIT (ej: 2M → 1M)
   - Desactivar USE_MULTI_BAND si no es necesario
   - Reducir RFI_ENABLE_ALL_FILTERS a False

5. PROBLEMAS DE MEMORIA:
   - Reducir MAX_SAMPLES_LIMIT (ej: 2M → 500K)
   - Verificar ENABLE_CHUNK_PROCESSING = True
   - Aumentar CHUNK_OVERLAP_SAMPLES si hay problemas

6. MALA CALIDAD EN PLOTS:
   - Ajustar DM_RANGE_FACTOR para mejor centrado
   - Verificar SLICE_DURATION_SECONDS apropiado
   - Habilitar RFI_SAVE_DIAGNOSTICS para análisis
"""

print(__doc__)
print(TROUBLESHOOTING_GUIDE)
