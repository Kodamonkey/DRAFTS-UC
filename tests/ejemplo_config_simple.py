#!/usr/bin/env python3
"""
Ejemplo de Uso de la Configuración Simplificada
==============================================

Este script demuestra cómo usar la nueva configuración simplificada
para procesar datos de FRB.
"""

def ejemplo_configuracion_simple():
    """Demuestra la configuración simplificada."""
    
    print("=== EJEMPLO: Configuración Simplificada para FRB ===")
    
    # 1. Mostrar configuración simple actual
    print("\n1. Configuración actual del usuario:")
    from DRAFTS.config_auto import print_user_config
    print_user_config()
    
    # 2. Mostrar cómo cambiar parámetros
    print("\n2. Ejemplo de modificación de parámetros:")
    print("   Edita config_simple.py con estos valores:")
    print("   ")
    print("   # Para búsqueda de alta sensibilidad:")
    print("   DM_min = 50")
    print("   DM_max = 2000")
    print("   DET_PROB = 0.05")
    print("   FRB_TARGETS = ['FRB20180301', 'B0355+54']")
    
    # 3. Cargar configuración completa automáticamente
    print("\n3. Carga automática de configuración completa:")
    from DRAFTS.config_auto import get_complete_config
    config = get_complete_config()
    print(f"   Total de parámetros configurados: {len(config)}")
    print(f"   Rango DM configurado: {config['DM_min']} - {config['DM_max']} pc cm⁻³")
    print(f"   Sensibilidad: {config['DET_PROB']}")
    print(f"   Visualización dinámica: {'Activada' if config['DM_DYNAMIC_RANGE_ENABLE'] else 'Desactivada'}")
    
    # 4. Mostrar diferencias con configuración anterior
    print("\n4. Comparación con configuración anterior:")
    print("   ANTES: 50+ parámetros en config.py")
    print("   AHORA: 6 parámetros en config_simple.py")
    print("   REDUCCIÓN: 88% menos parámetros para configurar")
    
    return config

def ejemplo_uso_pipeline():
    """Demuestra cómo usar el pipeline con la configuración simplificada."""
    
    print("\n=== EJEMPLO: Uso del Pipeline ===")
    
    # Cargar configuración automática
    from DRAFTS.config_auto import *
    
    print(f"Directorio de datos: {DATA_DIR}")
    print(f"Targets a procesar: {FRB_TARGETS}")
    print(f"Rango DM: {DM_min} - {DM_max} pc cm⁻³")
    print(f"Sensibilidad: {DET_PROB}")
    
    # Mostrar configuración automática relevante
    print("\nConfiguración automática aplicada:")
    print(f"- Procesamiento multi-banda: {'Sí' if USE_MULTI_BAND else 'No'}")
    print(f"- Visualización dinámica: {'Sí' if DM_DYNAMIC_RANGE_ENABLE else 'No'}")
    print(f"- Slice temporal automático: {'Sí' if SLICE_LEN_AUTO else 'No'}")
    
    # Simular procesamiento
    print("\nSimulando procesamiento:")
    print("1. Cargando metadatos de archivos...")
    print("2. Calculando slices temporales automáticamente...")
    print("3. Configurando rango DM dinámico para visualización...")
    print("4. Iniciando detección con modelos ResNet...")
    print("5. Generando plots centrados en candidatos...")
    print("✓ Procesamiento completo")

def ejemplo_casos_uso():
    """Muestra ejemplos de configuración para diferentes casos."""
    
    print("\n=== EJEMPLOS DE CASOS DE USO ===")
    
    casos = [
        {
            "nombre": "Búsqueda Exploratoria",
            "descripcion": "Para explorar nuevas regiones del espacio DM",
            "config": {
                "DM_min": 0,
                "DM_max": 3000,
                "DET_PROB": 0.1,
                "FRB_TARGETS": ["survey_field_1", "survey_field_2"]
            }
        },
        {
            "nombre": "Alta Precisión",
            "descripcion": "Para confirmar detecciones con alta confianza",
            "config": {
                "DM_min": 100,
                "DM_max": 1000,
                "DET_PROB": 0.05,
                "FRB_TARGETS": ["FRB20180301", "FRB20201124"]
            }
        },
        {
            "nombre": "Procesamiento Rápido",
            "descripcion": "Para análisis rápido de muchos archivos",
            "config": {
                "DM_min": 200,
                "DM_max": 800,
                "DET_PROB": 0.15,
                "FRB_TARGETS": ["batch_001", "batch_002", "batch_003"]
            }
        },
        {
            "nombre": "FRB Conocido",
            "descripcion": "Para análisis detallado de FRB con DM conocido",
            "config": {
                "DM_min": 300,
                "DM_max": 400,
                "DET_PROB": 0.05,
                "FRB_TARGETS": ["FRB20180301_repeat"]
            }
        }
    ]
    
    for i, caso in enumerate(casos, 1):
        print(f"\n{i}. {caso['nombre']}")
        print(f"   {caso['descripcion']}")
        print("   Configuración:")
        for key, value in caso['config'].items():
            print(f"     {key} = {value}")

def main():
    """Ejecuta todos los ejemplos."""
    
    print("Script de Ejemplo - Configuración Simplificada FRB")
    print("=" * 50)
    
    # Ejecutar ejemplos
    config = ejemplo_configuracion_simple()
    ejemplo_uso_pipeline()
    ejemplo_casos_uso()
    
    print("\n=== RESUMEN ===")
    print("✓ Configuración simplificada: solo 6 parámetros esenciales")
    print("✓ Configuración automática: 40+ parámetros técnicos")
    print("✓ Visualización dinámica: automática y centrada en candidatos")
    print("✓ Casos de uso: ejemplos para diferentes situaciones")
    print("\nPara usar el sistema:")
    print("1. Edita config_simple.py con tus parámetros")
    print("2. Ejecuta: from DRAFTS.config_auto import *")
    print("3. Procesa tus datos normalmente")

if __name__ == "__main__":
    main()
