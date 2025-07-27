#!/usr/bin/env python3
"""
Guía Práctica: Cómo Usar las Variables del Sistema DM Dinámico

Este script muestra ejemplos prácticos de cómo configurar y usar cada variable
del sistema de DM dinámico para diferentes tipos de observaciones.
"""

from DRAFTS import config
import numpy as np

def explain_dm_dynamic_variables():
    """Explica cada variable del sistema DM dinámico con ejemplos prácticos."""
    
    print("=" * 70)
    print("🎯 GUÍA DE USO: Variables del Sistema DM Dinámico")
    print("=" * 70)
    
    # 1. DM_DYNAMIC_RANGE_ENABLE
    print("\n1. 🔧 DM_DYNAMIC_RANGE_ENABLE")
    print("   Descripción: Activa/desactiva todo el sistema DM dinámico")
    print("   Valor actual:", getattr(config, 'DM_DYNAMIC_RANGE_ENABLE', True))
    print("   Uso:")
    print("   • True  = Los plots se centran automáticamente en candidatos")
    print("   • False = Usa rango completo DM_min a DM_max (comportamiento original)")
    print("\n   💡 Ejemplo de cuándo cambiar:")
    print("   config.DM_DYNAMIC_RANGE_ENABLE = False  # Para análisis de survey completo")
    print("   config.DM_DYNAMIC_RANGE_ENABLE = True   # Para análisis de candidatos individuales")
    
    # 2. DM_RANGE_FACTOR
    print("\n" + "-" * 50)
    print("2. 📏 DM_RANGE_FACTOR")
    print("   Descripción: Controla qué tan 'zoomeado' está el plot")
    print("   Valor actual:", getattr(config, 'DM_RANGE_FACTOR', 0.2))
    print("   Rango recomendado: 0.1 (mucho zoom) a 0.4 (poco zoom)")
    print("\n   📊 Ejemplos prácticos:")
    
    dm_candidato = 450.0  # Ejemplo: candidato en DM = 450
    
    for factor in [0.1, 0.2, 0.3]:
        rango_width = dm_candidato * factor * 2
        dm_min = dm_candidato - rango_width/2
        dm_max = dm_candidato + rango_width/2
        zoom_factor = 1024 / rango_width
        
        print(f"   Factor {factor}: Rango {dm_min:.0f}-{dm_max:.0f} pc cm⁻³ (zoom {zoom_factor:.1f}x)")
    
    print("\n   💡 Recomendaciones por tipo:")
    print("   • FRBs individuales: 0.15-0.2 (zoom alto para detalles)")
    print("   • Múltiples candidatos: 0.25-0.3 (zoom medio)")
    print("   • Análisis exploratorio: 0.3-0.4 (zoom bajo)")
    
    # 3. DM_RANGE_MIN_WIDTH y DM_RANGE_MAX_WIDTH
    print("\n" + "-" * 50)
    print("3. 📐 DM_RANGE_MIN_WIDTH y DM_RANGE_MAX_WIDTH")
    print("   Descripción: Limitan el ancho mínimo y máximo del rango DM")
    print("   Valores actuales:")
    print("   • MIN_WIDTH:", getattr(config, 'DM_RANGE_MIN_WIDTH', 50.0), "pc cm⁻³")
    print("   • MAX_WIDTH:", getattr(config, 'DM_RANGE_MAX_WIDTH', 200.0), "pc cm⁻³")
    print("\n   📊 Cómo funcionan:")
    print("   • Si el factor da un rango < MIN_WIDTH → usa MIN_WIDTH")
    print("   • Si el factor da un rango > MAX_WIDTH → usa MAX_WIDTH")
    print("   • Evita rangos demasiado estrechos o amplios")
    
    print("\n   💡 Configuraciones recomendadas:")
    print("   # Para FRBs de DM bajo (< 200 pc cm⁻³)")
    print("   config.DM_RANGE_MIN_WIDTH = 30.0")
    print("   config.DM_RANGE_MAX_WIDTH = 100.0")
    print("\n   # Para FRBs de DM alto (> 600 pc cm⁻³)")
    print("   config.DM_RANGE_MIN_WIDTH = 80.0")
    print("   config.DM_RANGE_MAX_WIDTH = 300.0")
    
    # 4. DM_RANGE_ADAPTIVE
    print("\n" + "-" * 50)
    print("4. 🧠 DM_RANGE_ADAPTIVE")
    print("   Descripción: Ajusta el rango basado en la confianza de detección")
    print("   Valor actual:", getattr(config, 'DM_RANGE_ADAPTIVE', True))
    print("\n   🎯 Cómo funciona:")
    print("   • Confianza alta (>0.9) → rango más estrecho (más zoom)")
    print("   • Confianza baja (<0.7) → rango más amplio (menos zoom)")
    print("   • Permite visualización adaptiva según calidad de detección")
    
    print("\n   💡 Cuándo usar:")
    print("   • True: Para análisis automático (recomendado)")
    print("   • False: Para rangos fijos independientes de confianza")
    
    # 5. DM_RANGE_DEFAULT_VISUALIZATION
    print("\n" + "-" * 50)
    print("5. 🎨 DM_RANGE_DEFAULT_VISUALIZATION")
    print("   Descripción: Tipo de visualización que define configuración automática")
    print("   Valor actual:", getattr(config, 'DM_RANGE_DEFAULT_VISUALIZATION', 'detailed'))
    print("\n   🎯 Opciones disponibles:")
    
    viz_configs = {
        'composite': {'factor': 0.15, 'min_w': 40, 'max_w': 150, 'uso': 'Resúmenes multi-candidato'},
        'patch': {'factor': 0.1, 'min_w': 30, 'max_w': 100, 'uso': 'Patches individuales detallados'},
        'detailed': {'factor': 0.2, 'min_w': 50, 'max_w': 200, 'uso': 'Análisis detallado (recomendado)'},
        'overview': {'factor': 0.3, 'min_w': 80, 'max_w': 300, 'uso': 'Vista general de múltiples candidatos'}
    }
    
    for viz_type, params in viz_configs.items():
        print(f"   • '{viz_type}': Factor {params['factor']}, Ancho {params['min_w']}-{params['max_w']}")
        print(f"     → Uso: {params['uso']}")
    
    print("\n" + "=" * 70)
    print("🚀 EJEMPLOS DE CONFIGURACIÓN COMPLETA")
    print("=" * 70)

def show_configuration_examples():
    """Muestra ejemplos completos de configuración para diferentes casos de uso."""
    
    print("\n📝 EJEMPLO 1: Survey de FRBs de DM bajo (Arecibo, < 200 pc cm⁻³)")
    print("-" * 60)
    print("config.DM_DYNAMIC_RANGE_ENABLE = True")
    print("config.DM_RANGE_FACTOR = 0.25           # Zoom moderado")
    print("config.DM_RANGE_MIN_WIDTH = 30.0        # Mínimo pequeño para DMs bajos")
    print("config.DM_RANGE_MAX_WIDTH = 120.0       # Máximo adecuado")
    print("config.DM_RANGE_ADAPTIVE = True")
    print("config.DM_RANGE_DEFAULT_VISUALIZATION = 'detailed'")
    
    print("\n📝 EJEMPLO 2: FRBs extragalácticos de DM alto (> 500 pc cm⁻³)")
    print("-" * 60)
    print("config.DM_DYNAMIC_RANGE_ENABLE = True")
    print("config.DM_RANGE_FACTOR = 0.15           # Zoom alto para precisión")
    print("config.DM_RANGE_MIN_WIDTH = 80.0        # Mínimo mayor para DMs altos")
    print("config.DM_RANGE_MAX_WIDTH = 300.0       # Máximo amplio")
    print("config.DM_RANGE_ADAPTIVE = True")
    print("config.DM_RANGE_DEFAULT_VISUALIZATION = 'detailed'")
    
    print("\n📝 EJEMPLO 3: Análisis de múltiples candidatos simultáneos")
    print("-" * 60)
    print("config.DM_DYNAMIC_RANGE_ENABLE = True")
    print("config.DM_RANGE_FACTOR = 0.3            # Zoom amplio para ver todos")
    print("config.DM_RANGE_MIN_WIDTH = 100.0       # Mínimo amplio")
    print("config.DM_RANGE_MAX_WIDTH = 400.0       # Máximo muy amplio")
    print("config.DM_RANGE_ADAPTIVE = False        # Sin adaptación")
    print("config.DM_RANGE_DEFAULT_VISUALIZATION = 'overview'")
    
    print("\n📝 EJEMPLO 4: Modo clásico (desactivar DM dinámico)")
    print("-" * 60)
    print("config.DM_DYNAMIC_RANGE_ENABLE = False  # Usar rango completo DM_min-DM_max")
    print("# Las demás variables se ignoran cuando está desactivado")
    
    print("\n📝 EJEMPLO 5: Patches individuales de alta precisión")
    print("-" * 60)
    print("config.DM_DYNAMIC_RANGE_ENABLE = True")
    print("config.DM_RANGE_FACTOR = 0.1            # Máximo zoom")
    print("config.DM_RANGE_MIN_WIDTH = 25.0        # Mínimo muy pequeño")
    print("config.DM_RANGE_MAX_WIDTH = 80.0        # Máximo pequeño")
    print("config.DM_RANGE_ADAPTIVE = True")
    print("config.DM_RANGE_DEFAULT_VISUALIZATION = 'patch'")

def show_practical_workflow():
    """Muestra el flujo de trabajo práctico."""
    
    print("\n" + "=" * 70)
    print("⚙️ FLUJO DE TRABAJO PRÁCTICO")
    print("=" * 70)
    
    print("\n🔧 PASO 1: Configurar antes de procesar")
    print("   1. Edita DRAFTS/config.py")
    print("   2. Ajusta las variables según tu tipo de observación")
    print("   3. Guarda el archivo")
    
    print("\n🚀 PASO 2: Ejecutar el pipeline (¡automático!)")
    print("   • El sistema detecta candidatos automáticamente")
    print("   • Calcula el DM óptimo del mejor candidato")
    print("   • Ajusta los ejes DM de todos los plots")
    print("   • NO necesitas cambiar tu código existente")
    
    print("\n📊 PASO 3: Verificar resultados")
    print("   • Los plots muestran '(auto)' en el título si usan DM dinámico")
    print("   • Los plots muestran '(full)' si usan rango completo")
    print("   • El rango DM aparece en el título del plot")
    
    print("\n🔍 PASO 4: Ajustar si es necesario")
    print("   • Si el zoom es demasiado: aumenta DM_RANGE_FACTOR")
    print("   • Si el zoom es poco: disminuye DM_RANGE_FACTOR")
    print("   • Si hay problemas: pon DM_DYNAMIC_RANGE_ENABLE = False")

def show_troubleshooting():
    """Muestra solución de problemas comunes."""
    
    print("\n" + "=" * 70)
    print("🔧 SOLUCIÓN DE PROBLEMAS COMUNES")
    print("=" * 70)
    
    print("\n❓ PROBLEMA: 'El plot no se centra en mi candidato'")
    print("   ✅ SOLUCIÓN:")
    print("   • Verifica: config.DM_DYNAMIC_RANGE_ENABLE = True")
    print("   • El sistema se centra en el candidato con MAYOR confianza")
    print("   • Si tienes múltiples candidatos, mejora la confianza del principal")
    
    print("\n❓ PROBLEMA: 'El rango es demasiado estrecho'")
    print("   ✅ SOLUCIÓN:")
    print("   • Aumenta: config.DM_RANGE_FACTOR (ej: de 0.2 a 0.3)")
    print("   • Aumenta: config.DM_RANGE_MIN_WIDTH (ej: de 50 a 100)")
    
    print("\n❓ PROBLEMA: 'El rango es demasiado amplio'")
    print("   ✅ SOLUCIÓN:")
    print("   • Disminuye: config.DM_RANGE_FACTOR (ej: de 0.3 a 0.15)")
    print("   • Disminuye: config.DM_RANGE_MAX_WIDTH (ej: de 200 a 100)")
    
    print("\n❓ PROBLEMA: 'Quiero el comportamiento original'")
    print("   ✅ SOLUCIÓN:")
    print("   • config.DM_DYNAMIC_RANGE_ENABLE = False")
    print("   • Los plots volverán a usar DM_min a DM_max completo")
    
    print("\n❓ PROBLEMA: 'No veo candidatos en mis datos'")
    print("   ✅ SOLUCIÓN:")
    print("   • Sin candidatos → el sistema usa automáticamente rango completo")
    print("   • Esto es normal y esperado")
    print("   • El sistema es robusto y maneja este caso automáticamente")

if __name__ == "__main__":
    print("Iniciando Guía de Uso del Sistema DM Dinámico...")
    
    explain_dm_dynamic_variables()
    show_configuration_examples()
    show_practical_workflow()
    show_troubleshooting()
    
    print("\n" + "=" * 70)
    print("✅ RESUMEN: El sistema funciona AUTOMÁTICAMENTE")
    print("✅ Solo necesitas configurar las variables una vez")
    print("✅ Tu código existente NO necesita cambios")
    print("✅ Los plots se optimizan automáticamente para cada candidato")
    print("=" * 70)
