#!/usr/bin/env python3
"""
Guía de Uso de Switches de Control - Config.py
==============================================

Este script demuestra cómo usar los switches de control para alternar
entre configuración manual y automática.
"""

def mostrar_configuracion_actual():
    """Muestra el estado actual de los switches de control."""
    from DRAFTS import config
    
    print("=== ESTADO ACTUAL DE SWITCHES DE CONTROL ===")
    print(f"🔧 SLICE_LEN_AUTO = {config.SLICE_LEN_AUTO}")
    print(f"🔧 SLICE_LEN_INTELLIGENT = {config.SLICE_LEN_INTELLIGENT}")
    print(f"🔧 SLICE_LEN_OVERRIDE_MANUAL = {config.SLICE_LEN_OVERRIDE_MANUAL}")
    print()
    print(f"📊 DM_DYNAMIC_RANGE_ENABLE = {config.DM_DYNAMIC_RANGE_ENABLE}")
    print(f"📊 DM_RANGE_ADAPTIVE = {config.DM_RANGE_ADAPTIVE}")
    print()
    print(f"🛡️ RFI_ENABLE_ALL_FILTERS = {config.RFI_ENABLE_ALL_FILTERS}")
    print(f"🛡️ RFI_INTERPOLATE_MASKED = {config.RFI_INTERPOLATE_MASKED}")
    print(f"🛡️ RFI_SAVE_DIAGNOSTICS = {config.RFI_SAVE_DIAGNOSTICS}")
    print()
    
    print("=== VALORES DE CONFIGURACIÓN MANUAL ===")
    print(f"⏱️ SLICE_LEN = {config.SLICE_LEN}")
    print(f"📈 DM_RANGE_FACTOR = {config.DM_RANGE_FACTOR}")
    print(f"📏 DM_PLOT_MARGIN_FACTOR = {config.DM_PLOT_MARGIN_FACTOR}")
    print("=" * 50)

def ejemplo_configuracion_manual():
    """Muestra cómo configurar para modo manual."""
    print("\n🔧 CONFIGURACIÓN PARA MODO MANUAL:")
    print("Edita config.py con estos valores:")
    print()
    print("# Para SLICE TEMPORAL manual:")
    print("SLICE_LEN_AUTO = False")
    print("SLICE_LEN_INTELLIGENT = False")
    print("SLICE_LEN_OVERRIDE_MANUAL = False")
    print("SLICE_LEN = 64  # Tu valor personalizado")
    print()
    print("# Para RANGO DM manual:")
    print("DM_DYNAMIC_RANGE_ENABLE = False")
    print("DM_RANGE_ADAPTIVE = False")
    print("DM_RANGE_FACTOR = 0.4  # Tu factor personalizado")
    print()
    print("# Para RFI básico:")
    print("RFI_ENABLE_ALL_FILTERS = False")
    print("RFI_INTERPOLATE_MASKED = False")
    print("RFI_SAVE_DIAGNOSTICS = False")

def ejemplo_configuracion_automatica():
    """Muestra cómo configurar para modo automático."""
    print("\n🤖 CONFIGURACIÓN PARA MODO AUTOMÁTICO:")
    print("Edita config.py con estos valores:")
    print()
    print("# Para SLICE TEMPORAL automático:")
    print("SLICE_LEN_AUTO = True")
    print("SLICE_LEN_INTELLIGENT = True")
    print("SLICE_LEN_OVERRIDE_MANUAL = True")
    print()
    print("# Para RANGO DM automático:")
    print("DM_DYNAMIC_RANGE_ENABLE = True")
    print("DM_RANGE_ADAPTIVE = True")
    print()
    print("# Para RFI completo:")
    print("RFI_ENABLE_ALL_FILTERS = True")
    print("RFI_INTERPOLATE_MASKED = True")
    print("RFI_SAVE_DIAGNOSTICS = True")

def casos_de_uso():
    """Muestra casos de uso típicos."""
    print("\n📋 CASOS DE USO TÍPICOS:")
    
    print("\n🔬 DESARROLLO Y PRUEBAS:")
    print("  - Slice temporal: MANUAL (control exacto)")
    print("  - Rango DM: MANUAL (plots fijos para comparar)")
    print("  - RFI: BÁSICO (procesamiento rápido)")
    
    print("\n🏭 PRODUCCIÓN AUTOMÁTICA:")
    print("  - Slice temporal: AUTOMÁTICO (optimizado por archivo)")
    print("  - Rango DM: AUTOMÁTICO (zoom en candidatos)")
    print("  - RFI: COMPLETO (limpieza máxima)")
    
    print("\n🐛 DEBUG Y AJUSTE FINO:")
    print("  - Slice temporal: MANUAL (valores específicos)")
    print("  - Rango DM: MANUAL (rangos controlados)")
    print("  - RFI: COMPLETO + DIAGNÓSTICOS (ver qué se filtra)")
    
    print("\n⚡ PROCESAMIENTO RÁPIDO:")
    print("  - Slice temporal: AUTOMÁTICO (optimizado)")
    print("  - Rango DM: AUTOMÁTICO (sin configuración extra)")
    print("  - RFI: BÁSICO (solo filtros esenciales)")

def main():
    """Ejecuta la guía completa."""
    print("Guía de Uso - Switches de Control")
    print("=" * 40)
    
    mostrar_configuracion_actual()
    ejemplo_configuracion_manual()
    ejemplo_configuracion_automatica()
    casos_de_uso()
    
    print("\n✅ CÓMO CAMBIAR CONFIGURACIÓN:")
    print("1. Edita DRAFTS/config.py")
    print("2. Modifica los switches en la sección 'SWITCHES DE CONTROL'")
    print("3. Ajusta los valores en las secciones correspondientes")
    print("4. Guarda el archivo")
    print("5. Reinicia tu script para aplicar cambios")

if __name__ == "__main__":
    main()
