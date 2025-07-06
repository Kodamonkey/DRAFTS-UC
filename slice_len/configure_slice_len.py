#!/usr/bin/env python3
"""
Herramienta interactiva para configurar SLICE_LEN dinámico.

Permite al usuario configurar fácilmente la duración temporal deseada
y actualiza automáticamente el archivo config.py.
"""

import sys
from pathlib import Path

# Configurar path
sys.path.append(str(Path(__file__).parent))

def interactive_slice_len_config():
    """Configuración interactiva de SLICE_LEN."""
    
    print("🛠️  === CONFIGURADOR INTERACTIVO SLICE_LEN ===\n")
    
    print("Este asistente te ayudará a configurar SLICE_LEN de manera intuitiva")
    print("basándote en la duración temporal que deseas para cada slice.\n")
    
    # Paso 1: Tipo de señal
    print("🎯 PASO 1: ¿Qué tipo de señales buscas?")
    print("   1. Pulsos muy cortos (< 20ms) - Ej: pulsars rápidos")
    print("   2. FRBs típicos (20-100ms) - Configuración estándar")
    print("   3. Señales largas (100-500ms) - Ej: FRBs dispersos")
    print("   4. Señales muy dispersas (> 500ms) - Casos extremos")
    print("   5. Configuración personalizada")
    
    while True:
        try:
            choice = input("\nSelecciona una opción (1-5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                break
            print("❌ Opción no válida. Usa números 1-5.")
        except KeyboardInterrupt:
            print("\n❌ Configuración cancelada.")
            return
    
    # Determinar duración según elección
    signal_types = {
        '1': ('short', 0.016, 'Pulsos muy cortos'),
        '2': ('medium', 0.032, 'FRBs típicos'),
        '3': ('long', 0.064, 'Señales largas'),
        '4': ('dispersed', 0.128, 'Señales muy dispersas'),
    }
    
    if choice in signal_types:
        signal_type, duration, description = signal_types[choice]
        print(f"\n✅ Seleccionado: {description}")
        print(f"   Duración recomendada: {duration:.3f} s ({duration*1000:.0f} ms)")
    else:
        # Configuración personalizada
        print("\n⚙️  CONFIGURACIÓN PERSONALIZADA:")
        while True:
            try:
                duration = float(input("Introduce duración deseada en segundos (ej: 0.032): "))
                if 0.001 <= duration <= 1.0:
                    break
                print("❌ Duración debe estar entre 0.001 y 1.0 segundos.")
            except ValueError:
                print("❌ Valor no válido. Usa formato decimal (ej: 0.032).")
            except KeyboardInterrupt:
                print("\n❌ Configuración cancelada.")
                return
        
        description = f"Personalizada ({duration*1000:.1f} ms)"
    
    # Paso 2: Configuración avanzada
    print(f"\n🔧 PASO 2: Configuración avanzada")
    
    # Límites de SLICE_LEN
    print("   ¿Quieres usar límites automáticos para SLICE_LEN? (recomendado)")
    use_auto_limits = input("   [Y/n]: ").strip().lower() not in ['n', 'no']
    
    if use_auto_limits:
        if duration <= 0.025:  # Señales cortas
            min_slice = 8
            max_slice = 128
        elif duration <= 0.075:  # Señales medias
            min_slice = 16
            max_slice = 256
        else:  # Señales largas
            min_slice = 32
            max_slice = 512
        
        print(f"   ✅ Límites automáticos: SLICE_LEN entre {min_slice} y {max_slice}")
    else:
        while True:
            try:
                min_slice = int(input("   SLICE_LEN mínimo (ej: 16): "))
                max_slice = int(input("   SLICE_LEN máximo (ej: 512): "))
                if min_slice < max_slice and min_slice >= 4 and max_slice <= 2048:
                    break
                print("❌ Valores no válidos. Min < Max, y dentro de 4-2048.")
            except ValueError:
                print("❌ Usa números enteros.")
            except KeyboardInterrupt:
                print("\n❌ Configuración cancelada.")
                return
    
    # Paso 3: Mostrar resumen y confirmar
    print(f"\n📋 RESUMEN DE CONFIGURACIÓN:")
    print(f"   • Tipo de señal: {description}")
    print(f"   • Duración objetivo: {duration:.3f} s ({duration*1000:.1f} ms)")
    print(f"   • SLICE_LEN automático: ✅ Habilitado")
    print(f"   • Límites SLICE_LEN: {min_slice} - {max_slice}")
    print(f"   • Preferir potencias de 2: ✅ Sí")
    
    confirm = input("\n¿Aplicar esta configuración? [Y/n]: ").strip().lower()
    if confirm in ['n', 'no']:
        print("❌ Configuración cancelada.")
        return
    
    # Paso 4: Actualizar config.py
    try:
        update_config_file(duration, min_slice, max_slice)
        print(f"\n✅ CONFIGURACIÓN APLICADA EXITOSAMENTE")
        
        # Mostrar ejemplo de uso
        print(f"\n🚀 PRÓXIMOS PASOS:")
        print(f"   1. Ejecuta tu pipeline normalmente: python main.py")
        print(f"   2. El sistema calculará automáticamente SLICE_LEN óptimo")
        print(f"   3. Revisa los logs para ver el valor calculado")
        
        # Mostrar comando para análisis
        print(f"\n🔍 PARA ANALIZAR LA CONFIGURACIÓN:")
        print(f"   python -c \"from DRAFTS.slice_len_utils import print_slice_len_analysis; from DRAFTS import config; print_slice_len_analysis(config)\"")
        
    except Exception as e:
        print(f"\n❌ Error actualizando configuración: {e}")
        print("   Revisa permisos de archivo y vuelve a intentar.")

def update_config_file(duration: float, min_slice: int, max_slice: int):
    """Actualiza el archivo config.py con la nueva configuración."""
    
    config_path = Path(__file__).parent / "DRAFTS" / "config.py"
    
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontró config.py en {config_path}")
    
    # Leer archivo actual
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Buscar y actualizar líneas relevantes
    updated_lines = []
    found_duration = False
    found_auto = False
    found_min = False
    found_max = False
    
    for line in lines:
        stripped = line.strip()
        
        if stripped.startswith('SLICE_DURATION_SECONDS:'):
            updated_lines.append(f'SLICE_DURATION_SECONDS: float = {duration:.3f}  # Duración deseada por slice en segundos\n')
            found_duration = True
        elif stripped.startswith('SLICE_LEN_AUTO:'):
            updated_lines.append('SLICE_LEN_AUTO: bool = True  # Calcular SLICE_LEN automáticamente\n')
            found_auto = True
        elif stripped.startswith('SLICE_LEN_MIN:'):
            updated_lines.append(f'SLICE_LEN_MIN: int = {min_slice}      # Valor mínimo permitido para SLICE_LEN\n')
            found_min = True
        elif stripped.startswith('SLICE_LEN_MAX:'):
            updated_lines.append(f'SLICE_LEN_MAX: int = {max_slice}     # Valor máximo permitido para SLICE_LEN\n')
            found_max = True
        else:
            updated_lines.append(line)
    
    # Verificar que se encontraron todas las configuraciones
    if not all([found_duration, found_auto, found_min, found_max]):
        missing = []
        if not found_duration: missing.append('SLICE_DURATION_SECONDS')
        if not found_auto: missing.append('SLICE_LEN_AUTO')
        if not found_min: missing.append('SLICE_LEN_MIN')
        if not found_max: missing.append('SLICE_LEN_MAX')
        
        raise ValueError(f"No se encontraron las siguientes configuraciones en config.py: {missing}")
    
    # Escribir archivo actualizado
    with open(config_path, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    
    print(f"✅ Archivo config.py actualizado en: {config_path}")

def main():
    """Función principal del configurador."""
    
    print("🎛️  CONFIGURADOR SLICE_LEN DINÁMICO")
    print("=" * 50)
    print("Herramienta para configurar SLICE_LEN de manera intuitiva")
    print("basándote en la duración temporal deseada.\n")
    
    try:
        interactive_slice_len_config()
    except KeyboardInterrupt:
        print("\n\n❌ Configuración interrumpida por el usuario.")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        print("   Por favor reporta este error.")

if __name__ == "__main__":
    main()
