#!/usr/bin/env python3
"""
Localización exacta de todos los usos de SLICE_LEN en el código.
Este script muestra dónde y cómo se usa SLICE_LEN en tu pipeline.
"""

import sys
from pathlib import Path

# Configurar path
sys.path.append(str(Path(__file__).parent))

def find_slice_len_usage():
    """Encuentra todos los usos de SLICE_LEN en el código."""
    
    print("🔍 === LOCALIZACIÓN DE SLICE_LEN EN EL CÓDIGO ===\n")
    
    # Archivos principales donde se usa SLICE_LEN
    files_to_check = [
        "DRAFTS/config.py",
        "DRAFTS/pipeline.py", 
        "DRAFTS/astro_conversions.py",
        "DRAFTS/image_utils.py",
        "DRAFTS/visualization.py"
    ]
    
    base_dir = Path(__file__).parent
    
    for file_path in files_to_check:
        full_path = base_dir / file_path
        if not full_path.exists():
            continue
            
        print(f"📁 {file_path}:")
        print("-" * 50)
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            found_usage = False
            for i, line in enumerate(lines, 1):
                if 'SLICE_LEN' in line and not line.strip().startswith('#'):
                    found_usage = True
                    context_start = max(0, i-2)
                    context_end = min(len(lines), i+2)
                    
                    print(f"   📍 Línea {i}:")
                    for j in range(context_start, context_end):
                        marker = ">>>" if j == i-1 else "   "
                        print(f"   {marker} {j+1:3d}: {lines[j].rstrip()}")
                    print()
            
            if not found_usage:
                print("   ❌ No se encontraron usos de SLICE_LEN")
                
        except Exception as e:
            print(f"   ❌ Error leyendo archivo: {e}")
        
        print()

def show_slice_len_flow():
    """Muestra el flujo de uso de SLICE_LEN en el pipeline."""
    
    print("🔄 === FLUJO DE SLICE_LEN EN EL PIPELINE ===\n")
    
    steps = [
        {
            'step': 1,
            'location': 'config.py:30',
            'description': 'Definición del parámetro',
            'code': 'SLICE_LEN: int = 64',
            'impact': 'Valor global usado en todo el pipeline'
        },
        {
            'step': 2,
            'location': 'pipeline.py:289',
            'description': 'Cálculo de parámetros de slicing',
            'code': 'slice_len, time_slice = _slice_parameters(width_total, config.SLICE_LEN)',
            'impact': 'Determina cuántos slices se crearán'
        },
        {
            'step': 3,
            'location': 'pipeline.py:331-332',
            'description': 'Extracción de datos por slice',
            'code': '''slice_cube = dm_time[:, :, slice_len * j : slice_len * (j + 1)]
waterfall_block = data[j * slice_len : (j + 1) * slice_len]''',
            'impact': 'Divide datos en ventanas temporales'
        },
        {
            'step': 4,
            'location': 'image_utils.py:29',
            'description': 'Redimensionamiento para CNN',
            'code': 'img = cv2.resize(img, (512, 512))',
            'impact': 'SLICE_LEN muestras → 512 pixels temporales'
        },
        {
            'step': 5,
            'location': 'astro_conversions.py:22',
            'description': 'Conversión pixel → físico',
            'code': 'scale_time = slice_len / 512.0',
            'impact': 'Convierte detecciones CNN a tiempo real'
        },
        {
            'step': 6,
            'location': 'image_utils.py:144,147',
            'description': 'Metadatos temporales',
            'code': 'time_start_slice = slice_idx * config.SLICE_LEN * config.TIME_RESO',
            'impact': 'Información temporal para visualización'
        },
        {
            'step': 7,
            'location': 'visualization.py:33,34,231',
            'description': 'Visualización de slices',
            'code': 'prev_len = config.SLICE_LEN; config.SLICE_LEN = slice_len',
            'impact': 'Ajusta parámetros para visualización correcta'
        }
    ]
    
    for step in steps:
        print(f"   {step['step']}️⃣  {step['description'].upper()}")
        print(f"      📍 Ubicación: {step['location']}")
        print(f"      📝 Código:")
        for line in step['code'].split('\n'):
            print(f"         {line.strip()}")
        print(f"      🎯 Impacto: {step['impact']}")
        print()

def show_slice_len_impact_summary():
    """Muestra un resumen del impacto de SLICE_LEN."""
    
    print("📊 === RESUMEN DEL IMPACTO DE SLICE_LEN ===\n")
    
    print("🎯 CAMBIOS DIRECTOS AL MODIFICAR SLICE_LEN:")
    impacts = [
        "✅ Número de slices: width_total // SLICE_LEN",
        "✅ Duración por slice: SLICE_LEN × TIME_RESO",
        "✅ Resolución temporal: SLICE_LEN / 512 muestras/pixel",
        "✅ Precisión localización: (SLICE_LEN / 512) × TIME_RESO segundos/pixel",
        "✅ Contexto temporal: SLICE_LEN × TIME_RESO segundos por ventana",
        "✅ Oportunidades detección: Más slices = más oportunidades"
    ]
    
    for impact in impacts:
        print(f"   {impact}")
    
    print(f"\n🔄 EFECTOS EN CASCADA:")
    cascades = [
        "📸 Más slices → Más imágenes 512x512 procesadas",
        "⏱️  Más slices → Mayor tiempo de procesamiento",
        "🎯 Más slices → Más oportunidades de detectar señales fragmentadas",
        "🖼️  Más slices → Más archivos composite generados",
        "📊 Resolución diferente → Diferentes patrones detectados por CNN",
        "🎨 Visualizaciones muestran diferentes ventanas temporales"
    ]
    
    for cascade in cascades:
        print(f"   {cascade}")

def main():
    """Función principal."""
    find_slice_len_usage()
    show_slice_len_flow()
    show_slice_len_impact_summary()
    
    print("\n💡 CONCLUSIÓN:")
    print("   SLICE_LEN es un parámetro central que afecta:")
    print("   1. División temporal de datos")
    print("   2. Resolución del análisis")
    print("   3. Entrada al modelo CNN")
    print("   4. Conversión de coordenadas")
    print("   5. Número de oportunidades de detección")
    print("   6. Visualizaciones generadas")
    print("\n   ¡Experimentar con diferentes valores puede revelar señales ocultas!")

if __name__ == "__main__":
    main()
