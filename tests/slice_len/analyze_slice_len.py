"""
Análisis completo de SLICE_LEN en el pipeline DRAFTS.

Este documento explica en detalle qué hace SLICE_LEN, dónde se usa, 
y por qué cambiar este valor afecta los resultados de detección.
"""

import sys
from pathlib import Path
import numpy as np

# Configurar path
sys.path.append(str(Path(__file__).parent))

def analyze_slice_len_impact():
    """Analiza el impacto de SLICE_LEN en el pipeline."""
    print("🔬 === ANÁLISIS COMPLETO DE SLICE_LEN ===\n")
    
    print("📋 DEFINICIÓN Y PROPÓSITO:")
    print("   SLICE_LEN = Longitud de cada slice temporal en muestras")
    print("   • Controla cómo se divide el archivo de datos en segmentos temporales")
    print("   • Cada slice se procesa independientemente por el modelo de detección")
    print("   • Valor actual en config.py: 64 muestras")
    
    print("\n🏗️  ARQUITECTURA DEL PROCESAMIENTO:")
    print("   1. Archivo completo → Múltiples slices temporales")
    print("   2. Cada slice → Imagen 512x512 para CNN")
    print("   3. CNN detecta candidatos en cada slice")
    print("   4. Coordenadas pixel → Coordenadas físicas (DM, tiempo)")
    
    print("\n📐 USOS PRINCIPALES DE SLICE_LEN:")
    
    print("\n   1️⃣  DIVISIÓN TEMPORAL DEL ARCHIVO:")
    print("      📍 Ubicación: pipeline.py, línea ~289")
    print("      📝 Código: slice_len, time_slice = _slice_parameters(width_total, config.SLICE_LEN)")
    print("      🎯 Función: Divide datos totales en segmentos manejables")
    print("      📊 Cálculo: time_slice = width_total // SLICE_LEN")
    print("      💡 Ejemplo: Si archivo = 1024 muestras, SLICE_LEN = 64")
    print("                  → time_slice = 1024 // 64 = 16 slices")
    
    print("\n   2️⃣  EXTRACCIÓN DE SLICE_CUBE:")
    print("      📍 Ubicación: pipeline.py, línea ~331")
    print("      📝 Código: slice_cube = dm_time[:, :, slice_len * j : slice_len * (j + 1)]")
    print("      🎯 Función: Extrae región DM vs. tiempo para análisis")
    print("      📊 Dimensiones: [bandas, DM, tiempo_slice]")
    print("      💡 Cada slice_cube contiene SLICE_LEN muestras temporales")
    
    print("\n   3️⃣  EXTRACCIÓN DE WATERFALL_BLOCK:")
    print("      📍 Ubicación: pipeline.py, línea ~332")
    print("      📝 Código: waterfall_block = data[j * slice_len : (j + 1) * slice_len]")
    print("      🎯 Función: Extrae waterfall temporal para visualización")
    print("      📊 Dimensiones: [tiempo_slice, frecuencia]")
    print("      💡 Para composites y análisis RFI")
    
    print("\n   4️⃣  CONVERSIÓN PIXEL → FÍSICAS:")
    print("      📍 Ubicación: astro_conversions.py, línea ~22")
    print("      📝 Código: scale_time = slice_len / 512.0")
    print("      🎯 Función: Convierte coordenadas CNN a tiempo real")
    print("      📊 Escala: pixel_time * (SLICE_LEN / 512)")
    print("      💡 Fundamental para localización temporal precisa")
    
    print("\n   5️⃣  REDIMENSIONAMIENTO PARA CNN:")
    print("      📍 Ubicación: image_utils.py, línea ~29")
    print("      📝 Código: img = cv2.resize(img, (512, 512))")
    print("      🎯 Función: Ajusta slice a entrada fija CNN")
    print("      📊 Mapeo: SLICE_LEN muestras → 512 pixels")
    print("      💡 Resolución temporal = SLICE_LEN / 512")
    
    print("\n   6️⃣  DEDISPERSIÓN DE BLOQUES:")
    print("      📍 Ubicación: pipeline.py, línea ~438")
    print("      📝 Código: dedisp_block = dedisperse_block(data, freq_down, first_dm, start, slice_len)")
    print("      🎯 Función: Dedispersa waterfall para DM específico")
    print("      📊 Tamaño: [SLICE_LEN, n_freq]")
    print("      💡 Para análisis post-detección")
    
    print("\n🎯 POR QUÉ SLICE_LEN AFECTA LOS RESULTADOS:")
    
    print("\n   📊 RESOLUCIÓN TEMPORAL:")
    print("      • SLICE_LEN pequeño → Alta resolución temporal, slices cortos")
    print("      • SLICE_LEN grande → Baja resolución temporal, slices largos")
    print("      • Resolución efectiva = (SLICE_LEN * TIME_RESO) / 512 pixels")
    
    print("\n   🎯 DETECCIÓN DE SEÑALES:")
    print("      • Señales cortas: Necesitan SLICE_LEN pequeño para resolución")
    print("      • Señales largas: Pueden usar SLICE_LEN grande para contexto")
    print("      • Señales intermedias: SLICE_LEN debe optimizarse")
    
    print("\n   🧠 ENTRADA AL MODELO CNN:")
    print("      • CNN espera imágenes 512x512 pixels")
    print("      • Cada pixel temporal representa SLICE_LEN/512 muestras")
    print("      • SLICE_LEN cambia la 'ventana temporal' que ve el modelo")
    
    print("\n   📍 PRECISIÓN DE LOCALIZACIÓN:")
    print("      • Coordenadas pixel se mapean a tiempo real via SLICE_LEN")
    print("      • SLICE_LEN pequeño → Mayor precisión temporal")
    print("      • SLICE_LEN grande → Menor precisión, más contexto")
    
    print("\n   🔄 NÚMERO DE DETECCIONES:")
    print("      • SLICE_LEN pequeño → Más slices → Más oportunidades detección")
    print("      • SLICE_LEN grande → Menos slices → Menos oportunidades")
    print("      • Pero también: menos fragmentación de señales largas")
    
    print("\n📈 EJEMPLOS NUMÉRICOS:")
    
    # Simulación con diferentes SLICE_LEN
    file_length = 2048  # muestras
    time_reso = 0.001   # segundos
    
    slice_lens = [32, 64, 128, 256, 512]
    
    print(f"\n   📊 Para archivo de {file_length} muestras ({file_length * time_reso:.2f} s):")
    print("   " + "="*70)
    print("   SLICE_LEN | N_Slices | Duración/Slice | Resolución/Pixel | Contexto")
    print("   " + "-"*70)
    
    for sl in slice_lens:
        n_slices = file_length // sl
        duration = sl * time_reso
        resolution = duration / 512
        context = "Bajo" if sl < 64 else "Medio" if sl < 256 else "Alto"
        
        print(f"   {sl:8d} | {n_slices:8d} | {duration:10.3f}s | {resolution:12.6f}s | {context}")
    
    print("   " + "="*70)
    
    print("\n💡 RECOMENDACIONES POR TIPO DE SEÑAL:")
    
    print("\n   🔥 SEÑALES CORTAS (< 10ms):")
    print("      • SLICE_LEN recomendado: 32-64")
    print("      • Razón: Alta resolución temporal necesaria")
    print("      • Trade-off: Más slices, menos contexto")
    
    print("\n   ⚡ SEÑALES MEDIAS (10-100ms):")
    print("      • SLICE_LEN recomendado: 64-128")
    print("      • Razón: Balance resolución/contexto")
    print("      • Trade-off: Moderado en ambos aspectos")
    
    print("\n   🌊 SEÑALES LARGAS (> 100ms):")
    print("      • SLICE_LEN recomendado: 128-512")
    print("      • Razón: Contexto temporal importante")
    print("      • Trade-off: Menos resolución, mejor contexto")
    
    print("\n   🎯 BÚSQUEDA GENERAL:")
    print("      • SLICE_LEN recomendado: 64-128")
    print("      • Razón: Versatilidad para diferentes duraciones")
    print("      • Trade-off: Compromiso equilibrado")
    
    print("\n⚙️  EFECTOS EN COMPONENTES ESPECÍFICOS:")
    
    print("\n   🖼️  VISUALIZACIONES:")
    print("      • Composites: Muestran ventana temporal = SLICE_LEN")
    print("      • Patches: Extraídos de región SLICE_LEN")
    print("      • Waterfalls: Limitados a duración SLICE_LEN")
    
    print("\n   📊 ANÁLISIS SNR:")
    print("      • Perfiles temporales: Longitud = SLICE_LEN")
    print("      • Regiones off-pulse: Relativas a SLICE_LEN")
    print("      • Picos SNR: Posición dentro de slice")
    
    print("\n   🧹 LIMPIEZA RFI:")
    print("      • Aplicada slice por slice")
    print("      • Estadísticas RFI por slice")
    print("      • Efectividad depende de duración slice")
    
    print("\n   🎯 MODELO CNN:")
    print("      • Entrenado con SLICE_LEN específico")
    print("      • Sensibilidad a escala temporal")
    print("      • Rendimiento óptimo en escala entrenamiento")
    
    print("\n🔧 CÓMO OPTIMIZAR SLICE_LEN:")
    
    print("\n   1️⃣  ANALIZAR SEÑALES OBJETIVO:")
    print("      • Duración típica de FRBs esperados")
    print("      • Dispersión temporal en banda observada")
    print("      • Características específicas telescopio")
    
    print("\n   2️⃣  EXPERIMENTAR CON VALORES:")
    print("      • Comenzar con 64 (valor actual)")
    print("      • Probar 32, 128 según señales")
    print("      • Evaluar número y calidad detecciones")
    
    print("\n   3️⃣  EVALUAR MÉTRICAS:")
    print("      • Número candidatos detectados")
    print("      • SNR promedio detecciones")
    print("      • Precisión localización temporal")
    print("      • Tasa falsos positivos")
    
    print("\n   4️⃣  CONSIDERAR HARDWARE:")
    print("      • Memoria GPU: slices grandes usan más")
    print("      • Velocidad: más slices = más tiempo")
    print("      • Almacenamiento: más composites generados")
    
    print("\n🎯 SLICE_LEN ACTUAL = 64:")
    
    print("\n   ✅ VENTAJAS:")
    print("      • Buena resolución temporal")
    print("      • Número moderado de slices")
    print("      • Compatible con modelo entrenado")
    print("      • Equilibrio memoria/velocidad")
    
    print("\n   ⚠️  CONSIDERACIONES:")
    print("      • Puede fragmentar señales > 64ms")
    print("      • Resolución ~0.125ms por pixel (si TIME_RESO=0.001)")
    print("      • 16 slices para archivo 1024 muestras")
    
    print("\n📝 MODIFICACIONES SUGERIDAS:")
    
    print("\n   🔬 PARA SEÑALES MUY CORTAS:")
    print("      • Cambiar SLICE_LEN = 32")
    print("      • Duplica resolución temporal")
    print("      • Duplica número de slices")
    
    print("\n   🌊 PARA SEÑALES LARGAS:")
    print("      • Cambiar SLICE_LEN = 128")
    print("      • Reduce fragmentación")
    print("      • Reduce número de slices")
    
    print("\n   ⚖️  PARA OPTIMIZACIÓN:")
    print("      • Analizar distribución duraciones FRB")
    print("      • Ajustar según pico distribución")
    print("      • Re-entrenar modelo si es necesario")


def demonstrate_slice_len_calculations():
    """Demuestra cálculos específicos con diferentes SLICE_LEN."""
    
    print("\n" + "="*80)
    print("🧮 DEMOSTRACIÓN DE CÁLCULOS SLICE_LEN")
    print("="*80)
    
    # Parámetros de ejemplo
    file_length = 1024  # muestras totales
    time_reso = 0.001   # resolución temporal (1ms)
    dm_range = 1024     # rango DM
    
    print(f"\n📊 PARÁMETROS DE EJEMPLO:")
    print(f"   • Archivo: {file_length} muestras ({file_length * time_reso:.2f} segundos)")
    print(f"   • Resolución temporal: {time_reso} s/muestra")
    print(f"   • Rango DM: 0-{dm_range} pc cm⁻³")
    
    slice_lens = [32, 64, 128, 256]
    
    print(f"\n📋 COMPARACIÓN DETALLADA:")
    
    for sl in slice_lens:
        print(f"\n   🔸 SLICE_LEN = {sl}")
        print(f"   " + "-"*40)
        
        # Cálculos básicos
        n_slices = file_length // sl
        slice_duration = sl * time_reso
        pixel_resolution = slice_duration / 512
        
        print(f"   • Número de slices: {n_slices}")
        print(f"   • Duración por slice: {slice_duration:.3f} s")
        print(f"   • Resolución por pixel: {pixel_resolution:.6f} s")
        
        # Ejemplo de conversión pixel to physical
        example_pixel_x = 256  # pixel central
        example_pixel_y = 256  # pixel central
        
        scale_time = sl / 512.0
        scale_dm = dm_range / 512.0
        
        sample_offset = example_pixel_x * scale_time
        dm_val = example_pixel_y * scale_dm
        t_seconds = sample_offset * time_reso
        
        print(f"   • Pixel (256, 256) → DM={dm_val:.1f}, t={t_seconds:.6f}s")
        print(f"   • Escala temporal: {scale_time:.3f} muestras/pixel")
        print(f"   • Escala DM: {scale_dm:.3f} pc cm⁻³/pixel")
        
        # Análisis de ventana temporal
        if sl < 64:
            analysis = "Alta resolución, contexto limitado"
        elif sl < 128:
            analysis = "Resolución equilibrada"
        elif sl < 256:
            analysis = "Buen contexto, resolución moderada"
        else:
            analysis = "Máximo contexto, baja resolución"
            
        print(f"   • Análisis: {analysis}")


def slice_len_recommendations():
    """Proporciona recomendaciones específicas para SLICE_LEN."""
    
    print("\n" + "="*80)
    print("💡 RECOMENDACIONES ESPECÍFICAS SLICE_LEN")
    print("="*80)
    
    print("\n🎯 BASADO EN TU CONFIGURACIÓN ACTUAL:")
    print("   • SLICE_LEN actual: 64")
    print("   • USE_MULTI_BAND: True") 
    print("   • DET_PROB: 0.5")
    print("   • TARGET: '3100_0001_00_8bit'")
    
    print("\n📊 ANÁLISIS PARA TU CASO:")
    
    print("\n   🔍 SI BUSCAS SEÑALES MUY CORTAS (<50ms):")
    print("      → Cambiar SLICE_LEN = 32")
    print("      → Duplica resolución temporal")
    print("      → Mejor para pulsos estrechos")
    print("      → Más slices → más detecciones fragmentadas")
    
    print("\n   ⚖️  SI BUSCAS EQUILIBRIO (50-200ms):")
    print("      → Mantener SLICE_LEN = 64 (actual)")
    print("      → Buen compromiso resolución/contexto")
    print("      → Apropiado para FRBs típicos")
    print("      → Configuración estándar probada")
    
    print("\n   🌊 SI BUSCAS SEÑALES LARGAS (>200ms):")
    print("      → Cambiar SLICE_LEN = 128")
    print("      → Mejor contexto temporal")
    print("      → Menos fragmentación señales largas")
    print("      → Menos slices → procesamiento más rápido")
    
    print("\n   🚀 SI QUIERES MÁXIMA SENSIBILIDAD:")
    print("      → Probar múltiples valores: [32, 64, 128]")
    print("      → Ejecutar pipeline con cada valor")
    print("      → Comparar número y calidad detecciones")
    print("      → Seleccionar óptimo para tus datos")
    
    print("\n⚙️  PASOS PARA CAMBIAR SLICE_LEN:")
    
    print("\n   1️⃣  MODIFICAR CONFIG.PY:")
    print("      ```python")
    print("      SLICE_LEN: int = 32  # o 128, 256, etc.")
    print("      ```")
    
    print("\n   2️⃣  EJECUTAR PIPELINE:")
    print("      ```bash")
    print("      python main.py")
    print("      ```")
    
    print("\n   3️⃣  COMPARAR RESULTADOS:")
    print("      • Número candidatos en CSV")
    print("      • SNR promedio detecciones")
    print("      • Calidad visual composites")
    print("      • Tiempo procesamiento")
    
    print("\n   4️⃣  EVALUAR TRADE-OFFS:")
    print("      • Más candidatos vs. más falsos positivos")
    print("      • Mayor resolución vs. más fragmentación")
    print("      • Mejor contexto vs. menor precisión")
    
    print("\n🎯 RECOMENDACIÓN FINAL:")
    print("   • Comenzar con SLICE_LEN = 32 para tus datos filterbank")
    print("   • Los archivos .fil suelen tener señales cortas")
    print("   • La mayor resolución temporal puede mejorar detección")
    print("   • Evaluar resultados vs. configuración actual (64)")
    print("   • Si no mejora, volver a 64 o probar 128")


if __name__ == "__main__":
    analyze_slice_len_impact()
    demonstrate_slice_len_calculations()
    slice_len_recommendations()
