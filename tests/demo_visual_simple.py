#!/usr/bin/env python3
"""
Demo visual SIMPLE: Comparación exacta de waterfalls entre bandas
"""
import numpy as np
import matplotlib.pyplot as plt

def get_band_frequency_range(frequencies, band_idx):
    """Versión simplificada para la demo"""
    freq_min, freq_max = frequencies[0], frequencies[-1]
    if band_idx == 0:  # Full band
        return (freq_min, freq_max)
    elif band_idx == 1:  # Low band
        return (freq_min, (freq_min + freq_max) / 2)
    else:  # High band
        return ((freq_min + freq_max) / 2, freq_max)

def create_visual_comparison():
    """Crea una comparación visual de cómo se ven los waterfalls en cada banda"""
    
    print("🎨 DEMO VISUAL: WATERFALLS EN SISTEMA MULTI-BANDA")
    print("="*70)
    print()
    
    # Usar datos simulados para la demostración
    print("📊 Generando datos de ejemplo...")
    
    # Simular datos de radiotelescopia
    n_freq = 200  # 200 canales de frecuencia
    n_time = 100  # 100 muestras de tiempo
    freq_start = 1200  # MHz
    freq_end = 1500    # MHz
    
    # Crear arrays de frecuencia y tiempo
    frequencies = np.linspace(freq_start, freq_end, n_freq)
    time_axis = np.linspace(0, 10, n_time)  # 10 segundos
    
    # Generar datos simulados con ruido + señal FRB
    np.random.seed(42)
    waterfall_data = np.random.normal(1.0, 0.2, (n_freq, n_time))
    
    # Agregar una señal FRB dispersa (más fuerte en frecuencias altas, llega antes)
    frb_time_center = 50
    for i, freq in enumerate(frequencies):
        # Dispersion delay simulado
        delay = int(5 * (1/freq**2 - 1/freq_end**2) * 1000)
        signal_time = frb_time_center + delay
        if 0 <= signal_time < n_time:
            # Señal más fuerte en frecuencias altas
            signal_strength = 2.0 + (freq - freq_start) / (freq_end - freq_start)
            waterfall_data[i, signal_time] += signal_strength
    
    # Crear la figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('COMPARACIÓN VISUAL: Waterfalls en Sistema Multi-Banda\n' + 
                 f'Datos: {freq_start}-{freq_end} MHz', fontsize=16, fontweight='bold')
    
    # Calcular rangos de frecuencia para cada banda
    full_range = get_band_frequency_range(frequencies, 0)
    low_range = get_band_frequency_range(frequencies, 1) 
    high_range = get_band_frequency_range(frequencies, 2)
    
    print(f"📈 Rangos calculados:")
    print(f"   Full Band: {full_range[0]:.1f} - {full_range[1]:.1f} MHz")
    print(f"   Low Band:  {low_range[0]:.1f} - {low_range[1]:.1f} MHz")
    print(f"   High Band: {high_range[0]:.1f} - {high_range[1]:.1f} MHz")
    print()
    
    # Datos originales completos
    axes[0,0].imshow(waterfall_data, aspect='auto', cmap='viridis', 
                     extent=[0, 10, freq_start, freq_end])
    axes[0,0].set_title('DATOS ORIGINALES COMPLETOS\n(Lo que lee el filterbank)', 
                        fontweight='bold')
    axes[0,0].set_ylabel('Frecuencia (MHz)')
    axes[0,0].set_xlabel('Tiempo (s)')
    
    # Full Band waterfall (idéntico a los datos originales)
    axes[0,1].imshow(waterfall_data, aspect='auto', cmap='viridis',
                     extent=[0, 10, freq_start, freq_end])
    axes[0,1].set_title(f'FULL BAND WATERFALL\nProcesa: {full_range[0]:.0f}-{full_range[1]:.0f} MHz\n' +
                        '⚠️ Muestra: TODO el rango de frecuencias', 
                        fontweight='bold', color='blue')
    axes[0,1].set_ylabel('Frecuencia (MHz)')
    axes[0,1].set_xlabel('Tiempo (s)')
    
    # Low Band waterfall (¡IDÉNTICO visualmente!)
    axes[1,0].imshow(waterfall_data, aspect='auto', cmap='viridis',
                     extent=[0, 10, freq_start, freq_end])
    axes[1,0].set_title(f'LOW BAND WATERFALL\nProcesa: {low_range[0]:.0f}-{low_range[1]:.0f} MHz\n' +
                        '⚠️ Muestra: TODO el rango de frecuencias', 
                        fontweight='bold', color='green')
    axes[1,0].set_ylabel('Frecuencia (MHz)')
    axes[1,0].set_xlabel('Tiempo (s)')
    
    # Destacar la región procesada en Low Band
    low_freq_start = low_range[0]
    low_freq_end = low_range[1]
    axes[1,0].axhspan(low_freq_start, low_freq_end, alpha=0.2, color='green', 
                      label=f'Región procesada: {low_freq_start:.0f}-{low_freq_end:.0f} MHz')
    axes[1,0].legend()
    
    # High Band waterfall (¡IDÉNTICO visualmente!)
    axes[1,1].imshow(waterfall_data, aspect='auto', cmap='viridis',
                     extent=[0, 10, freq_start, freq_end])
    axes[1,1].set_title(f'HIGH BAND WATERFALL\nProcesa: {high_range[0]:.0f}-{high_range[1]:.0f} MHz\n' +
                        '⚠️ Muestra: TODO el rango de frecuencias', 
                        fontweight='bold', color='red')
    axes[1,1].set_ylabel('Frecuencia (MHz)')
    axes[1,1].set_xlabel('Tiempo (s)')
    
    # Destacar la región procesada en High Band
    high_freq_start = high_range[0]
    high_freq_end = high_range[1]
    axes[1,1].axhspan(high_freq_start, high_freq_end, alpha=0.2, color='red',
                      label=f'Región procesada: {high_freq_start:.0f}-{high_freq_end:.0f} MHz')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('demo_visual_waterfalls_comparison.png', dpi=150, bbox_inches='tight')
    print("💾 Guardado: demo_visual_waterfalls_comparison.png")
    plt.close()
    
    print()
    print("🔍 PUNTOS CLAVE:")
    print("   1️⃣ TODOS los waterfalls son VISUALMENTE IDÉNTICOS")
    print("   2️⃣ Todos muestran el rango completo de frecuencias")
    print("   3️⃣ Solo cambia QUÉ datos se usan para DETECCIÓN")
    print("   4️⃣ Las regiones destacadas muestran qué frecuencias se procesan")
    print()
    
    return waterfall_data, frequencies

def create_dm_time_comparison(waterfall_data, frequencies):
    """Muestra cómo difieren los cubos DM-tiempo entre bandas"""
    
    print("📊 DEMO: DIFERENCIAS EN CUBOS DM-TIEMPO")
    print("="*50)
    
    # Simular diferentes DMs
    dm_values = np.linspace(0, 200, 50)
    time_samples = waterfall_data.shape[1]
    
    # Simular procesamiento para cada banda
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('CUBOS DM-TIEMPO: Lo que SÍ cambia entre bandas', fontsize=16, fontweight='bold')
    
    band_names = ['Full Band', 'Low Band', 'High Band']
    band_colors = ['blue', 'green', 'red']
    
    for band_idx in range(3):
        # Obtener rango de frecuencias para esta banda
        freq_range = get_band_frequency_range(frequencies, band_idx)
        
        # Simular cubo DM-tiempo (simplificado)
        dm_time_cube = np.random.normal(0.5, 0.1, (len(dm_values), time_samples))
        
        # Agregar señal simulada más fuerte en ciertos DMs
        if band_idx == 0:  # Full band - señal en DM ~100
            dm_time_cube[25, 45:55] += 2.0
        elif band_idx == 1:  # Low band - señal más débil
            dm_time_cube[25, 45:55] += 1.0
        else:  # High band - señal más fuerte
            dm_time_cube[25, 45:55] += 3.0
        
        # Plotear
        im = axes[band_idx].imshow(dm_time_cube, aspect='auto', cmap='plasma',
                                   extent=[0, 10, dm_values[0], dm_values[-1]])
        axes[band_idx].set_title(f'{band_names[band_idx]}\n{freq_range[0]:.0f}-{freq_range[1]:.0f} MHz',
                                 fontweight='bold', color=band_colors[band_idx])
        axes[band_idx].set_xlabel('Tiempo (s)')
        if band_idx == 0:
            axes[band_idx].set_ylabel('DM (pc cm⁻³)')
        
        # Colorbar
        plt.colorbar(im, ax=axes[band_idx], label='Intensidad')
    
    plt.tight_layout()
    plt.savefig('demo_dm_time_cubes_comparison.png', dpi=150, bbox_inches='tight')
    print("💾 Guardado: demo_dm_time_cubes_comparison.png")
    plt.close()
    
    print()
    print("🎯 ESTO SÍ CAMBIA ENTRE BANDAS:")
    print("   • Intensidad de las señales detectadas")
    print("   • Qué candidatos se identifican como significativos")
    print("   • Eficiencia de detección según el tipo de FRB")
    print()

def main():
    """Función principal"""
    print("🚀 INICIANDO DEMO VISUAL COMPLETO")
    print()
    
    # Crear comparación de waterfalls
    waterfall_data, frequencies = create_visual_comparison()
    
    print("\n" + "="*70 + "\n")
    
    # Crear comparación de cubos DM-tiempo
    create_dm_time_comparison(waterfall_data, frequencies)
    
    print("✅ DEMO COMPLETADO")
    print()
    print("📁 Archivos generados:")
    print("   • demo_visual_waterfalls_comparison.png")
    print("   • demo_dm_time_cubes_comparison.png")
    print()
    print("🤔 PREGUNTA RESPONDIDA:")
    print("   Los waterfalls de todas las bandas se ven EXACTAMENTE iguales")
    print("   porque todos muestran los datos originales completos.")
    print("   Lo que cambia es el PROCESAMIENTO interno para detectar señales.")
    print()
    print("📊 RESUMEN VISUAL:")
    print("   ┌─ Datos originales (1200-1500 MHz)")
    print("   ├─ Full Band: Waterfall = 1200-1500 MHz, Procesa = 1200-1500 MHz")
    print("   ├─ Low Band:  Waterfall = 1200-1500 MHz, Procesa = 1200-1350 MHz") 
    print("   └─ High Band: Waterfall = 1200-1500 MHz, Procesa = 1350-1500 MHz")
    print()
    print("   ⚠️  Los 3 waterfalls son visualmente IDÉNTICOS!")

if __name__ == "__main__":
    main()
