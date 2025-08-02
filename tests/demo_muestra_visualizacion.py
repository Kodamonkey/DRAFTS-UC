"""
Demostración visual de qué es una muestra en detección de FRB
=============================================================

Este script muestra diferentes representaciones de una muestra:
1. Como array de intensidades por frecuencia
2. Como parte de un waterfall
3. Como se procesa en el pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

# Configurar estilo
plt.style.use('default')
sns.set_palette("husl")

def crear_datos_simulados():
    """Crear datos simulados de radioastronomía."""
    # Parámetros simulados
    n_freq = 64  # Canales de frecuencia
    n_time = 100  # Muestras temporales
    frecuencias = np.linspace(1200, 1500, n_freq)  # MHz
    
    # Crear datos de fondo (ruido)
    datos = np.random.normal(0.5, 0.1, (n_time, n_freq))
    
    # Agregar un FRB simulado en el medio
    frb_time = 50
    frb_freq_center = 32
    frb_width_time = 3
    frb_width_freq = 10
    
    for t in range(max(0, frb_time - frb_width_time), min(n_time, frb_time + frb_width_time)):
        for f in range(max(0, frb_freq_center - frb_width_freq), min(n_freq, frb_freq_center + frb_width_freq)):
            distancia_t = abs(t - frb_time)
            distancia_f = abs(f - frb_freq_center)
            intensidad = 2.0 * np.exp(-(distancia_t**2 + distancia_f**2) / 10)
            datos[t, f] += intensidad
    
    return datos, frecuencias

def visualizar_muestra_individual(datos, frecuencias, tiempo_idx=50):
    """Visualizar una muestra individual como array de intensidades."""
    muestra = datos[tiempo_idx, :]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Gráfico 1: Intensidad vs Frecuencia
    ax1.plot(frecuencias, muestra, 'b-', linewidth=2, label=f'Muestra t={tiempo_idx}')
    ax1.set_xlabel('Frecuencia (MHz)')
    ax1.set_ylabel('Intensidad')
    ax1.set_title('Una Muestra = Array de Intensidades por Frecuencia')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Resaltar el FRB
    frb_region = (frecuencias >= 1350) & (frecuencias <= 1450)
    ax1.fill_between(frecuencias[frb_region], muestra[frb_region], 
                     alpha=0.3, color='red', label='Región FRB')
    ax1.legend()
    
    # Gráfico 2: Representación de barras
    bars = ax2.bar(range(len(muestra)), muestra, color='skyblue', alpha=0.7)
    ax2.set_xlabel('Canal de Frecuencia')
    ax2.set_ylabel('Intensidad')
    ax2.set_title('Representación de Barras de la Muestra')
    ax2.grid(True, alpha=0.3)
    
    # Resaltar canales del FRB
    for i in range(32, 42):  # Región del FRB
        if i < len(bars):
            bars[i].set_color('red')
            bars[i].set_alpha(0.8)
    
    plt.tight_layout()
    plt.savefig('tests/demo_muestra_individual.png', dpi=150, bbox_inches='tight')
    plt.show()

def visualizar_waterfall_completo(datos, frecuencias):
    """Visualizar el waterfall completo con la muestra resaltada."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Waterfall completo
    im1 = ax1.imshow(datos.T, aspect='auto', origin='lower', 
                     extent=[0, datos.shape[0], frecuencias[0], frecuencias[-1]],
                     cmap='viridis')
    ax1.set_xlabel('Tiempo (muestras)')
    ax1.set_ylabel('Frecuencia (MHz)')
    ax1.set_title('Waterfall Completo (Datos de Radio)')
    plt.colorbar(im1, ax=ax1, label='Intensidad')
    
    # Resaltar la muestra específica
    tiempo_idx = 50
    ax1.axhline(y=frecuencias[tiempo_idx], color='red', linestyle='--', linewidth=2, 
                label=f'Muestra t={tiempo_idx}')
    ax1.legend()
    
    # Zoom en la región del FRB
    zoom_t_start = max(0, tiempo_idx - 10)
    zoom_t_end = min(datos.shape[0], tiempo_idx + 10)
    zoom_f_start = 1350
    zoom_f_end = 1450
    
    zoom_data = datos[zoom_t_start:zoom_t_end, :]
    zoom_freq = frecuencias
    
    im2 = ax2.imshow(zoom_data.T, aspect='auto', origin='lower',
                     extent=[zoom_t_start, zoom_t_end, zoom_freq[0], zoom_freq[-1]],
                     cmap='viridis')
    ax2.set_xlabel('Tiempo (muestras)')
    ax2.set_ylabel('Frecuencia (MHz)')
    ax2.set_title(f'Zoom en FRB (Muestra {tiempo_idx} resaltada)')
    plt.colorbar(im2, ax=ax2, label='Intensidad')
    
    # Resaltar la muestra específica
    ax2.axhline(y=frecuencias[tiempo_idx], color='red', linestyle='--', linewidth=3)
    
    plt.tight_layout()
    plt.savefig('tests/demo_waterfall_completo.png', dpi=150, bbox_inches='tight')
    plt.show()

def visualizar_procesamiento_pipeline(datos, frecuencias):
    """Visualizar cómo se procesa una muestra en el pipeline."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Parámetros del pipeline
    slice_len = 20  # Tamaño del slice
    slice_idx = 2   # Índice del slice
    start_idx = slice_idx * slice_len
    end_idx = start_idx + slice_len
    
    # 1. Datos originales con slice marcado
    im1 = axes[0,0].imshow(datos.T, aspect='auto', origin='lower', cmap='viridis')
    axes[0,0].set_xlabel('Tiempo (muestras)')
    axes[0,0].set_ylabel('Frecuencia (MHz)')
    axes[0,0].set_title('Datos Originales con Slice Marcado')
    
    # Marcar el slice
    rect = Rectangle((start_idx, frecuencias[0]), slice_len, 
                     frecuencias[-1] - frecuencias[0], 
                     linewidth=3, edgecolor='red', facecolor='none')
    axes[0,0].add_patch(rect)
    axes[0,0].text(start_idx + slice_len/2, frecuencias[-1] + 10, 
                   f'Slice {slice_idx}', ha='center', color='red', fontweight='bold')
    
    # 2. Slice extraído
    slice_data = datos[start_idx:end_idx, :]
    im2 = axes[0,1].imshow(slice_data.T, aspect='auto', origin='lower', cmap='viridis')
    axes[0,1].set_xlabel('Tiempo dentro del slice')
    axes[0,1].set_ylabel('Frecuencia (MHz)')
    axes[0,1].set_title(f'Slice Extraído ({slice_len} muestras)')
    plt.colorbar(im2, ax=axes[0,1], label='Intensidad')
    
    # 3. Una muestra específica del slice
    muestra_idx = 10  # Muestra dentro del slice
    muestra = slice_data[muestra_idx, :]
    axes[1,0].plot(frecuencias, muestra, 'b-', linewidth=2)
    axes[1,0].set_xlabel('Frecuencia (MHz)')
    axes[1,0].set_ylabel('Intensidad')
    axes[1,0].set_title(f'Muestra {muestra_idx} del Slice {slice_idx}')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Procesamiento de detección
    # Simular detección de candidatos
    candidatos = []
    for t in range(slice_data.shape[0]):
        for f in range(slice_data.shape[1]):
            if slice_data[t, f] > 1.5:  # Umbral de detección
                candidatos.append((t, f))
    
    im3 = axes[1,1].imshow(slice_data.T, aspect='auto', origin='lower', cmap='viridis')
    axes[1,1].set_xlabel('Tiempo dentro del slice')
    axes[1,1].set_ylabel('Frecuencia (MHz)')
    axes[1,1].set_title('Detección de Candidatos FRB')
    
    # Marcar candidatos
    for t, f in candidatos:
        axes[1,1].plot(t, frecuencias[f], 'rx', markersize=10, markeredgewidth=2)
    
    plt.colorbar(im3, ax=axes[1,1], label='Intensidad')
    plt.tight_layout()
    plt.savefig('tests/demo_procesamiento_pipeline.png', dpi=150, bbox_inches='tight')
    plt.show()

def crear_resumen_conceptual():
    """Crear un resumen conceptual de qué es una muestra."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Configurar el gráfico
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Título
    ax.text(5, 7.5, '¿QUÉ ES UNA MUESTRA EN DETECCIÓN DE FRB?', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Definiciones
    definiciones = [
        "ASTRONÓMICO:",
        "• Una medición temporal de intensidad de radio",
        "• Contiene valores para cada canal de frecuencia",
        "• Representa un instante en el tiempo",
        "",
        "COMPUTACIONAL:",
        "• Array numpy de números (float32)",
        "• Shape: (FREQ_RESO,) - ej: (512,)",
        "• Cada valor = intensidad en una frecuencia",
        "",
        "VISUAL:",
        "• Una fila horizontal en el waterfall",
        "• Gráfico de intensidad vs frecuencia",
        "• Punto de datos en el tiempo"
    ]
    
    y_pos = 6.5
    for i, texto in enumerate(definiciones):
        if texto.startswith("•"):
            ax.text(1, y_pos, texto, fontsize=10, va='center')
        elif texto.endswith(":"):
            ax.text(0.5, y_pos, texto, fontsize=12, fontweight='bold', va='center')
        else:
            ax.text(0.5, y_pos, texto, fontsize=10, va='center')
        y_pos -= 0.4
    
    # Ejemplo visual
    ax.text(6, 6, "EJEMPLO PRÁCTICO:", fontsize=12, fontweight='bold')
    ax.text(6, 5.5, "Muestra_t=0 = [0.23, 0.45, 0.12, 0.67, ...]", fontsize=10)
    ax.text(6, 5.1, "Muestra_t=1 = [0.25, 0.43, 0.15, 0.65, ...]", fontsize=10)
    ax.text(6, 4.7, "Muestra_t=2 = [0.22, 0.47, 0.18, 0.69, ...]", fontsize=10)
    
    # Relación con el pipeline
    ax.text(6, 3.5, "EN EL PIPELINE:", fontsize=12, fontweight='bold')
    ax.text(6, 3, "• SLICE_LEN = número de muestras por slice", fontsize=10)
    ax.text(6, 2.6, "• start_idx = posición de inicio del slice", fontsize=10)
    ax.text(6, 2.2, "• end_idx = posición de fin del slice", fontsize=10)
    ax.text(6, 1.8, "• block.shape[0] = total de muestras en el bloque", fontsize=10)
    
    plt.tight_layout()
    plt.savefig('tests/demo_resumen_conceptual.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Ejecutar todas las visualizaciones."""
    print("Creando datos simulados...")
    datos, frecuencias = crear_datos_simulados()
    
    print("1. Visualizando muestra individual...")
    visualizar_muestra_individual(datos, frecuencias)
    
    print("2. Visualizando waterfall completo...")
    visualizar_waterfall_completo(datos, frecuencias)
    
    print("3. Visualizando procesamiento del pipeline...")
    visualizar_procesamiento_pipeline(datos, frecuencias)
    
    print("4. Creando resumen conceptual...")
    crear_resumen_conceptual()
    
    print("¡Visualizaciones completadas! Revisa los archivos PNG generados.")

if __name__ == "__main__":
    main() 