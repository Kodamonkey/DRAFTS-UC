#!/usr/bin/env python3
"""
Demo del Sistema de DM Dinámico Integrado en Plots de Candidatos

Este script demuestra cómo el sistema de DM dinámico ajusta automáticamente
los ejes de los plots para centrar los candidatos detectados, mejorando la
visualización y resolución de las detecciones.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import shutil

# Importar módulos del pipeline
from DRAFTS import config
from DRAFTS.dynamic_dm_range import get_dynamic_dm_range_for_candidate, dm_range_calculator
from DRAFTS.image_utils import _calculate_dynamic_dm_range, save_detection_plot

def simulate_detection_scenario():
    """Simula un escenario de detección con candidatos en DMs específicos."""
    
    print("=== DEMO: Sistema de DM Dinámico para Plots de Candidatos ===\n")
    
    # Configurar parámetros de simulación
    config.DM_min = 0
    config.DM_max = 1024
    config.SLICE_LEN = 64
    config.TIME_RESO = 0.001
    config.DOWN_TIME_RATE = 4
    config.FREQ = np.linspace(1200, 1500, 256)
    config.FREQ_RESO = 256
    config.DOWN_FREQ_RATE = 2
    config.FILE_LENG = 4096
    config.MODEL_NAME = "resnet50"
    config.DET_PROB = 0.7
    config.CLASS_PROB = 0.5
    
    # Habilitar DM dinámico
    config.DM_DYNAMIC_RANGE_ENABLE = True
    config.DM_RANGE_FACTOR = 0.2
    config.DM_RANGE_MIN_WIDTH = 50.0
    config.DM_RANGE_MAX_WIDTH = 200.0
    
    # Escenarios de prueba
    scenarios = [
        {
            'name': 'FRB de DM Bajo',
            'dm_candidates': [85.0],
            'confidences': [0.95],
            'description': 'Candidato único con DM bajo'
        },
        {
            'name': 'FRB de DM Alto',
            'dm_candidates': [750.0],
            'confidences': [0.88],
            'description': 'Candidato único con DM alto'
        },
        {
            'name': 'FRBs Múltiples Agrupados',
            'dm_candidates': [320.0, 335.0, 315.0],
            'confidences': [0.92, 0.78, 0.85],
            'description': 'Múltiples candidatos con DMs similares'
        },
        {
            'name': 'FRBs Dispersos',
            'dm_candidates': [150.0, 480.0, 820.0],
            'confidences': [0.76, 0.91, 0.83],
            'description': 'Candidatos dispersos en amplio rango DM'
        }
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n--- Escenario: {scenario['name']} ---")
        print(f"Descripción: {scenario['description']}")
        print(f"DMs candidatos: {scenario['dm_candidates']}")
        print(f"Confidencias: {scenario['confidences']}")
        
        # Simular bounding boxes basados en DMs
        top_boxes = []
        for dm_val in scenario['dm_candidates']:
            # Convertir DM a posición Y en pixel (0-512)
            dm_fraction = (dm_val - config.DM_min) / (config.DM_max - config.DM_min)
            y_center = dm_fraction * 512
            
            # Posición X aleatoria
            x_center = np.random.uniform(128, 384)
            
            # Crear bounding box
            box_width, box_height = 40, 30
            x1 = max(0, x_center - box_width/2)
            y1 = max(0, y_center - box_height/2)
            x2 = min(512, x_center + box_width/2)
            y2 = min(512, y_center + box_height/2)
            
            top_boxes.append([x1, y1, x2, y2])
        
        top_conf = scenario['confidences']
        
        # Calcular rango DM dinámico
        dm_plot_min, dm_plot_max = _calculate_dynamic_dm_range(
            top_boxes=top_boxes,
            slice_len=config.SLICE_LEN,
            confidence_scores=top_conf
        )
        
        # Calcular rango completo para comparación
        dm_full_min = config.DM_min
        dm_full_max = config.DM_max
        
        # Calcular estadísticas
        dm_candidates = scenario['dm_candidates']
        dm_center = np.mean(dm_candidates)
        dm_spread = np.max(dm_candidates) - np.min(dm_candidates)
        range_width = dm_plot_max - dm_plot_min
        full_range_width = dm_full_max - dm_full_min
        zoom_factor = full_range_width / range_width
        
        results[scenario['name']] = {
            'dm_candidates': dm_candidates,
            'dm_center': dm_center,
            'dm_spread': dm_spread,
            'dm_plot_min': dm_plot_min,
            'dm_plot_max': dm_plot_max,
            'range_width': range_width,
            'zoom_factor': zoom_factor,
            'top_boxes': top_boxes,
            'confidences': top_conf
        }
        
        print(f"  → DM center: {dm_center:.1f} pc cm⁻³")
        print(f"  → DM spread: {dm_spread:.1f} pc cm⁻³")
        print(f"  → Rango completo: {dm_full_min} - {dm_full_max} pc cm⁻³ (ancho: {full_range_width})")
        print(f"  → Rango dinámico: {dm_plot_min:.1f} - {dm_plot_max:.1f} pc cm⁻³ (ancho: {range_width:.1f})")
        print(f"  → Factor de zoom: {zoom_factor:.1f}x")
        print(f"  → Mejora de resolución: {zoom_factor:.1f}x más detalle en eje DM")
        
        # Verificar que candidatos están dentro del rango
        all_in_range = all(dm_plot_min <= dm <= dm_plot_max for dm in dm_candidates)
        print(f"  → Todos los candidatos en rango: {'✓' if all_in_range else '✗'}")
    
    return results

def create_comparison_plots(results):
    """Crea plots comparativos mostrando el efecto del DM dinámico."""
    
    print("\n=== Generando Plots Comparativos ===")
    
    # Crear directorio temporal
    temp_dir = Path(tempfile.mkdtemp())
    demo_dir = Path("demo_dm_dynamic_plots")
    demo_dir.mkdir(exist_ok=True)
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Sistema de DM Dinámico: Comparación de Rangos', fontsize=16, fontweight='bold')
        
        scenarios = list(results.keys())
        
        for idx, (scenario_name, data) in enumerate(results.items()):
            if idx >= 4:  # Solo primeros 4 escenarios
                break
                
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            # Configurar el plot
            dm_candidates = data['dm_candidates']
            dm_plot_min = data['dm_plot_min']
            dm_plot_max = data['dm_plot_max']
            zoom_factor = data['zoom_factor']
            
            # Rango completo
            full_range = np.linspace(config.DM_min, config.DM_max, 100)
            ax.plot(full_range, np.ones_like(full_range) * 0.5, 'lightgray', linewidth=10, 
                   alpha=0.7, label='Rango completo (0-1024)')
            
            # Rango dinámico
            dynamic_range = np.linspace(dm_plot_min, dm_plot_max, 50)
            ax.plot(dynamic_range, np.ones_like(dynamic_range) * 0.6, 'orange', linewidth=8,
                   alpha=0.8, label=f'Rango dinámico ({dm_plot_min:.0f}-{dm_plot_max:.0f})')
            
            # Candidatos
            for i, dm in enumerate(dm_candidates):
                ax.plot(dm, 0.7, 'ro', markersize=12, markeredgewidth=2, 
                       markeredgecolor='darkred', label=f'Candidato {i+1}' if i == 0 else None)
                ax.annotate(f'DM={dm:.0f}', (dm, 0.7), xytext=(dm, 0.8),
                           ha='center', fontweight='bold', fontsize=9)
            
            ax.set_xlim(0, 1024)
            ax.set_ylim(0.4, 0.9)
            ax.set_xlabel('Dispersion Measure (pc cm⁻³)')
            ax.set_title(f'{scenario_name}\nZoom: {zoom_factor:.1f}x', fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Añadir información adicional
            ax.text(0.02, 0.95, 
                   f'Candidatos: {len(dm_candidates)}\n'
                   f'Ancho dinámico: {dm_plot_max - dm_plot_min:.0f} pc cm⁻³\n'
                   f'Mejora resolución: {zoom_factor:.1f}x',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   fontsize=8)
        
        plt.tight_layout()
        plt.savefig(demo_dir / "dm_dynamic_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plot comparativo guardado: {demo_dir}/dm_dynamic_comparison.png")
        
        # Crear tabla resumen
        create_summary_table(results, demo_dir)
        
    finally:
        # Limpiar directorio temporal
        shutil.rmtree(temp_dir, ignore_errors=True)

def create_summary_table(results, output_dir):
    """Crea una tabla resumen de los resultados."""
    
    print("\n=== Tabla Resumen de Resultados ===")
    
    header = f"{'Escenario':<25} {'Candidatos':<12} {'DM Center':<10} {'Rango Dinámico':<20} {'Zoom':<8} {'Resolución':<12}"
    print(header)
    print("=" * len(header))
    
    summary_lines = [header, "=" * len(header)]
    
    for scenario_name, data in results.items():
        n_candidates = len(data['dm_candidates'])
        dm_center = data['dm_center']
        dm_range = f"{data['dm_plot_min']:.0f}-{data['dm_plot_max']:.0f}"
        zoom_factor = data['zoom_factor']
        resolution = f"{zoom_factor:.1f}x mejor"
        
        line = f"{scenario_name:<25} {n_candidates:<12} {dm_center:<10.0f} {dm_range:<20} {zoom_factor:<8.1f} {resolution:<12}"
        print(line)
        summary_lines.append(line)
    
    # Guardar tabla en archivo
    with open(output_dir / "dm_dynamic_summary.txt", 'w') as f:
        f.write("Sistema de DM Dinámico - Resumen de Resultados\n")
        f.write("=" * 50 + "\n\n")
        for line in summary_lines:
            f.write(line + "\n")
        
        f.write("\n\nBeneficios del Sistema de DM Dinámico:\n")
        f.write("- Mejora automática de la resolución del eje DM\n")
        f.write("- Centrado automático en candidatos detectados\n")
        f.write("- Reducción de espacio vacío en los plots\n")
        f.write("- Visualización optimizada para cada detección\n")
        f.write("- Mantenimiento de toda la información relevante\n")
    
    print(f"\n✓ Resumen guardado: {output_dir}/dm_dynamic_summary.txt")

def demonstrate_configuration_options():
    """Demuestra diferentes opciones de configuración."""
    
    print("\n=== Configuraciones Disponibles ===")
    
    configs = [
        ("Estándar", {'DM_RANGE_FACTOR': 0.2, 'DM_RANGE_MIN_WIDTH': 50.0, 'DM_RANGE_MAX_WIDTH': 200.0}),
        ("Zoom Alto", {'DM_RANGE_FACTOR': 0.1, 'DM_RANGE_MIN_WIDTH': 30.0, 'DM_RANGE_MAX_WIDTH': 100.0}),
        ("Conservador", {'DM_RANGE_FACTOR': 0.3, 'DM_RANGE_MIN_WIDTH': 100.0, 'DM_RANGE_MAX_WIDTH': 300.0}),
        ("Adaptativo", {'DM_RANGE_FACTOR': 0.15, 'DM_RANGE_MIN_WIDTH': 40.0, 'DM_RANGE_MAX_WIDTH': 250.0})
    ]
    
    # Candidato de prueba
    dm_optimal = 450.0
    confidence = 0.85
    
    print(f"Candidato de prueba: DM = {dm_optimal} pc cm⁻³, Confianza = {confidence:.2f}")
    print("-" * 80)
    
    for config_name, config_params in configs:
        # Aplicar configuración temporal
        original_values = {}
        for param, value in config_params.items():
            original_values[param] = getattr(config, param, None)
            setattr(config, param, value)
        
        try:
            # Calcular rango
            dm_plot_min, dm_plot_max = get_dynamic_dm_range_for_candidate(
                dm_optimal=dm_optimal,
                config_module=config,
                confidence=confidence
            )
            
            range_width = dm_plot_max - dm_plot_min
            zoom_factor = (config.DM_max - config.DM_min) / range_width
            
            print(f"{config_name:<12}: {dm_plot_min:>6.0f} - {dm_plot_max:>6.0f} pc cm⁻³ "
                  f"(ancho: {range_width:>5.0f}, zoom: {zoom_factor:>5.1f}x)")
        
        finally:
            # Restaurar valores originales
            for param, value in original_values.items():
                if value is not None:
                    setattr(config, param, value)

if __name__ == "__main__":
    print("Iniciando demostración del Sistema de DM Dinámico...")
    
    # Ejecutar simulación
    results = simulate_detection_scenario()
    
    # Crear plots comparativos
    create_comparison_plots(results)
    
    # Demostrar opciones de configuración
    demonstrate_configuration_options()
    
    print("\n" + "="*60)
    print("✓ Demostración completada exitosamente!")
    print("✓ Plots y resúmenes guardados en: demo_dm_dynamic_plots/")
    print("✓ El sistema de DM dinámico está listo para su uso en el pipeline")
    print("="*60)
