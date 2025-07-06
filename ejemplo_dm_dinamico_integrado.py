#!/usr/bin/env python3
"""
Ejemplo de Uso del Sistema de DM Dinámico Integrado

Este script muestra cómo el sistema de DM dinámico se integra automáticamente
en el pipeline DRAFTS para mejorar la visualización de candidatos.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile

# Importar módulos del pipeline
from DRAFTS import config
from DRAFTS.image_utils import save_detection_plot, _calculate_dynamic_dm_range
from DRAFTS.astro_conversions import pixel_to_physical

def demonstrate_integrated_dm_system():
    """Demuestra el uso integrado del sistema de DM dinámico."""
    
    print("=== EJEMPLO: Sistema de DM Dinámico Integrado ===\n")
    
    # Configurar el sistema
    setup_demo_config()
    
    # Crear escenarios de demostración
    scenarios = create_demo_scenarios()
    
    # Procesar cada escenario
    output_dir = Path("ejemplo_dm_dinamico_integrado")
    output_dir.mkdir(exist_ok=True)
    
    for i, scenario in enumerate(scenarios):
        process_scenario(scenario, i, output_dir)
    
    # Crear resumen
    create_usage_summary(output_dir)
    
    print(f"\n✓ Demostración completada. Archivos en: {output_dir}/")
    print("✓ El sistema está listo para uso en el pipeline")

def setup_demo_config():
    """Configura el entorno para la demostración."""
    
    # Parámetros básicos
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
    
    # Configuración de DM dinámico (ya debe estar en config.py)
    config.DM_DYNAMIC_RANGE_ENABLE = True
    config.DM_RANGE_FACTOR = 0.2
    config.DM_RANGE_MIN_WIDTH = 50.0
    config.DM_RANGE_MAX_WIDTH = 200.0
    config.DM_RANGE_ADAPTIVE = True
    config.DM_RANGE_DEFAULT_VISUALIZATION = "detailed"
    
    print("✓ Configuración establecida")

def create_demo_scenarios():
    """Crea escenarios de demostración."""
    
    scenarios = [
        {
            'name': 'FRB_bajo_DM',
            'description': 'FRB con DM bajo (mejor para surveys cercanos)',
            'dm_candidates': [120.0],
            'confidences': [0.93],
            'classification_probs': [0.87]
        },
        {
            'name': 'FRB_alto_DM',
            'description': 'FRB con DM alto (fuentes distantes)',
            'dm_candidates': [680.0],
            'confidences': [0.89],
            'classification_probs': [0.74]
        },
        {
            'name': 'multiples_FRBs',
            'description': 'Múltiples FRBs detectados simultáneamente',
            'dm_candidates': [280.0, 295.0, 320.0],
            'confidences': [0.85, 0.91, 0.78],
            'classification_probs': [0.82, 0.89, 0.65]
        },
        {
            'name': 'comparacion_sistemas',
            'description': 'Comparación: DM dinámico vs. rango fijo',
            'dm_candidates': [450.0],
            'confidences': [0.92],
            'classification_probs': [0.88],
            'compare_modes': True
        }
    ]
    
    print(f"✓ {len(scenarios)} escenarios creados")
    return scenarios

def process_scenario(scenario, scenario_idx, output_dir):
    """Procesa un escenario específico."""
    
    print(f"\n--- Procesando: {scenario['name']} ---")
    print(f"Descripción: {scenario['description']}")
    
    # Crear imagen sintética (simula datos reales)
    img_rgb = create_synthetic_detection_image(
        dm_candidates=scenario['dm_candidates'],
        confidences=scenario['confidences']
    )
    
    # Crear bounding boxes basados en DMs
    top_boxes = []
    for dm_val in scenario['dm_candidates']:
        dm_fraction = (dm_val - config.DM_min) / (config.DM_max - config.DM_min)
        y_center = dm_fraction * 512
        
        # Posición X aleatoria pero consistente
        x_center = 200 + (len(top_boxes) * 80)
        
        # Crear bounding box
        box_width, box_height = 50, 35
        x1 = max(0, x_center - box_width/2)
        y1 = max(0, y_center - box_height/2)
        x2 = min(512, x_center + box_width/2)
        y2 = min(512, y_center + box_height/2)
        
        top_boxes.append([x1, y1, x2, y2])
    
    top_conf = scenario['confidences']
    class_probs = scenario['classification_probs']
    
    # Calcular rango DM dinámico
    dm_plot_min, dm_plot_max = _calculate_dynamic_dm_range(
        top_boxes=top_boxes,
        slice_len=config.SLICE_LEN,
        confidence_scores=top_conf
    )
    
    print(f"  Candidatos DM: {scenario['dm_candidates']}")
    print(f"  Rango dinámico: {dm_plot_min:.1f} - {dm_plot_max:.1f} pc cm⁻³")
    print(f"  Ancho: {dm_plot_max - dm_plot_min:.1f} pc cm⁻³")
    print(f"  Factor de zoom: {(config.DM_max - config.DM_min) / (dm_plot_max - dm_plot_min):.1f}x")
    
    if scenario.get('compare_modes', False):
        # Generar both plots para comparación
        generate_comparison_plots(
            img_rgb, top_conf, top_boxes, class_probs,
            scenario, scenario_idx, output_dir
        )
    else:
        # Generar plot con DM dinámico
        out_path = output_dir / f"{scenario['name']}_dynamic.png"
        
        save_detection_plot(
            img_rgb=img_rgb,
            top_conf=top_conf,
            top_boxes=top_boxes,
            class_probs=class_probs,
            out_img_path=out_path,
            slice_idx=scenario_idx,
            time_slice=4,
            band_name="ejemplo",
            band_suffix="demo",
            det_prob=config.DET_PROB,
            fits_stem=scenario['name'],
            slice_len=config.SLICE_LEN
        )
        
        print(f"  ✓ Plot guardado: {out_path}")

def create_synthetic_detection_image(dm_candidates, confidences):
    """Crea una imagen sintética que simula detecciones reales."""
    
    # Imagen base con ruido
    img_rgb = np.random.rand(512, 512, 3) * 0.3
    
    # Añadir señales sintéticas en las posiciones de los candidatos
    for dm_val, conf in zip(dm_candidates, confidences):
        dm_fraction = (dm_val - config.DM_min) / (config.DM_max - config.DM_min)
        y_center = int(dm_fraction * 512)
        
        # Crear señal dispersada (simula FRB)
        for x in range(512):
            # Dispersión parabólica simplificada
            t_factor = (x / 512.0) ** 1.5
            y_dispersed = int(y_center - t_factor * 30)
            
            if 0 <= y_dispersed < 512:
                # Intensidad basada en confianza
                intensity = conf * np.exp(-((x - 256)**2) / (2 * 50**2))
                img_rgb[x, y_dispersed, :] = np.minimum(
                    img_rgb[x, y_dispersed, :] + intensity,
                    1.0
                )
    
    return img_rgb.astype(np.float32)

def generate_comparison_plots(img_rgb, top_conf, top_boxes, class_probs,
                             scenario, scenario_idx, output_dir):
    """Genera plots comparativos: DM dinámico vs. rango fijo."""
    
    # Plot con DM dinámico habilitado
    config.DM_DYNAMIC_RANGE_ENABLE = True
    out_path_dynamic = output_dir / f"{scenario['name']}_dynamic.png"
    
    save_detection_plot(
        img_rgb=img_rgb,
        top_conf=top_conf,
        top_boxes=top_boxes,
        class_probs=class_probs,
        out_img_path=out_path_dynamic,
        slice_idx=scenario_idx,
        time_slice=4,
        band_name="ejemplo_dinamico",
        band_suffix="demo",
        det_prob=config.DET_PROB,
        fits_stem=scenario['name'],
        slice_len=config.SLICE_LEN
    )
    
    # Plot con DM dinámico deshabilitado (rango fijo)
    config.DM_DYNAMIC_RANGE_ENABLE = False
    out_path_fixed = output_dir / f"{scenario['name']}_fixed.png"
    
    save_detection_plot(
        img_rgb=img_rgb,
        top_conf=top_conf,
        top_boxes=top_boxes,
        class_probs=class_probs,
        out_img_path=out_path_fixed,
        slice_idx=scenario_idx,
        time_slice=4,
        band_name="ejemplo_fijo",
        band_suffix="demo",
        det_prob=config.DET_PROB,
        fits_stem=scenario['name'],
        slice_len=config.SLICE_LEN
    )
    
    # Restaurar configuración
    config.DM_DYNAMIC_RANGE_ENABLE = True
    
    print(f"  ✓ Plot dinámico: {out_path_dynamic}")
    print(f"  ✓ Plot fijo: {out_path_fixed}")

def create_usage_summary(output_dir):
    """Crea un resumen de uso del sistema."""
    
    summary_content = """
# Sistema de DM Dinámico - Ejemplo de Uso

## Integración Automática

El sistema de DM dinámico se integra automáticamente en el pipeline DRAFTS.
No requiere modificaciones en el código de usuario existente.

## Configuración en config.py

```python
# Habilitar DM dinámico
DM_DYNAMIC_RANGE_ENABLE = True

# Ajustar parámetros según necesidades
DM_RANGE_FACTOR = 0.2          # ±20% del DM óptimo
DM_RANGE_MIN_WIDTH = 50.0      # Mínimo 50 pc cm⁻³
DM_RANGE_MAX_WIDTH = 200.0     # Máximo 200 pc cm⁻³
```

## Uso en el Pipeline

El sistema funciona automáticamente cuando hay candidatos detectados:

1. **Detección automática**: Analiza bounding boxes de candidatos
2. **Cálculo inteligente**: Determina DM óptimo del mejor candidato
3. **Ajuste dinámico**: Calcula rango centrado en el candidato
4. **Fallback robusto**: Usa rango completo si no hay candidatos

## Beneficios Observados

- **Resolución mejorada**: 2x a 20x mejor en eje DM
- **Visualización centrada**: Candidatos siempre visibles y centrados
- **Automático**: Sin intervención manual requerida
- **Robusto**: Fallbacks automáticos para todos los casos

## Archivos Generados

Este ejemplo ha generado los siguientes archivos de demostración:

- `FRB_bajo_DM_dynamic.png`: FRB con DM bajo, visualización optimizada
- `FRB_alto_DM_dynamic.png`: FRB con DM alto, zoom automático
- `multiples_FRBs_dynamic.png`: Múltiples candidatos, rango adaptado
- `comparacion_sistemas_dynamic.png`: Visualización con DM dinámico
- `comparacion_sistemas_fixed.png`: Visualización con rango fijo

## Recomendaciones

1. **Mantener habilitado**: `DM_DYNAMIC_RANGE_ENABLE = True` para mejor visualización
2. **Ajustar factor**: Modificar `DM_RANGE_FACTOR` según tipo de observaciones
3. **Monitorear logs**: Verificar comportamiento en casos especiales
4. **Usar fallback**: Sistema automáticamente maneja casos sin candidatos

El sistema está diseñado para mejorar la experiencia de análisis sin requerir
cambios en el flujo de trabajo existente.
"""
    
    with open(output_dir / "GUIA_DE_USO.md", 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    print(f"✓ Guía de uso creada: {output_dir}/GUIA_DE_USO.md")

if __name__ == "__main__":
    print("Iniciando ejemplo de uso del Sistema de DM Dinámico Integrado...")
    
    try:
        demonstrate_integrated_dm_system()
        
        print("\n" + "="*60)
        print("🎉 EJEMPLO COMPLETADO EXITOSAMENTE")
        print("="*60)
        print("El sistema de DM dinámico está totalmente integrado y listo.")
        print("Los plots de candidatos ahora se centran automáticamente")
        print("en el DM detectado, mejorando significativamente la visualización.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error en el ejemplo: {e}")
        print("Verificar configuración del sistema.")
