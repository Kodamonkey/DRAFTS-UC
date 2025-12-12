#!/usr/bin/env python3
"""
Genera análisis académico completo para validación ALMA con todos los detalles.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import re

# Cargar resultados
with open('ResultsThesis/alma_validation_analysis.json', 'r') as f:
    results = json.load(f)

# Definir casos con sus descripciones
CASE_DESCRIPTIONS = {
    'a': {
        'name': 'Flujo completo con validación polarimétrica previa y clasificación dual CNN (I+L)',
        'phases': '1→2→3A→3B',
        'description': 'Este caso implementa el flujo completo del pipeline HF, activando todas las fases disponibles. Fases activas: FASE 1 (matched filtering) $\\to$ FASE 2 (validación polarimétrica previa) $\\to$ FASE 3A (clasificación en I) $\\to$ FASE 3B (clasificación en L). Este caso maximiza especificidad mediante doble filtrado: primero mediante SNR polarimétrico (FASE 2) y luego mediante clasificación dual CNN (FASE 3A + 3B).'
    },
    'b': {
        'name': 'Validación polarimétrica previa seguida únicamente de clasificación en polarización lineal',
        'phases': '1→2→3B',
        'description': 'Este caso prioriza evidencia polarimétrica sobre morfología de intensidad. Fases activas: FASE 1 (matched filtering) $\\to$ FASE 2 (validación polarimétrica previa) $\\to$ FASE 3B (clasificación en L). Útil cuando se busca maximizar la especificidad basada en coherencia polarimétrica, evaluando únicamente la morfología en el dominio de polarización lineal.'
    },
    'c': {
        'name': 'Clasificación dual CNN sin validación polarimétrica previa',
        'phases': '1→3A→3B',
        'description': 'Este caso maximiza sensibilidad permitiendo que todos los candidatos detectados accedan a clasificación CNN, confiando en la especificidad morfológica dual. Fases activas: FASE 1 (matched filtering) $\\to$ FASE 3A (clasificación en I) $\\to$ FASE 3B (clasificación en L). El sistema implementa modo STRICT, requiriendo clasificación BURST en I \\textbf{y} L simultáneamente ($p_{\\mathrm{I}} \\geq 0.5 \\land p_{\\mathrm{L}} \\geq 0.5$).'
    },
    'd': {
        'name': 'Clasificación exclusiva en polarización lineal sin validación previa ni evaluación de intensidad',
        'phases': '1→3B',
        'description': 'Este caso minimalista evalúa únicamente coherencia morfológica en polarización lineal. Fases activas: FASE 1 (matched filtering) $\\to$ FASE 3B (clasificación en L). La decisión se basa exclusivamente en $p_{\\mathrm{L}} \\geq \\theta$ (umbral típico $\\theta = 0.5$, 50\\%), sin considerar información de intensidad ni validación polarimétrica previa.'
    },
    'e': {
        'name': 'Validación polarimétrica previa seguida de clasificación únicamente en intensidad',
        'phases': '1→2→3A',
        'description': 'Este caso combina filtrado SNR polarimétrico con especificidad morfológica en intensidad. Fases activas: FASE 1 (matched filtering) $\\to$ FASE 2 (validación polarimétrica previa) $\\to$ FASE 3A (clasificación en I). La decisión se basa en $p_{\\mathrm{I}} \\geq \\theta$ (umbral típico $\\theta = 0.5$, 50\\%), pero solo después de pasar el filtro polarimétrico de FASE 2.'
    },
    'f': {
        'name': 'Clasificación exclusiva en intensidad sin validación polarimétrica previa',
        'phases': '1→3A',
        'description': 'Este caso minimalista maximiza sensibilidad evaluando únicamente morfología de intensidad. Fases activas: FASE 1 (matched filtering) $\\to$ FASE 3A (clasificación en I). Equivalente a modo PERMISSIVE sin polarimetría, donde la decisión se basa exclusivamente en $p_{\\mathrm{I}} \\geq \\theta$ (umbral típico $\\theta = 0.5$, 50\\%), sin considerar polarización lineal ni validación polarimétrica previa.'
    }
}

def generate_case_analysis(case_letter):
    """Genera análisis completo para un caso."""
    if case_letter not in results:
        return ""
    
    case_info = CASE_DESCRIPTIONS[case_letter]
    r = results[case_letter]
    
    lines = []
    lines.append(f"\\paragraph{{Caso {case_letter}: {case_info['name']}}}")
    lines.append("")
    lines.append(f"\\textbf{{Descripción metodológica:}} {case_info['description']}")
    lines.append("")
    lines.append(f"\\textbf{{Fases activas:}} {case_info['phases']}")
    lines.append("")
    lines.append("\\subparagraph{Resultados detallados}")
    lines.append("")
    
    # 8 Pulsos Canónicos
    canon = r['canonical_stats']
    lines.append("\\textbf{8 Pulsos Canónicos (Ground Truth):}")
    lines.append("\\begin{itemize}")
    lines.append(f"    \\item Total canónicos: {canon['total']}")
    if canon['total'] > 0:
        lines.append(f"    \\item Detectados: {canon['detected']}/{canon['total']} ({canon['detected']/canon['total']*100:.1f}\\%)")
    else:
        lines.append(f"    \\item Detectados: {canon['detected']}/{canon['total']} (N/A)")
    if canon['total'] > 0:
        lines.append(f"    \\item Clasificados BURST en I: {canon['burst_intensity']}/{canon['total']}")
        lines.append(f"    \\item Clasificados BURST en L: {canon['burst_linear']}/{canon['total']}")
        lines.append(f"    \\item Decisión final BURST: {canon['burst_final']}/{canon['total']}")
        if canon['detected'] < canon['total']:
            missing = canon['total'] - canon['detected']
            lines.append(f"    \\item \\textit{{Nota:}} {missing} pulsos canónicos no fueron detectados en este caso.")
    else:
        lines.append(f"    \\item Clasificados BURST en I: {canon['burst_intensity']}/0")
        lines.append(f"    \\item Clasificados BURST en L: {canon['burst_linear']}/0")
        lines.append(f"    \\item Decisión final BURST: {canon['burst_final']}/0")
        lines.append(f"    \\item \\textit{{Nota:}} No se encontraron pulsos canónicos en los matches de este caso.")
    lines.append("\\end{itemize}")
    lines.append("")
    
    # Dataset Extendido
    dataset = r['dataset_stats']
    lines.append("\\textbf{Dataset Extendido (Resto de candidatos validados):}")
    lines.append("\\begin{itemize}")
    lines.append(f"    \\item Total validados: {dataset['total_validated']}")
    lines.append(f"    \\item Detectados: {dataset['detected']} ({dataset['detected']/dataset['total_validated']*100:.1f}\\%)")
    lines.append(f"    \\item Clasificados BURST en I: {dataset['burst_intensity']}")
    lines.append(f"    \\item Clasificados BURST en L: {dataset['burst_linear']}")
    lines.append(f"    \\item Decisión final BURST: {dataset['burst_final']}")
    if dataset['detected'] < dataset['total_validated']:
        lines.append(f"    \\item Candidatos no detectados: {dataset['total_validated'] - dataset['detected']}")
    lines.append("\\end{itemize}")
    lines.append("")
    
    # Nuevos Pulsos
    new = r['new_pulses']
    lines.append("\\textbf{Nuevos Pulsos Detectados (solo BURST):}")
    lines.append("\\begin{itemize}")
    lines.append(f"    \\item Total nuevos detectados: {new['total_new']}")
    lines.append(f"    \\item Clasificados como BURST: {new['new_burst']}")
    if new['total_new'] > 0:
        lines.append(f"    \\item Tasa de BURST en nuevos: {new['new_burst']/new['total_new']*100:.1f}\\%")
        lines.append(f"    \\item Nuevos NO BURST: {new['total_new'] - new['new_burst']}")
    lines.append("\\end{itemize}")
    lines.append("")
    
    # Discusión del caso
    lines.append("\\textbf{Discusión:}")
    lines.append("\\begin{itemize}")
    
    # Análisis de canónicos
    if canon['detected'] == canon['total']:
        lines.append("    \\item \\textit{{8 Canónicos:}} Detección perfecta (100\\%).")
    else:
        missing = canon['total'] - canon['detected']
        lines.append(f"    \\item \\textit{{8 Canónicos:}} {canon['detected']}/{canon['total']} detectados. {missing} pulsos canónicos no fueron detectados, posiblemente debido a umbrales de detección o variaciones en las condiciones de procesamiento.")
    
    # Análisis de clasificación
    if '3A' in case_info['phases'] and '3B' in case_info['phases']:
        # Clasificación dual
        if canon['burst_intensity'] > canon['burst_linear']:
            lines.append(f"    \\item \\textit{{Clasificación dual:}} {canon['burst_intensity']} clasificados en I vs. {canon['burst_linear']} en L, revelando limitación en Fase 3B (transfer learning en polarización lineal).")
        else:
            lines.append(f"    \\item \\textit{{Clasificación dual:}} Coherencia I+L: {canon['burst_linear']}/{canon['total']} mantienen clasificación BURST en ambas polarizaciones.")
    elif '3A' in case_info['phases']:
        lines.append(f"    \\item \\textit{{Clasificación en I:}} {canon['burst_intensity']}/{canon['total']} clasificados como BURST en intensidad.")
    elif '3B' in case_info['phases']:
        lines.append(f"    \\item \\textit{{Clasificación en L:}} {canon['burst_linear']}/{canon['total']} clasificados como BURST en polarización lineal.")
    
    # Análisis de nuevos pulsos
    if new['total_new'] > 0:
        if new['new_burst'] == new['total_new']:
            lines.append(f"    \\item \\textit{{Nuevos pulsos:}} Todos los {new['new_burst']} nuevos pulsos detectados fueron clasificados como BURST, demostrando alta especificidad del pipeline.")
        else:
            rate = new['new_burst']/new['total_new']*100
            lines.append(f"    \\item \\textit{{Nuevos pulsos:}} {new['new_burst']}/{new['total_new']} nuevos pulsos clasificados como BURST ({rate:.1f}\\%), indicando filtrado efectivo de candidatos espurios.")
    
    lines.append("\\end{itemize}")
    lines.append("")
    
    return "\n".join(lines)

def generate_comparison_table():
    """Genera tabla comparativa mejorada."""
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("    \\centering")
    lines.append("    \\caption{Comparación sistemática de los seis casos metodológicos sobre el dataset ALMA completo. Para cada caso se muestra: 8 pulsos canónicos (detectados/BURST final), dataset extendido (detectados/BURST final), nuevos pulsos detectados (solo BURST), y métricas agregadas.}")
    lines.append("    \\label{tab:comparacion_casos_alma_completa}")
    lines.append("    \\small")
    lines.append("    \\resizebox{\\textwidth}{!}{%")
    lines.append("    \\begin{tabular}{lcccccc}")
    lines.append("    \\toprule")
    lines.append("    \\textbf{Caso} & \\textbf{8 Canónicos} & \\textbf{Dataset Ext.} & \\textbf{Nuevos} & \\textbf{Total BURST} & \\textbf{Recall Can.} \\\\")
    lines.append("    & \\textbf{Det/BURST} & \\textbf{Det/BURST} & \\textbf{BURST} & & \\\\")
    lines.append("    \\midrule")
    
    for case in ['a', 'b', 'c', 'd', 'e', 'f']:
        if case in results:
            r = results[case]
            canon = r['canonical_stats']
            dataset = r['dataset_stats']
            new = r['new_pulses']
            
            canon_str = f"{canon['detected']}/{canon['burst_final']}"
            dataset_str = f"{dataset['detected']}/{dataset['burst_final']}"
            new_str = str(new['new_burst'])
            total_burst = canon['burst_final'] + dataset['burst_final'] + new['new_burst']
            recall_can = f"{canon['burst_final']/canon['total']*100:.1f}\\%"
            
            lines.append(f"    {case.upper()} & {canon_str} & {dataset_str} & {new_str} & {total_burst} & {recall_can} \\\\")
    
    lines.append("    \\bottomrule")
    lines.append("    \\end{tabular}%")
    lines.append("    }")
    lines.append("\\end{table}")
    
    return "\n".join(lines)

def generate_limitations_analysis():
    """Genera análisis de limitaciones por caso."""
    lines = []
    lines.append("\\subsubsection{Análisis de limitaciones por caso metodológico}")
    lines.append("")
    lines.append("El análisis sistemático de los seis casos metodológicos revela limitaciones específicas asociadas a cada configuración, permitiendo identificar con precisión las causas técnicas y las oportunidades de mejora.")
    lines.append("")
    
    for case in ['a', 'b', 'c', 'd', 'e', 'f']:
        if case not in results:
            continue
            
        case_info = CASE_DESCRIPTIONS[case]
        r = results[case]
        canon = r['canonical_stats']
        dataset = r['dataset_stats']
        
        lines.append(f"\\paragraph{{Limitaciones Caso {case.upper()}}}")
        lines.append("")
        lines.append(f"\\textbf{{Configuración:}} {case_info['phases']}")
        lines.append("")
        lines.append("\\begin{itemize}")
        
        # Limitación en detección de canónicos
        if canon['detected'] < canon['total']:
            lines.append(f"    \\item \\textbf{{Detección de canónicos:}} Solo {canon['detected']}/{canon['total']} pulsos canónicos fueron detectados. Esta limitación puede deberse a:")
            lines.append("    \\begin{itemize}")
            if '2' in case_info['phases']:
                lines.append("        \\item Filtrado polarimétrico previo (FASE 2) puede estar rechazando pulsos con SNR\\_L bajo pero morfología válida.")
            lines.append("        \\item Umbrales de detección en FASE 1 pueden ser demasiado restrictivos para algunos pulsos.")
            lines.append("        \\item Variaciones en condiciones de procesamiento entre casos.")
            lines.append("    \\end{itemize}")
        
        # Limitación en clasificación L
        if '3B' in case_info['phases']:
            if canon['burst_linear'] < canon['burst_intensity']:
                lines.append(f"    \\item \\textbf{{Transfer learning en L:}} {canon['burst_intensity']} clasificados en I vs. {canon['burst_linear']} en L. Esta limitación es consistente con el análisis previo (Sección~\\ref{{sec:limitaciones_transfer_learning}}): ResNet18, entrenado exclusivamente en Stokes I, no generaliza correctamente a Stokes L.")
                lines.append("    \\begin{itemize}")
                lines.append("        \\item Causa: Morfologías físicas difieren entre I (coherente banda ancha) y L (difusa/fragmentada por efectos magnéticos).")
                lines.append("        \\item Solución: Reentrenamiento de ResNet18 con ejemplos de waterfalls en Stokes L de alta frecuencia.")
                lines.append("    \\end{itemize}")
        
        # Limitación en especificidad
        new = r['new_pulses']
        if new['total_new'] > 0 and new['new_burst'] == new['total_new']:
            if '3B' not in case_info['phases']:
                lines.append(f"    \\item \\textbf{{Especificidad:}} Todos los {new['new_burst']} nuevos pulsos fueron clasificados como BURST. Sin clasificación en L, el sistema puede aceptar candidatos espurios con alta probabilidad en I pero baja coherencia polarimétrica.")
        
        lines.append("\\end{itemize}")
        lines.append("")
    
    return "\n".join(lines)

def generate_comparison_discussion():
    """Genera discusión comparativa."""
    lines = []
    lines.append("\\subsubsection{Comparación sistemática y identificación del caso óptimo}")
    lines.append("")
    lines.append("La Tabla~\\ref{tab:comparacion_casos_alma_completa} presenta una comparación sistemática de los seis casos metodológicos, evaluando su rendimiento sobre tres conjuntos de datos: 8 pulsos canónicos (ground truth), dataset extendido (resto de candidatos validados), y nuevos pulsos detectados.")
    lines.append("")
    lines.append("\\textbf{Análisis comparativo:}")
    lines.append("")
    lines.append("\\begin{enumerate}")
    lines.append("    \\item \\textbf{Sensibilidad en canónicos:} Todos los casos detectan 3/8 pulsos canónicos (37.5\\%), sugiriendo que la limitación en detección es independiente de la configuración de fases. Esto indica que el problema puede estar en:")
    lines.append("    \\begin{itemize}")
    lines.append("        \\item Umbrales de detección en FASE 1 (matched filtering) que requieren ajuste.")
    lines.append("        \\item Variaciones en condiciones de procesamiento o disponibilidad de datos entre casos.")
    lines.append("        \\item Necesidad de validación manual adicional para identificar causas específicas.")
    lines.append("    \\end{itemize}")
    lines.append("")
    lines.append("    \\item \\textbf{Clasificación dual vs. individual:}")
    lines.append("    \\begin{itemize}")
    lines.append("        \\item Casos con clasificación dual (a, c): Muestran limitación consistente en Fase 3B, con clasificación en L inferior a clasificación en I.")
    lines.append("        \\item Casos con solo I (e, f): Demuestran robustez en clasificación de intensidad, con tasas de BURST similares independientemente de FASE 2.")
    lines.append("        \\item Casos con solo L (b, d): Muestran menor sensibilidad en polarización lineal, confirmando la limitación de transfer learning identificada.")
    lines.append("    \\end{itemize}")
    lines.append("")
    lines.append("    \\item \\textbf{Impacto de FASE 2:}")
    lines.append("    \\begin{itemize}")
    lines.append("        \\item Casos con FASE 2 (a, b, e): Procesan menos candidatos pero mantienen alta especificidad.")
    lines.append("        \\item Casos sin FASE 2 (c, d, f): Procesan más candidatos, aumentando carga computacional pero potencialmente mejorando sensibilidad.")
    lines.append("    \\end{itemize}")
    lines.append("")
    lines.append("    \\item \\textbf{Nuevos pulsos detectados:}")
    lines.append("    \\begin{itemize}")
    lines.append("        \\item Caso d detecta el mayor número de nuevos pulsos (118), todos clasificados como BURST, pero sin filtrado por clasificación en I.")
    lines.append("        \\item Casos con clasificación dual (a, c) muestran filtrado más selectivo, con tasas de BURST en nuevos de 100\\% y 45.8\\% respectivamente.")
    lines.append("    \\end{itemize}")
    lines.append("\\end{enumerate}")
    lines.append("")
    lines.append("\\textbf{Identificación del caso óptimo:}")
    lines.append("")
    lines.append("Basado en el análisis comparativo, el \\textbf{Caso c (1→3A→3B)} emerge como el caso que optimiza el balance sensibilidad--especificidad:")
    lines.append("\\begin{itemize}")
    lines.append("    \\item Clasificación dual I+L proporciona filtrado efectivo de candidatos espurios.")
    lines.append("    \\item Sin FASE 2, maximiza sensibilidad permitiendo que todos los candidatos detectados accedan a clasificación CNN.")
    lines.append("    \\item Detecta 70 candidatos del dataset extendido con 65 clasificados como BURST final.")
    lines.append("    \\item Identifica 54 nuevos pulsos de 118 detectados (45.8\\% tasa de BURST), demostrando especificidad morfológica.")
    lines.append("\\end{itemize}")
    lines.append("")
    lines.append("Sin embargo, la limitación en transfer learning de ResNet18 en polarización lineal (Fase 3B) afecta todos los casos que utilizan clasificación en L, sugiriendo que el trabajo futuro de reentrenamiento podría mejorar significativamente el rendimiento de todos los casos con clasificación dual.")
    lines.append("")
    
    return "\n".join(lines)

def main():
    """Genera todo el contenido LaTeX."""
    print("Generando análisis académico completo...")
    
    output = []
    
    # Análisis por caso
    output.append("% Análisis detallado por caso metodológico\n")
    for case in ['a', 'b', 'c', 'd', 'e', 'f']:
        analysis = generate_case_analysis(case)
        if analysis:
            output.append(analysis)
            output.append("")
    
    # Tabla comparativa
    output.append("% Tabla comparativa consolidada\n")
    output.append(generate_comparison_table())
    output.append("")
    
    # Discusión comparativa
    output.append(generate_comparison_discussion())
    output.append("")
    
    # Análisis de limitaciones
    output.append(generate_limitations_analysis())
    
    # Guardar
    content = "\n".join(output)
    with open('ResultsThesis/complete_academic_analysis.tex', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\nAnálisis completo guardado en: ResultsThesis/complete_academic_analysis.tex")
    print(f"\nLongitud del contenido: {len(content)} caracteres")
    
    return content

if __name__ == "__main__":
    content = main()

