#!/usr/bin/env python3
"""
Genera tablas LaTeX detalladas para el análisis académico de validación ALMA.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Cargar resultados del análisis
with open('ResultsThesis/alma_validation_analysis.json', 'r') as f:
    results = json.load(f)

def generate_canonical_table(results_dict):
    """Genera tabla LaTeX para los 8 pulsos canónicos."""
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("    \\centering")
    lines.append("    \\caption{Validación de los 8 pulsos canónicos (ground truth) para los seis casos metodológicos.}")
    lines.append("    \\label{tab:canonical_pulses_validation}")
    lines.append("    \\small")
    lines.append("    \\resizebox{\\textwidth}{!}{%")
    lines.append("    \\begin{tabular}{lccccccc}")
    lines.append("    \\toprule")
    lines.append("    \\textbf{Pulso} & \\textbf{Caso a} & \\textbf{Caso b} & \\textbf{Caso c} & \\textbf{Caso d} & \\textbf{Caso e} & \\textbf{Caso f} \\\\")
    lines.append("    \\midrule")
    
    # Nombres de los 8 canónicos
    canonical_names = [
        "142\\_0003 (39.977 s)",
        "142\\_0006 (10.882 s)",
        "142\\_0006 (25.829 s)",
        "153\\_0006 (23.444 s)",
        "230\\_0002 (2.3 s)",
        "230\\_0002 (17.395 s)",
        "230\\_0003 (36.548 s)",
        "242\\_0005 (44.919 s)",
    ]
    
    # Por ahora, usar estadísticas agregadas
    # Nota: Para una tabla detallada por pulso, necesitaríamos cargar los datos completos
    lines.append("    \\multicolumn{7}{l}{\\textit{Estadísticas agregadas:}} \\\\")
    
    for case in ['a', 'b', 'c', 'd', 'e', 'f']:
        if case in results:
            stats = results[case]['canonical_stats']
            detected = stats['detected']
            total = stats['total']
            burst_final = stats['burst_final']
            lines.append(f"    Caso {case}: {detected}/{total} detectados, {burst_final}/{total} BURST final \\\\")
    
    lines.append("    \\bottomrule")
    lines.append("    \\end{tabular}%")
    lines.append("    }")
    lines.append("\\end{table}")
    
    return "\n".join(lines)

def generate_comparison_table(results_dict):
    """Genera tabla comparativa consolidada."""
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("    \\centering")
    lines.append("    \\caption{Comparación sistemática de los seis casos metodológicos sobre el dataset ALMA completo.}")
    lines.append("    \\label{tab:comparacion_casos_alma_completa}")
    lines.append("    \\small")
    lines.append("    \\resizebox{\\textwidth}{!}{%")
    lines.append("    \\begin{tabular}{lcccccc}")
    lines.append("    \\toprule")
    lines.append("    \\textbf{Caso} & \\textbf{8 Canónicos} & \\textbf{Dataset Ext.} & \\textbf{Nuevos} & \\textbf{Total BURST} & \\textbf{Especificidad} \\\\")
    lines.append("    & \\textbf{Det/BURST} & \\textbf{Det/BURST} & \\textbf{BURST} & & \\\\")
    lines.append("    \\midrule")
    
    for case in ['a', 'b', 'c', 'd', 'e', 'f']:
        if case in results_dict:
            r = results_dict[case]
            canon = r['canonical_stats']
            dataset = r['dataset_stats']
            new = r['new_pulses']
            
            canon_str = f"{canon['detected']}/{canon['burst_final']}"
            dataset_str = f"{dataset['detected']}/{dataset['burst_final']}"
            new_str = str(new['new_burst'])
            total_burst = canon['burst_final'] + dataset['burst_final'] + new['new_burst']
            
            # Calcular especificidad aproximada (BURST final / detectados)
            total_detected = canon['detected'] + dataset['detected']
            if total_detected > 0:
                specificity = f"{(total_burst / total_detected * 100):.1f}\\%"
            else:
                specificity = "N/A"
            
            lines.append(f"    {case.upper()} & {canon_str} & {dataset_str} & {new_str} & {total_burst} & {specificity} \\\\")
    
    lines.append("    \\bottomrule")
    lines.append("    \\end{tabular}%")
    lines.append("    }")
    lines.append("\\end{table}")
    
    return "\n".join(lines)

def generate_detailed_case_analysis(case_letter, results_dict):
    """Genera análisis detallado para un caso específico."""
    if case_letter not in results_dict:
        return ""
    
    r = results_dict[case_letter]
    lines = []
    
    lines.append(f"\\paragraph{{Caso {case_letter.upper()}: Análisis detallado}}")
    lines.append("")
    lines.append("\\textbf{8 Pulsos Canónicos (Ground Truth):}")
    lines.append("\\begin{itemize}")
    canon = r['canonical_stats']
    lines.append(f"    \\item Detectados: {canon['detected']}/{canon['total']} ({canon['detected']/canon['total']*100:.1f}\\%)")
    lines.append(f"    \\item Clasificados BURST en I: {canon['burst_intensity']}/{canon['total']}")
    lines.append(f"    \\item Clasificados BURST en L: {canon['burst_linear']}/{canon['total']}")
    lines.append(f"    \\item Decisión final BURST: {canon['burst_final']}/{canon['total']}")
    lines.append("\\end{itemize}")
    lines.append("")
    
    lines.append("\\textbf{Dataset Extendido (Resto de candidatos validados):}")
    lines.append("\\begin{itemize}")
    dataset = r['dataset_stats']
    lines.append(f"    \\item Total validados: {dataset['total_validated']}")
    lines.append(f"    \\item Detectados: {dataset['detected']} ({dataset['detected']/dataset['total_validated']*100:.1f}\\%)")
    lines.append(f"    \\item Clasificados BURST en I: {dataset['burst_intensity']}")
    lines.append(f"    \\item Clasificados BURST en L: {dataset['burst_linear']}")
    lines.append(f"    \\item Decisión final BURST: {dataset['burst_final']}")
    lines.append("\\end{itemize}")
    lines.append("")
    
    lines.append("\\textbf{Nuevos Pulsos Detectados (solo BURST):}")
    lines.append("\\begin{itemize}")
    new = r['new_pulses']
    lines.append(f"    \\item Total nuevos detectados: {new['total_new']}")
    lines.append(f"    \\item Clasificados como BURST: {new['new_burst']}")
    if new['total_new'] > 0:
        lines.append(f"    \\item Tasa de BURST en nuevos: {new['new_burst']/new['total_new']*100:.1f}\\%")
    lines.append("\\end{itemize}")
    lines.append("")
    
    return "\n".join(lines)

def main():
    """Genera todas las tablas LaTeX."""
    print("Generando tablas LaTeX...")
    
    # Tabla comparativa
    comparison_table = generate_comparison_table(results)
    print("\n=== TABLA COMPARATIVA ===")
    print(comparison_table)
    
    # Análisis detallado por caso
    print("\n\n=== ANÁLISIS DETALLADO POR CASO ===")
    for case in ['a', 'b', 'c', 'd', 'e', 'f']:
        analysis = generate_detailed_case_analysis(case, results)
        if analysis:
            print(f"\n--- Caso {case.upper()} ---")
            print(analysis)
    
    # Guardar en archivo
    output_file = "ResultsThesis/latex_tables_output.tex"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("% Tablas LaTeX generadas automáticamente\n")
        f.write("% Para análisis de validación ALMA\n\n")
        f.write(comparison_table)
        f.write("\n\n")
        for case in ['a', 'b', 'c', 'd', 'e', 'f']:
            analysis = generate_detailed_case_analysis(case, results)
            if analysis:
                f.write(analysis)
                f.write("\n\n")
    
    print(f"\n\nTablas guardadas en: {output_file}")

if __name__ == "__main__":
    main()

