#!/usr/bin/env python3
"""
Genera tabla detallada de los 8 pulsos canónicos para cada caso.
"""

import pandas as pd
import numpy as np
import json
import re

# Cargar tabla validada
df_validated = pd.read_csv("ResultsThesis/tabla_transformada.csv", sep="\t")
df_validated['candidato_tiempo_clean'] = df_validated['candidato tiempo'].astype(str).apply(
    lambda x: re.sub(r'\s*\([^)]*\)', '', x).strip() if pd.notna(x) else ""
)
df_validated['candidato_tiempo_clean'] = pd.to_numeric(df_validated['candidato_tiempo_clean'], errors='coerce')

# 8 Pulsos canónicos
CANONICAL_PULSES = [
    {"file": "2017-04-03-08_16_13_0003", "time": 39.977, "name": "142\\_0003"},
    {"file": "2017-04-03-08_16_13_0006", "time": 10.882, "name": "142\\_0006 (1)"},
    {"file": "2017-04-03-08_16_13_0006", "time": 25.829, "name": "142\\_0006 (2)"},
    {"file": "2017-04-03-08_55_22_0006", "time": 23.444, "name": "153\\_0006"},
    {"file": "2017-04-03-12_56_05_0002", "time": 2.3, "name": "230\\_0002 (1)"},
    {"file": "2017-04-03-12_56_05_0002", "time": 17.395, "name": "230\\_0002 (2)"},
    {"file": "2017-04-03-12_56_05_0003", "time": 36.548, "name": "230\\_0003"},
    {"file": "2017-04-03-13_38_31_0005", "time": 44.919, "name": "242\\_0005"},
]

CASE_MAPPING = {
    "matches_all.xlsx": "a",
    "matches_no-class-intensity.xlsx": "b",
    "matches_no-classification-linear.xlsx": "e",
    "matches_no-phase2-no-classification-intensity.xlsx": "d",
    "matches_no-phase2-no-classification-linear.xlsx": "c",
    "matches_no-phase2.xlsx": "f",
}

def normalize_filename(filename):
    """Normaliza nombres de archivo."""
    if pd.isna(filename):
        return ""
    filename = str(filename).lower().strip().replace('.fits', '').replace('_', '-').replace(' ', '')
    return filename

def files_match(file1, file2):
    """Verifica si dos nombres coinciden."""
    norm1 = normalize_filename(file1)
    norm2 = normalize_filename(file2)
    if norm1 == norm2:
        return True
    parts1 = set([p for p in norm1.split('-') if len(p) > 2])
    parts2 = set([p for p in norm2.split('-') if len(p) > 2])
    if len(parts1) > 0 and len(parts2) > 0:
        common = parts1.intersection(parts2)
        if len(common) >= min(3, len(parts1), len(parts2)):
            return True
    return False

def analyze_canonical_for_case(matches_file, case_letter):
    """Analiza los 8 canónicos para un caso específico."""
    df = pd.read_excel(f"ResultsThesis/{matches_file}")
    df.columns = [c.split('.')[0] if '.' in c else c for c in df.columns]
    df_validated_only = df[df['match_type'] == 'validated'].copy()
    
    canonical_details = []
    
    for pulse in CANONICAL_PULSES:
        # Buscar en matches
        file_col = 'validated_nombre_archivo'
        time_col = 'validated_candidato_tiempo'
        
        if file_col not in df_validated_only.columns:
            canonical_details.append({
                'name': pulse['name'],
                'file': pulse['file'],
                'time': pulse['time'],
                'detected': False,
                'burst_i': False,
                'burst_l': False,
                'burst_final': False,
                'detection_prob': np.nan,
                'class_prob_i': np.nan,
                'class_prob_l': np.nan,
            })
            continue
        
        file_series = df_validated_only[file_col]
        if isinstance(file_series, pd.DataFrame):
            file_series = file_series.iloc[:, 0]
        
        time_series = df_validated_only[time_col]
        if isinstance(time_series, pd.DataFrame):
            time_series = time_series.iloc[:, 0]
        
        # Matching
        file_match = file_series.apply(lambda x: files_match(x, pulse['file']))
        time_numeric = pd.to_numeric(time_series, errors='coerce')
        time_match = (time_numeric >= pulse['time'] - 0.1) & (time_numeric <= pulse['time'] + 0.1)
        
        matches = df_validated_only[file_match & time_match]
        
        if len(matches) > 0:
            match_row = matches.iloc[0]
            has_match = match_row.get('has_match', False)
            
            if has_match:
                canonical_details.append({
                    'name': pulse['name'],
                    'file': pulse['file'],
                    'time': pulse['time'],
                    'detected': True,
                    'burst_i': match_row.get('detected_is_burst_intensity', '') == 'burst',
                    'burst_l': match_row.get('detected_is_burst_linear', '') == 'burst',
                    'burst_final': match_row.get('detected_is_burst', '') == 'burst',
                    'detection_prob': match_row.get('detected_detection_prob', np.nan),
                    'class_prob_i': match_row.get('detected_class_prob_intensity', np.nan),
                    'class_prob_l': match_row.get('detected_class_prob_linear', np.nan),
                })
            else:
                canonical_details.append({
                    'name': pulse['name'],
                    'file': pulse['file'],
                    'time': pulse['time'],
                    'detected': False,
                    'burst_i': False,
                    'burst_l': False,
                    'burst_final': False,
                    'detection_prob': np.nan,
                    'class_prob_i': np.nan,
                    'class_prob_l': np.nan,
                })
        else:
            canonical_details.append({
                'name': pulse['name'],
                'file': pulse['file'],
                'time': pulse['time'],
                'detected': False,
                'burst_i': False,
                'burst_l': False,
                'burst_final': False,
                'detection_prob': np.nan,
                'class_prob_i': np.nan,
                'class_prob_l': np.nan,
            })
    
    return canonical_details

def generate_canonical_table_latex():
    """Genera tabla LaTeX detallada de canónicos."""
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("    \\centering")
    lines.append("    \\caption{Validación detallada de los 8 pulsos canónicos (ground truth) para los seis casos metodológicos. Para cada pulso se indica: detección (✓/✗), clasificación BURST en I (✓/✗), clasificación BURST en L (✓/✗), y decisión final BURST (✓/✗).}")
    lines.append("    \\label{tab:canonical_pulses_detailed}")
    lines.append("    \\footnotesize")
    lines.append("    \\resizebox{\\textwidth}{!}{%")
    lines.append("    \\begin{tabular}{lcccccc}")
    lines.append("    \\toprule")
    lines.append("    \\textbf{Pulso} & \\textbf{Caso a} & \\textbf{Caso b} & \\textbf{Caso c} & \\textbf{Caso d} & \\textbf{Caso e} & \\textbf{Caso f} \\\\")
    lines.append("    \\midrule")
    
    # Analizar cada caso
    all_canonical_details = {}
    for matches_file, case_letter in CASE_MAPPING.items():
        details = analyze_canonical_for_case(matches_file, case_letter)
        all_canonical_details[case_letter] = details
    
    # Para cada pulso canónico, mostrar resultados en todos los casos
    for i, pulse in enumerate(CANONICAL_PULSES):
        pulse_name = pulse['name']
        row = [pulse_name]
        
        for case in ['a', 'b', 'c', 'd', 'e', 'f']:
            if case in all_canonical_details and i < len(all_canonical_details[case]):
                detail = all_canonical_details[case][i]
                if detail['detected']:
                    if detail['burst_final']:
                        row.append("✓")
                    else:
                        row.append("✗")
                else:
                    row.append("--")
            else:
                row.append("--")
        
        lines.append("    " + " & ".join(row) + " \\\\")
    
    lines.append("    \\midrule")
    totals = []
    for case in ['a', 'b', 'c', 'd', 'e', 'f']:
        if case in all_canonical_details:
            detected = sum(1 for d in all_canonical_details[case] if d['detected'])
            totals.append(str(detected))
        else:
            totals.append("--")
    lines.append("    \\textbf{{Total detectados}} & " + " & ".join(totals) + " \\\\")
    
    burst_totals = []
    for case in ['a', 'b', 'c', 'd', 'e', 'f']:
        if case in all_canonical_details:
            burst = sum(1 for d in all_canonical_details[case] if d['burst_final'])
            burst_totals.append(str(burst))
        else:
            burst_totals.append("--")
    lines.append("    \\textbf{{Total BURST final}} & " + " & ".join(burst_totals) + " \\\\")
    
    lines.append("    \\bottomrule")
    lines.append("    \\end{tabular}%")
    lines.append("    }")
    lines.append("\\end{table}")
    
    return "\n".join(lines)

if __name__ == "__main__":
    table = generate_canonical_table_latex()
    print(table)
    with open('ResultsThesis/canonical_table_detailed.tex', 'w', encoding='utf-8') as f:
        f.write(table)

