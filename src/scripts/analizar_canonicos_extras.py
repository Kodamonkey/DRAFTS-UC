#!/usr/bin/env python3
"""Script para analizar canónicos y pulsos extras por caso"""

import pandas as pd
import re
from pathlib import Path

# Importar función de analyze_alma_validation.py
import sys
sys.path.insert(0, 'src/scripts')
from analyze_alma_validation import load_canonical_pulses_from_config

# Cargar tabla_transformada para identificar canónicos
tabla_path = Path("ResultsThesis/tabla_transformada.csv")
df_validated = pd.read_csv(tabla_path, sep='\t', encoding='utf-8')

# Normalizar nombre_archivo para matching
def normalize_filename(filename: str) -> str:
    if pd.isna(filename) or filename == '':
        return ''
    filename = str(filename).strip().replace('-', '_').lower()
    return filename

df_validated['nombre_archivo_normalized'] = df_validated['nombre_archivo'].apply(normalize_filename)

# Limpiar tiempo de candidato
df_validated['candidato_tiempo_clean'] = df_validated['candidato tiempo'].astype(str).apply(
    lambda x: re.sub(r'\s*\([^)]*\)', '', x).strip() if pd.notna(x) else ""
)
df_validated['candidato_tiempo_clean'] = pd.to_numeric(df_validated['candidato_tiempo_clean'], errors='coerce')

# Cargar canónicos desde config.yaml
canonical_pulses = load_canonical_pulses_from_config()

# Identificar canónicos en tabla_transformada
canonical_mask_final = pd.Series([False] * len(df_validated), index=df_validated.index)

for canon in canonical_pulses:
    file_pattern = canon['file_pattern']
    subfolder = canon['subfolder']
    time_val = canon['time']
    tolerance = canon.get('tolerance', 0.1)
    
    # Buscar matches
    file_match = df_validated['nombre_archivo_normalized'].str.contains(
        normalize_filename(file_pattern), case=False, na=False
    )
    subfolder_match = (
        (df_validated['subfolder'].astype(str) == subfolder) |
        (df_validated['subfolder'].astype(str) == str(int(subfolder)))
    )
    time_match = (
        (df_validated['candidato_tiempo_clean'] >= time_val - tolerance) &
        (df_validated['candidato_tiempo_clean'] <= time_val + tolerance)
    )
    
    matches = df_validated[file_match & subfolder_match & time_match]
    if len(matches) > 0:
        canonical_mask_final.loc[matches.index] = True

# Los extras son el resto
extras_mask = ~canonical_mask_final

print("="*80)
print("ANÁLISIS DE PULSOS CANÓNICOS Y EXTRAS POR CASO")
print("="*80)
print()

# Mapeo de casos
case_mapping = {
    "matches_all.xlsx": "Caso a (All - todas las fases)",
    "matches_no-class-intensity.xlsx": "Caso b (no-class-intensity)",
    "matches_no-classification-linear.xlsx": "Caso e (no-class-linear)",
    "matches_no-phase2-no-classification-intensity.xlsx": "Caso d (no-phase2-no-class-intensity)",
    "matches_no-phase2-no-classification-linear.xlsx": "Caso c (no-phase2-no-class-linear)",
    "matches_no-phase2.xlsx": "Caso f (no-phase2)",
}

base_path = Path("ResultsThesis")

results_summary = []

for match_file, case_name in case_mapping.items():
    match_path = base_path / match_file
    
    if not match_path.exists():
        continue
    
    df_matches = pd.read_excel(match_path)
    
    # Filtrar solo candidatos validados (match_type == 'validated')
    df_validated_only = df_matches[df_matches['match_type'] == 'validated'].copy()
    
    # Identificar canónicos en matches
    # Filtrar matches que corresponden a canónicos
    canonical_matches = []
    for idx, row in df_validated[canonical_mask_final].iterrows():
        canon_file = str(row.get('nombre_archivo', '')).strip()
        canon_time = row.get('candidato_tiempo_clean')
        
        if pd.isna(canon_file) or canon_file == '' or pd.isna(canon_time):
            continue
        
        # Buscar en matches
        file_match = df_validated_only['validated_nombre_archivo'].astype(str).str.strip() == canon_file
        time_match = pd.to_numeric(
            df_validated_only['validated_candidato_tiempo'].astype(str).str.replace(
                r'\s*\([^)]*\)', '', regex=True
            ), errors='coerce'
        )
        time_match = abs(time_match - canon_time) < 0.1
        
        matches = df_validated_only[file_match & time_match]
        if len(matches) > 0:
            canonical_matches.append(matches.iloc[0])
    
    canonical_detected = len(canonical_matches)
    canonical_burst_i = 0
    canonical_burst_l = 0
    canonical_burst_final = 0
    
    for match_row in canonical_matches:
        if 'detected_is_burst_intensity' in match_row and match_row['detected_is_burst_intensity'] == 'burst':
            canonical_burst_i += 1
        if 'detected_is_burst_linear' in match_row and match_row['detected_is_burst_linear'] == 'burst':
            canonical_burst_l += 1
        if 'detected_is_burst' in match_row and match_row['detected_is_burst'] == 'burst':
            canonical_burst_final += 1
    
    # Analizar extras (resto de validados que no son canónicos)
    extras_matches = []
    for idx, row in df_validated[~canonical_mask_final].iterrows():
        extra_file = str(row.get('nombre_archivo', '')).strip()
        extra_time = row.get('candidato_tiempo_clean')
        
        if pd.isna(extra_file) or extra_file == '' or pd.isna(extra_time):
            continue
        
        # Buscar en matches
        file_match = df_validated_only['validated_nombre_archivo'].astype(str).str.strip() == extra_file
        time_match = pd.to_numeric(
            df_validated_only['validated_candidato_tiempo'].astype(str).str.replace(
                r'\s*\([^)]*\)', '', regex=True
            ), errors='coerce'
        )
        time_match = abs(time_match - extra_time) < 0.1
        
        matches = df_validated_only[file_match & time_match]
        if len(matches) > 0:
            extras_matches.append(matches.iloc[0])
    
    extras_detected = len(extras_matches)
    extras_burst_i = 0
    extras_burst_l = 0
    extras_burst_final = 0
    
    for match_row in extras_matches:
        if 'detected_is_burst_intensity' in match_row and match_row['detected_is_burst_intensity'] == 'burst':
            extras_burst_i += 1
        if 'detected_is_burst_linear' in match_row and match_row['detected_is_burst_linear'] == 'burst':
            extras_burst_l += 1
        if 'detected_is_burst' in match_row and match_row['detected_is_burst'] == 'burst':
            extras_burst_final += 1
    
    total_canonical = canonical_mask_final.sum()
    total_extras = (~canonical_mask_final).sum()
    
    results_summary.append({
        'Caso': case_name,
        'Canónicos Total': total_canonical,
        'Canónicos Detectados': canonical_detected,
        'Canónicos BURST I': canonical_burst_i,
        'Canónicos BURST L': canonical_burst_l,
        'Canónicos BURST Final': canonical_burst_final,
        'Extras Total': total_extras,
        'Extras Detectados': extras_detected,
        'Extras BURST I': extras_burst_i,
        'Extras BURST L': extras_burst_l,
        'Extras BURST Final': extras_burst_final,
    })
    
    # Imprimir resumen
    print(f"{case_name}")
    print(f"  {'─'*76}")
    print(f"  8 Pulsos Canónicos:")
    print(f"    • Total: {total_canonical}")
    print(f"    • Detectados: {canonical_detected}/{total_canonical} ({canonical_detected/total_canonical*100:.1f}%)")
    print(f"    • BURST Intensity: {canonical_burst_i}/{total_canonical}")
    print(f"    • BURST Linear: {canonical_burst_l}/{total_canonical}")
    print(f"    • BURST Final: {canonical_burst_final}/{total_canonical}")
    print(f"  ")
    print(f"  Pulsos Extras (resto del dataset):")
    print(f"    • Total: {total_extras}")
    print(f"    • Detectados: {extras_detected}/{total_extras} ({extras_detected/total_extras*100:.1f}%)")
    print(f"    • BURST Intensity: {extras_burst_i}/{total_extras}")
    print(f"    • BURST Linear: {extras_burst_l}/{total_extras}")
    print(f"    • BURST Final: {extras_burst_final}/{total_extras}")
    print()

# Tabla comparativa
print("="*80)
print("TABLA COMPARATIVA")
print("="*80)
print()

print(f"{'Caso':<35} {'Can.Det':<8} {'Can.BI':<8} {'Can.BL':<8} {'Can.BF':<8} {'Ext.Det':<8} {'Ext.BI':<8} {'Ext.BL':<8} {'Ext.BF':<8}")
print("-"*100)
for row in results_summary:
    caso_short = row['Caso'].split('(')[0].strip()[:30]
    print(f"{caso_short:<35} {row['Canónicos Detectados']}/{row['Canónicos Total']:<6} "
          f"{row['Canónicos BURST I']}/{row['Canónicos Total']:<6} "
          f"{row['Canónicos BURST L']}/{row['Canónicos Total']:<6} "
          f"{row['Canónicos BURST Final']}/{row['Canónicos Total']:<6} "
          f"{row['Extras Detectados']}/{row['Extras Total']:<6} "
          f"{row['Extras BURST I']}/{row['Extras Total']:<6} "
          f"{row['Extras BURST L']}/{row['Extras Total']:<6} "
          f"{row['Extras BURST Final']}/{row['Extras Total']:<6}")

print()
print("Leyenda:")
print("  Can.Det = Canónicos Detectados/Total")
print("  Can.BI = Canónicos BURST Intensity/Total")
print("  Can.BL = Canónicos BURST Linear/Total")
print("  Can.BF = Canónicos BURST Final/Total")
print("  Ext.Det = Extras Detectados/Total")
print("  Ext.BI = Extras BURST Intensity/Total")
print("  Ext.BL = Extras BURST Linear/Total")
print("  Ext.BF = Extras BURST Final/Total")
print("="*80)

