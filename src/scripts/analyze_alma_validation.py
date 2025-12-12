#!/usr/bin/env python3
"""
Script para analizar los resultados de validación ALMA de los 6 casos metodológicos.
Genera estadísticas detalladas para los 8 pulsos canónicos, dataset extendido y nuevos pulsos.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

# Cargar canónicos desde config.yaml
def load_canonical_pulses_from_config():
    """Carga los 8 pulsos canónicos desde config.yaml."""
    import yaml
    
    canonical_pulses = []
    
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Buscar las líneas 47-52 que contienen los canónicos
        in_canonical_section = False
        for i, line in enumerate(lines):
            if i >= 46 and i <= 51:  # Líneas 47-52 (0-indexed: 46-51)
                stripped = line.strip()
                if stripped.startswith('- "') and stripped.endswith('"'):
                    # Parsear el formato: "2017-04-03-08_16_13_142_0006_t10.882_t25.829"
                    canonical_str = stripped[3:-1]  # Remover '- "' y '"'
                    
                    # Extraer fecha, subfolder y tiempos
                    # Patrón: fecha_subfolder_tiempo1_tiempo2...
                    # Ejemplo: "2017-04-03-08_16_13_142_0006_t10.882_t25.829"
                    
                    # Buscar el patrón de fecha (puede tener guiones o guiones bajos)
                    # Formato: "2017-04-03-08_16_13_142_0006_t10.882_t25.829"
                    # Fecha: "2017-04-03-08_16_13" (puede tener guiones o guiones bajos)
                    date_match = re.search(r'(\d{4}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2})', canonical_str)
                    if not date_match:
                        continue
                    
                    date_part = date_match.group(1)
                    remaining = canonical_str[len(date_part):]
                    
                    # Extraer subfolder (4 dígitos después del identificador y antes de _t)
                    # remaining sería: "_142_0006_t10.882_t25.829"
                    subfolder_match = re.search(r'_\d+_(\d{4})_t', remaining)
                    if not subfolder_match:
                        continue
                    
                    subfolder = subfolder_match.group(1)
                    
                    # Extraer todos los tiempos (patrón _tNUMBER)
                    times = re.findall(r'_t([\d.]+)', canonical_str)
                    
                    if not times:
                        continue
                    
                    # Normalizar fecha: convertir guiones a guiones bajos
                    date_normalized = date_part.replace('-', '_')
                    
                    # Crear entrada para cada tiempo
                    for time_str in times:
                        try:
                            time_val = float(time_str)
                            
                            # Ajuste especial para 242_0005: tiempo en config.yaml es 44.169 pero en tabla es 44.919
                            # Usar el tiempo de tabla_transformada.csv si está disponible
                            if subfolder == "0005" and date_normalized.endswith("13_38_31"):
                                # Verificar si hay un tiempo cercano en tabla_transformada.csv
                                # Por ahora, usar tolerancia más amplia o ajustar el tiempo
                                # El tiempo real en tabla es 44.919, así que ajustamos
                                if abs(time_val - 44.919) < 1.0:  # Si está cerca de 44.919
                                    time_val = 44.919  # Usar el tiempo de tabla_transformada.csv
                            
                            # Construir nombre de archivo esperado en tabla_transformada.csv
                            # Formato: fecha_subfolder (sin el identificador numérico)
                            # Ejemplo: "2017-04-03-08_16_13_0006" (sin el "142_")
                            expected_file = f"{date_normalized}_{subfolder}"
                            expected_file_alt = expected_file  # Mismo formato
                            
                            canonical_pulses.append({
                                "file_pattern": date_normalized,  # Patrón de fecha
                                "subfolder": subfolder,
                                "subfolder_int": str(int(subfolder)),  # Sin ceros iniciales
                                "time": time_val,
                                "tolerance": 0.1,
                                "name": f"{date_normalized.split('_')[-3]}_{subfolder}_{time_str}",
                                "expected_file": expected_file,
                                "expected_file_alt": expected_file_alt
                            })
                        except ValueError:
                            continue
    except Exception as e:
        print(f"Error cargando canónicos desde config.yaml: {e}")
        # Fallback a definición manual
        canonical_pulses = [
            {"file_pattern": "2017-04-03-08_16_13", "subfolder": "0006", "subfolder_int": "6", "time": 10.882, "tolerance": 0.1, "name": "142_0006_1", "expected_file": "2017-04-03-08_16_13_0006"},
            {"file_pattern": "2017-04-03-08_16_13", "subfolder": "0006", "subfolder_int": "6", "time": 25.829, "tolerance": 0.1, "name": "142_0006_2", "expected_file": "2017-04-03-08_16_13_0006"},
            {"file_pattern": "2017-04-03-08_55_22", "subfolder": "0006", "subfolder_int": "6", "time": 23.444, "tolerance": 0.1, "name": "153_0006", "expected_file": "2017-04-03-08_55_22_0006"},
            {"file_pattern": "2017-04-03-08_16_13", "subfolder": "0003", "subfolder_int": "3", "time": 39.977, "tolerance": 0.1, "name": "142_0003", "expected_file": "2017-04-03-08_16_13_0003"},
            {"file_pattern": "2017-04-03-12_56_05", "subfolder": "0002", "subfolder_int": "2", "time": 2.3, "tolerance": 0.1, "name": "230_0002_1", "expected_file": "2017-04-03-12_56_05_0002"},
            {"file_pattern": "2017-04-03-12_56_05", "subfolder": "0002", "subfolder_int": "2", "time": 17.395, "tolerance": 0.1, "name": "230_0002_2", "expected_file": "2017-04-03-12_56_05_0002"},
            {"file_pattern": "2017-04-03-12_56_05", "subfolder": "0003", "subfolder_int": "3", "time": 36.548, "tolerance": 0.1, "name": "230_0003", "expected_file": "2017-04-03-12_56_05_0003"},
            {"file_pattern": "2017-04-03-13_38_31", "subfolder": "0005", "subfolder_int": "5", "time": 44.169, "tolerance": 0.1, "name": "242_0005", "expected_file": "2017-04-03-13_38_31_0005"},
        ]
    
    return canonical_pulses

# Mapeo de archivos a casos
CASE_MAPPING = {
    "matches_all.xlsx": "a",
    "matches_no-class-intensity.xlsx": "b",
    "matches_no-classification-linear.xlsx": "e",
    "matches_no-phase2-no-classification-intensity.xlsx": "d",
    "matches_no-phase2-no-classification-linear.xlsx": "c",
    "matches_no-phase2.xlsx": "f",
}

# Cargar tabla validada para identificar canónicos
def load_validated_table():
    """Carga la tabla de candidatos validados."""
    df = pd.read_csv("ResultsThesis/tabla_transformada.csv", sep="\t")
    
    # Limpiar tiempo de candidato
    df['candidato_tiempo_clean'] = df['candidato tiempo'].astype(str).apply(
        lambda x: re.sub(r'\s*\([^)]*\)', '', x).strip() if pd.notna(x) else ""
    )
    df['candidato_tiempo_clean'] = pd.to_numeric(df['candidato_tiempo_clean'], errors='coerce')
    
    return df

def identify_canonical_pulses(df_validated):
    """Identifica los índices de los 8 pulsos canónicos en la tabla validada."""
    canonical_pulses = load_canonical_pulses_from_config()
    canonical_indices = []
    canonical_info = []
    
    for pulse in canonical_pulses:
        # Buscar por patrón de archivo y subfolder
        file_pattern = pulse['file_pattern']
        subfolder = pulse['subfolder']
        subfolder_int = pulse['subfolder_int']
        
        # El nombre_archivo en tabla_transformada.csv tiene formato: fecha-subfolder
        # Ejemplo: "2017-04-03-08_16_13_0003" (con guiones en la fecha, guiones bajos al final)
        # El file_pattern es: "2017_04_03_08_16_13" (todo con guiones bajos)
        # Necesitamos buscar que contenga el patrón de fecha (normalizando ambos)
        
        # Crear variantes del patrón (con guiones y con guiones bajos)
        file_pattern_with_dashes = file_pattern.replace('_', '-', 3)  # Solo los primeros 3 guiones bajos
        file_pattern_normalized = file_pattern  # Ya está con guiones bajos
        
        # Buscar archivos que contengan cualquiera de los patrones
        file_match_dashes = df_validated['nombre_archivo'].astype(str).str.contains(
            file_pattern_with_dashes, 
            case=False, 
            na=False, 
            regex=False
        )
        file_match_underscores = df_validated['nombre_archivo'].astype(str).str.contains(
            file_pattern_normalized, 
            case=False, 
            na=False, 
            regex=False
        )
        file_match = file_match_dashes | file_match_underscores
        
        # También verificar que termine con el subfolder (o que el subfolder coincida)
        # El nombre_archivo termina con _subfolder (ej: _0003)
        file_ends_with_subfolder = df_validated['nombre_archivo'].astype(str).str.endswith(
            f"_{subfolder}", na=False
        ) | df_validated['nombre_archivo'].astype(str).str.endswith(
            f"_{subfolder_int}", na=False
        )
        file_match = file_match & file_ends_with_subfolder
        
        # También verificar el subfolder en la columna subfolder
        subfolder_match = (
            (df_validated['subfolder'].astype(str) == subfolder) |
            (df_validated['subfolder'].astype(str) == subfolder_int)
        )
        
        # Buscar tiempo
        time_match = (
            (df_validated['candidato_tiempo_clean'] >= pulse['time'] - pulse['tolerance']) &
            (df_validated['candidato_tiempo_clean'] <= pulse['time'] + pulse['tolerance'])
        )
        
        matches = df_validated[file_match & subfolder_match & time_match]
        
        if len(matches) > 0:
            idx = matches.index[0]
            canonical_indices.append(idx)
            canonical_info.append({
                'index': idx,
                'name': pulse['name'],
                'file_pattern': pulse['file_pattern'],
                'subfolder': subfolder,
                'time': pulse['time'],
                'validated_time': matches.iloc[0]['candidato_tiempo_clean'],
                'validated_file': matches.iloc[0]['nombre_archivo']
            })
        else:
            # Buscar más flexible
            for idx, row in df_validated.iterrows():
                file_str = str(row['nombre_archivo']).lower()
                time_val = row['candidato_tiempo_clean']
                
                if pd.isna(time_val):
                    continue
                    
                # Verificar si el archivo contiene el patrón y el subfolder coincide
                file_pattern_normalized = pulse['file_pattern'].replace('-', '_')
                if file_pattern_normalized in file_str:
                    subfolder_val = str(row.get('subfolder', ''))
                    if subfolder_val == subfolder or subfolder_val == subfolder_int:
                        if abs(time_val - pulse['time']) <= pulse['tolerance']:
                            canonical_indices.append(idx)
                            canonical_info.append({
                                'index': idx,
                                'name': pulse['name'],
                                'file_pattern': pulse['file_pattern'],
                                'subfolder': subfolder,
                                'time': pulse['time'],
                                'validated_time': time_val,
                                'validated_file': row['nombre_archivo']
                            })
                            break
    
    return canonical_indices, canonical_info

def clean_column_name(col):
    """Limpia nombres de columnas duplicadas."""
    if col.endswith('.1'):
        return col[:-2]
    return col

def normalize_filename(filename):
    """Normaliza nombres de archivo para comparación."""
    if pd.isna(filename):
        return ""
    filename = str(filename).lower().strip()
    # Remover extensión .fits
    filename = filename.replace('.fits', '')
    # Normalizar guiones y guiones bajos
    filename = filename.replace('_', '-')
    # Remover espacios extra
    filename = filename.replace(' ', '')
    return filename

def files_match(file1, file2):
    """Verifica si dos nombres de archivo coinciden después de normalización."""
    norm1 = normalize_filename(file1)
    norm2 = normalize_filename(file2)
    
    # Comparación exacta
    if norm1 == norm2:
        return True
    
    # Comparación por partes (para manejar variaciones)
    parts1 = set([p for p in norm1.split('-') if len(p) > 2])
    parts2 = set([p for p in norm2.split('-') if len(p) > 2])
    
    # Si tienen suficientes partes en común
    if len(parts1) > 0 and len(parts2) > 0:
        common = parts1.intersection(parts2)
        if len(common) >= min(3, len(parts1), len(parts2)):
            return True
    
    return False

def analyze_case(matches_file, df_validated, canonical_indices, canonical_info):
    """Analiza un caso específico."""
    print(f"\n{'='*70}")
    print(f"Analizando: {matches_file} -> Caso {CASE_MAPPING[matches_file]}")
    print(f"{'='*70}")
    
    # Cargar archivo matches
    df = pd.read_excel(f"ResultsThesis/{matches_file}")
    
    # Limpiar columnas duplicadas - tomar solo la primera ocurrencia
    seen = {}
    new_cols = []
    for col in df.columns:
        base_col = col.split('.')[0]
        if base_col not in seen:
            seen[base_col] = col
            new_cols.append(base_col)
        else:
            new_cols.append(seen[base_col])
    df.columns = new_cols
    
    # Filtrar solo candidatos validados (no nuevos)
    df_validated_only = df[df['match_type'] == 'validated'].copy()
    
    # Identificar canónicos usando la información de canonical_info (parseada desde config.yaml)
    # Buscar en df_validated_only usando file_pattern, subfolder y tiempo
    canonical_pulses = load_canonical_pulses_from_config()
    
    canonicos_mask = pd.Series([False] * len(df_validated_only), index=df_validated_only.index)
    
    file_col = df_validated_only['validated_nombre_archivo']
    if isinstance(file_col, pd.DataFrame):
        file_col = file_col.iloc[:, 0]
    file_str = file_col.astype(str)
    
    time_col = df_validated_only['validated_candidato_tiempo']
    if isinstance(time_col, pd.DataFrame):
        time_col = time_col.iloc[:, 0]
    time_str = time_col.astype(str)
    time_numeric = pd.to_numeric(time_str.str.replace(r'\s*\([^)]*\)', '', regex=True), errors='coerce')
    
    subfolder_col = df_validated_only.get('validated_subfolder', None)
    if subfolder_col is not None:
        if isinstance(subfolder_col, pd.DataFrame):
            subfolder_col = subfolder_col.iloc[:, 0]
        subfolder_str = subfolder_col.astype(str)
    else:
        subfolder_str = pd.Series([''] * len(df_validated_only))
    
    # Para cada canónico, buscar matches
    for pulse in canonical_pulses:
        file_pattern = pulse['file_pattern'].replace('-', '_')
        subfolder = pulse['subfolder']
        subfolder_int = pulse['subfolder_int']
        time_val = pulse['time']
        tolerance = pulse['tolerance']
        
        # Buscar archivos que contengan el patrón
        file_match = file_str.str.contains(file_pattern, case=False, na=False, regex=False)
        
        # Buscar subfolder
        subfolder_match = (subfolder_str == subfolder) | (subfolder_str == subfolder_int)
        
        # Buscar tiempo
        time_match = (
            (time_numeric >= time_val - tolerance) &
            (time_numeric <= time_val + tolerance)
        )
        
        # Combinar
        pulse_mask = file_match & subfolder_match & time_match
        canonicos_mask = canonicos_mask | pulse_mask
    
    df_canonicos = df_validated_only[canonicos_mask]
    
    # Contar canónicos únicos (puede haber múltiples filas por canónico si hay múltiples matches)
    # Crear clave única para cada canónico: nombre_archivo + tiempo (limpiado)
    if len(df_canonicos) > 0:
        file_col_can = df_canonicos['validated_nombre_archivo']
        if isinstance(file_col_can, pd.DataFrame):
            file_col_can = file_col_can.iloc[:, 0]
        time_col_can = df_canonicos['validated_candidato_tiempo']
        if isinstance(time_col_can, pd.DataFrame):
            time_col_can = time_col_can.iloc[:, 0]
        
        # Limpiar tiempo y crear clave única
        time_clean = pd.to_numeric(time_col_can.astype(str).str.replace(r'\s*\([^)]*\)', '', regex=True), errors='coerce')
        canonico_keys = file_col_can.astype(str) + '_' + time_clean.astype(str)
        
        # Obtener canónicos únicos
        unique_keys = canonico_keys.unique()
        # total_canonicos debe ser siempre 8 (total de canónicos definidos)
        # No el número encontrado en matches, sino el total definido
        total_canonicos = len(canonical_info)  # Total de canónicos definidos (8)
        
        # Para cada canónico único, verificar si tiene match
        canonicos_detectados = 0
        canonicos_burst_i = 0
        canonicos_burst_l = 0
        canonicos_burst_final = 0
        
        for key in unique_keys:
            canonico_rows = df_canonicos[canonico_keys == key]
            # Si al menos una fila tiene match, el canónico está detectado
            if canonico_rows['has_match'].any():
                canonicos_detectados += 1
                # Para clasificaciones, verificar si alguna fila tiene burst
                if 'detected_is_burst_intensity' in canonico_rows.columns:
                    if (canonico_rows['detected_is_burst_intensity'] == 'burst').any():
                        canonicos_burst_i += 1
                if 'detected_is_burst_linear' in canonico_rows.columns:
                    if (canonico_rows['detected_is_burst_linear'] == 'burst').any():
                        canonicos_burst_l += 1
                if 'detected_is_burst' in canonico_rows.columns:
                    if (canonico_rows['detected_is_burst'] == 'burst').any():
                        canonicos_burst_final += 1
    else:
        # Si no se encontraron canónicos en matches, total_canonicos sigue siendo 8 (total definido)
        total_canonicos = len(canonical_info)  # Total de canónicos definidos (8)
        canonicos_detectados = 0
        canonicos_burst_i = 0
        canonicos_burst_l = 0
        canonicos_burst_final = 0
    
    # Dataset extendido (resto de validated sin canónicos)
    df_dataset_extendido = df_validated_only[~canonicos_mask]
    
    # Identificar canónicos en el dataframe matches (para detalles individuales)
    canonical_results = []
    for canon in canonical_info:
        # Buscar en matches
        file_col = 'validated_nombre_archivo'
        time_col = 'validated_candidato_tiempo'
        
        if file_col not in df_validated_only.columns or time_col not in df_validated_only.columns:
            continue
            
        # Obtener Series (asegurarse de que no sea DataFrame)
        if file_col not in df_validated_only.columns:
            matches = pd.DataFrame()
        else:
            file_series = df_validated_only[file_col]
            # Si es DataFrame (columnas duplicadas), tomar la primera columna
            if isinstance(file_series, pd.DataFrame):
                file_series = file_series.iloc[:, 0]
            
            time_series = df_validated_only[time_col]
            if isinstance(time_series, pd.DataFrame):
                time_series = time_series.iloc[:, 0]
            
            # Buscar usando file_pattern y subfolder
            file_str = file_series.astype(str)
            file_pattern = canon.get('file_pattern', '')
            if not file_pattern:
                # Fallback: usar validated_file si está disponible
                file_pattern = str(canon.get('validated_file', '')).split('_')[:-1]
                file_pattern = '_'.join(file_pattern) if file_pattern else ''
            
            file_pattern_normalized = file_pattern.replace('-', '_')
            file_match = file_str.str.contains(file_pattern_normalized, case=False, na=False, regex=False)
            
            # También verificar subfolder si está disponible
            subfolder_col = df_validated_only.get('validated_subfolder', None)
            if subfolder_col is not None:
                if isinstance(subfolder_col, pd.DataFrame):
                    subfolder_col = subfolder_col.iloc[:, 0]
                subfolder_str = subfolder_col.astype(str)
                subfolder = canon.get('subfolder', '')
                subfolder_int = str(int(subfolder)) if subfolder.isdigit() else subfolder
                subfolder_match = (subfolder_str == subfolder) | (subfolder_str == subfolder_int)
                file_match = file_match & subfolder_match
            
            # Convertir tiempo a numérico
            time_numeric = pd.to_numeric(time_series, errors='coerce')
            # Usar tolerancia del pulso canónico original
            tolerance = 0.1  # Tolerancia por defecto
            canonical_pulses_list = load_canonical_pulses_from_config()
            for orig_pulse in canonical_pulses_list:
                if orig_pulse['name'] == canon['name']:
                    tolerance = orig_pulse['tolerance']
                    break
            
            time_match = (
                (time_numeric >= canon['time'] - tolerance) &
                (time_numeric <= canon['time'] + tolerance)
            )
            
            matches = df_validated_only[file_match & time_match]
        
        if len(matches) > 0:
            # Tomar el primer match (puede haber múltiples)
            match_row = matches.iloc[0]
            
            has_match = match_row.get('has_match', False)
            is_burst_intensity = match_row.get('detected_is_burst_intensity', '') == 'burst'
            is_burst_linear = match_row.get('detected_is_burst_linear', '') == 'burst'
            is_burst = match_row.get('detected_is_burst', '') == 'burst'
            
            canonical_results.append({
                'name': canon['name'],
                'file': canon.get('expected_file', canon.get('file_pattern', '')),
                'time': canon['time'],
                'has_match': has_match,
                'is_burst_intensity': is_burst_intensity,
                'is_burst_linear': is_burst_linear,
                'is_burst': is_burst,
                'detection_prob': match_row.get('detected_detection_prob', np.nan),
                'class_prob_intensity': match_row.get('detected_class_prob_intensity', np.nan),
                'class_prob_linear': match_row.get('detected_class_prob_linear', np.nan),
                'snr_waterfall': match_row.get('detected_snr_waterfall', np.nan),
                'snr_patch_dedispersed': match_row.get('detected_snr_patch_dedispersed', np.nan),
            })
        else:
            # No encontrado
            canonical_results.append({
                'name': canon['name'],
                'file': canon.get('expected_file', canon.get('file_pattern', '')),
                'time': canon['time'],
                'has_match': False,
                'is_burst_intensity': False,
                'is_burst_linear': False,
                'is_burst': False,
                'detection_prob': np.nan,
                'class_prob_intensity': np.nan,
                'class_prob_linear': np.nan,
                'snr_waterfall': np.nan,
                'snr_patch_dedispersed': np.nan,
            })
    
    # Analizar dataset extendido (todos los validados excepto canónicos)
    total_validated = len(df_dataset_extendido)
    detected = df_dataset_extendido['has_match'].sum()
    
    # Clasificaciones para dataset extendido
    burst_intensity = (df_dataset_extendido['detected_is_burst_intensity'] == 'burst').sum() if 'detected_is_burst_intensity' in df_dataset_extendido.columns else 0
    burst_linear = (df_dataset_extendido['detected_is_burst_linear'] == 'burst').sum() if 'detected_is_burst_linear' in df_dataset_extendido.columns else 0
    burst_final = (df_dataset_extendido['detected_is_burst'] == 'burst').sum() if 'detected_is_burst' in df_dataset_extendido.columns else 0
    
    # Nuevos pulsos (solo BURST)
    df_new = df[df['match_type'] == 'new'].copy()
    new_burst = df_new[df_new['detected_is_burst'] == 'burst'].copy() if 'detected_is_burst' in df_new.columns else pd.DataFrame()
    
    # Usar estadísticas calculadas directamente de los dataframes
    canon_detected = canonicos_detectados
    canon_burst_intensity = canonicos_burst_i
    canon_burst_linear = canonicos_burst_l
    canon_burst_final = canonicos_burst_final
    
    return {
        'case': CASE_MAPPING[matches_file],
        'file': matches_file,
        'canonical_results': canonical_results,
        'canonical_stats': {
            'total': total_canonicos,  # Total de canónicos encontrados
            'detected': canon_detected,
            'burst_intensity': canon_burst_intensity,
            'burst_linear': canon_burst_linear,
            'burst_final': canon_burst_final,
        },
        'dataset_stats': {
            'total_validated': total_validated,
            'detected': detected,
            'burst_intensity': burst_intensity,
            'burst_linear': burst_linear,
            'burst_final': burst_final,
        },
        'new_pulses': {
            'total_new': len(df_new),
            'new_burst': len(new_burst),
            'new_burst_list': new_burst[[
                'detected_file', 'detected_t_sec_dm_time', 'detected_dm_pc_cm-3',
                'detected_detection_prob', 'detected_class_prob_intensity', 
                'detected_class_prob_linear', 'detected_snr_waterfall',
                'detected_is_burst'
            ]].to_dict('records') if len(new_burst) > 0 else []
        }
    }

def main():
    """Función principal."""
    print("="*70)
    print("ANÁLISIS DE VALIDACIÓN ALMA - 6 CASOS METODOLÓGICOS")
    print("="*70)
    
    # Cargar tabla validada
    df_validated = load_validated_table()
    print(f"\nCargados {len(df_validated)} candidatos validados")
    
    # Identificar canónicos
    canonical_indices, canonical_info = identify_canonical_pulses(df_validated)
    print(f"\nIdentificados {len(canonical_info)} pulsos canónicos:")
    for canon in canonical_info:
        print(f"  - {canon['name']}: {canon['validated_file']} @ {canon['validated_time']:.3f}s")
    
    # Analizar cada caso
    all_results = {}
    for matches_file in CASE_MAPPING.keys():
        try:
            results = analyze_case(matches_file, df_validated, canonical_indices, canonical_info)
            all_results[results['case']] = results
        except Exception as e:
            print(f"\nERROR analizando {matches_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generar resumen comparativo
    print(f"\n\n{'='*70}")
    print("RESUMEN COMPARATIVO")
    print(f"{'='*70}")
    
    print("\n8 Pulsos Canónicos:")
    print(f"{'Caso':<6} {'Detectados':<12} {'BURST I':<10} {'BURST L':<10} {'BURST Final':<12}")
    print("-" * 60)
    for case in ['a', 'b', 'c', 'd', 'e', 'f']:
        if case in all_results:
            stats = all_results[case]['canonical_stats']
            print(f"{case:<6} {stats['detected']}/{stats['total']:<11} "
                  f"{stats['burst_intensity']}/{stats['total']:<9} "
                  f"{stats['burst_linear']}/{stats['total']:<9} "
                  f"{stats['burst_final']}/{stats['total']:<11}")
    
    print("\nDataset Extendido:")
    print(f"{'Caso':<6} {'Validados':<12} {'Detectados':<12} {'BURST I':<12} {'BURST L':<12} {'BURST Final':<12}")
    print("-" * 70)
    for case in ['a', 'b', 'c', 'd', 'e', 'f']:
        if case in all_results:
            stats = all_results[case]['dataset_stats']
            print(f"{case:<6} {stats['total_validated']:<12} {stats['detected']:<12} "
                  f"{stats['burst_intensity']:<12} {stats['burst_linear']:<12} "
                  f"{stats['burst_final']:<12}")
    
    print("\nNuevos Pulsos Detectados (solo BURST):")
    print(f"{'Caso':<6} {'Total Nuevos':<15} {'Nuevos BURST':<15}")
    print("-" * 40)
    for case in ['a', 'b', 'c', 'd', 'e', 'f']:
        if case in all_results:
            new = all_results[case]['new_pulses']
            print(f"{case:<6} {new['total_new']:<15} {new['new_burst']:<15}")
    
    # Guardar resultados en JSON para uso posterior
    import json
    
    def convert_to_serializable(obj):
        """Convierte objetos numpy a tipos nativos de Python."""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    output = {}
    for case, results in all_results.items():
        output[case] = {
            'canonical_stats': convert_to_serializable(results['canonical_stats']),
            'dataset_stats': convert_to_serializable(results['dataset_stats']),
            'new_pulses': {
                'total_new': int(results['new_pulses']['total_new']),
                'new_burst': int(results['new_pulses']['new_burst']),
            }
        }
    
    with open('ResultsThesis/alma_validation_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\nResultados guardados en: ResultsThesis/alma_validation_analysis.json")
    
    return all_results

if __name__ == "__main__":
    results = main()

