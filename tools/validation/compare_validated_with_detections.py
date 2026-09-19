#!/usr/bin/env python3
"""
Script para comparar candidatos validados manualmente con detecciones automáticas.

Compara tabla_transformada.csv con múltiples archivos combined_candidates-*.csv
y genera archivos Excel con todos los matches encontrados.
"""

import re
import pandas as pd
from pathlib import Path

# Código compartido con analyze_alma_phases_validation.py (extraído en REF-14).
# El directorio del script ya está en sys.path cuando se ejecuta como
# `python src/scripts/compare_validated_with_detections.py`.
from _matching_common import (
    canonical_alma_key,
    find_matches,
    load_detected_candidates,
    save_to_excel,
)


def load_validated_candidates(csv_path: str) -> pd.DataFrame:
    """
    Carga y normaliza los candidatos validados manualmente.
    
    Args:
        csv_path: Ruta al archivo tabla_transformada.csv
        
    Returns:
        DataFrame con candidatos validados normalizados
    """
    print(f"Cargando candidatos validados desde: {csv_path}")
    df = pd.read_csv(csv_path, sep='\t', encoding='utf-8')
    
    # Normalizar nombres de columnas
    df.columns = df.columns.str.strip()
    
    # Normalizar campo 'nombre_archivo' para matching consistente
    if 'nombre_archivo' in df.columns:
        df['nombre_archivo_normalized'] = df['nombre_archivo'].apply(canonical_alma_key)
    else:
        print("⚠ Advertencia: No se encontró la columna 'nombre_archivo'")
        df['nombre_archivo_normalized'] = ''
    
    # Normalizar campo 'candidato tiempo' - remover texto entre paréntesis
    if 'candidato tiempo' in df.columns:
        df['candidato_tiempo_clean'] = df['candidato tiempo'].astype(str).apply(
            lambda x: re.sub(r'\s*\([^)]*\)', '', x).strip()
        )
        # Convertir a numérico
        df['candidato_tiempo_clean'] = pd.to_numeric(
            df['candidato_tiempo_clean'], 
            errors='coerce'
        )
    else:
        print("⚠ Advertencia: No se encontró la columna 'candidato tiempo'")
        df['candidato_tiempo_clean'] = None
    
    print(f"  [OK] Cargados {len(df)} candidatos validados")
    return df


def main():
    """Función principal del script."""
    base_path = Path("ResultsThesis")
    
    # Archivo de candidatos validados
    validated_path = base_path / "tabla_transformada.csv"
    
    # Archivos de detecciones a comparar
    detection_files = [
        ("combined_candidates-all.csv", "matches_all.xlsx"),
        ("combined_candidates-no-class-intensity.csv", "matches_no-class-intensity.xlsx"),
        ("combined_candidates-no-classification-linear.csv", "matches_no-classification-linear.xlsx"),
        ("combined_candidates-no-phase2.csv", "matches_no-phase2.xlsx"),
        ("combined_candidates-no-phase2-no-classification-intensity.csv", 
         "matches_no-phase2-no-classification-intensity.xlsx"),
        ("combined_candidates-no-phase2-no-classification-linear.csv", 
         "matches_no-phase2-no-classification-linear.xlsx"),
    ]
    
    # Cargar candidatos validados
    validated_df = load_validated_candidates(str(validated_path))
    
    # Procesar cada archivo de detecciones
    for detection_file, output_file in detection_files:
        detection_path = base_path / detection_file
        
        if not detection_path.exists():
            print(f"\n⚠ Advertencia: No se encontró {detection_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Procesando: {detection_file}")
        print(f"{'='*60}")
        
        # Cargar candidatos detectados
        detected_df = load_detected_candidates(str(detection_path))
        
        # Encontrar matches
        matches_df = find_matches(validated_df, detected_df, detection_file)
        
        # Guardar en Excel
        output_path = base_path / output_file
        try:
            save_to_excel(matches_df, str(output_path))
        except PermissionError:
            print(f"  ⚠ Saltando {output_file} - archivo está abierto. Por favor ciérralo y vuelve a ejecutar.")
            continue
    
    print(f"\n{'='*60}")
    print("[OK] Proceso completado!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
