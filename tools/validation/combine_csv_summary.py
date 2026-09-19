#!/usr/bin/env python3
"""
Script para combinar todos los archivos CSV de las subcarpetas dentro de Summary.

Uso:
    python combine_csv_summary.py <ruta_carpeta_all>
    
Ejemplo:
    python combine_csv_summary.py "D:/Seba - Dev/TESIS/DRAFTS-UC/ResultsThesis/ALMA-4phases/all"
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import List


def encontrar_csvs_en_summary(ruta_all: str) -> List[Path]:
    """
    Encuentra todos los archivos CSV dentro de las subcarpetas de Summary.
    
    Args:
        ruta_all: Ruta a la carpeta 'all'
        
    Returns:
        Lista de rutas a los archivos CSV encontrados
    """
    ruta_summary = Path(ruta_all) / "Summary"
    
    if not ruta_summary.exists():
        raise FileNotFoundError(f"No se encontró la carpeta Summary en: {ruta_all}")
    
    csv_files = []
    
    # Buscar recursivamente todos los archivos CSV en las subcarpetas
    for subcarpeta in ruta_summary.iterdir():
        if subcarpeta.is_dir():
            # Buscar archivos CSV en la subcarpeta
            csvs_en_subcarpeta = list(subcarpeta.glob("*.csv"))
            csv_files.extend(csvs_en_subcarpeta)
    
    return sorted(csv_files)


def combinar_csvs(csv_files: List[Path], archivo_salida: str) -> None:
    """
    Combina múltiples archivos CSV en uno solo.
    
    Args:
        csv_files: Lista de rutas a los archivos CSV
        archivo_salida: Ruta del archivo CSV de salida
    """
    if not csv_files:
        print("No se encontraron archivos CSV para combinar.")
        return
    
    print(f"Encontrados {len(csv_files)} archivos CSV para combinar...")
    
    dataframes = []
    
    for i, csv_file in enumerate(csv_files, 1):
        try:
            print(f"  [{i}/{len(csv_files)}] Procesando: {csv_file.name}")
            df = pd.read_csv(csv_file)
            dataframes.append(df)
        except Exception as e:
            print(f"    ERROR al leer {csv_file}: {e}")
            continue
    
    if not dataframes:
        print("No se pudieron leer archivos CSV válidos.")
        return
    
    # Combinar todos los dataframes
    print("\nCombinando dataframes...")
    df_combinado = pd.concat(dataframes, ignore_index=True)
    
    # Guardar el resultado
    print(f"Guardando archivo combinado en: {archivo_salida}")
    df_combinado.to_csv(archivo_salida, index=False)
    
    print(f"\n✓ Completado!")
    print(f"  Total de filas combinadas: {len(df_combinado)}")
    print(f"  Total de columnas: {len(df_combinado.columns)}")
    print(f"  Archivo guardado en: {archivo_salida}")


def main():
    """Función principal del script."""
    if len(sys.argv) < 2:
        print("Error: Debes proporcionar la ruta a la carpeta 'all'")
        print("\nUso:")
        print(f"  python {sys.argv[0]} <ruta_carpeta_all>")
        print("\nEjemplo:")
        print(f'  python {sys.argv[0]} "D:/Seba - Dev/TESIS/DRAFTS-UC/ResultsThesis/ALMA-4phases/all"')
        sys.exit(1)
    
    ruta_all = sys.argv[1].strip().rstrip('\\/').rstrip('"').rstrip("'")
    
    # Verificar que la ruta existe
    if not os.path.exists(ruta_all):
        print(f"Error: La ruta no existe: {ruta_all}")
        sys.exit(1)
    
    try:
        # Encontrar todos los archivos CSV
        print(f"Buscando archivos CSV en: {ruta_all}/Summary")
        csv_files = encontrar_csvs_en_summary(ruta_all)
        
        if not csv_files:
            print("No se encontraron archivos CSV en las subcarpetas de Summary.")
            sys.exit(1)
        
        # Definir el archivo de salida (en la misma carpeta 'all')
        ruta_all_path = Path(ruta_all)
        archivo_salida = ruta_all_path / "combined_candidates.csv"
        
        # Combinar los CSV
        combinar_csvs(csv_files, str(archivo_salida))
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

