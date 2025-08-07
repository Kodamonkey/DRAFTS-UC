#!/usr/bin/env python3
"""
Script para obtener el tiempo exacto de duración de archivos FITS y filterbank.
Uso: python scripts/get_exact_duration.py <archivo.fits|archivo.fil>
"""

import sys
from pathlib import Path
import numpy as np

# Agregar el directorio raíz al path para importar DRAFTS
sys.path.insert(0, str(Path(__file__).parent.parent))

from drafts import config
from drafts.input.data_loader import get_obparams, get_obparams_fil


def get_exact_duration(file_path: str) -> dict:
    """
    Obtener el tiempo exacto de duración de un archivo.
    
    Args:
        file_path: Ruta al archivo (.fits o .fil)
    
    Returns:
        dict: Diccionario con la información de duración
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"El archivo {file_path} no existe")
    
    # Cargar parámetros según el tipo de archivo
    if file_path.suffix.lower() == ".fits":
        get_obparams(str(file_path))
    elif file_path.suffix.lower() == ".fil":
        get_obparams_fil(str(file_path))
    else:
        raise ValueError(f"Formato de archivo no soportado: {file_path.suffix}")
    
    # Calcular duración exacta
    total_samples = config.FILE_LENG
    time_resolution = config.TIME_RESO
    down_time_rate = config.DOWN_TIME_RATE
    
    # Duración sin decimación (tiempo real del archivo)
    duration_raw_sec = total_samples * time_resolution
    duration_raw_min = duration_raw_sec / 60
    duration_raw_hour = duration_raw_min / 60
    
    # Duración después de decimación (tiempo usado en pipeline)
    duration_decimated_sec = total_samples * time_resolution * down_time_rate
    duration_decimated_min = duration_decimated_sec / 60
    duration_decimated_hour = duration_decimated_min / 60
    
    return {
        'file_name': file_path.name,
        'file_type': file_path.suffix.lower(),
        'total_samples': total_samples,
        'time_resolution_sec': time_resolution,
        'down_time_rate': down_time_rate,
        'duration_raw_sec': duration_raw_sec,
        'duration_raw_min': duration_raw_min,
        'duration_raw_hour': duration_raw_hour,
        'duration_decimated_sec': duration_decimated_sec,
        'duration_decimated_min': duration_decimated_min,
        'duration_decimated_hour': duration_decimated_hour
    }


def main():
    """Función principal."""
    if len(sys.argv) != 2:
        print("Uso: python scripts/get_exact_duration.py <archivo.fits|archivo.fil>")
        print()
        print("Ejemplos:")
        print("  python scripts/get_exact_duration.py Data/2017-04-03-08-16-13_142_0003_t39.977.fits")
        print("  python scripts/get_exact_duration.py Data/3100_0001_00_8bit.fil")
        return
    
    file_path = sys.argv[1]
    
    try:
        # Obtener duración exacta
        duration_info = get_exact_duration(file_path)
        
        # Mostrar solo el tiempo exacto como solicitó el usuario
        print(f"Archivo: {duration_info['file_name']}")
        print(f"TIEMPO EXACTO: {duration_info['duration_raw_sec']:.3f} segundos")
        print(f"TIEMPO EXACTO en minutos y horas: ({duration_info['duration_raw_min']:.2f} minutos, {duration_info['duration_raw_hour']:.2f} horas)")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
