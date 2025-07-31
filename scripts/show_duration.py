#!/usr/bin/env python3
"""
Script para mostrar la duración real de archivos FITS y filterbank.
Uso: python scripts/show_duration.py <archivo.fits|archivo.fil>
"""

import sys
from pathlib import Path
import numpy as np

# Agregar el directorio raíz al path para importar DRAFTS
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRAFTS.core import config
from DRAFTS.io.io import get_obparams_fits, get_obparams_fil


def show_file_duration(file_path: str) -> None:
    """Mostrar la duración de un archivo FITS o filterbank."""
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ Error: El archivo {file_path} no existe")
        return
    
    print(f"📁 Analizando archivo: {file_path.name}")
    print("=" * 60)
    
    try:
        # Cargar parámetros según el tipo de archivo
        if file_path.suffix.lower() == ".fits":
            get_obparams_fits(str(file_path))
        elif file_path.suffix.lower() == ".fil":
            get_obparams_fil(str(file_path))
        else:
            print(f"❌ Error: Formato de archivo no soportado: {file_path.suffix}")
            return
        
        # Calcular duración
        total_samples = config.FILE_LENG
        time_resolution = config.TIME_RESO
        down_time_rate = config.DOWN_TIME_RATE
        
        # Duración sin decimación
        duration_raw_sec = total_samples * time_resolution
        duration_raw_min = duration_raw_sec / 60
        duration_raw_hour = duration_raw_min / 60
        
        # Duración después de decimación
        duration_decimated_sec = total_samples * time_resolution * down_time_rate
        duration_decimated_min = duration_decimated_sec / 60
        duration_decimated_hour = duration_decimated_min / 60
        
        print("📊 INFORMACIÓN DEL ARCHIVO:")
        print(f"   📏 Muestras totales: {total_samples:,}")
        print(f"   ⏱️  Resolución temporal: {time_resolution:.6f} segundos")
        print(f"   🔽 Factor de decimación temporal: {down_time_rate}")
        print()
        
        print("⏰ DURACIÓN CALCULADA:")
        print(f"   🕐 Sin decimación:")
        print(f"      {duration_raw_sec:.3f} segundos")
        print(f"      {duration_raw_min:.2f} minutos")
        print(f"      {duration_raw_hour:.2f} horas")
        print()
        print(f"   🕐 Con decimación (usado en pipeline):")
        print(f"      {duration_decimated_sec:.3f} segundos")
        print(f"      {duration_decimated_min:.2f} minutos")
        print(f"      {duration_decimated_hour:.2f} horas")
        print()
        
        # Información adicional
        print("🔧 CONFIGURACIÓN ACTUAL:")
        print(f"   🎯 SLICE_DURATION_MS: {config.SLICE_DURATION_MS:.1f} ms")
        print(f"   📏 SLICE_LEN: {config.SLICE_LEN} muestras")
        print(f"   📊 DM_min: {config.DM_min} pc cm⁻³")
        print(f"   📊 DM_max: {config.DM_max} pc cm⁻³")
        print(f"   🔽 DOWN_FREQ_RATE: {config.DOWN_FREQ_RATE}")
        print(f"   🔽 DOWN_TIME_RATE: {config.DOWN_TIME_RATE}")
        print()
        
        # Calcular número de slices esperados
        width_total = total_samples  # Ya está decimado
        time_slice = (width_total + config.SLICE_LEN - 1) // config.SLICE_LEN
        
        print("📈 ESTIMACIÓN DE SLICES:")
        print(f"   📊 Slices esperados: {time_slice:,}")
        print(f"   📊 Slices por hora: {time_slice / duration_decimated_hour:.0f}")
        print()
        
        # Información de frecuencia
        freq_min = config.FREQ.min()
        freq_max = config.FREQ.max()
        print("📡 INFORMACIÓN DE FRECUENCIA:")
        print(f"   📡 Frecuencia mínima: {freq_min:.1f} MHz")
        print(f"   📡 Frecuencia máxima: {freq_max:.1f} MHz")
        print(f"   📡 Ancho de banda: {freq_max - freq_min:.1f} MHz")
        print(f"   📡 Canales de frecuencia: {config.FREQ_RESO}")
        print(f"   📡 Canales después de decimación: {config.FREQ_RESO // config.DOWN_FREQ_RATE}")
        
    except Exception as e:
        print(f"❌ Error al analizar el archivo: {e}")
        return


def main():
    """Función principal."""
    if len(sys.argv) != 2:
        print("Uso: python scripts/show_duration.py <archivo.fits|archivo.fil>")
        print()
        print("Ejemplos:")
        print("  python scripts/show_duration.py data/observacion.fits")
        print("  python scripts/show_duration.py data/observacion.fil")
        return
    
    file_path = sys.argv[1]
    show_file_duration(file_path)


if __name__ == "__main__":
    main() 
