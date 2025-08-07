#!/usr/bin/env python3
"""
Script de prueba para demostrar el nuevo sistema de logging detallado del chunking.
"""

import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drafts import config
from drafts.preprocessing.slice_len_calculator import get_processing_parameters
from drafts.logging.chunking_logging import display_detailed_chunking_info
from drafts.logging.logging_config import setup_logging, set_global_logger


def test_detailed_chunking_logging():
    """Prueba el sistema de logging detallado del chunking."""
    
    print("🧪 PRUEBA DEL SISTEMA DE LOGGING DETALLADO DEL CHUNKING")
    print("=" * 80)
    
    # Configurar el sistema de logging de DRAFTS
    logger = setup_logging(level="INFO", use_colors=True)
    set_global_logger(logger)
    
    # Configurar parámetros de prueba
    config.SLICE_DURATION_MS = 300.0  # 300ms por slice
    config.FILE_LENG = 65_917_985  # Muestras totales
    config.FREQ_RESO = 512  # Canales totales
    config.TIME_RESO = 0.000064  # Resolución temporal
    config.DOWN_TIME_RATE = 8  # Factor de downsampling temporal
    config.DOWN_FREQ_RATE = 1  # Factor de downsampling frecuencial
    
    print(f"📋 Configuración de prueba:")
    print(f"   • SLICE_DURATION_MS: {config.SLICE_DURATION_MS:.1f} ms")
    print(f"   • FILE_LENG: {config.FILE_LENG:,} muestras")
    print(f"   • FREQ_RESO: {config.FREQ_RESO:,} canales")
    print(f"   • TIME_RESO: {config.TIME_RESO:.2e} s")
    print(f"   • DOWN_TIME_RATE: {config.DOWN_TIME_RATE}")
    print(f"   • DOWN_FREQ_RATE: {config.DOWN_FREQ_RATE}")
    print()
    
    # Calcular parámetros de procesamiento
    processing_params = get_processing_parameters()
    
    # Mostrar información detallada del chunking
    display_detailed_chunking_info(processing_params)
    
    print()
    print("✅ Prueba completada exitosamente!")


if __name__ == "__main__":
    test_detailed_chunking_logging()
