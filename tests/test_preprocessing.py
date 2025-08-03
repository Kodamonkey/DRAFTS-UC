#!/usr/bin/env python3
"""
Test script for the refactored preprocessing module - PASO 2 del Pipeline DRAFTS
===============================================================================

Este script prueba todas las funcionalidades del módulo preprocessing:
- downsampling: Reducción de resolución de datos
- dedispersion: Aplicación de dedispersión GPU/CPU
- slice_calculator: Cálculo de tamaños óptimos
- dm_calculator: Cálculo de rangos DM

Para ejecutar: python tests/test_preprocessing.py
"""

import sys
import os
import numpy as np
import logging

# Agregar el directorio raíz al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_downsampling():
    """Prueba las funciones de downsampling."""
    logger.info("🧪 Probando módulo downsampling...")
    
    try:
        from drafts.preprocessing import downsample_data, validate_downsampling_parameters, get_downsampling_info
        
        # Crear datos de prueba
        test_data = np.random.rand(1000, 2, 512).astype(np.float32)
        logger.info(f"Datos de prueba creados: {test_data.shape}")
        
        # Probar downsampling
        downsampled_data = downsample_data(test_data)
        logger.info(f"Datos downsampled: {downsampled_data.shape}")
        
        # Probar validación de parámetros
        is_valid = validate_downsampling_parameters()
        logger.info(f"Parámetros de downsampling válidos: {is_valid}")
        
        # Probar información de downsampling
        info = get_downsampling_info(test_data.shape)
        logger.info(f"Información de downsampling: {info}")
        
        logger.info("✅ Módulo downsampling funcionando correctamente")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en downsampling: {e}")
        return False

def test_dedispersion():
    """Prueba las funciones de dedispersión."""
    logger.info("🧪 Probando módulo dedispersion...")
    
    try:
        from drafts.preprocessing import (
            d_dm_time_g, 
            dedisperse_patch, 
            dedisperse_block,
            validate_dedispersion_parameters
        )
        
        # Crear datos de prueba
        test_data = np.random.rand(1000, 256).astype(np.float32)
        freq_down = np.linspace(1000, 2000, 256)  # MHz
        
        logger.info(f"Datos de prueba creados: {test_data.shape}")
        logger.info(f"Frecuencias downsampled: {freq_down.shape}")
        
        # Probar validación de parámetros
        is_valid = validate_dedispersion_parameters(test_data, 100, 500)
        logger.info(f"Parámetros de dedispersión válidos: {is_valid}")
        
        # Probar dedispersión de patch
        patch, start = dedisperse_patch(test_data, freq_down, dm=100.0, sample=500, patch_len=256)
        logger.info(f"Patch dedispersado: {patch.shape}, start: {start}")
        
        # Probar dedispersión de bloque
        block = dedisperse_block(test_data, freq_down, dm=100.0, start=0, block_len=512)
        logger.info(f"Bloque dedispersado: {block.shape}")
        
        logger.info("✅ Módulo dedispersion funcionando correctamente")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en dedispersion: {e}")
        return False

def test_slice_calculator():
    """Prueba las funciones de cálculo de slices."""
    logger.info("🧪 Probando módulo slice_calculator...")
    
    try:
        from drafts.preprocessing import (
            calculate_slice_len_from_duration,
            calculate_optimal_chunk_size,
            get_processing_parameters,
            update_slice_len_dynamic,
            validate_processing_parameters,
            get_memory_usage_info
        )
        
        # Probar cálculo de slice_len
        slice_len, duration_ms = calculate_slice_len_from_duration()
        logger.info(f"Slice length calculado: {slice_len}, duración: {duration_ms:.1f} ms")
        
        # Probar cálculo de chunk_size
        chunk_size = calculate_optimal_chunk_size()
        logger.info(f"Chunk size calculado: {chunk_size}")
        
        # Probar parámetros de procesamiento
        params = get_processing_parameters()
        logger.info(f"Parámetros de procesamiento: {params}")
        
        # Probar validación de parámetros
        is_valid = validate_processing_parameters(params)
        logger.info(f"Parámetros válidos: {is_valid}")
        
        # Probar información de memoria
        memory_info = get_memory_usage_info()
        logger.info(f"Información de memoria: {memory_info}")
        
        # Probar actualización dinámica
        update_slice_len_dynamic()
        logger.info("Actualización dinámica completada")
        
        logger.info("✅ Módulo slice_calculator funcionando correctamente")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en slice_calculator: {e}")
        return False

def test_dm_calculator():
    """Prueba las funciones de cálculo DM."""
    logger.info("🧪 Probando módulo dm_calculator...")
    
    try:
        from drafts.preprocessing import (
            calculate_dm_range,
            validate_dm_parameters,
            get_dm_processing_config,
            optimize_dm_range_for_frb_detection
        )
        
        # Probar cálculo de rango DM
        dm_range = calculate_dm_range(freq_min=1000, freq_max=2000, time_resolution=0.001)
        logger.info(f"Rango DM calculado: {dm_range}")
        
        # Probar validación de parámetros DM
        is_valid = validate_dm_parameters(dm_min=0, dm_max=1000, dm_step=1.0)
        logger.info(f"Parámetros DM válidos: {is_valid}")
        
        # Probar configuración de procesamiento DM
        config = get_dm_processing_config()
        logger.info(f"Configuración DM: {config}")
        
        # Probar optimización para FRB
        optimized = optimize_dm_range_for_frb_detection()
        logger.info(f"Optimización FRB: {optimized}")
        
        logger.info("✅ Módulo dm_calculator funcionando correctamente")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en dm_calculator: {e}")
        return False

def test_legacy_compatibility():
    """Prueba la compatibilidad legacy."""
    logger.info("🧪 Probando compatibilidad legacy...")
    
    try:
        from drafts.preprocessing import (
            downsample_data_legacy,
            d_dm_time_g_legacy,
            dedisperse_patch_legacy,
            dedisperse_block_legacy,
            calculate_slice_len_from_duration_legacy,
            calculate_optimal_chunk_size_legacy,
            get_processing_parameters_legacy
        )
        
        # Probar funciones legacy
        test_data = np.random.rand(100, 2, 64).astype(np.float32)
        downsampled = downsample_data_legacy(test_data)
        logger.info(f"Downsampling legacy: {downsampled.shape}")
        
        slice_len, duration = calculate_slice_len_from_duration_legacy()
        logger.info(f"Slice calculation legacy: {slice_len}, {duration:.1f} ms")
        
        chunk_size = calculate_optimal_chunk_size_legacy()
        logger.info(f"Chunk size legacy: {chunk_size}")
        
        params = get_processing_parameters_legacy()
        logger.info(f"Processing parameters legacy: {params}")
        
        logger.info("✅ Compatibilidad legacy funcionando correctamente")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en compatibilidad legacy: {e}")
        return False

def test_integration_with_data_loading():
    """Prueba la integración con el módulo data_loading."""
    logger.info("🧪 Probando integración con data_loading...")
    
    try:
        # Probar que podemos importar ambos módulos sin conflictos
        from drafts.data_loading import load_fits_file_legacy, get_obparams
        from drafts.preprocessing import downsample_data, calculate_slice_len_from_duration
        
        # Simular datos cargados (sin archivo real)
        test_data = np.random.rand(2000, 2, 1024).astype(np.float32)
        logger.info(f"Datos simulados: {test_data.shape}")
        
        # Probar downsampling de datos cargados
        downsampled = downsample_data(test_data)
        logger.info(f"Datos downsampled: {downsampled.shape}")
        
        # Probar cálculo de slices para datos cargados
        slice_len, duration = calculate_slice_len_from_duration()
        logger.info(f"Slices calculados: {slice_len}, {duration:.1f} ms")
        
        logger.info("✅ Integración con data_loading funcionando correctamente")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en integración con data_loading: {e}")
        return False

def main():
    """Función principal de pruebas."""
    logger.info("🚀 Iniciando pruebas del módulo preprocessing...")
    logger.info("=" * 60)
    
    # Lista de pruebas
    tests = [
        ("Downsampling", test_downsampling),
        ("Dedispersion", test_dedispersion),
        ("Slice Calculator", test_slice_calculator),
        ("DM Calculator", test_dm_calculator),
        ("Legacy Compatibility", test_legacy_compatibility),
        ("Integration with Data Loading", test_integration_with_data_loading),
    ]
    
    # Ejecutar pruebas
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Ejecutando prueba: {test_name}")
        logger.info("-" * 40)
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                logger.info(f"✅ {test_name}: PASÓ")
            else:
                logger.error(f"❌ {test_name}: FALLÓ")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Resumen de resultados
    logger.info("\n" + "=" * 60)
    logger.info("📊 RESUMEN DE PRUEBAS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASÓ" if success else "❌ FALLÓ"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Resultado final: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        logger.info("🎉 ¡Todas las pruebas pasaron! El módulo preprocessing está funcionando correctamente.")
        return True
    else:
        logger.error(f"⚠️  {total - passed} pruebas fallaron. Revisar errores arriba.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 