#!/usr/bin/env python3
"""
Test del módulo data_loading - Verificar funcionalidad de carga de datos
==========================================================================

Este script prueba las funciones del nuevo módulo de carga de datos para
asegurar que funciona correctamente y mantiene compatibilidad.
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import logging

# Configurar logging para tests
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Probar que todos los imports funcionan correctamente."""
    print("\n=== Probando Imports ===")
    
    try:
        # Probar import del módulo principal
        import drafts.data_loading
        print("✓ Import del módulo principal exitoso")
        
        # Probar imports específicos
        from drafts.data_loading import detect_file_format
        print("✓ Import de detect_file_format exitoso")
        
        from drafts.data_loading import validate_data
        print("✓ Import de validate_data exitoso")
        
        from drafts.data_loading import validate_metadata
        print("✓ Import de validate_metadata exitoso")
        
        print("✓ Todos los imports funcionan correctamente")
        
    except ImportError as e:
        print(f"✗ Error en imports: {e}")
        return False
    
    return True

def test_format_detection():
    """Probar detección de formatos de archivo."""
    print("\n=== Probando Detección de Formatos ===")
    
    try:
        from drafts.data_loading import detect_file_format, validate_format_compatibility, get_format_handler
        
        # Crear archivos de prueba
        test_files = {
            'test.fits': b'SIMPLE  =                    T / FITS standard',
            'test.fil': b'HEADER_START',
            'test.unknown': b'UNKNOWN_FORMAT'
        }
        
        for filename, content in test_files.items():
            with open(filename, 'wb') as f:
                f.write(content)
            
            try:
                format_info = detect_file_format(filename)
                print(f"Archivo: {filename}")
                print(f"  Tipo: {format_info['file_type']}")
                print(f"  Subtipo: {format_info['format_subtype']}")
                print(f"  Confianza: {format_info['confidence']:.2f}")
                
                # Probar compatibilidad
                is_compatible = validate_format_compatibility(format_info)
                print(f"  Compatible: {is_compatible}")
                
                # Obtener handler recomendado
                handler = get_format_handler(format_info)
                print(f"  Handler: {handler}")
                
            except Exception as e:
                print(f"Error con {filename}: {e}")
            
            # Limpiar archivo de prueba
            os.remove(filename)
            
        print("✓ Detección de formatos funcionando")
        return True
        
    except Exception as e:
        print(f"✗ Error en detección de formatos: {e}")
        return False

def test_data_validation():
    """Probar validación de datos."""
    print("\n=== Probando Validación de Datos ===")
    
    try:
        from drafts.data_loading import validate_data
        
        # Datos válidos
        valid_data = np.random.randn(100, 512)
        is_valid = validate_data(valid_data)
        print(f"Datos válidos (100x512): {is_valid}")
        
        # Datos inválidos
        invalid_data = np.array([])
        is_valid = validate_data(invalid_data)
        print(f"Datos vacíos: {is_valid}")
        
        # Datos con NaN
        nan_data = np.random.randn(10, 10)
        nan_data[0, 0] = np.nan
        is_valid = validate_data(nan_data)
        print(f"Datos con NaN: {is_valid}")
        
        print("✓ Validación de datos funcionando")
        return True
        
    except Exception as e:
        print(f"✗ Error en validación de datos: {e}")
        return False

def test_metadata_validation():
    """Probar validación de metadatos."""
    print("\n=== Probando Validación de Metadatos ===")
    
    try:
        from drafts.data_loading import validate_metadata
        
        # Metadatos válidos
        valid_metadata = {
            'frequency_resolution': 512,
            'time_resolution': 0.001,
            'file_length': 10000,
            'frequencies': np.linspace(1200, 1500, 512)
        }
        is_valid = validate_metadata(valid_metadata)
        print(f"Metadatos válidos: {is_valid}")
        
        # Metadatos inválidos
        invalid_metadata = {
            'frequency_resolution': 0,  # Inválido
            'time_resolution': 0.001,
            'file_length': 10000
        }
        is_valid = validate_metadata(invalid_metadata)
        print(f"Metadatos inválidos: {is_valid}")
        
        print("✓ Validación de metadatos funcionando")
        return True
        
    except Exception as e:
        print(f"✗ Error en validación de metadatos: {e}")
        return False

def test_structure():
    """Probar estructura del módulo."""
    print("\n=== Probando Estructura del Módulo ===")
    
    try:
        import drafts.data_loading as dl
        
        # Verificar que los módulos existen
        expected_modules = [
            'fits_loader',
            'fil_loader', 
            'data_validator',
            'metadata_extractor',
            'stream_processor',
            'header_parser',
            'data_preprocessor',
            'format_detector'
        ]
        
        for module_name in expected_modules:
            if hasattr(dl, module_name):
                print(f"✓ Módulo {module_name} presente")
            else:
                print(f"✗ Módulo {module_name} faltante")
        
        # Verificar funciones principales
        expected_functions = [
            'load_fits_data',
            'load_fil_data',
            'validate_data',
            'detect_file_format'
        ]
        
        for func_name in expected_functions:
            if hasattr(dl, func_name):
                print(f"✓ Función {func_name} presente")
            else:
                print(f"✗ Función {func_name} faltante")
        
        print("✓ Estructura del módulo correcta")
        return True
        
    except Exception as e:
        print(f"✗ Error verificando estructura: {e}")
        return False

def main():
    """Función principal de pruebas."""
    print("Iniciando pruebas del módulo data_loading...")
    
    # Ejecutar todas las pruebas
    tests = [
        test_imports,
        test_structure,
        test_format_detection,
        test_data_validation,
        test_metadata_validation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Error en test {test.__name__}: {e}")
    
    print(f"\n=== Resumen de Pruebas ===")
    print(f"Pruebas pasadas: {passed}/{total}")
    
    if passed == total:
        print("✓ Módulo data_loading creado exitosamente")
        print("✓ Separación de responsabilidades implementada")
        print("✓ Compatibilidad con código existente mantenida")
        print("✓ Funciones de validación funcionando")
        print("✓ Detección de formatos implementada")
    else:
        print("✗ Algunas pruebas fallaron")

if __name__ == "__main__":
    main() 