#!/usr/bin/env python3
"""
Script de Prueba para Estrategias Milimétricas en DRAFTS-MB
=========================================================

Este script demuestra cómo ejecutar el pipeline con las estrategias
milimétricas E1/E2 activadas.

Autor: DRAFTS-MB Team
Fecha: 2024
"""

import sys
import os
from pathlib import Path

# Añadir el directorio del proyecto al path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

def setup_millimeter_config():
    """Configura el entorno para usar estrategias milimétricas."""
    print("🔧 Configurando estrategias milimétricas...")
    
    # Importar y configurar
    from drafts import config
    
    # Activar estrategias milimétricas
    config.STRATEGY_DM_EXPAND = {
        'enabled': True,
        'dm_max': 2000,
        'smear_frac': 0.25,
        'min_dm_sigmas': 3.0,
    }
    
    config.STRATEGY_FISH_NEAR_ZERO = {
        'enabled': True,
        'dm_fish_max': 50,
        'fish_thresh': 0.3,
        'refine': {
            'dm_local_max': 300,
            'ddm_local': 1,
            'min_delta_snr': 2.0,
            'min_dm_star': 5,
            'subband_consistency_pc': 20,
        }
    }
    
    # Configuración de logging para mejor visibilidad
    config.LOG_LEVEL = "INFO"
    config.LOG_COLORS = True
    
    print("✅ Estrategias E1 y E2 activadas")
    return config

def verify_dependencies():
    """Verifica que todos los módulos necesarios estén disponibles."""
    print("🔍 Verificando dependencias...")
    
    try:
        from drafts.preprocessing.dm_planner import build_dm_grids
        print("✅ DM Planner disponible")
    except ImportError as e:
        print(f"❌ Error: DM Planner no disponible - {e}")
        return False
    
    try:
        from drafts.validators.dm_validator import DMValidator
        print("✅ DM Validator disponible")
    except ImportError as e:
        print(f"❌ Error: DM Validator no disponible - {e}")
        return False
    
    try:
        from drafts.preprocessing.dedispersion import dm_index_to_physical
        print("✅ Dedispersion con metadatos disponible")
    except ImportError as e:
        print(f"❌ Error: Dedispersion no disponible - {e}")
        return False
    
    try:
        from drafts.output.candidate_manager import Candidate
        candidate = Candidate(
            file="test", chunk_id=0, slice_id=0, band_id=0, prob=0.5,
            dm=25.0, t_sec=1.0, t_sample=1000, box=(10,20,30,40), snr=8.0
        )
        # Verificar nuevos campos
        assert hasattr(candidate, 'dm_star')
        assert hasattr(candidate, 'validation_passed')
        assert hasattr(candidate, 'strategy')
        print("✅ Candidate con campos milimétricas disponible")
    except Exception as e:
        print(f"❌ Error: Candidate extendido no disponible - {e}")
        return False
    
    return True

def check_models():
    """Verifica que los modelos estén disponibles."""
    print("🤖 Verificando modelos...")
    
    models_dir = project_dir / "models"
    
    detection_model = models_dir / "cent_resnet50.pth"
    if detection_model.exists():
        print(f"✅ Modelo de detección: {detection_model}")
    else:
        print(f"❌ Modelo de detección no encontrado: {detection_model}")
        return False
    
    classification_model = models_dir / "class_resnet50.pth"
    if classification_model.exists():
        print(f"✅ Modelo de clasificación: {classification_model}")
    else:
        print(f"❌ Modelo de clasificación no encontrado: {classification_model}")
        return False
    
    return True

def check_data():
    """Verifica que haya datos de entrada disponibles."""
    print("📁 Verificando datos de entrada...")
    
    data_dir = project_dir / "Data" / "raw"
    
    if not data_dir.exists():
        print(f"❌ Directorio de datos no encontrado: {data_dir}")
        return False
    
    fits_files = list(data_dir.glob("*.fits"))
    fil_files = list(data_dir.glob("*.fil"))
    
    if fits_files:
        print(f"✅ Archivos FITS encontrados: {len(fits_files)}")
        for f in fits_files[:3]:  # Mostrar solo los primeros 3
            print(f"   - {f.name}")
        if len(fits_files) > 3:
            print(f"   - ... y {len(fits_files) - 3} más")
    
    if fil_files:
        print(f"✅ Archivos FIL encontrados: {len(fil_files)}")
        for f in fil_files[:3]:  # Mostrar solo los primeros 3
            print(f"   - {f.name}")
        if len(fil_files) > 3:
            print(f"   - ... y {len(fil_files) - 3} más")
    
    if not fits_files and not fil_files:
        print("❌ No se encontraron archivos de datos (.fits o .fil)")
        return False
    
    return True

def run_millimeter_pipeline():
    """Ejecuta el pipeline con estrategias milimétricas."""
    print("\n🚀 Iniciando pipeline con estrategias milimétricas...")
    
    try:
        from drafts.pipeline import run_pipeline
        
        # Ejecutar pipeline
        run_pipeline(chunk_samples=0)  # 0 = automático
        
        print("✅ Pipeline ejecutado exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error ejecutando pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_results():
    """Muestra un resumen de los resultados."""
    print("\n📊 Verificando resultados...")
    
    results_dir = project_dir / "Results" / "ObjectDetection"
    
    if not results_dir.exists():
        print("❌ No se encontró directorio de resultados")
        return
    
    # Buscar archivos CSV de candidatos
    csv_files = list(results_dir.rglob("candidates.csv"))
    
    if csv_files:
        print(f"✅ Archivos de candidatos encontrados: {len(csv_files)}")
        for csv_file in csv_files:
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
                
                # Verificar columnas nuevas de estrategias milimétricas
                millimeter_cols = ['dm_star', 'delta_snr', 'subband_agreement', 'validation_passed', 'strategy']
                has_millimeter = any(col in df.columns for col in millimeter_cols)
                
                print(f"   - {csv_file.relative_to(project_dir)}: {len(df)} candidatos")
                if has_millimeter:
                    # Contar candidatos por estrategia
                    if 'strategy' in df.columns:
                        strategy_counts = df['strategy'].value_counts()
                        print(f"     • Estrategias: {dict(strategy_counts)}")
                    
                    if 'validation_passed' in df.columns:
                        validated = df['validation_passed'].sum() if 'validation_passed' in df.columns else 0
                        print(f"     • Candidatos validados: {validated}")
                    
                    print(f"     • ✅ Columnas milimétricas presentes")
                else:
                    print(f"     • ⚠️ Sin columnas milimétricas (modo clásico)")
                
            except Exception as e:
                print(f"   - {csv_file.relative_to(project_dir)}: Error leyendo - {e}")
    else:
        print("❌ No se encontraron archivos de candidatos")

def main():
    """Función principal del script de prueba."""
    print("="*80)
    print("🎯 SCRIPT DE PRUEBA: ESTRATEGIAS MILIMÉTRICAS DRAFTS-MB")
    print("="*80)
    
    # Verificaciones previas
    if not verify_dependencies():
        print("\n❌ Faltan dependencias necesarias")
        return 1
    
    if not check_models():
        print("\n❌ Faltan modelos necesarios")
        print("💡 Asegúrate de que los modelos estén en la carpeta 'models/'")
        return 1
    
    if not check_data():
        print("\n❌ No hay datos de entrada")
        print("💡 Coloca archivos .fits o .fil en 'Data/raw/'")
        return 1
    
    # Configurar estrategias milimétricas
    config = setup_millimeter_config()
    
    # Ejecutar pipeline
    success = run_millimeter_pipeline()
    
    if success:
        # Mostrar resultados
        show_results()
        
        print("\n🎉 ¡Prueba completada exitosamente!")
        print("\n📋 Próximos pasos:")
        print("   1. Revisar archivos CSV en Results/ObjectDetection/")
        print("   2. Analizar candidatos con strategy='E1_expand' o 'E2_fish'")
        print("   3. Verificar plots en las carpetas Composite/, Detections/, etc.")
        print("   4. Ajustar parámetros en user_config.py según sea necesario")
        
        return 0
    else:
        print("\n❌ Error durante la ejecución")
        print("💡 Consulta los logs para más detalles")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
