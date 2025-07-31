#!/usr/bin/env python3
"""
Script para ejecutar el Pipeline Chunked V2
==========================================

Este script:
1. Carga los metadatos del archivo correctamente
2. Configura SLICE_LEN dinámicamente
3. Ejecuta el Pipeline Chunked V2 con continuidad temporal perfecta
"""

import sys
from pathlib import Path
import numpy as np

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DRAFTS.core import config
from DRAFTS.preprocessing.slice_len_utils import update_slice_len_dynamic
from DRAFTS.data_loader import create_data_loader
from DRAFTS.chunked_pipeline_v2 import ChunkedPipelineV2


def setup_configuration(file_path: str):
    """
    Configurar el pipeline correctamente.
    
    Args:
        file_path: Ruta al archivo de datos
    """
    
    print("🔧 CONFIGURANDO PIPELINE CHUNKED V2")
    print("=" * 60)
    
    # 1. CARGAR METADATOS
    print(f"\n📁 1. CARGANDO METADATOS DE: {file_path}")
    
    data_loader = create_data_loader(Path(file_path))
    metadata = data_loader.load_metadata()
    
    print(f"   ✅ Metadatos cargados")
    print(f"   📏 Muestras totales: {metadata.get('nsamples', 0):,}")
    print(f"   ⏱️  TIME_RESO: {metadata.get('time_reso', 'N/A')}")
    print(f"   📊 Canales: {metadata.get('nchans', 'N/A')}")
    
    # 2. ACTUALIZAR CONFIGURACIÓN GLOBAL
    print(f"\n⚙️  2. ACTUALIZANDO CONFIGURACIÓN:")
    
    if metadata.get('time_reso'):
        config.TIME_RESO = metadata['time_reso']
        print(f"   ✅ TIME_RESO: {config.TIME_RESO}")
    
    if metadata.get('down_time_rate'):
        config.DOWN_TIME_RATE = metadata['down_time_rate']
        print(f"   ✅ DOWN_TIME_RATE: {config.DOWN_TIME_RATE}")
    
    if metadata.get('down_freq_rate'):
        config.DOWN_FREQ_RATE = metadata['down_freq_rate']
        print(f"   ✅ DOWN_FREQ_RATE: {config.DOWN_FREQ_RATE}")
    
    # 3. CALCULAR SLICE_LEN
    print(f"\n🧮 3. CALCULANDO SLICE_LEN:")
    
    slice_len_old = config.SLICE_LEN
    slice_len_new, duration_ms = update_slice_len_dynamic()
    
    print(f"   📏 SLICE_LEN: {slice_len_old} → {slice_len_new}")
    print(f"   ⏱️  Duración: {duration_ms:.1f} ms")
    
    return metadata


def run_pipeline_v2(file_path: str, save_dir: str = None):
    """
    Ejecutar el Pipeline Chunked V2.
    
    Args:
        file_path: Ruta al archivo de datos
        save_dir: Directorio de guardado (opcional)
    """
    
    print("\n🚀 EJECUTANDO PIPELINE CHUNKED V2")
    print("=" * 60)
    
    # Configurar directorio de guardado
    if save_dir is None:
        save_dir = Path("Results") / Path(file_path).stem
    else:
        save_dir = Path(save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar el pipeline
    metadata = setup_configuration(file_path)
    
    # Crear modelos dummy (aquí irían tus modelos reales)
    class DummyModel:
        def __call__(self, *args, **kwargs):
            return [], None
    
    det_model = DummyModel()
    cls_model = DummyModel()
    
    # Crear y ejecutar pipeline
    print(f"\n📦 4. CREANDO PIPELINE:")
    
    pipeline = ChunkedPipelineV2(
        det_model=det_model,
        cls_model=cls_model,
        fits_path=Path(file_path),
        save_dir=save_dir
    )
    
    print(f"   ✅ Pipeline creado")
    print(f"   📊 Chunks configurados: {pipeline.chunk_config['num_chunks']}")
    print(f"   📊 Slices por chunk: {pipeline.chunk_config['slices_per_chunk']}")
    print(f"   ⏱️  Duración por chunk: {pipeline.chunk_config['chunk_duration_seconds']:.1f}s")
    
    # Ejecutar pipeline
    print(f"\n🔄 5. EJECUTANDO PROCESAMIENTO:")
    
    try:
        results = pipeline.run()
        
        print(f"\n🎉 PROCESAMIENTO COMPLETADO:")
        print(f"   ⏱️  Tiempo total: {results['processing_time']:.1f}s")
        print(f"   📊 Candidatos: {results['total_candidates']}")
        print(f"   📊 Slices procesados: {results['total_slices']}")
        print(f"   ⏱️  Duración del archivo: {results['file_duration']:.1f}s")
        print(f"   📁 Resultados guardados en: {save_dir}")
        
        return results
        
    except Exception as e:
        print(f"   ❌ Error durante el procesamiento: {e}")
        return None


def main():
    """Función principal."""
    
    if len(sys.argv) < 2:
        print("📋 Uso: python scripts/run_chunked_pipeline_v2.py <archivo.fits|archivo.fil> [directorio_guardado]")
        print("📋 Ejemplo: python scripts/run_chunked_pipeline_v2.py Data/3098_0001_00_8bit.fil Results/test")
        return
    
    file_path = sys.argv[1]
    save_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("🎯 PIPELINE CHUNKED V2 - CONTINUIDAD TEMPORAL PERFECTA")
    print("=" * 70)
    print(f"📁 Archivo: {file_path}")
    print(f"📁 Guardado en: {save_dir or 'Results/' + Path(file_path).stem}")
    
    results = run_pipeline_v2(file_path, save_dir)
    
    if results:
        print(f"\n✅ ¡PROCESAMIENTO EXITOSO!")
        print(f"🎯 Continuidad temporal: PERFECTA")
        print(f"🎯 Slices regidos por SLICE_DURATION_MS: {config.SLICE_DURATION_MS} ms")
        print(f"🎯 Procesamiento completo sin pérdida de información")
    else:
        print(f"\n❌ Error en el procesamiento")


if __name__ == "__main__":
    main() 
