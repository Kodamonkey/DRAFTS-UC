#!/usr/bin/env python3
"""
Script de prueba para demostrar el sistema de summaries con timestamp.
"""

import json
from pathlib import Path
from drafts.output.summary_manager import (
    _write_summary_with_timestamp,
    get_execution_history,
    get_cumulative_stats
)

def create_mock_summary(execution_id: int) -> dict:
    """Crear un summary simulado para pruebas."""
    return {
        "pipeline_info": {
            "timestamp": "2024-01-01 12:00:00",
            "model": "resnet50",
            "dm_range": "0-1024",
            "slice_duration_ms": 64.0,
            "execution_id": execution_id
        },
        "files_processed": {
            f"archivo_{execution_id}_1.fits": {
                "n_candidates": 5 + execution_id,
                "n_bursts": 2,
                "processing_time": 120.5,
                "status": "completed"
            },
            f"archivo_{execution_id}_2.fits": {
                "n_candidates": 3,
                "n_bursts": 1,
                "processing_time": 95.2,
                "status": "completed"
            }
        },
        "global_stats": {
            "total_files": 2,
            "total_candidates": 8 + execution_id,
            "total_bursts": 3,
            "total_processing_time": 215.7
        }
    }

def test_summary_timestamp_system():
    """Probar el sistema de summaries con timestamp."""
    
    # Crear directorio de prueba
    test_dir = Path("./tests/test_summary_output")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print("🧪 Probando sistema de summaries con timestamp...")
    print(f"📁 Directorio de prueba: {test_dir}")
    
    # Simular múltiples ejecuciones
    for i in range(1, 4):
        print(f"\n📊 Simulando ejecución {i}...")
        
        # Crear summary simulado
        mock_summary = create_mock_summary(i)
        
        # Escribir con timestamp y preservar historial
        _write_summary_with_timestamp(mock_summary, test_dir, preserve_history=True)
        
        print(f"✅ Ejecución {i} completada")
    
    # Mostrar historial de ejecuciones
    print(f"\n📋 Historial de ejecuciones:")
    history = get_execution_history(test_dir, limit=10)
    
    for i, record in enumerate(history, 1):
        print(f"  {i}. {record['datetime']} - {record['files_processed']} archivos, "
              f"{record['total_candidates']} candidatos, {record['total_bursts']} bursts")
    
    # Mostrar estadísticas acumuladas
    print(f"\n📈 Estadísticas acumuladas:")
    cumulative = get_cumulative_stats(test_dir)
    
    if cumulative:
        print(f"  Total ejecuciones: {cumulative['total_executions']}")
        print(f"  Total archivos procesados: {cumulative['total_files_processed']}")
        print(f"  Total candidatos detectados: {cumulative['total_candidates_detected']}")
        print(f"  Total bursts detectados: {cumulative['total_bursts_detected']}")
        print(f"  Tiempo total de procesamiento: {cumulative['total_processing_time']:.2f}s")
    else:
        print("  No hay estadísticas acumuladas disponibles")
    
    # Listar archivos creados
    print(f"\n📁 Archivos creados:")
    for file_path in test_dir.glob("*.json"):
        file_size = file_path.stat().st_size / 1024  # KB
        print(f"  {file_path.name} ({file_size:.1f} KB)")
    
    print(f"\n✅ Prueba completada. Revisa los archivos en: {test_dir}")

if __name__ == "__main__":
    test_summary_timestamp_system()