"""Wrapper for running the refactored Effelsberg FRB detection pipeline."""
import argparse
from pathlib import Path
from drafts.pipeline import run_pipeline
from drafts import config


def main():
    """Función principal con argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Pipeline de detección de FRB con soporte para chunking"
    )
    parser.add_argument(
        "--chunk-samples", 
        type=int, 
        default=2_097_152,
        help="Número de muestras por bloque para archivos .fil (0 = modo antiguo, default: 2M)"
    )
    parser.add_argument(
        "--data-dir", 
        type=Path, 
        help="Directorio con archivos de datos (default: ./Data)"
    )
    parser.add_argument(
        "--results-dir", 
        type=Path, 
        help="Directorio para resultados (default: ./Results/ObjectDetection)"
    )
    parser.add_argument(
        "--det-model", 
        type=Path, 
        help="Ruta al modelo de detección"
    )
    parser.add_argument(
        "--class-model", 
        type=Path, 
        help="Ruta al modelo de clasificación"
    )
    
    args = parser.parse_args()
    
    # Configurar parámetros si se proporcionan
    if args.data_dir:
        config.DATA_DIR = args.data_dir
    if args.results_dir:
        config.RESULTS_DIR = args.results_dir
    if args.det_model:
        config.MODEL_PATH = args.det_model
    if args.class_model:
        config.CLASS_MODEL_PATH = args.class_model
    
    # Ejecutar pipeline
    run_pipeline(chunk_samples=args.chunk_samples)


if __name__ == "__main__":
    main()
