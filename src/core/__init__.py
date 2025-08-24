"""
Módulo core que contiene los componentes principales del pipeline de detección de FRB.

Este módulo incluye:
- detection_engine: Motor de detección y clasificación de candidatos
- pipeline: Orquestador principal del pipeline de procesamiento
"""

from .detection_engine import process_slice
from .pipeline import run_pipeline

__all__ = ['process_slice', 'run_pipeline']
