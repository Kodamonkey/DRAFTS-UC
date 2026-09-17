#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plomería compartida por los scripts analyze_case_*.py (REF-18).

Contiene sólo el andamiaje que era byte a byte idéntico entre
``analyze_case_alma_psr1745.py``, ``analyze_case_b0355.py``,
``analyze_case_fast_frex.py`` y ``analyze_case_frb121102.py``:

* configuración de ``sys.path`` y de ``logging``;
* localización de los JSON de validación y de los CSV de candidatos;
* construcción del ``ArgumentParser`` y resolución/validación de rutas;
* escritura del informe JSON y despacho a ``print_summary``.

El contenido científico de cada caso (``analyze_file`` / ``generate_report`` /
``print_summary`` y los ``extract_*_metrics``) NO vive aquí: es distinto en cada
script y debe seguir siéndolo.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ``import logging`` ARRIBA, antes de tocar sys.path: src/ contiene paquetes con
# nombres propios del proyecto y en el pasado uno de ellos (src/logging/, hoy
# src/log_utils/) ensombreció la biblioteca estándar. Además se usa
# sys.path.append con ruta ABSOLUTA derivada de __file__ — y no
# sys.path.insert(0, ...) como hacían los cuatro scripts — para que src/ quede
# al final de sys.path y no pueda ganarle a un módulo del stdlib.
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent
PROJECT_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.append(str(_SRC_DIR))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_validation_json(results_dir: Path, file_stem: str) -> Optional[Path]:
    """Find validation JSON file for a given file stem."""
    # Look in Validation/ directory (direct files)
    validation_dir = results_dir / 'Validation'
    if validation_dir.exists():
        # Try direct match
        for json_file in validation_dir.glob(f'*{file_stem}*.json'):
            return json_file
        # Try in subdirectories
        for subdir in validation_dir.iterdir():
            if subdir.is_dir() and file_stem in subdir.name:
                for json_file in subdir.glob(f'*{file_stem}*.json'):
                    return json_file

    # Look in Summary/*/Validation/validation_metrics.json (legacy structure)
    for summary_dir in results_dir.glob('Summary/*'):
        if file_stem in summary_dir.name:
            validation_dir = summary_dir / 'Validation'
            if validation_dir.exists():
                json_file = validation_dir / 'validation_metrics.json'
                if json_file.exists():
                    return json_file
    return None


def find_candidates_csvs(results_dir: Path, file_pattern: str) -> List[Path]:
    """
    Find ALL candidate CSV files whose Summary/* directory matches ``file_pattern``.

    Ésta es la semántica de analyze_case_alma_psr1745.py, que procesa todos los
    archivos que casan con el patrón. Para la semántica de "sólo el primero"
    (analyze_case_b0355.py, analyze_case_frb121102.py) usar
    ``find_candidates_csv``. Los dos nombres son distintos a propósito: antes
    eran dos funciones con el MISMO nombre y tipos de retorno incompatibles
    (List[Path] vs Optional[Path]).
    """
    csv_files = []
    for summary_dir in results_dir.glob('Summary/*'):
        if file_pattern in summary_dir.name:
            csv_file = summary_dir / f"{summary_dir.name}.candidates.csv"
            if csv_file.exists():
                csv_files.append(csv_file)
    return csv_files


def find_candidates_csv(results_dir: Path, file_stem: str) -> Optional[Path]:
    """Find the FIRST candidates CSV file for a given file stem (or None)."""
    for csv_file in find_candidates_csvs(results_dir, file_stem):
        return csv_file
    return None


def load_validation_metrics(json_path: Path) -> Dict[str, Any]:
    """Load validation metrics from JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {json_path}: {e}")
        return {}


def build_arg_parser(description: str, default_results_dir: str) -> argparse.ArgumentParser:
    """
    Construye el ArgumentParser común a los cuatro scripts.

    Args:
        description: Descripción del script (distinta en cada caso).
        default_results_dir: Valor por defecto de --results-dir. OJO: NO es el
            mismo en todos los scripts.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--results-dir',
        type=str,
        default=default_results_dir,
        help=f'Results directory (default: {default_results_dir})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON report file (optional)'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print summary statistics'
    )
    return parser


def resolve_results_dir(results_dir_arg: str) -> Tuple[Path, Path]:
    """
    Resuelve (project_root, results_dir) y aborta si results_dir no existe.

    Returns:
        Tupla (project_root, results_dir).
    """
    project_root = PROJECT_ROOT
    results_dir = project_root / results_dir_arg

    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        sys.exit(1)

    return project_root, results_dir


def emit_report(
    report: Dict[str, Any],
    args: argparse.Namespace,
    project_root: Path,
    print_summary: Callable[[Dict[str, Any]], None],
    done_message: str,
) -> None:
    """
    Escribe el informe JSON si se pidió --output y despacha a print_summary.

    Args:
        report: Informe ya generado por el script.
        args: Namespace de argparse (usa .output y .summary).
        project_root: Raíz del proyecto, para resolver --output.
        print_summary: Función print_summary propia del script.
        done_message: Mensaje informativo cuando NO se pidió --summary.
    """
    # Save report if requested
    if args.output:
        output_path = project_root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to {output_path}")

    # Print summary
    if args.summary:
        print_summary(report)
    else:
        logger.info(done_message)
        logger.info("Use --summary to see detailed statistics")
