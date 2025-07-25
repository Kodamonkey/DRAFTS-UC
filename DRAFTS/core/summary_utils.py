import json
import logging
import time
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)


def _write_summary(summary: dict, save_path: Path) -> None:
    """Write global summary information to ``summary.json``."""

    summary_path = save_path / "summary.json"
    with summary_path.open("w") as f_json:
        json.dump(summary, f_json, indent=2)
    logger.info("Resumen global escrito en %s", summary_path)


def _load_or_create_summary(save_path: Path) -> dict:
    """Load existing summary.json or create a new one."""
    summary_path = save_path / "summary.json"

    if summary_path.exists():
        try:
            with summary_path.open("r") as f:
                summary = json.load(f)
            # Ensure required keys exist
            if "files_processed" not in summary:
                summary["files_processed"] = {}
            if "pipeline_info" not in summary:
                summary["pipeline_info"] = {}
            if "global_stats" not in summary:
                summary["global_stats"] = {
                    "total_files": 0,
                    "total_candidates": 0,
                    "total_bursts": 0,
                    "total_processing_time": 0.0,
                }
            return summary
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning(f"Error leyendo {summary_path}, creando nuevo summary")

    # Crear nuevo summary con estructura inicial
    return {
        "pipeline_info": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": config.MODEL_NAME,
            "dm_range": f"{config.DM_min}-{config.DM_max}",
            "slice_duration_ms": config.SLICE_DURATION_MS,
            "debug_enabled": config.DEBUG_FREQUENCY_ORDER,
        },
        "files_processed": {},
        "global_stats": {
            "total_files": 0,
            "total_candidates": 0,
            "total_bursts": 0,
            "total_processing_time": 0.0,
        },
    }


def _update_summary_with_file_debug(
    save_path: Path, filename: str, debug_info: dict
) -> None:
    """Update summary.json immediately with file debug information."""

    summary = _load_or_create_summary(save_path)

    if filename not in summary["files_processed"]:
        summary["files_processed"][filename] = {}

    summary["files_processed"][filename]["debug_info"] = debug_info
    summary["files_processed"][filename]["status"] = "debug_completed"
    summary["files_processed"][filename]["debug_timestamp"] = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    summary_path = save_path / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    if config.DEBUG_FREQUENCY_ORDER:
        logger.info(f"Debug info guardada para {filename} en {summary_path}")


def _update_summary_with_results(
    save_path: Path, filename: str, results_info: dict
) -> None:
    """Update summary.json with processing results for a file."""

    summary = _load_or_create_summary(save_path)

    if filename not in summary["files_processed"]:
        summary["files_processed"][filename] = {}

    summary["files_processed"][filename].update(results_info)
    summary["files_processed"][filename]["status"] = "processing_completed"
    summary["files_processed"][filename]["results_timestamp"] = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    summary["global_stats"]["total_files"] = len(summary["files_processed"])
    summary["global_stats"]["total_candidates"] += results_info.get("n_candidates", 0)
    summary["global_stats"]["total_bursts"] += results_info.get("n_bursts", 0)
    summary["global_stats"]["total_processing_time"] += results_info.get(
        "processing_time", 0.0
    )

    summary_path = save_path / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Resultados guardados para {filename} en {summary_path}")
