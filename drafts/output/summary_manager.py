"""Summary manager for FRB pipeline - handles logging, statistics, and progress tracking."""
import json
import logging
import time
from pathlib import Path
from datetime import datetime

from .. import config

logger = logging.getLogger(__name__)


def _write_summary(summary: dict, save_path: Path) -> None:
    """Write global summary information to ``summary.json``."""

    summary_path = save_path / "summary.json"
    with summary_path.open("w") as f_json:
        json.dump(summary, f_json, indent=2)
    logger.info("Resumen global escrito en %s", summary_path)


def _write_summary_with_timestamp(summary: dict, save_path: Path, preserve_history: bool = True) -> None:
    """
    Write summary with timestamp and optionally preserve historical summaries.
    
    Parameters
    ----------
    summary : dict
        Summary data to write
    save_path : Path
        Directory to save the summary
    preserve_history : bool
        If True, creates timestamped copies and maintains a master summary
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if preserve_history:
        # Crear summary con timestamp
        timestamped_path = save_path / f"summary_{timestamp}.json"
        with timestamped_path.open("w") as f_json:
            json.dump(summary, f_json, indent=2)
        logger.info("Resumen con timestamp escrito en %s", timestamped_path)
        
        # Actualizar summary maestro con historial
        master_summary_path = save_path / "summary_master.json"
        master_summary = _load_or_create_master_summary(master_summary_path)
        
        # Agregar esta ejecución al historial
        execution_record = {
            "timestamp": timestamp,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "files_processed": len(summary.get("files_processed", {})),
            "total_candidates": summary.get("global_stats", {}).get("total_candidates", 0),
            "total_bursts": summary.get("global_stats", {}).get("total_bursts", 0),
            "total_processing_time": summary.get("global_stats", {}).get("total_processing_time", 0.0),
            "pipeline_config": summary.get("pipeline_info", {}),
            "summary_file": f"summary_{timestamp}.json"
        }
        
        if "execution_history" not in master_summary:
            master_summary["execution_history"] = []
        
        master_summary["execution_history"].append(execution_record)
        
        # Mantener solo las últimas 10 ejecuciones para evitar archivos muy grandes
        if len(master_summary["execution_history"]) > 10:
            master_summary["execution_history"] = master_summary["execution_history"][-10:]
        
        # Actualizar estadísticas globales acumuladas
        if "cumulative_stats" not in master_summary:
            master_summary["cumulative_stats"] = {
                "total_executions": 0,
                "total_files_processed": 0,
                "total_candidates_detected": 0,
                "total_bursts_detected": 0,
                "total_processing_time": 0.0
            }
        
        master_summary["cumulative_stats"]["total_executions"] += 1
        master_summary["cumulative_stats"]["total_files_processed"] += execution_record["files_processed"]
        master_summary["cumulative_stats"]["total_candidates_detected"] += execution_record["total_candidates"]
        master_summary["cumulative_stats"]["total_bursts_detected"] += execution_record["total_bursts"]
        master_summary["cumulative_stats"]["total_processing_time"] += execution_record["total_processing_time"]
        
        # Escribir summary maestro
        with master_summary_path.open("w") as f_json:
            json.dump(master_summary, f_json, indent=2)
        logger.info("Summary maestro actualizado en %s", master_summary_path)
        
        # También escribir el summary actual (sin timestamp) para compatibilidad
        current_summary_path = save_path / "summary.json"
        with current_summary_path.open("w") as f_json:
            json.dump(summary, f_json, indent=2)
        logger.info("Summary actual escrito en %s", current_summary_path)
        
    else:
        # Modo simple: solo escribir con timestamp
        timestamped_path = save_path / f"summary_{timestamp}.json"
        with timestamped_path.open("w") as f_json:
            json.dump(summary, f_json, indent=2)
        logger.info("Resumen con timestamp escrito en %s", timestamped_path)


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


def _load_or_create_master_summary(master_path: Path) -> dict:
    """Load existing master summary or create a new one."""
    if master_path.exists():
        try:
            with master_path.open("r") as f:
                master_summary = json.load(f)
            return master_summary
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning(f"Error leyendo {master_path}, creando nuevo master summary")
    
    # Crear nuevo master summary
    return {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "execution_history": [],
        "cumulative_stats": {
            "total_executions": 0,
            "total_files_processed": 0,
            "total_candidates_detected": 0,
            "total_bursts_detected": 0,
            "total_processing_time": 0.0
        }
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


def get_execution_history(save_path: Path, limit: int = 10) -> list:
    """
    Get recent execution history from master summary.
    
    Parameters
    ----------
    save_path : Path
        Directory containing the master summary
    limit : int
        Maximum number of recent executions to return
        
    Returns
    -------
    list
        List of recent execution records
    """
    master_path = save_path / "summary_master.json"
    if not master_path.exists():
        return []
    
    try:
        with master_path.open("r") as f:
            master_summary = json.load(f)
        
        history = master_summary.get("execution_history", [])
        return history[-limit:] if limit > 0 else history
        
    except Exception as e:
        logger.error(f"Error leyendo historial de ejecuciones: {e}")
        return []


def get_cumulative_stats(save_path: Path) -> dict:
    """
    Get cumulative statistics from all executions.
    
    Parameters
    ----------
    save_path : Path
        Directory containing the master summary
        
    Returns
    -------
    dict
        Cumulative statistics
    """
    master_path = save_path / "summary_master.json"
    if not master_path.exists():
        return {}
    
    try:
        with master_path.open("r") as f:
            master_summary = json.load(f)
        
        return master_summary.get("cumulative_stats", {})
        
    except Exception as e:
        logger.error(f"Error leyendo estadísticas acumuladas: {e}")
        return {} 
