# This module manages pipeline summary metadata.

"""Summary manager for FRB pipeline - handles logging, statistics, and progress tracking."""
                          
import json
import logging
import time
from datetime import datetime
from pathlib import Path

               
from ..config import config

              
logger = logging.getLogger(__name__)

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
                                     
        timestamped_path = save_path / f"summary_{timestamp}.json"
        with timestamped_path.open("w") as f_json:
            json.dump(summary, f_json, indent=2)
        logger.info("Timestamped summary written to %s", timestamped_path)
        
                                                  
        master_summary_path = save_path / "summary_master.json"
        master_summary = _load_or_create_master_summary(master_summary_path)
        
                                             
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
        
                                                                                   
        if len(master_summary["execution_history"]) > 10:
            master_summary["execution_history"] = master_summary["execution_history"][-10:]
        
                                                     
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
        
                                  
        with master_summary_path.open("w") as f_json:
            json.dump(master_summary, f_json, indent=2)
        logger.info("Master summary updated at %s", master_summary_path)
        
                                                                                
        current_summary_path = save_path / "summary.json"
        with current_summary_path.open("w") as f_json:
            json.dump(summary, f_json, indent=2)
        logger.info("Current summary written to %s", current_summary_path)
        
    else:
                                                  
        timestamped_path = save_path / f"summary_{timestamp}.json"
        with timestamped_path.open("w") as f_json:
            json.dump(summary, f_json, indent=2)
        logger.info("Timestamped summary written to %s", timestamped_path)


def _load_or_create_summary(save_path: Path) -> dict:
    """Load existing summary.json or create a new one."""
    summary_path = save_path / "summary.json"

    if summary_path.exists():
        try:
            with summary_path.open("r") as f:
                summary = json.load(f)
                                        
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
            logger.warning(f"Error reading {summary_path}, creating new summary")

                                                
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
            logger.warning(f"Error reading {master_path}, creating new master summary")
    
                                
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
        logger.info(f"Debug info saved for {filename} at {summary_path}")


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

    logger.info(f"Results saved for {filename} at {summary_path}")
