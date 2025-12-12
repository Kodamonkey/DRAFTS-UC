"""Execution summary generator - creates JSON summary with phase metrics for pipeline execution."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, TYPE_CHECKING

from ..config import config

if TYPE_CHECKING:
    from .phase_metrics import PhaseMetricsTracker

logger = logging.getLogger(__name__)


def generate_execution_summary(
    phase_metrics_by_file: Dict[str, 'PhaseMetricsTracker'],
    save_dir: Path,
    file_summary: Dict[str, Dict],
) -> None:
    """Generate execution summary JSON with phase metrics.
    
    Args:
        phase_metrics_by_file: Dictionary mapping filename to PhaseMetricsTracker
        save_dir: Directory where results are saved (from config.RESULTS_DIR)
        file_summary: Summary dictionary from pipeline execution
    """
    # Create Summary-execution directory
    summary_exec_dir = save_dir / "Summary-execution"
    summary_exec_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate global execution summary
    global_tracker = None
    files_data = []
    
    for filename, tracker in phase_metrics_by_file.items():
        # Initialize global tracker with first file's tracker
        if global_tracker is None:
            from .phase_metrics import PhaseMetricsTracker
            global_tracker = PhaseMetricsTracker()
        
        # Merge file metrics into global tracker
        global_tracker.merge(tracker)
        
        # Get file summary data
        file_info = file_summary.get(filename, {})
        
        # Create file entry
        file_data = {
            "filename": filename,
            "total_candidates": file_info.get("n_candidates", 0),
            "total_burst": file_info.get("n_bursts", 0),
            "total_no_burst": file_info.get("n_no_bursts", 0),
            "by_phase": tracker.to_dict()["by_phase"],
        }
        files_data.append(file_data)
    
    # If no metrics were collected, create empty summary
    if global_tracker is None:
        from .phase_metrics import PhaseMetricsTracker
        global_tracker = PhaseMetricsTracker()
    
    # Build execution summary structure
    execution_data = {
        "execution_info": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_files_processed": len(phase_metrics_by_file),
            "pipeline_config": {
                "detection_probability": float(config.DET_PROB),
                "classification_probability": float(config.CLASS_PROB),
                "classification_probability_linear": float(getattr(config, 'CLASS_PROB_LINEAR', config.CLASS_PROB)),
                "snr_threshold": float(config.SNR_THRESH),
                "snr_threshold_linear": float(getattr(config, 'SNR_THRESH_LINEAR', config.SNR_THRESH)),
                "enable_linear_validation": bool(getattr(config, 'ENABLE_LINEAR_VALIDATION', False)),
                "enable_intensity_classification": bool(getattr(config, 'ENABLE_INTENSITY_CLASSIFICATION', True)),
                "enable_linear_classification": bool(getattr(config, 'ENABLE_LINEAR_CLASSIFICATION', True)),
                "save_only_burst": bool(config.SAVE_ONLY_BURST),
                "dm_min": float(config.DM_min),
                "dm_max": float(config.DM_max),
                "slice_duration_ms": float(config.SLICE_DURATION_MS),
                "down_freq_rate": int(config.DOWN_FREQ_RATE),
                "down_time_rate": int(config.DOWN_TIME_RATE),
            }
        },
        "execution_summary": global_tracker.to_dict(),
        "files": files_data,
    }
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = summary_exec_dir / f"execution_metrics_{timestamp}.json"
    
    # Write timestamped file
    try:
        with timestamped_path.open("w", encoding="utf-8") as f:
            json.dump(execution_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Execution summary written to: {timestamped_path}")
    except Exception as e:
        logger.error(f"Failed to write execution summary: {e}")
        raise
    
    # Also write latest file (overwrite)
    latest_path = summary_exec_dir / "execution_metrics_latest.json"
    try:
        with latest_path.open("w", encoding="utf-8") as f:
            json.dump(execution_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Latest execution summary written to: {latest_path}")
    except Exception as e:
        logger.warning(f"Failed to write latest execution summary: {e}")

