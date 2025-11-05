# This module configures the logging infrastructure.

"""
Centralized logging system for the DRAFTS pipeline
=================================================

This module provides a unified, configurable, and professional logging system
for the FRB detection pipeline. It includes:

- Logging level configuration
- Custom color formatters
- Handlers for multiple output types
- Utility functions for structured logging
- Automatic logging to a .log file
"""

                          
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

                            
class Colors:
    """ANSI colors for message formatting."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
                                     
    PIPELINE = '\033[38;5;39m'              
    DETECTION = '\033[38;5;82m'               
    PROCESSING = '\033[38;5;220m'                  
    GPU = '\033[38;5;213m'                 
    FILE = '\033[38;5;87m'              
    ERROR = '\033[38;5;196m'                  


class DRAFTSFormatter(logging.Formatter):
    """Custom formatter for DRAFTS with colors and structure."""
    
    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors
        
                                          
        self.formats = {
            logging.DEBUG: self._format_debug,
            logging.INFO: self._format_info,
            logging.WARNING: self._format_warning,
            logging.ERROR: self._format_error,
            logging.CRITICAL: self._format_critical,
        }
    
    def format(self, record):
        """Apply formatting according to the log level."""
        if record.levelno in self.formats:
            return self.formats[record.levelno](record)
        return super().format(record)
    
    def _format_debug(self, record):
        """Format DEBUG messages."""
        if self.use_colors:
            return f"{Colors.OKCYAN} [DEBUG] {record.getMessage()}{Colors.ENDC}"
        return f"[DEBUG] {record.getMessage()}"
    
    def _format_info(self, record):
        """Format INFO messages."""
        if self.use_colors:
            return f"{Colors.OKGREEN}[INFO] {record.getMessage()}{Colors.ENDC}"
        return f"[INFO] {record.getMessage()}"
    
    def _format_warning(self, record):
        """Format WARNING messages."""
        if self.use_colors:
            return f"{Colors.WARNING}[WARN] {record.getMessage()}{Colors.ENDC}"
        return f"{record.getMessage()}"
    
    def _format_error(self, record):
        """Format ERROR messages."""
        if self.use_colors:
            return f"{Colors.ERROR}[ERROR] {record.getMessage()}{Colors.ENDC}"
        return f"[ERROR] {record.getMessage()}"
    
    def _format_critical(self, record):
        """Format CRITICAL messages."""
        if self.use_colors:
            return f"{Colors.FAIL}{Colors.BOLD}[CRITICAL] {record.getMessage()}{Colors.ENDC}"
        return f"[CRITICAL] {record.getMessage()}"


class DRAFTSLogger:
    """Central logger used across the DRAFTS pipeline."""

    def __init__(
        self,
        name: str = "DRAFTS",
        level: str = "INFO",
        log_file: Optional[Path] = None,
        use_colors: bool = True,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        if self.logger.handlers:
            self.logger.propagate = False
            return

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(DRAFTSFormatter(use_colors))

        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path(__file__).parent / "log"
            log_file = log_dir / f"drafts_pipeline_{timestamp}.log"

        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(DRAFTSFormatter(use_colors=False))

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False

        root_logger = logging.getLogger()
        root_logger.setLevel(self.logger.level)
        root_logger.handlers.clear()
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

        self.logger.info("Logs stored in: %s", log_file.absolute())

    def pipeline_start(self, config: Dict[str, Any]) -> None:
        """Emit a concise summary when the pipeline starts."""

        self.logger.info("Starting FRB detection pipeline")
        self.logger.info(
            "Paths • data=%s • results=%s",
            config.get("data_dir", "N/A"),
            config.get("results_dir", "N/A"),
        )
        targets = config.get("targets", [])
        if targets:
            self.logger.info("Targets • %s", ", ".join(map(str, targets)))
        chunk_samples = config.get("chunk_samples", 0)
        if chunk_samples:
            self.logger.info("Chunk override • %s samples", f"{chunk_samples:,}")

    def pipeline_end(self, summary: Dict[str, Any]) -> None:
        """Log a compact summary of the pipeline execution."""

        total_files = len(summary)
        total_candidates = sum(r.get("n_candidates", 0) for r in summary.values())
        total_bursts = sum(r.get("n_bursts", 0) for r in summary.values())
        total_time = sum(r.get("runtime_s", 0.0) for r in summary.values())

        self.logger.info("Pipeline finished")
        self.logger.info(
            "Summary • files=%d • candidates=%d • bursts=%d • runtime=%.1fs",
            total_files,
            total_candidates,
            total_bursts,
            total_time,
        )

    def file_processing_start(self, filename: str, file_info: Dict[str, Any]) -> None:
        """Log that a new file is being processed."""

        self.logger.info(
            "Processing file • %s • samples=%s • duration=%.1f min • channels=%d",
            filename,
            f"{file_info.get('samples', 0):,}",
            file_info.get("duration_min", 0.0),
            file_info.get("channels", 0),
        )

    def file_processing_end(self, filename: str, results: Dict[str, Any]) -> None:
        """Log the result obtained after processing a file."""

        self.logger.info(
            "File result • %s • candidates=%d • bursts=%d • max_prob=%.2f • runtime=%.1fs",
            filename,
            results.get("n_candidates", 0),
            results.get("n_bursts", 0),
            results.get("max_prob", 0.0),
            results.get("runtime_s", 0.0),
        )

    def chunk_processing(self, chunk_idx: int, chunk_info: Dict[str, Any]) -> None:
        """Log a high-level summary for a chunk."""

        self.logger.info(
            "Chunk %03d • samples=%s • slices=%d",
            chunk_idx,
            f"{chunk_info.get('samples', 0):,}",
            chunk_info.get("slices", 0),
        )

    def slice_processing(self, slice_idx: int, slice_info: Dict[str, Any]) -> None:
        """Log slice information only when something relevant happened."""

        candidates = slice_info.get("candidates", 0)
        if candidates > 0:
            bursts = slice_info.get("bursts", 0)
            no_bursts = slice_info.get("no_bursts", candidates - bursts)
            self.logger.info(
                "Slice %03d • candidates=%d (bursts=%d • non-bursts=%d)",
                slice_idx,
                candidates,
                bursts,
                no_bursts,
            )

    def gpu_info(self, message: str, level: str = "INFO") -> None:
        """Log GPU related messages respecting their level."""

        if level.upper() == "DEBUG":
            self.logger.debug(message)
        else:
            self.logger.info(message)

    def debug_file_info(self, file_info: Dict[str, Any]) -> None:
        """Emit a compact debug line with file metadata."""

        self.logger.debug(
            "File info • Δt=%0.2e s • channels=%d • f_start=%.1f MHz • bandwidth=%.1f MHz",
            file_info.get("time_reso", 0.0),
            file_info.get("freq_reso", 0),
            file_info.get("freq_start", 0.0),
            file_info.get("bandwidth", 0.0),
        )

    def slice_config(self, config_info: Dict[str, Any]) -> None:
        """Log the slice configuration in a single line."""

        self.logger.info(
            "Slice configuration • target=%.1f ms • length=%d samples • actual=%.1f ms",
            config_info.get("target_ms", 0.0),
            config_info.get("slice_len", 0),
            config_info.get("real_ms", 0.0),
        )

    def slice_progress(self, current_slice: int, total_slices: int,
                       chunk_idx: Optional[int] = None) -> None:
        """Emit slice progress at debug level to avoid flooding the console."""

        percentage = (current_slice / total_slices) * 100 if total_slices else 0.0
        chunk_info = f"chunk {chunk_idx:03d}" if chunk_idx is not None else "pipeline"
        self.logger.debug(
            "Slice progress • %s • %03d/%03d (%.1f%%)",
            chunk_info,
            current_slice,
            max(total_slices - 1, 0),
            percentage,
        )

    def candidate_detected(self, dm: float, time: float, confidence: float,
                           class_prob: float, is_burst: bool, snr_raw: float,
                           snr_patch: float) -> None:
        """Log a candidate detection event."""

        burst_status = "burst" if is_burst else "noise"
        self.logger.info(
            "Candidate • DM=%.1f • t=%.3fs • detect=%.2f • class=%.2f • SNR(raw=%.2f, patch=%.2f) • %s",
            dm,
            time,
            confidence,
            class_prob,
            snr_raw,
            snr_patch,
            burst_status,
        )

    def slice_completed(self, slice_idx: int, candidates: int, bursts: int,
                        no_bursts: int) -> None:
        """Log aggregate information after finishing a slice."""

        if candidates > 0:
            self.logger.info(
                "Slice %03d complete • candidates=%d (bursts=%d • non-bursts=%d)",
                slice_idx,
                candidates,
                bursts,
                no_bursts,
            )

    def chunk_completed(self, chunk_idx: int, total_candidates: int,
                        total_bursts: int, total_no_bursts: int) -> None:
        """Log the summary for a completed chunk."""

        self.logger.info(
            "Chunk %03d complete • candidates=%d (bursts=%d • non-bursts=%d)",
            chunk_idx,
            total_candidates,
            total_bursts,
            total_no_bursts,
        )

    def processing_band(self, band_name: str, slice_idx: int) -> None:
        """Debug log for per-band processing steps."""

        self.logger.debug("Processing %s for slice %d", band_name, slice_idx)

    def band_candidates(self, band_name: str, candidate_count: int) -> None:
        """Log band statistics when candidates are found."""

        if candidate_count > 0:
            self.logger.info("%s • %d candidates", band_name, candidate_count)
        else:
            self.logger.debug("%s • no candidates", band_name)

    def creating_waterfall(self, waterfall_type: str, slice_idx: int,
                           dm: Optional[float] = None) -> None:
        """Debug log when generating waterfall plots."""

        dm_info = f" (DM={dm:.1f})" if dm is not None else ""
        self.logger.debug(
            "Creating %s waterfall for slice %d%s",
            waterfall_type,
            slice_idx,
            dm_info,
        )

    def generating_plots(self) -> None:
        """Debug log used when building plots."""

        self.logger.debug("Generating composite and detection plots")


def setup_logging(
    level: str = "INFO", log_file: Optional[Path] = None, use_colors: bool = True
) -> DRAFTSLogger:
    """Configure the logging system for DRAFTS.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: File path to store logs (optional, created automatically if ``None``)
        use_colors: Whether to use colors in the console output

    Returns:
        Configured :class:`DRAFTSLogger`
    """
    return DRAFTSLogger("DRAFTS", level, log_file, use_colors)


def get_logger(name: str = "DRAFTS") -> logging.Logger:
    """Get a configured logger."""
    return logging.getLogger(name)


                      
_global_logger: Optional[DRAFTSLogger] = None

def get_global_logger() -> DRAFTSLogger:
    """Retrieve the configured global logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logging()
    return _global_logger

def set_global_logger(logger: DRAFTSLogger):
    """Set the global logger."""
    global _global_logger
    _global_logger = logger 
