# This module configures the logging infrastructure.

"""
Sistema de Logging Centralizado para DRAFTS Pipeline
===================================================

Este módulo proporciona un sistema de logging unificado, configurable y profesional
para el pipeline de detección de FRB. Incluye:

- Configuración de niveles de logging
- Formateadores personalizados con colores
- Handlers para diferentes tipos de salida
- Funciones de utilidad para logging estructurado
- Logging automático a archivo .log
"""

                          
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

                            
class Colors:
    """Colores ANSI para formateo de mensajes."""
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
    """Formateador personalizado para DRAFTS con colores y estructura."""
    
    # This function initializes the formatter.
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
    
    # This function formats log records.
    def format(self, record):
        """Aplica formato según el nivel del log."""
        if record.levelno in self.formats:
            return self.formats[record.levelno](record)
        return super().format(record)
    
    # This function formats debug messages.
    def _format_debug(self, record):
        """Formato para mensajes DEBUG."""
        if self.use_colors:
            return f"{Colors.OKCYAN} [DEBUG] {record.getMessage()}{Colors.ENDC}"
        return f"[DEBUG] {record.getMessage()}"
    
    # This function formats info messages.
    def _format_info(self, record):
        """Formato para mensajes INFO."""
        if self.use_colors:
            return f"{Colors.OKGREEN}[INFO] {record.getMessage()}{Colors.ENDC}"
        return f"[INFO] {record.getMessage()}"
    
    # This function formats warning messages.
    def _format_warning(self, record):
        """Formato para mensajes WARNING."""
        if self.use_colors:
            return f"{Colors.WARNING}[WARN] {record.getMessage()}{Colors.ENDC}"
        return f"{record.getMessage()}"
    
    # This function formats error messages.
    def _format_error(self, record):
        """Formato para mensajes ERROR."""
        if self.use_colors:
            return f"{Colors.ERROR}[ERROR] {record.getMessage()}{Colors.ENDC}"
        return f"[ERROR] {record.getMessage()}"
    
    # This function formats critical messages.
    def _format_critical(self, record):
        """Formato para mensajes CRITICAL."""
        if self.use_colors:
            return f"{Colors.FAIL}{Colors.BOLD}[CRITICAL] {record.getMessage()}{Colors.ENDC}"
        return f"[CRITICAL] {record.getMessage()}"


class DRAFTSLogger:
    """Logger principal para DRAFTS con funcionalidades especializadas."""
    
    # This function initializes the logger.
    def __init__(self, name: str = "DRAFTS", level: str = "INFO", 
                 log_file: Optional[Path] = None, use_colors: bool = True):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
                                        
        if self.logger.handlers:
            return
        
                              
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(DRAFTSFormatter(use_colors))
        self.logger.addHandler(console_handler)
        
                                                               
        if log_file is None:
                                                                     
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path(__file__).parent / "log"
            log_file = log_dir / f"drafts_pipeline_{timestamp}.log"
        
                                                              
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(DRAFTSFormatter(use_colors=False))
        self.logger.addHandler(file_handler)
        
                                       
        self.logger.info(f"Logs guardándose en: {log_file.absolute()}")
    
    # This function logs pipeline startup information.
    def pipeline_start(self, config: Dict[str, Any]):
        """Log del inicio del pipeline."""
        self.logger.info(f"{Colors.PIPELINE}INICIANDO PIPELINE DE DETECCIÓN DE FRB{Colors.ENDC}")
        
                                                     
        self.logger.info(f"{Colors.OKBLUE}Leyenda de Colores:{Colors.ENDC}")
        self.logger.info(f"{Colors.PIPELINE} Pipeline: Operaciones principales del pipeline{Colors.ENDC}")
        self.logger.info(f"{Colors.FILE} Archivos: Procesamiento de archivos{Colors.ENDC}")
        self.logger.info(f"{Colors.PROCESSING} Procesamiento: Chunks y Slices{Colors.ENDC}")
        self.logger.info(f"{Colors.DETECTION} Detección: Candidatos detectados{Colors.ENDC}")
        self.logger.info(f"{Colors.GPU} GPU: Operaciones de GPU{Colors.ENDC}")
        self.logger.info(f"{Colors.WARNING} Advertencia: Problemas no críticos{Colors.ENDC}")
        self.logger.info(f"{Colors.ERROR} Error: Fallos críticos{Colors.ENDC}")
        self.logger.info(f"{Colors.OKCYAN} Debug: Información detallada (solo en modo DEBUG){Colors.ENDC}")
        self.logger.info("")                                  
        
        self.logger.info(f"Datos: {config.get('data_dir', 'N/A')}")
        self.logger.info(f"Resultados: {config.get('results_dir', 'N/A')}")
        self.logger.info(f"Targets: {config.get('targets', [])}")
        if config.get('chunk_samples', 0) > 0:
            self.logger.info(f"Chunking: {config['chunk_samples']:,} muestras/bloque")
    
    # This function logs pipeline completion details.
    def pipeline_end(self, summary: Dict[str, Any]):
        """Log del fin del pipeline."""
        total_files = len(summary)
        total_candidates = sum(r.get('n_candidates', 0) for r in summary.values())
        total_bursts = sum(r.get('n_bursts', 0) for r in summary.values())
        total_time = sum(r.get('runtime_s', 0) for r in summary.values())
        
        self.logger.info(f"{Colors.PIPELINE} PIPELINE COMPLETADO{Colors.ENDC}")
        self.logger.info(f"Resumen: {total_files} archivos, {total_candidates} candidatos, {total_bursts} bursts")
        self.logger.info(f"Tiempo total: {total_time:.1f}s")
    
    # This function logs the start of file processing.
    def file_processing_start(self, filename: str, file_info: Dict[str, Any]):
        """Log del inicio de procesamiento de archivo."""
        self.logger.info(f"{Colors.FILE} Procesando: {filename}{Colors.ENDC}")
        self.logger.info(f"Muestras: {file_info.get('samples', 0):,}")
        self.logger.info(f"Duración: {file_info.get('duration_min', 0):.1f} min")
        self.logger.info(f"Canales: {file_info.get('channels', 0)}")
    
    # This function logs the end of file processing.
    def file_processing_end(self, filename: str, results: Dict[str, Any]):
        """Log del fin de procesamiento de archivo."""
        self.logger.info(f"{Colors.FILE} {filename}: {results.get('n_candidates', 0)} candidatos, "
                        f"max prob {results.get('max_prob', 0):.2f}, "
                        f"{results.get('runtime_s', 0):.1f}s{Colors.ENDC}")
    
    # This function logs chunk processing.
    def chunk_processing(self, chunk_idx: int, chunk_info: Dict[str, Any]):
        """Log del procesamiento de chunk."""
        self.logger.info(f"{Colors.PROCESSING} Chunk {chunk_idx:03d}: "
                        f"{chunk_info.get('samples', 0):,} muestras → "
                        f"{chunk_info.get('slices', 0)} slices{Colors.ENDC}")
    
    # This function logs slice processing.
    def slice_processing(self, slice_idx: int, slice_info: Dict[str, Any]):
        """Log del procesamiento de slice."""
        candidates = slice_info.get('candidates', 0)
        if candidates > 0:
            self.logger.info(f"{Colors.DETECTION}Slice {slice_idx}: {candidates} candidatos detectados{Colors.ENDC}")
    
    # This function logs GPU information.
    def gpu_info(self, message: str, level: str = "INFO"):
        """Log de información GPU."""
        if level.upper() == "INFO":
            self.logger.info(f"{Colors.GPU} {message}{Colors.ENDC}")
        elif level.upper() == "DEBUG":
            self.logger.debug(f"{Colors.GPU} {message}{Colors.ENDC}")
    
    # This function logs detailed file information.
    def debug_file_info(self, file_info: Dict[str, Any]):
        """Log de información detallada de archivo (solo en DEBUG)."""
        self.logger.debug(f"Información del archivo:")
        self.logger.debug(f"   - Resolución temporal: {file_info.get('time_reso', 0):.2e} s")
        self.logger.debug(f"   - Resolución frecuencia: {file_info.get('freq_reso', 0)} canales")
        self.logger.debug(f"   - Frecuencia inicial: {file_info.get('freq_start', 0):.1f} MHz")
        self.logger.debug(f"   - Ancho de banda: {file_info.get('bandwidth', 0):.1f} MHz")
    
    # This function logs slice configuration.
    def slice_config(self, config_info: Dict[str, Any]):
        """Log de configuración de slice."""
        self.logger.info(f"Configuración de slice:")
        self.logger.info(f"Duración objetivo: {config_info.get('target_ms', 0):.1f} ms")
        self.logger.info(f"SLICE_LEN: {config_info.get('slice_len', 0)} muestras")
        self.logger.info(f"Duración real: {config_info.get('real_ms', 0):.1f} ms")
    
    # This function logs slice progress.
    def slice_progress(self, current_slice: int, total_slices: int, chunk_idx: Optional[int] = None):
        """Log de progreso de slices."""
        percentage = (current_slice / total_slices) * 100 if total_slices > 0 else 0
        chunk_info = f" (chunk {chunk_idx:03d})" if chunk_idx is not None else ""
        self.logger.info(f"{Colors.PROCESSING} Progreso: slice {current_slice:03d}/{total_slices-1:03d} ({percentage:.1f}%){chunk_info}{Colors.ENDC}")
    
    # This function logs detected candidates.
    def candidate_detected(self, dm: float, time: float, confidence: float, class_prob: float, is_burst: bool, snr_raw: float, snr_patch: float):
        """Log de candidato detectado."""
        burst_status = "BURST" if is_burst else "NO burst"
        self.logger.info(f"{Colors.DETECTION} Candidato encontrado: DM={dm:.1f} t={time:.3f}s conf={confidence:.2f} class={class_prob:.2f} → {burst_status}{Colors.ENDC}")
        self.logger.info(f"{Colors.OKCYAN} SNR Raw: {snr_raw:.2f}σ, SNR Patch: {snr_patch:.2f}σ{Colors.ENDC}")
    
    # This function logs slice completion.
    def slice_completed(self, slice_idx: int, candidates: int, bursts: int, no_bursts: int):
        """Log de slice completado."""
        if candidates > 0:
            self.logger.info(f"{Colors.DETECTION} Slice {slice_idx:03d} completado: {candidates} candidatos ({bursts} bursts, {no_bursts} no bursts){Colors.ENDC}")
    
    # This function logs chunk completion.
    def chunk_completed(self, chunk_idx: int, total_candidates: int, total_bursts: int, total_no_bursts: int):
        """Log de chunk completado."""
        self.logger.info(f"{Colors.PROCESSING} Chunk {chunk_idx:03d} completado: {total_candidates} candidatos totales ({total_bursts} bursts, {total_no_bursts} no bursts){Colors.ENDC}")
    
    # This function logs band processing.
    def processing_band(self, band_name: str, slice_idx: int):
        """Log de procesamiento de banda."""
        self.logger.info(f"{Colors.PROCESSING} Procesando {band_name} (slice {slice_idx}){Colors.ENDC}")
    
    # This function logs candidates per band.
    def band_candidates(self, band_name: str, candidate_count: int):
        """Log de candidatos por banda."""
        if candidate_count > 0:
            self.logger.info(f"{Colors.DETECTION} {band_name}: {candidate_count} candidatos detectados{Colors.ENDC}")
        else:
            self.logger.debug(f"{Colors.OKCYAN} {band_name}: Sin candidatos detectados{Colors.ENDC}")
    
    # This function logs waterfall creation.
    def creating_waterfall(self, waterfall_type: str, slice_idx: int, dm: Optional[float] = None):
        """Log de creación de waterfall."""
        dm_info = f" (DM={dm:.1f})" if dm is not None else ""
        self.logger.debug(f"{Colors.OKCYAN} Creando waterfall {waterfall_type} para slice {slice_idx}{dm_info}{Colors.ENDC}")
    
    # This function logs plot generation.
    def generating_plots(self):
        """Log de generación de plots."""
        self.logger.debug(f"{Colors.OKCYAN} Generando plots compuestos y de detección{Colors.ENDC}")


# This function sets up logging.
def setup_logging(level: str = "INFO", log_file: Optional[Path] = None, 
                 use_colors: bool = True) -> DRAFTSLogger:
    """
    Configura el sistema de logging para DRAFTS.
    
    Args:
        level: Nivel de logging ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Archivo para guardar logs (opcional, se crea automáticamente si es None)
        use_colors: Si usar colores en consola
    
    Returns:
        DRAFTSLogger configurado
    """
    return DRAFTSLogger("DRAFTS", level, log_file, use_colors)


# This function retrieves a configured logger.
def get_logger(name: str = "DRAFTS") -> logging.Logger:
    """Obtiene un logger configurado."""
    return logging.getLogger(name)


                      
_global_logger: Optional[DRAFTSLogger] = None

# This function retrieves the global logger.
def get_global_logger() -> DRAFTSLogger:
    """Obtiene el logger global configurado."""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logging()
    return _global_logger

# This function sets the global logger.
def set_global_logger(logger: DRAFTSLogger):
    """Establece el logger global."""
    global _global_logger
    _global_logger = logger 
