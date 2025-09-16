# This module orchestrates the end-to-end FRB processing pipeline.

from __future__ import annotations

                          
import gc
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

                     
import numpy as np

                              
try:
    import torch
except ImportError:
    torch = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

               
from ..config import config
from .detection_engine import process_slice_with_multiple_bands
from .data_flow_manager import (
    build_dm_time_cube,
    create_chunk_directories,
    downsample_chunk,
    get_chunk_processing_parameters,
    plan_slices,
    trim_valid_window,
    validate_slice_indices,
)
from .pipeline_parameters import calculate_absolute_slice_time, calculate_frequency_downsampled
from ..input.parameter_extractor import extract_parameters_auto
from ..input.streaming_orchestrator import get_streaming_function
from ..input.file_finder import find_data_files
from ..logging import (
    log_block_processing,
    log_pipeline_file_completion,
    log_pipeline_file_processing,
    log_processing_summary,
    log_streaming_parameters,
)
from ..output.candidate_manager import ensure_csv_header
from ..output.summary_manager import _update_summary_with_results, _write_summary_with_timestamp

              
logger = logging.getLogger(__name__)


@dataclass
class DetectionStats:
    """Accumulate detection metrics for a chunk or file."""

    n_candidates: int = 0
    n_bursts: int = 0
    n_no_bursts: int = 0
    max_prob: float = 0.0
    snr_values: list[float] = field(default_factory=list)

    # This function updates detection counters.
    def update(self, candidates: int, bursts: int, no_bursts: int, prob_max: float) -> None:
        """Update counters with the result of a slice or chunk."""

        self.n_candidates += candidates
        self.n_bursts += bursts
        self.n_no_bursts += no_bursts
        self.max_prob = max(self.max_prob, float(prob_max))

    # This function merges detection statistics.
    def merge(self, other: "DetectionStats") -> None:
        """Merge metrics coming from another :class:`DetectionStats` instance."""

        self.update(other.n_candidates, other.n_bursts, other.n_no_bursts, other.max_prob)
        if other.snr_values:
            self.snr_values.extend(other.snr_values)

    # This function returns the mean SNR value.
    def mean_snr(self) -> float:
        return float(np.mean(self.snr_values)) if self.snr_values else 0.0

    # This function computes effective candidate counts.
    def effective_counts(self, save_only_burst: bool) -> tuple[int, int, int]:
        """Return counts respecting the SAVE_ONLY_BURST flag."""

        if save_only_burst:
            return self.n_bursts, self.n_bursts, 0
        return self.n_candidates, self.n_bursts, self.n_no_bursts

# This function logs informational messages.
def _trace_info(message: str, *args) -> None:
    try:
        from ..logging.logging_config import get_global_logger
        gl = get_global_logger()
        gl.logger.info(message % args if args else message)
    except Exception:
        logger.info(message, *args)

# This function optimizes memory.
def _optimize_memory(aggressive: bool = False) -> None:
    """Optimiza el uso de memoria del sistema.
    
    Args:
        aggressive: Si True, realiza limpieza más agresiva
    """
                               
    gc.collect()
    
                                                               
    if plt is not None:
        plt.close('all')                                          
    
                                         
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        if aggressive:
                                           
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    
                                                     
    if aggressive:
        time.sleep(0.05)                               
    else:
        time.sleep(0.01)                             


# This function loads detection model.
def _load_detection_model() -> torch.nn.Module:
    """Load the CenterNet model configured in :mod:`config`."""
    if torch is None:
        raise ImportError("torch is required to load models")

    from ..models.ObjectDet.centernet_model import centernet
    model = centernet(model_name=config.MODEL_NAME).to(config.DEVICE)
    state = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

# This function loads class model.
def _load_class_model() -> torch.nn.Module:
    """Load the binary classification model configured in :mod:`config`."""
    if torch is None:
        raise ImportError("torch is required to load models")

    from ..models.BinaryClass.binary_model import BinaryNet
    model = BinaryNet(config.CLASS_MODEL_NAME, num_classes=2).to(config.DEVICE)
    state = torch.load(config.CLASS_MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

# This function processes block.
def _process_block(
    det_model: torch.nn.Module,
    cls_model: torch.nn.Module,
    block: np.ndarray,
    metadata: dict,
    fits_path: Path,
    save_dir: Path,
    chunk_idx: int,
    csv_file: Path,
) -> DetectionStats:
    """Procesa un bloque de datos y retorna estadísticas consolidadas."""

    chunk_samples = int(metadata.get("actual_chunk_size", block.shape[0]))
    total_samples = int(metadata.get("total_samples", 0)) or chunk_samples
    start_sample = int(metadata.get("start_sample", 0))
    end_sample = int(metadata.get("end_sample", start_sample + chunk_samples))

    chunk_start_time_sec = start_sample * config.TIME_RESO
    chunk_duration_sec = chunk_samples * config.TIME_RESO

    logger.info(
        f"INICIANDO CHUNK {chunk_idx:03d}:\n"
        f"   • Muestras en chunk: {chunk_samples:,} / {total_samples:,} totales\n"
        f"   • Rango de muestras: [{start_sample:,} → {end_sample:,}]\n"
        f"   • Tiempo absoluto: {chunk_start_time_sec:.3f}s → {chunk_start_time_sec + chunk_duration_sec:.3f}s\n"
        f"   • Duración del chunk: {chunk_duration_sec:.2f}s\n"
        f"   • Progreso del archivo: {(start_sample / max(total_samples, 1)) * 100:.1f}%"
    )

    logger.info(
        f"Chunk {chunk_idx:03d}: Tiempo {chunk_start_time_sec:.2f}s - {chunk_start_time_sec + chunk_duration_sec:.2f}s "
        f"(duración: {chunk_duration_sec:.2f}s)"
    )

    block, dt_ds = downsample_chunk(block)

    _trace_info(
        "[TRACE] Chunk %03d: tsamp=%.9fs DOWN_TIME_RATE=%dx Δt=%.9fs start_sample_raw=%d end_sample_raw=%d",
        chunk_idx,
        config.TIME_RESO,
        int(config.DOWN_TIME_RATE),
        dt_ds,
        metadata.get("start_sample", -1),
        metadata.get("end_sample", -1),
    )

    chunk_params = get_chunk_processing_parameters(metadata)
    freq_down = chunk_params['freq_down']
    height = chunk_params['height']
    slice_len = chunk_params['slice_len']
    time_slice = chunk_params['time_slice']
    overlap_left_ds = chunk_params['overlap_left_ds']
    overlap_right_ds = chunk_params['overlap_right_ds']

    logger.info(
        "[TRACE] Overlap raw→ds: left=%d→%d (R=%d) right=%d→%d",
        int(metadata.get("overlap_left", 0)),
        overlap_left_ds,
        int(config.DOWN_TIME_RATE),
        int(metadata.get("overlap_right", 0)),
        overlap_right_ds,
    )

    logger.info(f"Chunk {chunk_idx:03d}: {chunk_samples} muestras → {time_slice} slices")

    dm_time_full = build_dm_time_cube(block, height=height, dm_min=config.DM_min, dm_max=config.DM_max)
    block, dm_time, valid_start_ds, valid_end_ds = trim_valid_window(
        block, dm_time_full, overlap_left_ds, overlap_right_ds
    )

    _trace_info(
        "[TRACE] Chunk %03d: valid_start_ds=%d valid_end_ds=%d (N_valid=%d)",
        chunk_idx,
        valid_start_ds,
        valid_end_ds,
        (valid_end_ds - valid_start_ds),
    )

    band_configs = config.get_band_configs()
    chunk_stats = DetectionStats()
    snr_list = chunk_stats.snr_values
    slices_to_process = plan_slices(block, slice_len, chunk_idx)

    for j, start_idx, end_idx in slices_to_process:
        if j % 10 == 0 or j == 0:
            try:
                from ..logging.logging_config import get_global_logger

                global_logger = get_global_logger()
                global_logger.slice_progress(j, time_slice, chunk_idx)
            except ImportError:
                pass

        es_valido, start_idx_ajustado, end_idx_ajustado, razon = validate_slice_indices(
            start_idx, end_idx, block.shape[0], slice_len, j, chunk_idx
        )

        if not es_valido:
            logger.warning(f"SALTANDO SLICE {j} (chunk {chunk_idx}): {razon}")
            continue

        start_idx, end_idx = start_idx_ajustado, end_idx_ajustado

        dt_ds_local = config.TIME_RESO * config.DOWN_TIME_RATE
        slice_abs_start_preview = chunk_start_time_sec + (start_idx * dt_ds_local)
        slice_info = {
            'slice_idx': j,
            'slice_len': slice_len,
            'start_idx': start_idx,
            'end_idx_calculado': end_idx,
            'block_shape': block.shape[0],
            'chunk_idx': chunk_idx,
            'tiempo_absoluto_inicio': slice_abs_start_preview,
            'duracion_slice_esperada_ms': slice_len * config.TIME_RESO * config.DOWN_TIME_RATE * 1000,
        }

        slice_cube = dm_time[:, :, start_idx:end_idx]
        waterfall_block = block[start_idx:end_idx]

        slice_tiempo_real_ms = (end_idx - start_idx) * config.TIME_RESO * config.DOWN_TIME_RATE * 1000
        logger.info(
            f"PROCESANDO SLICE {j} (chunk {chunk_idx}):\n"
            f"   • Rango de muestras: [{start_idx} → {end_idx}] ({end_idx - start_idx} muestras)\n"
            f"   • Tiempo absoluto: {slice_info['tiempo_absoluto_inicio']:.3f}s\n"
            f"   • Duración real: {slice_tiempo_real_ms:.1f}ms\n"
            f"   • Shape del slice: {slice_cube.shape}\n"
            f"   • Datos del waterfall: {waterfall_block.shape}"
        )

        if slice_cube.size == 0 or waterfall_block.size == 0:
            logger.warning(
                f"SALTANDO SLICE {j} (chunk {chunk_idx}) - DATOS VACÍOS:\n"
                f"   • slice_cube.size: {slice_cube.size}\n"
                f"   • waterfall_block.size: {waterfall_block.size}\n"
                f"   • Rango de muestras: [{start_idx} → {end_idx}]\n"
                f"   • Tiempo absoluto: {slice_info['tiempo_absoluto_inicio']:.3f}s\n"
                f"   • Razón: No hay datos útiles para procesar"
            )
            continue

        slice_start_time_sec = calculate_absolute_slice_time(
            chunk_start_time_sec, start_idx, dt_ds
        )

        _trace_info(
            "[TRACE] Slice %03d (chunk %03d): start_idx=%d end_idx=%d N=%d | abs_start=%.9fs abs_end=%.9fs Δt=%.9fs",
            j,
            chunk_idx,
            start_idx,
            end_idx,
            (end_idx - start_idx),
            slice_start_time_sec,
            slice_start_time_sec + (end_idx - start_idx) * dt_ds,
            dt_ds,
        )

        composite_dir, detections_dir, patches_dir = create_chunk_directories(
            save_dir, fits_path, chunk_idx
        )

        cands, bursts, no_bursts, max_prob = process_slice_with_multiple_bands(
            j,
            dm_time,
            block,
            slice_len,
            det_model,
            cls_model,
            fits_path,
            save_dir,
            freq_down,
            csv_file,
            config.TIME_RESO * config.DOWN_TIME_RATE,
            band_configs,
            snr_list,
            config,
            absolute_start_time=slice_start_time_sec,
            composite_dir=composite_dir,
            detections_dir=detections_dir,
            patches_dir=patches_dir,
            chunk_idx=chunk_idx,
            force_plots=config.FORCE_PLOTS,
            slice_start_idx=start_idx,
            slice_end_idx=end_idx,
        )

        chunk_stats.update(cands, bursts, no_bursts, max_prob)

        if j % 10 == 0:
            _optimize_memory(aggressive=False)
        else:
            if plt is not None:
                plt.close('all')
            gc.collect()

    try:
        from ..logging.logging_config import get_global_logger

        global_logger = get_global_logger()
        global_logger.chunk_completed(
            chunk_idx, chunk_stats.n_candidates, chunk_stats.n_bursts, chunk_stats.n_no_bursts
        )
    except ImportError:
        pass

    if not config.SAVE_ONLY_BURST and chunk_stats.n_bursts > 0:
        file_folder_name = fits_path.stem
        chunk_folder_name = f"chunk{chunk_idx:03d}"
        try:
            chunks_with_frbs_dir = save_dir / "Composite" / file_folder_name / "ChunksWithFRBs"
            chunks_with_frbs_dir.mkdir(parents=True, exist_ok=True)

            chunk_dir = save_dir / "Composite" / file_folder_name / chunk_folder_name
            if chunk_dir.exists():
                png_files = list(chunk_dir.glob("*.png"))
                if png_files:
                    destination_dir = chunks_with_frbs_dir / chunk_folder_name
                    if destination_dir.exists():
                        shutil.rmtree(destination_dir)
                    shutil.move(str(chunk_dir), str(destination_dir))
                    logger.info(
                        f"Chunk {chunk_idx:03d} movido a ChunksWithFRBs "
                        f"(contiene {chunk_stats.n_bursts} candidatos BURST)"
                    )
                else:
                    logger.warning(
                        f"Chunk {chunk_idx:03d} tiene {chunk_stats.n_bursts} BURST pero no contiene plots, "
                        "no se moverá a ChunksWithFRBs"
                    )
            else:
                logger.warning(f"Carpeta del chunk {chunk_idx:03d} no existe, no se puede mover")
        except Exception as e:
            logger.error(f"Error moviendo chunk {chunk_idx:03d} a ChunksWithFRBs: {e}")
    elif config.SAVE_ONLY_BURST and chunk_stats.n_bursts > 0:
        logger.info(
            f"Chunk {chunk_idx:03d} contiene {chunk_stats.n_bursts} candidatos BURST "
            f"(SAVE_ONLY_BURST=True, no se reorganiza - todos los chunks con candidatos son chunks con FRBs)"
        )

    return chunk_stats


# This function processes file chunked.
def _process_file_chunked(
    det_model: torch.nn.Module,
    cls_model: torch.nn.Module,
    fits_path: Path,
    save_dir: Path,
    chunk_samples: int,
) -> dict:
    """Procesa un archivo en bloques usando stream_fil o stream_fits."""
    
                                                          
    logger.info(f"Analizando estructura del archivo: {fits_path.name}")
    
                                                                                       
    total_samples = config.FILE_LENG                              
    
    if chunk_samples <= 0:
        raise ValueError("chunk_samples debe ser mayor que cero")

                                                   
    if total_samples <= chunk_samples:
        logger.info(f"Archivo pequeño detectado ({total_samples:,} muestras), "
                    f"procesando en un solo chunk optimizado")
                                                                    
        effective_chunk_samples = total_samples
        chunk_count = 1
        logger.info(f"   • Ajuste automático: chunk_samples = {effective_chunk_samples:,} (archivo completo)")
    else:
        effective_chunk_samples = chunk_samples
        chunk_count = (total_samples + chunk_samples - 1) // chunk_samples                          
        logger.info(f"   • Usando chunking estándar: {chunk_count} chunks")
    
    total_duration_sec = total_samples * config.TIME_RESO                                           
    chunk_duration_sec = effective_chunk_samples * config.TIME_RESO                                       
    
    logger.info(f"RESUMEN DEL ARCHIVO:")
    logger.info(f"   Total de chunks estimado: {chunk_count}")
    logger.info(f"   Muestras totales: {total_samples:,}")
    logger.info(f"   Duración total: {total_duration_sec:.2f} segundos ({total_duration_sec/60:.1f} minutos)")
    logger.info(f"   Tamaño de chunk efectivo: {effective_chunk_samples:,} muestras ({chunk_duration_sec:.2f}s)")
    if total_samples <= chunk_samples: 
        logger.info(f"   Modo optimizado: archivo pequeño procesado en un solo chunk")
    else:
        logger.info(f"   Modo chunking estándar: múltiples chunks")
    logger.info(f"   Iniciando procesamiento...")
    
    csv_file = save_dir / f"{fits_path.stem}.candidates.csv"                       
    ensure_csv_header(csv_file)                                               
    
    t_start = time.time()                                     
    actual_chunk_count = 0                                
    file_stats = DetectionStats()
    
    try:
        if total_samples <= 0:
            raise ValueError(f"Archivo inválido: {total_samples} muestras")
        if total_samples > 1_000_000_000:                      
            logger.warning(f"Archivo muy grande detectado ({total_samples:,} muestras), "
                          f"puede requerir mucho tiempo de procesamiento")
        
                                                               
        try:
            freq_ds = calculate_frequency_downsampled()
        except ValueError as exc:
            logger.warning("No se pudo calcular la decimación de frecuencia (%s). Se utilizará el eje original.", exc)
            if config.FREQ is None or len(config.FREQ) == 0:
                raise
            freq_ds = config.FREQ
        nu_min = float(freq_ds.min())
        nu_max = float(freq_ds.max())
        dt_max_sec = 4.1488e3 * config.DM_max * (nu_min**-2 - nu_max**-2)

        if config.TIME_RESO <= 0:
            logger.warning(f"TIME_RESO inválido ({config.TIME_RESO}), usando valor por defecto para overlap")
            overlap_raw = 1024
        else:
            overlap_raw = max(0, int(np.ceil(dt_max_sec / config.TIME_RESO)))

                                                                        
        streaming_func, file_type = get_streaming_function(fits_path) 
        logger.info(f"DETECTADO: Archivo {file_type.upper()} - {fits_path.name}")
        logger.info(f"Usando streaming {file_type.upper()}: {streaming_func.__name__}")
        
        log_streaming_parameters(effective_chunk_samples, overlap_raw, total_samples, chunk_samples, streaming_func, file_type)
        
                                               
        for block, metadata in streaming_func(str(fits_path), effective_chunk_samples, overlap_samples=overlap_raw):
            actual_chunk_count += 1                                               
            
            log_block_processing(actual_chunk_count, block.shape, str(block.dtype), metadata)
            
            logger.info(f"Procesando chunk {metadata['chunk_idx']:03d} "                       
                       f"({metadata['start_sample']:,} - {metadata['end_sample']:,})")                                      
            
                                                           
            try:
                block_stats = _process_block(
                    det_model,
                    cls_model,
                    block,
                    metadata,
                    fits_path,
                    save_dir,
                    metadata['chunk_idx'],
                    csv_file,
                )
                file_stats.merge(block_stats)
            except Exception as chunk_error:
                logger.exception(f"Error procesando chunk {metadata['chunk_idx']:03d}: {chunk_error}")

                                          
            del block                             
            _optimize_memory(aggressive=(actual_chunk_count % 5 == 0))                                   

                                                        
        log_processing_summary(actual_chunk_count, chunk_count, file_stats.n_candidates, file_stats.n_bursts)

        runtime = time.time() - t_start                      
        logger.info(
            f"Archivo completado: {actual_chunk_count} chunks procesados, "
            f"{file_stats.n_candidates} candidatos, max prob {file_stats.max_prob:.2f}, ⏱️ {runtime:.1f}s"
        )

        n_candidates, n_bursts, n_no_bursts = file_stats.effective_counts(config.SAVE_ONLY_BURST)

        return {
            "n_candidates": n_candidates,
            "n_bursts": n_bursts,
            "n_no_bursts": n_no_bursts,
            "runtime_s": runtime,
            "max_prob": file_stats.max_prob,
            "mean_snr": file_stats.mean_snr,
            "status": "SUCCESS_CHUNKED",
            "chunks_processed": actual_chunk_count,
            "total_chunks": chunk_count,
            "file_size_samples": total_samples,
            "processing_mode": "small_file_optimized" if total_samples <= chunk_samples else "standard_chunking"
        }
        
    except Exception as e:
                                                         
        error_msg = str(e).lower()
        if "corrupted" in error_msg or "invalid" in error_msg or "corrupt" in error_msg:
            status = "ERROR_CORRUPTED_FILE"
            logger.error(f"Archivo corrupto detectado: {fits_path.name} - {e}")
        elif "memory" in error_msg or "out of memory" in error_msg or "oom" in error_msg:
            status = "ERROR_MEMORY"
            logger.error(f"Error de memoria procesando: {fits_path.name} - {e}")
        elif "file not found" in error_msg or "no such file" in error_msg:
            status = "ERROR_FILE_NOT_FOUND"
            logger.error(f"Archivo no encontrado: {fits_path.name} - {e}")
        elif "permission" in error_msg or "access denied" in error_msg:
            status = "ERROR_PERMISSION"
            logger.error(f"Error de permisos: {fits_path.name} - {e}")
        else:
            status = "ERROR_CHUNKED"
            logger.error(f"Error general procesando: {fits_path.name} - {e}")
        
        return {
            "n_candidates": 0,
            "n_bursts": 0,
            "n_no_bursts": 0,
            "runtime_s": time.time() - t_start,
            "max_prob": 0.0,
            "mean_snr": 0.0,
            "status": status,
            "error_details": str(e)
        }

# This function runs pipeline.
def run_pipeline(chunk_samples: int = 0) -> None:
    from ..logging.logging_config import setup_logging, set_global_logger
    
    logger = setup_logging(level="INFO", use_colors=True)                     
    set_global_logger(logger)                              
    
                                
    pipeline_config = {
        'data_dir': str(config.DATA_DIR),                      
        'results_dir': str(config.RESULTS_DIR),                           
        'targets': config.FRB_TARGETS,                     
        'chunk_samples': chunk_samples                                                   
    }
    
    logger.pipeline_start(pipeline_config) 

                                                     
    if config.SAVE_ONLY_BURST:
        logger.logger.info("CONFIGURACIÓN ACTIVADA: SAVE_ONLY_BURST=True")
        logger.logger.info("   → Solo se guardarán y mostrarán candidatos clasificados como BURST")
        logger.logger.info("   → Los candidatos NO BURST serán detectados pero no guardados ni visualizados")
        logger.logger.info("   → NO se reorganizarán chunks a 'ChunksWithFRBs' (todos los chunks con candidatos son chunks con FRBs)")
    else:
        logger.logger.info("CONFIGURACIÓN: SAVE_ONLY_BURST=False")
        logger.logger.info("   → Se guardarán y mostrarán TODOS los candidatos (BURST y NO BURST)")
        logger.logger.info("   → SÍ se reorganizarán chunks con FRBs a 'ChunksWithFRBs' para separar chunks con/sin FRBs")

    save_dir = config.RESULTS_DIR                                                       
    save_dir.mkdir(parents=True, exist_ok=True)                                    
    
    logger.logger.info("Cargando modelos...")
    det_model = _load_detection_model()                                
    cls_model = _load_class_model()                                    
    logger.logger.info("Modelos cargados exitosamente")                          

    summary: dict[str, dict] = {}
    for frb in config.FRB_TARGETS:                                 
        logger.logger.info(f"Buscando archivos para target: {frb}")                              
        file_list = find_data_files(frb)                                                               
        logger.logger.info(f"Archivos encontrados: {[f.name for f in file_list]}")                              
        if not file_list:
            logger.logger.warning(f"No se encontraron archivos para {frb}")
            continue
        try:
                                                                                    
            first_file = file_list[0]
            logger.logger.info(f"Extrayendo parámetros automáticamente desde: {first_file.name}")
            
            extraction_result = extract_parameters_auto(first_file)
            if extraction_result['success']:
                logger.logger.info(f"Parámetros extraídos exitosamente: {', '.join(extraction_result['parameters_extracted'])}")
            else:
                logger.logger.error(f"Error extrayendo parámetros: {', '.join(extraction_result['errors'])}")
                continue
            
                                                                  
            from ..preprocessing.slice_len_calculator import get_processing_parameters, validate_processing_parameters
            from ..logging.chunking_logging import display_detailed_chunking_info
            
            if chunk_samples == 0:                   
                processing_params = get_processing_parameters()                                          
                if validate_processing_parameters(processing_params):                                          
                    chunk_samples = processing_params['chunk_samples']                                           
                                                                           
                    display_detailed_chunking_info(processing_params)                                                        
                else:
                    logger.logger.error("Parámetros calculados inválidos, usando valores por defecto")
                    chunk_samples = 2_097_152                   
            else:
                logger.logger.info(f"Usando chunk_samples manual: {chunk_samples:,}")
                
        except Exception as e:
            logger.logger.error(f"Error obteniendo parámetros: {e}")
            continue
            
        for fits_path in file_list:                                 
            try:
                                                      
                file_info = {
                    'samples': config.FILE_LENG,
                    'duration_min': (config.FILE_LENG * config.TIME_RESO) / 60,
                    'channels': config.FREQ_RESO
                }
                logger.file_processing_start(fits_path.name, file_info) 
                
                log_pipeline_file_processing(fits_path.name, fits_path.suffix.lower(), config.FILE_LENG, chunk_samples) 
                
                results = _process_file_chunked(det_model, cls_model, fits_path, save_dir, chunk_samples)                               
                summary[fits_path.name] = results                                       
                
                log_pipeline_file_completion(fits_path.name, results)
                
                                                                           
                _update_summary_with_results(save_dir, fits_path.stem, {
                    "n_candidates": results.get("n_candidates", 0),
                    "n_bursts": results.get("n_bursts", 0),
                    "n_no_bursts": results.get("n_no_bursts", 0),
                    "processing_time": results.get("runtime_s", 0.0),
                    "max_detection_prob": results.get("max_prob", 0.0),
                    "mean_snr": results.get("mean_snr", 0.0),
                    "status": "completed"
                })
                
                logger.file_processing_end(fits_path.name, results)
            except Exception as e:
                logger.logger.error(f"Error procesando {fits_path.name}: {e}")
                error_results = {
                    "n_candidates": 0,
                    "n_bursts": 0,
                    "n_no_bursts": 0,
                    "runtime_s": 0,
                    "max_prob": 0.0,
                    "mean_snr": 0.0,
                    "status": "ERROR"
                }
                summary[fits_path.name] = error_results
                
                                                                    
                _update_summary_with_results(save_dir, fits_path.stem, {
                    "n_candidates": 0,
                    "n_bursts": 0,
                    "n_no_bursts": 0,
                    "processing_time": 0.0,
                    "max_detection_prob": 0.0,
                    "mean_snr": 0.0,
                    "status": "ERROR",
                    "error_message": str(e)
                })

    logger.logger.info("Escribiendo resumen final...")
    _write_summary_with_timestamp(summary, save_dir)
    logger.pipeline_end(summary)

if __name__ == "__main__":
                                                     
    run_pipeline()
