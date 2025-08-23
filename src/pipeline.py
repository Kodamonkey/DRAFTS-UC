# =============================================================================
# PIPELINE PRINCIPAL DE DETECCIÓN DE FRB #
#
# Orquesta la carga de modelos, procesamiento de archivos FITS/FIL, y guarda los resultados.
# =============================================================================
"""High level pipeline for FRB detection with CenterNet."""
from __future__ import annotations

import logging
import time
import gc
from pathlib import Path
from typing import List
import shutil

try:
    import torch
except ImportError:  
    torch = None
import numpy as np

# Importar matplotlib al nivel del módulo para evitar problemas de scope
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from . import config
from .input.data_loader import get_obparams, stream_fil, stream_fits, get_obparams_fil
from .output.candidate_manager import ensure_csv_header

from .config import get_band_configs
from .detection_engine import process_slice
from .preprocessing.chunk_planner import plan_slices_for_chunk
from .preprocessing.dedispersion import d_dm_time_g
from .output.summary_manager import (
    _update_summary_with_results,
    _write_summary_with_timestamp, 
)
from .logging import (
    log_streaming_parameters,
    log_block_processing,
    log_processing_summary,
    log_file_detection,
    log_fits_detected,
    log_fil_detected,
    log_unsupported_file_type,
    log_pipeline_file_processing,
    log_pipeline_file_completion
)

logger = logging.getLogger(__name__)

def _trace_info(message: str, *args) -> None:
    try:
        from .logging.logging_config import get_global_logger
        gl = get_global_logger()
        gl.logger.info(message % args if args else message)
    except Exception:
        logger.info(message, *args)

def _optimize_memory(aggressive: bool = False) -> None:
    """Optimiza el uso de memoria del sistema.
    
    Args:
        aggressive: Si True, realiza limpieza más agresiva
    """
    # Limpieza básica de Python
    gc.collect()
    
    # Limpieza de matplotlib para evitar acumulación de figuras
    if plt is not None:
        plt.close('all')  # Cerrar todas las figuras de matplotlib
    
    # Limpieza de CUDA si está disponible
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        if aggressive:
            # Limpieza más agresiva de CUDA
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    
    # Pausa breve para permitir liberación de memoria
    if aggressive:
        time.sleep(0.05)  # 50ms para limpieza agresiva
    else:
        time.sleep(0.01)  # 10ms para limpieza normal


# =============================================================================
# CARGA DE MODELOS DE DETECCIÓN Y CLASIFICACIÓN #
#
# Funciones para cargar los modelos de CenterNet y clasificación binaria desde disco.
# =============================================================================
def _load_detection_model() -> torch.nn.Module:
    """Load the CenterNet model configured in :mod:`config`."""
    if torch is None:
        raise ImportError("torch is required to load models")

    from training.ObjectDet.centernet_model import centernet
    model = centernet(model_name=config.MODEL_NAME).to(config.DEVICE)
    state = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

def _load_class_model() -> torch.nn.Module:
    """Load the binary classification model configured in :mod:`config`."""
    if torch is None:
        raise ImportError("torch is required to load models")

    from training.BinaryClass.binary_model import BinaryNet
    model = BinaryNet(config.CLASS_MODEL_NAME, num_classes=2).to(config.DEVICE)
    state = torch.load(config.CLASS_MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


# Helpers de orquestación

def _downsample_chunk(block: np.ndarray) -> tuple[np.ndarray, float]:
    """Aplica downsampling temporal (suma) y frecuencial (promedio) al chunk completo.

    Returns:
        block_ds: bloque decimado (tiempo, freq)
        dt_ds: resolución temporal efectiva (s)
    """
    from .preprocessing.data_downsampler import downsample_data
    block_ds = downsample_data(block)
    dt_ds = config.TIME_RESO * config.DOWN_TIME_RATE
    return block_ds, dt_ds


def _build_dm_time_cube(block_ds: np.ndarray, height: int, dm_min: float, dm_max: float) -> np.ndarray:
    """Construye el cubo DM–tiempo para el bloque decimado completo."""
    width = block_ds.shape[0]
    from .preprocessing.dedispersion import d_dm_time_g
    return d_dm_time_g(block_ds, height=height, width=width, dm_min=dm_min, dm_max=dm_max)


def _trim_valid_window(block_ds: np.ndarray, dm_time_full: np.ndarray, overlap_left_ds: int, overlap_right_ds: int) -> tuple[np.ndarray, np.ndarray, int, int]:

    valid_start_ds = max(0, overlap_left_ds)
    # Conservar el lado derecho completo para continuidad
    valid_end_ds = block_ds.shape[0]
    if valid_end_ds <= valid_start_ds:
        valid_start_ds, valid_end_ds = 0, block_ds.shape[0]
    dm_time = dm_time_full[:, :, valid_start_ds:valid_end_ds]
    block_valid = block_ds[valid_start_ds:valid_end_ds]
    return block_valid, dm_time, valid_start_ds, valid_end_ds


def _plan_slices(block_valid: np.ndarray, slice_len: int, chunk_idx: int) -> list[tuple[int, int, int]]:
    """Genera (j, start_idx, end_idx) por slice para el bloque válido."""
    if getattr(config, 'USE_PLANNED_CHUNKING', False):
        time_reso_ds = config.TIME_RESO * config.DOWN_TIME_RATE
        plan = plan_slices_for_chunk(
            num_samples_decimated=block_valid.shape[0],
            target_duration_ms=config.SLICE_DURATION_MS,
            time_reso_decimated_s=time_reso_ds,
            max_slice_count=getattr(config, 'MAX_SLICE_COUNT', 5000),
            time_tol_ms=getattr(config, 'TIME_TOL_MS', 0.1),
        )
        try:
            from .logging.chunking_logging import log_slice_plan_summary
            log_slice_plan_summary(chunk_idx, plan)
        except Exception:
            pass
        return [(idx, sl.start_idx, sl.end_idx) for idx, sl in enumerate(plan["slices"])]
    else:
        time_slice = (block_valid.shape[0] + slice_len - 1) // slice_len
        return [(j, j * slice_len, min((j + 1) * slice_len, block_valid.shape[0])) for j in range(time_slice)]


def _absolute_slice_time(chunk_start_time_sec: float, start_idx: int, dt_ds: float) -> float:
    """Calcula el inicio absoluto del slice en segundos desde el inicio del archivo.

    Nota: chunk_start_time_sec ya corresponde al inicio válido del chunk
    (sin solape izquierdo), por lo que NO se suma ningún desplazamiento adicional.
    """
    return chunk_start_time_sec + (start_idx * dt_ds)


def _process_block(
    det_model: torch.nn.Module, # Modelo de detección
    cls_model: torch.nn.Module, # Modelo de clasificación
    block: np.ndarray, # Bloque de datos
    metadata: dict, # Metadatos del bloque
    fits_path: Path, # Path del archivo FITS
    save_dir: Path, # Path de guardado
    chunk_idx: int, # Índice del chunk
    csv_file: Path,  # CSV file por archivo
) -> dict: 
    """Procesa un bloque de datos y retorna estadísticas del bloque."""
    
    original_file_leng = config.FILE_LENG # Tamaño original del archivo
    
    config.FILE_LENG = metadata["actual_chunk_size"] # Tamaño del chunk
    
    chunk_start_time_sec = metadata["start_sample"] * config.TIME_RESO # Tiempo de inicio del chunk
    chunk_duration_sec = metadata["actual_chunk_size"] * config.TIME_RESO # Duración del chunk
    
    # =============================================================================
    # LOGGING INFORMATIVO DETALLADO DEL CHUNK
    # =============================================================================
    logger.info(
        f"INICIANDO CHUNK {chunk_idx:03d}:\n"
        f"   • Muestras en chunk: {metadata['actual_chunk_size']:,} / {metadata['total_samples']:,} totales\n"
        f"   • Rango de muestras: [{metadata['start_sample']:,} → {metadata['end_sample']:,}]\n"
        f"   • Tiempo absoluto: {chunk_start_time_sec:.3f}s → {chunk_start_time_sec + chunk_duration_sec:.3f}s\n"
        f"   • Duración del chunk: {chunk_duration_sec:.2f}s\n"
        f"   • Progreso del archivo: {(metadata['start_sample'] / metadata['total_samples']) * 100:.1f}%"
    )
    
    logger.info(f"Chunk {chunk_idx:03d}: Tiempo {chunk_start_time_sec:.2f}s - {chunk_start_time_sec + chunk_duration_sec:.2f}s "
               f"(duración: {chunk_duration_sec:.2f}s)")
    
    try:
        # Aplicar downsampling al bloque
        block, dt_ds = _downsample_chunk(block)

        dt_raw = config.TIME_RESO
        _trace_info(
            "[TRACE] Chunk %03d: tsamp=%.9fs DOWN_TIME_RATE=%dx Δt=%.9fs start_sample_raw=%d end_sample_raw=%d",
            chunk_idx, dt_raw, int(config.DOWN_TIME_RATE), dt_ds,
            metadata.get("start_sample", -1), metadata.get("end_sample", -1)
        )
        
        # Calcular parámetros para este bloque
        freq_down = np.mean(
            config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE),
            axis=1,
        )
        
        height = config.DM_max - config.DM_min + 1

        width_total = config.FILE_LENG // config.DOWN_TIME_RATE

        # Calcular solapamiento equivalente en muestras decimadas
        # IMPORTANTE: usar ceil para no incluir bins decimados contaminados por el solape parcial
        _ol_raw = int(metadata.get("overlap_left", 0))
        _or_raw = int(metadata.get("overlap_right", 0))
        R = int(config.DOWN_TIME_RATE)
        overlap_left_ds = (_ol_raw + R - 1) // R
        overlap_right_ds = (_or_raw + R - 1) // R
        # TRACE: mostrar cómo se mapean los solapes
        logger.info(
            "[TRACE] Overlap raw→ds: left=%d→%d (R=%d) right=%d→%d",
            _ol_raw, overlap_left_ds, R, _or_raw, overlap_right_ds
        )
        
        # Calcular slice_len dinámicamente
        from .preprocessing.slice_len_calculator import update_slice_len_dynamic # Importar la función de actualización de slice_len
        slice_len, real_duration_ms = update_slice_len_dynamic() # Actualiza slice_len según la configuración
        time_slice = (width_total + slice_len - 1) // slice_len # Número de slices por chunk
        
        logger.info(f"Chunk {chunk_idx:03d}: {metadata['actual_chunk_size']} muestras → {time_slice} slices")
        
        # Generar DM-time cube del bloque decimado completo
        dm_time_full = _build_dm_time_cube(block, height=height, dm_min=config.DM_min, dm_max=config.DM_max)

        # Extraer ventana válida (sin bordes inválidos) usando los solapes decimados
        block, dm_time, valid_start_ds, valid_end_ds = _trim_valid_window(
            block, dm_time_full, overlap_left_ds, overlap_right_ds
        )

        _trace_info(
            "[TRACE] Chunk %03d: valid_start_ds=%d valid_end_ds=%d (N_valid=%d)",
            chunk_idx, valid_start_ds, valid_end_ds, (valid_end_ds - valid_start_ds)
        )
        
        # Configurar bandas
        band_configs = get_band_configs() # Configuración de bandas (fullband, lowband, highband)
        
        # Procesar slices
        cand_counter = 0 # Contador de candidatos
        n_bursts = 0 # Contador de candidatos de tipo burst
        n_no_bursts = 0 # Contador de candidatos de tipo no burst
        prob_max = 0.0 # Probabilidad máxima de detección
        snr_list = [] # Lista de SNRs de los candidatos
        
        # Plan de slices por chunk con contigüidad y ajuste divisor
        time_slice = (config.FILE_LENG // config.DOWN_TIME_RATE + slice_len - 1) // slice_len
        slices_to_process = _plan_slices(block, slice_len, chunk_idx)

        for j, start_idx, end_idx in slices_to_process: # Procesar cada slice
            # Mensaje de progreso cada 10 slices o en el primer slice
            if j % 10 == 0 or j == 0:
                try:
                    from .logging.logging_config import get_global_logger
                    global_logger = get_global_logger()
                    global_logger.slice_progress(j, time_slice, chunk_idx)
                except ImportError:
                    pass
            
            # Verificar que tenemos suficientes datos para este slice (índices ya calculados)

            # =============================================================================
            # LOGGING INFORMATIVO DETALLADO PARA AJUSTES Y SALTOS
            # =============================================================================
            
            # Información base del slice (tiempo absoluto calculado con recorte válido)
            dt_ds_local = config.TIME_RESO * config.DOWN_TIME_RATE
            # El índice 0 de 'block' (tras recorte izquierdo) corresponde exactamente a chunk_start_time_sec
            slice_abs_start_preview = chunk_start_time_sec + (start_idx * dt_ds_local)
            slice_info = {
                'slice_idx': j,
                'slice_len': slice_len,
                'start_idx': start_idx,
                'end_idx_calculado': end_idx,
                'block_shape': block.shape[0],
                'chunk_idx': chunk_idx,
                'tiempo_absoluto_inicio': slice_abs_start_preview,
                'duracion_slice_esperada_ms': slice_len * config.TIME_RESO * config.DOWN_TIME_RATE * 1000
            }

            # Verificar que no excedemos los límites del bloque
            if start_idx >= block.shape[0]: # Si el índice de inicio es mayor o igual al tamaño del bloque
                logger.warning(
                    f"SALTANDO SLICE {j} (chunk {chunk_idx}):\n"
                    f"   • start_idx ({start_idx}) >= block.shape[0] ({block.shape[0]})\n"
                    f"   • Slice fuera de límites - no hay datos que procesar\n"
                    f"   • Tiempo absoluto: {slice_info['tiempo_absoluto_inicio']:.3f}s\n"
                    f"   • Duración esperada: {slice_info['duracion_slice_esperada_ms']:.1f}ms\n"
                    f"   • Datos disponibles en bloque: {block.shape[0]} muestras\n"
                    f"   • Razón: El slice empieza después del final del bloque"
                )
                continue

            if end_idx > block.shape[0]: # Si el índice de fin es mayor al tamaño del bloque
                # Calcular información antes del ajuste
                muestras_esperadas = end_idx - start_idx
                muestras_disponibles = block.shape[0] - start_idx
                porcentaje_ajuste = ((end_idx - block.shape[0]) / (end_idx - start_idx)) * 100
                
                logger.warning(
                    f"AJUSTANDO SLICE {j} (chunk {chunk_idx}):\n"
                    f"   • end_idx calculado ({end_idx}) > block.shape[0] ({block.shape[0]})\n"
                    f"   • Muestras esperadas: {muestras_esperadas}\n"
                    f"   • Muestras disponibles: {muestras_disponibles}\n"
                    f"   • Ajuste necesario: {end_idx - block.shape[0]} muestras ({porcentaje_ajuste:.1f}%)\n"
                    f"   • Tiempo absoluto: {slice_info['tiempo_absoluto_inicio']:.3f}s\n"
                    f"   • Duración esperada: {slice_info['duracion_slice_esperada_ms']:.1f}ms\n"
                    f"   • Razón: Último slice del chunk con datos residuales"
                )
                
                end_idx = block.shape[0] # Ajustar el índice de fin al tamaño del bloque
                
                # Si el slice es muy pequeño, saltarlo
                if end_idx - start_idx < slice_len // 2:
                    logger.warning(
                        f"SALTANDO SLICE {j} (chunk {chunk_idx}) - MUY PEQUEÑO:\n"
                        f"   • Tamaño después del ajuste: {end_idx - start_idx} muestras\n"
                        f"   • Tamaño mínimo requerido: {slice_len // 2} muestras\n"
                        f"   • Porcentaje del tamaño esperado: {((end_idx - start_idx) / slice_len) * 100:.1f}%\n"
                        f"   • Tiempo absoluto: {slice_info['tiempo_absoluto_inicio']:.3f}s\n"
                        f"   • Razón: Slice demasiado pequeño para procesamiento efectivo"
                    )
                    continue

            # Procesamiento del slice
            
            slice_cube = dm_time[:, :, start_idx : end_idx] # Cubo DM-time para este slice
            waterfall_block = block[start_idx : end_idx] # Bloque de datos para este slice

            # Log informativo del slice que se va a procesar
            slice_tiempo_real_ms = (end_idx - start_idx) * config.TIME_RESO * config.DOWN_TIME_RATE * 1000
            logger.info(
                f"PROCESANDO SLICE {j} (chunk {chunk_idx}):\n"
                f"   • Rango de muestras: [{start_idx} → {end_idx}] ({end_idx - start_idx} muestras)\n"
                f"   • Tiempo absoluto: {slice_info['tiempo_absoluto_inicio']:.3f}s\n"
                f"   • Duración real: {slice_tiempo_real_ms:.1f}ms\n"
                f"   • Shape del slice: {slice_cube.shape}\n"
                f"   • Datos del waterfall: {waterfall_block.shape}"
            )

            if slice_cube.size == 0 or waterfall_block.size == 0: # Si el cubo o el bloque están vacíos
                logger.warning(
                    f"SALTANDO SLICE {j} (chunk {chunk_idx}) - DATOS VACÍOS:\n"
                    f"   • slice_cube.size: {slice_cube.size}\n"
                    f"   • waterfall_block.size: {waterfall_block.size}\n"
                    f"   • Rango de muestras: [{start_idx} → {end_idx}]\n"
                    f"   • Tiempo absoluto: {slice_info['tiempo_absoluto_inicio']:.3f}s\n"
                    f"   • Razón: No hay datos útiles para procesar"
                )
                continue

            # Calcular tiempo absoluto para este slice específico
            # Importante: tras recorte izquierdo, el índice 0 de 'block' corresponde a chunk_start_time_sec
            slice_start_time_sec = _absolute_slice_time(
                chunk_start_time_sec, start_idx, dt_ds
            )

            _trace_info(
                "[TRACE] Slice %03d (chunk %03d): start_idx=%d end_idx=%d N=%d | abs_start=%.9fs abs_end=%.9fs Δt=%.9fs",
                j, chunk_idx, start_idx, end_idx, (end_idx - start_idx),
                slice_start_time_sec,
                slice_start_time_sec + (end_idx - start_idx) * dt_ds,
                dt_ds,
            )

            # Crear carpeta principal del archivo
            file_folder_name = fits_path.stem
            chunk_folder_name = f"chunk{chunk_idx:03d}"
            

            waterfall_dispersion_dir = save_dir / "waterfall_dispersion" / file_folder_name / chunk_folder_name
            waterfall_dedispersion_dir = save_dir / "waterfall_dedispersion" / file_folder_name / chunk_folder_name

            # Estructura de carpetas para plots (se crearán solo si hay candidatos)
            # Results/ObjectDetection/Composite/3096_0001_00_8bit/chunk000/
            composite_dir = save_dir / "Composite" / file_folder_name / chunk_folder_name
            detections_dir = save_dir / "Detections" / file_folder_name / chunk_folder_name
            patches_dir = save_dir / "Patches" / file_folder_name / chunk_folder_name

            # Pass chunked paths to process_slice (these will be used in plot_manager)
            cands, bursts, no_bursts, max_prob = process_slice( # Procesar el slice
                j, dm_time, block, slice_len, det_model, cls_model, fits_path, save_dir,
                freq_down, csv_file, config.TIME_RESO * config.DOWN_TIME_RATE, band_configs,
                snr_list, waterfall_dispersion_dir, waterfall_dedispersion_dir, config,
                absolute_start_time=slice_start_time_sec,
                composite_dir=composite_dir,
                detections_dir=detections_dir,
                patches_dir=patches_dir,
                chunk_idx=chunk_idx,
                force_plots=config.FORCE_PLOTS,
                slice_start_idx=start_idx,
                slice_end_idx=end_idx,
            )

            cand_counter += cands
            n_bursts += bursts
            n_no_bursts += no_bursts
            prob_max = max(prob_max, max_prob)
            
            # Evitar log duplicado: el resumen de slice se registra dentro de detection_engine.process_slice

            # LIMPIEZA DE MEMORIA DESPUÉS DE CADA SLICE
            if j % 10 == 0:  # Cada 10 slices
                _optimize_memory(aggressive=False)
            else:
                # Limpieza básica después de cada slice
                if plt is not None:
                    plt.close('all')
                gc.collect()

        # Mensaje de resumen del chunk
        try:
            from .logging.logging_config import get_global_logger
            global_logger = get_global_logger()
            global_logger.chunk_completed(chunk_idx, cand_counter, n_bursts, n_no_bursts)
        except ImportError:
            pass
        
        # =============================================================================
        # REORGANIZACIÓN DE CHUNKS CON FRBs
        # =============================================================================
        # Si este chunk contiene al menos un candidato BURST, moverlo a ChunksWithFRBs
        if n_bursts > 0:
            try:
                # Crear la carpeta ChunksWithFRBs si no existe
                chunks_with_frbs_dir = save_dir / "Composite" / file_folder_name / "ChunksWithFRBs"
                chunks_with_frbs_dir.mkdir(parents=True, exist_ok=True)
                
                # Verificar si la carpeta del chunk existe y contiene plots
                chunk_dir = save_dir / "Composite" / file_folder_name / chunk_folder_name
                if chunk_dir.exists():
                    # Verificar si hay al menos un plot en la carpeta
                    png_files = list(chunk_dir.glob("*.png"))
                    if png_files:
                        # Mover la carpeta completa del chunk a ChunksWithFRBs
                        destination_dir = chunks_with_frbs_dir / chunk_folder_name
                        
                        # Si la carpeta de destino ya existe, eliminarla primero
                        if destination_dir.exists():
                            shutil.rmtree(destination_dir)
                        
                        # Mover la carpeta
                        shutil.move(str(chunk_dir), str(destination_dir))
                        
                        logger.info(f"Chunk {chunk_idx:03d} movido a ChunksWithFRBs "
                                  f"(contiene {n_bursts} candidatos BURST)")
                        

                    else:
                        logger.warning(f"Chunk {chunk_idx:03d} tiene {n_bursts} BURST pero no contiene plots, "
                                     f"no se moverá a ChunksWithFRBs")
                else:
                    logger.warning(f"Carpeta del chunk {chunk_idx:03d} no existe, no se puede mover")
                    
            except Exception as e:
                logger.error(f"Error moviendo chunk {chunk_idx:03d} a ChunksWithFRBs: {e}")
        
        return {
            "n_candidates": cand_counter,
            "n_bursts": n_bursts,
            "n_no_bursts": n_no_bursts,
            "max_prob": prob_max,
            "mean_snr": float(np.mean(snr_list)) if snr_list else 0.0,
            "time_slice": time_slice,
        }
        
    finally:
        # Restaurar configuración original
        config.FILE_LENG = original_file_leng


def _process_file_chunked(
    det_model: torch.nn.Module,
    cls_model: torch.nn.Module,
    fits_path: Path,
    save_dir: Path,
    chunk_samples: int,
) -> dict:
    """Procesa un archivo en bloques usando stream_fil o stream_fits."""
    
    # CALCULAR INFORMACIÓN DEL ARCHIVO DE MANERA EFICIENTE
    logger.info(f"Analizando estructura del archivo: {fits_path.name}")
    
    # Calcular información basada en config.FILE_LENG (ya cargado por get_obparams_fil)
    total_samples = config.FILE_LENG # Longitud total del archivo 
    
    # DETECTAR ARCHIVOS PEQUEÑOS Y AJUSTAR CHUNKING
    if total_samples <= chunk_samples:
        logger.info(f"Archivo pequeño detectado ({total_samples:,} muestras), "
                    f"procesando en un solo chunk optimizado")
        # Para archivos pequeños, usar el tamaño completo como chunk
        effective_chunk_samples = total_samples
        chunk_count = 1
        logger.info(f"   • Ajuste automático: chunk_samples = {effective_chunk_samples:,} (archivo completo)")
    else:
        effective_chunk_samples = chunk_samples
        chunk_count = (total_samples + chunk_samples - 1) // chunk_samples  # Redondear hacia arriba
        logger.info(f"   • Usando chunking estándar: {chunk_count} chunks")
    
    total_duration_sec = total_samples * config.TIME_RESO # Duración total del archivo (en segundos)
    chunk_duration_sec = effective_chunk_samples * config.TIME_RESO # Duración de cada chunk (en segundos)
    
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
    
    csv_file = save_dir / f"{fits_path.stem}.candidates.csv" # CSV file por archivo
    ensure_csv_header(csv_file) # Asegurar que el header del CSV esté presente
    
    t_start = time.time() # Tiempo de inicio del procesamiento 
    cand_counter_total = 0 # Contador de candidatos totales
    n_bursts_total = 0 # Contador de candidatos de tipo burst
    n_no_bursts_total = 0 # Contador de candidatos de tipo no burst
    prob_max_total = 0.0 # Probabilidad máxima de detección
    snr_list_total = [] # Lista de SNRs de los candidatos
    actual_chunk_count = 0 # Contador de chunks procesados
    
    try:
        if total_samples <= 0:
            raise ValueError(f"Archivo inválido: {total_samples} muestras")
        if total_samples > 1_000_000_000:  # 1B muestras = ~1TB
            logger.warning(f"Archivo muy grande detectado ({total_samples:,} muestras), "
                          f"puede requerir mucho tiempo de procesamiento")
        
        # Calcular Δt_max para determinar solapamiento en bruto
        freq_ds = np.mean(
            config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE),
            axis=1,
        )
        nu_min = float(freq_ds.min()) # Frecuencia mínima
        nu_max = float(freq_ds.max()) # Frecuencia máxima
        # Usamos constante para frecuencias en MHz: Δt_sec = 4.1488e3 * DM * (ν_min^-2 − ν_max^-2)
        dt_max_sec = 4.1488e3 * config.DM_max * (nu_min**-2 - nu_max**-2) # Tiempo máximo de solapamiento (en segundos)
        overlap_raw = int(np.ceil(dt_max_sec / config.TIME_RESO)) # Tiempo de solapamiento (en muestras)

        # Obtener función de streaming apropiada para el tipo de archivo
        streaming_func, file_type = _get_streaming_function(fits_path) 
        logger.info(f"DETECTADO: Archivo {file_type.upper()} - {fits_path.name}")
        logger.info(f"Usando streaming {file_type.upper()}: {streaming_func.__name__}")
        
        log_streaming_parameters(effective_chunk_samples, overlap_raw, total_samples, chunk_samples, streaming_func, file_type)
        
        # Procesar cada bloque con solapamiento
        for block, metadata in streaming_func(str(fits_path), effective_chunk_samples, overlap_samples=overlap_raw):
            actual_chunk_count += 1 # Incrementar el contador de chunks procesados
            
            log_block_processing(actual_chunk_count, block.shape, str(block.dtype), metadata)
            
            logger.info(f"Procesando chunk {metadata['chunk_idx']:03d} " # Log del chunk actual
                       f"({metadata['start_sample']:,} - {metadata['end_sample']:,})") # Log del rango de muestras del chunk
            
            # Procesar bloque con manejo de errores robusto
            try:
                block_results = _process_block(  
                    det_model, cls_model, block, metadata, # Modelos de detección y clasificación
                    fits_path, save_dir, metadata['chunk_idx'], csv_file 
                ) 
                
                # Acumular resultados
                cand_counter_total += block_results["n_candidates"] # Acumular el número de candidatos
                n_bursts_total += block_results["n_bursts"] # Acumular el número de candidatos de tipo burst
                n_no_bursts_total += block_results["n_no_bursts"] # Acumular el número de candidatos de tipo no burst
                prob_max_total = max(prob_max_total, block_results["max_prob"]) # Actualizar la probabilidad máxima de detección
                
                # Acumular SNR si está disponible
                if "mean_snr" in block_results and block_results["mean_snr"] > 0:
                    snr_list_total.append(block_results["mean_snr"])
                
            except Exception as chunk_error:
                # Log con traceback completo para ubicar la línea exacta del fallo
                logger.exception(f"Error procesando chunk {metadata['chunk_idx']:03d}: {chunk_error}")
                # Continuar con el siguiente chunk en lugar de fallar completamente
                block_results = {
                    "n_candidates": 0,
                    "n_bursts": 0,
                    "n_no_bursts": 0,
                    "max_prob": 0.0,
                    "mean_snr": 0.0,
                    "time_slice": 0,
                }
                # Acumular resultados de error (ceros)
                cand_counter_total += 0
                n_bursts_total += 0
                n_no_bursts_total += 0
                prob_max_total = max(prob_max_total, 0.0)
            
            # LIMPIEZA AGRESIVA DE MEMORIA
            del block # Liberar memoria del bloque
            _optimize_memory(aggressive=(actual_chunk_count % 5 == 0))  # Limpieza agresiva cada 5 chunks
        
        # *** DEBUG CRÍTICO: CONFIRMAR RESUMEN FINAL ***
        log_processing_summary(actual_chunk_count, chunk_count, cand_counter_total, n_bursts_total)
        
        runtime = time.time() - t_start # Tiempo de ejecución
        logger.info(
            f"Archivo completado: {actual_chunk_count} chunks procesados, "
            f"{cand_counter_total} candidatos, max prob {prob_max_total:.2f}, ⏱️ {runtime:.1f}s"
        )
        
        return {
            "n_candidates": cand_counter_total,
            "n_bursts": n_bursts_total,
            "n_no_bursts": n_no_bursts_total,
            "runtime_s": runtime,
            "max_prob": prob_max_total,
            "mean_snr": float(np.mean(snr_list_total)) if snr_list_total else 0.0,
            "status": "SUCCESS_CHUNKED",
            "chunks_processed": actual_chunk_count,
            "total_chunks": chunk_count,
            "file_size_samples": total_samples,
            "processing_mode": "small_file_optimized" if total_samples <= chunk_samples else "standard_chunking"
        }
        
    except Exception as e:
        # Determinar tipo de error para mejor diagnóstico
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


def _get_streaming_function(file_path: Path):
    """
    Detecta automáticamente el tipo de archivo y retorna la función de streaming apropiada.
    
    Args:
        file_path: Path al archivo a procesar
        
    Returns:
        Tuple[streaming_function, file_type]: Función de streaming y tipo de archivo
    """
    # *** DEBUG CRÍTICO: CONFIRMAR DETECCIÓN DE TIPO DE ARCHIVO ***
    log_file_detection(str(file_path), file_path.suffix.lower(), str(file_path))
    
    if str(file_path).endswith('.fits'):
        log_fits_detected(str(file_path))
        return stream_fits, "fits"
    elif str(file_path).endswith('.fil'):
        log_fil_detected(str(file_path))
        return stream_fil, "fil"
    else:
        log_unsupported_file_type(str(file_path))
        raise ValueError(f"Tipo de archivo no soportado: {file_path}. Solo se soportan .fits y .fil")


# =============================================================================
# BÚSQUEDA DE ARCHIVOS DE DATOS #
#
# Encuentra archivos FITS o FIL que coincidan con el target FRB en el directorio de datos.
# =============================================================================
def _find_data_files(frb: str) -> List[Path]:
    """Return FITS or filterbank files matching ``frb`` within ``config.DATA_DIR``."""

    files = list(config.DATA_DIR.glob("*.fits")) + list(config.DATA_DIR.glob("*.fil"))
    return sorted(f for f in files if frb in f.name)


# =============================================================================
# EJECUCIÓN DEL PIPELINE COMPLETO #
#
# Controla el flujo principal: carga modelos, busca archivos, procesa cada archivo y guarda el resumen global.
# =============================================================================
def run_pipeline(chunk_samples: int = 0) -> None:
    from .logging.logging_config import setup_logging, set_global_logger
    
    logger = setup_logging(level="INFO", use_colors=True) # Configurar logging
    set_global_logger(logger) # Establecer el logger global
    
    # Configuración del pipeline
    pipeline_config = {
        'data_dir': str(config.DATA_DIR), # Directorio de datos
        'results_dir': str(config.RESULTS_DIR), # Directorio de resultados
        'targets': config.FRB_TARGETS, # Targets a procesar
        'chunk_samples': chunk_samples # Número de muestras por bloque para archivos .fil
    }
    
    logger.pipeline_start(pipeline_config) 

    save_dir = config.RESULTS_DIR / config.MODEL_NAME # Directorio de resultados
    save_dir.mkdir(parents=True, exist_ok=True) # Crear el directorio de resultados
    
    logger.logger.info("Cargando modelos...")
    det_model = _load_detection_model() # Cargar el modelo de detección
    cls_model = _load_class_model() # Cargar el modelo de clasificación
    logger.logger.info("Modelos cargados exitosamente") # Log de carga de modelos

    summary: dict[str, dict] = {}
    for frb in config.FRB_TARGETS: # Procesar cada archivo de datos
        logger.logger.info(f"Buscando archivos para target: {frb}") # Log de búsqueda de archivos
        file_list = _find_data_files(frb) # Buscar archivos de datos
        logger.logger.info(f"Archivos encontrados: {[f.name for f in file_list]}") # Log de archivos encontrados
        if not file_list:
            logger.logger.warning(f"No se encontraron archivos para {frb}")
            continue

        try:
            # Obtener los parámetros del archivo
            first_file = file_list[0] # Obtener el primer archivo de la lista
            logger.logger.info(f"Leyendo parámetros desde: {first_file.name}") # Log de lectura de parámetros
            if first_file.suffix.lower() == ".fits":
                get_obparams(str(first_file)) # Obtener los parámetros del archivo
            else:
                get_obparams_fil(str(first_file)) # Obtener los parámetros del archivo
            logger.logger.info("Parámetros de observación cargados")
            
            # CALCULAR PARÁMETROS DE PROCESAMIENTO AUTOMÁTICAMENTE
            from .preprocessing.slice_len_calculator import get_processing_parameters, validate_processing_parameters
            from .logging.chunking_logging import display_detailed_chunking_info
            
            if chunk_samples == 0:  # Modo automático
                processing_params = get_processing_parameters() # Obtener los parámetros de procesamiento
                if validate_processing_parameters(processing_params): # Validar los parámetros de procesamiento
                    chunk_samples = processing_params['chunk_samples'] # Obtener el número de muestras por bloque
                    # Mostrar información detallada del sistema de chunking
                    display_detailed_chunking_info(processing_params) # Mostrar información detallada del sistema de chunking
                else:
                    logger.logger.error("Parámetros calculados inválidos, usando valores por defecto")
                    chunk_samples = 2_097_152  # 2MB por defecto
            else:
                logger.logger.info(f"Usando chunk_samples manual: {chunk_samples:,}")
                
        except Exception as e:
            logger.logger.error(f"Error obteniendo parámetros: {e}")
            continue
            
        for fits_path in file_list: # Procesar cada archivo de datos
            try:
                # Información del archivo para logging
                file_info = {
                    'samples': config.FILE_LENG,
                    'duration_min': (config.FILE_LENG * config.TIME_RESO) / 60,
                    'channels': config.FREQ_RESO
                }
                logger.file_processing_start(fits_path.name, file_info) 
                
                log_pipeline_file_processing(fits_path.name, fits_path.suffix.lower(), config.FILE_LENG, chunk_samples) 
                
                results = _process_file_chunked(det_model, cls_model, fits_path, save_dir, chunk_samples) # Procesar el archivo de datos
                summary[fits_path.name] = results # Guardar los resultados en el resumen
                
                log_pipeline_file_completion(fits_path.name, results)
                
                # *** GUARDAR RESULTADOS INMEDIATAMENTE EN SUMMARY.JSON ***
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
                
                # *** GUARDAR RESULTADOS DE ERROR INMEDIATAMENTE ***
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
    import argparse

    parser = argparse.ArgumentParser(description="Run FRB detection pipeline")
    parser.add_argument("--data-dir", type=Path, help="Directory with FITS files")
    parser.add_argument("--results-dir", type=Path, help="Directory for results")
    parser.add_argument("--det-model", type=Path, help="Detection model path")
    parser.add_argument("--class-model", type=Path, help="Classification model path")
    parser.add_argument("--chunk-samples", type=int, default=0, 
                       help="Número de muestras por bloque para archivos .fil (0 = modo antiguo)")
    args = parser.parse_args()

    if args.data_dir:
        config.DATA_DIR = args.data_dir
    if args.results_dir:
        config.RESULTS_DIR = args.results_dir
    if args.det_model:
        config.MODEL_PATH = args.det_model
    if args.class_model:
        config.CLASS_MODEL_PATH = args.class_model

    run_pipeline(chunk_samples=args.chunk_samples)
   
