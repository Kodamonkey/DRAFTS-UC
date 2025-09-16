# This module manages chunk-level data flow operations.

"""Gestor del flujo de datos: chunks, slices y planificación del procesamiento."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..config import config
from ..preprocessing.chunk_planner import plan_slices_for_chunk
from ..preprocessing.dedispersion import d_dm_time_g
from .pipeline_parameters import (
    calculate_dm_height,
    calculate_frequency_downsampled,
    calculate_overlap_decimated,
    calculate_slice_parameters,
    calculate_time_slice,
    calculate_width_total,
)

              
logger = logging.getLogger(__name__)


# This function downsamples chunk.
def downsample_chunk(block: np.ndarray) -> tuple[np.ndarray, float]:
    """Aplica downsampling temporal (suma) y frecuencial (promedio) al chunk completo.

    Args:
        block: Bloque de datos original
        
    Returns:
        tuple[np.ndarray, float]: (block_ds, dt_ds)
            - block_ds: bloque decimado (tiempo, freq)
            - dt_ds: resolución temporal efectiva (s)
    """
    from ..preprocessing.data_downsampler import downsample_data
    block_ds = downsample_data(block)
    dt_ds = config.TIME_RESO * config.DOWN_TIME_RATE
    return block_ds, dt_ds


# This function builds DM time cube.
def build_dm_time_cube(block_ds: np.ndarray, height: int, dm_min: float, dm_max: float) -> np.ndarray:
    """Construye el cubo DM–tiempo para el bloque decimado completo.
    
    Args:
        block_ds: Bloque decimado
        height: Altura del cubo DM
        dm_min: DM mínimo
        dm_max: DM máximo
        
    Returns:
        np.ndarray: Cubo DM-tiempo
    """
    width = block_ds.shape[0]
    from ..preprocessing.dedispersion import d_dm_time_g
    return d_dm_time_g(block_ds, height=height, width=width, dm_min=dm_min, dm_max=dm_max)


# This function trims valid window.
def trim_valid_window(
    block_ds: np.ndarray, 
    dm_time_full: np.ndarray, 
    overlap_left_ds: int, 
    overlap_right_ds: int
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Extrae la ventana válida del bloque, eliminando bordes contaminados por solapamiento.
    
    Args:
        block_ds: Bloque decimado
        dm_time_full: Cubo DM-tiempo completo
        overlap_left_ds: Solapamiento izquierdo decimado
        overlap_right_ds: Solapamiento derecho decimado
        
    Returns:
        tuple[np.ndarray, np.ndarray, int, int]: (block_valid, dm_time, valid_start_ds, valid_end_ds)
    """
    valid_start_ds = max(0, overlap_left_ds)
                                                         
    valid_end_ds = block_ds.shape[0]
    if valid_end_ds <= valid_start_ds:
        valid_start_ds, valid_end_ds = 0, block_ds.shape[0]
    
    dm_time = dm_time_full[:, :, valid_start_ds:valid_end_ds]
    block_valid = block_ds[valid_start_ds:valid_end_ds]
    return block_valid, dm_time, valid_start_ds, valid_end_ds


# This function plans slices.
def plan_slices(block_valid: np.ndarray, slice_len: int, chunk_idx: int) -> list[tuple[int, int, int]]:
    """Genera (j, start_idx, end_idx) por slice para el bloque válido.
    
    Args:
        block_valid: Bloque válido (sin bordes contaminados)
        slice_len: Longitud de cada slice
        chunk_idx: Índice del chunk
        
    Returns:
        list[tuple[int, int, int]]: Lista de tuplas (slice_idx, start_idx, end_idx)
    """
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
            from ..logging.chunking_logging import log_slice_plan_summary
            log_slice_plan_summary(chunk_idx, plan)
        except Exception:
            pass
        return [(idx, sl.start_idx, sl.end_idx) for idx, sl in enumerate(plan["slices"])]
    else:
        time_slice = (block_valid.shape[0] + slice_len - 1) // slice_len
        return [(j, j * slice_len, min((j + 1) * slice_len, block_valid.shape[0])) for j in range(time_slice)]





# This function validates slice indices.
def validate_slice_indices(
    start_idx: int, 
    end_idx: int, 
    block_shape: int, 
    slice_len: int, 
    j: int, 
    chunk_idx: int
) -> tuple[bool, int, int, str]:
    """Valida y ajusta los índices de un slice si es necesario.
    
    Args:
        start_idx: Índice de inicio del slice
        end_idx: Índice de fin del slice
        block_shape: Tamaño del bloque
        slice_len: Longitud esperada del slice
        j: Índice del slice
        chunk_idx: Índice del chunk
        
    Returns:
        tuple[bool, int, int, str]: (es_valido, start_idx_ajustado, end_idx_ajustado, razon)
    """
                                                       
    if start_idx >= block_shape:
        return False, start_idx, end_idx, "Slice fuera de límites - no hay datos que procesar"
    
    if end_idx > block_shape:
                                                       
        end_idx_ajustado = block_shape
        
                                                                 
        if end_idx_ajustado - start_idx < slice_len // 2:
            return False, start_idx, end_idx_ajustado, "Slice demasiado pequeño para procesamiento efectivo"
        
        return True, start_idx, end_idx_ajustado, "Slice ajustado - último slice del chunk con datos residuales"
    
    return True, start_idx, end_idx, "Slice válido"


# This function creates chunk directories.
def create_chunk_directories(
    save_dir: Path, 
    fits_path: Path, 
    chunk_idx: int
) -> tuple[Path, Path, Path]:
    """Crea las carpetas necesarias para un chunk específico.
    
    Args:
        save_dir: Directorio base de guardado
        fits_path: Path del archivo FITS
        chunk_idx: Índice del chunk
        
    Returns:
        tuple[Path, Path, Path]: (composite_dir, detections_dir, patches_dir)
    """
    file_folder_name = fits_path.stem
    chunk_folder_name = f"chunk{chunk_idx:03d}"
    
    composite_dir = save_dir / "Composite" / file_folder_name / chunk_folder_name
    detections_dir = save_dir / "Detections" / file_folder_name / chunk_folder_name
    patches_dir = save_dir / "Patches" / file_folder_name / chunk_folder_name
    
    return composite_dir, detections_dir, patches_dir


# This function gets chunk processing parameters.
def get_chunk_processing_parameters(metadata: dict) -> dict:
    """Obtiene todos los parámetros necesarios para procesar un chunk.
    
    Args:
        metadata: Metadatos del chunk
        
    Returns:
        dict: Parámetros del chunk
    """
                                 
    chunk_samples = int(metadata.get("actual_chunk_size", config.FILE_LENG))
    freq_down = calculate_frequency_downsampled()
    height = calculate_dm_height()
    width_total = calculate_width_total(chunk_samples)
    slice_len, real_duration_ms = calculate_slice_parameters()
    time_slice = calculate_time_slice(width_total, slice_len)
    
                                    
    _ol_raw = int(metadata.get("overlap_left", 0))
    _or_raw = int(metadata.get("overlap_right", 0))
    overlap_left_ds, overlap_right_ds = calculate_overlap_decimated(_ol_raw, _or_raw)
    
    return {
        'freq_down': freq_down,
        'height': height,
        'width_total': width_total,
        'slice_len': slice_len,
        'real_duration_ms': real_duration_ms,
        'time_slice': time_slice,
        'overlap_left_ds': overlap_left_ds,
        'overlap_right_ds': overlap_right_ds,
    }
