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
from .input.data_loader import load_and_preprocess_data, get_obparams, stream_fil, get_obparams_fil
from .output.candidate_manager import ensure_csv_header
from .preprocessing.dedispersion import d_dm_time_g
from .config import get_band_configs
from .detection_engine import get_pipeline_parameters, process_slice
from .output.summary_manager import (
    _update_summary_with_file_debug,
    _update_summary_with_results,
    _write_summary_with_timestamp,  # NUEVA FUNCIÓN
)
from .detection.model_interface import detect, classify_patch
logger = logging.getLogger(__name__)


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
    
    # Configurar parámetros temporales para este bloque
    original_file_leng = config.FILE_LENG # Guardar longitud original del archivo
    config.FILE_LENG = metadata["actual_chunk_size"] # Actualizar longitud del archivo al tamaño del bloque actual
    
    # CALCULAR TIEMPO ABSOLUTO DESDE INICIO DEL ARCHIVO
    # Tiempo de inicio del chunk en segundos desde el inicio del archivo
    chunk_start_time_sec = metadata["start_sample"] * config.TIME_RESO # Tiempo de inicio del chunk en segundos desde el inicio del archivo
    chunk_duration_sec = metadata["actual_chunk_size"] * config.TIME_RESO # Duración del chunk en segundos
    
    logger.info(f"🕐 Chunk {chunk_idx:03d}: Tiempo {chunk_start_time_sec:.2f}s - {chunk_start_time_sec + chunk_duration_sec:.2f}s "
               f"(duración: {chunk_duration_sec:.2f}s)")
    
    try:
        # Aplicar downsampling al bloque
        from .preprocessing.data_downsampler import downsample_data # Importar la función de downsampling
        block = downsample_data(block) # Aplica downsampling según la configuración 
        
        # Calcular parámetros para este bloque
        freq_down = np.mean(
            config.FREQ.reshape(config.FREQ_RESO // config.DOWN_FREQ_RATE, config.DOWN_FREQ_RATE),
            axis=1,
        ) # Promedio de frecuencias después del downsampling
        
        height = config.DM_max - config.DM_min + 1 # Altura del cubo DM
        width_total = config.FILE_LENG // config.DOWN_TIME_RATE # Ancho total del cubo DM-time
        
        # Calcular slice_len dinámicamente
        from .preprocessing.slice_len_calculator import update_slice_len_dynamic # Importar la función de actualización de slice_len
        slice_len, real_duration_ms = update_slice_len_dynamic() # Actualiza slice_len según la configuración
        time_slice = (width_total + slice_len - 1) // slice_len # Número de slices por chunk
        
        logger.info(f"🧩 Chunk {chunk_idx:03d}: {metadata['actual_chunk_size']} muestras → {time_slice} slices")
        
        # Generar DM-time cube
        dm_time = d_dm_time_g(block, height=height, width=width_total) # dedispersion del bloque
        
        # Configurar bandas
        band_configs = get_band_configs() # Configuración de bandas (fullband, lowband, highband)
        
        # Procesar slices
        cand_counter = 0 # Contador de candidatos
        n_bursts = 0 # Contador de candidatos de tipo burst
        n_no_bursts = 0 # Contador de candidatos de tipo no burst
        prob_max = 0.0 # Probabilidad máxima de detección
        snr_list = [] # Lista de SNRs de los candidatos
        
        for j in range(time_slice): # Procesar cada slice
            # Verificar que tenemos suficientes datos para este slice
            start_idx = slice_len * j # Índice de inicio del slice
            end_idx = slice_len * (j + 1) # Índice de fin del slice

            # Verificar que no excedemos los límites del bloque
            if start_idx >= block.shape[0]: # Si el índice de inicio es mayor o igual al tamaño del bloque
                logger.warning(f"Slice {j}: start_idx ({start_idx}) >= block.shape[0] ({block.shape[0]}), saltando...")
                continue

            if end_idx > block.shape[0]: # Si el índice de fin es mayor al tamaño del bloque
                logger.warning(f"Slice {j}: end_idx ({end_idx}) > block.shape[0] ({block.shape[0]}), ajustando...")
                end_idx = block.shape[0] # Ajustar el índice de fin al tamaño del bloque
                # Si el slice es muy pequeño, saltarlo
                if end_idx - start_idx < slice_len // 2:
                    logger.warning(f"Slice {j}: muy pequeño ({end_idx - start_idx} muestras), saltando...")
                    continue

            slice_cube = dm_time[:, :, start_idx : end_idx] # Cubo DM-time para este slice
            waterfall_block = block[start_idx : end_idx] # Bloque de datos para este slice

            if slice_cube.size == 0 or waterfall_block.size == 0: # Si el cubo o el bloque están vacíos
                logger.warning(f"Slice {j}: datos vacíos, saltando...")
                continue

            # Calcular tiempo absoluto para este slice específico
            slice_start_time_sec = chunk_start_time_sec + (j * slice_len * config.TIME_RESO * config.DOWN_TIME_RATE)

            # Crear carpeta principal del archivo
            file_folder_name = fits_path.stem
            chunk_folder_name = f"chunk{chunk_idx:03d}"
            
            # Estructura: Results/ObjectDetection/Composite/3096_0001_00_8bit/chunk000/
            composite_dir = save_dir / "Composite" / file_folder_name / chunk_folder_name
            composite_dir.mkdir(parents=True, exist_ok=True)
            detections_dir = save_dir / "Detections" / file_folder_name / chunk_folder_name
            detections_dir.mkdir(parents=True, exist_ok=True)
            patches_dir = save_dir / "Patches" / file_folder_name / chunk_folder_name
            patches_dir.mkdir(parents=True, exist_ok=True)

            # Candidate CSV and waterfall folders (already chunked)
            # csv_file = save_dir / f"{chunk_folder_name}.candidates.csv" # Moved outside _process_block
            # ensure_csv_header(csv_file) # Moved outside _process_block
            waterfall_dispersion_dir = save_dir / "waterfall_dispersion" / file_folder_name / chunk_folder_name
            waterfall_dedispersion_dir = save_dir / "waterfall_dedispersion" / file_folder_name / chunk_folder_name

            # Pass chunked paths to process_slice (these will be used in plot_manager)
            cands, bursts, no_bursts, max_prob = process_slice( # Procesar el slice
                j, dm_time, block, slice_len, det_model, cls_model, fits_path, save_dir,
                freq_down, csv_file, config.TIME_RESO * config.DOWN_TIME_RATE, band_configs,
                snr_list, waterfall_dispersion_dir, waterfall_dedispersion_dir, config,
                absolute_start_time=slice_start_time_sec,
                composite_dir=composite_dir,
                detections_dir=detections_dir,
                patches_dir=patches_dir,
                chunk_idx=chunk_idx 
            )

            cand_counter += cands
            n_bursts += bursts
            n_no_bursts += no_bursts
            prob_max = max(prob_max, max_prob)

            # LIMPIEZA DE MEMORIA DESPUÉS DE CADA SLICE
            if j % 10 == 0:  # Cada 10 slices
                _optimize_memory(aggressive=False)
            else:
                # Limpieza básica después de cada slice
                if plt is not None:
                    plt.close('all')
                gc.collect()

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
    """Procesa un archivo .fil en bloques usando stream_fil."""
    
    # 📊 CALCULAR INFORMACIÓN DEL ARCHIVO DE MANERA EFICIENTE
    logger.info(f"📁 Analizando estructura del archivo: {fits_path.name}")
    
    # Calcular información basada en config.FILE_LENG (ya cargado por get_obparams_fil)
    total_samples = config.FILE_LENG # Longitud total del archivo
    chunk_count = (total_samples + chunk_samples - 1) // chunk_samples  # Redondear hacia arriba
    
    total_duration_sec = total_samples * config.TIME_RESO # Duración total del archivo
    chunk_duration_sec = chunk_samples * config.TIME_RESO # Duración de cada chunk
    
    logger.info(f"📊 RESUMEN DEL ARCHIVO:")
    logger.info(f"   🧩 Total de chunks estimado: {chunk_count}")
    logger.info(f"   📊 Muestras totales: {total_samples:,}")
    logger.info(f"   🕐 Duración total: {total_duration_sec:.2f} segundos ({total_duration_sec/60:.1f} minutos)")
    logger.info(f"   📦 Tamaño de chunk: {chunk_samples:,} muestras ({chunk_duration_sec:.2f}s)")
    logger.info(f"   🔄 Iniciando procesamiento...")
    
    # Crear un CSV por archivo en lugar de por chunk
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
        # Procesar cada bloque (UNA SOLA PASADA)
        for block, metadata in stream_fil(str(fits_path), chunk_samples): # Procesar cada bloque
            actual_chunk_count += 1 # Incrementar el contador de chunks procesados
            logger.info(f"🧩 Procesando chunk {metadata['chunk_idx']:03d} " # Log del chunk actual
                       f"({metadata['start_sample']:,} - {metadata['end_sample']:,})") # Log del rango de muestras del chunk
            
            # Procesar bloque
            block_results = _process_block( 
                det_model, cls_model, block, metadata, # Modelos de detección y clasificación
                fits_path, save_dir, metadata['chunk_idx'], csv_file  # PASAR CSV_FILE
            ) 
            
            # Acumular resultados
            cand_counter_total += block_results["n_candidates"] # Acumular el número de candidatos
            n_bursts_total += block_results["n_bursts"] # Acumular el número de candidatos de tipo burst
            n_no_bursts_total += block_results["n_no_bursts"] # Acumular el número de candidatos de tipo no burst
            prob_max_total = max(prob_max_total, block_results["max_prob"]) # Actualizar la probabilidad máxima de detección
            
            # LIMPIEZA AGRESIVA DE MEMORIA
            del block # Liberar memoria del bloque
            _optimize_memory(aggressive=(actual_chunk_count % 5 == 0))  # Limpieza agresiva cada 5 chunks
        
        runtime = time.time() - t_start # Tiempo de ejecución
        logger.info(
            f"🧩 Archivo completado: {actual_chunk_count} chunks procesados, "
            f"{cand_counter_total} candidatos, max prob {prob_max_total:.2f}, ⏱️ {runtime:.1f}s"
        )
        
        return {
            "n_candidates": cand_counter_total,
            "n_bursts": n_bursts_total,
            "n_no_bursts": n_no_bursts_total,
            "runtime_s": runtime,
            "max_prob": prob_max_total,
            "mean_snr": float(np.mean(snr_list_total)) if snr_list_total else 0.0,
            "status": "completed_chunked"
        }
        
    except Exception as e:
        logger.error(f"Error procesando archivo por bloques: {e}")
        return {
            "n_candidates": 0,
            "n_bursts": 0,
            "n_no_bursts": 0,
            "runtime_s": time.time() - t_start,
            "max_prob": 0.0,
            "mean_snr": 0.0,
            "status": "ERROR_CHUNKED"
        }


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
# PROCESAMIENTO DE ARCHIVO INDIVIDUAL #
#
# Procesa un archivo FITS/FIL: carga datos, ejecuta el pipeline de slices, guarda candidatos y estadísticas.
# =============================================================================
def _process_file(
    det_model,
    cls_model,
    fits_path: Path,
    save_dir: Path,
    chunk_samples: int = 0,
) -> dict:
    """Process a single FITS or fil file and return summary information."""
    t_start = time.time()
    logger.info("Procesando %s", fits_path.name)
    
    # Determinar si usar procesamiento por bloques
    use_chunking = chunk_samples > 0 and fits_path.suffix.lower() == ".fil"
    
    if use_chunking:
        logger.info("🧩 Procesando archivo .fil en bloques de %d muestras", chunk_samples)
        return _process_file_chunked(det_model, cls_model, fits_path, save_dir, chunk_samples)
    
    # Procesamiento normal (modo estandar)
    cand_counter = 0
    n_bursts = 0
    n_no_bursts = 0
    prob_max = 0.0
    
    try:
        data = load_and_preprocess_data(fits_path)
    except ValueError as e:
        if "corrupto" in str(e).lower():
            logger.error("Archivo corrupto detectado: %s - SALTANDO", fits_path.name)
            return {
                "n_candidates": 0,
                "n_bursts": 0,
                "n_no_bursts": 0,
                "runtime_s": time.time() - t_start,
                "max_prob": 0.0,
                "mean_snr": 0.0,
                "status": "CORRUPTED_FILE"
            }
        else:
            raise
    freq_down, height, width_total, slice_len, real_duration_ms, time_slice, slice_duration = get_pipeline_parameters(config)
    logger.info("✅ Sistema de slice simplificado:")
    logger.info(f"   🎯 Duración objetivo: {config.SLICE_DURATION_MS:.1f} ms")
    logger.info(f"   🧩 SLICE_LEN calculado: {slice_len} muestras")
    logger.info(f"   ⏱️  Duración real obtenida: {real_duration_ms:.1f} ms")
    logger.info(f"   📊 Archivo: {config.FILE_LENG} muestras → {time_slice} slices")
    dm_time = d_dm_time_g(data, height=height, width=width_total)
    logger.info(
        "Análisis de %s con %d slices de %d muestras (%.3f s cada uno)",
        fits_path.name,
        time_slice,
        slice_len,
        slice_duration,
    )
    csv_file = save_dir / f"{fits_path.stem}.candidates.csv"
    ensure_csv_header(csv_file)
    band_configs = get_band_configs()
    # === HIERARCHICAL FOLDER STRUCTURE FOR NON-CHUNKED FILES ===
    file_folder_name = fits_path.stem
    chunk_folder_name = "chunk000"  # Para archivos no chunked, usar chunk000
    
    # Estructura: Results/ObjectDetection/Composite/3096_0001_00_8bit/chunk000/
    composite_dir = save_dir / "Composite" / file_folder_name / chunk_folder_name
    composite_dir.mkdir(parents=True, exist_ok=True)
    detections_dir = save_dir / "Detections" / file_folder_name / chunk_folder_name
    detections_dir.mkdir(parents=True, exist_ok=True)
    patches_dir = save_dir / "Patches" / file_folder_name / chunk_folder_name
    patches_dir.mkdir(parents=True, exist_ok=True)
    
    # Estructura: Results/ObjectDetection/waterfall_dispersion/3096_0001_00_8bit/chunk000/
    waterfall_dispersion_dir = save_dir / "waterfall_dispersion" / file_folder_name / chunk_folder_name
    waterfall_dedispersion_dir = save_dir / "waterfall_dedispersion" / file_folder_name / chunk_folder_name
    freq_ds = freq_down
    time_reso_ds = config.TIME_RESO * config.DOWN_TIME_RATE
    snr_list: List[float] = []
    for j in range(time_slice):
        cands, bursts, no_bursts, max_prob = process_slice(
            j, dm_time, data, slice_len, det_model, cls_model, fits_path, save_dir, freq_down, csv_file, time_reso_ds, band_configs, snr_list, waterfall_dispersion_dir, waterfall_dedispersion_dir, config,
            chunk_idx=0,  # chunk_idx=0 para archivos no chunked
            composite_dir=composite_dir,  # pasar directorio composite
            detections_dir=detections_dir,  # pasar directorio detections
            patches_dir=patches_dir  # pasar directorio patches
        )
        cand_counter += cands
        n_bursts += bursts
        n_no_bursts += no_bursts
        prob_max = max(prob_max, max_prob)
    runtime = time.time() - t_start
    logger.info(
        "\u25b6 %s: %d candidatos, max prob %.2f, \u23f1 %.1f s",
        fits_path.name,
        cand_counter,
        prob_max,
        runtime,
    )
    return {
        "n_candidates": cand_counter,
        "n_bursts": n_bursts,
        "n_no_bursts": n_no_bursts,
        "runtime_s": runtime,
        "max_prob": float(prob_max),
        "mean_snr": float(np.mean(snr_list)) if snr_list else 0.0,
    }


# =============================================================================
# EJECUCIÓN DEL PIPELINE COMPLETO #
#
# Controla el flujo principal: carga modelos, busca archivos, procesa cada archivo y guarda el resumen global.
# =============================================================================
def run_pipeline(chunk_samples: int = 0) -> None:
    """Run the full FRB detection pipeline.
    
    Args:
        chunk_samples: Número de muestras por bloque para archivos .fil (0 = modo antiguo)
    """

    print("=== INICIANDO PIPELINE DE DETECCIÓN DE FRB ===")
    print(f"Directorio de datos: {config.DATA_DIR}")
    print(f"Directorio de resultados: {config.RESULTS_DIR}")
    print(f"Targets FRB: {config.FRB_TARGETS}")
    if chunk_samples > 0:
        print(f"Modo chunking habilitado: {chunk_samples:,} muestras por bloque")
    else:
        print("Modo normal: carga completa en memoria")
    
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    save_dir = config.RESULTS_DIR / config.MODEL_NAME
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Directorio de guardado: {save_dir}")
    
    print("Cargando modelos...")
    det_model = _load_detection_model()
    cls_model = _load_class_model()
    print("Modelos cargados exitosamente")

    summary: dict[str, dict] = {}
    for frb in config.FRB_TARGETS:
        print(f"\nBuscando archivos para target: {frb}")
        file_list = _find_data_files(frb)
        print(f"Archivos encontrados: {[f.name for f in file_list]}")
        if not file_list:
            print(f"No se encontraron archivos para {frb}")
            continue

        try:
            first_file = file_list[0]
            print(f"Leyendo parámetros de observación desde: {first_file.name}")
            if first_file.suffix.lower() == ".fits":
                get_obparams(str(first_file))
            else:
                get_obparams_fil(str(first_file))
            print("Parámetros de observación cargados exitosamente")
        except Exception as e:
            print(f"[ERROR] Error obteniendo parámetros de observación: {e}")
            logger.error("Error obteniendo parámetros de observación: %s", e)
            continue
            
        for fits_path in file_list:
            try:
                print(f"Procesando archivo: {fits_path.name}")
                results = _process_file(det_model, cls_model, fits_path, save_dir, chunk_samples)
                summary[fits_path.name] = results
                
                # *** GUARDAR RESULTADOS INMEDIATAMENTE EN SUMMARY.JSON ***
                _update_summary_with_results(save_dir, fits_path.stem, {
                    "n_candidates": results.get("n_candidates", 0),
                    "n_bursts": results.get("n_bursts", 0),
                    "n_no_bursts": results.get("n_no_bursts", 0),
                    "processing_time": results.get("runtime_s", 0.0),
                    "max_detection_prob": results.get("max_prob", 0.0),
                    "mean_snr": results.get("mean_snr", 0.0),
                    "status": results.get("status", "completed")
                })
                
                print(f"Archivo {fits_path.name} procesado exitosamente")
            except Exception as e:
                print(f"[ERROR] Error procesando {fits_path.name}: {e}")
                logger.error("Error procesando %s: %s", fits_path.name, e)
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

    print("Escribiendo resumen final...")
    _write_summary_with_timestamp(summary, save_dir)
    print("=== PIPELINE COMPLETADO ===")

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
   
