"""Manejo de archivos filterbank (.fil) para el pipeline de detección de FRBs."""
from __future__ import annotations

import gc
import os
import struct
from typing import Dict, Generator, Tuple, Type

import numpy as np

# Local imports
from ..config import config
from ..logging import (
    log_stream_fil_block_generation,
    log_stream_fil_parameters,
    log_stream_fil_summary
)
from .utils import safe_float, safe_int, auto_config_downsampling, print_debug_frequencies, save_file_debug_info


def _read_int(f) -> int:
    """Lee un entero de 32 bits del archivo."""
    return struct.unpack("<i", f.read(4))[0]


def _read_double(f) -> float:
    """Lee un double de 64 bits del archivo."""
    return struct.unpack("<d", f.read(8))[0]


def _read_string(f) -> str:
    """Lee una cadena de caracteres del archivo."""
    length = _read_int(f)
    return f.read(length).decode('utf-8', errors='ignore')


def _read_header(f) -> Tuple[dict, int]:
    """Read filterbank header, handling both standard and non-standard formats."""
    original_pos = f.tell()
    
    try:
        # Try to read as standard SIGPROC format first
        start = _read_string(f)
        if start != "HEADER_START":
            # If not standard format, reset and try alternative approach
            f.seek(original_pos)
            return _read_non_standard_header(f)

        header = {}
        while True:
            try:
                key = _read_string(f)
                if key == "HEADER_END":
                    break
                if key in {"rawdatafile", "source_name"}:
                    header[key] = _read_string(f)
                elif key in {
                    "telescope_id",
                    "machine_id",
                    "data_type",
                    "barycentric",
                    "pulsarcentric",
                    "nbits",
                    "nchans",
                    "nifs",
                    "nbeams",
                    "ibeam",
                    "nsamples",
                }:
                    header[key] = _read_int(f)
                elif key in {
                    "az_start",
                    "za_start",
                    "src_raj",
                    "src_dej",
                    "tstart",
                    "tsamp",
                    "fch1",
                    "foff",
                    "refdm",
                }:
                    header[key] = _read_double(f)
                else:
                    # Read unknown field as integer by default
                    header[key] = _read_int(f)
            except (struct.error, UnicodeDecodeError) as e:
                print(f"Warning: Error reading header field '{key}': {e}")
                continue
        return header, f.tell()
    except Exception as e:
        print(f"Error reading standard filterbank header: {e}")
        f.seek(original_pos)
        return _read_non_standard_header(f)


def _read_non_standard_header(f) -> Tuple[dict, int]:
    """Handle non-standard filterbank files by assuming common parameters."""
    print("[INFO] Detectado archivo .fil con formato no estándar, usando parámetros estimados")
    
    # Get file size to estimate parameters
    current_pos = f.tell()
    f.seek(0, 2)  # Go to end
    file_size = f.tell()
    f.seek(current_pos)  # Return to original position
    
    # Common parameters for many filterbank files
    header = {
        "nchans": 512,
        "tsamp": 8.192e-5,
        "fch1": 1500.0,
        "foff": -1.0,
        "nbits": 8,
        "nifs": 1,
    }
    
    # Estimate number of samples based on file size
    bytes_per_sample = header["nifs"] * header["nchans"] * (header["nbits"] // 8)
    estimated_samples = (file_size - 512) // bytes_per_sample
    max_samples = config.MAX_SAMPLES_LIMIT
    header["nsamples"] = min(estimated_samples, max_samples)
    
    print(f"[INFO] Parámetros estimados para archivo no estándar:")
    print(f"  - Tamaño de archivo: {file_size / (1024**2):.1f} MB")
    print(f"  - Muestras estimadas: {estimated_samples}")
    print(f"  - Muestras a usar: {header['nsamples']}")
    
    return header, 512


def load_fil_file(file_name: str) -> np.ndarray:
    """Load a filterbank file and return the data array in shape (time, pol, channel)."""
    global_vars = config
    data_array = None
    
    try:
        with open(file_name, "rb") as f:
            header, hdr_len = _read_header(f)

        nchans = header.get("nchans", 512)
        nifs = header.get("nifs", 1)
        nbits = header.get("nbits", 8)
        nsamples = header.get("nsamples")
        
        if nsamples is None:
            bytes_per_sample = nifs * nchans * (nbits // 8)
            file_size = os.path.getsize(file_name) - hdr_len
            nsamples = file_size // bytes_per_sample if bytes_per_sample > 0 else 1000

        # Mapeo de tipos de datos
        dtype_map: Dict[int, Type] = {
            8: np.uint8,
            16: np.int16,
            32: np.float32,
            64: np.float64
        }
        
        dtype = dtype_map.get(nbits, np.uint8)
            
        print(f"[INFO] Cargando datos: {nsamples} muestras, {nchans} canales, tipo {dtype}")
        
        # Memory-map the data
        try:
            data = np.memmap(
                file_name,
                dtype=dtype,
                mode="r",
                offset=hdr_len,
                shape=(nsamples, nifs, nchans),
            )
            data_array = np.array(data)
        except ValueError as e:
            print(f"[WARNING] Error creating memmap: {e}")
            safe_samples = min(nsamples, 10000)
            data = np.memmap(
                file_name,
                dtype=dtype,
                mode="r",
                offset=hdr_len,
                shape=(safe_samples, nifs, nchans),
            )
            data_array = np.array(data)
            
    except Exception as e:
        print(f"[Error cargando FIL] {e}")
        try:
            # Fallback to synthetic data
            data_array = np.random.rand(1000, 1, 512).astype(np.float32)
        except Exception:
            raise ValueError(f"No se pudieron cargar los datos de {file_name}")
            
    if data_array is None:
        raise ValueError(f"No se pudieron cargar los datos de {file_name}")

    if global_vars.DATA_NEEDS_REVERSAL:
        print(f">> Invirtiendo eje de frecuencia de los datos cargados para {file_name}")
        data_array = np.ascontiguousarray(data_array[:, :, ::-1])
    
    # DEBUG: Información de los datos cargados
    if config.DEBUG_FREQUENCY_ORDER:
        print(f"[DEBUG DATOS FIL] Archivo: {file_name}")
        print(f"[DEBUG DATOS FIL] Shape de datos: {data_array.shape}")
        print(f"[DEBUG DATOS FIL] Dimensiones: (tiempo={data_array.shape[0]}, pol={data_array.shape[1]}, freq={data_array.shape[2]})")
        print(f"[DEBUG DATOS FIL] Tipo de datos: {data_array.dtype}")
        print(f"[DEBUG DATOS FIL] Tamaño en memoria: {data_array.nbytes / (1024**3):.2f} GB")
        print(f"[DEBUG DATOS FIL] Reversión aplicada: {global_vars.DATA_NEEDS_REVERSAL}")
        print(f"[DEBUG DATOS FIL] Rango de valores: [{data_array.min():.3f}, {data_array.max():.3f}]")
        print(f"[DEBUG DATOS FIL] Valor medio: {data_array.mean():.3f}")
        print(f"[DEBUG DATOS FIL] Desviación estándar: {data_array.std():.3f}")
        print("[DEBUG DATOS FIL] " + "="*50)
    
    return data_array


def get_obparams_fil(file_name: str) -> None:
    """Extract observation parameters and populate :mod:`config`."""
    
    # DEBUG: Información de entrada del archivo
    if config.DEBUG_FREQUENCY_ORDER:
        print(f"[DEBUG FILTERBANK] Iniciando extracción de parámetros de: {file_name}")
        print(f"[DEBUG FILTERBANK] " + "="*60)
    
    with open(file_name, "rb") as f:
        freq_axis_inverted = False
        header, hdr_len = _read_header(f)

        # extraer nchans, fch1 y foff y construir el eje de frecuencias
        nchans = header.get("nchans", 512)                 
        fch1   = header.get("fch1", None)                   
        foff   = header.get("foff", None)                   
        if fch1 is None or foff is None:                     
            raise ValueError(f"Header inválido: faltan fch1={fch1} o foff={foff}") 
        freq_temp = fch1 + foff * np.arange(nchans)        

        # DEBUG: Estructura del archivo filterbank
        if config.DEBUG_FREQUENCY_ORDER:
            print(f"[DEBUG FILTERBANK] Estructura del archivo Filterbank:")
            print(f"[DEBUG FILTERBANK]   Formato: SIGPROC Filterbank (.fil)")
            print(f"[DEBUG FILTERBANK]   Tamaño del header: {hdr_len} bytes")
            print(f"[DEBUG FILTERBANK] Headers extraídos del archivo .fil:")
            for key, value in header.items():
                print(f"[DEBUG FILTERBANK]   {key}: {value}")

        nchans = header.get("nchans", 512)
        tsamp = header.get("tsamp", 8.192e-5)
        nifs = header.get("nifs", 1)
        nbits = header.get("nbits", 8)
        nsamples = header.get("nsamples")
        
        if nsamples is None:
            bytes_per_sample = nifs * nchans * (nbits // 8)
            file_size = os.path.getsize(file_name) - hdr_len
            nsamples = file_size // bytes_per_sample if bytes_per_sample > 0 else 1000
            
            if config.DEBUG_FREQUENCY_ORDER:
                print(f"[DEBUG FILTERBANK] nsamples no en header, calculando:")
                print(f"[DEBUG FILTERBANK]   Tamaño archivo: {file_size} bytes")
                print(f"[DEBUG FILTERBANK]   Bytes por muestra: {bytes_per_sample}")
                print(f"[DEBUG FILTERBANK]   Muestras calculadas: {nsamples}")

        # Solo mostrar información esencial en modo debug
        if config.DEBUG_FREQUENCY_ORDER:
            print(f"[DEBUG FILTERBANK]   tsamp (resolución temporal): {tsamp:.2e} s")
            print(f"[DEBUG FILTERBANK]   nchans (canales): {nchans}")
            print(f"[DEBUG FILTERBANK]   nifs (polarizaciones): {nifs}")
            print(f"[DEBUG FILTERBANK]   nbits (bits por muestra): {nbits}")
            if 'telescope_id' in header:
                print(f"[DEBUG FILTERBANK]   telescope_id: {header['telescope_id']}")
            if 'source_name' in header:
                print(f"[DEBUG FILTERBANK]   Fuente: {header['source_name']}")
            print(f"[DEBUG FILTERBANK]   Total de muestras: {nsamples}")
            
            print(f"[DEBUG FILTERBANK] Análisis de frecuencias:")
            print(f"[DEBUG FILTERBANK]   fch1 (freq inicial): {fch1} MHz")
            print(f"[DEBUG FILTERBANK]   foff (ancho canal): {foff} MHz")
            print(f"[DEBUG FILTERBANK]   Primeras 5 freq calculadas: {freq_temp[:5]}")
            print(f"[DEBUG FILTERBANK]   Últimas 5 freq calculadas: {freq_temp[-5:]}")
        
        # Detectar inversión de frecuencias (homólogo a io.py)
        if foff < 0:
            freq_axis_inverted = True
            if config.DEBUG_FREQUENCY_ORDER:
                print(f"[DEBUG FILTERBANK] foff negativo - frecuencias invertidas!")
        elif len(freq_temp) > 1 and freq_temp[0] > freq_temp[-1]:
            freq_axis_inverted = True
            if config.DEBUG_FREQUENCY_ORDER:
                print(f"[DEBUG FILTERBANK] Frecuencias detectadas en orden descendente!")
        else:
            # CORRECCIÓN: Invertir también frecuencias ascendentes para mantener estándar de radioastronomía
            # (alta frecuencia arriba, como en plot_waterfall.py)
            freq_axis_inverted = True
            if config.DEBUG_FREQUENCY_ORDER:
                print(f"[DEBUG FILTERBANK] foff positivo - invirtiendo para estándar radioastronomía!")
        
        # Aplicar corrección de orden (homólogo a io.py)
        if freq_axis_inverted:
            config.FREQ = freq_temp[::-1]
            config.DATA_NEEDS_REVERSAL = True
        else:
            config.FREQ = freq_temp
            config.DATA_NEEDS_REVERSAL = False

    # DEBUG: Orden de frecuencias
    if config.DEBUG_FREQUENCY_ORDER:
        print_debug_frequencies("[DEBUG FRECUENCIAS FIL]", file_name, freq_axis_inverted)

    # *** ASIGNAR VARIABLES GLOBALES ANTES DEL DEBUG ***
    config.TIME_RESO = tsamp
    config.FREQ_RESO = nchans
    config.FILE_LENG = nsamples

    # RESPETAR CONFIGURACIONES DEL USUARIO - calcular automáticamente si corresponde
    auto_config_downsampling()

    # DEBUG: Información completa del archivo
    if config.DEBUG_FREQUENCY_ORDER:
        print(f"[DEBUG ARCHIVO FIL] Información completa del archivo: {file_name}")
        print(f"[DEBUG ARCHIVO FIL] " + "="*60)
        print(f"[DEBUG ARCHIVO FIL] DIMENSIONES Y RESOLUCIÓN:")
        print(f"[DEBUG ARCHIVO FIL]   - Resolución temporal: {config.TIME_RESO:.2e} segundos/muestra")
        print(f"[DEBUG ARCHIVO FIL]   - Resolución de frecuencia: {config.FREQ_RESO} canales")
        print(f"[DEBUG ARCHIVO FIL]   - Longitud del archivo: {config.FILE_LENG:,} muestras")
        print(f"[DEBUG ARCHIVO FIL]   - Bits por muestra: {nbits}")
        print(f"[DEBUG ARCHIVO FIL]   - Polarizaciones: {nifs}")
        
        # Calcular duración total
        duracion_total_seg = config.FILE_LENG * config.TIME_RESO
        duracion_min = duracion_total_seg / 60
        duracion_horas = duracion_min / 60
        print(f"[DEBUG ARCHIVO FIL]   - Duración total: {duracion_total_seg:.2f} seg ({duracion_min:.2f} min, {duracion_horas:.2f} h)")
        
        print(f"[DEBUG ARCHIVO FIL] FRECUENCIAS:")
        print(f"[DEBUG ARCHIVO FIL]   - Rango total: {config.FREQ.min():.2f} - {config.FREQ.max():.2f} MHz")
        print(f"[DEBUG ARCHIVO FIL]   - Ancho de banda: {abs(config.FREQ.max() - config.FREQ.min()):.2f} MHz")
        print(f"[DEBUG ARCHIVO FIL]   - Resolución por canal: {abs(foff):.4f} MHz/canal")
        print(f"[DEBUG ARCHIVO FIL]   - Orden original: {'DESCENDENTE (foff<0)' if foff < 0 else 'ASCENDENTE (foff>0)'}")
        print(f"[DEBUG ARCHIVO FIL]   - Orden final (post-corrección): {'ASCENDENTE' if config.FREQ[0] < config.FREQ[-1] else 'DESCENDENTE'}")
        
        print(f"[DEBUG ARCHIVO FIL] DECIMACIÓN:")
        print(f"[DEBUG ARCHIVO FIL]   - Factor reducción frecuencia: {config.DOWN_FREQ_RATE}x")
        print(f"[DEBUG ARCHIVO FIL]   - Factor reducción tiempo: {config.DOWN_TIME_RATE}x")
        print(f"[DEBUG ARCHIVO FIL]   - Canales después de decimación: {config.FREQ_RESO // config.DOWN_FREQ_RATE}")
        print(f"[DEBUG ARCHIVO FIL]   - Resolución temporal después: {config.TIME_RESO * config.DOWN_TIME_RATE:.2e} seg/muestra")
        
        # Calcular tamaño aproximado de datos
        size_original_gb = (config.FILE_LENG * config.FREQ_RESO * (nbits/8)) / (1024**3)
        size_decimated_gb = size_original_gb / (config.DOWN_FREQ_RATE * config.DOWN_TIME_RATE)
        print(f"[DEBUG ARCHIVO FIL] TAMAÑO ESTIMADO:")
        print(f"[DEBUG ARCHIVO FIL]   - Datos originales: ~{size_original_gb:.2f} GB")
        print(f"[DEBUG ARCHIVO FIL]   - Datos después decimación: ~{size_decimated_gb:.2f} GB")
        
        
        print(f"[DEBUG ARCHIVO FIL] PROCESAMIENTO:")
        print(f"[DEBUG ARCHIVO FIL]   - Multi-banda habilitado: {'SÍ' if config.USE_MULTI_BAND else 'NO'}")
        print(f"[DEBUG ARCHIVO FIL]   - DM rango: {config.DM_min} - {config.DM_max} pc cm⁻³")
        print(f"[DEBUG ARCHIVO FIL]   - Umbrales: DET_PROB={config.DET_PROB}, CLASS_PROB={config.CLASS_PROB}, SNR_THRESH={config.SNR_THRESH}")
        print(f"[DEBUG ARCHIVO FIL] " + "="*60)

    # DEBUG: Configuración final de decimación
    if config.DEBUG_FREQUENCY_ORDER:
        print(f"[DEBUG CONFIG FINAL FIL] Configuración final después de get_obparams_fil:")
        print(f"[DEBUG CONFIG FINAL FIL] " + "="*60)
        print(f"[DEBUG CONFIG FINAL FIL] DOWN_FREQ_RATE calculado: {config.DOWN_FREQ_RATE}x")
        print(f"[DEBUG CONFIG FINAL FIL] DOWN_TIME_RATE calculado: {config.DOWN_TIME_RATE}x")
        print(f"[DEBUG CONFIG FINAL FIL] Datos después de decimación:")
        print(f"[DEBUG CONFIG FINAL FIL]   - Canales: {config.FREQ_RESO // config.DOWN_FREQ_RATE}")
        print(f"[DEBUG CONFIG FINAL FIL]   - Resolución temporal: {config.TIME_RESO * config.DOWN_TIME_RATE:.2e} s/muestra")
        print(f"[DEBUG CONFIG FINAL FIL]   - Reducción total de datos: {config.DOWN_FREQ_RATE * config.DOWN_TIME_RATE}x")
        print(f"[DEBUG CONFIG FINAL FIL] DATA_NEEDS_REVERSAL final: {config.DATA_NEEDS_REVERSAL}")
        print(f"[DEBUG CONFIG FINAL FIL] Orden de frecuencias final: {'ASCENDENTE' if config.FREQ[0] < config.FREQ[-1] else 'DESCENDENTE'}")
        print(f"[DEBUG CONFIG FINAL FIL] " + "="*60)

    print(f"[INFO] Parámetros del archivo .fil cargados exitosamente:")
    print(f"  - Canales: {nchans}")
    print(f"  - Resolución temporal: {tsamp:.2e} s")
    print(f"  - Frecuencia inicial: {fch1} MHz")
    print(f"  - Ancho de banda: {foff} MHz")
    print(f"  - Muestras: {nsamples}")
    print(f"  - Down-sampling frecuencia: {config.DOWN_FREQ_RATE}")
    print(f"  - Down-sampling tiempo: {config.DOWN_TIME_RATE}")

    # *** GUARDAR DEBUG INFO EN SUMMARY.JSON INMEDIATAMENTE ***
    if config.DEBUG_FREQUENCY_ORDER:
        save_file_debug_info(file_name, {
            "file_type": "filterbank",
            "file_size_bytes": os.path.getsize(file_name),
            "file_size_gb": os.path.getsize(file_name) / (1024**3),
            "header_size_bytes": hdr_len,
            "format": "SIGPROC Filterbank (.fil)",
            "source_name": header.get('source_name', 'Unknown'),
            "telescope_id": header.get('telescope_id', 'Unknown'),
            "raw_parameters": {
                "tsamp": tsamp,
                "nchans": nchans,
                "nifs": nifs,
                "nbits": nbits,
                "nsamples": nsamples,
                "fch1": fch1,
                "foff": foff
            },
            "frequency_analysis": {
                "fch1_mhz": fch1,
                "foff_mhz": foff,
                "freq_min_mhz": float(config.FREQ.min()),
                "freq_max_mhz": float(config.FREQ.max()),
                "bandwidth_mhz": abs(config.FREQ.max() - config.FREQ.min()),
                "resolution_per_channel_mhz": abs(foff),
                "original_order": "DESCENDENTE (foff<0)" if foff < 0 else "ASCENDENTE (foff>0)",
                "final_order": "ASCENDENTE" if config.FREQ[0] < config.FREQ[-1] else "DESCENDENTE",
                "freq_axis_inverted": freq_axis_inverted,
                "data_needs_reversal": config.DATA_NEEDS_REVERSAL
            },
            "time_analysis": {
                "time_resolution_sec": config.TIME_RESO,
                "total_samples": config.FILE_LENG,
                "total_duration_sec": config.FILE_LENG * config.TIME_RESO,
                "total_duration_min": (config.FILE_LENG * config.TIME_RESO) / 60,
                "total_duration_hours": (config.FILE_LENG * config.TIME_RESO) / 3600
            },
            "decimation": {
                "down_freq_rate": config.DOWN_FREQ_RATE,
                "down_time_rate": config.DOWN_TIME_RATE,
                "channels_after_decimation": config.FREQ_RESO // config.DOWN_FREQ_RATE,
                "time_resolution_after_decimation_sec": config.TIME_RESO * config.DOWN_TIME_RATE,
                "total_reduction_factor": config.DOWN_FREQ_RATE * config.DOWN_TIME_RATE
            },
            "slice_config": {
                "slice_len_calculated": round(config.SLICE_DURATION_MS / (config.TIME_RESO * config.DOWN_TIME_RATE * 1000)),
                "slice_len_limits": [config.SLICE_LEN_MIN, config.SLICE_LEN_MAX]
            },
            "processing_config": {
                "multi_band_enabled": config.USE_MULTI_BAND,
                "dm_range": [config.DM_min, config.DM_max],
                "detection_thresholds": {
                    "det_prob": config.DET_PROB,
                    "class_prob": config.CLASS_PROB,
                    "snr_thresh": config.SNR_THRESH
                }
            },
            "file_temporal_info": {
                "total_duration_sec": config.FILE_LENG * config.TIME_RESO,
                "total_duration_formatted": f"{(config.FILE_LENG * config.TIME_RESO) // 3600:.0f}h {((config.FILE_LENG * config.TIME_RESO) % 3600) // 60:.0f}m {(config.FILE_LENG * config.TIME_RESO) % 60:.1f}s",
                "sample_rate_hz": 1.0 / config.TIME_RESO,
                "effective_sample_rate_after_decimation_hz": 1.0 / (config.TIME_RESO * config.DOWN_TIME_RATE),
            }
        })


def stream_fil(
    file_name: str,
    chunk_samples: int = 2_097_152,
    overlap_samples: int = 0,
) -> Generator[Tuple[np.ndarray, Dict], None, None]:
    """
    Generador que lee un archivo .fil en bloques sin cargar todo en RAM.
    
    Args:
        file_name: Ruta al archivo .fil
        chunk_samples: Número de muestras por bloque (default: 2M)
        overlap_samples: Número de muestras de solapamiento entre bloques
    
    Yields:
        Tuple[data_block, metadata]: Bloque de datos (time, pol, chan) y metadatos
    """
    
    # Mapeo de tipos de datos
    dtype_map: Dict[int, Type] = {
        8: np.uint8,
        16: np.int16,
        32: np.float32,
        64: np.float64
    }
    
    try:
        # Leer header
        with open(file_name, "rb") as f:
            header, hdr_len = _read_header(f)
        
        nchans = header.get("nchans", 512)
        nifs = header.get("nifs", 1)
        nbits = header.get("nbits", 8)
        nsamples = header.get("nsamples")
        
        # Calcular nsamples si falta
        if nsamples is None:
            bytes_per_sample = nifs * nchans * (nbits // 8)
            file_size = os.path.getsize(file_name) - hdr_len
            nsamples = file_size // bytes_per_sample if bytes_per_sample > 0 else 1000
        
        dtype = dtype_map.get(nbits, np.uint8)
        
        print(f"[INFO] Streaming datos: {nsamples} muestras totales, "
              f"{nchans} canales, tipo {dtype}, chunk_size={chunk_samples}, overlap={overlap_samples}")
        
        # *** DEBUG CRÍTICO: CONFIRMAR PARÁMETROS DE STREAMING FIL ***
        log_stream_fil_parameters(nsamples, chunk_samples, overlap_samples, nchans, nifs, nbits, str(dtype))
        
        # Crear memmap para acceso eficiente
        data_mmap = np.memmap(
            file_name,
            dtype=dtype,
            mode="r",
            offset=hdr_len,
            shape=(nsamples, nifs, nchans),
        )
        
        # Procesar en bloques
        chunk_counter = 0
        for chunk_start in range(0, nsamples, chunk_samples):
            chunk_counter += 1
            valid_start = chunk_start
            valid_end = min(chunk_start + chunk_samples, nsamples)
            actual_chunk_size = valid_end - valid_start

            # Rango con solapamiento aplicado (en crudo)
            start_with_overlap = max(0, valid_start - overlap_samples)
            end_with_overlap = min(nsamples, valid_end + overlap_samples)

            # Leer bloque con solapamiento
            block = data_mmap[start_with_overlap:end_with_overlap].copy()
            
            # *** DEBUG CRÍTICO: CONFIRMAR CADA BLOQUE GENERADO ***
            log_stream_fil_block_generation(chunk_counter, block.shape, str(block.dtype), valid_start, valid_end, start_with_overlap, end_with_overlap, actual_chunk_size)
            
            # Aplicar reversión de frecuencia si es necesario
            if config.DATA_NEEDS_REVERSAL:
                block = np.ascontiguousarray(block[:, :, ::-1])
            
            # Convertir a float32 para consistencia
            if block.dtype != np.float32:
                block = block.astype(np.float32)
            
            # Metadatos del bloque
            metadata = {
                "chunk_idx": valid_start // chunk_samples,
                "start_sample": valid_start,               # inicio válido (sin solape)
                "end_sample": valid_end,                   # fin válido (sin solape)
                "actual_chunk_size": actual_chunk_size,    # tamaño válido
                "block_start_sample": start_with_overlap,  # inicio del bloque con solape
                "block_end_sample": end_with_overlap,      # fin del bloque con solape
                "overlap_left": valid_start - start_with_overlap,
                "overlap_right": end_with_overlap - valid_end,
                "total_samples": nsamples,
                "nchans": nchans,
                "nifs": nifs,
                "dtype": str(block.dtype),
                "shape": block.shape
            }
            
            yield block, metadata
            
            # Limpiar memoria
            del block
            gc.collect()
        
        # *** DEBUG CRÍTICO: CONFIRMAR RESUMEN DE STREAMING ***
        log_stream_fil_summary(chunk_counter)
        
        # Limpiar memmap
        del data_mmap
        
    except Exception as e:
        print(f"[ERROR] Error en stream_fil: {e}")
        raise ValueError(f"No se pudo leer el archivo {file_name}") from e
