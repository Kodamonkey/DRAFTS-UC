"""Helper functions to read SIGPROC filterbank (.fil) files."""
from __future__ import annotations

import os
import struct
import gc
from pathlib import Path
from typing import Tuple, Generator, Dict, Type

import numpy as np

from . import config
from .summary_utils import _update_summary_with_file_debug


def _read_int(f) -> int:
    return struct.unpack("<i", f.read(4))[0]


def _read_double(f) -> float:
    return struct.unpack("<d", f.read(8))[0]


def _read_string(f) -> str:
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


def stream_fil(file_name: str, chunk_samples: int = 2_097_152) -> Generator[Tuple[np.ndarray, Dict], None, None]:
    """
    Generador que lee un archivo .fil en bloques sin cargar todo en RAM.
    
    Args:
        file_name: Ruta al archivo .fil
        chunk_samples: Número de muestras por bloque (default: 2M)
    
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
              f"{nchans} canales, tipo {dtype}, chunk_size={chunk_samples}")
        
        # Crear memmap para acceso eficiente
        data_mmap = np.memmap(
            file_name,
            dtype=dtype,
            mode="r",
            offset=hdr_len,
            shape=(nsamples, nifs, nchans),
        )
        
        # Procesar en bloques
        for chunk_idx in range(0, nsamples, chunk_samples):
            end_sample = min(chunk_idx + chunk_samples, nsamples)
            actual_chunk_size = end_sample - chunk_idx
            
            # Leer bloque actual
            block = data_mmap[chunk_idx:end_sample].copy()  # Copia solo este bloque
            
            # Aplicar reversión de frecuencia si es necesario
            if config.DATA_NEEDS_REVERSAL:
                block = np.ascontiguousarray(block[:, :, ::-1])
            
            # Convertir a float32 para consistencia
            if block.dtype != np.float32:
                block = block.astype(np.float32)
            
            # Metadatos del bloque
            metadata = {
                "chunk_idx": chunk_idx // chunk_samples,
                "start_sample": chunk_idx,
                "end_sample": end_sample,
                "actual_chunk_size": actual_chunk_size,
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
        
        # Limpiar memmap
        del data_mmap
        gc.collect()
        
    except Exception as e:
        print(f"[ERROR] Error en stream_fil: {e}")
        raise ValueError(f"No se pudo leer el archivo {file_name}") from e


def load_fil_file(file_name: str) -> np.ndarray:
    """
    Carga un archivo SIGPROC Filterbank (.fil) y devuelve un array
    con forma (time, pol, channel) en float32.
    """
    global_vars = config
    data_array = None

    try:
        # ────────────────────── 1. LEER HEADER ───────────────────────────
        with open(file_name, "rb") as f:
            header, hdr_len = _read_header(f)

        nchans   = header.get("nchans", 512)
        nifs     = header.get("nifs",   1)
        nbits    = header.get("nbits",  8)
        nsamples = header.get("nsamples")

        # ───── 2. ELEGIR dtype CORRECTO  (Reparo A) ───────────────────────
        dtype_map = {8:  np.uint8,
                     16: np.int16,
                     32: np.float32,
                     64: np.float64}
        dtype = dtype_map.get(nbits, np.uint8)

        # ───── 3. CALCULAR nsamples si falta en header ────────────────────
        if nsamples is None:
            bytes_per_sample = nifs * nchans * (nbits // 8)
            file_size = os.path.getsize(file_name) - hdr_len
            nsamples  = file_size // bytes_per_sample if bytes_per_sample > 0 else 1000

        print(f"[INFO] Cargando datos: {nsamples} muestras, "
              f"{nchans} canales, tipo {dtype}")

        # ────────────────────── 4. MEMMAP + COPIA ────────────────────────
        try:
            data = np.memmap(
                file_name,
                dtype=dtype,
                mode="r",
                offset=hdr_len,
                shape=(nsamples, nifs, nchans),
            )
            data_array = np.array(data)        # ⚠️ copia completa a RAM
        except ValueError as e:
            print(f"[WARNING] Error creating memmap: {e}")
            safe_samples = min(nsamples, 10_000)
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
            # Fallback sintético
            data_array = np.random.rand(1_000, 1, 512).astype(np.float32)
        except Exception:
            raise ValueError(f"No se pudieron cargar los datos de {file_name}")

    if data_array is None:
        raise ValueError(f"No se pudieron cargar los datos de {file_name}")

    # ────────────────────── 5. INVERSIÓN FREQ OPCIONAL ───────────────────
    if global_vars.DATA_NEEDS_REVERSAL:
        print(f">> Invirtiendo eje de frecuencia de los datos cargados "
              f"para {file_name}")
        data_array = np.ascontiguousarray(data_array[:, :, ::-1])
    
    # DEBUG: Información de los datos cargados
    if config.DEBUG_FREQUENCY_ORDER:
        print(f"💾 [DEBUG DATOS FIL] Archivo: {file_name}")
        print(f"💾 [DEBUG DATOS FIL] Shape de datos: {data_array.shape}")
        print(f"💾 [DEBUG DATOS FIL] Dimensiones: (tiempo={data_array.shape[0]}, pol={data_array.shape[1]}, freq={data_array.shape[2]})")
        print(f"💾 [DEBUG DATOS FIL] Tipo de datos: {data_array.dtype}")
        print(f"💾 [DEBUG DATOS FIL] Tamaño en memoria: {data_array.nbytes / (1024**3):.2f} GB")
        print(f"💾 [DEBUG DATOS FIL] Reversión aplicada: {global_vars.DATA_NEEDS_REVERSAL}")
        print(f"💾 [DEBUG DATOS FIL] Rango de valores: [{data_array.min():.3f}, {data_array.max():.3f}]")
        print(f"💾 [DEBUG DATOS FIL] Valor medio: {data_array.mean():.3f}")
        print(f"💾 [DEBUG DATOS FIL] Desviación estándar: {data_array.std():.3f}")
        print("💾 [DEBUG DATOS FIL] " + "="*50)
    
    return data_array


def get_obparams_fil(file_name: str) -> None:
    """Extract observation parameters and populate :mod:`config`."""
    
    # DEBUG: Información de entrada del archivo
    if config.DEBUG_FREQUENCY_ORDER:
        print(f"📋 [DEBUG FILTERBANK] Iniciando extracción de parámetros de: {file_name}")
        print(f"📋 [DEBUG FILTERBANK] " + "="*60)
    
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
            print(f"📋 [DEBUG FILTERBANK] Estructura del archivo Filterbank:")
            print(f"📋 [DEBUG FILTERBANK]   Formato: SIGPROC Filterbank (.fil)")
            print(f"📋 [DEBUG FILTERBANK]   Tamaño del header: {hdr_len} bytes")
            print(f"📋 [DEBUG FILTERBANK] Headers extraídos del archivo .fil:")
            for key, value in header.items():
                print(f"📋 [DEBUG FILTERBANK]   {key}: {value}")

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
                print(f"📋 [DEBUG FILTERBANK] nsamples no en header, calculando:")
                print(f"📋 [DEBUG FILTERBANK]   Tamaño archivo: {file_size} bytes")
                print(f"📋 [DEBUG FILTERBANK]   Bytes por muestra: {bytes_per_sample}")
                print(f"📋 [DEBUG FILTERBANK]   Muestras calculadas: {nsamples}")

            print(f"📋 [DEBUG FILTERBANK]   tsamp (resolución temporal): {tsamp:.2e} s")
            print(f"📋 [DEBUG FILTERBANK]   nchans (canales): {nchans}")
            print(f"📋 [DEBUG FILTERBANK]   nifs (polarizaciones): {nifs}")
            print(f"📋 [DEBUG FILTERBANK]   nbits (bits por muestra): {nbits}")
            if 'telescope_id' in header:
                print(f"📋 [DEBUG FILTERBANK]   telescope_id: {header['telescope_id']}")
            if 'source_name' in header:
                print(f"📋 [DEBUG FILTERBANK]   Fuente: {header['source_name']}")
            print(f"📋 [DEBUG FILTERBANK]   Total de muestras: {nsamples}")
            
            print(f"📋 [DEBUG FILTERBANK] Análisis de frecuencias:")
            print(f"📋 [DEBUG FILTERBANK]   fch1 (freq inicial): {fch1} MHz")
            print(f"📋 [DEBUG FILTERBANK]   foff (ancho canal): {foff} MHz")
            print(f"📋 [DEBUG FILTERBANK]   Primeras 5 freq calculadas: {freq_temp[:5]}")
            print(f"📋 [DEBUG FILTERBANK]   Últimas 5 freq calculadas: {freq_temp[-5:]}")
        
        # Detectar inversión de frecuencias (homólogo a io.py)
        if foff < 0:
            freq_axis_inverted = True
            if config.DEBUG_FREQUENCY_ORDER:
                print(f"📋 [DEBUG FILTERBANK]   ⚠️ foff negativo - frecuencias invertidas!")
        elif len(freq_temp) > 1 and freq_temp[0] > freq_temp[-1]:
            freq_axis_inverted = True
            if config.DEBUG_FREQUENCY_ORDER:
                print(f"📋 [DEBUG FILTERBANK]   ⚠️ Frecuencias detectadas en orden descendente!")
        
        # Aplicar corrección de orden (homólogo a io.py)
        if freq_axis_inverted:
            config.FREQ = freq_temp[::-1]
            config.DATA_NEEDS_REVERSAL = True
        else:
            config.FREQ = freq_temp
            config.DATA_NEEDS_REVERSAL = False

    # DEBUG: Orden de frecuencias
    if config.DEBUG_FREQUENCY_ORDER:
        print(f"🔍 [DEBUG FRECUENCIAS FIL] Archivo: {file_name}")
        print(f"🔍 [DEBUG FRECUENCIAS FIL] freq_axis_inverted detectado: {freq_axis_inverted}")
        print(f"🔍 [DEBUG FRECUENCIAS FIL] DATA_NEEDS_REVERSAL configurado: {config.DATA_NEEDS_REVERSAL}")
        print(f"🔍 [DEBUG FRECUENCIAS FIL] Primeras 5 frecuencias: {config.FREQ[:5]}")
        print(f"🔍 [DEBUG FRECUENCIAS FIL] Últimas 5 frecuencias: {config.FREQ[-5:]}")
        print(f"🔍 [DEBUG FRECUENCIAS FIL] Frecuencia mínima: {config.FREQ.min():.2f} MHz")
        print(f"🔍 [DEBUG FRECUENCIAS FIL] Frecuencia máxima: {config.FREQ.max():.2f} MHz")
        print(f"🔍 [DEBUG FRECUENCIAS FIL] Orden esperado: frecuencias ASCENDENTES (menor a mayor)")
        if config.FREQ[0] < config.FREQ[-1]:
            print(f"✅ [DEBUG FRECUENCIAS FIL] Orden CORRECTO: {config.FREQ[0]:.2f} < {config.FREQ[-1]:.2f}")
        else:
            print(f"❌ [DEBUG FRECUENCIAS FIL] Orden INCORRECTO: {config.FREQ[0]:.2f} > {config.FREQ[-1]:.2f}")
        print(f"🔍 [DEBUG FRECUENCIAS FIL] DOWN_FREQ_RATE: {config.DOWN_FREQ_RATE}")
        print(f"🔍 [DEBUG FRECUENCIAS FIL] DOWN_TIME_RATE: {config.DOWN_TIME_RATE}")
        print("🔍 [DEBUG FRECUENCIAS FIL] " + "="*50)

    # *** ASIGNAR VARIABLES GLOBALES ANTES DEL DEBUG ***
    config.TIME_RESO = tsamp
    config.FREQ_RESO = nchans
    config.FILE_LENG = nsamples

    if config.FREQ_RESO >= 512:
        config.DOWN_FREQ_RATE = max(1, int(round(config.FREQ_RESO / 512)))
    else:
        config.DOWN_FREQ_RATE = 1
    if config.TIME_RESO > 1e-9:
        config.DOWN_TIME_RATE = max(1, int((49.152 * 16 / 1e6) / config.TIME_RESO))
    else:
        config.DOWN_TIME_RATE = 15

    # DEBUG: Información completa del archivo
    if config.DEBUG_FREQUENCY_ORDER:
        print(f"📁 [DEBUG ARCHIVO FIL] Información completa del archivo: {file_name}")
        print(f"📁 [DEBUG ARCHIVO FIL] " + "="*60)
        print(f"📁 [DEBUG ARCHIVO FIL] DIMENSIONES Y RESOLUCIÓN:")
        print(f"📁 [DEBUG ARCHIVO FIL]   - Resolución temporal: {config.TIME_RESO:.2e} segundos/muestra")
        print(f"📁 [DEBUG ARCHIVO FIL]   - Resolución de frecuencia: {config.FREQ_RESO} canales")
        print(f"📁 [DEBUG ARCHIVO FIL]   - Longitud del archivo: {config.FILE_LENG:,} muestras")
        print(f"📁 [DEBUG ARCHIVO FIL]   - Bits por muestra: {nbits}")
        print(f"📁 [DEBUG ARCHIVO FIL]   - Polarizaciones: {nifs}")
        
        # Calcular duración total
        duracion_total_seg = config.FILE_LENG * config.TIME_RESO
        duracion_min = duracion_total_seg / 60
        duracion_horas = duracion_min / 60
        print(f"📁 [DEBUG ARCHIVO FIL]   - Duración total: {duracion_total_seg:.2f} seg ({duracion_min:.2f} min, {duracion_horas:.2f} h)")
        
        print(f"📁 [DEBUG ARCHIVO FIL] FRECUENCIAS:")
        print(f"📁 [DEBUG ARCHIVO FIL]   - Rango total: {config.FREQ.min():.2f} - {config.FREQ.max():.2f} MHz")
        print(f"📁 [DEBUG ARCHIVO FIL]   - Ancho de banda: {abs(config.FREQ.max() - config.FREQ.min()):.2f} MHz")
        print(f"📁 [DEBUG ARCHIVO FIL]   - Resolución por canal: {abs(foff):.4f} MHz/canal")
        print(f"📁 [DEBUG ARCHIVO FIL]   - Orden original: {'DESCENDENTE (foff<0)' if foff < 0 else 'ASCENDENTE (foff>0)'}")
        print(f"📁 [DEBUG ARCHIVO FIL]   - Orden final (post-corrección): {'ASCENDENTE' if config.FREQ[0] < config.FREQ[-1] else 'DESCENDENTE'}")
        
        print(f"📁 [DEBUG ARCHIVO FIL] DECIMACIÓN:")
        print(f"📁 [DEBUG ARCHIVO FIL]   - Factor reducción frecuencia: {config.DOWN_FREQ_RATE}x")
        print(f"📁 [DEBUG ARCHIVO FIL]   - Factor reducción tiempo: {config.DOWN_TIME_RATE}x")
        print(f"📁 [DEBUG ARCHIVO FIL]   - Canales después de decimación: {config.FREQ_RESO // config.DOWN_FREQ_RATE}")
        print(f"📁 [DEBUG ARCHIVO FIL]   - Resolución temporal después: {config.TIME_RESO * config.DOWN_TIME_RATE:.2e} seg/muestra")
        
        # Calcular tamaño aproximado de datos
        size_original_gb = (config.FILE_LENG * config.FREQ_RESO * (nbits/8)) / (1024**3)
        size_decimated_gb = size_original_gb / (config.DOWN_FREQ_RATE * config.DOWN_TIME_RATE)
        print(f"📁 [DEBUG ARCHIVO FIL] TAMAÑO ESTIMADO:")
        print(f"📁 [DEBUG ARCHIVO FIL]   - Datos originales: ~{size_original_gb:.2f} GB")
        print(f"📁 [DEBUG ARCHIVO FIL]   - Datos después decimación: ~{size_decimated_gb:.2f} GB")
        
        
        print(f"📁 [DEBUG ARCHIVO FIL] PROCESAMIENTO:")
        print(f"📁 [DEBUG ARCHIVO FIL]   - Multi-banda habilitado: {'SÍ' if config.USE_MULTI_BAND else 'NO'}")
        print(f"📁 [DEBUG ARCHIVO FIL]   - DM rango: {config.DM_min} - {config.DM_max} pc cm⁻³")
        print(f"📁 [DEBUG ARCHIVO FIL]   - Umbrales: DET_PROB={config.DET_PROB}, CLASS_PROB={config.CLASS_PROB}, SNR_THRESH={config.SNR_THRESH}")
        print(f"📁 [DEBUG ARCHIVO FIL] " + "="*60)

    # DEBUG: Configuración final de decimación
    if config.DEBUG_FREQUENCY_ORDER:
        print(f"⚙️ [DEBUG CONFIG FINAL FIL] Configuración final después de get_obparams_fil:")
        print(f"⚙️ [DEBUG CONFIG FINAL FIL] " + "="*60)
        print(f"⚙️ [DEBUG CONFIG FINAL FIL] DOWN_FREQ_RATE calculado: {config.DOWN_FREQ_RATE}x")
        print(f"⚙️ [DEBUG CONFIG FINAL FIL] DOWN_TIME_RATE calculado: {config.DOWN_TIME_RATE}x")
        print(f"⚙️ [DEBUG CONFIG FINAL FIL] Datos después de decimación:")
        print(f"⚙️ [DEBUG CONFIG FINAL FIL]   - Canales: {config.FREQ_RESO // config.DOWN_FREQ_RATE}")
        print(f"⚙️ [DEBUG CONFIG FINAL FIL]   - Resolución temporal: {config.TIME_RESO * config.DOWN_TIME_RATE:.2e} s/muestra")
        print(f"⚙️ [DEBUG CONFIG FINAL FIL]   - Reducción total de datos: {config.DOWN_FREQ_RATE * config.DOWN_TIME_RATE}x")
        print(f"⚙️ [DEBUG CONFIG FINAL FIL] DATA_NEEDS_REVERSAL final: {config.DATA_NEEDS_REVERSAL}")
        print(f"⚙️ [DEBUG CONFIG FINAL FIL] Orden de frecuencias final: {'ASCENDENTE' if config.FREQ[0] < config.FREQ[-1] else 'DESCENDENTE'}")
        print(f"⚙️ [DEBUG CONFIG FINAL FIL] " + "="*60)

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
        _save_file_debug_info_fil(file_name, {
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


def _save_file_debug_info_fil(file_name: str, debug_info: dict) -> None:
    """Save debug information for a filterbank file to summary.json immediately."""
    try:
        from pathlib import Path

        # Determinar el directorio de guardado
        results_dir = getattr(config, 'RESULTS_DIR', Path('./Results/ObjectDetection'))
        model_dir = results_dir / config.MODEL_NAME
        
        # Asegurar que el directorio existe
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Obtener solo el nombre del archivo sin path
        filename = Path(file_name).stem
        
        # Guardar debug info inmediatamente
        _update_summary_with_file_debug(model_dir, filename, debug_info)
        
    except Exception as e:
        print(f"[WARNING] Error guardando debug info para {file_name}: {e}")


# ...existing code...
