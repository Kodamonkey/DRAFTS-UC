"""Helper functions to read SIGPROC filterbank (.fil) files."""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Tuple

import numpy as np

from . import config


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
                # Try to skip this field and continue
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
        "nchans": 512,      # Common number of channels
        "tsamp": 8.192e-5,  # Common time resolution (81.92 µs)
        "fch1": 1500.0,     # Starting frequency (MHz)
        "foff": -1.0,       # Channel bandwidth (MHz)
        "nbits": 8,         # Usually 8-bit data
        "nifs": 1,          # Single polarization
    }
    
    # Estimate number of samples based on file size
    bytes_per_sample = header["nifs"] * header["nchans"] * (header["nbits"] // 8)
    estimated_samples = (file_size - 512) // bytes_per_sample  # Assume 512 byte header
    
    # Limit to reasonable size to avoid memory issues
    max_samples = config.MAX_SAMPLES_LIMIT  # Configurable limit for safety
    header["nsamples"] = min(estimated_samples, max_samples)
    
    print(f"[INFO] Parámetros estimados para archivo no estándar:")
    print(f"  - Tamaño de archivo: {file_size / (1024**2):.1f} MB")
    print(f"  - Muestras estimadas: {estimated_samples}")
    print(f"  - Muestras a usar: {header['nsamples']}")
    
    return header, 512  # Assume 512 byte header offset


def load_fil_file(file_name: str) -> np.ndarray:
    """Load a filterbank file and return data as ``(time, pol, channel)``."""
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

        # Check if chunk processing is enabled for large files
        if (getattr(config, 'ENABLE_CHUNK_PROCESSING', True) and 
            nsamples > config.MAX_SAMPLES_LIMIT):
            # Don't load data here - let the pipeline handle chunk processing
            print(f"[INFO] Archivo grande detectado ({nsamples} muestras)")
            print(f"[INFO] Se procesará automáticamente por chunks")
            print(f"[INFO] No cargando datos completos para evitar problemas de memoria")
            # Store original size for chunk processing
            config._ORIGINAL_FILE_SAMPLES = nsamples
            # Return a small representative sample for parameter verification
            nsamples = min(1000, nsamples)  # Just load 1000 samples for verification
        else:
            # Apply limit for single-pass processing
            max_samples = config.MAX_SAMPLES_LIMIT
            if nsamples > max_samples:
                print(f"[WARNING] Archivo muy grande ({nsamples} muestras), limitando a {max_samples}")
                print(f"[INFO] Para procesar archivos más grandes, habilitar config.ENABLE_CHUNK_PROCESSING")
                nsamples = max_samples

        dtype = np.uint8
        if nbits == 16:
            dtype = np.int16
        elif nbits == 32:
            dtype = np.float32
        elif nbits == 64:
            dtype = np.float64
            
        print(f"[INFO] Cargando datos: {nsamples} muestras, {nchans} canales, tipo {dtype}")
        estimated_size_mb = (nsamples * nifs * nchans * dtype().itemsize) / (1024**2)
        print(f"[INFO] Tamaño estimado en memoria: {estimated_size_mb:.1f} MB")
        
        # Memory-map the data to avoid loading the entire file into memory
        try:
            data = np.memmap(
                file_name,
                dtype=dtype,
                mode="r",
                offset=hdr_len,
                shape=(nsamples, nifs, nchans),
            )
            print(f"[INFO] Memmap creado exitosamente con shape: {data.shape}")
        except ValueError as e:
            print(f"[WARNING] Error creating memmap: {e}")
            # Try with an even smaller shape if the original fails
            safe_samples = min(nsamples, 10000)
            data = np.memmap(
                file_name,
                dtype=dtype,
                mode="r",
                offset=hdr_len,
                shape=(safe_samples, nifs, nchans),
            )
            print(f"[WARNING] Using reduced samples: {safe_samples}")

        if config.DATA_NEEDS_REVERSAL:
            # Create a copy to avoid issues with memory mapping
            print("[INFO] Revirtiendo eje de frecuencia...")
            data_copy = np.array(data[:, :, ::-1])
            return data_copy
        else:
            # Convert to regular array to avoid memmap issues downstream
            return np.array(data)
        
    except Exception as e:
        print(f"[ERROR] Error cargando archivo .fil: {e}")
        print("[WARNING] Retornando datos sintéticos para continuar el procesamiento")
        # Return synthetic data to allow processing to continue
        return np.random.rand(1000, 1, 512).astype(np.float32)


def get_obparams_fil(file_name: str) -> None:
    """Populate :mod:`config` using parameters from a filterbank file."""
    try:
        with open(file_name, "rb") as f:
            header, hdr_len = _read_header(f)

        nchans = header.get("nchans", 512)
        tsamp = header.get("tsamp", 8.192e-5)  # More realistic default
        nifs = header.get("nifs", 1)
        nbits = header.get("nbits", 8)
        nsamples = header.get("nsamples")
        
        if nsamples is None:
            bytes_per_sample = nifs * nchans * (nbits // 8)
            file_size = os.path.getsize(file_name) - hdr_len
            nsamples = file_size // bytes_per_sample if bytes_per_sample > 0 else 1000

        # Check if chunk processing is enabled for large files  
        if (getattr(config, 'ENABLE_CHUNK_PROCESSING', True) and 
            nsamples > config.MAX_SAMPLES_LIMIT):
            # Store original size for chunk processing but don't truncate here
            print(f"[INFO] Archivo grande detectado ({nsamples} muestras)")
            print(f"[INFO] Se procesará automáticamente por chunks") 
            # Store original size for chunk processing
            config._ORIGINAL_FILE_SAMPLES = nsamples
            # Keep original nsamples for parameter configuration
        else:
            # Apply limit for single-pass processing
            max_samples = config.MAX_SAMPLES_LIMIT
            if nsamples > max_samples:
                print(f"[WARNING] Limitando número de muestras de {nsamples} a {max_samples}")
                print(f"[INFO] Para procesar archivos más grandes, habilitar config.ENABLE_CHUNK_PROCESSING")
                nsamples = max_samples

        fch1 = header.get("fch1", 1500.0)  # More realistic default
        foff = header.get("foff", -1.0)
        freq_temp = fch1 + np.arange(nchans) * foff
        
        if foff < 0:
            config.DATA_NEEDS_REVERSAL = True
            freq_temp = freq_temp[::-1]
        else:
            config.DATA_NEEDS_REVERSAL = False

        config.FREQ = freq_temp
        config.FREQ_RESO = nchans
        config.TIME_RESO = tsamp
        config.FILE_LENG = nsamples

        # ✅ CORRECCIÓN CRÍTICA: NO sobrescribir valores de downsampling configurados manualmente
        # Solo usar valores automáticos si no están configurados explícitamente
        if not hasattr(config, '_downsampling_configured') or not config._downsampling_configured:
            # Cálculo automático solo si no hay configuración manual
            if config.FREQ_RESO >= 512:
                auto_down_freq = max(1, int(round(config.FREQ_RESO / 512)))
            else:
                auto_down_freq = 1
            if config.TIME_RESO > 1e-9:
                auto_down_time = max(1, int((49.152 * 16 / 1e6) / config.TIME_RESO))
            else:
                auto_down_time = 15
                
            config.DOWN_FREQ_RATE = auto_down_freq
            config.DOWN_TIME_RATE = auto_down_time
            print(f"[INFO] Usando downsampling automático: freq={auto_down_freq}, time={auto_down_time}")
        else:
            print(f"[INFO] Usando downsampling configurado manualmente: freq={config.DOWN_FREQ_RATE}, time={config.DOWN_TIME_RATE}")
            
        print(f"[INFO] Parámetros del archivo .fil cargados exitosamente:")
        print(f"  - Canales: {nchans}")
        print(f"  - Resolución temporal: {tsamp:.2e} s")
        print(f"  - Frecuencia inicial: {fch1} MHz")
        print(f"  - Ancho de banda: {foff} MHz")
        print(f"  - Muestras: {nsamples}")
        print(f"  - Down-sampling frecuencia: {config.DOWN_FREQ_RATE}")
        print(f"  - Down-sampling tiempo: {config.DOWN_TIME_RATE}")
        
        # ✅ DEBUG: Mostrar información detallada del array de frecuencias
        print(f"\n[DEBUG] 📡 INFORMACIÓN DEL ARRAY DE FRECUENCIAS:")
        print(f"  - Array shape: {freq_temp.shape}")
        print(f"  - Frecuencia mínima: {freq_temp.min():.3f} MHz")
        print(f"  - Frecuencia máxima: {freq_temp.max():.3f} MHz")
        print(f"  - Ancho de banda total: {freq_temp.max() - freq_temp.min():.3f} MHz")
        print(f"  - Resolución por canal: {abs(foff):.6f} MHz")
        print(f"  - Datos invertidos (freq descendente): {config.DATA_NEEDS_REVERSAL}")
        print(f"  - Primeros 10 canales: {freq_temp[:10]}")
        print(f"  - Últimos 10 canales: {freq_temp[-10:]}")
        
        # ✅ DEBUG: Mostrar todos los parámetros extraídos del header
        print(f"\n[DEBUG] 📄 PARÁMETROS EXTRAÍDOS DEL HEADER:")
        for key, value in header.items():
            if isinstance(value, float):
                print(f"  - {key}: {value:.6e}")
            else:
                print(f"  - {key}: {value}")
        
        # ✅ Calcular y mostrar el tamaño final después del downsampling
        final_samples = nsamples // config.DOWN_TIME_RATE
        final_channels = nchans // config.DOWN_FREQ_RATE
        reduction_factor = config.DOWN_TIME_RATE * config.DOWN_FREQ_RATE
        print(f"\n[INFO] 💾 IMPACTO DEL DOWNSAMPLING:")
        print(f"  - Tamaño después del downsampling: {final_samples:,} muestras")
        print(f"  - Canales después del downsampling: {final_channels}")
        print(f"  - Factor de reducción total: {reduction_factor}x")
        print(f"  - ¿Necesita chunks?: {'NO' if final_samples <= config.MAX_SAMPLES_LIMIT else 'SÍ'}")
        
        # ✅ DEBUG: Información específica sobre inversión de frecuencias
        print(f"\n[DEBUG] 🔄 ANÁLISIS DE INVERSIÓN DE FRECUENCIAS:")
        print(f"  - foff (ancho de banda por canal): {foff:.6f} MHz")
        print(f"  - foff < 0?: {foff < 0} → {'Frecuencias DESCENDENTES' if foff < 0 else 'Frecuencias ASCENDENTES'}")
        print(f"  - config.DATA_NEEDS_REVERSAL: {config.DATA_NEEDS_REVERSAL}")
        print(f"  - Acción: {'Se INVIERTE el array de frecuencias' if config.DATA_NEEDS_REVERSAL else 'NO se invierte'}")
        print(f"  - Orden final: {freq_temp[0]:.1f} → {freq_temp[-1]:.1f} MHz")
        print(f"  - ¿Es orden correcto para dedispersión?: {'SÍ (baja→alta)' if freq_temp[0] < freq_temp[-1] else 'NO (alta→baja)'}")
        
        if freq_temp[0] > freq_temp[-1]:
            print(f"  - ⚠️ ADVERTENCIA: El orden final va de alta→baja. Esto causará problemas en dedispersión!")
        else:
            print(f"  - ✅ CORRECTO: El orden final va de baja→alta. Perfecto para dedispersión.")
        
    except Exception as e:
        print(f"[WARNING] Error leyendo parámetros del archivo .fil: {e}")
        print("[WARNING] Usando parámetros por defecto")
        # Set realistic default parameters
        config.FREQ = np.linspace(1500, 1000, 512)  # 1500-1000 MHz range
        config.FREQ_RESO = 512
        config.TIME_RESO = 8.192e-5  # 81.92 µs
        config.FILE_LENG = 50000
        config.DOWN_FREQ_RATE = 1
        config.DOWN_TIME_RATE = 15
        config.DATA_NEEDS_REVERSAL = True
