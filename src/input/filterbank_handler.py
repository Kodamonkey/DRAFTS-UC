# This module handles SIGPROC filterbank file ingestion.

"""Manage SIGPROC filterbank (.fil) files for the FRB detection pipeline."""
from __future__ import annotations

import gc
import os
import struct
from typing import Dict, Generator, Tuple, Type

import numpy as np
import logging

               
from ..config import config
from ..logging import (
    log_stream_fil_block_generation,
    log_stream_fil_parameters,
    log_stream_fil_summary
)
from .utils import safe_float, safe_int, auto_config_downsampling, print_debug_frequencies, save_file_debug_info


logger = logging.getLogger(__name__)


def _read_int(f) -> int:
    """Read a 32-bit integer from the file."""
    return struct.unpack("<i", f.read(4))[0]


def _read_double(f) -> float:
    """Read a 64-bit floating-point value from the file."""
    return struct.unpack("<d", f.read(8))[0]


def _read_string(f) -> str:
    """Read a length-prefixed string from the file."""
    length = _read_int(f)
    return f.read(length).decode('utf-8', errors='ignore')


def _read_header(f) -> Tuple[dict, int]:
    """Read filterbank header, handling both standard and non-standard formats."""
    original_pos = f.tell()
    
    try:
                                                      
        start = _read_string(f)
        if start != "HEADER_START":
                                                                        
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
                                                              
                    header[key] = _read_int(f)
            except (struct.error, UnicodeDecodeError) as e:
                logger.debug(f"Warning: Error reading header field '{key}': {e}")
                continue
        return header, f.tell()
    except Exception as e:
        logger.debug(f"Error reading standard filterbank header: {e}")
        f.seek(original_pos)
        return _read_non_standard_header(f)


def _read_non_standard_header(f) -> Tuple[dict, int]:
    """Handle non-standard filterbank files by assuming common parameters."""
    logger.info("Detected non-standard .fil file; using estimated parameters")
    
                                          
    current_pos = f.tell()
    f.seek(0, 2)             
    file_size = f.tell()
    f.seek(current_pos)                               
    
                                                 
    header = {
        "nchans": 512,
        "tsamp": 8.192e-5,
        "fch1": 1500.0,
        "foff": -1.0,
        "nbits": 8,
        "nifs": 1,
    }
    
                                                   
    bytes_per_sample = header["nifs"] * header["nchans"] * (header["nbits"] // 8)
    estimated_samples = (file_size - 512) // bytes_per_sample
    max_samples = config.MAX_SAMPLES_LIMIT
    header["nsamples"] = min(estimated_samples, max_samples)
    
    logger.info("Estimated parameters for non-standard file:")
    logger.debug("  - File size: %.1f MB", file_size / (1024**2))
    logger.debug("  - Estimated samples: %d", estimated_samples)
    logger.debug("  - Samples used: %d", header['nsamples'])
    
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

                                 
        dtype_map: Dict[int, Type] = {
            8: np.uint8,
            16: np.int16,
            32: np.float32,
            64: np.float64
        }
        
        dtype = dtype_map.get(nbits, np.uint8)
            
        logger.info(
            "Loading filterbank data: samples=%d, channels=%d, dtype=%s",
            nsamples,
            nchans,
            dtype,
        )
        
                             
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
            logger.warning("Error creating memmap: %s", e)
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
        logger.error("Error loading filterbank file with numpy: %s", e)
        raise ValueError(f"Could not load data from {file_name}: {e}") from e
            
    if data_array is None:
        raise ValueError(f"Could not load data from {file_name}")

    if global_vars.DATA_NEEDS_REVERSAL:
        logger.debug(f">> Reversing frequency axis of loaded data for {file_name}")
        data_array = np.ascontiguousarray(data_array[:, :, ::-1])
    
                                              
    if config.DEBUG_FREQUENCY_ORDER:
        logger.debug(f"[DEBUG FIL DATA] File: {file_name}")
        logger.debug(f"[DEBUG FIL DATA] Data shape: {data_array.shape}")
        logger.debug(f"[DEBUG FIL DATA] Dimensions: (time={data_array.shape[0]}, pol={data_array.shape[1]}, freq={data_array.shape[2]})")
        logger.debug(f"[DEBUG FIL DATA] Data type: {data_array.dtype}")
        logger.debug(f"[DEBUG FIL DATA] Memory size: {data_array.nbytes / (1024**3):.2f} GB")
        logger.debug(f"[DEBUG FIL DATA] Reversal applied: {global_vars.DATA_NEEDS_REVERSAL}")
        logger.debug(f"[DEBUG FIL DATA] Value range: [{data_array.min():.3f}, {data_array.max():.3f}]")
        logger.debug(f"[DEBUG FIL DATA] Mean value: {data_array.mean():.3f}")
        logger.debug(f"[DEBUG FIL DATA] Standard deviation: {data_array.std():.3f}")
        logger.debug("[DEBUG FIL DATA] " + "="*50)
    
    return data_array


def get_obparams_fil(file_name: str) -> None:
    """Extract observation parameters and populate :mod:`config`."""
    
                                               
    if config.DEBUG_FREQUENCY_ORDER:
        logger.debug(f"[DEBUG FILTERBANK] Starting parameter extraction from: {file_name}")
        logger.debug(f"[DEBUG FILTERBANK] " + "="*60)
    
    with open(file_name, "rb") as f:
        freq_axis_inverted = False
        header, hdr_len = _read_header(f)

                                                                       
        nchans = header.get("nchans", 512)                 
        fch1   = header.get("fch1", None)                   
        foff   = header.get("foff", None)                   
        if fch1 is None or foff is None:                     
            raise ValueError(f"Invalid header: missing fch1={fch1} or foff={foff}") 
        freq_temp = fch1 + foff * np.arange(nchans)        

                                                  
        if config.DEBUG_FREQUENCY_ORDER:
            logger.debug(f"[DEBUG FILTERBANK] Filterbank file structure:")
            logger.debug(f"[DEBUG FILTERBANK]   Format: SIGPROC Filterbank (.fil)")
            logger.debug(f"[DEBUG FILTERBANK]   Header size: {hdr_len} bytes")
            logger.debug(f"[DEBUG FILTERBANK] Headers extracted from .fil file:")
            for key, value in header.items():
                logger.debug(f"[DEBUG FILTERBANK]   {key}: {value}")

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
                logger.debug(f"[DEBUG FILTERBANK] nsamples not in header, calculating:")
                logger.debug(f"[DEBUG FILTERBANK]   File size: {file_size} bytes")
                logger.debug(f"[DEBUG FILTERBANK]   Bytes per sample: {bytes_per_sample}")
                logger.debug(f"[DEBUG FILTERBANK]   Calculated samples: {nsamples}")

                                                         
        if config.DEBUG_FREQUENCY_ORDER:
            logger.debug(f"[DEBUG FILTERBANK]   tsamp (temporal resolution): {tsamp:.2e} s")
            logger.debug(f"[DEBUG FILTERBANK]   nchans (channels): {nchans}")
            logger.debug(f"[DEBUG FILTERBANK]   nifs (polarizations): {nifs}")
            logger.debug(f"[DEBUG FILTERBANK]   nbits (bits per sample): {nbits}")
            if 'telescope_id' in header:
                logger.debug(f"[DEBUG FILTERBANK]   telescope_id: {header['telescope_id']}")
            if 'source_name' in header:
                logger.debug(f"[DEBUG FILTERBANK]   Source: {header['source_name']}")
            logger.debug(f"[DEBUG FILTERBANK]   Total samples: {nsamples}")
            
            logger.debug(f"[DEBUG FILTERBANK] Frequency analysis:")
            logger.debug(f"[DEBUG FILTERBANK]   fch1 (initial freq): {fch1} MHz")
            logger.debug(f"[DEBUG FILTERBANK]   foff (channel width): {foff} MHz")
            logger.debug(f"[DEBUG FILTERBANK]   First 5 calculated freq: {freq_temp[:5]}")
            logger.debug(f"[DEBUG FILTERBANK]   Last 5 calculated freq: {freq_temp[-5:]}")
        
                                                              
        if foff < 0:
            freq_axis_inverted = True
            if config.DEBUG_FREQUENCY_ORDER:
                logger.debug(f"[DEBUG FILTERBANK] Negative foff - frequencies inverted!")
        elif len(freq_temp) > 1 and freq_temp[0] > freq_temp[-1]:
            freq_axis_inverted = True
            if config.DEBUG_FREQUENCY_ORDER:
                logger.debug(f"[DEBUG FILTERBANK] Frequencies detected in descending order!")
        else:
            freq_axis_inverted = False
            if config.DEBUG_FREQUENCY_ORDER:
                logger.debug(f"[DEBUG FILTERBANK] Positive foff - keeping ascending frequency order")
        
                                                        
        if freq_axis_inverted:
            config.FREQ = freq_temp[::-1]
            config.DATA_NEEDS_REVERSAL = True
        else:
            config.FREQ = freq_temp
            config.DATA_NEEDS_REVERSAL = False

                                 
    if config.DEBUG_FREQUENCY_ORDER:
        print_debug_frequencies("[DEBUG FRECUENCIAS FIL]", file_name, freq_axis_inverted)

                                                        
    config.TIME_RESO = tsamp
    config.FREQ_RESO = nchans
    config.FILE_LENG = nsamples

                                                                                    
    auto_config_downsampling()

                                             
    if config.DEBUG_FREQUENCY_ORDER:
        logger.debug(f"[DEBUG FIL FILE] Complete file information: {file_name}")
        logger.debug(f"[DEBUG FIL FILE] " + "="*60)
        logger.debug(f"[DEBUG FIL FILE] DIMENSIONS AND RESOLUTION:")
        logger.debug(
            f"[DEBUG FIL FILE]   - Temporal resolution: {config.TIME_RESO:.2e} seconds/sample"
        )
        logger.debug(
            f"[DEBUG FIL FILE]   - Frequency resolution: {config.FREQ_RESO} channels"
        )
        logger.debug(
            f"[DEBUG FIL FILE]   - File length: {config.FILE_LENG:,} samples"
        )
        logger.debug(f"[DEBUG FIL FILE]   - Bits per sample: {nbits}")
        logger.debug(f"[DEBUG FIL FILE]   - Polarisations: {nifs}")
        
                                 
        duracion_total_seg = config.FILE_LENG * config.TIME_RESO
        duracion_min = duracion_total_seg / 60
        duracion_horas = duracion_min / 60
        logger.debug(
            f"[DEBUG FIL FILE]   - Total duration: {duracion_total_seg:.2f} s ({duracion_min:.2f} min, {duracion_horas:.2f} h)"
        )
        
        logger.debug(f"[DEBUG FIL FILE] FREQUENCIES:")
        logger.debug(
            f"[DEBUG FIL FILE]   - Total range: {config.FREQ.min():.2f} - {config.FREQ.max():.2f} MHz"
        )
        logger.debug(
            f"[DEBUG FIL FILE]   - Bandwidth: {abs(config.FREQ.max() - config.FREQ.min()):.2f} MHz"
        )
        logger.debug(
            f"[DEBUG FIL FILE]   - Channel resolution: {abs(foff):.4f} MHz/channel"
        )
        logger.debug(
            f"[DEBUG FIL FILE]   - Original order: {'DESCENDING (foff<0)' if foff < 0 else 'ASCENDING (foff>0)'}"
        )
        logger.debug(
            f"[DEBUG FIL FILE]   - Final order (post-correction): {'ASCENDING' if config.FREQ[0] < config.FREQ[-1] else 'DESCENDING'}"
        )
        
        logger.debug(f"[DEBUG FIL FILE] DECIMATION:")
        logger.debug(
            f"[DEBUG FIL FILE]   - Frequency reduction factor: {config.DOWN_FREQ_RATE}x"
        )
        logger.debug(
            f"[DEBUG FIL FILE]   - Time reduction factor: {config.DOWN_TIME_RATE}x"
        )
        logger.debug(
            f"[DEBUG FIL FILE]   - Channels after decimation: {config.FREQ_RESO // config.DOWN_FREQ_RATE}"
        )
        logger.debug(
            f"[DEBUG FIL FILE]   - Temporal resolution after: {config.TIME_RESO * config.DOWN_TIME_RATE:.2e} s/sample"
        )
        
                                             
        size_original_gb = (config.FILE_LENG * config.FREQ_RESO * (nbits/8)) / (1024**3)
        size_decimated_gb = size_original_gb / (config.DOWN_FREQ_RATE * config.DOWN_TIME_RATE)
        logger.debug(f"[DEBUG FIL FILE] ESTIMATED SIZE:")
        logger.debug(f"[DEBUG FIL FILE]   - Original data: ~{size_original_gb:.2f} GB")
        logger.debug(
            f"[DEBUG FIL FILE]   - Data after decimation: ~{size_decimated_gb:.2f} GB"
        )
        
        
        logger.debug(f"[DEBUG FIL FILE] PROCESSING:")
        logger.debug(
            f"[DEBUG FIL FILE]   - Multi-band enabled: {'YES' if config.USE_MULTI_BAND else 'NO'}"
        )
        logger.debug(
            f"[DEBUG FIL FILE]   - DM range: {config.DM_min} - {config.DM_max} pc cm⁻³"
        )
        logger.debug(
            f"[DEBUG FIL FILE]   - Thresholds: DET_PROB={config.DET_PROB}, CLASS_PROB={config.CLASS_PROB}, SNR_THRESH={config.SNR_THRESH}"
        )
        logger.debug(f"[DEBUG FIL FILE] " + "="*60)

                                              
    if config.DEBUG_FREQUENCY_ORDER:
        logger.debug(
            f"[DEBUG FINAL FIL CONFIG] Final configuration after get_obparams_fil:"
        )
        logger.debug(f"[DEBUG FINAL FIL CONFIG] " + "="*60)
        logger.debug(
            f"[DEBUG FINAL FIL CONFIG] Calculated DOWN_FREQ_RATE: {config.DOWN_FREQ_RATE}x"
        )
        logger.debug(
            f"[DEBUG FINAL FIL CONFIG] Calculated DOWN_TIME_RATE: {config.DOWN_TIME_RATE}x"
        )
        logger.debug("[DEBUG FINAL FIL CONFIG] Data after decimation:")
        logger.debug(
            f"[DEBUG FINAL FIL CONFIG]   - Channels: {config.FREQ_RESO // config.DOWN_FREQ_RATE}"
        )
        logger.debug(
            f"[DEBUG FINAL FIL CONFIG]   - Temporal resolution: {config.TIME_RESO * config.DOWN_TIME_RATE:.2e} s/sample"
        )
        logger.debug(
            f"[DEBUG FINAL FIL CONFIG]   - Total data reduction: {config.DOWN_FREQ_RATE * config.DOWN_TIME_RATE}x"
        )
        logger.debug(
            f"[DEBUG FINAL FIL CONFIG] Final DATA_NEEDS_REVERSAL: {config.DATA_NEEDS_REVERSAL}"
        )
        logger.debug(
            f"[DEBUG FINAL FIL CONFIG] Final frequency order: {'ASCENDING' if config.FREQ[0] < config.FREQ[-1] else 'DESCENDING'}"
        )
        logger.debug(f"[DEBUG FINAL FIL CONFIG] " + "="*60)

    logger.info("Filterbank parameters loaded successfully:")
    logger.debug("  - Channels: %d", nchans)
    logger.debug("  - Time resolution: %.2e s", tsamp)
    logger.debug("  - Start frequency: %.2f MHz", fch1)
    logger.debug("  - Channel width: %.2f MHz", foff)
    logger.debug("  - Samples: %d", nsamples)
    logger.debug("  - Frequency downsampling: %d", config.DOWN_FREQ_RATE)
    logger.debug("  - Time downsampling: %d", config.DOWN_TIME_RATE)

                                                               
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
                "original_order": "DESCENDING (foff<0)" if foff < 0 else "ASCENDING (foff>0)",
                "final_order": "ASCENDING" if config.FREQ[0] < config.FREQ[-1] else "DESCENDING",
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
    """Generator that reads a .fil file in blocks without loading everything into RAM.

    Args:
        file_name: Path to the .fil file
        chunk_samples: Number of samples per block (default: 2M)
        overlap_samples: Number of samples overlapping between blocks

    Yields:
        Tuple[data_block, metadata]: Data block (time, pol, chan) with metadata
    """


    dtype_map: Dict[int, Type] = {
        8: np.uint8,
        16: np.int16,
        32: np.float32,
        64: np.float64
    }

    try:

        if chunk_samples <= 0:
            raise ValueError("chunk_samples must be a positive integer")
        if overlap_samples < 0:
            raise ValueError("overlap_samples must be non-negative")
        if overlap_samples >= chunk_samples:
            raise ValueError("overlap_samples must be smaller than chunk_samples")

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

        if nsamples <= 0:
            raise ValueError(f"Invalid sample count ({nsamples}) derived from file header")
        
        dtype = dtype_map.get(nbits, np.uint8)
        
        logger.info(
            "Streaming filterbank data: total_samples=%d, channels=%d, dtype=%s, chunk_size=%d, overlap=%d",
            nsamples,
            nchans,
            dtype,
            chunk_samples,
            overlap_samples,
        )
        
                                                                      
        log_stream_fil_parameters(nsamples, chunk_samples, overlap_samples, nchans, nifs, nbits, str(dtype))
        
                                            
        data_mmap = np.memmap(
            file_name,
            dtype=dtype,
            mode="r",
            offset=hdr_len,
            shape=(nsamples, nifs, nchans),
        )
        
                             
        chunk_counter = 0
        for chunk_start in range(0, nsamples, chunk_samples):
            chunk_counter += 1
            valid_start = chunk_start
            valid_end = min(chunk_start + chunk_samples, nsamples)
            actual_chunk_size = valid_end - valid_start

                                                        
            start_with_overlap = max(0, valid_start - overlap_samples)
            end_with_overlap = min(nsamples, valid_end + overlap_samples)

                                          
            block = data_mmap[start_with_overlap:end_with_overlap].copy()
            
                                                                   
            log_stream_fil_block_generation(chunk_counter, block.shape, str(block.dtype), valid_start, valid_end, start_with_overlap, end_with_overlap, actual_chunk_size)
            
                                                             
            if config.DATA_NEEDS_REVERSAL:
                block = np.ascontiguousarray(block[:, :, ::-1])
            
                                                   
            if block.dtype != np.float32:
                block = block.astype(np.float32)
            
                                  
            metadata = {
                "chunk_idx": valid_start // chunk_samples,
                "start_sample": valid_start,                                           
                "end_sample": valid_end,                                            
                "actual_chunk_size": actual_chunk_size,                   
                "block_start_sample": start_with_overlap,                                
                "block_end_sample": end_with_overlap,                                 
                "overlap_left": valid_start - start_with_overlap,
                "overlap_right": end_with_overlap - valid_end,
                "total_samples": nsamples,
                "nchans": nchans,
                "nifs": nifs,
                "dtype": str(block.dtype),
                "shape": block.shape
            }
            
            yield block, metadata
            
                             
            del block
            gc.collect()
        
                                                               
        log_stream_fil_summary(chunk_counter)
        
                        
        del data_mmap
        
    except Exception as e:
        logger.debug(f"[DEBUG ERROR] Error in stream_fil: {e}")
        raise ValueError(f"Could not read file {file_name}") from e
