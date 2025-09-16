# This module calculates chunking and slice length parameters.

"""Slice length calculator for FRB pipeline - dynamically calculates optimal temporal segmentation."""
                          
import logging
import math
from typing import Optional, Tuple

                     
import psutil

               
from ..config import config

              
logger = logging.getLogger(__name__)


def calculate_slice_len_from_duration() -> Tuple[int, float]:
    """
    Calcula SLICE_LEN dinámicamente basado en SLICE_DURATION_MS y metadatos del archivo.
    
    Fórmula inversa (dominio decimado):
        dt_ds = TIME_RESO × DOWN_TIME_RATE
        SLICE_LEN = floor( (SLICE_DURATION_MS/1000) / dt_ds + 0.5 )  # round half up estable
    
    Returns:
        Tuple[int, float]: (slice_len_calculado, duracion_real_ms)
    """
    if config.TIME_RESO <= 0 or config.DOWN_TIME_RATE < 1:
        logger.warning("TIME_RESO no está configurado, usando SLICE_LEN_MIN")
        return config.SLICE_LEN_MIN, config.SLICE_DURATION_MS
    
                                                  
                                                   
    target_duration_s = config.SLICE_DURATION_MS / 1000.0
    dt_ds = config.TIME_RESO * config.DOWN_TIME_RATE
                                                                                   
    calculated_slice_len = int(math.floor((target_duration_s / dt_ds) + 0.5))
    
                             
    slice_len = max(config.SLICE_LEN_MIN, min(config.SLICE_LEN_MAX, calculated_slice_len))
    
                                                            
    real_duration_s = slice_len * dt_ds
    real_duration_ms = real_duration_s * 1000.0
    
                                                        
    config.SLICE_LEN = slice_len
    
                                               
    if abs(real_duration_ms - config.SLICE_DURATION_MS) > 5.0:
        logger.warning(f"Diferencia significativa entre objetivo ({config.SLICE_DURATION_MS:.1f} ms) "
                      f"y obtenido ({real_duration_ms:.1f} ms)")
    
    return slice_len, real_duration_ms


def calculate_optimal_chunk_size(slice_len: Optional[int] = None) -> int:
    """Determina cuántas muestras puede contener un chunk.

    El cálculo se basa únicamente en la duración objetivo de cada slice y en la
    memoria disponible del sistema. Si el archivo completo (tras la reducción en
    frecuencia y tiempo) cabe en memoria, se procesa en un solo chunk. De lo
    contrario se calcula el máximo número de muestras que ocupa como mucho el
    25%% de la memoria disponible. El resultado siempre es múltiplo de
    ``slice_len`` para que los slices encajen exactamente.

    Args:
        slice_len: Número de muestras de un slice. Si es ``None`` se calcula a
            partir de :data:`config.SLICE_DURATION_MS`.

    Returns:
        int: muestras por chunk.
    """
    if slice_len is None:
        slice_len, _ = calculate_slice_len_from_duration()

    if config.FILE_LENG <= 0 or config.FREQ_RESO <= 0 or config.TIME_RESO <= 0:
        logger.warning("Metadatos del archivo no disponibles, usando chunk por defecto")
        return slice_len * 200

    total_samples = config.FILE_LENG // max(1, config.DOWN_TIME_RATE)                           
    n_channels = max(1, config.FREQ_RESO // max(1, config.DOWN_FREQ_RATE))                    
    bytes_per_sample = 4 * n_channels                    
    available_bytes = psutil.virtual_memory().available                    
    file_bytes = total_samples * bytes_per_sample

    if file_bytes <= available_bytes * 0.8:
        chunk_samples = total_samples
    else:
        max_samples = int((available_bytes * 0.25) / bytes_per_sample)
        chunk_samples = max(slice_len, min(max_samples, total_samples))

    chunk_samples = (chunk_samples // slice_len) * slice_len
    if chunk_samples == 0:
        chunk_samples = slice_len

    chunk_duration_sec = chunk_samples * config.TIME_RESO * config.DOWN_TIME_RATE
    slices_per_chunk = chunk_samples // slice_len
    logger.info(
        f"Chunk óptimo calculado: {chunk_samples:,} muestras "
        f"({chunk_duration_sec:.1f}s, {slices_per_chunk} slices)"
    )

    return chunk_samples


def get_processing_parameters() -> dict:
    """Calcula automáticamente todos los parámetros de chunking y slicing.

    A partir de :data:`config.SLICE_DURATION_MS` se determina ``slice_len`` y el
    número máximo de muestras que puede manejar un chunk sin agotar la memoria
    disponible.  También se calculan el número total de chunks y slices y las
    muestras residuales que no forman un slice completo.

    Returns:
        dict: Parámetros de procesamiento calculados.
    """
    slice_len, real_duration_ms = calculate_slice_len_from_duration()                                        
                                                                   
    if getattr(config, 'USE_PLANNED_CHUNKING', False):
                                                                         
        down_time_rate = max(1, config.DOWN_TIME_RATE)
        down_freq_rate = max(1, config.DOWN_FREQ_RATE)
        total_channels_downsampled = max(1, config.FREQ_RESO // down_freq_rate)
        bytes_per_sample = 4 * total_channels_downsampled           

                                
        if getattr(config, 'MAX_CHUNK_BYTES', None):
            usable_bytes = config.MAX_CHUNK_BYTES / max(1.0, getattr(config, 'OVERHEAD_FACTOR', 1.3))
        else:
            vm = psutil.virtual_memory()
            usable_bytes = (vm.available * getattr(config, 'MAX_RAM_FRACTION', 0.25)) / max(1.0, getattr(config, 'OVERHEAD_FACTOR', 1.3))

        nsamp_max = max(1, int(usable_bytes // bytes_per_sample))

                                                                         
        if nsamp_max < slice_len:
            chunk_samples = slice_len
        else:
                                                                       
            chunk_samples = (nsamp_max // slice_len) * slice_len
            if chunk_samples == 0:
                chunk_samples = slice_len

                                        
        try:
            from ..logging.chunking_logging import log_chunk_budget
            import psutil as _ps
            vm = _psutil = psutil.virtual_memory()
            log_chunk_budget({
                'bytes_per_sample': bytes_per_sample,
                'available_bytes': vm.available,
                'usable_bytes': usable_bytes,
                'nsamp_max_raw': nsamp_max,
                'nsamp_max_aligned': (nsamp_max // slice_len) * slice_len,
                'slice_len': slice_len,
                'chunk_samples': chunk_samples,
                'down_time_rate': down_time_rate,
                'down_freq_rate': down_freq_rate,
            })
        except Exception:
            pass
    else:
        chunk_samples = calculate_optimal_chunk_size(slice_len)

                                                                                   
                                                 
                                                                                   
    
                                     
    total_samples_original = config.FILE_LENG if config.FILE_LENG > 0 else 0
    total_channels_original = config.FREQ_RESO if config.FREQ_RESO > 0 else 0
    time_reso_original = config.TIME_RESO if config.TIME_RESO > 0 else 0.000064
    
                                         
    down_time_rate = max(1, config.DOWN_TIME_RATE)
    down_freq_rate = max(1, config.DOWN_FREQ_RATE)
    total_samples_downsampled = total_samples_original // down_time_rate
    total_channels_downsampled = total_channels_original // down_freq_rate
    time_reso_downsampled = time_reso_original * down_time_rate
    
                          
    total_duration_sec = total_samples_original * time_reso_original
    total_duration_min = total_duration_sec / 60.0
    
                                              
    samples_per_slice = slice_len
    slices_per_chunk = chunk_samples // slice_len
    total_slices = total_samples_downsampled // slice_len
    leftover_samples = total_samples_downsampled % slice_len
    
                                            
    total_chunks = (total_samples_downsampled + chunk_samples - 1) // chunk_samples
    chunk_duration_sec = chunk_samples * time_reso_downsampled
    
                       
    bytes_per_sample = 4 * total_channels_downsampled           
    total_file_size_gb = (total_samples_downsampled * bytes_per_sample) / (1024**3)
    chunk_size_gb = (chunk_samples * bytes_per_sample) / (1024**3)
    
                        
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    
                                                  
    can_load_full_file = total_file_size_gb <= available_memory_gb * 0.8

                                                        
                                                                                      
                                                                                         
                                                                                          
    stream_chunk_samples_original = chunk_samples
    stream_chunk_samples_decimated = max(1, stream_chunk_samples_original // down_time_rate)
    slices_per_chunk_stream_estimate = stream_chunk_samples_decimated // slice_len
    chunk_duration_stream_sec = stream_chunk_samples_original * time_reso_original
    
    parameters = {
                                
        'slice_duration_ms_target': config.SLICE_DURATION_MS,
        'down_time_rate': down_time_rate,
        'down_freq_rate': down_freq_rate,
        
                                         
        'total_samples_original': total_samples_original,
        'total_channels_original': total_channels_original,
        'time_reso_original': time_reso_original,
        'total_duration_sec': total_duration_sec,
        'total_duration_min': total_duration_min,
        
                                             
        'total_samples_downsampled': total_samples_downsampled,
        'total_channels_downsampled': total_channels_downsampled,
        'time_reso_downsampled': time_reso_downsampled,
        
                               
        'slice_len': slice_len,
        'slice_duration_ms_real': real_duration_ms,
        'samples_per_slice': samples_per_slice,
                                                                
                                                                
        'chunk_samples_stream_original': stream_chunk_samples_original,
        'chunk_samples_stream_decimated': stream_chunk_samples_decimated,
        'slices_per_chunk_stream_estimate': slices_per_chunk_stream_estimate,
        'chunk_duration_stream_sec': chunk_duration_stream_sec,
        'chunk_samples': chunk_samples,
        'chunk_duration_sec': chunk_duration_sec,
        'slices_per_chunk': min(slices_per_chunk, total_slices),
        'total_chunks': total_chunks,
        'total_slices': total_slices,
        'leftover_samples': leftover_samples,
        
                                
        'total_file_size_gb': total_file_size_gb,
        'chunk_size_gb': chunk_size_gb,
        'available_memory_gb': available_memory_gb,
        'total_memory_gb': total_memory_gb,
        'can_load_full_file': can_load_full_file,
        
                         
        'memory_optimized': True,
        'has_leftover_samples': leftover_samples > 0,
        
                                                               
        'slice_duration_ms': real_duration_ms,                                            
        'total_duration_sec': total_duration_sec,                                            
    }

    return parameters


def update_slice_len_dynamic():
    """
    Actualiza config.SLICE_LEN basado en SLICE_DURATION_MS.
    Debe llamarse después de cargar metadatos del archivo.
    """
    slice_len, real_duration_ms = calculate_slice_len_from_duration()
    
                                              
    try:
        from ..logging.logging_config import get_global_logger
        global_logger = get_global_logger()
        global_logger.slice_config({
            'target_ms': config.SLICE_DURATION_MS,
            'slice_len': slice_len,
            'real_ms': real_duration_ms
        })
    except ImportError:
                                  
        logger.info(f"Slice configurado: {slice_len} muestras = {real_duration_ms:.1f} ms")
    
    return slice_len, real_duration_ms


def validate_processing_parameters(parameters: dict) -> bool:
    """
    Valida que los parámetros calculados sean razonables.
    
    Args:
        parameters: Diccionario con parámetros de procesamiento
        
    Returns:
        bool: True si los parámetros son válidos
    """
    errors = []
    
                       
    if parameters['slice_len'] < config.SLICE_LEN_MIN:
        errors.append(f"slice_len ({parameters['slice_len']}) < mínimo ({config.SLICE_LEN_MIN})")
    
    if parameters['slice_len'] > config.SLICE_LEN_MAX:
        errors.append(f"slice_len ({parameters['slice_len']}) > máximo ({config.SLICE_LEN_MAX})")
    
                           
    if parameters['chunk_samples'] < parameters['slice_len'] * 10:
        errors.append(f"chunk_samples ({parameters['chunk_samples']}) muy pequeño para {parameters['slice_len']} slice_len")
    
    if parameters['chunk_samples'] > 50_000_000:                       
        errors.append(f"chunk_samples ({parameters['chunk_samples']}) muy grande")
    
                              
    if parameters['slices_per_chunk'] < 5:
        errors.append(f"slices_per_chunk ({parameters['slices_per_chunk']}) muy pequeño")
    
    if parameters['slices_per_chunk'] > 5000:                                            
        errors.append(f"slices_per_chunk ({parameters['slices_per_chunk']}) muy grande")

    leftover = parameters.get('leftover_samples', 0)
    if leftover >= parameters['slice_len']:
        errors.append(
            f"leftover_samples ({leftover}) debería ser menor que slice_len ({parameters['slice_len']})"
        )
    
    if errors:
        logger.error("Parámetros de procesamiento inválidos:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    return True

