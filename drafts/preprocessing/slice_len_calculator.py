"""Slice length calculator for FRB pipeline - dynamically calculates optimal temporal segmentation."""
import logging
from typing import Tuple, Optional
import psutil

from .. import config

logger = logging.getLogger(__name__)


def calculate_slice_len_from_duration() -> Tuple[int, float]:
    """
    Calcula SLICE_LEN dinámicamente basado en SLICE_DURATION_MS y metadatos del archivo.
    
    Fórmula inversa: SLICE_LEN = round(SLICE_DURATION_MS / (TIME_RESO × DOWN_TIME_RATE × 1000))
    
    Returns:
        Tuple[int, float]: (slice_len_calculado, duracion_real_ms)
    """
    if config.TIME_RESO <= 0:
        logger.warning("TIME_RESO no está configurado, usando SLICE_LEN_MIN")
        return config.SLICE_LEN_MIN, config.SLICE_DURATION_MS
    
    # SLICE_LEN se calcula para datos YA decimados
    # Por lo tanto, usar TIME_RESO * DOWN_TIME_RATE
    target_duration_s = config.SLICE_DURATION_MS / 1000.0
    calculated_slice_len = round(target_duration_s / (config.TIME_RESO * config.DOWN_TIME_RATE))
    
    # Aplicar límites mín/máx
    slice_len = max(config.SLICE_LEN_MIN, min(config.SLICE_LEN_MAX, calculated_slice_len))
    
    # Calcular duración real obtenida (para datos decimados)
    real_duration_s = slice_len * config.TIME_RESO * config.DOWN_TIME_RATE
    real_duration_ms = real_duration_s * 1000.0
    
    # Actualizar config.SLICE_LEN con el valor calculado
    config.SLICE_LEN = slice_len
    
    # Solo mostrar información esencial en INFO
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
    chunk_samples = calculate_optimal_chunk_size(slice_len)

    if config.FILE_LENG > 0:
        total_samples = config.FILE_LENG // max(1, config.DOWN_TIME_RATE)
        total_slices = total_samples // slice_len
        leftover_samples = total_samples % slice_len
        total_chunks = (total_samples + chunk_samples - 1) // chunk_samples
        total_duration_sec = config.FILE_LENG * config.TIME_RESO
    else:
        total_samples = 0
        total_slices = 0
        leftover_samples = 0
        total_chunks = 0
        total_duration_sec = 0

    chunk_duration_sec = chunk_samples * config.TIME_RESO * config.DOWN_TIME_RATE
    slices_per_chunk = chunk_samples // slice_len

    parameters = {
        'slice_len': slice_len,
        'slice_duration_ms': real_duration_ms,
        'chunk_samples': chunk_samples,
        'chunk_duration_sec': chunk_duration_sec,
        'slices_per_chunk': slices_per_chunk,
        'total_slices': total_slices,
        'total_chunks': total_chunks,
        'total_duration_sec': total_duration_sec,
        'leftover_samples': leftover_samples,
        'memory_optimized': True
    }

    if leftover_samples > 0:
        logger.info(
            f"Último chunk tendrá {leftover_samples} muestras sin completar un slice"
        )

    return parameters


def update_slice_len_dynamic():
    """
    Actualiza config.SLICE_LEN basado en SLICE_DURATION_MS.
    Debe llamarse después de cargar metadatos del archivo.
    """
    slice_len, real_duration_ms = calculate_slice_len_from_duration()
    
    # Usar el logger global si está disponible
    try:
        from ..logging.logging_config import get_global_logger
        global_logger = get_global_logger()
        global_logger.slice_config({
            'target_ms': config.SLICE_DURATION_MS,
            'slice_len': slice_len,
            'real_ms': real_duration_ms
        })
    except ImportError:
        # Fallback al logger local
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
    
    # Validar slice_len
    if parameters['slice_len'] < config.SLICE_LEN_MIN:
        errors.append(f"slice_len ({parameters['slice_len']}) < mínimo ({config.SLICE_LEN_MIN})")
    
    if parameters['slice_len'] > config.SLICE_LEN_MAX:
        errors.append(f"slice_len ({parameters['slice_len']}) > máximo ({config.SLICE_LEN_MAX})")
    
    # Validar chunk_samples
    if parameters['chunk_samples'] < parameters['slice_len'] * 10:
        errors.append(f"chunk_samples ({parameters['chunk_samples']}) muy pequeño para {parameters['slice_len']} slice_len")
    
    if parameters['chunk_samples'] > 50_000_000:  # 50M muestras máximo
        errors.append(f"chunk_samples ({parameters['chunk_samples']}) muy grande")
    
    # Validar slices_per_chunk
    if parameters['slices_per_chunk'] < 10:
        errors.append(f"slices_per_chunk ({parameters['slices_per_chunk']}) muy pequeño")
    
    if parameters['slices_per_chunk'] > 2000:
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

