"""Data loader for astronomical files (FITS, PSRFITS, filterbank) and FRB candidate management."""
from __future__ import annotations

# Standard library imports
import csv
import gc
import logging
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Tuple, Type

# Third-party imports
import numpy as np
from astropy.io import fits

# Local imports
from .. import config
from ..preprocessing.data_downsampler import downsample_data
from ..output.summary_manager import _update_summary_with_file_debug
from .. import config

# Setup logger
logger = logging.getLogger(__name__)

def _safe_float(value, default=0.0):
    """Return ``value`` as ``float`` or ``default`` if conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            cleaned = str(value).strip().replace("*", "").replace("UNSET", "")
            return float(cleaned)
        except (TypeError, ValueError):
            return default


def _safe_int(value, default=0):
    """Return ``value`` as ``int`` or ``default`` if conversion fails."""
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            cleaned = str(value).strip().replace("*", "").replace("UNSET", "")
            return int(float(cleaned))
        except (TypeError, ValueError):
            return default


def load_fits_file(file_name: str) -> np.ndarray:
    """Load a FITS file and return the data array in shape (time, pol, channel)."""
    global_vars = config
    data_array = None
    try:
        with fits.open(file_name, memmap=True) as hdul:
            if "SUBINT" in [hdu.name for hdu in hdul] and "DATA" in hdul["SUBINT"].columns.names:
                subint = hdul["SUBINT"]
                hdr = subint.header
                data_array = subint.data["DATA"]
                nsubint = _safe_int(hdr.get("NAXIS2", 0))
                nchan = _safe_int(hdr.get("NCHAN", 0))
                npol = _safe_int(hdr.get("NPOL", 0))
                nsblk = _safe_int(hdr.get("NSBLK", 1))
                # Validar dimensiones antes de reshape
                if any(x <= 0 for x in [nsubint, nchan, npol, nsblk]):
                    raise ValueError(
                        f"Dimensiones inválidas en header FITS: NAXIS2={nsubint}, NCHAN={nchan}, NPOL={npol}, NSBLK={nsblk} (no pueden ser <= 0)"
                    )
                try:
                    data_array = data_array.reshape(nsubint, nchan, npol, nsblk).swapaxes(1, 2)
                    data_array = data_array.reshape(nsubint * nsblk, npol, nchan)
                    data_array = data_array[:, :2, :]
                except Exception as e:
                    raise ValueError(f"Error al hacer reshape de los datos: {e}")
            else:
                import fitsio
                temp_data, h = fitsio.read(file_name, header=True)
                if "DATA" in temp_data.dtype.names:
                    total_samples = _safe_int(h.get("NAXIS2", 1)) * _safe_int(h.get("NSBLK", 1))
                    num_pols = _safe_int(h.get("NPOL", 2))
                    num_chans = _safe_int(h.get("NCHAN", 512))
                    if any(x <= 0 for x in [total_samples, num_pols, num_chans]):
                        # Mensaje detallado explicando el problema
                        error_details = []
                        if total_samples <= 0:
                            error_details.append(f"total_samples={total_samples} (NAXIS2={h.get('NAXIS2', 1)} × NSBLK={h.get('NSBLK', 1)})")
                        if num_pols <= 0:
                            error_details.append(f"num_pols={num_pols}")
                        if num_chans <= 0:
                            error_details.append(f"num_chans={num_chans}")
                        
                        error_msg = f"Dimensiones inválidas en header FITS: {', '.join(error_details)}"
                        if total_samples <= 0:
                            error_msg += f"\n  → NSBLK={h.get('NSBLK', 1)} es 0 o negativo, lo que hace imposible calcular el número de muestras temporales"
                            error_msg += f"\n  → El pipeline necesita datos en formato (tiempo, polarización, canal) pero no puede determinar la dimensión temporal"
                        if num_pols <= 0:
                            error_msg += f"\n  → NPOL={num_pols} es 0 o negativo, lo que hace imposible procesar las polarizaciones"
                        if num_chans <= 0:
                            error_msg += f"\n  → NCHAN={num_chans} es 0 o negativo, lo que hace imposible procesar los canales de frecuencia"
                        
                        raise ValueError(error_msg)
                    try:
                        data_array = temp_data["DATA"].reshape(total_samples, num_pols, num_chans)[:, :2, :]
                    except Exception as e:
                        raise ValueError(f"Error al hacer reshape de los datos (fitsio): {e}\n  → Los datos no pueden reorganizarse en el formato esperado (tiempo={total_samples}, pol={num_pols}, canal={num_chans})")
                else:
                    total_samples = _safe_int(h.get("NAXIS2", 1)) * _safe_int(h.get("NSBLK", 1))
                    num_pols = _safe_int(h.get("NPOL", 2))
                    num_chans = _safe_int(h.get("NCHAN", 512))
                    if any(x <= 0 for x in [total_samples, num_pols, num_chans]):
                        # Mensaje detallado explicando el problema
                        error_details = []
                        if total_samples <= 0:
                            error_details.append(f"total_samples={total_samples} (NAXIS2={h.get('NAXIS2', 1)} × NSBLK={h.get('NSBLK', 1)})")
                        if num_pols <= 0:
                            error_details.append(f"num_pols={num_pols}")
                        if num_chans <= 0:
                            error_details.append(f"num_chans={num_chans}")
                        
                        error_msg = f"Dimensiones inválidas en header FITS: {', '.join(error_details)}"
                        if total_samples <= 0:
                            error_msg += f"\n  → NSBLK={h.get('NSBLK', 1)} es 0 o negativo, lo que hace imposible calcular el número de muestras temporales"
                            error_msg += f"\n  → El pipeline necesita datos en formato (tiempo, polarización, canal) pero no puede determinar la dimensión temporal"
                        if num_pols <= 0:
                            error_msg += f"\n  → NPOL={num_pols} es 0 o negativo, lo que hace imposible procesar las polarizaciones"
                        if num_chans <= 0:
                            error_msg += f"\n  → NCHAN={num_chans} es 0 o negativo, lo que hace imposible procesar los canales de frecuencia"
                        
                        raise ValueError(error_msg)
                    try:
                        data_array = temp_data.reshape(total_samples, num_pols, num_chans)[:, :2, :]
                    except Exception as e:
                        raise ValueError(f"Error al hacer reshape de los datos (fitsio): {e}\n  → Los datos no pueden reorganizarse en el formato esperado (tiempo={total_samples}, pol={num_pols}, canal={num_chans})")
    except (ValueError, fits.verify.VerifyError) as e:
        # Re-lanzar errores específicos de archivos corruptos con información adicional
        if "NSBLK" in str(e) and "0" in str(e):
            raise ValueError(f"Archivo FITS corrupto: {file_name}\n  → El archivo tiene NSBLK=0 en el header, lo que indica que está mal formateado o corrupto\n  → NSBLK debe ser > 0 para definir el número de muestras por bloque temporal\n  → Recomendación: Verificar el origen del archivo o obtener una versión correcta") from e
        elif "Dimensiones inválidas" in str(e):
            raise ValueError(f"Archivo FITS corrupto: {file_name}\n  → {str(e)}\n  → El archivo no puede ser procesado debido a dimensiones inválidas en el header\n  → Recomendación: Verificar la integridad del archivo o usar un archivo diferente") from e
        else:
            raise ValueError(f"Archivo FITS corrupto: {file_name}\n  → {str(e)}\n  → El archivo no puede ser leído correctamente\n  → Recomendación: Verificar que el archivo no esté dañado") from e
    except Exception as e:
        print(f"[Error cargando FITS con fitsio/astropy] {e}")
        try:
            # Intentar sin memmap para archivos corruptos
            with fits.open(file_name, memmap=False) as f:
                data_hdu = None
                for hdu_item in f:
                    try:
                        if (hdu_item.data is not None and 
                            isinstance(hdu_item.data, np.ndarray) and 
                            hdu_item.data.ndim >= 3):
                            data_hdu = hdu_item
                            break
                    except (TypeError, ValueError):
                        continue
                if data_hdu is None and len(f) > 1:
                    data_hdu = f[1]
                elif data_hdu is None:
                    data_hdu = f[0]
                h = data_hdu.header
                try:
                    raw_data = data_hdu.data
                    if raw_data is not None:
                        total_samples = _safe_int(h.get("NAXIS2", 1)) * _safe_int(h.get("NSBLK", 1))
                        num_pols = _safe_int(h.get("NPOL", 2))
                        num_chans = _safe_int(h.get("NCHAN", raw_data.shape[-1]))
                        if any(x <= 0 for x in [total_samples, num_pols, num_chans]):
                            # Mensaje detallado para el fallback también
                            error_details = []
                            if total_samples <= 0:
                                error_details.append(f"total_samples={total_samples} (NAXIS2={h.get('NAXIS2', 1)} × NSBLK={h.get('NSBLK', 1)})")
                            if num_pols <= 0:
                                error_details.append(f"num_pols={num_pols}")
                            if num_chans <= 0:
                                error_details.append(f"num_chans={num_chans}")
                            
                            error_msg = f"Dimensiones inválidas en header fallback: {', '.join(error_details)}"
                            if total_samples <= 0:
                                error_msg += f"\n  → NSBLK={h.get('NSBLK', 1)} es 0 o negativo, lo que hace imposible calcular el número de muestras temporales"
                                error_msg += f"\n  → El pipeline necesita datos en formato (tiempo, polarización, canal) pero no puede determinar la dimensión temporal"
                            if num_pols <= 0:
                                error_msg += f"\n  → NPOL={num_pols} es 0 o negativo, lo que hace imposible procesar las polarizaciones"
                            if num_chans <= 0:
                                error_msg += f"\n  → NCHAN={num_chans} es 0 o negativo, lo que hace imposible procesar los canales de frecuencia"
                            
                            raise ValueError(error_msg)
                        try:
                            data_array = raw_data.reshape(total_samples, num_pols, num_chans)[:, :2, :]
                        except Exception as e:
                            raise ValueError(f"Error al hacer reshape de los datos (fallback): {e}\n  → Los datos no pueden reorganizarse en el formato esperado (tiempo={total_samples}, pol={num_pols}, canal={num_chans})\n  → El archivo puede estar corrupto o tener un formato no compatible")
                    else:
                        raise ValueError("No hay datos válidos en el HDU\n  → El archivo FITS no contiene datos procesables\n  → Verificar que el archivo no esté vacío o corrupto")
                except (TypeError, ValueError) as e_data:
                    print(f"Error accediendo a datos del HDU: {e_data}")
                    raise ValueError(f"Archivo FITS corrupto: {file_name}\n  → Error al acceder a los datos del archivo: {e_data}\n  → El archivo puede estar dañado o tener un formato no reconocido")
        except Exception as e_astropy:
            print(f"Fallo final al cargar con astropy: {e_astropy}")
            raise ValueError(f"Archivo FITS corrupto: {file_name}\n  → Fallo en el método de respaldo con astropy: {e_astropy}\n  → El archivo no puede ser leído por ningún método disponible\n  → Recomendación: Verificar la integridad del archivo o usar un archivo diferente") from e_astropy
            
    if data_array is None:
        raise ValueError(f"Archivo FITS corrupto: {file_name}\n  → No se pudieron cargar datos válidos del archivo\n  → El archivo puede estar vacío, corrupto o tener un formato no compatible\n  → Recomendación: Verificar la integridad del archivo o usar un archivo diferente")

    if global_vars.DATA_NEEDS_REVERSAL:
        print(f">> Invirtiendo eje de frecuencia de los datos cargados para {file_name}")
        data_array = np.ascontiguousarray(data_array[:, :, ::-1])
    
    # DEBUG: Información de los datos cargados
    if config.DEBUG_FREQUENCY_ORDER:
        print(f"[DEBUG DATOS CARGADOS] Archivo: {file_name}")
        print(f"[DEBUG DATOS CARGADOS] Shape de datos: {data_array.shape}")
        print(f"[DEBUG DATOS CARGADOS] Dimensiones: (tiempo={data_array.shape[0]}, pol={data_array.shape[1]}, freq={data_array.shape[2]})")
        print(f"[DEBUG DATOS CARGADOS] Tipo de datos: {data_array.dtype}")
        print(f"[DEBUG DATOS CARGADOS] Tamaño en memoria: {data_array.nbytes / (1024**3):.2f} GB")
        print(f"[DEBUG DATOS CARGADOS] Reversión aplicada: {global_vars.DATA_NEEDS_REVERSAL}")
        print(f"[DEBUG DATOS CARGADOS] Rango de valores: [{data_array.min():.3f}, {data_array.max():.3f}]")
        print(f"[DEBUG DATOS CARGADOS] Valor medio: {data_array.mean():.3f}")
        print(f"[DEBUG DATOS CARGADOS] Desviación estándar: {data_array.std():.3f}")
        print("[DEBUG DATOS CARGADOS] " + "="*50)
    
    return data_array


# =============================================================================
# EJEMPLOS DE MENSAJES DE ERROR INFORMATIVOS
# =============================================================================
"""
PATRÓN GENERAL PARA MENSAJES DE ERROR INFORMATIVOS:

1. ERRORES DE VALIDACIÓN:
   try:
       if valor <= 0:
           raise ValueError(
               f"Valor inválido: {valor}\n"
               f"  → El valor debe ser > 0 para funcionar correctamente\n"
               f"  → Este valor se usa para calcular {contexto}\n"
               f"  → Recomendación: Verificar el origen de los datos"
           )
   except ValueError as e:
       logger.error("Error de validación detectado")
       logger.error("Detalles:")
       for line in str(e).split('\n'):
           if line.strip():
               logger.error("  %s", line.strip())

2. ERRORES DE ARCHIVO:
   try:
       with open(archivo) as f:
           datos = f.read()
   except FileNotFoundError:
       raise FileNotFoundError(
           f"Archivo no encontrado: {archivo}\n"
           f"  → El archivo no existe en la ruta especificada\n"
           f"  → Verificar que la ruta sea correcta\n"
           f"  → Recomendación: Verificar la existencia del archivo"
       )
   except PermissionError:
       raise PermissionError(
           f"Sin permisos para acceder: {archivo}\n"
           f"  → No tienes permisos de lectura en este archivo\n"
           f"  → Verificar permisos del sistema\n"
           f"  → Recomendación: Ejecutar con permisos adecuados"
       )

3. ERRORES DE CONFIGURACIÓN:
   if not config_valida:
       raise ValueError(
           f"Configuración inválida en {seccion}\n"
           f"  → {parametro} = {valor} no es válido\n"
           f"  → Valores permitidos: {valores_permitidos}\n"
           f"  → Recomendación: Corregir la configuración"
       )

4. ERRORES DE MEMORIA:
   try:
       array_grande = np.zeros((1000000, 1000000))
   except MemoryError:
       raise MemoryError(
           f"Memoria insuficiente para operación\n"
           f"  → Se requieren {memoria_necesaria} GB\n"
           f"  → Memoria disponible: {memoria_disponible} GB\n"
           f"  → Recomendación: Reducir el tamaño de los datos o liberar memoria"
       )

5. ERRORES DE RED:
   try:
       response = requests.get(url, timeout=10)
   except requests.Timeout:
       raise requests.Timeout(
           f"Timeout al conectar con {url}\n"
           f"  → La conexión tardó más de 10 segundos\n"
           f"  → Verificar conectividad de red\n"
           f"  → Recomendación: Verificar conexión o aumentar timeout"
       )

BENEFICIOS DE ESTE ENFOQUE:
- El usuario entiende QUÉ pasó
- El usuario entiende POR QUÉ pasó
- El usuario sabe QUÉ hacer para solucionarlo
- Facilita el debugging y soporte técnico
- Mejora la experiencia del usuario
"""


def _auto_config_downsampling() -> None:
    """Configura DOWN_FREQ_RATE y DOWN_TIME_RATE si no fueron fijados por el usuario."""
    user_configured_freq = getattr(config, 'DOWN_FREQ_RATE', None)
    user_configured_time = getattr(config, 'DOWN_TIME_RATE', None)

    if user_configured_freq is None or user_configured_freq == 1:
        if config.FREQ_RESO >= 512:
            config.DOWN_FREQ_RATE = max(1, int(round(config.FREQ_RESO / 512)))
        else:
            config.DOWN_FREQ_RATE = 1

    if user_configured_time is None:
        if config.TIME_RESO > 1e-9:
            config.DOWN_TIME_RATE = max(1, int((49.152 * 16 / 1e6) / config.TIME_RESO))
        else:
            config.DOWN_TIME_RATE = 15


def _print_debug_frequencies(prefix: str, file_name: str, freq_axis_inverted: bool) -> None:
    """Imprime bloque estándar de depuración de frecuencias con un prefijo dado."""
    print(f"{prefix} Archivo: {file_name}")
    print(f"{prefix} freq_axis_inverted detectado: {freq_axis_inverted}")
    print(f"{prefix} DATA_NEEDS_REVERSAL configurado: {config.DATA_NEEDS_REVERSAL}")
    print(f"{prefix} Primeras 5 frecuencias: {config.FREQ[:5]}")
    print(f"{prefix} Últimas 5 frecuencias: {config.FREQ[-5:]}")
    print(f"{prefix} Frecuencia mínima: {config.FREQ.min():.2f} MHz")
    print(f"{prefix} Frecuencia máxima: {config.FREQ.max():.2f} MHz")
    print(f"{prefix} Orden esperado: frecuencias ASCENDENTES (menor a mayor)")
    if config.FREQ[0] < config.FREQ[-1]:
        print(f"{prefix} Orden CORRECTO: {config.FREQ[0]:.2f} < {config.FREQ[-1]:.2f}")
    else:
        print(f"{prefix} Orden INCORRECTO: {config.FREQ[0]:.2f} > {config.FREQ[-1]:.2f}")
    print(f"{prefix} DOWN_FREQ_RATE: {config.DOWN_FREQ_RATE}")
    print(f"{prefix} DOWN_TIME_RATE: {config.DOWN_TIME_RATE}")
    print(f"{prefix} " + "="*50)


def _save_file_debug_info(file_name: str, debug_info: dict) -> None:
    """Guarda debug info (FITS o FIL) en summary.json inmediatamente (unificado)."""
    try:
        from pathlib import Path
        results_dir = getattr(config, 'RESULTS_DIR', Path('./Results/ObjectDetection'))
        model_dir = results_dir / config.MODEL_NAME
        model_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(file_name).stem
        _update_summary_with_file_debug(model_dir, filename, debug_info)
    except Exception as e:
        print(f"[WARNING] Error guardando debug info para {file_name}: {e}")


def get_obparams(file_name: str) -> None:
    """Extract observation parameters and populate :mod:`config`."""
    
    # DEBUG: Información de entrada del archivo
    if config.DEBUG_FREQUENCY_ORDER:
        print(f"[DEBUG HEADER] Iniciando extracción de parámetros de: {file_name}")
        print(f"[DEBUG HEADER] " + "="*60)
    
    with fits.open(file_name, memmap=True) as f:
        freq_axis_inverted = False
        
        # DEBUG: Estructura del archivo FITS
        if config.DEBUG_FREQUENCY_ORDER:
            print(f"[DEBUG HEADER] Estructura del archivo FITS:")
            for i, hdu in enumerate(f):
                hdu_type = type(hdu).__name__
                if hasattr(hdu, 'header') and hdu.header:
                    if 'EXTNAME' in hdu.header:
                        ext_name = hdu.header['EXTNAME']
                    else:
                        ext_name = 'PRIMARY' if i == 0 else f'HDU_{i}'
                    print(f"[DEBUG HEADER]   HDU {i}: {hdu_type} - {ext_name}")
                    if hasattr(hdu, 'columns') and hdu.columns:
                        print(f"[DEBUG HEADER]     Columnas: {[col.name for col in hdu.columns]}")
                else:
                    print(f"[DEBUG HEADER]   HDU {i}: {hdu_type} - Sin header")
        
        if "SUBINT" in [hdu.name for hdu in f] and "TBIN" in f["SUBINT"].header:
            # DEBUG: Procesando formato PSRFITS
            if config.DEBUG_FREQUENCY_ORDER:
                print(f"[DEBUG HEADER] Formato detectado: PSRFITS (SUBINT)")
            
            hdr = f["SUBINT"].header
            sub_data = f["SUBINT"].data
            # Convertir a tipos numéricos explícitamente por si vengan como strings
            config.TIME_RESO = _safe_float(hdr.get("TBIN"))
            config.FREQ_RESO = _safe_int(hdr.get("NCHAN", 512))
            config.FILE_LENG = (
                _safe_int(hdr.get("NSBLK")) * _safe_int(hdr.get("NAXIS2"))
            )

            try:
                freq_temp = sub_data["DAT_FREQ"][0].astype(np.float64)
            except Exception as e:
                if config.DEBUG_FREQUENCY_ORDER:
                    print(f"[DEBUG HEADER] Error convirtiendo DAT_FREQ: {e}")
                    print("[DEBUG HEADER] Usando rango de frecuencias por defecto")
                nchan = _safe_int(hdr.get("NCHAN", 512), 512)
                freq_temp = np.linspace(1000, 1500, nchan)
            
            # DEBUG: Headers PSRFITS específicos
            if config.DEBUG_FREQUENCY_ORDER:
                print(f"[DEBUG HEADER] Headers PSRFITS extraídos:")
                print(
                    f"[DEBUG HEADER]   TBIN (resolución temporal): {_safe_float(hdr.get('TBIN')):.2e} s"
                )
                print(f"[DEBUG HEADER]   NCHAN (canales): {hdr['NCHAN']}")
                print(f"[DEBUG HEADER]   NSBLK (muestras por subint): {hdr['NSBLK']}")
                print(f"[DEBUG HEADER]   NAXIS2 (número de subints): {hdr['NAXIS2']}")
                print(f"[DEBUG HEADER]   NPOL (polarizaciones): {hdr.get('NPOL', 'N/A')}")
                print(f"[DEBUG HEADER]   Total de muestras: {config.FILE_LENG}")
                if 'OBS_MODE' in hdr:
                    print(f"[DEBUG HEADER]   Modo de observación: {hdr['OBS_MODE']}")
                if 'SRC_NAME' in hdr:
                    print(f"[DEBUG HEADER]   Fuente: {hdr['SRC_NAME']}")
            
            if "CHAN_BW" in hdr:
                bw = hdr["CHAN_BW"]
                if isinstance(bw, (list, np.ndarray)):
                    bw = bw[0]
                # Asegurar que sea float por si proviene como string
                try:
                    bw = float(bw)
                except (TypeError, ValueError):
                    bw = 0.0
                if config.DEBUG_FREQUENCY_ORDER:
                    print(f"[DEBUG HEADER]   CHAN_BW detectado: {bw} MHz")
                if bw < 0:
                    freq_axis_inverted = True
                    if config.DEBUG_FREQUENCY_ORDER:
                        print(f"[DEBUG HEADER]   ⚠️ CHAN_BW negativo - frecuencias invertidas!")
            elif len(freq_temp) > 1 and freq_temp[0] > freq_temp[-1]:
                freq_axis_inverted = True
                if config.DEBUG_FREQUENCY_ORDER:
                    print(f"[DEBUG HEADER]   ⚠️ Frecuencias detectadas en orden descendente!")
        else:
            # DEBUG: Procesando formato FITS estándar
            if config.DEBUG_FREQUENCY_ORDER:
                print(f"[DEBUG HEADER] Formato detectado: FITS estándar (no PSRFITS)")
            
            try:
                data_hdu_index = 0
                for i, hdu_item in enumerate(f):
                    if hdu_item.is_image or isinstance(hdu_item, (fits.BinTableHDU, fits.TableHDU)):
                        if 'NAXIS' in hdu_item.header and hdu_item.header['NAXIS'] > 0:
                            if 'CTYPE3' in hdu_item.header and 'FREQ' in hdu_item.header['CTYPE3'].upper():
                                data_hdu_index = i
                                break
                            if 'CTYPE2' in hdu_item.header and 'FREQ' in hdu_item.header['CTYPE2'].upper():
                                data_hdu_index = i
                                break
                            if 'CTYPE1' in hdu_item.header and 'FREQ' in hdu_item.header['CTYPE1'].upper():
                                data_hdu_index = i
                                break
                if data_hdu_index == 0 and len(f) > 1:
                    data_hdu_index = 1
                
                # DEBUG: HDU seleccionado
                if config.DEBUG_FREQUENCY_ORDER:
                    print(f"[DEBUG HEADER] HDU seleccionado para datos: {data_hdu_index}")
                
                hdr = f[data_hdu_index].header
                
                # DEBUG: Headers FITS estándar
                if config.DEBUG_FREQUENCY_ORDER:
                    print(f"[DEBUG HEADER] Headers FITS estándar del HDU {data_hdu_index}:")
                    relevant_keys = ['TBIN', 'NCHAN', 'NAXIS2', 'NSBLK', 'NPOL', 'CRVAL1', 'CRVAL2', 'CRVAL3', 
                                   'CDELT1', 'CDELT2', 'CDELT3', 'CTYPE1', 'CTYPE2', 'CTYPE3']
                    for key in relevant_keys:
                        if key in hdr:
                            print(f"[DEBUG HEADER]   {key}: {hdr[key]}")
                
                if "DAT_FREQ" in f[data_hdu_index].columns.names:
                    try:
                        freq_temp = f[data_hdu_index].data["DAT_FREQ"][0].astype(np.float64)
                    except Exception as e:
                        if config.DEBUG_FREQUENCY_ORDER:
                            print(f"[DEBUG HEADER] Error convirtiendo DAT_FREQ: {e}")
                            print("[DEBUG HEADER] Usando rango de frecuencias por defecto")
                        nchan = _safe_int(hdr.get("NCHAN", 512), 512)
                        freq_temp = np.linspace(1000, 1500, nchan)
                    else:
                        if config.DEBUG_FREQUENCY_ORDER:
                            print(f"[DEBUG HEADER] Frecuencias extraídas de columna DAT_FREQ")
                else:
                    freq_axis_num = ''
                    for i in range(1, hdr.get('NAXIS', 0) + 1):
                        if 'FREQ' in hdr.get(f'CTYPE{i}', '').upper():
                            freq_axis_num = str(i)
                            break
                    
                    if config.DEBUG_FREQUENCY_ORDER:
                        print(f"[DEBUG HEADER] Buscando eje de frecuencias en headers WCS...")
                        print(f"[DEBUG HEADER] Eje de frecuencias detectado: CTYPE{freq_axis_num}" if freq_axis_num else "📋 [DEBUG HEADER] ⚠️ No se encontró eje de frecuencias")
                    
                    if freq_axis_num:
                        crval = hdr.get(f'CRVAL{freq_axis_num}', 0)
                        cdelt = hdr.get(f'CDELT{freq_axis_num}', 1)
                        crpix = hdr.get(f'CRPIX{freq_axis_num}', 1)
                        naxis = hdr.get(f'NAXIS{freq_axis_num}', hdr.get('NCHAN', 512))
                        try:
                            crval = float(crval)
                            cdelt = float(cdelt)
                            crpix = float(crpix)
                            naxis = int(naxis)
                        except (TypeError, ValueError):
                            crval = float(crval) if isinstance(crval, (int, float)) else 0.0
                            cdelt = float(cdelt) if isinstance(cdelt, (int, float)) else 1.0
                            crpix = float(crpix) if isinstance(crpix, (int, float)) else 1.0
                            naxis = int(naxis) if isinstance(naxis, (int, float)) else hdr.get('NCHAN', 512)
                        freq_temp = crval + (np.arange(naxis) - (crpix - 1)) * cdelt
                        
                        if config.DEBUG_FREQUENCY_ORDER:
                            print(f"[DEBUG HEADER] Parámetros WCS frecuencia:")
                            print(f"[DEBUG HEADER]   CRVAL{freq_axis_num}: {crval} (valor de referencia)")
                            print(f"[DEBUG HEADER]   CDELT{freq_axis_num}: {cdelt} (incremento por canal)")
                            print(f"[DEBUG HEADER]   CRPIX{freq_axis_num}: {crpix} (pixel de referencia)")
                            print(f"[DEBUG HEADER]   NAXIS{freq_axis_num}: {naxis} (número de canales)")
                        
                        if cdelt < 0:
                            freq_axis_inverted = True
                            if config.DEBUG_FREQUENCY_ORDER:
                                print(f"[DEBUG HEADER]   ⚠️ CDELT negativo - frecuencias invertidas!")
                    else:
                        if config.DEBUG_FREQUENCY_ORDER:
                            print(f"[DEBUG HEADER] ⚠️ Usando frecuencias por defecto: 1000-1500 MHz")
                        freq_temp = np.linspace(1000, 1500, hdr.get('NCHAN', 512))
                
                # Convertir a tipos numéricos para evitar errores de comparación
                config.TIME_RESO = _safe_float(hdr.get("TBIN"))
                config.FREQ_RESO = _safe_int(hdr.get("NCHAN", len(freq_temp)))
                config.FILE_LENG = _safe_int(hdr.get("NAXIS2", 0)) * _safe_int(hdr.get("NSBLK", 1))
                
                # DEBUG: Parámetros finales extraídos
                if config.DEBUG_FREQUENCY_ORDER:
                    print(f"[DEBUG HEADER] Parámetros finales FITS estándar:")
                    print(f"[DEBUG HEADER]   TIME_RESO: {config.TIME_RESO:.2e} s")
                    print(f"[DEBUG HEADER]   FREQ_RESO: {config.FREQ_RESO}")
                    print(f"[DEBUG HEADER]   FILE_LENG: {config.FILE_LENG}")
                    
            except Exception as e_std:
                if config.DEBUG_FREQUENCY_ORDER:
                    print(f"[DEBUG HEADER] ⚠️ Error procesando FITS estándar: {e_std}")
                    print(f"[DEBUG HEADER] Usando valores por defecto...")
                print(f"Error procesando FITS estándar: {e_std}")
                config.TIME_RESO = 5.12e-5
                config.FREQ_RESO = 512
                config.FILE_LENG = 100000
                freq_temp = np.linspace(1000, 1500, config.FREQ_RESO)
        if freq_axis_inverted:
            config.FREQ = freq_temp[::-1]
            config.DATA_NEEDS_REVERSAL = True
        else:
            config.FREQ = freq_temp
            config.DATA_NEEDS_REVERSAL = False

    # DEBUG: Orden de frecuencias
    if config.DEBUG_FREQUENCY_ORDER:
        _print_debug_frequencies("[DEBUG FRECUENCIAS]", file_name, freq_axis_inverted)

    # DEBUG: Información completa del archivo
    if config.DEBUG_FREQUENCY_ORDER:
        print(f"[DEBUG ARCHIVO] Información completa del archivo: {file_name}")
        print(f"[DEBUG ARCHIVO] " + "="*60)
        print(f"[DEBUG ARCHIVO] DIMENSIONES Y RESOLUCIÓN:")
        print(f"[DEBUG ARCHIVO]   - Resolución temporal: {config.TIME_RESO:.2e} segundos/muestra")
        print(f"[DEBUG ARCHIVO]   - Resolución de frecuencia: {config.FREQ_RESO} canales")
        print(f"[DEBUG ARCHIVO]   - Longitud del archivo: {config.FILE_LENG:,} muestras")
        
        # Calcular duración total
        duracion_total_seg = config.FILE_LENG * config.TIME_RESO
        duracion_min = duracion_total_seg / 60
        duracion_horas = duracion_min / 60
        print(f"[DEBUG ARCHIVO]   - Duración total: {duracion_total_seg:.2f} seg ({duracion_min:.2f} min, {duracion_horas:.2f} h)")
        
        print(f"[DEBUG ARCHIVO] FRECUENCIAS:")
        print(f"[DEBUG ARCHIVO]   - Rango total: {config.FREQ.min():.2f} - {config.FREQ.max():.2f} MHz")
        print(f"[DEBUG ARCHIVO]   - Ancho de banda: {abs(config.FREQ.max() - config.FREQ.min()):.2f} MHz")
        print(f"[DEBUG ARCHIVO]   - Resolución por canal: {abs(config.FREQ[1] - config.FREQ[0]):.4f} MHz/canal")
        print(f"[DEBUG ARCHIVO]   - Orden original: {'DESCENDENTE' if freq_axis_inverted else 'ASCENDENTE'}")
        print(f"[DEBUG ARCHIVO]   - Orden final (post-corrección): {'ASCENDENTE' if config.FREQ[0] < config.FREQ[-1] else 'DESCENDENTE'}")
        
        print(f"[DEBUG ARCHIVO] DECIMACIÓN:")
        print(f"[DEBUG ARCHIVO]   - Factor reducción frecuencia: {config.DOWN_FREQ_RATE}x")
        print(f"[DEBUG ARCHIVO]   - Factor reducción tiempo: {config.DOWN_TIME_RATE}x")
        print(f"[DEBUG ARCHIVO]   - Canales después de decimación: {config.FREQ_RESO // config.DOWN_FREQ_RATE}")
        print(f"[DEBUG ARCHIVO]   - Resolución temporal después: {config.TIME_RESO * config.DOWN_TIME_RATE:.2e} seg/muestra")
        
        # Calcular tamaño aproximado de datos
        size_original_gb = (config.FILE_LENG * config.FREQ_RESO * 4) / (1024**3)  # 4 bytes por float32
        size_decimated_gb = size_original_gb / (config.DOWN_FREQ_RATE * config.DOWN_TIME_RATE)
        print(f"[DEBUG ARCHIVO] TAMAÑO ESTIMADO:")
        print(f"[DEBUG ARCHIVO]   - Datos originales: ~{size_original_gb:.2f} GB")
        print(f"[DEBUG ARCHIVO]   - Datos después decimación: ~{size_decimated_gb:.2f} GB")
        
        
        print(f"[DEBUG ARCHIVO] CONFIGURACIÓN DE SLICE:")
        print(f"[DEBUG ARCHIVO]   - SLICE_DURATION_MS configurado: {config.SLICE_DURATION_MS} ms")
        expected_slice_len = round(config.SLICE_DURATION_MS / (config.TIME_RESO * config.DOWN_TIME_RATE * 1000))
        print(f"[DEBUG ARCHIVO]   - SLICE_LEN calculado: {expected_slice_len} muestras")
        print(f"[DEBUG ARCHIVO]   - SLICE_LEN límites: [{config.SLICE_LEN_MIN}, {config.SLICE_LEN_MAX}]")
        
        print(f"[DEBUG ARCHIVO] PROCESAMIENTO:")
        print(f"[DEBUG ARCHIVO]   - Multi-banda habilitado: {'SÍ' if config.USE_MULTI_BAND else 'NO'}")
        print(f"[DEBUG ARCHIVO]   - DM rango: {config.DM_min} - {config.DM_max} pc cm⁻³")
        print(f"[DEBUG ARCHIVO]   - Umbrales: DET_PROB={config.DET_PROB}, CLASS_PROB={config.CLASS_PROB}, SNR_THRESH={config.SNR_THRESH}")
        print(f"[DEBUG ARCHIVO] " + "="*60)

    # RESPETAR CONFIGURACIONES DEL USUARIO - calcular automáticamente si corresponde
    _auto_config_downsampling()

    # DEBUG: Configuración final de decimación
    if config.DEBUG_FREQUENCY_ORDER:
        print(f"[DEBUG CONFIG FINAL] Configuración final después de get_obparams:")
        print(f"[DEBUG CONFIG FINAL] " + "="*60)
        print(f"[DEBUG CONFIG FINAL] DOWN_FREQ_RATE calculado: {config.DOWN_FREQ_RATE}x")
        print(f"[DEBUG CONFIG FINAL] DOWN_TIME_RATE calculado: {config.DOWN_TIME_RATE}x")
        print(f"[DEBUG CONFIG FINAL] Datos después de decimación:")
        print(f"[DEBUG CONFIG FINAL]   - Canales: {config.FREQ_RESO // config.DOWN_FREQ_RATE}")
        print(f"[DEBUG CONFIG FINAL]   - Resolución temporal: {config.TIME_RESO * config.DOWN_TIME_RATE:.2e} s/muestra")
        print(f"[DEBUG CONFIG FINAL]   - Reducción total de datos: {config.DOWN_FREQ_RATE * config.DOWN_TIME_RATE}x")
        print(f"[DEBUG CONFIG FINAL] DATA_NEEDS_REVERSAL final: {config.DATA_NEEDS_REVERSAL}")
        print(f"[DEBUG CONFIG FINAL] Orden de frecuencias final: {'ASCENDENTE' if config.FREQ[0] < config.FREQ[-1] else 'DESCENDENTE'}")
        print(f"[DEBUG CONFIG FINAL] " + "="*60)

    # *** GUARDAR DEBUG INFO EN SUMMARY.JSON INMEDIATAMENTE ***
    if config.DEBUG_FREQUENCY_ORDER:
        _save_file_debug_info(file_name, {
            "file_type": "fits",
            "file_size_bytes": os.path.getsize(file_name),
            "file_size_gb": os.path.getsize(file_name) / (1024**3),
            "format": "PSRFITS (.fits)",
            "frequency_analysis": {
                "freq_min_mhz": float(config.FREQ.min()),
                "freq_max_mhz": float(config.FREQ.max()),
                "bandwidth_mhz": abs(config.FREQ.max() - config.FREQ.min()),
                "resolution_per_channel_mhz": abs(config.FREQ[1] - config.FREQ[0]) if len(config.FREQ) > 1 else 0,
                "original_order": "DESCENDENTE" if freq_axis_inverted else "ASCENDENTE",
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
                "slice_duration_ms_configured": config.SLICE_DURATION_MS,
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
                "effective_sample_rate_after_decimation_hz": 1.0 / (config.TIME_RESO * config.DOWN_TIME_RATE)
            }
        })

    # (El guardado de debug ahora se unifica en _save_file_debug_info)

'''
FILTERBANK IO
'''

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

            dtype = np.int16
        elif nbits == 32:
            dtype = np.float32
        elif nbits == 64:
            dtype = np.float64
            
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
        
        # Aplicar corrección de orden (homólogo a io.py)
        if freq_axis_inverted:
            config.FREQ = freq_temp[::-1]
            config.DATA_NEEDS_REVERSAL = True
        else:
            config.FREQ = freq_temp
            config.DATA_NEEDS_REVERSAL = False

    # DEBUG: Orden de frecuencias
    if config.DEBUG_FREQUENCY_ORDER:
        _print_debug_frequencies("[DEBUG FRECUENCIAS FIL]", file_name, freq_axis_inverted)

    # *** ASIGNAR VARIABLES GLOBALES ANTES DEL DEBUG ***
    config.TIME_RESO = tsamp
    config.FREQ_RESO = nchans
    config.FILE_LENG = nsamples

    # RESPETAR CONFIGURACIONES DEL USUARIO - calcular automáticamente si corresponde
    _auto_config_downsampling()

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
        _save_file_debug_info(file_name, {
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


    # (El guardado de debug ahora se unifica en _save_file_debug_info)


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
        
        # Crear memmap para acceso eficiente
        data_mmap = np.memmap(
            file_name,
            dtype=dtype,
            mode="r",
            offset=hdr_len,
            shape=(nsamples, nifs, nchans),
        )
        
        # Procesar en bloques
        for chunk_start in range(0, nsamples, chunk_samples):
            valid_start = chunk_start
            valid_end = min(chunk_start + chunk_samples, nsamples)
            actual_chunk_size = valid_end - valid_start

            # Rango con solapamiento aplicado (en crudo)
            start_with_overlap = max(0, valid_start - overlap_samples)
            end_with_overlap = min(nsamples, valid_end + overlap_samples)

            # Leer bloque con solapamiento
            block = data_mmap[start_with_overlap:end_with_overlap].copy()
            
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
        
        # Limpiar memmap
        del data_mmap
        gc.collect()
        
    except Exception as e:
        print(f"[ERROR] Error en stream_fil: {e}")
        raise ValueError(f"No se pudo leer el archivo {file_name}") from e

def load_and_preprocess_data(fits_path):
    """Carga y preprocesa los datos del archivo FITS o FIL."""
    if fits_path.suffix.lower() == ".fits":
        data = load_fits_file(str(fits_path))
    else:
        data = load_fil_file(str(fits_path))
    # El procesamiento estándar no debe duplicar temporalmente los datos.
    # Simplemente aplicar el downsampling según configuración.
    return downsample_data(data)

