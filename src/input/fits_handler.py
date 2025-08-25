"""Manejo de archivos FITS y PSRFITS para el pipeline de detección de FRBs."""
from __future__ import annotations

import os
from typing import Generator, Tuple, Dict

import numpy as np
from astropy.io import fits

# Optional third-party imports
try:
    import fitsio
except ImportError:
    fitsio = None

# Local imports
from ..config import config
from ..logging import (
    log_stream_fits_block_generation,
    log_stream_fits_load_strategy,
    log_stream_fits_parameters,
    log_stream_fits_summary
)
from .utils import safe_float, safe_int, auto_config_downsampling, print_debug_frequencies, save_file_debug_info


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
                nsubint = safe_int(hdr.get("NAXIS2", 0))
                nchan = safe_int(hdr.get("NCHAN", 0))
                npol = safe_int(hdr.get("NPOL", 0))
                nsblk = safe_int(hdr.get("NSBLK", 1))
                # Validar dimensiones antes de reshape
                if any(x <= 0 for x in [nsubint, nchan, npol, nsblk]):
                    raise ValueError(
                        f"Dimensiones inválidas en header FITS: NAXIS2={nsubint}, NCHAN={nchan}, NPOL={npol}, NSBLK={nsblk} (no pueden ser <= 0)"
                    )
                try:
                    data_array = data_array.reshape(nsubint, nchan, npol, nsblk).swapaxes(1, 2)
                    data_array = data_array.reshape(nsubint * nsblk, npol, nchan)
                    # Selección de polarización al estilo PRESTO: usar Stokes I si está disponible
                    pol_type = str(hdr.get("POL_TYPE", "")).upper() if hdr.get("POL_TYPE") is not None else ""
                    if npol >= 1:
                        if "IQUV" in pol_type:
                            # Orden estándar PSRFITS: I,Q,U,V → índice 0 es Stokes I
                            data_array = data_array[:, 0:1, :]
                        else:
                            # Si no hay POL_TYPE o no es IQUV, usar la primera polarización/IF
                            data_array = data_array[:, 0:1, :]
                    else:
                        # Seguridad: forzar dimensión de pol=1 si npol inválido
                        data_array = data_array.reshape(data_array.shape[0], 1, data_array.shape[-1])
                except Exception as e:
                    raise ValueError(f"Error al hacer reshape de los datos: {e}")
            else:
                if fitsio is None:
                    raise ImportError("fitsio no está instalado. Instale con: pip install fitsio")
                temp_data, h = fitsio.read(file_name, header=True)
                if "DATA" in temp_data.dtype.names:
                    total_samples = safe_int(h.get("NAXIS2", 1)) * safe_int(h.get("NSBLK", 1))
                    num_pols = safe_int(h.get("NPOL", 2))
                    num_chans = safe_int(h.get("NCHAN", 512))
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
                        data_array = temp_data["DATA"].reshape(total_samples, num_pols, num_chans)
                        # Selección de polarización al estilo PRESTO
                        pol_type = str(h.get("POL_TYPE", "")).upper() if h.get("POL_TYPE") is not None else ""
                        if num_pols >= 1:
                            if "IQUV" in pol_type:
                                data_array = data_array[:, 0:1, :]
                            else:
                                data_array = data_array[:, 0:1, :]
                        else:
                            data_array = data_array.reshape(data_array.shape[0], 1, data_array.shape[-1])
                    except Exception as e:
                        raise ValueError(f"Error al hacer reshape de los datos (fitsio): {e}\n  → Los datos no pueden reorganizarse en el formato esperado (tiempo={total_samples}, pol={num_pols}, canal={num_chans})")
                else:
                    total_samples = safe_int(h.get("NAXIS2", 1)) * safe_int(h.get("NSBLK", 1))
                    num_pols = safe_int(h.get("NPOL", 2))
                    num_chans = safe_int(h.get("NCHAN", 512))
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
                        data_array = temp_data.reshape(total_samples, num_pols, num_chans)
                        # Selección de polarización al estilo PRESTO (sin POL_TYPE disponible aquí)
                        data_array = data_array[:, 0:1, :]
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
                        total_samples = safe_int(h.get("NAXIS2", 1)) * safe_int(h.get("NSBLK", 1))
                        num_pols = safe_int(h.get("NPOL", 2))
                        num_chans = safe_int(h.get("NCHAN", raw_data.shape[-1]))
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
                            data_array = raw_data.reshape(total_samples, num_pols, num_chans)
                            pol_type = str(h.get("POL_TYPE", "")).upper() if h.get("POL_TYPE") is not None else ""
                            if num_pols >= 1:
                                if "IQUV" in pol_type:
                                    data_array = data_array[:, 0:1, :]
                                else:
                                    data_array = data_array[:, 0:1, :]
                            else:
                                data_array = data_array.reshape(data_array.shape[0], 1, data_array.shape[-1])
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
            primary = f["PRIMARY"].header if "PRIMARY" in [h.name for h in f] else {}
            sub_data = f["SUBINT"].data
            # Convertir a tipos numéricos explícitamente por si vengan como strings
            config.TIME_RESO = safe_float(hdr.get("TBIN"))
            config.FREQ_RESO = safe_int(hdr.get("NCHAN", 512))
            config.FILE_LENG = (
                safe_int(hdr.get("NSBLK")) * safe_int(hdr.get("NAXIS2"))
            )
            # Guardar parámetros PSRFITS relevantes al estilo PRESTO
            try:
                config.NBITS = safe_int(hdr.get("NBITS", 8))
            except Exception:
                config.NBITS = 8
            try:
                config.NPOL = safe_int(hdr.get("NPOL", 1))
            except Exception:
                config.NPOL = 1
            try:
                config.POL_TYPE = str(hdr.get("POL_TYPE", "")).upper()
            except Exception:
                config.POL_TYPE = ""
            try:
                # Tiempo absoluto de inicio (MJD) como PRESTO
                imjd = safe_int(primary.get("STT_IMJD", 0))
                smjd = safe_float(primary.get("STT_SMJD", 0.0))
                offs = safe_float(primary.get("STT_OFFS", 0.0))
                config.TSTART_MJD = float(imjd) + (float(smjd) + float(offs)) / 86400.0
            except Exception:
                # No disponible → mantener ausente
                pass
            # Desplazamiento de subintegraciones iniciales
            try:
                config.NSUBOFFS = safe_int(hdr.get("NSUBOFFS", 0))
            except Exception:
                config.NSUBOFFS = 0

            try:
                freq_temp = sub_data["DAT_FREQ"][0].astype(np.float64)
            except Exception as e:
                if config.DEBUG_FREQUENCY_ORDER:
                    print(f"[DEBUG HEADER] Error convirtiendo DAT_FREQ: {e}")
                    print("[DEBUG HEADER] Usando rango de frecuencias por defecto")
                nchan = safe_int(hdr.get("NCHAN", 512), 512)
                freq_temp = np.linspace(1000, 1500, nchan)
            
            # DEBUG: Headers PSRFITS específicos
            if config.DEBUG_FREQUENCY_ORDER:
                print(f"[DEBUG HEADER] Headers PSRFITS extraídos:")
                print(
                    f"[DEBUG HEADER]   TBIN (resolución temporal): {safe_float(hdr.get('TBIN')):.2e} s"
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
            
            # Decidir orientación como PRESTO: usar el signo de df=DAT_FREQ[1]-DAT_FREQ[0]
            if len(freq_temp) > 1:
                df = float(freq_temp[1] - freq_temp[0])
                if df < 0:
                    freq_axis_inverted = True
                    if config.DEBUG_FREQUENCY_ORDER:
                        print(f"[DEBUG HEADER] DAT_FREQ descendente → invertir banda (estilo PRESTO)")
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
                        nchan = safe_int(hdr.get("NCHAN", 512), 512)
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
                config.TIME_RESO = safe_float(hdr.get("TBIN"))
                config.FREQ_RESO = safe_int(hdr.get("NCHAN", len(freq_temp)))
                config.FILE_LENG = safe_int(hdr.get("NAXIS2", 0)) * safe_int(hdr.get("NSBLK", 1))
                
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
            # PRESTO marcaría need_flipband, aquí invertimos para mantener orden ascendente interno
            config.FREQ = freq_temp[::-1]
            config.DATA_NEEDS_REVERSAL = True
            try:
                config.NEED_FLIPBAND = True
            except Exception:
                pass
        else:
            config.FREQ = freq_temp
            config.DATA_NEEDS_REVERSAL = False
            try:
                config.NEED_FLIPBAND = False
            except Exception:
                pass

    # DEBUG: Orden de frecuencias
    if config.DEBUG_FREQUENCY_ORDER:
        print_debug_frequencies("[DEBUG FRECUENCIAS]", file_name, freq_axis_inverted)

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
    auto_config_downsampling()

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
        save_file_debug_info(file_name, {
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


def stream_fits(
    file_name: str,
    chunk_samples: int = 2_097_152,
    overlap_samples: int = 0,
) -> Generator[Tuple[np.ndarray, Dict], None, None]:
    """
    Generador que lee un archivo FITS en bloques sin cargar todo en RAM.
    
    Args:
        file_name: Ruta al archivo .fits
        chunk_samples: Número de muestras por bloque (default: 2M)
        overlap_samples: Número de muestras de solapamiento entre bloques
    
    Yields:
        Tuple[data_block, metadata]: Bloque de datos (time, pol, chan) y metadatos
    """
    
    try:
        print(f"[INFO] Streaming datos FITS: chunk_size={chunk_samples}, overlap={overlap_samples}")
        
        # Para archivos FITS, necesitamos cargar el header primero para obtener dimensiones
        # Intentar usar memmap si es posible para archivos grandes
        try:
            with fits.open(file_name, memmap=True) as hdul:
                # Obtener dimensiones del header
                if "SUBINT" in [hdu.name for hdu in hdul] and "DATA" in hdul["SUBINT"].columns.names:
                    subint = hdul["SUBINT"]
                    hdr = subint.header
                    nsubint = safe_int(hdr.get("NAXIS2", 0))
                    nchan = safe_int(hdr.get("NCHAN", 0))
                    npol = safe_int(hdr.get("NPOL", 0))
                    nsblk = safe_int(hdr.get("NSBLK", 1))
                    nsamples = nsubint * nsblk
                    npols = npol
                    nchans = nchan
                else:
                    # Fallback para otros formatos FITS
                    if fitsio is None:
                        raise ImportError("fitsio no está instalado. Instale con: pip install fitsio")
                    temp_data, h = fitsio.read(file_name, header=True)
                    nsamples = safe_int(h.get("NAXIS2", 1)) * safe_int(h.get("NSBLK", 1))
                    npols = safe_int(h.get("NPOL", 2))
                    nchans = safe_int(h.get("NCHAN", 512))
                
                print(f"[INFO] Datos FITS detectados: {nsamples} muestras, {npols} pols, {nchans} canales")
                
                log_stream_fits_parameters(nsamples, chunk_samples, overlap_samples, 
                                         nsubint if 'nsubint' in locals() else None, 
                                         nchan if 'nchan' in locals() else None,
                                         npol if 'npol' in locals() else None, 
                                         nsblk if 'nsblk' in locals() else None)
                
                # Cargar datos completos solo si el archivo no es demasiado grande
                if nsamples * npols * nchans * 4 < 2 * 1024**3:  # < 2GB
                    print(f"[INFO] Archivo FITS pequeño, cargando en memoria")
                    data_array = load_fits_file(file_name)
                    use_memmap = False
                else:
                    print(f"[INFO] Archivo FITS grande, usando memmap para streaming eficiente")
                    # Para archivos grandes, usar memmap del HDU de datos
                    data_hdu = None
                    for hdu in hdul:
                        if hdu.data is not None and hdu.data.ndim >= 3:
                            data_hdu = hdu
                            break
                    
                    if data_hdu is None:
                        raise ValueError("No se encontró HDU con datos válidos")
                    
                    # Crear memmap del HDU de datos
                    data_array = data_hdu.data
                    use_memmap = True
        except Exception as e:
            print(f"[WARN] Fallback a carga completa: {e}")
            data_array = load_fits_file(file_name)
            use_memmap = False
            nsamples, npols, nchans = data_array.shape
        
        # *** DEBUG CRÍTICO: CONFIRMAR ESTRATEGIA DE CARGA ***
        log_stream_fits_load_strategy(use_memmap, data_array.shape, str(data_array.dtype))
        
        # Procesar en bloques
        chunk_counter = 0
        for chunk_start in range(0, nsamples, chunk_samples):
            chunk_counter += 1
            valid_start = chunk_start
            valid_end = min(chunk_start + chunk_samples, nsamples)
            actual_chunk_size = valid_end - valid_start

            # Rango con solapamiento aplicado
            start_with_overlap = max(0, valid_start - overlap_samples)
            end_with_overlap = min(nsamples, valid_end + overlap_samples)

            # Extraer bloque con solapamiento
            if use_memmap:
                # Para memmap, hacer slice directo
                block = data_array[start_with_overlap:end_with_overlap].copy()
            else:
                # Para array en memoria, hacer slice
                block = data_array[start_with_overlap:end_with_overlap].copy()
            
            # *** DEBUG CRÍTICO: CONFIRMAR CADA BLOQUE GENERADO ***
            log_stream_fits_block_generation(chunk_counter, block.shape, str(block.dtype), valid_start, valid_end, start_with_overlap, end_with_overlap, actual_chunk_size)
            
            # Convertir a float32 para consistencia
            if block.dtype != np.float32:
                block = block.astype(np.float32)
            
            # Metadatos del bloque (consistente con stream_fil)
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
                "nifs": npols,  # nifs = npols para FITS
                "dtype": str(block.dtype),
                "shape": block.shape,
                "file_type": "fits"
            }
            
            yield block, metadata
            
            # Limpiar memoria del bloque
            del block
            import gc
            gc.collect()
        
        # *** DEBUG CRÍTICO: CONFIRMAR RESUMEN DE STREAMING ***
        log_stream_fits_summary(chunk_counter)
        
        # Limpiar array principal solo si no es memmap
        if not use_memmap:
            del data_array
            gc.collect()
        
    except Exception as e:
        print(f"[ERROR] Error en stream_fits: {e}")
        raise ValueError(f"No se pudo leer el archivo FITS {file_name}") from e
