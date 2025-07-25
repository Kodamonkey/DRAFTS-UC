"""Input/output helpers for PSRFITS and standard FITS files."""
from __future__ import annotations

import os
from typing import List

import numpy as np
from astropy.io import fits


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

from . import config
from .summary_utils import _update_summary_with_file_debug


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
                        raise ValueError(
                            f"Dimensiones inválidas en header FITSIO: NAXIS2={h.get('NAXIS2', 1)}, NSBLK={h.get('NSBLK', 1)}, NPOL={num_pols}, NCHAN={num_chans} (no pueden ser <= 0)"
                        )
                    try:
                        data_array = temp_data["DATA"].reshape(total_samples, num_pols, num_chans)[:, :2, :]
                    except Exception as e:
                        raise ValueError(f"Error al hacer reshape de los datos (fitsio): {e}")
                else:
                    total_samples = _safe_int(h.get("NAXIS2", 1)) * _safe_int(h.get("NSBLK", 1))
                    num_pols = _safe_int(h.get("NPOL", 2))
                    num_chans = _safe_int(h.get("NCHAN", 512))
                    if any(x <= 0 for x in [total_samples, num_pols, num_chans]):
                        raise ValueError(
                            f"Dimensiones inválidas en header FITSIO: NAXIS2={h.get('NAXIS2', 1)}, NSBLK={h.get('NSBLK', 1)}, NPOL={num_pols}, NCHAN={num_chans} (no pueden ser <= 0)"
                        )
                    try:
                        data_array = temp_data.reshape(total_samples, num_pols, num_chans)[:, :2, :]
                    except Exception as e:
                        raise ValueError(f"Error al hacer reshape de los datos (fitsio): {e}")
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
                            raise ValueError(
                                f"Dimensiones inválidas en header fallback: NAXIS2={h.get('NAXIS2', 1)}, NSBLK={h.get('NSBLK', 1)}, NPOL={num_pols}, NCHAN={num_chans} (no pueden ser <= 0)"
                            )
                        try:
                            data_array = raw_data.reshape(total_samples, num_pols, num_chans)[:, :2, :]
                        except Exception as e:
                            raise ValueError(f"Error al hacer reshape de los datos (fallback): {e}")
                    else:
                        raise ValueError("No hay datos válidos en el HDU")
                except (TypeError, ValueError) as e_data:
                    print(f"Error accediendo a datos del HDU: {e_data}")
                    raise ValueError(f"Archivo FITS corrupto: {file_name}")
        except Exception as e_astropy:
            print(f"Fallo final al cargar con astropy: {e_astropy}")
            raise ValueError(f"Archivo FITS corrupto: {file_name}") from e_astropy
            
    if data_array is None:
        raise ValueError(f"Archivo FITS corrupto: {file_name}")

    if global_vars.DATA_NEEDS_REVERSAL:
        print(f">> Invirtiendo eje de frecuencia de los datos cargados para {file_name}")
        data_array = np.ascontiguousarray(data_array[:, :, ::-1])
    
    # DEBUG: Información de los datos cargados
    if config.DEBUG_FREQUENCY_ORDER:
        print(f"💾 [DEBUG DATOS CARGADOS] Archivo: {file_name}")
        print(f"💾 [DEBUG DATOS CARGADOS] Shape de datos: {data_array.shape}")
        print(f"💾 [DEBUG DATOS CARGADOS] Dimensiones: (tiempo={data_array.shape[0]}, pol={data_array.shape[1]}, freq={data_array.shape[2]})")
        print(f"💾 [DEBUG DATOS CARGADOS] Tipo de datos: {data_array.dtype}")
        print(f"💾 [DEBUG DATOS CARGADOS] Tamaño en memoria: {data_array.nbytes / (1024**3):.2f} GB")
        print(f"💾 [DEBUG DATOS CARGADOS] Reversión aplicada: {global_vars.DATA_NEEDS_REVERSAL}")
        print(f"💾 [DEBUG DATOS CARGADOS] Rango de valores: [{data_array.min():.3f}, {data_array.max():.3f}]")
        print(f"💾 [DEBUG DATOS CARGADOS] Valor medio: {data_array.mean():.3f}")
        print(f"💾 [DEBUG DATOS CARGADOS] Desviación estándar: {data_array.std():.3f}")
        print("💾 [DEBUG DATOS CARGADOS] " + "="*50)
    
    return data_array


def get_obparams(file_name: str) -> None:
    """Extract observation parameters and populate :mod:`config`."""
    
    # DEBUG: Información de entrada del archivo
    if config.DEBUG_FREQUENCY_ORDER:
        print(f"📋 [DEBUG HEADER] Iniciando extracción de parámetros de: {file_name}")
        print(f"📋 [DEBUG HEADER] " + "="*60)
    
    with fits.open(file_name, memmap=True) as f:
        freq_axis_inverted = False
        
        # DEBUG: Estructura del archivo FITS
        if config.DEBUG_FREQUENCY_ORDER:
            print(f"📋 [DEBUG HEADER] Estructura del archivo FITS:")
            for i, hdu in enumerate(f):
                hdu_type = type(hdu).__name__
                if hasattr(hdu, 'header') and hdu.header:
                    if 'EXTNAME' in hdu.header:
                        ext_name = hdu.header['EXTNAME']
                    else:
                        ext_name = 'PRIMARY' if i == 0 else f'HDU_{i}'
                    print(f"📋 [DEBUG HEADER]   HDU {i}: {hdu_type} - {ext_name}")
                    if hasattr(hdu, 'columns') and hdu.columns:
                        print(f"📋 [DEBUG HEADER]     Columnas: {[col.name for col in hdu.columns]}")
                else:
                    print(f"📋 [DEBUG HEADER]   HDU {i}: {hdu_type} - Sin header")
        
        if "SUBINT" in [hdu.name for hdu in f] and "TBIN" in f["SUBINT"].header:
            # DEBUG: Procesando formato PSRFITS
            if config.DEBUG_FREQUENCY_ORDER:
                print(f"📋 [DEBUG HEADER] Formato detectado: PSRFITS (SUBINT)")
            
            hdr = f["SUBINT"].header
            sub_data = f["SUBINT"].data
            # Convertir a tipos numéricos explícitamente por si vengan como strings
            config.TIME_RESO = _safe_float(hdr.get("TBIN"))
            config.FREQ_RESO = _safe_int(hdr.get("NCHAN"))
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
                print(f"📋 [DEBUG HEADER] Headers PSRFITS extraídos:")
                print(
                    f"📋 [DEBUG HEADER]   TBIN (resolución temporal): {_safe_float(hdr.get('TBIN')):.2e} s"
                )
                print(f"📋 [DEBUG HEADER]   NCHAN (canales): {hdr['NCHAN']}")
                print(f"📋 [DEBUG HEADER]   NSBLK (muestras por subint): {hdr['NSBLK']}")
                print(f"📋 [DEBUG HEADER]   NAXIS2 (número de subints): {hdr['NAXIS2']}")
                print(f"📋 [DEBUG HEADER]   NPOL (polarizaciones): {hdr.get('NPOL', 'N/A')}")
                print(f"📋 [DEBUG HEADER]   Total de muestras: {config.FILE_LENG}")
                if 'OBS_MODE' in hdr:
                    print(f"📋 [DEBUG HEADER]   Modo de observación: {hdr['OBS_MODE']}")
                if 'SRC_NAME' in hdr:
                    print(f"📋 [DEBUG HEADER]   Fuente: {hdr['SRC_NAME']}")
            
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
                    print(f"📋 [DEBUG HEADER]   CHAN_BW detectado: {bw} MHz")
                if bw < 0:
                    freq_axis_inverted = True
                    if config.DEBUG_FREQUENCY_ORDER:
                        print(f"📋 [DEBUG HEADER]   ⚠️ CHAN_BW negativo - frecuencias invertidas!")
            elif len(freq_temp) > 1 and freq_temp[0] > freq_temp[-1]:
                freq_axis_inverted = True
                if config.DEBUG_FREQUENCY_ORDER:
                    print(f"📋 [DEBUG HEADER]   ⚠️ Frecuencias detectadas en orden descendente!")
        else:
            # DEBUG: Procesando formato FITS estándar
            if config.DEBUG_FREQUENCY_ORDER:
                print(f"📋 [DEBUG HEADER] Formato detectado: FITS estándar (no PSRFITS)")
            
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
                    print(f"📋 [DEBUG HEADER] HDU seleccionado para datos: {data_hdu_index}")
                
                hdr = f[data_hdu_index].header
                
                # DEBUG: Headers FITS estándar
                if config.DEBUG_FREQUENCY_ORDER:
                    print(f"📋 [DEBUG HEADER] Headers FITS estándar del HDU {data_hdu_index}:")
                    relevant_keys = ['TBIN', 'NCHAN', 'NAXIS2', 'NSBLK', 'NPOL', 'CRVAL1', 'CRVAL2', 'CRVAL3', 
                                   'CDELT1', 'CDELT2', 'CDELT3', 'CTYPE1', 'CTYPE2', 'CTYPE3']
                    for key in relevant_keys:
                        if key in hdr:
                            print(f"📋 [DEBUG HEADER]   {key}: {hdr[key]}")
                
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
                            print(f"📋 [DEBUG HEADER] Frecuencias extraídas de columna DAT_FREQ")
                else:
                    freq_axis_num = ''
                    for i in range(1, hdr.get('NAXIS', 0) + 1):
                        if 'FREQ' in hdr.get(f'CTYPE{i}', '').upper():
                            freq_axis_num = str(i)
                            break
                    
                    if config.DEBUG_FREQUENCY_ORDER:
                        print(f"📋 [DEBUG HEADER] Buscando eje de frecuencias en headers WCS...")
                        print(f"📋 [DEBUG HEADER] Eje de frecuencias detectado: CTYPE{freq_axis_num}" if freq_axis_num else "📋 [DEBUG HEADER] ⚠️ No se encontró eje de frecuencias")
                    
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
                            print(f"📋 [DEBUG HEADER] Parámetros WCS frecuencia:")
                            print(f"📋 [DEBUG HEADER]   CRVAL{freq_axis_num}: {crval} (valor de referencia)")
                            print(f"📋 [DEBUG HEADER]   CDELT{freq_axis_num}: {cdelt} (incremento por canal)")
                            print(f"📋 [DEBUG HEADER]   CRPIX{freq_axis_num}: {crpix} (pixel de referencia)")
                            print(f"📋 [DEBUG HEADER]   NAXIS{freq_axis_num}: {naxis} (número de canales)")
                        
                        if cdelt < 0:
                            freq_axis_inverted = True
                            if config.DEBUG_FREQUENCY_ORDER:
                                print(f"📋 [DEBUG HEADER]   ⚠️ CDELT negativo - frecuencias invertidas!")
                    else:
                        if config.DEBUG_FREQUENCY_ORDER:
                            print(f"📋 [DEBUG HEADER] ⚠️ Usando frecuencias por defecto: 1000-1500 MHz")
                        freq_temp = np.linspace(1000, 1500, hdr.get('NCHAN', 512))
                
                # Convertir a tipos numéricos para evitar errores de comparación
                config.TIME_RESO = _safe_float(hdr.get("TBIN"))
                config.FREQ_RESO = _safe_int(hdr.get("NCHAN", len(freq_temp)))
                config.FILE_LENG = _safe_int(hdr.get("NAXIS2", 0)) * _safe_int(hdr.get("NSBLK", 1))
                
                # DEBUG: Parámetros finales extraídos
                if config.DEBUG_FREQUENCY_ORDER:
                    print(f"📋 [DEBUG HEADER] Parámetros finales FITS estándar:")
                    print(f"📋 [DEBUG HEADER]   TIME_RESO: {config.TIME_RESO:.2e} s")
                    print(f"📋 [DEBUG HEADER]   FREQ_RESO: {config.FREQ_RESO}")
                    print(f"📋 [DEBUG HEADER]   FILE_LENG: {config.FILE_LENG}")
                    
            except Exception as e_std:
                if config.DEBUG_FREQUENCY_ORDER:
                    print(f"📋 [DEBUG HEADER] ⚠️ Error procesando FITS estándar: {e_std}")
                    print(f"📋 [DEBUG HEADER] Usando valores por defecto...")
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
        print(f"🔍 [DEBUG FRECUENCIAS] Archivo: {file_name}")
        print(f"🔍 [DEBUG FRECUENCIAS] freq_axis_inverted detectado: {freq_axis_inverted}")
        print(f"🔍 [DEBUG FRECUENCIAS] DATA_NEEDS_REVERSAL configurado: {config.DATA_NEEDS_REVERSAL}")
        print(f"🔍 [DEBUG FRECUENCIAS] Primeras 5 frecuencias: {config.FREQ[:5]}")
        print(f"🔍 [DEBUG FRECUENCIAS] Últimas 5 frecuencias: {config.FREQ[-5:]}")
        print(f"🔍 [DEBUG FRECUENCIAS] Frecuencia mínima: {config.FREQ.min():.2f} MHz")
        print(f"🔍 [DEBUG FRECUENCIAS] Frecuencia máxima: {config.FREQ.max():.2f} MHz")
        print(f"🔍 [DEBUG FRECUENCIAS] Orden esperado: frecuencias ASCENDENTES (menor a mayor)")
        if config.FREQ[0] < config.FREQ[-1]:
            print(f"✅ [DEBUG FRECUENCIAS] Orden CORRECTO: {config.FREQ[0]:.2f} < {config.FREQ[-1]:.2f}")
        else:
            print(f"❌ [DEBUG FRECUENCIAS] Orden INCORRECTO: {config.FREQ[0]:.2f} > {config.FREQ[-1]:.2f}")
        print(f"🔍 [DEBUG FRECUENCIAS] DOWN_FREQ_RATE: {config.DOWN_FREQ_RATE}")
        print(f"🔍 [DEBUG FRECUENCIAS] DOWN_TIME_RATE: {config.DOWN_TIME_RATE}")
        print("🔍 [DEBUG FRECUENCIAS] " + "="*50)

    # DEBUG: Información completa del archivo
    if config.DEBUG_FREQUENCY_ORDER:
        print(f"📁 [DEBUG ARCHIVO] Información completa del archivo: {file_name}")
        print(f"📁 [DEBUG ARCHIVO] " + "="*60)
        print(f"📁 [DEBUG ARCHIVO] DIMENSIONES Y RESOLUCIÓN:")
        print(f"📁 [DEBUG ARCHIVO]   - Resolución temporal: {config.TIME_RESO:.2e} segundos/muestra")
        print(f"📁 [DEBUG ARCHIVO]   - Resolución de frecuencia: {config.FREQ_RESO} canales")
        print(f"📁 [DEBUG ARCHIVO]   - Longitud del archivo: {config.FILE_LENG:,} muestras")
        
        # Calcular duración total
        duracion_total_seg = config.FILE_LENG * config.TIME_RESO
        duracion_min = duracion_total_seg / 60
        duracion_horas = duracion_min / 60
        print(f"📁 [DEBUG ARCHIVO]   - Duración total: {duracion_total_seg:.2f} seg ({duracion_min:.2f} min, {duracion_horas:.2f} h)")
        
        print(f"📁 [DEBUG ARCHIVO] FRECUENCIAS:")
        print(f"📁 [DEBUG ARCHIVO]   - Rango total: {config.FREQ.min():.2f} - {config.FREQ.max():.2f} MHz")
        print(f"📁 [DEBUG ARCHIVO]   - Ancho de banda: {abs(config.FREQ.max() - config.FREQ.min()):.2f} MHz")
        print(f"📁 [DEBUG ARCHIVO]   - Resolución por canal: {abs(config.FREQ[1] - config.FREQ[0]):.4f} MHz/canal")
        print(f"📁 [DEBUG ARCHIVO]   - Orden original: {'DESCENDENTE' if freq_axis_inverted else 'ASCENDENTE'}")
        print(f"📁 [DEBUG ARCHIVO]   - Orden final (post-corrección): {'ASCENDENTE' if config.FREQ[0] < config.FREQ[-1] else 'DESCENDENTE'}")
        
        print(f"📁 [DEBUG ARCHIVO] DECIMACIÓN:")
        print(f"📁 [DEBUG ARCHIVO]   - Factor reducción frecuencia: {config.DOWN_FREQ_RATE}x")
        print(f"📁 [DEBUG ARCHIVO]   - Factor reducción tiempo: {config.DOWN_TIME_RATE}x")
        print(f"📁 [DEBUG ARCHIVO]   - Canales después de decimación: {config.FREQ_RESO // config.DOWN_FREQ_RATE}")
        print(f"📁 [DEBUG ARCHIVO]   - Resolución temporal después: {config.TIME_RESO * config.DOWN_TIME_RATE:.2e} seg/muestra")
        
        # Calcular tamaño aproximado de datos
        size_original_gb = (config.FILE_LENG * config.FREQ_RESO * 4) / (1024**3)  # 4 bytes por float32
        size_decimated_gb = size_original_gb / (config.DOWN_FREQ_RATE * config.DOWN_TIME_RATE)
        print(f"📁 [DEBUG ARCHIVO] TAMAÑO ESTIMADO:")
        print(f"📁 [DEBUG ARCHIVO]   - Datos originales: ~{size_original_gb:.2f} GB")
        print(f"📁 [DEBUG ARCHIVO]   - Datos después decimación: ~{size_decimated_gb:.2f} GB")
        
        
        print(f"📁 [DEBUG ARCHIVO] CONFIGURACIÓN DE SLICE:")
        print(f"📁 [DEBUG ARCHIVO]   - SLICE_DURATION_MS configurado: {config.SLICE_DURATION_MS} ms")
        expected_slice_len = round(config.SLICE_DURATION_MS / (config.TIME_RESO * config.DOWN_TIME_RATE * 1000))
        print(f"📁 [DEBUG ARCHIVO]   - SLICE_LEN calculado: {expected_slice_len} muestras")
        print(f"📁 [DEBUG ARCHIVO]   - SLICE_LEN límites: [{config.SLICE_LEN_MIN}, {config.SLICE_LEN_MAX}]")
        
        print(f"📁 [DEBUG ARCHIVO] PROCESAMIENTO:")
        print(f"📁 [DEBUG ARCHIVO]   - Multi-banda habilitado: {'SÍ' if config.USE_MULTI_BAND else 'NO'}")
        print(f"📁 [DEBUG ARCHIVO]   - DM rango: {config.DM_min} - {config.DM_max} pc cm⁻³")
        print(f"📁 [DEBUG ARCHIVO]   - Umbrales: DET_PROB={config.DET_PROB}, CLASS_PROB={config.CLASS_PROB}, SNR_THRESH={config.SNR_THRESH}")
        print(f"📁 [DEBUG ARCHIVO] " + "="*60)

    if config.FREQ_RESO >= 512:
        config.DOWN_FREQ_RATE = max(1, int(round(config.FREQ_RESO / 512)))
    else:
        config.DOWN_FREQ_RATE = 1
    if config.TIME_RESO > 1e-9:
        config.DOWN_TIME_RATE = max(1, int((49.152 * 16 / 1e6) / config.TIME_RESO))
    else:
        config.DOWN_TIME_RATE = 15

    # DEBUG: Configuración final de decimación
    if config.DEBUG_FREQUENCY_ORDER:
        print(f"⚙️ [DEBUG CONFIG FINAL] Configuración final después de get_obparams:")
        print(f"⚙️ [DEBUG CONFIG FINAL] " + "="*60)
        print(f"⚙️ [DEBUG CONFIG FINAL] DOWN_FREQ_RATE calculado: {config.DOWN_FREQ_RATE}x")
        print(f"⚙️ [DEBUG CONFIG FINAL] DOWN_TIME_RATE calculado: {config.DOWN_TIME_RATE}x")
        print(f"⚙️ [DEBUG CONFIG FINAL] Datos después de decimación:")
        print(f"⚙️ [DEBUG CONFIG FINAL]   - Canales: {config.FREQ_RESO // config.DOWN_FREQ_RATE}")
        print(f"⚙️ [DEBUG CONFIG FINAL]   - Resolución temporal: {config.TIME_RESO * config.DOWN_TIME_RATE:.2e} s/muestra")
        print(f"⚙️ [DEBUG CONFIG FINAL]   - Reducción total de datos: {config.DOWN_FREQ_RATE * config.DOWN_TIME_RATE}x")
        print(f"⚙️ [DEBUG CONFIG FINAL] DATA_NEEDS_REVERSAL final: {config.DATA_NEEDS_REVERSAL}")
        print(f"⚙️ [DEBUG CONFIG FINAL] Orden de frecuencias final: {'ASCENDENTE' if config.FREQ[0] < config.FREQ[-1] else 'DESCENDENTE'}")
        print(f"⚙️ [DEBUG CONFIG FINAL] " + "="*60)

    # *** GUARDAR DEBUG INFO EN SUMMARY.JSON INMEDIATAMENTE ***
    if config.DEBUG_FREQUENCY_ORDER:
        _save_file_debug_info_fits(file_name, {
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


def _save_file_debug_info_fits(file_name: str, debug_info: dict) -> None:
    """Save debug information for a FITS file to summary.json immediately."""
    try:
        from pathlib import Path
        import os
        
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
