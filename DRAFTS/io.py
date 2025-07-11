"""Input/output helpers for PSRFITS and standard FITS files."""
from __future__ import annotations

from typing import List

import numpy as np
from astropy.io import fits

from . import config


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
                nsubint = hdr["NAXIS2"]
                nchan = hdr["NCHAN"]
                npol = hdr["NPOL"]
                nsblk = hdr["NSBLK"]
                data_array = data_array.reshape(nsubint, nchan, npol, nsblk).swapaxes(1, 2)
                data_array = data_array.reshape(nsubint * nsblk, npol, nchan)
                data_array = data_array[:, :2, :]
            else:
                import fitsio
                temp_data, h = fitsio.read(file_name, header=True)
                if "DATA" in temp_data.dtype.names:
                    data_array = temp_data["DATA"].reshape(h["NAXIS2"] * h["NSBLK"], h["NPOL"], h["NCHAN"])[:, :2, :]
                else:
                    total_samples = h.get("NAXIS2", 1) * h.get("NSBLK", 1)
                    num_pols = h.get("NPOL", 2)
                    num_chans = h.get("NCHAN", 512)
                    data_array = temp_data.reshape(total_samples, num_pols, num_chans)[:, :2, :]
    except Exception as e:
        print(f"[Error cargando FITS con fitsio/astropy] {e}")
        try:
            # Intentar sin memmap para archivos corruptos
            with fits.open(file_name, memmap=False) as f:
                data_hdu = None
                for hdu_item in f:
                    # Evitar acceder a .data directamente, usar hasattr primero
                    try:
                        if (hdu_item.data is not None and 
                            isinstance(hdu_item.data, np.ndarray) and 
                            hdu_item.data.ndim >= 3):
                            data_hdu = hdu_item
                            break
                    except (TypeError, ValueError):
                        # Si no se puede acceder a los datos, saltar este HDU
                        continue
                        
                if data_hdu is None and len(f) > 1:
                    data_hdu = f[1]
                elif data_hdu is None:
                    data_hdu = f[0]
                    
                h = data_hdu.header
                try:
                    raw_data = data_hdu.data
                    if raw_data is not None:
                        data_array = raw_data.reshape(h["NAXIS2"] * h.get("NSBLK", 1), h.get("NPOL", 2), h.get("NCHAN", raw_data.shape[-1]))[:, :2, :]
                    else:
                        raise ValueError("No hay datos válidos en el HDU")
                except (TypeError, ValueError) as e_data:
                    print(f"Error accediendo a datos del HDU: {e_data}")
                    raise ValueError(f"Archivo FITS corrupto: {file_name}")
                    
        except Exception as e_astropy:
            print(f"Fallo final al cargar con astropy: {e_astropy}")
            raise ValueError(f"No se puede leer el archivo FITS corrupto: {file_name}")
            
    if data_array is None:
        raise ValueError(f"No se pudieron cargar los datos de {file_name}")

    if global_vars.DATA_NEEDS_REVERSAL:
        print(f">> Invirtiendo eje de frecuencia de los datos cargados para {file_name}")
        data_array = np.ascontiguousarray(data_array[:, :, ::-1])
    return data_array


def get_obparams(file_name: str) -> None:
    """Extract observation parameters and populate :mod:`config`."""
    with fits.open(file_name, memmap=True) as f:
        freq_axis_inverted = False
        if "SUBINT" in [hdu.name for hdu in f] and "TBIN" in f["SUBINT"].header:
            hdr = f["SUBINT"].header
            sub_data = f["SUBINT"].data
            config.TIME_RESO = hdr["TBIN"]
            config.FREQ_RESO = hdr["NCHAN"]
            config.FILE_LENG = hdr["NSBLK"] * hdr["NAXIS2"]
            freq_temp = sub_data["DAT_FREQ"][0].astype(np.float64)
            if "CHAN_BW" in hdr:
                bw = hdr["CHAN_BW"]
                if isinstance(bw, (list, np.ndarray)):
                    bw = bw[0]
                if bw < 0:
                    freq_axis_inverted = True
            elif len(freq_temp) > 1 and freq_temp[0] > freq_temp[-1]:
                freq_axis_inverted = True
        else:
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
                hdr = f[data_hdu_index].header
                if "DAT_FREQ" in f[data_hdu_index].columns.names:
                    freq_temp = f[data_hdu_index].data["DAT_FREQ"][0].astype(np.float64)
                else:
                    freq_axis_num = ''
                    for i in range(1, hdr.get('NAXIS', 0) + 1):
                        if 'FREQ' in hdr.get(f'CTYPE{i}', '').upper():
                            freq_axis_num = str(i)
                            break
                    if freq_axis_num:
                        crval = hdr.get(f'CRVAL{freq_axis_num}', 0)
                        cdelt = hdr.get(f'CDELT{freq_axis_num}', 1)
                        crpix = hdr.get(f'CRPIX{freq_axis_num}', 1)
                        naxis = hdr.get(f'NAXIS{freq_axis_num}', hdr.get('NCHAN', 512))
                        freq_temp = crval + (np.arange(naxis) - (crpix - 1)) * cdelt
                        if cdelt < 0:
                            freq_axis_inverted = True
                    else:
                        freq_temp = np.linspace(1000, 1500, hdr.get('NCHAN', 512))
                config.TIME_RESO = hdr["TBIN"]
                config.FREQ_RESO = hdr.get("NCHAN", len(freq_temp))
                config.FILE_LENG = hdr.get("NAXIS2", 0) * hdr.get("NSBLK", 1)
            except Exception as e_std:
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

    # ✅ DEBUG: Mostrar información detallada para archivos FITS
    print(f"\n[DEBUG] 📡 INFORMACIÓN DEL ARRAY DE FRECUENCIAS (FITS):")
    print(f"  - Array shape: {config.FREQ.shape}")
    print(f"  - Frecuencia mínima: {config.FREQ.min():.3f} MHz")
    print(f"  - Frecuencia máxima: {config.FREQ.max():.3f} MHz")
    print(f"  - Ancho de banda total: {config.FREQ.max() - config.FREQ.min():.3f} MHz")
    print(f"  - Resolución por canal: {abs(config.FREQ[1] - config.FREQ[0]):.6f} MHz")
    print(f"  - Datos invertidos (freq descendente): {config.DATA_NEEDS_REVERSAL}")
    print(f"  - Primeros 10 canales: {config.FREQ[:10]}")
    print(f"  - Últimos 10 canales: {config.FREQ[-10:]}")
    
    print(f"\n[DEBUG] 📄 PARÁMETROS EXTRAÍDOS DEL FITS:")
    print(f"  - TIME_RESO: {config.TIME_RESO:.6e} s")
    print(f"  - FREQ_RESO: {config.FREQ_RESO}")
    print(f"  - FILE_LENG: {config.FILE_LENG}")
    print(f"  - freq_axis_inverted: {freq_axis_inverted}")
    
    # ✅ Calcular y mostrar el tamaño final después del downsampling
    final_samples = config.FILE_LENG // config.DOWN_TIME_RATE
    final_channels = config.FREQ_RESO // config.DOWN_FREQ_RATE
    reduction_factor = config.DOWN_TIME_RATE * config.DOWN_FREQ_RATE
    print(f"\n[INFO] 💾 IMPACTO DEL DOWNSAMPLING (FITS):")
    print(f"  - Tamaño después del downsampling: {final_samples:,} muestras")
    print(f"  - Canales después del downsampling: {final_channels}")
    print(f"  - Factor de reducción total: {reduction_factor}x")
    print(f"  - ¿Necesita chunks?: {'NO' if final_samples <= getattr(config, 'MAX_SAMPLES_LIMIT', 2000000) else 'SÍ'}")
