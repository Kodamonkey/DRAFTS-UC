import numpy as np
from astropy.io import fits
import warnings

def load_fits_file(file_name, reverse_flag=False):
    """
    Carga archivos FITS/PSRFITS de manera resiliente, manejando diferentes estructuras.
    
    Parameters:
    -----------
    file_name : str
        Ruta al archivo FITS
    reverse_flag : bool
        Si True, invierte el orden de frecuencias
        
    Returns:
    --------
    data : numpy.ndarray
        Datos con forma (time_samples, polarizations, frequency_channels)
    """
    
    # Intentar diferentes métodos de lectura
    data, header = None, None
    
    # Método 1: fitsio (más rápido)
    try:
        import fitsio
        data, header = fitsio.read(file_name, header=True)
    except ImportError:
        warnings.warn("fitsio no disponible, usando astropy")
    except Exception as e:
        warnings.warn(f"Error con fitsio: {e}, intentando con astropy")
    
    # Método 2: astropy.io.fits (fallback)
    if data is None or header is None:
        try:
            with fits.open(file_name) as f:
                # Buscar la extensión SUBINT (datos principales)
                subint_hdu = None
                for hdu in f:
                    if hasattr(hdu, 'header') and hdu.header.get('EXTNAME') == 'SUBINT':
                        subint_hdu = hdu
                        break
                
                if subint_hdu is None:
                    # Si no encuentra SUBINT, usar HDU[1] por defecto
                    subint_hdu = f[1]
                
                header = dict(subint_hdu.header)
                data = subint_hdu.data
        except Exception as e:
            raise Exception(f"Error al leer archivo FITS: {e}")
    
    # Extraer y reshapear los datos
    try:
        # Para archivos PSRFITS, los datos están en la extensión SUBINT
        # y necesitamos acceder a la columna 'DATA' correctamente
        raw_data = None
        
        if isinstance(data, np.ndarray) and hasattr(data, 'dtype') and data.dtype.names:
            # Datos estructurados (record array) - típico para PSRFITS
            if 'DATA' in data.dtype.names:
                raw_data = data['DATA']
                # Para PSRFITS, esto es típicamente un array 3D o 4D
                if raw_data.ndim == 1 and len(raw_data) == 1:
                    # Un solo registro, extraer el array interno
                    raw_data = raw_data[0]
            else:
                # Buscar otras posibles columnas de datos
                for col_name in ['data', 'Data', 'SUBDATA']:
                    if col_name in data.dtype.names:
                        raw_data = data[col_name]
                        if raw_data.ndim == 1 and len(raw_data) == 1:
                            raw_data = raw_data[0]
                        break
        
        if raw_data is None:
            raise ValueError("No se pudo encontrar columna de datos en el archivo FITS")
        
        # Obtener parámetros de forma desde el header
        naxis2 = header.get('NAXIS2', 1)  # Número de filas/subintegraciones
        nsblk = header.get('NSBLK', 1)    # Samples por bloque
        npol = header.get('NPOL', 1)      # Número de polarizaciones
        nchan = header.get('NCHAN', 512)  # Número de canales
        
        # Para archivos PSR mode, usar NBIN en lugar de NSBLK
        nbin = header.get('NBIN', 1024)   # Número de bins por período
        
        # Determinar el número total de muestras temporales
        if isinstance(nsblk, str) and nsblk == '*':
            # Modo PSR - usar NBIN
            total_time_samples = naxis2 * nbin
        else:
            # Modo SEARCH - usar NSBLK
            total_time_samples = naxis2 * nsblk
        
        print(f"Dimensiones detectadas: NAXIS2={naxis2}, NBIN={nbin}, NSBLK={nsblk}")
        print(f"NCHAN={nchan}, NPOL={npol}")
        print(f"Shape de raw_data: {raw_data.shape}")
        print(f"Total time samples calculado: {total_time_samples}")
        
        # Reshapear basado en las dimensiones actuales del array
        if raw_data.ndim == 1:
            # Array 1D - necesita reshape completo
            expected_size = total_time_samples * npol * nchan
            if len(raw_data) == expected_size:
                data_reshaped = raw_data.reshape(total_time_samples, npol, nchan)
            else:
                # Inferir dimensiones del tamaño actual
                total_size = len(raw_data)
                inferred_time_samples = total_size // (npol * nchan)
                print(f"Infiriendo {inferred_time_samples} muestras temporales del tamaño {total_size}")
                data_reshaped = raw_data.reshape(inferred_time_samples, npol, nchan)
                
        elif raw_data.ndim == 2:
            # Array 2D - podría ser (time, freq) o (subint, datos_por_subint)
            if raw_data.shape[1] == nchan:
                # Formato (time, freq) - añadir dimensión de polarización
                data_reshaped = raw_data[:, np.newaxis, :]
            else:
                # Cada fila es una subintegración con todos los datos
                data_per_subint = raw_data.shape[1]
                time_per_subint = data_per_subint // (npol * nchan)
                data_reshaped = raw_data.reshape(naxis2 * time_per_subint, npol, nchan)
                
        elif raw_data.ndim == 3:
            # Ya tiene 3 dimensiones - verificar orden
            if raw_data.shape == (naxis2, nbin, nchan) or raw_data.shape == (naxis2, nchan, nbin):
                # Typical PSRFITS format: (subint, time_or_phase, freq)
                if raw_data.shape[2] == nchan:
                    # (subint, time, freq)
                    data_reshaped = raw_data.reshape(-1, 1, nchan)  # Colapsar a (time*subint, 1_pol, freq)
                else:
                    # (subint, freq, time) - transponer
                    data_reshaped = raw_data.transpose(0, 2, 1).reshape(-1, 1, nchan)
            else:
                # Asumir formato (time, pol, freq) o similar
                data_reshaped = raw_data
                
        elif raw_data.ndim == 4:
            # 4D array típico: (subint, pol, time, freq) o (subint, time, pol, freq)
            if raw_data.shape[1] == npol:
                # (subint, pol, time, freq)
                data_reshaped = raw_data.transpose(0, 2, 1, 3).reshape(-1, npol, nchan)
            else:
                # (subint, time, pol, freq)
                data_reshaped = raw_data.reshape(-1, npol, nchan)
        
        else:
            raise ValueError(f"Dimensiones de datos no soportadas: {raw_data.shape}")
        
        print(f"Shape después del reshape: {data_reshaped.shape}")
        
        # Asegurar que tenemos al menos 2 polarizaciones para compatibilidad
        if data_reshaped.shape[1] >= 2:
            data_final = data_reshaped[:, :2, :]  # Tomar primeras 2 polarizaciones
        else:
            # Si solo hay 1 polarización, duplicarla
            if data_reshaped.shape[1] == 1:
                pol_data = data_reshaped[:, 0:1, :]
                data_final = np.concatenate([pol_data, pol_data], axis=1)
            else:
                data_final = data_reshaped
        
        # Aplicar reversión de frecuencias si se solicita
        if reverse_flag:
            data_final = data_final[:, :, ::-1]
        
        print(f"Shape final: {data_final.shape}")
        return data_final
        
    except Exception as e:
        raise Exception(f"Error al procesar datos FITS: {e}")


def get_obparams(file_name):
    """
    Extrae parámetros de observación de archivos FITS de manera resiliente.
    
    Parameters:
    -----------
    file_name : str
        Ruta al archivo FITS
        
    Returns:
    --------
    dict : Diccionario con parámetros de observación
    """
    
    params = {}
    
    try:
        with fits.open(file_name) as f:
            # Buscar extensión SUBINT o usar HDU[1]
            subint_hdu = None
            primary_hdu = f[0]  # Header primario
            
            for hdu in f:
                if hasattr(hdu, 'header') and hdu.header.get('EXTNAME') == 'SUBINT':
                    subint_hdu = hdu
                    break
            
            if subint_hdu is None:
                subint_hdu = f[1]
            
            header = subint_hdu.header
            primary_header = primary_hdu.header
            
            # Extraer resolución temporal
            # Diferentes posibles nombres de campo
            time_reso_fields = ['TBIN', 'TTYPE9', 'time_reso']
            time_reso = None
            
            for field in time_reso_fields:
                if field in header:
                    time_reso = header[field]
                    if isinstance(time_reso, str) and time_reso != '*':
                        try:
                            time_reso = float(time_reso)
                        except:
                            time_reso = None
                    elif time_reso == '*' or time_reso is None:
                        time_reso = None
                    break
            
            # Si no encontramos TBIN, intentar calcularlo
            if time_reso is None or time_reso == 0:
                # Valor por defecto basado en ejemplos típicos
                time_reso = 4.9152e-05  # ~49 microsegundos
                warnings.warn(f"No se pudo determinar resolución temporal, usando valor por defecto: {time_reso}")
            
            params['time_reso'] = float(time_reso)
            
            # Extraer número de canales de frecuencia
            freq_reso = header.get('NCHAN', 0)
            if freq_reso == 0:
                freq_reso = primary_header.get('OBSNCHAN', 1024)  # valor por defecto
            params['freq_reso'] = int(freq_reso)
            
            # Calcular longitud total del archivo
            naxis2 = header.get('NAXIS2', 1)
            nsblk = header.get('NSBLK', 1)
            file_leng = naxis2 * nsblk
            params['file_leng'] = int(file_leng)
            
            # Extraer frecuencias
            try:
                if 'DAT_FREQ' in subint_hdu.data.dtype.names:
                    freq_array = subint_hdu.data['DAT_FREQ'][0, :].astype(np.float64)
                else:
                    # Generar array de frecuencias basado en header
                    obsfreq = primary_header.get('OBSFREQ', 1400.0)  # MHz
                    obsbw = primary_header.get('OBSBW', 400.0)      # MHz
                    chan_bw = header.get('CHAN_BW', obsbw / freq_reso)
                    
                    # Generar frecuencias centradas
                    freq_start = obsfreq - obsbw/2.0 + chan_bw/2.0
                    freq_array = np.linspace(freq_start, 
                                           freq_start + (freq_reso-1)*abs(chan_bw), 
                                           freq_reso)
                    
                params['freq'] = freq_array
                
            except Exception as e:
                warnings.warn(f"Error al extraer frecuencias: {e}")
                # Generar frecuencias por defecto
                params['freq'] = np.linspace(1000, 1500, freq_reso)
            
            # Calcular factores de down-sampling
            target_freq_channels = 512
            params['down_freq_rate'] = max(1, int(freq_reso / target_freq_channels))
            
            # Factor de down-sampling temporal basado en resolución objetivo
            target_time_reso = 49.152 * 16 / 1e6  # ~0.786 ms
            params['down_time_rate'] = max(1, int(target_time_reso / time_reso))
            
            # Información adicional útil
            params['telescope'] = primary_header.get('TELESCOP', 'Unknown')
            params['source'] = primary_header.get('SRC_NAME', 'Unknown')
            params['obs_mode'] = primary_header.get('OBS_MODE', 'Unknown')
            params['backend'] = primary_header.get('BACKEND', 'Unknown')
            params['npol'] = header.get('NPOL', 1)
            
            return params
            
    except Exception as e:
        raise Exception(f"Error al leer parámetros del archivo FITS: {e}")


def set_global_params(file_name):
    """
    Función de compatibilidad que establece variables globales como en el código original.
    
    Parameters:
    -----------
    file_name : str
        Ruta al archivo FITS
    """
    global freq, freq_reso, time_reso, file_leng, down_freq_rate, down_time_rate
    
    params = get_obparams(file_name)
    
    freq = params['freq']
    freq_reso = params['freq_reso']
    time_reso = params['time_reso']
    file_leng = params['file_leng']
    down_freq_rate = params['down_freq_rate']
    down_time_rate = params['down_time_rate']
    
    print(f"Parámetros cargados:")
    print(f"  Telescopio: {params['telescope']}")
    print(f"  Fuente: {params['source']}")
    print(f"  Canales de frecuencia: {freq_reso}")
    print(f"  Resolución temporal: {time_reso:.2e} s")
    print(f"  Longitud del archivo: {file_leng} muestras")
    print(f"  Rango de frecuencias: {freq.min():.1f} - {freq.max():.1f} MHz")


# Función de utilidad para diagnosticar archivos FITS
def diagnose_fits_file(file_name):
    """
    Función de diagnóstico para inspeccionar la estructura de archivos FITS.
    
    Parameters:
    -----------
    file_name : str
        Ruta al archivo FITS
    """
    try:
        with fits.open(file_name) as f:
            print(f"\n=== Diagnóstico de {file_name} ===")
            print(f"Número de HDUs: {len(f)}")
            
            for i, hdu in enumerate(f):
                print(f"\nHDU {i}: {type(hdu).__name__}")
                if hasattr(hdu, 'header'):
                    extname = hdu.header.get('EXTNAME', 'N/A')
                    print(f"  EXTNAME: {extname}")
                    
                    if hasattr(hdu, 'data') and hdu.data is not None:
                        if isinstance(hdu.data, np.ndarray):
                            if len(hdu.data.dtype.names or []) > 0:
                                print(f"  Columnas: {list(hdu.data.dtype.names)}")
                                print(f"  Forma: {hdu.data.shape}")
                            else:
                                print(f"  Forma de datos: {hdu.data.shape}")
                        
                        # Mostrar campos clave si es SUBINT
                        if extname == 'SUBINT':
                            key_fields = ['NCHAN', 'NPOL', 'NAXIS2', 'NSBLK', 'TBIN']
                            for field in key_fields:
                                if field in hdu.header:
                                    print(f"  {field}: {hdu.header[field]}")
                                    
    except Exception as e:
        print(f"Error en diagnóstico: {e}")


# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de cómo usar las funciones
    try:
        file_name = "./Data/11P_1257sec_tot.fits"  # Reemplazar con archivo real
        
        # Diagnosticar archivo
        diagnose_fits_file(file_name)
        
        # Cargar datos
        data = load_fits_file(file_name, reverse_flag=False)
        print(f"Datos cargados con forma: {data.shape}")
        
        # Obtener parámetros
        params = get_obparams(file_name)
        print(f"Parámetros extraídos: {list(params.keys())}")
        
        # O usar método compatible con código original
        set_global_params(file_name)
        
    except Exception as e:
        print(f"Error: {e}")