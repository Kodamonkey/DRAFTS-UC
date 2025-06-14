import os 
import sys
import time
import csv
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap

if "mako" not in plt.colormaps():
    plt.register_cmap(
        name="mako",
        cmap=ListedColormap(sns.color_palette("mako", as_cmap=True)(np.linspace(0, 1, 256)))
    )

from astropy.io import fits
from numba import cuda, njit, prange
import torch

# ----------------------------------------------------------------------------
# Configuración general -------------------------------------------------------
# ----------------------------------------------------------------------------

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Selección de GPU
plt.style.use("default")

# ---------------------------------------------------------------------------
# Inserta el directorio raíz del proyecto en sys.path para importar módulos ----
# ---------------------------------------------------------------------------
_current_script_directory = Path(__file__).resolve().parent
_project_src_directory = _current_script_directory.parent
if str(_project_src_directory) not in sys.path:
    sys.path.insert(0, str(_project_src_directory))

# Importación local (modelo CenterNet y utilidades) --------------------------
from ObjectDet.centernet_utils import get_res  
from ObjectDet.centernet_model import centernet 

# Dispositivo (CPU / GPU) -----------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------
#  Variables globales de configuración --------------------------------------
# ----------------------------------------------------------------------------

# Variables globales de observación (se llenan con get_obparams) -------------
FREQ: np.ndarray  # vector de frecuencias promediadas
FREQ_RESO: int  # número de canales originales
TIME_RESO: float  # segundos por muestra (TBIN)
FILE_LENG: int  # número total de muestras temporales
DOWN_FREQ_RATE: int  # factor de submuestreo en frecuencia
DOWN_TIME_RATE: int  # factor de submuestreo temporal
DATA_NEEDS_REVERSAL: bool  # True si los datos originales del FITS necesitan ser invertidos para coincidir con FREQ ascendente

# Variables globales de configuración -----------------------------------

USE_MULTI_BAND = False  # Cambiar a True si se quiere las 3 bandas
SLICE_LEN = 512  # longitud de cada segmento temporal (número de muestras)
DET_PROB = 0.5 # probabilidad de detección mínima para considerar un candidato

# Rango de DM a explorar (en pc cm⁻³) -----------------------------------
DM_min = 0 # valor mínimo de DM
DM_max = 129 # valor máximo de DM


# ----------------------------------------------------------------------------
# Funciones auxiliares de I/O FITS -------------------------------------------
# ----------------------------------------------------------------------------

def load_fits_file(file_name: str) -> np.ndarray: # reverse_flag ya no es necesario como argumento
    """
    Carga un archivo FITS y devuelve la matriz (tiempo, pol, canal).
    El eje de frecuencia de los datos devueltos se invierte si DATA_NEEDS_REVERSAL es True,
    para que coincida con el orden ascendente de la variable global FREQ.
    """
    global DATA_NEEDS_REVERSAL # Accede a la bandera global
    data_array = None # Inicializar
    try:
        with fits.open(file_name, memmap=True) as hdul:
            if "SUBINT" in [hdu.name for hdu in hdul] and "DATA" in hdul["SUBINT"].columns.names:
                # ... (lógica de carga para Effelsberg como antes) ...
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

            else: # FAST / ALMA u otros casos genéricos
                import fitsio
                # ... (lógica de carga para fitsio como antes) ...
                temp_data, h = fitsio.read(file_name, header=True) # Renombrar para evitar conflicto
                # Asumiendo que 'DATA' es la columna correcta, esto puede variar
                if "DATA" in temp_data.dtype.names:
                    data_array = temp_data["DATA"].reshape(h["NAXIS2"] * h["NSBLK"], h["NPOL"], h["NCHAN"])[:, :2, :]
                else: # Fallback si no hay columna 'DATA', intenta usar el array directamente si es posible
                    # Esto es una suposición y puede fallar para formatos desconocidos
                    print("Advertencia: No se encontró la columna 'DATA' en FITS genérico con fitsio. Intentando usar el array directamente.")
                    # Necesitas una lógica más robusta aquí para determinar la forma correcta
                    # Por ahora, asumimos que temp_data es el array de datos crudos y necesita reshape
                    # Esta parte es muy especulativa y depende del formato FITS
                    total_samples = h.get("NAXIS2",1) * h.get("NSBLK",1)
                    num_pols = h.get("NPOL",2)
                    num_chans = h.get("NCHAN",512) # O FREQ_RESO
                    try:
                        data_array = temp_data.reshape(total_samples, num_pols, num_chans)[:,:2,:]
                    except ValueError as ve:
                        print(f"Error en reshape para FITS genérico con fitsio: {ve}. El array puede no ser compatible.")
                        raise # Re-lanza la excepción o maneja de otra forma


    except Exception as e:
        print(f"[Error cargando FITS con fitsio/astropy (rama principal), se intenta astropy genérico] {e}")
        try:
            with fits.open(file_name) as f:
                # Intenta encontrar la HDU de datos más probable
                data_hdu = None
                for hdu_item in f:
                    if hdu_item.data is not None and isinstance(hdu_item.data, np.ndarray) and hdu_item.data.ndim >= 3: # Busca un array 3D+
                        data_hdu = hdu_item
                        break
                if data_hdu is None and len(f) > 1: # Fallback a la HDU 1 si no se encontró nada mejor
                    data_hdu = f[1]
                elif data_hdu is None:
                    data_hdu = f[0] # Último recurso

                h = data_hdu.header
                raw_data = data_hdu.data # Renombrar
                # La lógica de reshape aquí es crítica y depende del formato FITS
                # Esto es una suposición general
                # (tiempo, pol, canal)
                # Necesitas obtener n_time, n_pol, n_chan de la cabecera h
                n_time_samples = h.get('NAXIS2', h.get('NAXIS1', raw_data.shape[0] if raw_data.ndim > 0 else 1)) # Muy especulativo
                n_pols = h.get('NPOL', 2) # O inferir del shape
                n_chans = h.get('NCHAN', raw_data.shape[-1] if raw_data.ndim > 0 else 1) # Muy especulativo

                # Intentar un reshape común, pero esto es propenso a errores sin conocer el formato
                # data_array = raw_data.reshape(n_time_samples, n_pols, n_chans)[:, :2, :]
                # Una forma más segura es basarse en las dimensiones conocidas si es posible
                # Por ahora, si es Effelsberg, ya se manejó. Si no, es más difícil.
                # La lógica original era:
                data_array = raw_data.reshape(h["NAXIS2"] * h["NSBLK"], h["NPOL"], h["NCHAN"])[:, :2, :]


        except Exception as e_astropy:
            print(f"Fallo final al cargar con astropy: {e_astropy}")
            raise # Re-lanza la excepción si todo falla

    if data_array is None:
        raise ValueError(f"No se pudieron cargar los datos de {file_name}")

    if DATA_NEEDS_REVERSAL:
        print(f">> Invirtiendo eje de frecuencia de los datos cargados para {file_name}")
        data_array = np.ascontiguousarray(data_array[:, :, ::-1])
    return data_array


def get_obparams(file_name: str) -> None:
    """
    Extrae parámetros clave del FITS y los expone como globales.
    Asegura que FREQ global esté siempre en orden ascendente.
    Establece DATA_NEEDS_REVERSAL si los datos originales del FITS
    necesitan ser invertidos para coincidir con FREQ ascendente.
    """
    global FREQ, FREQ_RESO, TIME_RESO, FILE_LENG, DOWN_FREQ_RATE, DOWN_TIME_RATE, DATA_NEEDS_REVERSAL
    with fits.open(file_name, memmap=True) as f:
        freq_axis_inverted_in_file = False # Asumimos orden ascendente por defecto
        
        # Detectar formato Effelsberg PSRFITS o FITS estándar ----------------
        if "SUBINT" in [hdu.name for hdu in f] and "TBIN" in f["SUBINT"].header:
            print(">> Detected Effelsberg PSRFITS format (HDU=2)")
            hdr = f["SUBINT"].header
            sub_data = f["SUBINT"].data # Renombrado para evitar confusión con la variable 'data' en main

            TIME_RESO = hdr["TBIN"] # Asumiendo que TBIN es la resolución temporal
            FREQ_RESO = hdr["NCHAN"] # Número de canales originales
            FILE_LENG = hdr["NSBLK"] * hdr["NAXIS2"] # Número total de muestras temporales
            # FREQ_temp es como está en el archivo
            FREQ_temp = sub_data["DAT_FREQ"][0].astype(np.float64) 

            # Para PSRFITS, CHAN_BW es una buena indicación de si las frecuencias están en orden ascendente o descendente
            if "CHAN_BW" in hdr:
                chan_bw_val = hdr["CHAN_BW"]
                if isinstance(chan_bw_val, (list, np.ndarray)): # A veces es un array
                    chan_bw_val = chan_bw_val[0]
                if chan_bw_val < 0:
                    freq_axis_inverted_in_file = True
                    print(">> CHAN_BW es negativo, frecuencias en archivo probablemente descendentes.")
            elif len(FREQ_temp) > 1 and FREQ_temp[0] > FREQ_temp[-1]: # Fallback si no hay CHAN_BW
                freq_axis_inverted_in_file = True
                print(">> Frecuencias en archivo parecen descendentes (FREQ[0] > FREQ[-1]).")


        else: # Caso general (FAST/GBT)
            print(">> Detected standard FITS (e.g., FAST/GBT)")
            # Intentar con HDU 1, pero podría ser otra para datos espectrales
            # Es crucial saber qué HDU contiene los datos y el eje de frecuencia
            try:
                # Asumimos que los datos relevantes están en la primera extensión de imagen/tabla útil
                # Esto puede necesitar ser más robusto para diferentes tipos de FITS
                data_hdu_index = 0
                for i, hdu_item in enumerate(f):
                    if hdu_item.is_image or isinstance(hdu_item, (fits.BinTableHDU, fits.TableHDU)):
                        if 'NAXIS' in hdu_item.header and hdu_item.header['NAXIS'] > 0: # Tiene datos
                             # Buscamos un eje que parezca de frecuencia
                            if 'CTYPE3' in hdu_item.header and 'FREQ' in hdu_item.header['CTYPE3'].upper():
                                data_hdu_index = i
                                break
                            if 'CTYPE2' in hdu_item.header and 'FREQ' in hdu_item.header['CTYPE2'].upper():
                                data_hdu_index = i
                                break
                            if 'CTYPE1' in hdu_item.header and 'FREQ' in hdu_item.header['CTYPE1'].upper():
                                data_hdu_index = i
                                break
                if data_hdu_index == 0 and len(f) > 1: # Si no se encontró CTYPE, probar con HDU 1 por defecto
                    data_hdu_index = 1

                hdr = f[data_hdu_index].header
                # Para datos no PSRFITS, necesitamos encontrar DAT_FREQ o construir FREQ desde CRVAL, CDELT, CRPIX
                if "DAT_FREQ" in f[data_hdu_index].columns.names: # Si es una tabla con DAT_FREQ
                     FREQ_temp = f[data_hdu_index].data["DAT_FREQ"][0].astype(np.float64)
                else: # Intentar construir desde WCS (esto es simplificado)
                    # Necesitas identificar cuál es el eje de frecuencia (ej. NAXIS3)
                    # Esto es un placeholder y necesita lógica robusta de WCS
                    freq_axis_num_str = ''
                    for i in range(1, hdr.get('NAXIS', 0) + 1):
                        if 'FREQ' in hdr.get(f'CTYPE{i}', '').upper():
                            freq_axis_num_str = str(i)
                            break
                    if freq_axis_num_str:
                        crval = hdr.get(f'CRVAL{freq_axis_num_str}', 0)
                        cdelt = hdr.get(f'CDELT{freq_axis_num_str}', 1)
                        crpix = hdr.get(f'CRPIX{freq_axis_num_str}', 1)
                        naxis = hdr.get(f'NAXIS{freq_axis_num_str}', FREQ_RESO if 'FREQ_RESO' in globals() and FREQ_RESO else hdr.get('NCHAN', 512)) # Fallback
                        FREQ_temp = crval + (np.arange(naxis) - (crpix - 1)) * cdelt
                        if cdelt < 0:
                            freq_axis_inverted_in_file = True # CDELT negativo implica descendente
                            print(f">> CDELT del eje de frecuencia es negativo.")

                    else: # Fallback si no se puede construir
                        print("Advertencia: No se pudo determinar el eje de frecuencia o construir FREQ desde WCS.")
                        # Asumir un array de placeholder si es necesario, o fallar
                        FREQ_temp = np.linspace(1000,1500, hdr.get('NCHAN', 512)) # Placeholder muy genérico

                TIME_RESO = hdr["TBIN"] # Asumiendo que TBIN existe
                FREQ_RESO = hdr.get("NCHAN", len(FREQ_temp)) # Número de canales originales
                FILE_LENG = hdr.get("NAXIS2", 0) * hdr.get("NSBLK", 1) # Esto es específico de PSRFITS, ajustar para otros
                if FILE_LENG == 0 and 'NAXIS1' in hdr and 'NAXIS2' in hdr : # Para espectros (tiempo, freq) o (freq, tiempo)
                    # Necesitas saber cuál eje es tiempo
                    pass # Ajustar FILE_LENG para formatos no PSRFITS

            except Exception as e_std:
                print(f"Error procesando FITS estándar: {e_std}")
                # Fallback a valores por defecto o error
                TIME_RESO = 5.12e-5
                FREQ_RESO = 512 
                FILE_LENG = 100000 # Placeholder
                FREQ_temp = np.linspace(1000,1500, FREQ_RESO) # Placeholder

        # Asegurar que FREQ global esté siempre en orden ascendente
        if freq_axis_inverted_in_file:
            FREQ = FREQ_temp[::-1]
            DATA_NEEDS_REVERSAL = True # Los datos del archivo necesitarán ser invertidos
            print(">> FREQ global ha sido invertido a orden ascendente.")
        else:
            FREQ = FREQ_temp
            DATA_NEEDS_REVERSAL = False # Los datos del archivo ya coinciden con FREQ ascendente
            print(">> FREQ global ya está (o se asume) en orden ascendente.")


    # Queremos terminar con 512 canales después de submuestreo
    if FREQ_RESO > 0 and FREQ_RESO >= 512 : # Evitar división por cero o ratio < 1
        DOWN_FREQ_RATE = int(round(FREQ_RESO / 512))
        DOWN_FREQ_RATE = max(1, DOWN_FREQ_RATE) # Asegurar que sea al menos 1
    else:
        DOWN_FREQ_RATE = 1 # No hacer downsampling si hay pocos canales o FREQ_RESO es inválido

    if TIME_RESO > 1e-9: # Evitar división por cero
        DOWN_TIME_RATE = int((49.152 * 16 / 1e6) / TIME_RESO)
        DOWN_TIME_RATE = max(1, DOWN_TIME_RATE) # Asegurar que sea al menos 1
    else:
        DOWN_TIME_RATE = 15 # Un valor por defecto si TIME_RESO es inválido
# ----------------------------------------------------------------------------
# Generación de waterfall (frecuencia vs tiempo) ------------------------
# ----------------------------------------------------------------------------

def plot_waterfall_block(
    data_block: np.ndarray,
    freq: np.ndarray,
    time_reso: float,
    block_size: int,
    block_idx: int,
    save_dir: Path,
    filename: str
):
    """
    Dibuja y guarda el waterfall de un bloque temporal.

    Args:
      data_block: array 2D (block_size, nchan) — dynspec preprocesado
      freq:        1D array de frecuencias (MHz)
      time_reso:   resolución temporal por muestra (s)
      block_size:  número de muestras en este bloque
      block_idx:   índice del bloque dentro de la observación
      save_dir:    Path donde guardar la imagen
      filename:    nombre base para el archivo (sin extensión)
    """
    # 1) calcula perfil y tiempo de pico
    profile    = data_block.mean(axis=1)
    time_start = block_idx * block_size * time_reso
    peak_time  = time_start + np.argmax(profile) * time_reso

    # 2) prepara figura con 4 filas (1 de perfil + 3 de waterfall)
    fig = plt.figure(figsize=(5, 5))
    gs  = gridspec.GridSpec(4, 1, height_ratios=[1, 4, 4, 4], hspace=0.05)

    # 2a) Perfil temporal encima
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(profile, color='royalblue', alpha=0.8, lw=1)
    ax0.set_xlim(0, block_size)
    ax0.set_xticks([])
    ax0.set_yticks([])

    # 2b) Waterfall abajo
    ax1 = fig.add_subplot(gs[1:, 0])
    im  = ax1.imshow(
        data_block.T,
        origin='lower',
        cmap='mako',
        aspect='auto',
        vmin=np.nanpercentile(data_block, 1),
        vmax=np.nanpercentile(data_block, 99)
    )
    # Etiquetas de frecuencia
    nchan = data_block.shape[1]
    ax1.set_yticks(np.linspace(0, nchan, 6))
    ax1.set_yticklabels(np.round(np.linspace(freq.min(), freq.max(), 6)).astype(int))
    # Etiquetas de tiempo
    ax1.set_xticks(np.linspace(0, block_size, 6))
    ax1.set_xticklabels(np.round(time_start + np.linspace(0, block_size, 6)*time_reso, 2))
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (MHz)")

    # 3) guarda y cierra
    out_path = save_dir / f"{filename}-block{block_idx:03d}-peak{peak_time:.2f}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ----------------------------------------------------------------------------
# Generación de mapa DM–tiempo (CPU y GPU) -----------------------------------
# ----------------------------------------------------------------------------


@cuda.jit
def _de_disp_gpu(dm_time, data, freq, index, start_offset, mid_channel):
    """Kernel de dedispersión (GPU)."""
    x, y = cuda.grid(2)
    if x < dm_time.shape[1] and y < dm_time.shape[2]:
        td_i = 0.0
        DM = x + start_offset  # DM actual (x es el índice de DM)
        for idx in index:
            delay = (
                4.15
                * DM
                * ((freq[idx]) ** -2 - (freq[-1] ** -2))
                * 1e3
                / TIME_RESO
                / DOWN_TIME_RATE
            )
            pos = int(delay + y)
            if 0 <= pos < data.shape[0]:
                td_i += data[pos, idx]
                if idx == mid_channel:  # canal medio dinámico
                    dm_time[1, x, y] = td_i  # slice central
        dm_time[2, x, y] = td_i - dm_time[1, x, y]
        dm_time[0, x, y] = td_i

@njit(parallel=True)
def _d_dm_time_cpu(data, height: int, width: int) -> np.ndarray:
    """Versión CPU (numba) del mapa DM–tiempo."""
    out = np.zeros((3, height, width), dtype=np.float32)
    nchan_ds = FREQ_RESO // DOWN_FREQ_RATE
    freq_index = np.arange(0, nchan_ds)
    mid_channel = nchan_ds // 2  # Canal medio dinámico
    
    for DM in prange(height):
        delays = (
            4.15
            * DM
            * (FREQ ** -2 - FREQ.max() ** -2)
            * 1e3
            / TIME_RESO
            / DOWN_TIME_RATE
        ).astype(np.int64)
        time_series = np.zeros(width, dtype=np.float32)
        for j in freq_index:
            time_series += data[delays[j] : delays[j] + width, j]
            if j == mid_channel:  # Consistente con GPU
                out[1, DM] = time_series
        out[0, DM] = time_series
        out[2, DM] = time_series - out[1, DM]
    return out

def d_dm_time_g(data: np.ndarray, height: int, width: int, chunk_size: int = 128) -> np.ndarray:
    """DDM en GPU con streaming por bloques para no saturar VRAM."""
    result = np.zeros((3, height, width), dtype=np.float32)
    try:
        # Constantes en GPU
        freq_values = np.mean(FREQ.reshape(FREQ_RESO // DOWN_FREQ_RATE, DOWN_FREQ_RATE), axis=1)
        freq_gpu = cuda.to_device(freq_values)
        
        # Número de canales tras downsampling
        nchan_ds = FREQ_RESO // DOWN_FREQ_RATE
        
        # excluir bordes RFI (10% inferior y superior)
        #lo = int(0.1 * nchan_ds)
        #hi = int(0.9 * nchan_ds)
        #index_values = np.arange(lo, hi)
        
        index_values = np.arange(0, nchan_ds)  # Usar todos los canales tras downsampling
        
        # Calcular el canal medio dinámicamente
        mid_channel = nchan_ds // 2
        print(f">> Canal medio para división de sub-bandas: {mid_channel} de {nchan_ds} canales")
        
        index_gpu = cuda.to_device(index_values)
        data_gpu = cuda.to_device(data)

        for start_dm in range(0, height, chunk_size):
            end_dm = min(start_dm + chunk_size, height)
            current_height = end_dm - start_dm
            dm_time_gpu = cuda.to_device(np.zeros((3, current_height, width), dtype=np.float32))

            nthreads = (8, 128)
            nblocks = (current_height // nthreads[0] + 1, width // nthreads[1] + 1)
            _de_disp_gpu[nblocks, nthreads](dm_time_gpu, data_gpu, freq_gpu, index_gpu, start_dm, mid_channel)
            cuda.synchronize()
            result[:, start_dm:end_dm, :] = dm_time_gpu.copy_to_host()
            del dm_time_gpu  # libera bloque

        return result
    except cuda.cudadrv.driver.CudaAPIError:
        # Fallback a CPU si hay problemas de memoria
        return _d_dm_time_cpu(data, height, width)

# ----------------------------------------------------------------------------
# Pre/post–procesado de imágenes ---------------------------------------------
# ----------------------------------------------------------------------------


def preprocess_img(img: np.ndarray) -> np.ndarray:
    img = (img - img.min()) / np.ptp(img)
    img = (img - img.mean()) / img.std()
    img = cv2.resize(img, (512, 512))
    img = np.clip(img, *np.percentile(img, (0.1, 99.9)))
    img = (img - img.min()) / np.ptp(img)
    img = plt.get_cmap("mako")(img)[..., :3]
    img -= [0.485, 0.456, 0.406]
    img /= [0.229, 0.224, 0.225]
    return img.transpose(2, 0, 1)  # (C,H,W)


def postprocess_img(img_tensor: np.ndarray) -> np.ndarray:
    img = img_tensor.transpose(1, 2, 0)
    img *= [0.229, 0.224, 0.225]
    img += [0.485, 0.456, 0.406]
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------------------
# Candidate dataclass ---------------------------------------------------------
# ----------------------------------------------------------------------------


class Candidate:
    """Contenedor ligero para un candidato detectado."""

    __slots__ = (
        "file",
        "slice_id",
        "band_id",
        "prob",
        "dm",
        "t_sec",
        "t_sample",
        "box",
        "snr",
    )

    def __init__(
        self,
        file: str,
        slice_id: int,
        band_id: int,
        prob: float,
        dm: float,
        t_sec: float,
        t_sample: int,
        box: Tuple[int, int, int, int],
        snr: float,
    ):
        self.file = file
        self.slice_id = slice_id
        self.band_id = band_id
        self.prob = prob
        self.dm = dm
        self.t_sec = t_sec
        self.t_sample = t_sample
        self.box = box
        self.snr = snr

    def to_row(self) -> List:
        return [
            self.file,
            self.slice_id,
            self.band_id,
            f"{self.prob:.3f}",
            f"{self.dm:.2f}",
            f"{self.t_sec:.6f}",
            self.t_sample,
            *self.box,
            f"{self.snr:.2f}",
        ]


# ----------------------------------------------------------------------------
# Funciones de utilería -------------------------------------------------------
# ----------------------------------------------------------------------------


def pixel_to_physical(px: float, py: float, time_SLICE_LEN: int) -> Tuple[float, float, int]:
    """Convierte coordenadas de pixel (x, y) a (dm, t_sec, t_sample)."""
    dm_range = DM_max - DM_min + 1
    scale_dm = dm_range / 512.0            # pc cm⁻³ por píxel vertical
    scale_time = time_SLICE_LEN / 512.0    # muestras por píxel horizontal
    dm_val      = DM_min + py * scale_dm 
    sample_off  = px * scale_time
    t_sample    = int(sample_off)
    t_seconds   = t_sample * TIME_RESO * DOWN_TIME_RATE
    return dm_val, t_seconds, t_sample


def compute_snr(slice_band: np.ndarray, box: Tuple[int, int, int, int]) -> float:
    """Calcula un S/N simple dentro de una caja bounding-box."""
    x1, y1, x2, y2 = map(int, box)
    box_data = slice_band[y1:y2, x1:x2]
    if box_data.size == 0:
        return 0.0
    signal = box_data.mean()
    noise = np.median(slice_band)
    std = slice_band.std(ddof=1)
    return (signal - noise) / (std + 1e-6)


# ----------------------------------------------------------------------------
# Bucle principal -------------------------------------------------------------
# ----------------------------------------------------------------------------


def main():
    global DET_PROB, USE_MULTI_BAND, SLICE_LEN
    det_prob = DET_PROB
    base_model = "resnet50"
    root_path = Path("./Data")
    save_path = Path(f"./Results/ObjectDetection/{base_model}")
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Configuración de bandas ---------------------------------------------
    band_configs = [
        (0, "fullband", "Full Band"),
        (1, "lowband", "Low Band"), 
        (2, "highband", "High Band")
    ] if USE_MULTI_BAND else [(0, "fullband", "Full Band")]

    # Cargar modelo CenterNet -------------------------------------------------
    model = centernet(model_name=base_model).to(DEVICE)
    model.load_state_dict(torch.load(f"cent_{base_model}.pth", map_location=DEVICE))
    model.eval()

    summary = {}

    for frb in ["B0355+54"]: # "FRB20121102", "FRB20201124", "FRB20180301
        file_list = sorted([f for f in root_path.glob("*.fits") if frb in f.name])
        if not file_list:
            continue

        # Extrae parámetros del primer archivo -------------------------------
        get_obparams(str(file_list[0])) # Esto llena las variables globales
        

        for fits_path in file_list:
            t_start = time.time()
            print(f"Procesando {fits_path.name}")

            # Carga y downsampling preliminar --------------------------------
            data = load_fits_file(str(fits_path))
            
            # Si solo hay una polarización, duplicarla
            if data.shape[1] == 1:
                data = np.repeat(data, 2, axis=1)
            
            # Simetria en polarización
            data = np.vstack([data, data[::-1, :]])  # simetría en pol
            
            # Debug de dimensiones
            print(">> data.shape:", data.shape)
            print(">> FILE_LENG =", FILE_LENG)
            print(">> FREQ_RESO =", FREQ_RESO)
            print(">> DOWN_TIME_RATE =", DOWN_TIME_RATE)
            print(">> DOWN_FREQ_RATE =", DOWN_FREQ_RATE)
            print(">> FREQ =", FREQ)
            print(">> FREQ.shape =", FREQ.shape)
            print(">> DM_min =", DM_min)
            print(">> DM_max =", DM_max)
            #print(">> DM_span =", DM_span)
            print(">> TIME_RESO =", TIME_RESO)
            
            # Verifica de dimensiones y ajustes -----------------------------
            # Cálculo de Duración Temporal
            # Duración real de cada slice
            slice_duration = SLICE_LEN * TIME_RESO * DOWN_TIME_RATE
            print(f"Duración de cada slice: {slice_duration:.3f} segundos")

                        
            # Recorte defensivo para reshape seguro
            n_time = (data.shape[0] // DOWN_TIME_RATE) * DOWN_TIME_RATE
            n_freq = (data.shape[2] // DOWN_FREQ_RATE) * DOWN_FREQ_RATE
            data = data[:n_time, :, :n_freq]

            # Reduce dimensiones con media en tiempo y frecuencia
            data = (
                np.mean(
                    data.reshape(
                        n_time // DOWN_TIME_RATE,
                        DOWN_TIME_RATE,
                        2,  # polarización ya en eje 1
                        n_freq // DOWN_FREQ_RATE,
                        DOWN_FREQ_RATE,
                    ),
                    axis=(1, 4)  # promediamos en tiempo y frecuencia
                )
                .mean(axis=1)  # promediamos también en polarización (eje 2 ahora)
                .astype(np.float32)
            )
            height  = DM_max - DM_min + 1
            width_total = FILE_LENG // DOWN_TIME_RATE
            print("height =", height) # número de DM
            print("width_total =", width_total) # número de muestras temporales
            dm_time = d_dm_time_g(data, height=height, width=width_total) # DM–tiempo
            print("dm_time.shape =", dm_time.shape) # (3, height, width_total)

            # Segmentación temporal antes -----------------------------------------
            # time_length = dm_time.shape[2] # longitud temporal total
            # time_slice = 1 # número de segmentos temporales
            # SLICE_LEN = time_length // time_slice # longitud de cada segmento
            
            if width_total == 0: # Evitar división por cero si no hay datos de tiempo
                print("Advertencia: width_total es 0, no se pueden crear slices.")
                time_slice = 0
            elif width_total < SLICE_LEN: # Manejar casos donde el ancho total es menor que el SLICE_LEN deseado
                SLICE_LEN = width_total
                time_slice = 1
            else:
                time_slice = width_total // SLICE_LEN # número de segmentos temporales
            
            # Asegurar al menos un slice si width_total es pequeño pero no cero y time_slice es 0
            if time_slice == 0 and width_total > 0:
                time_slice = 1
                SLICE_LEN = width_total # Asegúrate que SLICE_LEN no sea 0 si width_total > 0
            
            print(f"Análisis de {fits_path.name} con {time_slice} slices de {SLICE_LEN} muestras cada uno.")
            
            
            csv_file = save_path / f"{fits_path.stem}.candidates.csv"

            csv_file = save_path / f"{fits_path.stem}.candidates.csv"
            if not csv_file.exists():
                with csv_file.open("w", newline="") as f_csv:
                    writer = csv.writer(f_csv)
                    writer.writerow(
                        [
                            "file",
                            "slice",
                            "band",
                            "prob",
                            "dm_pc_cm-3",
                            "t_sec",
                            "t_sample",
                            "x1",
                            "y1",
                            "x2",
                            "y2",
                            "snr",
                        ]
                    )

            cand_counter = 0
            prob_max = 0.0
            snr_list: List[float] = []

            # Itera cada slice temporal --------------------------------------
            for j in range(time_slice):
                slice_cube = dm_time[:, :, SLICE_LEN * j : SLICE_LEN * (j + 1)]
                
                 
                # Configuración de bandas -----------------------------------
                
                for band_idx, band_suffix, band_name in band_configs: 
                    band_img = slice_cube[band_idx]  # Selecciona la banda correspondiente
                    img_tensor = preprocess_img(band_img) # Preprocesa la imagen

                    with torch.no_grad(): # Desactiva gradientes para inferencia
                        hm, wh, offset = model(torch.from_numpy(img_tensor).to(DEVICE).float().unsqueeze(0)) # Añade batch dimension
                    top_conf, top_boxes = get_res(hm, wh, offset, confidence=det_prob) # Obtiene detecciones

                    if top_boxes is None:
                        continue

                    img_rgb = postprocess_img(img_tensor)

                    # Procesa cada caja detectada ---------------------------
                    for conf, box in zip(top_conf, top_boxes):
                        x1, y1, x2, y2 = map(int, box)
                        dm_val, t_sec, t_sample = pixel_to_physical(
                            (x1 + x2) / 2, (y1 + y2) / 2, SLICE_LEN
                        )
                        snr_val = compute_snr(band_img, (x1, y1, x2, y2))
                        snr_list.append(snr_val)

                        cand = Candidate(
                            fits_path.name, # Nombre del archivo FITS
                            j, # ID del slice temporal
                            band_idx, # ID de la banda (0=full, 1=low, 2=high)
                            float(conf), # Confianza de la detección
                            dm_val, # DM calculado
                            t_sec, # Tiempo en segundos
                            t_sample, # Muestra temporal
                            (x1, y1, x2, y2), # Caja delimitadora (bounding box)
                            snr_val, # SNR calculado
                        )

                        # Guarda CSV --------------------------------------
                        with csv_file.open("a", newline="") as f_csv:
                            writer = csv.writer(f_csv)
                            writer.writerow(cand.to_row())

                        # Dibuja y guarda PNG -----------------------------
                        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 220, 0), 1)
                        cand_counter += 1
                        prob_max = max(prob_max, float(conf))

                    # Guarda la imagen anotada por slice/band -------------
                    out_img_path = save_path / f"{fits_path.stem}_slice{j}_{band_suffix}.png"  # Nombre de archivo con slice y banda
                    
                    # Crear figura con etiquetas científicas apropiadas
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Mostrar la imagen
                    im = ax.imshow(img_rgb, origin="lower", aspect='auto')
                    
                    # Configurar etiquetas de los ejes con valores físicos
                    # Eje X (Tiempo)
                    n_time_ticks = 6
                    time_positions = np.linspace(0, 512, n_time_ticks)
                    time_start_slice = j * SLICE_LEN * TIME_RESO * DOWN_TIME_RATE
                    time_values = time_start_slice + (time_positions / 512.0) * SLICE_LEN * TIME_RESO * DOWN_TIME_RATE
                    ax.set_xticks(time_positions)
                    ax.set_xticklabels([f"{t:.3f}" for t in time_values])
                    ax.set_xlabel("Time (s)", fontsize=12, fontweight='bold')
                    
                    # Eje Y (DM)
                    n_dm_ticks = 8
                    dm_positions = np.linspace(0, 512, n_dm_ticks)
                    dm_values = DM_min + (dm_positions / 512.0) * (DM_max - DM_min)
                    ax.set_yticks(dm_positions)
                    ax.set_yticklabels([f"{dm:.0f}" for dm in dm_values])
                    ax.set_ylabel("Dispersion Measure (pc cm⁻³)", fontsize=12, fontweight='bold')
                    
                    # Título científico detallado
                    band_names = ["Full Band", "Low Band", "High Band"]
                    freq_range = f"{FREQ.min():.1f}–{FREQ.max():.1f} MHz"
                    title = (f"{fits_path.stem} - {band_name} ({freq_range})\n"
                            f"Slice {j+1}/{time_slice} | "
                            f"Time Resolution: {TIME_RESO*DOWN_TIME_RATE*1e6:.1f} μs | "
                            f"DM Range: {DM_min}–{DM_max} pc cm⁻³")
                    ax.set_title(title, fontsize=11, fontweight='bold', pad=20)
                    
                    # Agregar información de detecciones como texto
                    if top_boxes is not None and len(top_boxes) > 0:
                        detection_info = f"Detections: {len(top_boxes)}"
                        ax.text(0.02, 0.98, detection_info, transform=ax.transAxes, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                               fontsize=10, verticalalignment='top', fontweight='bold')
                    
                    # Agregar información técnica en la esquina inferior derecha
                    tech_info = (f"Model: {base_model.upper()}\n"
                                f"Confidence: {det_prob:.1f}\n"
                                f"Channels: {FREQ_RESO}→{FREQ_RESO//DOWN_FREQ_RATE}\n"
                                f"Time samples: {FILE_LENG}→{FILE_LENG//DOWN_TIME_RATE}")
                    ax.text(0.98, 0.02, tech_info, transform=ax.transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                           fontsize=8, verticalalignment='bottom', horizontalalignment='right')
                    
                    # Mejorar las anotaciones de las cajas de detección
                    if top_boxes is not None:
                        for idx, (conf, box) in enumerate(zip(top_conf, top_boxes)):
                            x1, y1, x2, y2 = map(int, box)
                            # Dibujar rectángulo de detección
                            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                               linewidth=2, edgecolor='lime', facecolor='none')
                            ax.add_patch(rect)
                            
                            # Calcular valores físicos para la etiqueta
                            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                            dm_val, t_sec, t_sample = pixel_to_physical(center_x, center_y, SLICE_LEN)
                            
                            # Etiqueta con información de la detección
                            label = f"#{idx+1}\nDM: {dm_val:.1f}\nP: {conf:.2f}"
                            ax.annotate(label, xy=(center_x, center_y), 
                                       xytext=(center_x, y2 + 15),
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lime", alpha=0.8),
                                       fontsize=8, ha='center', fontweight='bold',
                                       arrowprops=dict(arrowstyle='->', color='lime', lw=1))
                    
                    # Agregar grid sutil para mejor lectura
                    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                    
                    # Mejorar el espaciado y diseño
                    plt.tight_layout()
                    
                    # Guardar con alta resolución
                    plt.savefig(out_img_path, dpi=300, bbox_inches="tight", 
                               facecolor='white', edgecolor='none')
                    plt.close()
                    
                    # Opcional: Guardar también una versión con colorbar para la banda completa
                    if band_suffix == "fullband":  # Solo para la banda completa    
                        fig_cb, ax_cb = plt.subplots(figsize=(13, 8))
                        
                        # Convertir de RGB a escala de grises para colorbar
                        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                        im_cb = ax_cb.imshow(img_gray, origin="lower", aspect='auto', cmap='mako')
                        
                        # Aplicar las mismas etiquetas
                        ax_cb.set_xticks(time_positions)
                        ax_cb.set_xticklabels([f"{t:.3f}" for t in time_values])
                        ax_cb.set_xlabel("Time (s)", fontsize=12, fontweight='bold')
                        
                        ax_cb.set_yticks(dm_positions)
                        ax_cb.set_yticklabels([f"{dm:.0f}" for dm in dm_values])
                        ax_cb.set_ylabel("Dispersion Measure (pc cm⁻³)", fontsize=12, fontweight='bold')
                        
                        ax_cb.set_title(title, fontsize=11, fontweight='bold', pad=20)
                        
                        # Agregar colorbar
                        cbar = plt.colorbar(im_cb, ax=ax_cb, shrink=0.8, pad=0.02)
                        cbar.set_label('Normalized Intensity', fontsize=10, fontweight='bold')
                        
                        # Agregar detecciones
                        if top_boxes is not None:
                            for idx, (conf, box) in enumerate(zip(top_conf, top_boxes)):
                                x1, y1, x2, y2 = map(int, box)
                                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                   linewidth=2, edgecolor='cyan', facecolor='none')
                                ax_cb.add_patch(rect)
                        
                        ax_cb.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                        plt.tight_layout()
                        
                        # Guardar versión con colorbar
                        cb_path = save_path / f"{fits_path.stem}_slice{j}_{band_suffix}_colorbar.png"
                        plt.savefig(cb_path, dpi=300, bbox_inches="tight", 
                                   facecolor='white', edgecolor='none')
                        plt.close()

            # Resumen por archivo -------------------------------------------
            runtime = time.time() - t_start
            summary[fits_path.name] = {
                "n_candidates": cand_counter,
                "runtime_s": runtime,
                "max_prob": prob_max,
                "mean_snr": float(np.mean(snr_list)) if snr_list else 0.0,
            }
            print(
                f"▶ {fits_path.name}: {cand_counter} candidatos, max prob {prob_max:.2f}, "
                f"⏱ {runtime:.1f} s"
            )

    # Guardar resumen global --------------------------------------------------
    summary_path = save_path / "summary.json"
    with summary_path.open("w") as f_json:
        json.dump(summary, f_json, indent=2)
    print(f"Resumen global escrito en {summary_path}")


if __name__ == "__main__":
    main()
