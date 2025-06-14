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
from ObjectDet.centernet_utils import get_res  # noqa: E402
from ObjectDet.centernet_model import centernet  # noqa: E402

# Dispositivo (CPU / GPU) -----------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------
# Funciones auxiliares de I/O FITS -------------------------------------------
# ----------------------------------------------------------------------------


def load_fits_file(file_name: str, reverse_flag: bool = False) -> np.ndarray:
    """Carga un archivo FITS y devuelve la matriz (tiempo, pol, canal).

    Solo se conservan las dos primeras polarizaciones.
    """
    try:
        import fitsio

        data, h = fitsio.read(file_name, header=True)
    except Exception:
        with fits.open(file_name) as f:
            h = f[1].header
            data = f[1].data
    # Reorganiza a (time, pol, chan)
    data = data["DATA"].reshape(h["NAXIS2"] * h["NSBLK"], h["NPOL"], h["NCHAN"])[:, :2, :]
    if reverse_flag:
        data = np.ascontiguousarray(data[:, :, ::-1])
    return data


# Variables globales de observación (se llenan con get_obparams) -------------
FREQ: np.ndarray  # vector de frecuencias promediadas
FREQ_RESO: int  # número de canales originales
TIME_RESO: float  # segundos por muestra (TBIN)
FILE_LENG: int  # número total de muestras temporales
DOWN_FREQ_RATE: int  # factor de submuestreo en frecuencia
DOWN_TIME_RATE: int  # factor de submuestreo temporal



def get_obparams(file_name: str) -> None:
    """Extrae parámetros clave del FITS y los expone como globales."""
    global FREQ, FREQ_RESO, TIME_RESO, FILE_LENG, DOWN_FREQ_RATE, DOWN_TIME_RATE

    with fits.open(file_name) as f:
        hdr = f[1].header
        TIME_RESO = hdr["TBIN"]
        FREQ_RESO = hdr["NCHAN"]
        FILE_LENG = hdr["NAXIS2"] * hdr["NSBLK"]
        FREQ = f[1].data["DAT_FREQ"][0, :].astype(np.float64)

    # Queremos terminar con 512 canales después de submuestreo
    DOWN_FREQ_RATE = int(FREQ_RESO / 512)
    # Ajuste para usar ~1 ms en band 3 (default del paper DRAFTS)
    DOWN_TIME_RATE = int((49.152 * 16 / 1e6) / TIME_RESO)


# ----------------------------------------------------------------------------
# Generación de mapa DM–tiempo (CPU y GPU) -----------------------------------
# ----------------------------------------------------------------------------


@cuda.jit
def _de_disp_gpu(dm_time, data, freq, index, start_offset):
    """Kernel de dedispersión (GPU)."""
    x, y = cuda.grid(2)
    if x < dm_time.shape[1] and y < dm_time.shape[2]:
        td_i = 0.0
        DM = x + start_offset  # DM actual (x es el índice de DM)
        for idx in index:
            delay = (
                4.15
                * DM
                * (freq[idx] ** -2 - freq[-1] ** -2)
                * 1e3
                / TIME_RESO
                / DOWN_TIME_RATE
            )
            pos = int(delay + y)
            if 0 <= pos < data.shape[0]:
                td_i += data[pos, idx]
                if idx == 256:
                    dm_time[1, x, y] = td_i  # slice central
        dm_time[2, x, y] = td_i - dm_time[1, x, y]
        dm_time[0, x, y] = td_i


@njit(parallel=True)
def _d_dm_time_cpu(data, height: int, width: int) -> np.ndarray:
    """Versión CPU (numba) del mapa DM–tiempo."""
    out = np.zeros((3, height, width), dtype=np.float32)
    freq_index = np.append(
        np.arange(int(10 / 4096 * FREQ_RESO // DOWN_FREQ_RATE), int(650 / 4096 * FREQ_RESO // DOWN_FREQ_RATE)),
        np.arange(int(820 / 4096 * FREQ_RESO // DOWN_FREQ_RATE), int(4050 / 4096 * FREQ_RESO // DOWN_FREQ_RATE)),
    )
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
            if j == int(FREQ_RESO // 2):
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
        index_values = np.append(
            np.arange(int(10 / 4096 * FREQ_RESO // DOWN_FREQ_RATE), int(650 / 4096 * FREQ_RESO // DOWN_FREQ_RATE)),
            np.arange(int(820 / 4096 * FREQ_RESO // DOWN_FREQ_RATE), int(4050 / 4096 * FREQ_RESO // DOWN_FREQ_RATE)),
        )
        index_gpu = cuda.to_device(index_values)
        data_gpu = cuda.to_device(data)

        for start_dm in range(0, height, chunk_size):
            end_dm = min(start_dm + chunk_size, height)
            current_height = end_dm - start_dm
            dm_time_gpu = cuda.to_device(np.zeros((3, current_height, width), dtype=np.float32))

            nthreads = (8, 128)
            nblocks = (current_height // nthreads[0] + 1, width // nthreads[1] + 1)
            _de_disp_gpu[nblocks, nthreads](dm_time_gpu, data_gpu, freq_gpu, index_gpu, start_dm)
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


def pixel_to_physical(px: float, py: float, time_slice_len: int) -> Tuple[float, float, int]:
    """Convierte coordenadas de pixel (x, y) a (dm, t_sec, t_sample)."""
    scale_dm = 2  # cada pixel vertical son 2 unidades DM (512 px -> 1024 DM)
    scale_time = time_slice_len / 512  # muestras por pixel horizontal
    dm_val = py * scale_dm
    sample_offset = px * scale_time
    global_sample = int(sample_offset)
    t_seconds = global_sample * TIME_RESO * DOWN_TIME_RATE
    return dm_val, t_seconds, global_sample


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
    det_prob = 0.5
    base_model = "resnet50"
    root_path = Path("./Data")
    save_path = Path(f"./Results/ObjectDetection/{base_model}")
    save_path.mkdir(parents=True, exist_ok=True)

    # Cargar modelo CenterNet -------------------------------------------------
    model = centernet(model_name=base_model).to(DEVICE)
    model.load_state_dict(torch.load(f"cent_{base_model}.pth", map_location=DEVICE))
    model.eval()

    summary = {}

    for frb in ["FRB20121102", "FRB20201124", "FRB20180301"]:
        file_list = sorted([f for f in root_path.glob("*.fits") if frb in f.name])
        if not file_list:
            continue

        # Extrae parámetros del primer archivo -------------------------------
        get_obparams(str(file_list[0]))

        for fits_path in file_list:
            t_start = time.time()
            print(f"Procesando {fits_path.name}")

            # Carga y downsampling preliminar --------------------------------
            data = load_fits_file(str(fits_path))
            data = np.vstack([data, data[::-1, :]])  # simetría en pol
            data = (
                np.mean(
                    data.reshape(
                        data.shape[0] // DOWN_TIME_RATE,
                        DOWN_TIME_RATE,
                        FREQ_RESO // DOWN_FREQ_RATE,
                        DOWN_FREQ_RATE,
                    ),
                    axis=(1, 3),
                )
                .astype(np.float32)
            )

            dm_time = d_dm_time_g(data, height=1024, width=FILE_LENG // DOWN_TIME_RATE)

            # Segmentación temporal -----------------------------------------
            time_length = dm_time.shape[2]
            time_slice = 4
            slice_len = time_length // time_slice

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
                slice_cube = dm_time[:, :, slice_len * j : slice_len * (j + 1)]

                for k in range(3):  # 3 bandas (low/high/full)
                    band_img = np.mean(slice_cube[k].reshape(512, 2, slice_len), axis=1)
                    img_tensor = preprocess_img(band_img)

                    with torch.no_grad():
                        hm, wh, offset = model(torch.from_numpy(img_tensor).to(DEVICE).float().unsqueeze(0))
                    top_conf, top_boxes = get_res(hm, wh, offset, confidence=det_prob)

                    if top_boxes is None:
                        continue

                    img_rgb = postprocess_img(img_tensor)

                    # Procesa cada caja detectada ---------------------------
                    for conf, box in zip(top_conf, top_boxes):
                        x1, y1, x2, y2 = map(int, box)
                        dm_val, t_sec, t_sample = pixel_to_physical(
                            (x1 + x2) / 2, (y1 + y2) / 2, slice_len
                        )
                        snr_val = compute_snr(band_img, (x1, y1, x2, y2))
                        snr_list.append(snr_val)

                        cand = Candidate(
                            fits_path.name,
                            j,
                            k,
                            float(conf),
                            dm_val,
                            t_sec,
                            t_sample,
                            (x1, y1, x2, y2),
                            snr_val,
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
                    out_img_path = save_path / f"{fits_path.stem}_slice{j}_band{k}.png"
                    plt.figure()
                    plt.imshow(img_rgb, origin="lower")
                    plt.axis("off")
                    plt.savefig(out_img_path, dpi=300, bbox_inches="tight")
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
