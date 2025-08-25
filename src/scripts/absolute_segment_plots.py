"""Script for generating absolute segment plots from FRB data files."""
from __future__ import annotations

# Standard library imports
import argparse
from pathlib import Path

# Third-party imports
import numpy as np

# Local imports
from ..config import config
from ..input.data_loader import (
    get_obparams,
    get_obparams_fil,
    load_fil_file,
    load_fits_file
)
from ..preprocessing.dedispersion import dedisperse_block
from ..visualization.plot_waterfall_dispersed import save_waterfall_dispersed_plot
from ..visualization.plot_waterfall_dedispersed import save_waterfall_dedispersed_plot


def _infer_file_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".fits":
        return "fits"
    if suffix == ".fil":
        return "fil"
    raise ValueError(f"Tipo de archivo no soportado: {suffix}")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _compute_indices(start_sec: float, duration_sec: float) -> tuple[int, int, float]:
    """Convierte tiempo absoluto a índices en dominio decimado del pipeline.

    Retorna (start_idx_ds, end_idx_ds_exclusive, dt_ds).
    """
    dt_ds = config.TIME_RESO * config.DOWN_TIME_RATE
    start_idx = int(round(start_sec / dt_ds))
    nbin = int(round(duration_sec / dt_ds))
    end_idx = start_idx + max(nbin, 1)
    return start_idx, end_idx, dt_ds


def _compute_freq_down() -> np.ndarray:
    """Devuelve el eje de frecuencias decimado exactamente como el pipeline (promedio por grupos)."""
    freq = np.asarray(config.FREQ).reshape(-1)
    dfr = int(getattr(config, "DOWN_FREQ_RATE", 1))
    if dfr <= 1:
        return freq
    n = (freq.size // dfr) * dfr
    if n == 0:
        return freq
    freq_trim = freq[:n]
    freq_ds = freq_trim.reshape(n // dfr, dfr).mean(axis=1)
    return freq_ds


def run_single_segment(
    filename: Path,
    start: float,
    duration: float,
    dm: float,
    out_dir: Path,
    normalize: bool = True,
) -> None:
    # 1) Cargar headers → configura config.TIME_RESO, FREQ, FILE_LENG, DOWN_* si aplica
    ftype = _infer_file_type(filename)
    if ftype == "fits":
        get_obparams(str(filename))
    else:
        get_obparams_fil(str(filename))

    # 2) Cargar datos RAW (sin downsampling) y construir un bloque alineado a 'start'
    if ftype == "fits":
        data_raw = load_fits_file(str(filename))
    else:
        data_raw = load_fil_file(str(filename))
    # (time, pol, chan) → asegurar forma
    if data_raw.ndim != 3:
        raise ValueError("Datos inesperados: se espera (time, pol, chan)")
    # Seleccionar Stokes I ya aplicado por loader → tomar pol 0
    data_raw = data_raw[:, 0:1, :]
    # Mapear tiempo absoluto → índices RAW y extraer bloque exacto
    tsamp = float(config.TIME_RESO)
    R = int(config.DOWN_TIME_RATE)
    start_raw = int(round(start / tsamp))
    n_raw = int(round(duration / tsamp))
    # Asegurar longitud múltiplo de R para decimar después
    pad = (R - (n_raw % R)) % R
    end_raw = min(data_raw.shape[0], start_raw + max(1, n_raw + pad))
    start_raw = max(0, min(start_raw, data_raw.shape[0] - 1))
    raw_block = data_raw[start_raw:end_raw]  # (T_raw, 1, F)

    # 3) Downsampling al estilo PRESTO anclado en 'start' (decimar este bloque)
    from ..preprocessing.data_downsampler import downsample_data
    block_ds = downsample_data(raw_block)
    block = block_ds[:, :]  # (T_ds, F_ds)
    time_len, nchan = block.shape
    # Tiempo efectivo y dt tras decimar
    _, _, dt_ds = _compute_indices(0.0, 1.0)
    dt_ds = dt_ds  # claridad

    # 4) Directorios de salida y nombres
    file_stem = filename.stem
    seg_dir = out_dir / file_stem / f"start{start:.3f}_dur{duration:.3f}"
    _ensure_dir(seg_dir)

    # 5) Waterfall dispersado (segmento crudo decimado) con eje absoluto
    # NOTA: plot_waterfall_block ha sido reemplazado por save_waterfall_dispersed_plot
    # que genera plots idénticos al composite
    waterfall_dispersed_path = seg_dir / f"{file_stem}_disp_waterfall_dispersed.png"
    save_waterfall_dispersed_plot(
        waterfall_block=block,
        out_path=waterfall_dispersed_path,
        slice_idx=0,
        time_slice=1,
        band_name="Segment",
        band_suffix="segment",
        fits_stem=file_stem,
        slice_len=block.shape[0],
        normalize=normalize,
        band_idx=0,
        absolute_start_time=start,
        slice_samples=block.shape[0],
        dm_value=float(dm),  # Agregar DM para corrección de dispersión
    )

    # 6) Waterfall dedispersado en la misma ventana (estilo PRESTO)
    try:
        dedisp = dedisperse_block(
            data=block,
            freq_down=_compute_freq_down(),
            dm=float(dm),
            start=0,
            block_len=block.shape[0],
        )
        if dedisp is None or dedisp.size == 0:
            # fallback: usar el bloque crudo si no hay dedispersado
            dedisp = block
    except Exception:
        dedisp = block

    # NOTA: plot_waterfall_block ha sido reemplazado por save_waterfall_dedispersed_plot
    # que genera plots idénticos al composite
    waterfall_dedispersed_path = seg_dir / f"{file_stem}_dedisp_dm{int(round(dm))}_waterfall_dedispersed.png"
    save_waterfall_dedispersed_plot(
        dedispersed_block=dedisp,
        waterfall_block=block,  # Usar el bloque original como referencia
        top_conf=[],  # No hay detecciones en este script
        top_boxes=[],  # No hay detecciones en este script
        out_path=waterfall_dedispersed_path,
        slice_idx=0,
        time_slice=1,
        band_name="Segment",
        band_suffix="segment",
        fits_stem=file_stem,
        slice_len=dedisp.shape[0],
        normalize=normalize,
        band_idx=0,
        absolute_start_time=start,
        slice_samples=dedisp.shape[0],
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Genera plots de un único tramo con eje absoluto (estilo PRESTO) "
            "para waterfall dispersado y dedispersado."
        )
    )
    parser.add_argument("--filename", required=True, type=Path, help="Ruta al archivo .fits/.fil")
    parser.add_argument("--start", required=True, type=float, help="Tiempo absoluto de inicio (s)")
    parser.add_argument("--duration", required=True, type=float, help="Duración del tramo (s)")
    parser.add_argument("--dm", required=True, type=float, help="DM para dedispersado")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Results/AbsoluteSegments"),
        help="Directorio raíz para guardar resultados",
    )
    parser.add_argument("--no-normalize", action="store_true", help="Desactivar normalización visual")

    args = parser.parse_args()

    run_single_segment(
        filename=args.filename,
        start=args.start,
        duration=args.duration,
        dm=args.dm,
        out_dir=args.output_dir,
        normalize=not args.no_normalize,
    )


if __name__ == "__main__":
    main()


