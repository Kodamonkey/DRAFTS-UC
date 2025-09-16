from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation, solar_system_ephemeris
import astropy.units as u
import numpy as np
import argparse
from pathlib import Path
import pandas as pd

try:
    from blimpy import Waterfall  # type: ignore
except Exception:
    Waterfall = None

try:
    from your import Your  # type: ignore
except Exception:
    Your = None

K_DM = 4.148808e3  # s MHz^2 pc^-1 cm^3

def get_topo_MJD(mjd_files, times_s):
    return mjd_files + times_s / 86400.0

def get_bary_MJD(topo_mjds, RA, DEC, DM, freq_MHz, location="Effelsberg", ephem="de432s"):
    # 1) Sitio
    loc = EarthLocation.of_site(location) if isinstance(location, str) else location

    # 2) Tiempo con localización; trabajar en TDB explícitamente
    times_utc = Time(topo_mjds, format="mjd", scale="utc", location=loc)
    times_tdb = times_utc.tdb

    # 3) Efemérides JPL precisas
    solar_system_ephemeris.set(ephem)

    # 4) Fuente y corrección baricéntrica (TDB seconds)
    src = SkyCoord(RA, DEC, unit=(u.hourangle, u.deg), frame="icrs")
    ltt_bary = times_tdb.light_travel_time(src)  # TimeDelta in TDB seconds

    # 5) Corrección por dispersión a ν→∞
    dmcorr = TimeDelta(K_DM * DM / (freq_MHz**2), format="sec")

    # 6) Resultado en MJD(TDB)
    bary_mjds = (times_tdb + ltt_bary - dmcorr).mjd
    return times_utc.mjd, bary_mjds

def read_file_mjd_start(file_path: str) -> float:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo de datos: {file_path}")
    if p.suffix.lower() == ".fil":
        if Waterfall is None:
            raise ImportError("Falta dependencia 'blimpy' para leer .fil")
        wf = Waterfall(str(p), load_data=False)
        return float(wf.header["tstart"])  # MJD(UTC)
    elif p.suffix.lower() == ".fits":
        if Your is None:
            raise ImportError("Falta dependencia 'your' para leer .fits")
        f = Your(str(p))
        imjd = float(f.header["STT_IMJD"])  # días MJD
        smjd = float(f.header["STT_SMJD"])  # segundos del día
        offs = float(f.header.get("STT_OFFS", 0.0))
        return imjd + (smjd + offs) / 86400.0
    else:
        raise ValueError("Extensión no soportada, use .fil o .fits")

def get_file_metadata(file_path: str) -> dict:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo de datos: {file_path}")
    meta: dict = {"path": str(p)}
    if p.suffix.lower() == ".fil":
        if Waterfall is None:
            raise ImportError("Falta dependencia 'blimpy' para leer .fil")
        wf = Waterfall(str(p), load_data=False)
        h = wf.header
        meta["tstart_mjd"] = float(h["tstart"])  # MJD UTC
        meta["tsamp_s"] = float(h.get("tsamp", 0.0))  # s
        meta["nchans"] = int(h.get("nchans", 0))
        # Filtro SIGPROC: fch1 (MHz) (primer canal), foff (MHz) paso por canal
        fch1 = h.get("fch1")
        foff = h.get("foff")
        if fch1 is not None and foff is not None and meta["nchans"]:
            fch1 = float(fch1)
            foff = float(foff)
            freqs = fch1 + np.arange(meta["nchans"]) * foff
            meta["chan_freqs_MHz"] = np.array(freqs, dtype=float)
        # Intentar RA/DEC si están en header (formato SIGPROC: hhmmss.s / ddmmss.s)
        src_raj = h.get("src_raj")
        src_dej = h.get("src_dej")
        def _fmt_sigproc_angle(val, is_ra=True):
            try:
                v = float(val)
            except Exception:
                return None
            v = abs(v)
            hh = int(v // 10000)
            mm = int((v - hh*10000)//100)
            ss = v - hh*10000 - mm*100
            sign = '-' if (not is_ra and float(val) < 0) else ''
            return f"{sign}{hh:02d}:{mm:02d}:{ss:06.3f}"
        if src_raj is not None and src_dej is not None:
            meta["ra_hms"] = _fmt_sigproc_angle(src_raj, True)
            meta["dec_dms"] = _fmt_sigproc_angle(src_dej, False)
        return meta
    elif p.suffix.lower() == ".fits":
        if Your is None:
            raise ImportError("Falta dependencia 'your' para leer .fits")
        f = Your(str(p))
        imjd = float(f.header["STT_IMJD"])  # días MJD
        smjd = float(f.header["STT_SMJD"])  # s dentro del día
        offs = float(f.header.get("STT_OFFS", 0.0))
        meta["tstart_mjd"] = imjd + (smjd + offs) / 86400.0
        meta["tsamp_s"] = float(f.native_tsamp())
        meta["nchans"] = int(f.nchans)
        fch1 = float(f.fch1)
        foff = float(f.foff)
        freqs = fch1 + np.arange(meta["nchans"]) * foff
        meta["chan_freqs_MHz"] = np.array(freqs, dtype=float)
        # RA/DEC si disponibles
        try:
            meta["ra_hms"] = f.ra_str
            meta["dec_dms"] = f.dec_str
        except Exception:
            pass
        return meta
    else:
        raise ValueError("Extensión no soportada, use .fil o .fits")

def _compute_barycentric_columns(
    df: pd.DataFrame,
    ra: str,
    dec: str,
    freq_mhz: float,
    location: str,
    ephem: str,
    default_dm: float | None,
    ra_series: pd.Series | None = None,
    dec_series: pd.Series | None = None,
) -> pd.DataFrame:
    # Preparar ubicación, fuente y tiempos
    loc = EarthLocation.of_site(location) if isinstance(location, str) else location
    src = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame="icrs")
    solar_system_ephemeris.set(ephem)

    # RA/DEC por fila si se entregan, si no usar escalares
    if ra_series is not None and dec_series is not None:
        ra_vals = ra_series.to_numpy()
        dec_vals = dec_series.to_numpy()
        src = SkyCoord(ra_vals, dec_vals, unit=(u.hourangle, u.deg), frame="icrs")
    # Tiempos topo en UTC y TDB
    times_utc = Time(df["mjd_utc"].to_numpy(), format="mjd", scale="utc", location=loc)
    times_tdb = times_utc.tdb
    # Corrección baricéntrica (segundos TDB)
    ltt_bary = times_tdb.light_travel_time(src)
    bary_tdb = (times_tdb + ltt_bary)
    df["mjd_bary_tdb"] = bary_tdb.mjd
    # También en escala UTC
    bary_utc = bary_tdb.utc
    df["mjd_bary_utc"] = bary_utc.mjd

    # Corrección de dispersión a frecuencia infinita
    if "dm_pc_cm-3" in df.columns:
        dm_vals = df["dm_pc_cm-3"].astype(float).to_numpy()
    elif default_dm is not None:
        dm_vals = np.full(len(df), float(default_dm))
    else:
        raise ValueError("No hay columna 'dm_pc_cm-3' ni --default_dm especificado para corrección de dispersión")

    # Preferir frecuencia efectiva del header si está disponible (columna nu_eff_inv2)
    if "nu_eff_inv2" in df.columns:
        nu_eff_inv2 = df["nu_eff_inv2"].astype(float).to_numpy()
        dmcorr_sec = K_DM * dm_vals * nu_eff_inv2
    else:
        dmcorr_sec = K_DM * dm_vals / (freq_mhz ** 2)
    dmcorr_days = dmcorr_sec / 86400.0
    # TDB a frecuencia infinita
    df["mjd_bary_tdb_inf"] = df["mjd_bary_tdb"] - dmcorr_days
    # UTC a frecuencia infinita
    df["mjd_bary_utc_inf"] = df["mjd_bary_utc"] - dmcorr_days
    return df

def convert_times_to_mjd(
    file_path: str,
    times_sec: list[float],
    output_csv: str,
    compute_bary: bool = True,
    ra: str = "05:31:58.70",
    dec: str = "33:08:52.5",
    freq_mhz: float = 1400.0,
    location: str = "Effelsberg",
    ephem: str = "de432s",
    default_dm: float | None = 565.0,
    center_sample: bool = True,
    nu_ref_mode: str = "header_mean",
):
    # Metadatos del archivo
    meta = get_file_metadata(file_path)
    mjd_start = float(meta["tstart_mjd"])
    tsamp_s = float(meta.get("tsamp_s", 0.0))

    # Preparar filas base
    df = pd.DataFrame({
        "file": [Path(file_path).name] * len(times_sec),
        "t_sec": times_sec,
    })

    # Centrar en medio de sample si corresponde
    if center_sample:
        t_sec_centered = df["t_sec"].astype(float) + (tsamp_s / 2.0)
    else:
        t_sec_centered = df["t_sec"].astype(float)
    df["t_sec_centered"] = t_sec_centered

    # MJD topocéntrico UTC
    df["mjd_start_file"] = mjd_start
    df["mjd_utc"] = df["mjd_start_file"] + (t_sec_centered / 86400.0)

    # Frecuencia efectiva <1/nu^2> desde header si se solicita
    if nu_ref_mode == "header_mean" and "chan_freqs_MHz" in meta:
        freqs = np.asarray(meta["chan_freqs_MHz"], dtype=float)
        valid = freqs > 0
        if valid.any():
            df["nu_eff_inv2"] = float(np.mean(1.0 / (freqs[valid] ** 2)))

    # Añadir DM por defecto si no hay columna
    if default_dm is not None:
        df["dm_pc_cm-3"] = float(default_dm)

    # Baricéntrico si corresponde
    if compute_bary:
        df = _compute_barycentric_columns(
            df,
            ra=ra,
            dec=dec,
            freq_mhz=freq_mhz,
            location=location,
            ephem=ephem,
            default_dm=default_dm,
        )

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return str(out_path)

def convert_candidates_to_mjd(candidates_csv: str, data_dir: str, output_csv: str,
                              compute_bary: bool = False,
                              ra: str = "05:31:58.70",
                              dec: str = "33:08:52.5",
                              freq_mhz: float = 1400.0,
                              location: str = "Effelsberg",
                              ephem: str = "de432s",
                              default_dm: float | None = 565.0,
                              center_sample: bool = True,
                              nu_ref_mode: str = "header_mean") -> str:
    df = pd.read_csv(candidates_csv)
    if "file" not in df.columns or "t_sec" not in df.columns:
        raise ValueError("El CSV debe contener columnas 'file' y 't_sec'")

    # Calcular MJD_start por archivo
    filenames = df["file"].unique()
    file_to_mjdstart = {}
    file_to_tsamp = {}
    file_to_nu_eff_inv2 = {}
    file_to_ra = {}
    file_to_dec = {}
    data_dir_path = Path(data_dir)
    for fname in filenames:
        # Resolver ruta
        cand_path = data_dir_path / fname
        if not cand_path.exists():
            # Intentar usar el path tal cual
            cand_path = Path(fname)
        # Metadatos precisos
        meta = get_file_metadata(str(cand_path))
        mjd_start = float(meta["tstart_mjd"])
        tsamp_s = float(meta.get("tsamp_s", 0.0))
        file_to_tsamp[fname] = tsamp_s
        # Frec efectiva: <1/nu^2> sobre canales del header
        if nu_ref_mode == "header_mean" and "chan_freqs_MHz" in meta:
            freqs = np.asarray(meta["chan_freqs_MHz"], dtype=float)
            # Evitar ceros o negativos
            valid = freqs > 0
            if valid.any():
                nu_eff_inv2 = np.mean(1.0 / (freqs[valid] ** 2))
                file_to_nu_eff_inv2[fname] = float(nu_eff_inv2)
        # RA/DEC si el header las trae
        if meta.get("ra_hms") and meta.get("dec_dms"):
            file_to_ra[fname] = meta["ra_hms"]
            file_to_dec[fname] = meta["dec_dms"]
        file_to_mjdstart[fname] = mjd_start

    # Añadir columnas topo
    df["mjd_start_file"] = df["file"].map(file_to_mjdstart)
    # Centrar en el medio del sample si se solicita
    if center_sample:
        # sumar tsamp/2 por fila según el archivo correspondiente
        tsamp_series = df["file"].map(file_to_tsamp).astype(float).fillna(0.0)
        t_sec_centered = df["t_sec"].astype(float) + (tsamp_series / 2.0)
    else:
        t_sec_centered = df["t_sec"].astype(float)
    df["t_sec_centered"] = t_sec_centered
    df["mjd_utc"] = df["mjd_start_file"] + (t_sec_centered / 86400.0)

    # Añadir nu_eff_inv2 si lo tenemos
    if file_to_nu_eff_inv2:
        df["nu_eff_inv2"] = df["file"].map(file_to_nu_eff_inv2)

    # Opcional: añadir columnas baricéntricas (TDB) y a frecuencia infinita
    if compute_bary:
        # RA/DEC por fila, si hay disponibles desde el header
        ra_series = df["file"].map(file_to_ra) if file_to_ra else None
        dec_series = df["file"].map(file_to_dec) if file_to_dec else None
        df = _compute_barycentric_columns(
            df,
            ra=ra,
            dec=dec,
            freq_mhz=freq_mhz,
            location=location,
            ephem=ephem,
            default_dm=default_dm,
            ra_series=ra_series,
            dec_series=dec_series,
        )

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return str(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utilidades MJD: convertir t_sec de candidatos a MJD absoluto y opcionalmente baricéntrico")
    parser.add_argument("--candidates_csv", type=str, help="Ruta al CSV de candidatos con columnas 'file' y 't_sec'")
    parser.add_argument("--data_dir", type=str, default="Data/raw", help="Directorio base donde están los .fil/.fits")
    parser.add_argument("--output", type=str, help="Ruta de salida para CSV con columna 'mjd_utc'")
    # Calcular para tiempos específicos de un archivo
    parser.add_argument("--file", type=str, help="Ruta a un .fil/.fits para calcular tiempos específicos")
    parser.add_argument("--times", type=str, help="Lista de tiempos en segundos separados por coma")
    parser.add_argument("--compute_bary", action="store_true", help="Añadir columnas baricéntricas (TDB/UTC) y a freq infinita")
    parser.add_argument("--ra", type=str, default="05:31:58.70", help="RA de la fuente en HH:MM:SS.ss")
    parser.add_argument("--dec", type=str, default="33:08:52.5", help="DEC de la fuente en DD:MM:SS.s")
    parser.add_argument("--freq_mhz", type=float, default=1400.0, help="Frecuencia de referencia en MHz")
    parser.add_argument("--location", type=str, default="Effelsberg", help="Observatorio (EarthLocation.of_site)")
    parser.add_argument("--ephem", type=str, default="de432s", help="Efemérides JPL (ej. de432s)")
    parser.add_argument("--default_dm", type=float, default=565.0, help="DM por defecto si no está en el CSV")
    parser.add_argument("--no_center_sample", action="store_true", help="No sumar tsamp/2 al t_sec (por defecto se suma)")
    parser.add_argument("--nu_ref_mode", type=str, default="header_mean", choices=["header_mean","fixed"], help="'header_mean': usa <1/nu^2> del header; 'fixed': usa --freq_mhz")

    # Parámetros opcionales para ejemplo baricéntrico anterior
    parser.add_argument("--example_bary", action="store_true", help="Ejecutar ejemplo de conversión baricéntrica")
    args = parser.parse_args()

    if args.candidates_csv:
        in_csv = args.candidates_csv
        if args.output:
            out_csv = args.output
        else:
            base = Path(in_csv).with_suffix("")
            out_csv = str(Path("Results") / (base.name + "_with_mjd.csv"))
        out_path = convert_candidates_to_mjd(
            in_csv, args.data_dir, out_csv,
            compute_bary=args.compute_bary,
            ra=args.ra, dec=args.dec,
            freq_mhz=args.freq_mhz,
            location=args.location,
            ephem=args.ephem,
            default_dm=args.default_dm,
            center_sample=(not args.no_center_sample),
            nu_ref_mode=args.nu_ref_mode,
        )
        print(f"✅ Escrito {out_path}")
    elif args.file and args.times:
        times_list = [float(x) for x in args.times.replace(';', ',').split(',') if x.strip()]
        out_csv = args.output or str(Path("Results") / (Path(args.file).stem + "_times_mjd.csv"))
        out_path = convert_times_to_mjd(
            file_path=args.file,
            times_sec=times_list,
            output_csv=out_csv,
            compute_bary=args.compute_bary or True,
            ra=args.ra,
            dec=args.dec,
            freq_mhz=args.freq_mhz,
            location=args.location,
            ephem=args.ephem,
            default_dm=args.default_dm,
            center_sample=(not args.no_center_sample),
            nu_ref_mode=args.nu_ref_mode,
        )
        print(f"✅ Escrito {out_path}")
    elif args.example_bary:
        # Ejemplo anterior (requiere archivo con columnas esperadas)
        filename = "candidates.txt"
        mjd_files, times_s = np.loadtxt(filename, usecols=(7, 3), unpack=True, skiprows=1)
        topo_mjds = get_topo_MJD(mjd_files, times_s)
        topo, bary = get_bary_MJD(
            topo_mjds,
            RA="05:31:58.70",
            DEC="33:08:52.5",
            DM=565,
            freq_MHz=1400.0,
            location="Effelsberg",
        )
        out = np.column_stack((topo, bary))
        out = out[np.argsort(out[:, 0])]
        np.savetxt("bary_toas.txt", out, fmt="%.12f", header="Topo_MJD  Bary_MJD")
        print("✅ Saved barycentric TOAs to bary_toas.txt")
    else:
        print("Nada que hacer: proporcione --candidates_csv o --example_bary")