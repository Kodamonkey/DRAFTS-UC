import os
import sys
import argparse

# Usar backend headless
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Headless PRESTO waterfaller plotter")
    parser.add_argument("--filename", required=True, help="Ruta al archivo .fits o .fil")
    parser.add_argument("--start", type=float, required=True, help="Tiempo de inicio en segundos")
    parser.add_argument("--duration", type=float, required=True, help="Duración en segundos")
    parser.add_argument("--dm", type=float, default=0.0, help="DM para dedispersar (default 0)")
    parser.add_argument("--nsub", type=int, default=None, help="Número de subbandas (opcional)")
    parser.add_argument("--downsamp", type=int, default=1, help="Factor de downsample temporal")
    parser.add_argument("--output", required=True, help="PNG de salida")
    args = parser.parse_args()

    # Importar módulos de PRESTO
    # waterfaller.py está instalado como script en /software/presto5/installation/bin
    # lo agregamos al sys.path para importarlo como módulo
    sys.path.insert(0, "/software/presto5/installation/bin")
    import waterfaller as WF  # type: ignore
    from presto import psrfits, filterbank  # type: ignore

    fn = args.filename
    if fn.endswith(".fits"):
        raw = psrfits.PsrfitsFile(fn)
    elif fn.endswith(".fil"):
        raw = filterbank.FilterbankFile(fn)
    else:
        raise ValueError("Extensión no soportada. Use .fits o .fil")

    data, _nbinsextra, _nbins, start_corr = WF.waterfall(
        raw,
        args.start,
        args.duration,
        dm=args.dm,
        nsub=args.nsub,
        downsamp=args.downsamp,
    )

    # interactive=False evita plt.show() dentro de plot_waterfall
    WF.plot_waterfall(
        data,
        start_corr,
        args.duration,
        integrate_ts=True,
        integrate_spec=True,
        show_cb=True,
        interactive=False,
    )

    out = args.output
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()


