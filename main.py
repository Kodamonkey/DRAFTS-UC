import argparse
from pathlib import Path
from typing import List, Optional
from src.core.pipeline import run_pipeline


def parse_args():
    """Parse command-line arguments for pipeline configuration."""
    parser = argparse.ArgumentParser(
        description="Pipeline de Detección de FRB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  
  # Ejecutar con archivos específicos y umbrales personalizados:
  python main.py --target "2017-04-03-08_55_22" --det-prob 0.5 --class-prob 0.6
  
  # Cambiar directorios y configurar DM range:
  python main.py --data-dir "./Data/raw/" --results-dir "./Results/" --dm-max 512
  
  # Activar multi-band analysis:
  python main.py --multi-band --slice-duration 3000.0
  
  # Procesamiento de alta frecuencia con umbral personalizado:
  python main.py --auto-high-freq --high-freq-threshold 7500.0
        """
    )
    
    # =============================================================================
    # DATA AND FILE CONFIGURATION
    # =============================================================================
    parser.add_argument("--data-dir", type=str, required=True,
                        help="[OBLIGATORIO] Directorio con archivos de entrada (.fits, .fil)")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="[OBLIGATORIO] Directorio donde se almacenan los resultados")
    parser.add_argument("--target", "--targets", dest="targets", type=str, nargs="+", required=True,
                        help="[OBLIGATORIO] Lista de archivos a procesar (patrones de búsqueda)")
    
    # =============================================================================
    # TEMPORAL ANALYSIS CONFIGURATION
    # =============================================================================
    parser.add_argument("--slice-duration", type=float, default=300.0,
                        help="Duración de cada segmento temporal (ms) [por defecto: 300.0]")
    
    # =============================================================================
    # DOWNSAMPLING CONFIGURATION
    # =============================================================================
    parser.add_argument("--down-freq-rate", type=int, default=1,
                        help="Factor de reducción en frecuencia [por defecto: 1]")
    parser.add_argument("--down-time-rate", type=int, default=8,
                        help="Factor de reducción en tiempo [por defecto: 8]")
    
    # =============================================================================
    # DISPERSION MEASURE CONFIGURATION (DM)
    # =============================================================================
    parser.add_argument("--dm-min", type=int, default=0,
                        help="DM mínimo en pc cm⁻³ [por defecto: 0]")
    parser.add_argument("--dm-max", type=int, default=1024,
                        help="DM máximo en pc cm⁻³ [por defecto: 1024]")
    
    # =============================================================================
    # DETECTION THRESHOLDS
    # =============================================================================
    parser.add_argument("--det-prob", type=float, default=0.3,
                        help="Probabilidad mínima para considerar una detección válida [por defecto: 0.3]")
    parser.add_argument("--class-prob", type=float, default=0.5,
                        help="Probabilidad mínima para clasificar como burst [por defecto: 0.5]")
    parser.add_argument("--snr-thresh", type=float, default=5.0,
                        help="Umbral SNR usado en visualizaciones [por defecto: 5.0]")
    
    # =============================================================================
    # MULTI-BAND ANALYSIS CONFIGURATION
    # =============================================================================
    parser.add_argument("--multi-band", dest="multi_band", action="store_true",
                        help="Activar análisis multi-band (Full/Low/High)")
    
    # =============================================================================
    # HIGH-FREQUENCY PIPELINE CONFIGURATION
    # =============================================================================
    parser.add_argument("--auto-high-freq", dest="auto_high_freq", action="store_true",
                        help="Activar pipeline de alta frecuencia automáticamente")
    parser.add_argument("--no-auto-high-freq", dest="auto_high_freq", action="store_false",
                        help="Desactivar pipeline de alta frecuencia automático")
    parser.add_argument("--high-freq-threshold", type=float, default=8000.0,
                        help="Umbral de frecuencia central (MHz) para considerar 'alta frecuencia'")
    
    # =============================================================================
    # POLARIZATION CONFIGURATION
    # =============================================================================
    parser.add_argument("--polarization-mode", type=str, default="intensity",
                        choices=["intensity", "linear", "circular", "pol0", "pol1", "pol2", "pol3"],
                        help="Modo de polarización para PSRFITS (intensity/linear/circular/pol0-3)")
    parser.add_argument("--polarization-index", type=int, default=0,
                        help="Índice por defecto cuando IQUV no está disponible")
    
    # =============================================================================
    # LOGGING AND DEBUG CONFIGURATION
    # =============================================================================
    parser.add_argument("--debug-frequency", dest="debug_frequency", action="store_true",
                        help="Mostrar información detallada de frecuencia y archivos")
    parser.add_argument("--force-plots", dest="force_plots", action="store_true",
                        help="Siempre generar gráficos (incluso sin candidatos)")
    parser.add_argument("--no-force-plots", dest="force_plots", action="store_false",
                        help="No generar gráficos cuando no hay candidatos")
    
    # =============================================================================
    # CANDIDATE FILTERING CONFIGURATION
    # =============================================================================
    parser.add_argument("--save-all", dest="save_only_burst", action="store_false",
                        help="Guardar todos los candidatos (no solo BURST)")
    parser.add_argument("--save-only-burst", dest="save_only_burst", action="store_true",
                        help="Guardar solo candidatos clasificados como BURST")
    
    # =============================================================================
    # DEFAULTS para argumentos booleanos
    # =============================================================================
    parser.set_defaults(
        multi_band=False,           # Multi-band: desactivado
        auto_high_freq=True,        # Auto high-freq: activado
        debug_frequency=False,      # Debug: desactivado
        force_plots=False,          # Force plots: desactivado
        save_only_burst=True        # Save only BURST: activado
    )
    
    args = parser.parse_args()
    return args


def main():
    import sys
    
    # Configurar encoding UTF-8 para Windows
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    args = parse_args()
    
    # Convertir argumentos a diccionario de configuración
    config_dict = {
        "DATA_DIR": Path(args.data_dir),
        "RESULTS_DIR": Path(args.results_dir),
        "FRB_TARGETS": args.targets,
        "SLICE_DURATION_MS": args.slice_duration,
        "DOWN_FREQ_RATE": args.down_freq_rate,
        "DOWN_TIME_RATE": args.down_time_rate,
        "DM_min": args.dm_min,
        "DM_max": args.dm_max,
        "DET_PROB": args.det_prob,
        "CLASS_PROB": args.class_prob,
        "SNR_THRESH": args.snr_thresh,
        "USE_MULTI_BAND": args.multi_band,
        "AUTO_HIGH_FREQ_PIPELINE": args.auto_high_freq,
        "HIGH_FREQ_THRESHOLD_MHZ": args.high_freq_threshold,
        "POLARIZATION_MODE": args.polarization_mode,
        "POLARIZATION_INDEX": args.polarization_index,
        "DEBUG_FREQUENCY_ORDER": args.debug_frequency,
        "FORCE_PLOTS": args.force_plots,
        "SAVE_ONLY_BURST": args.save_only_burst,
    }
    
    # Mostrar configuración que se está usando
    print("=" * 80)
    print("CONFIGURACIÓN DEL PIPELINE")
    print("=" * 80)
    print(f"Directorio de datos:        {config_dict['DATA_DIR']}")
    print(f"Directorio de resultados:   {config_dict['RESULTS_DIR']}")
    print(f"Archivos objetivo:          {', '.join(config_dict['FRB_TARGETS'])}")
    print(f"Duración de slice:          {config_dict['SLICE_DURATION_MS']} ms")
    print(f"Reducción en frecuencia:    {config_dict['DOWN_FREQ_RATE']}x")
    print(f"Reducción en tiempo:        {config_dict['DOWN_TIME_RATE']}x")
    print(f"Rango DM:                   {config_dict['DM_min']} - {config_dict['DM_max']} pc cm⁻³")
    print(f"Umbral de detección:        {config_dict['DET_PROB']}")
    print(f"Umbral de clasificación:    {config_dict['CLASS_PROB']}")
    print(f"Umbral SNR:                 {config_dict['SNR_THRESH']}")
    print(f"Multi-band:                 {'[ACTIVADO]' if config_dict['USE_MULTI_BAND'] else '[Desactivado]'}")
    print(f"Pipeline alta frecuencia:   {'[ACTIVADO]' if config_dict['AUTO_HIGH_FREQ_PIPELINE'] else '[Desactivado]'}")
    print(f"Forzar gráficos:            {'[SI]' if config_dict['FORCE_PLOTS'] else '[No]'}")
    print(f"Guardar solo BURST:         {'[SI]' if config_dict['SAVE_ONLY_BURST'] else '[No - todos]'}")
    print(f"Debug frecuencia:           {'[ACTIVADO]' if config_dict['DEBUG_FREQUENCY_ORDER'] else '[Desactivado]'}")
    print("=" * 80)
    print()
    
    # Ejecutar pipeline con la configuración proporcionada
    run_pipeline(config_dict=config_dict)


if __name__ == "__main__":
    main()
