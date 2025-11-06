import argparse
from pathlib import Path
from typing import List, Optional
from src.core.pipeline import run_pipeline


def parse_args():
    """Parse command-line arguments for pipeline configuration."""
    parser = argparse.ArgumentParser(
        description="FRB Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  
  # Run with user_config.py (no CLI args):
  python main.py
  
  # Execute with specific files and custom thresholds:
  python main.py --data-dir "./Data/raw/" --results-dir "./Results/" \
    --target "2017-04-03-08_55_22" --det-prob 0.5 --class-prob 0.6
  
  # Override only specific parameters (rest from user_config.py):
  python main.py --dm-max 512 --det-prob 0.6
  
  # Enable multi-band analysis:
  python main.py --multi-band --slice-duration 3000.0
  
  # High-frequency processing with custom threshold:
  python main.py --auto-high-freq --high-freq-threshold 7500.0
        """
    )
    
    # =============================================================================
    # DATA AND FILE CONFIGURATION
    # =============================================================================
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing input files (.fits, .fil). If not provided, uses user_config.py")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory where results will be stored. If not provided, uses user_config.py")
    parser.add_argument("--target", "--targets", dest="targets", type=str, nargs="+", default=None,
                        help="List of files to process (search patterns). If not provided, uses user_config.py")
    
    # =============================================================================
    # TEMPORAL ANALYSIS CONFIGURATION
    # =============================================================================
    parser.add_argument("--slice-duration", type=float, default=300.0,
                        help="Duration of each temporal segment (ms) [default: 300.0]")
    
    # =============================================================================
    # DOWNSAMPLING CONFIGURATION
    # =============================================================================
    parser.add_argument("--down-freq-rate", type=int, default=1,
                        help="Frequency reduction factor [default: 1]")
    parser.add_argument("--down-time-rate", type=int, default=8,
                        help="Time reduction factor [default: 8]")
    
    # =============================================================================
    # DISPERSION MEASURE CONFIGURATION (DM)
    # =============================================================================
    parser.add_argument("--dm-min", type=int, default=0,
                        help="Minimum DM in pc cm⁻³ [default: 0]")
    parser.add_argument("--dm-max", type=int, default=1024,
                        help="Maximum DM in pc cm⁻³ [default: 1024]")
    
    # =============================================================================
    # DETECTION THRESHOLDS
    # =============================================================================
    parser.add_argument("--det-prob", type=float, default=0.3,
                        help="Minimum probability to consider a valid detection [default: 0.3]")
    parser.add_argument("--class-prob", type=float, default=0.5,
                        help="Minimum probability to classify as burst [default: 0.5]")
    parser.add_argument("--snr-thresh", type=float, default=5.0,
                        help="SNR threshold used in visualizations [default: 5.0]")
    
    # =============================================================================
    # MULTI-BAND ANALYSIS CONFIGURATION
    # =============================================================================
    parser.add_argument("--multi-band", dest="multi_band", action="store_true",
                        help="Enable multi-band analysis (Full/Low/High)")
    
    # =============================================================================
    # HIGH-FREQUENCY PIPELINE CONFIGURATION
    # =============================================================================
    parser.add_argument("--auto-high-freq", dest="auto_high_freq", action="store_true",
                        help="Automatically enable high-frequency pipeline")
    parser.add_argument("--no-auto-high-freq", dest="auto_high_freq", action="store_false",
                        help="Disable automatic high-frequency pipeline")
    parser.add_argument("--high-freq-threshold", type=float, default=8000.0,
                        help="Central frequency threshold (MHz) to consider 'high frequency'")
    
    # =============================================================================
    # POLARIZATION CONFIGURATION
    # =============================================================================
    parser.add_argument("--polarization-mode", type=str, default="intensity",
                        choices=["intensity", "linear", "circular", "pol0", "pol1", "pol2", "pol3"],
                        help="Polarization mode for PSRFITS (intensity/linear/circular/pol0-3)")
    parser.add_argument("--polarization-index", type=int, default=0,
                        help="Default index when IQUV is not available")
    
    # =============================================================================
    # LOGGING AND DEBUG CONFIGURATION
    # =============================================================================
    parser.add_argument("--debug-frequency", dest="debug_frequency", action="store_true",
                        help="Show detailed frequency and file information")
    parser.add_argument("--force-plots", dest="force_plots", action="store_true",
                        help="Always generate plots (even without candidates)")
    parser.add_argument("--no-force-plots", dest="force_plots", action="store_false",
                        help="Do not generate plots when no candidates found")
    
    # =============================================================================
    # CANDIDATE FILTERING CONFIGURATION
    # =============================================================================
    parser.add_argument("--save-all", dest="save_only_burst", action="store_false",
                        help="Save all candidates (not only BURST)")
    parser.add_argument("--save-only-burst", dest="save_only_burst", action="store_true",
                        help="Save only candidates classified as BURST")
    
    # =============================================================================
    # DEFAULTS for boolean arguments
    # =============================================================================
    parser.set_defaults(
        multi_band=False,           # Multi-band: disabled
        auto_high_freq=True,        # Auto high-freq: enabled
        debug_frequency=False,      # Debug: disabled
        force_plots=False,          # Force plots: disabled
        save_only_burst=True        # Save only BURST: enabled
    )
    
    args = parser.parse_args()
    return args


def main():
    import sys
    
    # Configure UTF-8 encoding for Windows
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    args = parse_args()
    
    # Load defaults from user_config.py
    from src.config import user_config
    
    # Determine configuration source
    using_cli_args = (args.data_dir is not None or args.results_dir is not None or args.targets is not None)
    
    if not using_cli_args:
        # Use user_config.py entirely
        print("=" * 80)
        print("CONFIGURATION SOURCE: user_config.py")
        print("=" * 80)
        print("(Use CLI arguments to override specific parameters)")
        print()
        
        config_dict = {
            "DATA_DIR": user_config.DATA_DIR,
            "RESULTS_DIR": user_config.RESULTS_DIR,
            "FRB_TARGETS": user_config.FRB_TARGETS,
            "SLICE_DURATION_MS": user_config.SLICE_DURATION_MS,
            "DOWN_FREQ_RATE": user_config.DOWN_FREQ_RATE,
            "DOWN_TIME_RATE": user_config.DOWN_TIME_RATE,
            "DM_min": user_config.DM_min,
            "DM_max": user_config.DM_max,
            "DET_PROB": user_config.DET_PROB,
            "CLASS_PROB": user_config.CLASS_PROB,
            "SNR_THRESH": user_config.SNR_THRESH,
            "USE_MULTI_BAND": user_config.USE_MULTI_BAND,
            "AUTO_HIGH_FREQ_PIPELINE": user_config.AUTO_HIGH_FREQ_PIPELINE,
            "HIGH_FREQ_THRESHOLD_MHZ": user_config.HIGH_FREQ_THRESHOLD_MHZ,
            "POLARIZATION_MODE": user_config.POLARIZATION_MODE,
            "POLARIZATION_INDEX": user_config.POLARIZATION_INDEX,
            "DEBUG_FREQUENCY_ORDER": user_config.DEBUG_FREQUENCY_ORDER,
            "FORCE_PLOTS": user_config.FORCE_PLOTS,
            "SAVE_ONLY_BURST": user_config.SAVE_ONLY_BURST,
        }
    else:
        # CLI arguments take priority, but use user_config.py as fallback for missing values
        print("=" * 80)
        print("CONFIGURATION SOURCE: CLI Arguments (with user_config.py fallback)")
        print("=" * 80)
        
        config_dict = {
            "DATA_DIR": Path(args.data_dir) if args.data_dir is not None else user_config.DATA_DIR,
            "RESULTS_DIR": Path(args.results_dir) if args.results_dir is not None else user_config.RESULTS_DIR,
            "FRB_TARGETS": args.targets if args.targets is not None else user_config.FRB_TARGETS,
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
    
    # Validate critical parameters
    if not isinstance(config_dict['DATA_DIR'], Path):
        config_dict['DATA_DIR'] = Path(config_dict['DATA_DIR'])
    if not isinstance(config_dict['RESULTS_DIR'], Path):
        config_dict['RESULTS_DIR'] = Path(config_dict['RESULTS_DIR'])
    
    if not config_dict['DATA_DIR'].exists():
        print(f"ERROR: Data directory does not exist: {config_dict['DATA_DIR']}")
        print("Please create the directory or specify a valid path.")
        sys.exit(1)
    
    # Display configuration being used
    print("PIPELINE CONFIGURATION")
    print("=" * 80)
    print(f"Data directory:             {config_dict['DATA_DIR']}")
    print(f"Results directory:          {config_dict['RESULTS_DIR']}")
    print(f"Target files:               {', '.join(config_dict['FRB_TARGETS'])}")
    print(f"Slice duration:             {config_dict['SLICE_DURATION_MS']} ms")
    print(f"Frequency reduction:        {config_dict['DOWN_FREQ_RATE']}x")
    print(f"Time reduction:             {config_dict['DOWN_TIME_RATE']}x")
    print(f"DM range:                   {config_dict['DM_min']} - {config_dict['DM_max']} pc cm⁻³")
    print(f"Detection threshold:        {config_dict['DET_PROB']}")
    print(f"Classification threshold:   {config_dict['CLASS_PROB']}")
    print(f"SNR threshold:              {config_dict['SNR_THRESH']}")
    print(f"Multi-band:                 {'[ENABLED]' if config_dict['USE_MULTI_BAND'] else '[Disabled]'}")
    print(f"High-freq pipeline:         {'[ENABLED]' if config_dict['AUTO_HIGH_FREQ_PIPELINE'] else '[Disabled]'}")
    print(f"Polarization mode:          {config_dict['POLARIZATION_MODE']}")
    print(f"Force plots:                {'[YES]' if config_dict['FORCE_PLOTS'] else '[No]'}")
    print(f"Save only BURST:            {'[YES]' if config_dict['SAVE_ONLY_BURST'] else '[No - all]'}")
    print(f"Debug frequency:            {'[ENABLED]' if config_dict['DEBUG_FREQUENCY_ORDER'] else '[Disabled]'}")
    print("=" * 80)
    print()
    
    # Run pipeline with provided configuration
    run_pipeline(config_dict=config_dict)


if __name__ == "__main__":
    main()
