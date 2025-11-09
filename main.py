"""
DRAFTS++ Pipeline Entry Point
==============================
Runs the FRB detection pipeline with configuration from config.yaml

Usage:
    python main.py                      # Use config.yaml
    python main.py --dm-max 512         # Override specific parameters
    python main.py --help               # Show all options

Configuration:
    Edit config.yaml in the project root to customize parameters.
    CLI arguments override config.yaml values.
"""

import sys
import argparse
from pathlib import Path
from src.core.pipeline import run_pipeline


def parse_args():
    """Parse optional command-line arguments to override config.yaml values."""
    parser = argparse.ArgumentParser(
        description="DRAFTS++ FRB Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config.yaml entirely
  python main.py
  
  # Override specific parameters
  python main.py --dm-max 512 --det-prob 0.5
  
  # Override data location (Local execution only)
  python main.py --data-dir "./MyData/" --target "FRB20180301"
  
  # Adjust thresholds for more sensitive detection
  python main.py --det-prob 0.2 --class-prob 0.4

Note: For Docker, data paths should be changed in docker-compose.yml, not via CLI.
        """
    )
    
    # Data and file configuration
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--data-dir', type=str, 
                           help='Input data directory (overrides config.yaml)')
    data_group.add_argument('--results-dir', type=str,
                           help='Results output directory (overrides config.yaml)')
    data_group.add_argument('--target', '--targets', dest='targets', type=str, nargs='+',
                           help='File patterns to process (overrides config.yaml)')
    
    # Temporal analysis
    temporal_group = parser.add_argument_group('Temporal Analysis')
    temporal_group.add_argument('--slice-duration', type=float,
                               help='Temporal slice duration in milliseconds (default: from config.yaml)')
    
    # Downsampling
    downsample_group = parser.add_argument_group('Downsampling')
    downsample_group.add_argument('--down-freq-rate', type=int,
                                 help='Frequency reduction factor (default: from config.yaml)')
    downsample_group.add_argument('--down-time-rate', type=int,
                                 help='Time reduction factor (default: from config.yaml)')
    
    # Dispersion measure
    dm_group = parser.add_argument_group('Dispersion Measure')
    dm_group.add_argument('--dm-min', type=int,
                         help='Minimum DM in pc cm⁻³ (default: from config.yaml)')
    dm_group.add_argument('--dm-max', type=int,
                         help='Maximum DM in pc cm⁻³ (default: from config.yaml)')
    
    # Detection thresholds
    threshold_group = parser.add_argument_group('Detection Thresholds')
    threshold_group.add_argument('--det-prob', type=float,
                                help='Detection probability threshold (0.0-1.0, default: from config.yaml)')
    threshold_group.add_argument('--class-prob', type=float,
                                help='Classification probability threshold (0.0-1.0, default: from config.yaml)')
    threshold_group.add_argument('--snr-thresh', type=float,
                                help='SNR threshold for visualization (default: from config.yaml)')
    
    # Multi-band and features
    features_group = parser.add_argument_group('Features')
    features_group.add_argument('--multi-band', action='store_true',
                               help='Enable multi-band analysis')
    features_group.add_argument('--no-multi-band', action='store_false', dest='multi_band',
                               help='Disable multi-band analysis')
    
    # High-frequency pipeline
    hf_group = parser.add_argument_group('High-Frequency Pipeline')
    hf_group.add_argument('--auto-high-freq', action='store_true', dest='auto_high_freq',
                         help='Enable automatic high-frequency pipeline')
    hf_group.add_argument('--no-auto-high-freq', action='store_false', dest='auto_high_freq',
                         help='Disable automatic high-frequency pipeline')
    hf_group.add_argument('--high-freq-threshold', type=float,
                         help='High-frequency threshold in MHz (default: from config.yaml)')
    
    # Polarization
    pol_group = parser.add_argument_group('Polarization (PSRFITS)')
    pol_group.add_argument('--polarization-mode', type=str,
                          choices=['intensity', 'linear', 'circular', 'pol0', 'pol1', 'pol2', 'pol3'],
                          help='Polarization mode (default: from config.yaml)')
    pol_group.add_argument('--polarization-index', type=int,
                          help='Default polarization index (default: from config.yaml)')
    
    # Debug and output
    debug_group = parser.add_argument_group('Debug and Output')
    debug_group.add_argument('--debug-frequency', action='store_true',
                            help='Show detailed frequency information')
    debug_group.add_argument('--force-plots', action='store_true',
                            help='Generate plots even without candidates')
    debug_group.add_argument('--save-all', action='store_true',
                            help='Save all candidates (not only BURST)')
    debug_group.add_argument('--save-only-burst', action='store_true',
                            help='Save only BURST candidates (default)')
    
    parser.set_defaults(multi_band=None, auto_high_freq=None)
    
    return parser.parse_args()


def main():
    # Configure UTF-8 encoding for Windows
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    # Parse command-line arguments
    args = parse_args()
    
    # Load configuration from config.yaml (via user_config.py)
    from src.config import user_config
    
    # Build configuration dictionary from config.yaml
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
    
    # Override with CLI arguments (if provided)
    cli_overrides = []
    
    if args.data_dir is not None:
        config_dict["DATA_DIR"] = Path(args.data_dir)
        cli_overrides.append(f"data_dir={args.data_dir}")
    
    if args.results_dir is not None:
        config_dict["RESULTS_DIR"] = Path(args.results_dir)
        cli_overrides.append(f"results_dir={args.results_dir}")
    
    if args.targets is not None:
        config_dict["FRB_TARGETS"] = args.targets
        cli_overrides.append(f"targets={args.targets}")
    
    if args.slice_duration is not None:
        config_dict["SLICE_DURATION_MS"] = args.slice_duration
        cli_overrides.append(f"slice_duration={args.slice_duration}ms")
    
    if args.down_freq_rate is not None:
        config_dict["DOWN_FREQ_RATE"] = args.down_freq_rate
        cli_overrides.append(f"down_freq_rate={args.down_freq_rate}")
    
    if args.down_time_rate is not None:
        config_dict["DOWN_TIME_RATE"] = args.down_time_rate
        cli_overrides.append(f"down_time_rate={args.down_time_rate}")
    
    if args.dm_min is not None:
        config_dict["DM_min"] = args.dm_min
        cli_overrides.append(f"dm_min={args.dm_min}")
    
    if args.dm_max is not None:
        config_dict["DM_max"] = args.dm_max
        cli_overrides.append(f"dm_max={args.dm_max}")
    
    if args.det_prob is not None:
        config_dict["DET_PROB"] = args.det_prob
        cli_overrides.append(f"det_prob={args.det_prob}")
    
    if args.class_prob is not None:
        config_dict["CLASS_PROB"] = args.class_prob
        cli_overrides.append(f"class_prob={args.class_prob}")
    
    if args.snr_thresh is not None:
        config_dict["SNR_THRESH"] = args.snr_thresh
        cli_overrides.append(f"snr_thresh={args.snr_thresh}")
    
    if args.multi_band is not None:
        config_dict["USE_MULTI_BAND"] = args.multi_band
        cli_overrides.append(f"multi_band={args.multi_band}")
    
    if args.auto_high_freq is not None:
        config_dict["AUTO_HIGH_FREQ_PIPELINE"] = args.auto_high_freq
        cli_overrides.append(f"auto_high_freq={args.auto_high_freq}")
    
    if args.high_freq_threshold is not None:
        config_dict["HIGH_FREQ_THRESHOLD_MHZ"] = args.high_freq_threshold
        cli_overrides.append(f"high_freq_threshold={args.high_freq_threshold}MHz")
    
    if args.polarization_mode is not None:
        config_dict["POLARIZATION_MODE"] = args.polarization_mode
        cli_overrides.append(f"polarization_mode={args.polarization_mode}")
    
    if args.polarization_index is not None:
        config_dict["POLARIZATION_INDEX"] = args.polarization_index
        cli_overrides.append(f"polarization_index={args.polarization_index}")
    
    if args.debug_frequency:
        config_dict["DEBUG_FREQUENCY_ORDER"] = True
        cli_overrides.append("debug_frequency=True")
    
    if args.force_plots:
        config_dict["FORCE_PLOTS"] = True
        cli_overrides.append("force_plots=True")
    
    if args.save_all:
        config_dict["SAVE_ONLY_BURST"] = False
        cli_overrides.append("save_only_burst=False")
    elif args.save_only_burst:
        config_dict["SAVE_ONLY_BURST"] = True
        cli_overrides.append("save_only_burst=True")
    
    # Validate critical parameters
    if not isinstance(config_dict['DATA_DIR'], Path):
        config_dict['DATA_DIR'] = Path(config_dict['DATA_DIR'])
    if not isinstance(config_dict['RESULTS_DIR'], Path):
        config_dict['RESULTS_DIR'] = Path(config_dict['RESULTS_DIR'])
    
    if not config_dict['DATA_DIR'].exists():
        print(f"ERROR: Data directory does not exist: {config_dict['DATA_DIR']}")
        print("Please create the directory or specify a valid path in config.yaml")
        sys.exit(1)
    
    # Display configuration source
    print("=" * 80)
    if cli_overrides:
        print("DRAFTS++ PIPELINE - Configuration from config.yaml + CLI overrides")
        print("=" * 80)
        print(f"CLI Overrides: {', '.join(cli_overrides)}")
    else:
        print("DRAFTS++ PIPELINE - Configuration from config.yaml")
        print("=" * 80)
    print()
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
    
    # Run pipeline with configuration
    run_pipeline(config_dict=config_dict)


if __name__ == "__main__":
    main()
