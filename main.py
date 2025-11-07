"""
DRAFTS++ Pipeline Entry Point
==============================
Runs the FRB detection pipeline with configuration from config.yaml

Usage:
    python main.py

Configuration:
    Edit config.yaml in the project root to customize parameters.
"""

import sys
from pathlib import Path
from src.core.pipeline import run_pipeline


def main():
    # Configure UTF-8 encoding for Windows
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    # Load configuration from config.yaml (via user_config.py)
    from src.config import user_config
    
    # Build configuration dictionary
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
    
    # Validate critical parameters
    if not isinstance(config_dict['DATA_DIR'], Path):
        config_dict['DATA_DIR'] = Path(config_dict['DATA_DIR'])
    if not isinstance(config_dict['RESULTS_DIR'], Path):
        config_dict['RESULTS_DIR'] = Path(config_dict['RESULTS_DIR'])
    
    if not config_dict['DATA_DIR'].exists():
        print(f"ERROR: Data directory does not exist: {config_dict['DATA_DIR']}")
        print("Please create the directory or specify a valid path in config.yaml")
        sys.exit(1)
    
    # Display configuration
    print("=" * 80)
    print("DRAFTS++ PIPELINE - Configuration Loaded from config.yaml")
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
