#!/usr/bin/env python3
"""Simple test script for SNR functions."""

import numpy as np
import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from DRAFTS.snr_utils import compute_snr_profile, find_snr_peak, inject_synthetic_frb
    print("✓ SNR utils imported successfully")
    
    # Test 1: Basic SNR calculation
    print("\n=== Test 1: Basic SNR Calculation ===")
    np.random.seed(42)
    data = np.random.normal(0, 1, (500, 256))
    snr, sigma = compute_snr_profile(data)
    print(f"Data shape: {data.shape}")
    print(f"SNR shape: {snr.shape}")
    print(f"SNR mean: {snr.mean():.3f} (should be ~0)")
    print(f"SNR std: {snr.std():.3f} (should be ~1)")
    print(f"Sigma: {sigma:.3f}")
    
    # Test 2: Injected FRB
    print("\n=== Test 2: Injected FRB ===")
    np.random.seed(123)
    noise = np.random.normal(0, 1, (1000, 256))
    frb_data = inject_synthetic_frb(noise, 500, 128, 8.0)
    snr_frb, sigma_frb = compute_snr_profile(frb_data)
    peak_snr, peak_time, peak_idx = find_snr_peak(snr_frb)
    
    print(f"Injected amplitude: 8.0")
    print(f"Detected peak SNR: {peak_snr:.2f}")
    print(f"Peak position: {peak_idx} (injected at 500)")
    print(f"Sigma: {sigma_frb:.3f}")
    
    # Test 3: Off regions
    print("\n=== Test 3: Off Regions ===")
    off_regions = [(50, 150), (850, 950)]
    snr_off, sigma_off = compute_snr_profile(frb_data, off_regions)
    peak_snr_off, _, _ = find_snr_peak(snr_off)
    
    print(f"SNR with off regions - Peak: {peak_snr_off:.2f}")
    print(f"Sigma with off regions: {sigma_off:.3f}")
    
    print("\n✓ All tests completed successfully!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()
