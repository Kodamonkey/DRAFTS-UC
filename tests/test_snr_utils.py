"""Tests for SNR calculation utilities."""
from __future__ import annotations

import numpy as np
import pytest

from DRAFTS.snr_utils import (
    compute_snr_profile,
    estimate_sigma_iqr,
    find_snr_peak,
    inject_synthetic_frb,
    create_snr_regions_around_peak,
    compute_detection_significance,
)


def test_snr_consistency():
    """Test that SNR calculation works correctly with pure Gaussian noise."""
    np.random.seed(42)
    
    # Generate pure Gaussian noise waterfall
    n_time, n_freq = 1000, 256
    noise_waterfall = np.random.normal(0, 1, (n_time, n_freq))
    
    # Calculate SNR profile
    snr, sigma = compute_snr_profile(noise_waterfall)
    
    # For pure noise, SNR should be approximately N(0,1)
    assert abs(np.mean(snr)) < 0.1, f"SNR mean should be ~0, got {np.mean(snr)}"
    assert abs(np.std(snr) - 1.0) < 0.1, f"SNR std should be ~1, got {np.std(snr)}"
    assert sigma > 0, "Sigma should be positive"


def test_injected_frb():
    """Test SNR calculation with an injected synthetic FRB."""
    np.random.seed(123)
    
    # Generate background noise
    n_time, n_freq = 1000, 256
    noise_waterfall = np.random.normal(0, 1, (n_time, n_freq))
    
    # Inject synthetic FRB with amplitude 10 sigma
    target_amplitude = 10.0
    peak_time = n_time // 2
    peak_freq = n_freq // 2
    
    waterfall_with_frb = inject_synthetic_frb(
        noise_waterfall, peak_time, peak_freq, target_amplitude
    )
    
    # Calculate SNR
    snr, sigma = compute_snr_profile(waterfall_with_frb)
    
    # Find peak
    peak_snr, peak_time_found, peak_idx = find_snr_peak(snr)
    
    # Peak should be approximately 10 sigma
    assert abs(peak_snr - target_amplitude) < 0.5, \
        f"Peak SNR should be ~{target_amplitude}, got {peak_snr}"
    
    # Peak should be at correct location
    assert abs(peak_idx - peak_time) < 5, \
        f"Peak should be near {peak_time}, found at {peak_idx}"


def test_estimate_sigma_iqr():
    """Test IQR-based sigma estimation."""
    np.random.seed(456)
    
    # Test with pure Gaussian noise
    data = np.random.normal(0, 2.0, 10000)
    estimated_sigma = estimate_sigma_iqr(data)
    
    # Should recover the true sigma within 10%
    assert abs(estimated_sigma - 2.0) < 0.2, \
        f"Should estimate sigma~2.0, got {estimated_sigma}"


def test_snr_with_off_regions():
    """Test SNR calculation using specified off-pulse regions."""
    np.random.seed(789)
    
    # Create waterfall with a pulse in the middle
    n_time, n_freq = 500, 128
    waterfall = np.random.normal(0, 1, (n_time, n_freq))
    
    # Add pulse at center
    pulse_start = n_time // 2 - 10
    pulse_end = n_time // 2 + 10
    waterfall[pulse_start:pulse_end, :] += 5.0
    
    # Define off regions that avoid the pulse
    off_regions = [(50, 150), (350, 450)]
    
    snr, sigma = compute_snr_profile(waterfall, off_regions)
    
    # The pulse region should have high SNR
    pulse_snr = np.mean(snr[pulse_start:pulse_end])
    assert pulse_snr > 3.0, f"Pulse region should have high SNR, got {pulse_snr}"
    
    # Off regions should have low SNR
    off_snr_1 = np.mean(snr[off_regions[0][0]:off_regions[0][1]])
    off_snr_2 = np.mean(snr[off_regions[1][0]:off_regions[1][1]])
    
    assert abs(off_snr_1) < 1.0, f"Off region 1 should have low SNR, got {off_snr_1}"
    assert abs(off_snr_2) < 1.0, f"Off region 2 should have low SNR, got {off_snr_2}"


def test_create_snr_regions_around_peak():
    """Test creation of off-pulse regions around a detected peak."""
    profile_length = 1000
    peak_idx = 500
    region_width = 50
    
    regions = create_snr_regions_around_peak(peak_idx, profile_length, region_width)
    
    # Should create regions that don't overlap with peak
    for start, end in regions:
        assert start >= 0 and end <= profile_length, "Regions should be within bounds"
        assert start < end, "Region start should be before end"
        assert end < peak_idx - region_width or start > peak_idx + region_width, \
            "Regions should not overlap with peak area"


def test_dedispersion_snr():
    """Test that SNR calculation works correctly with dedispersed data."""
    np.random.seed(999)
    
    # Simulate dispersed pulse across frequency
    n_time, n_freq = 800, 256
    waterfall = np.random.normal(0, 1, (n_time, n_freq))
    
    # Add dispersed pulse (simple linear sweep)
    pulse_amplitude = 8.0
    center_time = n_time // 2
    
    for i in range(n_freq):
        # Simple dispersion delay proportional to frequency index
        delay = int(i * 0.5)  # Simplified dispersion
        pulse_time = center_time + delay
        if 0 <= pulse_time < n_time:
            # Add Gaussian pulse
            for dt in range(-5, 6):
                if 0 <= pulse_time + dt < n_time:
                    waterfall[pulse_time + dt, i] += pulse_amplitude * np.exp(-0.5 * (dt/2)**2)
    
    # Calculate SNR on "dedispersed" data (in reality just the original)
    # In a real case, this would be dedispersed first
    snr, sigma = compute_snr_profile(waterfall)
    
    # Should detect some signal
    peak_snr, _, _ = find_snr_peak(snr)
    assert peak_snr > 3.0, f"Should detect dispersed signal, peak SNR = {peak_snr}"


def test_detection_significance():
    """Test calculation of detection significance."""
    # Test simple case
    snr_peak = 6.0
    n_samples = 1000
    n_trials = 100
    
    significance = compute_detection_significance(snr_peak, n_samples, n_trials)
    
    # Significance should be positive but less than peak SNR due to trials correction
    assert 0 <= significance <= snr_peak, \
        f"Significance should be between 0 and {snr_peak}, got {significance}"


def test_find_snr_peak_with_time_axis():
    """Test peak finding with custom time axis."""
    # Create SNR profile with known peak
    snr = np.array([0, 1, 2, 5, 8, 3, 1, 0])
    time_axis = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    
    peak_snr, peak_time, peak_idx = find_snr_peak(snr, time_axis)
    
    assert peak_snr == 8, f"Peak SNR should be 8, got {peak_snr}"
    assert peak_time == 0.4, f"Peak time should be 0.4, got {peak_time}"
    assert peak_idx == 4, f"Peak index should be 4, got {peak_idx}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
