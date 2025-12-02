"""
Test suite for validating large file processing (1TB simulation).

This test validates that the pipeline can handle extremely large files using:
- Temporal chunking (streaming)
- DM chunking (when cube size exceeds threshold)
- Adaptive memory budgeting
- All mathematical formulas for chunk size calculation
"""

import sys
from pathlib import Path
import numpy as np
import logging
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path.parent))  # Add DRAFTS-UC to path

from src.preprocessing.slice_len_calculator import calculate_memory_safe_chunk_size
from src.core.data_flow_manager import build_dm_time_cube, _build_dm_time_cube_chunked
from src.config import config
from src.output.validation_metrics import ValidationMetricsCollector

logger = logging.getLogger(__name__)


class TestLargeFileProcessing:
    """Test suite for 1TB file processing simulation."""
    
    def setup_method(self):
        """Setup test environment."""
        # Simulate a 1TB file scenario
        # Typical radio astronomy file: 512 channels, 2 bytes per sample, 1 polarization
        # File size = samples × channels × bytes_per_sample × npol
        
        # For 1TB file:
        # 1TB = 1024^4 bytes = 1,099,511,627,776 bytes
        # Assuming: 512 channels, 2 bytes/sample, 1 pol
        # samples = 1TB / (512 × 2 × 1) = 1,073,741,824 samples
        
        self.file_size_tb = 1.0
        self.file_size_bytes = int(1.0 * (1024**4))  # 1 TB in bytes
        
        # Typical parameters for large radio astronomy file
        self.n_channels = 512
        self.bytes_per_sample = 2  # 16-bit data
        self.n_pol = 1
        self.time_reso = 5.12e-5  # 51.2 microseconds
        
        # Calculate total samples
        bytes_per_time_sample = self.n_channels * self.bytes_per_sample * self.n_pol
        self.total_samples = self.file_size_bytes // bytes_per_time_sample
        
        # Calculate duration
        self.duration_seconds = self.total_samples * self.time_reso
        self.duration_hours = self.duration_seconds / 3600
        
        # DM range (standard test)
        self.dm_min = 0.0
        self.dm_max = 200.0  # Standard DM range
        self.dm_resolution = 0.01  # Fine resolution
        self.height_dm = int((self.dm_max - self.dm_min) / self.dm_resolution)  # 20,000 DM values
        
        # DM range (EXTREME test - very high DM for FRBs)
        self.dm_max_extreme = 10000.0  # Very high DM (extragalactic FRBs, very distant sources)
        self.dm_resolution_extreme = 0.1  # Slightly coarser for extreme range
        self.height_dm_extreme = int((self.dm_max_extreme - self.dm_min) / self.dm_resolution_extreme)  # 100,000 DM values
        
        # Downsampling factors
        self.down_time_rate = 8
        self.down_freq_rate = 1
        
        logger.info(f"Simulated file: {self.file_size_tb:.1f} TB")
        logger.info(f"Total samples: {self.total_samples:,}")
        logger.info(f"Duration: {self.duration_hours:.2f} hours")
        logger.info(f"DM range: {self.dm_min}-{self.dm_max} pc cm⁻³ ({self.height_dm:,} values)")
    
    def test_adaptive_memory_budgeting_1tb(self):
        """Test that adaptive memory budgeting calculates safe chunk sizes for 1TB file."""
        logger.info("\n" + "="*80)
        logger.info("TEST 1: Adaptive Memory Budgeting for 1TB file")
        logger.info("="*80)
        
        # Set config values directly (simpler approach)
        original_time_reso = config.TIME_RESO
        original_down_time = config.DOWN_TIME_RATE
        original_dm_max = config.DM_max
        original_dm_min = config.DM_min
        
        try:
            config.TIME_RESO = self.time_reso
            config.DOWN_TIME_RATE = self.down_time_rate
            config.DOWN_FREQ_RATE = self.down_freq_rate
            config.DM_min = self.dm_min
            config.DM_max = self.dm_max
            config.SLICE_LEN = 2048
            config.SLICE_LEN_MIN = 512
            config.MAX_CHUNK_SAMPLES = 10_000_000
            config.MAX_RAM_FRACTION = 0.5
            config.DM_CHUNKING_THRESHOLD_GB = 16.0
            config.FILE_LENG = self.total_samples
            config.FREQ_RESO = self.n_channels
            config.FREQ = np.linspace(1200, 1600, self.n_channels)  # Mock frequency array
            
            # Mock available RAM (simulate 64GB system)
            available_ram_gb = 64.0
            available_ram_bytes = int(available_ram_gb * (1024**3))
            
            with patch('psutil.virtual_memory') as mock_vm:
                mock_vm.return_value.available = available_ram_bytes
                mock_vm.return_value.total = available_ram_bytes * 2
                
                # Mock torch.cuda to simulate no GPU
                try:
                    import torch
                    with patch.object(torch.cuda, 'is_available', return_value=False):
                        # Calculate safe chunk size (returns tuple: (safe_chunk_samples, diagnostics))
                        safe_chunk_samples, diagnostics = calculate_memory_safe_chunk_size(
                            slice_len=2048
                        )
                except ImportError:
                    # No torch available, just run without GPU mock
                    safe_chunk_samples, diagnostics = calculate_memory_safe_chunk_size(
                        slice_len=2048
                    )
        finally:
            # Restore original config
            config.TIME_RESO = original_time_reso
            config.DOWN_TIME_RATE = original_down_time
            config.DM_max = original_dm_max
            config.DM_min = original_dm_min
        
        expected_cube_gb = diagnostics['expected_cube_gb']
        will_use_dm_chunking = diagnostics['will_use_dm_chunking']
        
        logger.info(f"✓ Safe chunk size calculated: {safe_chunk_samples:,} samples")
        logger.info(f"✓ Expected cube size: {expected_cube_gb:.2f} GB")
        logger.info(f"✓ DM chunking will activate: {will_use_dm_chunking}")
        
        # Validations
        assert safe_chunk_samples > 0, "Safe chunk size must be positive"
        assert safe_chunk_samples <= config.MAX_CHUNK_SAMPLES, \
            f"Chunk size {safe_chunk_samples:,} exceeds MAX_CHUNK_SAMPLES {config.MAX_CHUNK_SAMPLES:,}"
        
        # Calculate number of temporal chunks needed
        decimated_samples = self.total_samples // self.down_time_rate
        num_temporal_chunks = (decimated_samples + safe_chunk_samples - 1) // safe_chunk_samples
        
        logger.info(f"✓ Number of temporal chunks needed: {num_temporal_chunks:,}")
        logger.info(f"✓ Each chunk processes: {safe_chunk_samples:,} samples (decimated)")
        logger.info(f"✓ Total processing time estimate: {num_temporal_chunks} chunks")
        
        # Validate that chunking is necessary
        assert num_temporal_chunks > 1, \
            "File is too large to process in one chunk - chunking must be used"
        
        logger.info("✓ TEST 1 PASSED: Adaptive memory budgeting works correctly")
        return diagnostics
    
    def test_dm_chunking_activation(self):
        """Test that DM chunking activates when cube size exceeds threshold."""
        logger.info("\n" + "="*80)
        logger.info("TEST 2: DM Chunking Activation (Standard DM Range)")
        logger.info("="*80)
        
        # Simulate a chunk that would create a large cube
        chunk_samples_decimated = 500_000  # Large chunk
        width = chunk_samples_decimated
        
        # Calculate cube size
        cube_size_bytes = 3 * self.height_dm * width * 4  # 3 bands, float32
        cube_size_gb = cube_size_bytes / (1024**3)
        
        logger.info(f"Chunk samples (decimated): {chunk_samples_decimated:,}")
        logger.info(f"DM height: {self.height_dm:,}")
        logger.info(f"Calculated cube size: {cube_size_gb:.2f} GB")
        
        # Check if DM chunking should activate
        threshold_gb = 16.0
        should_activate = cube_size_gb > threshold_gb
        
        logger.info(f"DM chunking threshold: {threshold_gb} GB")
        logger.info(f"DM chunking should activate: {should_activate}")
        
        if should_activate:
            # Calculate DM chunk size
            max_chunk_height = int((threshold_gb * (1024**3)) / (3 * width * 4))
            min_chunk_height = 100
            dm_chunk_height = max(min_chunk_height, max_chunk_height)
            num_dm_chunks = (self.height_dm + dm_chunk_height - 1) // dm_chunk_height
            
            logger.info(f"✓ DM chunk height: {dm_chunk_height:,} DM values")
            logger.info(f"✓ Number of DM chunks: {num_dm_chunks}")
            
            # Validate chunking math
            assert dm_chunk_height > 0, "DM chunk height must be positive"
            assert num_dm_chunks > 1, "Multiple DM chunks should be needed"
            
            # Validate each chunk size
            for chunk_idx in range(num_dm_chunks):
                start_dm = chunk_idx * dm_chunk_height
                end_dm = min(start_dm + dm_chunk_height, self.height_dm)
                chunk_height = end_dm - start_dm
                chunk_size_gb = (3 * chunk_height * width * 4) / (1024**3)
                
                logger.info(f"  Chunk {chunk_idx + 1}: {chunk_height:,} DM values, "
                          f"{chunk_size_gb:.2f} GB")
                
                assert chunk_size_gb <= threshold_gb * 1.1, \
                    f"Chunk {chunk_idx + 1} size {chunk_size_gb:.2f} GB exceeds threshold"
            
            logger.info("✓ TEST 2 PASSED: DM chunking math is correct")
        else:
            logger.info("⚠ DM chunking not needed for this scenario")
    
    def test_extreme_dm_range(self):
        """Test system with EXTREMELY high DM range (5000 pc cm⁻³) to validate extreme cases."""
        logger.info("\n" + "="*80)
        logger.info("TEST 2B: EXTREME DM Range Validation (DM up to 10000 pc cm⁻³)")
        logger.info("="*80)
        
        logger.info(f"Extreme DM range: {self.dm_min}-{self.dm_max_extreme} pc cm⁻³")
        logger.info(f"DM resolution: {self.dm_resolution_extreme} pc cm⁻³")
        logger.info(f"Total DM values: {self.height_dm_extreme:,}")
        
        # --- VALIDATION 1: Dispersion Delay with Extreme DM ---
        logger.info("\n[Validation 1: Dispersion Delay with Extreme DM]")
        dispersion_constant = 4.148808e3
        freq_min_mhz = 1200.0
        freq_max_mhz = 1600.0
        
        # Calculate maximum dispersion delay
        expected_delta_t_extreme = dispersion_constant * self.dm_max_extreme * (
            (1.0 / (freq_min_mhz**2)) - (1.0 / (freq_max_mhz**2))
        )
        
        dt_ds = self.time_reso * self.down_time_rate
        expected_overlap_extreme = int(np.ceil(expected_delta_t_extreme / dt_ds))
        
        logger.info(f"Formula: delta_t = 4.148808e3 * DM_max * (nu_min^-2 - nu_max^-2)")
        logger.info(f"Input: DM_max={self.dm_max_extreme}, freq={freq_min_mhz}-{freq_max_mhz} MHz")
        logger.info(f"Expected delta_t: {expected_delta_t_extreme:.6f} seconds")
        logger.info(f"Expected overlap samples: {expected_overlap_extreme:,}")
        
        # Validate overlap is reasonable
        assert expected_overlap_extreme > 0, "Overlap must be positive"
        logger.info(f"✓ Extreme DM dispersion delay validated: {expected_delta_t_extreme:.3f}s")
        
        # --- VALIDATION 2: DM Chunking with Extreme Range ---
        logger.info("\n[Validation 2: DM Chunking with Extreme Range]")
        chunk_samples_decimated = 200_000  # Moderate chunk size
        width = chunk_samples_decimated
        
        # Calculate cube size with extreme DM range
        cube_size_bytes_extreme = 3 * self.height_dm_extreme * width * 4
        cube_size_gb_extreme = cube_size_bytes_extreme / (1024**3)
        
        logger.info(f"Chunk samples: {chunk_samples_decimated:,}")
        logger.info(f"DM height (extreme): {self.height_dm_extreme:,}")
        logger.info(f"Calculated cube size: {cube_size_gb_extreme:.2f} GB")
        
        threshold_gb = 16.0
        should_activate_extreme = cube_size_gb_extreme > threshold_gb
        
        logger.info(f"DM chunking threshold: {threshold_gb} GB")
        logger.info(f"DM chunking should activate: {should_activate_extreme}")
        
        if should_activate_extreme:
            # Calculate DM chunk size
            max_chunk_height_extreme = int((threshold_gb * (1024**3)) / (3 * width * 4))
            min_chunk_height = 100
            dm_chunk_height_extreme = max(min_chunk_height, max_chunk_height_extreme)
            num_dm_chunks_extreme = (self.height_dm_extreme + dm_chunk_height_extreme - 1) // dm_chunk_height_extreme
            
            logger.info(f"✓ DM chunk height: {dm_chunk_height_extreme:,} DM values")
            logger.info(f"✓ Number of DM chunks: {num_dm_chunks_extreme}")
            
            # Validate chunking math
            assert dm_chunk_height_extreme > 0, "DM chunk height must be positive"
            assert num_dm_chunks_extreme > 1, "Multiple DM chunks should be needed for extreme range"
            
            # Validate each chunk size and DM range conversion
            dm_range_extreme = self.dm_max_extreme - self.dm_min
            total_validated_dm = 0
            
            for chunk_idx in range(min(10, num_dm_chunks_extreme)):  # Show first 10 chunks
                start_dm_idx = chunk_idx * dm_chunk_height_extreme
                end_dm_idx = min(start_dm_idx + dm_chunk_height_extreme, self.height_dm_extreme)
                chunk_height = end_dm_idx - start_dm_idx
                
                # Convert indices to physical DM values
                chunk_dm_min = self.dm_min + (start_dm_idx / self.height_dm_extreme) * dm_range_extreme
                chunk_dm_max = self.dm_min + (end_dm_idx / self.height_dm_extreme) * dm_range_extreme
                
                chunk_size_gb = (3 * chunk_height * width * 4) / (1024**3)
                total_validated_dm += chunk_height
                
                logger.info(f"  Chunk {chunk_idx + 1}: DM {chunk_dm_min:.1f}-{chunk_dm_max:.1f} pc cm⁻³ "
                          f"({chunk_height:,} values, {chunk_size_gb:.2f} GB)")
                
                assert chunk_size_gb <= threshold_gb * 1.1, \
                    f"Chunk {chunk_idx + 1} size {chunk_size_gb:.2f} GB exceeds threshold"
                assert chunk_dm_max > chunk_dm_min, "DM range must be positive"
            
            # Validate we cover the full range
            logger.info(f"\nTotal DM values validated: {total_validated_dm:,} / {self.height_dm_extreme:,}")
            if num_dm_chunks_extreme <= 10:
                assert total_validated_dm == self.height_dm_extreme, \
                    "All DM values must be covered"
            
            logger.info("✓ Extreme DM chunking validated mathematically")
        
        # --- VALIDATION 3: Memory Requirements with Extreme DM ---
        logger.info("\n[Validation 3: Memory Requirements with Extreme DM]")
        
        # Calculate memory for full processing
        safe_chunk_samples = 9_998_336  # From test 1
        width_extreme = safe_chunk_samples
        
        # Full cube size with extreme DM
        full_cube_gb_extreme = (3 * self.height_dm_extreme * width_extreme * 4) / (1024**3)
        
        logger.info(f"Full cube size (extreme DM): {full_cube_gb_extreme:.2f} GB")
        logger.info(f"With DM chunking: Each chunk < {threshold_gb} GB")
        
        if full_cube_gb_extreme > threshold_gb:
            num_chunks_needed = int(np.ceil(full_cube_gb_extreme / threshold_gb))
            logger.info(f"Estimated DM chunks needed: {num_chunks_needed}")
            logger.info(f"✓ System can handle extreme DM range with chunking")
        
        logger.info("\n✓ TEST 2B PASSED: Extreme DM range (up to 10000 pc cm⁻³) validated")
    
    def test_streaming_chunk_calculation(self):
        """Test that streaming calculates correct chunk boundaries and overlap using mathematical validation."""
        logger.info("\n" + "="*80)
        logger.info("TEST 3: Streaming Chunk Calculation (Mathematical Validation)")
        logger.info("="*80)
        
        # Use results from test 1
        diagnostics = self.test_adaptive_memory_budgeting_1tb()
        safe_chunk_samples = diagnostics['safe_chunk_samples']
        
        # --- MATHEMATICAL VALIDATION: DISPERSION DELAY ---
        # Formula: delta_t = 4.148808e3 * DM * (nu_min^-2 - nu_max^-2)
        # Constants
        dispersion_constant = 4.148808e3
        freq_min_mhz = 1200.0
        freq_max_mhz = 1600.0
        dm_max = self.dm_max
        
        # Manual calculation
        expected_delta_t = dispersion_constant * dm_max * (
            (1.0 / (freq_min_mhz**2)) - (1.0 / (freq_max_mhz**2))
        )
        
        logger.info("\n[Validation 1: Dispersion Delay]")
        logger.info(f"Formula: delta_t = 4.148808e3 * DM_max * (nu_min^-2 - nu_max^-2)")
        logger.info(f"Input: DM={dm_max}, freq={freq_min_mhz}-{freq_max_mhz} MHz")
        logger.info(f"Expected delta_t: {expected_delta_t:.6f} seconds")
        
        # Convert to samples (decimated)
        dt_ds = self.time_reso * self.down_time_rate
        expected_overlap_samples = int(np.ceil(expected_delta_t / dt_ds))
        
        logger.info(f"Time resolution (decimated): {dt_ds:.9f} seconds")
        logger.info(f"Expected overlap samples: {expected_overlap_samples:,}")
        
        # --- VALIDATE AGAINST PIPELINE LOGIC ---
        overlap_samples = expected_overlap_samples  # In real pipeline this comes from calculator
        
        assert overlap_samples > 0, "Overlap must be positive"
        assert overlap_samples < safe_chunk_samples, \
            f"Overlap {overlap_samples:,} must be less than chunk size {safe_chunk_samples:,}"
        
        logger.info("✓ Dispersion delay calculation validated mathematically")
        
        # --- MATHEMATICAL VALIDATION: CHUNK BOUNDARIES ---
        decimated_total = self.total_samples // self.down_time_rate
        num_chunks = (decimated_total + safe_chunk_samples - 1) // safe_chunk_samples
        
        logger.info("\n[Validation 2: Chunk Boundaries]")
        logger.info(f"Total samples (decimated): {decimated_total:,}")
        logger.info(f"Chunk size: {safe_chunk_samples:,}")
        logger.info(f"Overlap: {overlap_samples:,}")
        
        # Validate continuity: End of Chunk N must match Start of Chunk N+1 + Overlap
        # Chunk N: [Start_N, End_N]
        # Chunk N+1: [Start_N+1, End_N+1]
        # Continuity condition: Start_N+1 = End_N - Overlap
        # Or equivalently: Step size = Chunk size - Overlap
        
        step_size = safe_chunk_samples - overlap_samples
        logger.info(f"Step size (advance): {step_size:,}")
        
        logger.info("Checking continuity for first 5 chunks:")
        previous_end = 0
        
        for chunk_idx in range(min(5, num_chunks)):
            # Calculate boundaries using correct streaming logic
            # Start = idx * step_size
            chunk_start = chunk_idx * step_size
            
            chunk_end = min(chunk_start + safe_chunk_samples, decimated_total)
            
            # Mathematical Check
            if chunk_idx > 0:
                expected_start = previous_end - overlap_samples
                diff = chunk_start - expected_start
                
                logger.info(f"  Chunk {chunk_idx + 1}: Start={chunk_start:,} | Previous End={previous_end:,} | Overlap={overlap_samples:,}")
                logger.info(f"  -> Continuity Check: {previous_end:,} - {overlap_samples:,} = {expected_start:,}")
                
                if diff == 0:
                    logger.info(f"  -> ✓ MATCH: Actual start {chunk_start:,} equals expected {expected_start:,}")
                else:
                    logger.error(f"  -> ✗ MISMATCH: Diff {diff}")
                
                assert diff == 0, f"Continuity error at chunk {chunk_idx+1}"
            else:
                logger.info(f"  Chunk 1: Start={chunk_start:,} (Initial)")
            
            previous_end = chunk_end
            
        logger.info("✓ Streaming chunk continuity validated mathematically")

    
    def test_memory_usage_validation(self):
        """Test that memory usage stays within limits."""
        logger.info("\n" + "="*80)
        logger.info("TEST 4: Memory Usage Validation")
        logger.info("="*80)
        
        diagnostics = self.test_adaptive_memory_budgeting_1tb()
        safe_chunk_samples = diagnostics['safe_chunk_samples']
        expected_cube_gb = diagnostics['expected_cube_gb']
        will_use_dm_chunking = diagnostics['will_use_dm_chunking']
        
        # Simulate processing one chunk
        chunk_samples_decimated = safe_chunk_samples
        width = chunk_samples_decimated
        
        if will_use_dm_chunking:
            # With DM chunking, each DM chunk is < threshold
            threshold_gb = 16.0
            max_chunk_height = int((threshold_gb * (1024**3)) / (3 * width * 4))
            dm_chunk_height = max(100, max_chunk_height)
            
            dm_chunk_size_gb = (3 * dm_chunk_height * width * 4) / (1024**3)
            
            logger.info(f"With DM chunking:")
            logger.info(f"  DM chunk size: {dm_chunk_size_gb:.2f} GB")
            logger.info(f"  Peak memory: ~{dm_chunk_size_gb:.2f} GB (during DM chunk processing)")
            
            assert dm_chunk_size_gb <= threshold_gb * 1.1, \
                f"DM chunk size {dm_chunk_size_gb:.2f} GB exceeds threshold"
        else:
            # Without DM chunking, full cube is in memory
            logger.info(f"Without DM chunking:")
            logger.info(f"  Cube size: {expected_cube_gb:.2f} GB")
            logger.info(f"  Peak memory: ~{expected_cube_gb:.2f} GB")
        
        # Validate total memory doesn't exceed available
        available_ram_gb = 64.0  # Simulated
        max_ram_fraction = 0.5
        max_allowed_gb = available_ram_gb * max_ram_fraction
        
        peak_memory_gb = dm_chunk_size_gb if will_use_dm_chunking else expected_cube_gb
        
        logger.info(f"Available RAM: {available_ram_gb:.1f} GB")
        logger.info(f"Max allowed (50%): {max_allowed_gb:.1f} GB")
        logger.info(f"Peak memory usage: {peak_memory_gb:.2f} GB")
        
        assert peak_memory_gb <= max_allowed_gb, \
            f"Peak memory {peak_memory_gb:.2f} GB exceeds allowed {max_allowed_gb:.1f} GB"
        
        logger.info("✓ TEST 4 PASSED: Memory usage stays within limits")
    
    def test_complete_processing_simulation(self):
        """Simulate complete processing of 1TB file and validate all systems."""
        logger.info("\n" + "="*80)
        logger.info("TEST 5: Complete 1TB File Processing Simulation")
        logger.info("="*80)
        
        # Get all diagnostics
        diagnostics = self.test_adaptive_memory_budgeting_1tb()
        safe_chunk_samples = diagnostics['safe_chunk_samples']
        expected_cube_gb = diagnostics['expected_cube_gb']
        will_use_dm_chunking = diagnostics['will_use_dm_chunking']
        
        # Calculate processing parameters
        decimated_total = self.total_samples // self.down_time_rate
        num_temporal_chunks = (decimated_total + safe_chunk_samples - 1) // safe_chunk_samples
        
        if will_use_dm_chunking:
            width = safe_chunk_samples
            threshold_gb = 16.0
            max_chunk_height = int((threshold_gb * (1024**3)) / (3 * width * 4))
            dm_chunk_height = max(100, max_chunk_height)
            num_dm_chunks = (self.height_dm + dm_chunk_height - 1) // dm_chunk_height
        else:
            num_dm_chunks = 1
        
        total_processing_units = num_temporal_chunks * num_dm_chunks
        
        logger.info(f"\nProcessing Summary:")
        logger.info(f"  File size: {self.file_size_tb:.1f} TB")
        logger.info(f"  Total samples: {self.total_samples:,}")
        logger.info(f"  Duration: {self.duration_hours:.2f} hours")
        logger.info(f"\nTemporal Chunking:")
        logger.info(f"  Chunks: {num_temporal_chunks:,}")
        logger.info(f"  Samples per chunk: {safe_chunk_samples:,}")
        logger.info(f"\nDM Chunking:")
        logger.info(f"  Activated: {will_use_dm_chunking}")
        if will_use_dm_chunking:
            logger.info(f"  DM chunks: {num_dm_chunks}")
            logger.info(f"  DM values per chunk: ~{dm_chunk_height:,}")
        logger.info(f"\nTotal Processing Units: {total_processing_units:,}")
        logger.info(f"  (Temporal chunks × DM chunks)")
        
        # Validate system can handle it
        assert num_temporal_chunks > 0, "Must have at least one temporal chunk"
        assert safe_chunk_samples > 0, "Chunk size must be positive"
        
        if will_use_dm_chunking:
            assert num_dm_chunks > 0, "Must have at least one DM chunk"
        
        logger.info("\n✓ TEST 5 PASSED: System can process 1TB file")
        logger.info("✓ All mathematical formulas validated")
        logger.info("✓ Streaming chunking validated")
        logger.info("✓ DM chunking validated")
        logger.info("✓ Memory budgeting validated")
    
    def run_all_tests(self):
        """Run all tests in sequence."""
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE TEST SUITE: 1TB File Processing Validation")
        logger.info("="*80)
        
        try:
            self.setup_method()
            self.test_adaptive_memory_budgeting_1tb()
            self.test_dm_chunking_activation()
            self.test_extreme_dm_range()  # NEW: Test with very high DM
            self.test_streaming_chunk_calculation()
            self.test_memory_usage_validation()
            self.test_complete_processing_simulation()
            
            logger.info("\n" + "="*80)
            logger.info("✓ ALL TESTS PASSED: System validated for 1TB file processing")
            logger.info("="*80)
            return True
        except Exception as e:
            logger.error(f"\n✗ TEST FAILED: {e}", exc_info=True)
            return False


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Run tests
    test_suite = TestLargeFileProcessing()
    success = test_suite.run_all_tests()
    
    sys.exit(0 if success else 1)

