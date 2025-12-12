"""Phase metrics tracker for FRB pipeline - tracks candidates through each pipeline phase."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PhaseMetrics:
    """Metrics for a single pipeline phase."""
    detected: int = 0  # Number of candidates that entered this phase
    passed: int = 0    # Number of candidates that passed this phase
    failed: int = 0    # Number of candidates that failed this phase
    burst: int = 0     # Number of candidates classified as BURST in this phase
    no_burst: int = 0  # Number of candidates classified as NO_BURST in this phase
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "detected": self.detected,
            "passed": self.passed,
            "failed": self.failed,
            "burst": self.burst,
            "no_burst": self.no_burst,
        }


class PhaseMetricsTracker:
    """Track metrics for candidates through each phase of the pipeline."""
    
    def __init__(self):
        """Initialize phase metrics tracker."""
        # Phase 1: SNR Detection in Intensity (always enabled, all pass)
        self.phase_1: PhaseMetrics = PhaseMetrics()
        
        # Phase 2: SNR Validation in Linear Polarization (conditional)
        self.phase_2: PhaseMetrics = PhaseMetrics()
        
        # Phase 3a: ResNet Classification in Intensity (conditional)
        self.phase_3a_intensity: PhaseMetrics = PhaseMetrics()
        
        # Phase 3b: ResNet Classification in Linear Polarization (conditional)
        self.phase_3b_linear: PhaseMetrics = PhaseMetrics()
        
        # Final classification counts (after all phases)
        self.total_candidates: int = 0
        self.total_burst: int = 0
        self.total_no_burst: int = 0
    
    def record_phase_1(self, num_detected: int) -> None:
        """Record Phase 1 metrics (SNR Detection in Intensity).
        
        All candidates detected in Phase 1 are considered to have "passed"
        since Phase 1 is the entry point.
        """
        self.phase_1.detected += num_detected
        self.phase_1.passed += num_detected
        # No failed in Phase 1 (it's the detection phase)
        self.phase_1.failed = 0
        # Phase 1 doesn't classify, so no burst/no_burst counts
    
    def record_phase_2(self, num_entered: int, num_passed: int, 
                       num_burst: int = 0, num_no_burst: int = 0) -> None:
        """Record Phase 2 metrics (SNR Validation in Linear Polarization).
        
        Args:
            num_entered: Number of candidates that entered Phase 2
            num_passed: Number of candidates that passed Phase 2 (SNR_L >= threshold)
            num_burst: Number classified as BURST (usually 0 for Phase 2, it's just SNR validation)
            num_no_burst: Number classified as NO_BURST (usually 0 for Phase 2)
        """
        self.phase_2.detected += num_entered
        self.phase_2.passed += num_passed
        self.phase_2.failed += (num_entered - num_passed)
        self.phase_2.burst += num_burst
        self.phase_2.no_burst += num_no_burst
    
    def record_phase_3a(self, num_entered: int, num_passed: int, 
                        num_burst: int, num_no_burst: int) -> None:
        """Record Phase 3a metrics (ResNet Classification in Intensity).
        
        Args:
            num_entered: Number of candidates that entered Phase 3a
            num_passed: Number that passed (is_burst_intensity == True)
            num_burst: Number classified as BURST in Intensity
            num_no_burst: Number classified as NO_BURST in Intensity
        """
        self.phase_3a_intensity.detected += num_entered
        self.phase_3a_intensity.passed += num_passed
        self.phase_3a_intensity.failed += (num_entered - num_passed)
        self.phase_3a_intensity.burst += num_burst
        self.phase_3a_intensity.no_burst += num_no_burst
    
    def record_phase_3b(self, num_entered: int, num_passed: int,
                        num_burst: int, num_no_burst: int) -> None:
        """Record Phase 3b metrics (ResNet Classification in Linear Polarization).
        
        Args:
            num_entered: Number of candidates that entered Phase 3b
            num_passed: Number that passed (is_burst_linear == True)
            num_burst: Number classified as BURST in Linear
            num_no_burst: Number classified as NO_BURST in Linear
        """
        self.phase_3b_linear.detected += num_entered
        self.phase_3b_linear.passed += num_passed
        self.phase_3b_linear.failed += (num_entered - num_passed)
        self.phase_3b_linear.burst += num_burst
        self.phase_3b_linear.no_burst += num_no_burst
    
    def record_final_classification(self, num_total: int, num_burst: int, num_no_burst: int) -> None:
        """Record final classification counts.
        
        Args:
            num_total: Total number of candidates processed
            num_burst: Total number classified as BURST (final)
            num_no_burst: Total number classified as NO_BURST (final)
        """
        self.total_candidates += num_total
        self.total_burst += num_burst
        self.total_no_burst += num_no_burst
    
    def merge(self, other: PhaseMetricsTracker) -> None:
        """Merge metrics from another tracker (accumulate across slices/chunks)."""
        # Phase 1
        self.phase_1.detected += other.phase_1.detected
        self.phase_1.passed += other.phase_1.passed
        self.phase_1.failed += other.phase_1.failed
        self.phase_1.burst += other.phase_1.burst
        self.phase_1.no_burst += other.phase_1.no_burst
        
        # Phase 2
        self.phase_2.detected += other.phase_2.detected
        self.phase_2.passed += other.phase_2.passed
        self.phase_2.failed += other.phase_2.failed
        self.phase_2.burst += other.phase_2.burst
        self.phase_2.no_burst += other.phase_2.no_burst
        
        # Phase 3a
        self.phase_3a_intensity.detected += other.phase_3a_intensity.detected
        self.phase_3a_intensity.passed += other.phase_3a_intensity.passed
        self.phase_3a_intensity.failed += other.phase_3a_intensity.failed
        self.phase_3a_intensity.burst += other.phase_3a_intensity.burst
        self.phase_3a_intensity.no_burst += other.phase_3a_intensity.no_burst
        
        # Phase 3b
        self.phase_3b_linear.detected += other.phase_3b_linear.detected
        self.phase_3b_linear.passed += other.phase_3b_linear.passed
        self.phase_3b_linear.failed += other.phase_3b_linear.failed
        self.phase_3b_linear.burst += other.phase_3b_linear.burst
        self.phase_3b_linear.no_burst += other.phase_3b_linear.no_burst
        
        # Final counts
        self.total_candidates += other.total_candidates
        self.total_burst += other.total_burst
        self.total_no_burst += other.total_no_burst
    
    def to_dict(self) -> Dict:
        """Convert all metrics to dictionary for JSON serialization."""
        return {
            "total_candidates": self.total_candidates,
            "total_burst": self.total_burst,
            "total_no_burst": self.total_no_burst,
            "by_phase": {
                "phase_1": self.phase_1.to_dict(),
                "phase_2": self.phase_2.to_dict(),
                "phase_3a_intensity": self.phase_3a_intensity.to_dict(),
                "phase_3b_linear": self.phase_3b_linear.to_dict(),
            }
        }
    
    def reset(self) -> None:
        """Reset all metrics (for new file processing)."""
        self.phase_1 = PhaseMetrics()
        self.phase_2 = PhaseMetrics()
        self.phase_3a_intensity = PhaseMetrics()
        self.phase_3b_linear = PhaseMetrics()
        self.total_candidates = 0
        self.total_burst = 0
        self.total_no_burst = 0
