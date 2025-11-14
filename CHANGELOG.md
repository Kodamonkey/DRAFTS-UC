# Changelog

All notable changes to DRAFTS++ will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-14

### Added

- Initial release of DRAFTS++ enhanced pipeline
- Two-stage deep learning approach: CenterNet (detection) + ResNet (classification)
- YAML-based configuration system (`config.yaml`)
- Docker support for both GPU and CPU environments
- Smart chunking system for handling large files with automatic memory management
- Comprehensive logging system with timestamped, color-coded console output
- CLI flexibility with optional command-line parameter overrides
- Multi-polarization support for high-frequency observations (≥8 GHz)
- High-frequency pipeline with IQUV analysis and linear polarization validation
- Graceful error handling with fallbacks (GPU→CPU, Torch→Numba→CPU)
- Support for both `.fits` and `.fil` file formats
- Automatic frequency ordering detection and correction
- Temporal slice generation with configurable duration
- DM (Dispersion Measure) search with configurable range
- Candidate filtering with configurable probability thresholds
- Output modes: strict (BURST only) and permissive (all detections)
- CSV output with candidate properties (DM, time, SNR, classification probability)
- Dynamic spectrum plots with detected candidates highlighted
- Multi-band analysis option (optional, ~3x slower)

### Technical Details

- Built on PyTorch 2.0+ with CUDA support
- Python 3.8+ compatibility
- Based on original DRAFTS by Zhang et al. (arXiv:2410.03200)
