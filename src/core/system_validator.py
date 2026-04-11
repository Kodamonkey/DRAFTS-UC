"""Startup validation of system requirements before pipeline execution.

Consolidates hardware checks, model file verification, and configuration
sanity tests that were previously scattered across config.py.  Called once
from ``run_pipeline()`` so that problems surface early with clear messages
instead of cryptic mid-run crashes.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .hardware_profile import HardwareProfile

logger = logging.getLogger(__name__)

# Minimum requirements
_MIN_RAM_GB = 4.0
_MIN_DISK_FREE_GB = 1.0
_MIN_GPU_VRAM_GB = 2.0


class SystemRequirements:
    """Static helper that validates whether the host can run the pipeline."""

    @staticmethod
    def validate(profile: HardwareProfile, config) -> list[str]:
        """Run all pre-flight checks and return a list of warning/error strings.

        The caller decides whether to abort or continue.  An empty list means
        all checks passed.

        Parameters
        ----------
        profile : HardwareProfile
            Snapshot of the current machine (from ``detect_hardware()``).
        config : module
            The ``src.config.config`` module (used to read MODEL_PATH, etc.).
        """
        issues: list[str] = []

        # -- RAM ---------------------------------------------------------------
        if profile.ram_available_gb < _MIN_RAM_GB:
            issues.append(
                f"Available RAM is {profile.ram_available_gb:.1f} GB "
                f"(minimum recommended: {_MIN_RAM_GB:.0f} GB). "
                f"Processing may fail on large files."
            )

        # -- Disk space --------------------------------------------------------
        if profile.disk_free_gb < _MIN_DISK_FREE_GB:
            issues.append(
                f"Free disk space is {profile.disk_free_gb:.1f} GB "
                f"(minimum recommended: {_MIN_DISK_FREE_GB:.0f} GB). "
                f"Results may fail to write."
            )

        # -- GPU ---------------------------------------------------------------
        if profile.gpu_available and profile.gpu_backend == "cuda":
            if profile.gpu_vram_total_gb < _MIN_GPU_VRAM_GB:
                issues.append(
                    f"GPU VRAM is {profile.gpu_vram_total_gb:.1f} GB "
                    f"(minimum recommended: {_MIN_GPU_VRAM_GB:.0f} GB). "
                    f"Consider using CPU mode."
                )
            # Validate CUDA actually works
            try:
                import torch
                t = torch.zeros(1, device="cuda")
                del t
            except Exception as exc:
                issues.append(
                    f"CUDA is reported available but failed a smoke test: {exc}. "
                    f"Pipeline will fall back to CPU."
                )

        # -- Model files -------------------------------------------------------
        model_path = getattr(config, "MODEL_PATH", None)
        class_model_path = getattr(config, "CLASS_MODEL_PATH", None)

        if model_path and not Path(model_path).exists():
            issues.append(
                f"Detection model not found: {model_path}. "
                f"Ensure cent_resnet18.pth is in src/models/."
            )
        if class_model_path and not Path(class_model_path).exists():
            issues.append(
                f"Classification model not found: {class_model_path}. "
                f"Ensure class_resnet18.pth is in src/models/."
            )

        # -- Data parameters (only after extraction) ---------------------------
        freq_reso = getattr(config, "FREQ_RESO", 0)
        time_reso = getattr(config, "TIME_RESO", 0.0)
        file_leng = getattr(config, "FILE_LENG", 0)

        if freq_reso > 0 and time_reso > 0 and file_leng > 0:
            # Estimate peak memory for the configured chunk
            _check_memory_budget(profile, config, issues)

        # -- Slice parameters --------------------------------------------------
        slice_len = getattr(config, "SLICE_LEN", 512)
        slice_min = getattr(config, "SLICE_LEN_MIN", 32)
        slice_max = getattr(config, "SLICE_LEN_MAX", 2048)
        if not (slice_min <= slice_len <= slice_max):
            issues.append(
                f"SLICE_LEN={slice_len} outside [{slice_min}, {slice_max}]. "
                f"Adjust SLICE_DURATION_MS in config.yaml."
            )

        # -- DM ranges ---------------------------------------------------------
        dm_min_w = getattr(config, "DM_RANGE_MIN_WIDTH", 80.0)
        dm_max_w = getattr(config, "DM_RANGE_MAX_WIDTH", 300.0)
        if dm_min_w <= 0 or dm_max_w <= 0:
            issues.append(f"DM range widths must be > 0 (got min={dm_min_w}, max={dm_max_w}).")
        if dm_min_w >= dm_max_w:
            issues.append(f"DM_RANGE_MIN_WIDTH ({dm_min_w}) must be < DM_RANGE_MAX_WIDTH ({dm_max_w}).")

        return issues


def _check_memory_budget(
    profile: HardwareProfile,
    config,
    issues: list[str],
) -> None:
    """Estimate peak memory for the current configuration and warn if it
    exceeds available RAM."""
    import numpy as np

    freq_reso = int(getattr(config, "FREQ_RESO", 0))
    down_freq = max(1, int(getattr(config, "DOWN_FREQ_RATE", 1)))
    down_time = max(1, int(getattr(config, "DOWN_TIME_RATE", 8)))
    dm_min = float(getattr(config, "DM_min", 0))
    dm_max = float(getattr(config, "DM_max", 1024))
    time_reso = float(getattr(config, "TIME_RESO", 0.0))
    max_chunk = int(getattr(config, "MAX_CHUNK_SAMPLES", 1_000_000))
    prewhiten = bool(getattr(config, "PREWHITEN_BEFORE_DM", True))

    if freq_reso <= 0 or time_reso <= 0:
        return

    nchan_ds = freq_reso // down_freq
    # DM height (number of DM trial values)
    height_dm = max(1, int(np.ceil((dm_max - dm_min) / max(1.0, time_reso * down_time * 1e3 / 4.15))))
    height_dm = max(height_dm, 100)  # minimum

    # Decimated chunk width
    chunk_ds = max_chunk // down_time

    # Per-sample costs (decimated domain)
    cost_dm_cube = 3 * height_dm * 4  # float32
    cost_raw_block = nchan_ds * 4
    cost_ds_peak = nchan_ds * 4 * 2  # downsampling intermediates
    cost_prewhiten = nchan_ds * 4 * 2 if prewhiten else 0
    cost_total = cost_dm_cube + cost_raw_block + cost_ds_peak + cost_prewhiten

    peak_bytes = cost_total * chunk_ds
    peak_gb = peak_bytes / (1024 ** 3)
    avail_gb = profile.ram_available_gb

    if peak_gb > avail_gb * 0.8:
        issues.append(
            f"Estimated peak memory ({peak_gb:.1f} GB) may exceed available RAM "
            f"({avail_gb:.1f} GB) with current config (chunk={max_chunk:,}, "
            f"DM={dm_min:.0f}-{dm_max:.0f}, channels={freq_reso}). "
            f"Reduce max_chunk_samples or DM range in config.yaml."
        )
