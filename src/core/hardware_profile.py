"""Hardware detection and profiling for adaptive pipeline configuration.

Provides a single source of truth for system capabilities (CPU, RAM, GPU,
platform) so that every subsystem can derive its parameters from the same
profile instead of scattering ad-hoc psutil/torch calls across the codebase.
"""
from __future__ import annotations

import logging
import os
import platform
import shutil
from dataclasses import dataclass, field

import psutil

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareProfile:
    """Immutable snapshot of the host machine's capabilities."""

    # CPU
    cpu_cores_physical: int
    cpu_cores_logical: int
    cpu_arch: str  # e.g. "x86_64", "arm64"

    # RAM (bytes)
    ram_total_bytes: int
    ram_available_bytes: int

    # GPU
    gpu_available: bool
    gpu_backend: str  # "cuda", "mps", "cpu"
    gpu_name: str | None = None
    gpu_vram_total_bytes: int = 0
    gpu_vram_available_bytes: int = 0
    gpu_compute_capability: tuple[int, int] | None = None
    gpu_count: int = 0

    # Platform
    platform_system: str = ""  # "Linux", "Darwin", "Windows"
    platform_node: str = ""

    # Disk (results directory)
    disk_free_bytes: int = 0

    # --- Derived helpers ---------------------------------------------------

    @property
    def ram_total_gb(self) -> float:
        return self.ram_total_bytes / (1024 ** 3)

    @property
    def ram_available_gb(self) -> float:
        return self.ram_available_bytes / (1024 ** 3)

    @property
    def gpu_vram_total_gb(self) -> float:
        return self.gpu_vram_total_bytes / (1024 ** 3)

    @property
    def gpu_vram_available_gb(self) -> float:
        return self.gpu_vram_available_bytes / (1024 ** 3)

    @property
    def disk_free_gb(self) -> float:
        return self.disk_free_bytes / (1024 ** 3)

    @property
    def recommended_threads(self) -> int:
        """Number of threads to use for parallel workloads (Numba, OMP, MKL)."""
        return min(self.cpu_cores_physical, 8)

    def usable_ram_bytes(self, fraction: float = 0.25, overhead: float = 1.3) -> int:
        """RAM budget for pipeline data, after safety margins."""
        return int(self.ram_available_bytes * fraction / overhead)

    def usable_vram_bytes(self, fraction: float = 0.70) -> int:
        """GPU VRAM budget for pipeline data."""
        return int(self.gpu_vram_available_bytes * fraction)

    def summary(self) -> str:
        parts = [
            f"CPU: {self.cpu_cores_physical}p/{self.cpu_cores_logical}l cores ({self.cpu_arch})",
            f"RAM: {self.ram_available_gb:.1f}/{self.ram_total_gb:.1f} GB available",
        ]
        if self.gpu_available:
            parts.append(
                f"GPU: {self.gpu_name} ({self.gpu_vram_available_gb:.1f}/"
                f"{self.gpu_vram_total_gb:.1f} GB VRAM, {self.gpu_backend})"
            )
        else:
            parts.append("GPU: none (CPU-only mode)")
        parts.append(f"Disk free: {self.disk_free_gb:.1f} GB")
        parts.append(f"Platform: {self.platform_system} ({self.platform_node})")
        return " | ".join(parts)


def detect_hardware(results_dir: str | None = None) -> HardwareProfile:
    """Probe the current machine and return an immutable *HardwareProfile*.

    Parameters
    ----------
    results_dir : str, optional
        Path used to measure free disk space.  Falls back to the current
        working directory when *None*.
    """
    # -- CPU ----------------------------------------------------------------
    phys = psutil.cpu_count(logical=False) or 1
    logi = psutil.cpu_count(logical=True) or phys
    arch = platform.machine() or "unknown"

    # -- RAM ----------------------------------------------------------------
    vm = psutil.virtual_memory()
    ram_total = vm.total
    ram_avail = vm.available

    # -- GPU ----------------------------------------------------------------
    gpu_available = False
    gpu_backend = "cpu"
    gpu_name: str | None = None
    gpu_vram_total = 0
    gpu_vram_avail = 0
    gpu_cc: tuple[int, int] | None = None
    gpu_count = 0

    try:
        import torch

        if torch.cuda.is_available():
            gpu_available = True
            gpu_backend = "cuda"
            gpu_count = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            gpu_vram_total = props.total_memory
            gpu_vram_avail = gpu_vram_total - torch.cuda.memory_reserved(0)
            gpu_cc = (props.major, props.minor)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_available = True
            gpu_backend = "mps"
            gpu_name = "Apple Silicon (MPS)"
            gpu_count = 1
            # MPS doesn't expose VRAM — assume shared memory
    except ImportError:
        pass

    # -- Platform -----------------------------------------------------------
    sys_name = platform.system()  # "Linux", "Darwin", "Windows"
    node_name = platform.node()

    # -- Disk ---------------------------------------------------------------
    disk_path = results_dir or os.getcwd()
    try:
        usage = shutil.disk_usage(disk_path)
        disk_free = usage.free
    except OSError:
        disk_free = 0

    profile = HardwareProfile(
        cpu_cores_physical=phys,
        cpu_cores_logical=logi,
        cpu_arch=arch,
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_avail,
        gpu_available=gpu_available,
        gpu_backend=gpu_backend,
        gpu_name=gpu_name,
        gpu_vram_total_bytes=gpu_vram_total,
        gpu_vram_available_bytes=gpu_vram_avail,
        gpu_compute_capability=gpu_cc,
        gpu_count=gpu_count,
        platform_system=sys_name,
        platform_node=node_name,
        disk_free_bytes=disk_free,
    )

    logger.info("Hardware profile: %s", profile.summary())
    return profile


def apply_thread_settings(profile: HardwareProfile, user_threads: int = 0) -> int:
    """Configure thread counts for Numba, OMP and MKL from the profile.

    Parameters
    ----------
    user_threads : int
        Override from ``CPU_THREADS`` config.  ``0`` means auto-detect.

    Returns
    -------
    int
        The thread count that was applied.
    """
    threads = user_threads if user_threads > 0 else profile.recommended_threads

    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["NUMBA_NUM_THREADS"] = str(threads)

    try:
        import numba
        numba.set_num_threads(threads)
    except (ImportError, RuntimeError):
        pass

    logger.info("Thread counts set to %d (physical=%d)", threads, profile.cpu_cores_physical)
    return threads
