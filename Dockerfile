# ==============================================================================
# DRAFTS-UC Pipeline - Complete Dockerfile
# ==============================================================================
# Multi-stage build optimized for CPU and GPU
# Includes all necessary dependencies for FRB data processing
# ==============================================================================

# ==============================================================================
# Stage 1: Base CPU
# ==============================================================================
FROM python:3.12-slim AS base-cpu

LABEL maintainer="Sebastian Salgado Polanco"
LABEL description="DRAFTS-UC/DRAFTS++: Pipeline FRB"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install complete system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Compilers and build tools
    gcc \
    g++ \
    gfortran \
    make \
    # Math libraries
    libopenblas-dev \
    liblapack-dev \
    # HDF5 for astropy/fitsio
    libhdf5-dev \
    # CFITSIO for FITS file handling
    libcfitsio-dev \
    # OpenCV dependencies (headless)
    # libgl1, not libgl1-mesa-glx: that name was dropped in Debian trixie,
    # which is what python:3.12-slim resolves to now, and the CI build failed
    # on it with "has no installation candidate". libgl1 is the real provider
    # and exists on trixie and on the Ubuntu 22.04 base the GPU stage uses, so
    # this no longer depends on which distribution the base image tracks.
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # SSL and crypto
    libffi-dev \
    libssl-dev \
    # Utilities
    wget \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash draftsuser

# ==============================================================================
# Stage 2: Builder CPU - Python dependencies installation
# ==============================================================================
FROM base-cpu AS builder-cpu

WORKDIR /tmp

# The lockfile is the source of truth for versions (requirements.lock.txt is
# universal and carries the CUDA wheels, so the CPU image installs the same
# versions from the CPU index instead of the full lock).
COPY requirements.txt requirements.lock.txt ./

RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch==2.11.0 torchvision==0.26.0 \
 && pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Stage 3: Final CPU image
# ==============================================================================
FROM base-cpu AS cpu-final

# Copy installed Python packages
COPY --from=builder-cpu /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder-cpu /usr/local/bin /usr/local/bin

# Create directory structure
WORKDIR /app
RUN mkdir -p /app/Data/raw /app/Data/processed /app/Results /app/models /app/logs && \
    chown -R draftsuser:draftsuser /app

# Copy source code and configuration
COPY --chown=draftsuser:draftsuser src/ /app/src/
COPY --chown=draftsuser:draftsuser main.py /app/
COPY --chown=draftsuser:draftsuser config.yaml /app/
COPY --chown=draftsuser:draftsuser advanced-config/ /app/advanced-config/
COPY --chown=draftsuser:draftsuser README.md /app/

# Non-root user
USER draftsuser

# Default command
ENTRYPOINT ["python", "main.py"]
CMD []

# ==============================================================================
# Stage 4: Base GPU with CUDA
# ==============================================================================
# NOTE: torch 2.11 in requirements.lock.txt brings its own CUDA runtime through
# the nvidia-*-cu13 wheels, so this base supplies the driver interface rather
# than the toolkit. The 11.8 tag no longer matches the wheels and should be
# revisited on a machine with a GPU, which is why it is left explicit here
# instead of being changed untested.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base-gpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    # Compilers
    gcc \
    g++ \
    gfortran \
    make \
    # Math libraries
    libopenblas-dev \
    liblapack-dev \
    # HDF5
    libhdf5-dev \
    # CFITSIO
    libcfitsio-dev \
    # OpenCV (headless, no GUI)
    # libgl1, not libgl1-mesa-glx: that name was dropped in Debian trixie,
    # which is what python:3.12-slim resolves to now, and the CI build failed
    # on it with "has no installation candidate". libgl1 is the real provider
    # and exists on trixie and on the Ubuntu 22.04 base the GPU stage uses, so
    # this no longer depends on which distribution the base image tracks.
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # SSL
    libffi-dev \
    libssl-dev \
    # Utilities
    wget \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash draftsuser

# ==============================================================================
# Stage 5: Builder GPU - Python dependencies installation
# ==============================================================================
FROM base-gpu AS builder-gpu

WORKDIR /tmp

COPY requirements.lock.txt ./

# The GPU image installs the audited environment verbatim: every package pinned
# and hash-checked. This is what CI tests; the previous hand-written list was
# not (torch 2.1 vs 2.11, numpy 1.24 vs 2.4, fifteen packages unpinned).
RUN pip install --no-cache-dir --require-hashes -r requirements.lock.txt

# ==============================================================================
# Stage 6: Final GPU image
# ==============================================================================
FROM base-gpu AS gpu-final

# Copy installed Python packages
COPY --from=builder-gpu /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder-gpu /usr/local/bin /usr/local/bin

# Create directory structure
WORKDIR /app
RUN mkdir -p /app/Data/raw /app/Data/processed /app/Results /app/models /app/logs && \
    chown -R draftsuser:draftsuser /app

# Copy source code and configuration
COPY --chown=draftsuser:draftsuser src/ /app/src/
COPY --chown=draftsuser:draftsuser main.py /app/
COPY --chown=draftsuser:draftsuser config.yaml /app/
COPY --chown=draftsuser:draftsuser advanced-config/ /app/advanced-config/
COPY --chown=draftsuser:draftsuser README.md /app/

# Non-root user

USER draftsuser

# Default command
ENTRYPOINT ["python", "main.py"]
CMD []
