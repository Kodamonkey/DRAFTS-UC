# ==============================================================================
# DRAFTS-UC Pipeline - Complete Dockerfile
# ==============================================================================
# Multi-stage build optimized for CPU and GPU
# Includes all necessary dependencies for FRB data processing
# ==============================================================================

# ==============================================================================
# Stage 1: Base CPU
# ==============================================================================
FROM python:3.10-slim as base-cpu

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
    libgl1-mesa-glx \
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
FROM base-cpu as builder-cpu

WORKDIR /tmp

# Copy requirements
COPY requirements.txt .

# Install PyTorch CPU + all dependencies
# Optimized order: torch first, then the rest
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    numpy==1.24.3 \
    numba==0.58.0 \
    scipy \
    astropy \
    fitsio \
    matplotlib \
    opencv-python-headless \
    pandas \
    psutil \
    pyyaml \
    seaborn \
    scikit-image \
    scikit-learn \
    timm \
    tqdm \
    your \
    blimpy \
    jplephem

# ==============================================================================
# Stage 3: Final CPU image
# ==============================================================================
FROM base-cpu as cpu-final

# Copy installed Python packages
COPY --from=builder-cpu /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder-cpu /usr/local/bin /usr/local/bin

# Create directory structure
WORKDIR /app
RUN mkdir -p /app/Data/raw /app/Data/processed /app/Results /app/models /app/logs && \
    chown -R draftsuser:draftsuser /app

# Copy source code and configuration
COPY --chown=draftsuser:draftsuser src/ /app/src/
COPY --chown=draftsuser:draftsuser main.py /app/
COPY --chown=draftsuser:draftsuser config.yaml /app/
COPY --chown=draftsuser:draftsuser README.md /app/

# Non-root user
USER draftsuser

# Default command
CMD ["python", "main.py"]

# ==============================================================================
# Stage 4: Base GPU with CUDA
# ==============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base-gpu

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
    libgl1-mesa-glx \
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
FROM base-gpu as builder-gpu

WORKDIR /tmp

COPY requirements.txt .

# Install PyTorch with CUDA 11.8 + all dependencies
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir \
    numpy==1.24.3 \
    numba==0.58.0 \
    scipy \
    astropy \
    fitsio \
    matplotlib \
    opencv-python-headless \
    pandas \
    psutil \
    pyyaml \
    seaborn \
    scikit-image \
    scikit-learn \
    timm \
    tqdm \
    your \
    blimpy \
    jplephem

# ==============================================================================
# Stage 6: Final GPU image
# ==============================================================================
FROM base-gpu as gpu-final

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
COPY --chown=draftsuser:draftsuser README.md /app/

# Non-root user

USER draftsuser

# Default command
CMD ["python", "main.py"]
