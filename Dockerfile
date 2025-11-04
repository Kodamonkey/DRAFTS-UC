# ==============================================================================
# DRAFTS-UC Pipeline - Dockerfile Completo
# ==============================================================================
# Multi-stage build optimizado para CPU y GPU
# Incluye todas las dependencias necesarias para procesamiento de datos FRB
# ==============================================================================

# ==============================================================================
# Stage 1: Base CPU
# ==============================================================================
FROM python:3.10-slim as base-cpu

LABEL maintainer="Sebastian Salgado Polanco"
LABEL description="DRAFTS-UC: Pipeline de Detección de FRB"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema completas
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Compiladores y build tools
    gcc \
    g++ \
    gfortran \
    make \
    # Librerías matemáticas
    libopenblas-dev \
    liblapack-dev \
    # HDF5 para astropy/fitsio
    libhdf5-dev \
    # CFITSIO para manejo de archivos FITS
    libcfitsio-dev \
    # OpenCV dependencias (sin GUI)
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # SSL y crypto
    libffi-dev \
    libssl-dev \
    # Utilidades
    wget \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root
RUN useradd -m -u 1000 -s /bin/bash draftsuser

# ==============================================================================
# Stage 2: Builder CPU - Instalación de dependencias Python
# ==============================================================================
FROM base-cpu as builder-cpu

WORKDIR /tmp

# Copiar requirements
COPY requirements.txt .

# Instalar PyTorch CPU + todas las dependencias
# Orden optimizado: torch primero, luego el resto
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
    seaborn \
    scikit-image \
    scikit-learn \
    timm \
    tqdm \
    your \
    blimpy \
    jplephem

# ==============================================================================
# Stage 3: Imagen final CPU
# ==============================================================================
FROM base-cpu as cpu-final

# Copiar Python packages instalados
COPY --from=builder-cpu /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder-cpu /usr/local/bin /usr/local/bin

# Crear estructura de directorios
WORKDIR /app
RUN mkdir -p /app/Data/raw /app/Data/processed /app/Results /app/models /app/logs && \
    chown -R draftsuser:draftsuser /app

# Copiar código fuente
COPY --chown=draftsuser:draftsuser src/ /app/src/
COPY --chown=draftsuser:draftsuser main.py /app/
COPY --chown=draftsuser:draftsuser README.md /app/

# Usuario no-root
USER draftsuser

# Entrypoint y comando
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]

# ==============================================================================
# Stage 4: Base GPU con CUDA
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

# Instalar Python 3.10 y dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    # Compiladores
    gcc \
    g++ \
    gfortran \
    make \
    # Librerías matemáticas
    libopenblas-dev \
    liblapack-dev \
    # HDF5
    libhdf5-dev \
    # CFITSIO
    libcfitsio-dev \
    # OpenCV (headless, sin GUI)
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # SSL
    libffi-dev \
    libssl-dev \
    # Utilidades
    wget \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Crear enlaces simbólicos para python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Crear usuario no-root
RUN useradd -m -u 1000 -s /bin/bash draftsuser

# ==============================================================================
# Stage 5: Builder GPU - Instalación de dependencias Python
# ==============================================================================
FROM base-gpu as builder-gpu

WORKDIR /tmp

COPY requirements.txt .

# Instalar PyTorch con CUDA 11.8 + todas las dependencias
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
    seaborn \
    scikit-image \
    scikit-learn \
    timm \
    tqdm \
    your \
    blimpy \
    jplephem

# ==============================================================================
# Stage 6: Imagen final GPU
# ==============================================================================
FROM base-gpu as gpu-final

# Copiar Python packages instalados
COPY --from=builder-gpu /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder-gpu /usr/local/bin /usr/local/bin

# Crear estructura de directorios
WORKDIR /app
RUN mkdir -p /app/Data/raw /app/Data/processed /app/Results /app/models /app/logs && \
    chown -R draftsuser:draftsuser /app

# Copiar código fuente
COPY --chown=draftsuser:draftsuser src/ /app/src/
COPY --chown=draftsuser:draftsuser main.py /app/
COPY --chown=draftsuser:draftsuser README.md /app/

# Usuario no-root
USER draftsuser

# Entrypoint y comando
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
