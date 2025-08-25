# 🚀 DRAFTS-UC Pipeline

<div align="center">

✨ **FRB Detection Pipeline based on DRAFTS for UC** ✨

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

[Description](#description) •
[Installation](#installation) •
[Configuration](#configuration) •
[Usage](#usage) •
[Models](#models) •
[Features](#features) •
[Contributing](#contributing)

</div>

![DRAFTS WorkFlow](./WorkFlow.png)

## 📖 Description

**DRAFTS-UC** is an FRB (Fast Radio Bursts) detection pipeline based on the original **DRAFTS** project (Deep learning-based RAdio Fast Transient Search), but adapted and optimized for specific use.

### 🎯 What is DRAFTS?

**DRAFTS** is a deep learning-based radio fast transient search pipeline designed to address limitations in traditional single-pulse search techniques like Presto and Heimdall.

### 🔄 How does this project relate?

This project is a **specialized fork** of DRAFTS that:

- ✅ **Uses** pre-trained models and neural network architectures from DRAFTS
- ✅ **Maintains** core detection and classification functionality
- ✅ **Adapts** the pipeline for specific use at UC
- ✅ **Optimizes** processing for local data
- ✅ **Simplifies** configuration and execution

### 🚀 Key Advantages

- **Written entirely in Python** for easy cross-platform installation
- **CUDA acceleration** for faster dedispersion
- **Pre-trained models** ready to use
- **Improved detection** compared to traditional methods
- **Binary classification** with >99% accuracy on FAST and GBT data
- **Significant reduction** in manual verification required

## 🆕 Special Features

### 🎯 Dispersion Correction for Waterfall Plots

**New functionality!** The pipeline now includes automatic dispersion correction that allows visualization of the **natural burst parabola** instead of the time offset introduced by interstellar dispersion.

#### ✨ Benefits

- **Visualization of the natural parabola** of the burst
- **Better analysis** of intrinsic morphology
- **Correct temporal alignment** of burst start
- **Facilitates identification** of burst characteristics vs. propagation effects

#### 🔧 Usage

```python
# With correction to show natural parabola
save_waterfall_dispersed_plot(
    waterfall_block=data,
    out_path=Path("output/waterfall.png"),
    # ... other parameters
    dm_value=100.0,  # 🆕 DM value for correction
)
```

### 📊 Automatic Parameter System

**Simplified configuration!** You only need to configure `SLICE_DURATION_MS` in `user_config.py` and the system automatically calculates all other optimized parameters.

#### 🚀 Basic Usage

```python
# In user_config.py - Just this:
SLICE_DURATION_MS: float = 64.0  # Desired duration of each slice in ms

# Execution:
python main.py  # Automatic mode by default
```

#### 📊 Typical Configurations

| Use Case     | SLICE_DURATION_MS | Description                                 |
| ------------ | ----------------- | ------------------------------------------- |
| Fast FRBs    | 32.0 ms           | Short slices for very fast pulses           |
| General FRBs | 64.0 ms           | Balance between sensitivity and speed       |
| Long pulses  | 128.0 ms          | Longer slices for extended pulses           |
| Weak signals | 256.0 ms          | Greater temporal integration for better SNR |

### 🎨 Signal-to-Noise Ratio (SNR) Analysis

The pipeline includes comprehensive SNR analysis capabilities that enhance FRB candidate detection and visualization:

#### ✨ Key Features

- **Automatic SNR calculation**: All candidate patches show temporal profiles in SNR units (σ)
- **Robust noise estimation**: Uses Interquartile Range (IQR) method robust to multiple pulses
- **Configurable thresholds**: Set `SNR_THRESH` in `user_config.py` to highlight significant detections
- **Enhanced visualizations**: SNR profiles with peak annotations, threshold lines, color-coded regions

## 🛠️ Installation

### Prerequisites

- **Python 3.8 or higher** (recommended Python 3.8+)
- **CUDA 11.0+** (for GPU acceleration)
- **Git** to clone the repository
- **Virtual environment** recommended (`venv` or `conda`)

### 🔧 Installation Steps

1. **Clone the Repository**

   ```bash
   git clone <REPOSITORY_URL>
   cd DRAFTS-UC
   ```

2. **Create and Activate Virtual Environment**

   ```bash
   # Using venv (recommended)
   python -m venv venv_drafts

   # On Windows:
   venv_drafts\Scripts\activate

   # On Linux/Mac:
   source venv_drafts/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**

   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## ⚙️ Configuration

### 📁 Main Configuration File

The file `src/config/user_config.py` contains all customizable configurations:

```python
# =============================================================================
# DATA AND FILE CONFIGURATION
# =============================================================================
DATA_DIR = Path("./Data/raw")              # Directory with .fits/.fil files
RESULTS_DIR = Path("./Results")            # Directory to save results

# List of files to process
FRB_TARGETS = [
   "3096_0001_00_8bit"                    # Specific files to analyze
]

# =============================================================================
# TEMPORAL ANALYSIS CONFIGURATION
# =============================================================================
SLICE_DURATION_MS: float = 300.0           # Duration of each slice in ms

# =============================================================================
# DISPERSION (DM) CONFIGURATION
# =============================================================================
DM_min: int = 0                            # Minimum DM in pc cm⁻³
DM_max: int = 1024                         # Maximum DM in pc cm⁻³

# =============================================================================
# DETECTION THRESHOLDS
# =============================================================================
DET_PROB: float = 0.3                      # Minimum probability for detection
CLASS_PROB: float = 0.5                    # Minimum probability for classification
SNR_THRESH: float = 3.0                    # SNR threshold for visualizations
```

### 🔧 Advanced Configurations

#### Multi-Band Analysis

```python
# Multi-band analysis (Full/Low/High)
USE_MULTI_BAND: bool = False               # True = use multi-band analysis
```

#### Processing Optimization

```python
# Reduction factors to optimize processing
DOWN_FREQ_RATE: int = 1                    # Frequency reduction factor
DOWN_TIME_RATE: int = 8                    # Time reduction factor
```

#### Debug and Logging

```python
# Frequency and file debug
DEBUG_FREQUENCY_ORDER: bool = False        # Detailed frequency information
FORCE_PLOTS: bool = False                  # Generate plots even without candidates
```

## 🚀 Usage

### 🎯 Basic Execution

1. **Configure parameters** in `src/config/user_config.py`
2. **Place data files** in `Data/raw/`
3. **Run the pipeline**:

   ```bash
   python main.py
   ```

### 📊 Pipeline Flow

The pipeline automatically executes:

1. **Preprocessing** - Loads and prepares FITS/FIL files
2. **Dedispersion** - Applies dispersion correction with CUDA acceleration
3. **Detection** - Uses object detection model to identify candidates
4. **Classification** - Verifies authenticity with binary classification model
5. **Visualization** - Generates plots and detailed reports
6. **Results** - Saves candidates and metrics in `Results/`

### 🔍 Specific Use Cases

#### Single File Analysis

```bash
# Process specific file
python main.py
```

#### Specific Plot Scripts

```bash
# Generate absolute segment plots
python src/scripts/absolute_segment_plots.py --filename file.fits --start 10.0 --duration 5.0 --dm 100.0
```

#### Individual Plot Generation

```python
from src.visualization.plot_individual_components import generate_individual_plots

# Generate individual plots for each component
generate_individual_plots(
    waterfall_block=data,
    dedispersed_block=dedisp_data,
    # ... other parameters
    dm_val=100.0,  # DM for dispersion correction
)
```

## 🧠 Models

### 📥 Pre-trained Models Download

Pre-trained models are available in the `models/` directory:

```
models/
├── cent_resnet18.pth      # ResNet18 detection model
├── cent_resnet50.pth      # ResNet50 detection model
├── class_resnet18.pth     # ResNet18 classification model
├── class_resnet50.pth     # ResNet50 classification model
└── README.md              # Model information
```

### 🎯 Model Types

#### 1. Object Detection Models (CenterNet)

- **`cent_resnet18.pth`**: Fast detection with ResNet18 backbone
- **`cent_resnet50.pth`**: Precise detection with ResNet50 backbone

#### 2. Binary Classification Models

- **`class_resnet18.pth`**: Fast classification with ResNet18
- **`class_resnet50.pth`**: Precise classification with ResNet50

### 🔧 Model Training (Optional)

If you want to train your own models:

#### Detection Training

```bash
cd src/models/ObjectDet/
python centernet_train.py resnet18  # Use 'resnet50' for ResNet50
```

#### Classification Training

```bash
cd src/models/BinaryClass/
python binary_train.py resnet18 BinaryNet  # Use 'resnet50' for ResNet50
```

## 📁 Project Structure

```
DRAFTS-UC/
├── 📁 Data/                           # Input and output data
│   ├── raw/                           # Original .fits/.fil files
│   └── processed/                     # Processed data (auto-generated)
├── 📁 models/                         # Pre-trained models
│   ├── cent_resnet18.pth             # ResNet18 detection model
│   ├── cent_resnet50.pth             # ResNet50 detection model
│   ├── class_resnet18.pth            # ResNet18 classification model
│   └── class_resnet50.pth            # ResNet50 classification model
├── 📁 src/                           # Pipeline source code
│   ├── 📁 analysis/                  # Analysis utilities (SNR, etc.)
│   ├── 📁 config/                    # System configurations
│   ├── 📁 core/                      # Main pipeline logic
│   ├── 📁 detection/                 # Model interfaces
│   ├── 📁 input/                     # Data loading and processing
│   ├── 📁 logging/                   # Centralized logging system
│   ├── 📁 models/                    # Model training code
│   ├── 📁 output/                    # Results and candidate management
│   ├── 📁 preprocessing/             # Preprocessing and dedispersion
│   ├── 📁 scripts/                   # Utility scripts
│   └── 📁 visualization/             # Plot and visualization generation
├── 📁 Results/                       # Pipeline results (auto-generated)
├── 📁 venv_drafts/                   # Virtual environment (auto-generated)
├── 📄 main.py                        # Main entry point
├── 📄 requirements.txt                # Python dependencies
├── 📄 README.md                       # This file
└── 📄 WorkFlow.png                    # Workflow diagram
```

## 🧪 Testing and Validation

### ✅ Verify Functionality

```bash
# Verify all dependencies are installed
python -c "import torch, numpy, matplotlib, astropy; print('✅ All dependencies are available')"

# Verify GPU access (if available)
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
```

### 🔍 Logs and Debug

The system includes detailed logging:

- **Automatic logs** in `src/logging/log/`
- **Logging configuration** in `src/logging/logging_config.py`
- **Configurable levels**: DEBUG, INFO, WARNING, ERROR

### 📊 Test Plot Generation

```bash
# Generate plots even without candidates (debug mode)
# In user_config.py: FORCE_PLOTS = True
python main.py
```

## 🚀 Optimization and Performance

### 💾 Memory Management

- **Automatic chunking** for large files
- **Memory optimization** based on available hardware
- **Automatic configuration** of chunking parameters

### ⚡ GPU Acceleration

- **CUDA dedispersion** for fast processing
- **PyTorch models** optimized for GPU
- **Automatic fallback** to CPU if GPU unavailable

### 📈 Scalability

- **Parallel processing** when possible
- **Automatic parameter optimization**
- **Efficient management** of large files

## 🔧 Troubleshooting

### ❌ Common Problems

#### 1. CUDA Error

```bash
# Verify CUDA installation
nvidia-smi
python -c "import torch; print(torch.version.cuda)"
```

#### 2. Missing Dependencies

```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

#### 3. Memory Problems

```bash
# Reduce chunk size in user_config.py
SLICE_DURATION_MS = 64.0  # Reduce from 300.0 to 64.0
```

#### 4. Files Not Found

```bash
# Verify directory structure
ls -la Data/raw/
ls -la models/
```

### 📋 Verification Checklist

- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] Models downloaded in `models/`
- [ ] Data files in `Data/raw/`
- [ ] Configuration in `user_config.py` correct
- [ ] GPU available (optional but recommended)

## 🤝 Contributing

### 📝 How to Contribute

1. **Fork** this repository
2. **Create a branch** for your feature: `git checkout -b new-feature`
3. **Make your changes** and commit: `git commit -m "Add new feature"`
4. **Push your changes**: `git push origin new-feature`
5. **Open a Pull Request**

### 🐛 Report Issues

- Use the **Issues** system on GitHub
- Include **complete error logs**
- Describe the **environment** (OS, Python, CUDA version)
- Provide **steps to reproduce** the problem

## 📚 References and Resources

### 🔬 Original DRAFTS Publication

- **Paper**: [DRAFTS: A Deep Learning-Based Radio Fast Transient Search Pipeline](https://arxiv.org/abs/2410.03200)
- **Original Repository**: [SukiYume/DRAFTS](https://github.com/SukiYume/DRAFTS)

### 📖 Additional Documentation

- **Automatic Parameter System**: Optimized automatic configuration
- **SNR Analysis**: Signal-to-noise ratio analysis capabilities
- **Dispersion Correction**: Natural burst parabola visualization

### 🛠️ Related Tools

- **Presto**: Traditional pulse search pipeline
- **Heimdall**: Transient search pipeline
- **FAST**: Five-hundred-meter Aperture Spherical Telescope
- **GBT**: Green Bank Telescope

## 📞 Contact and Support

### 👥 Development Team

- **Main Maintainer**: [Your Name]
- **Institution**: UC
- **Email**: [your.email@uc.cl]

### 💬 Community

- **Issues**: [GitHub Issues](https://github.com/your-username/DRAFTS-UC/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/DRAFTS-UC/discussions)
- **Wiki**: [Detailed Documentation](https://github.com/your-username/DRAFTS-UC/wiki)

---

<div align="center">
  <sub>🔭✨📡 Searching for cosmic signals with DRAFTS-UC</sub>
</div>

---

## 📋 Project Status

- **Version**: 1.0.0
- **Status**: ✅ Active and in development
- **Last Update**: December 2024
- **Compatibility**: Python 3.8+, PyTorch 1.8+, CUDA 11.0+

### ✅ Implemented Features

- [x] Complete FRB detection pipeline
- [x] Pre-trained models (ResNet18/50)
- [x] CUDA acceleration for dedispersion
- [x] Automatic parameter system
- [x] Integrated SNR analysis
- [x] Dispersion correction for waterfall plots
- [x] Centralized logging system
- [x] Automatic plot generation
- [x] Multi-band support
- [x] Automatic memory optimization

### 🚧 In Development

- [ ] Web interface for monitoring
- [ ] More pre-trained models
- [ ] Integration with more telescopes
- [ ] Real-time analysis
- [ ] REST API for integration

### 📈 Performance Metrics

- **Speed**: Up to 10x faster than traditional methods
- **Accuracy**: >99% in candidate classification
- **Detection**: Double bursts detected vs. Heimdall
- **Memory**: Automatic optimization for available hardware

---

## 🔄 Relationship with Original DRAFTS

This project is a **specialized fork** of the original DRAFTS pipeline that:

- **Preserves** the core neural network architectures and pre-trained models
- **Maintains** compatibility with the original detection and classification workflows
- **Enhances** the pipeline with UC-specific optimizations and features
- **Simplifies** the configuration and execution process
- **Adds** new capabilities like dispersion correction and automatic parameter optimization

### 📚 Original DRAFTS Resources

- **Repository**: [https://github.com/SukiYume/DRAFTS](https://github.com/SukiYume/DRAFTS)
- **Paper**: [arXiv:2410.03200](https://arxiv.org/abs/2410.03200)
- **Documentation**: [Original DRAFTS README](README-old-Drafts.md)

### 🎯 What This Fork Provides

- **Ready-to-use pipeline** with minimal configuration
- **Optimized for UC research** and local data processing
- **Enhanced visualization** with dispersion correction
- **Automatic parameter optimization** based on hardware
- **Comprehensive logging** and debugging capabilities
