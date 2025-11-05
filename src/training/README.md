# DRAFTS Model Training

This directory contains training scripts for both detection and classification models used in the DRAFTS pipeline.

## Overview

The DRAFTS pipeline uses two deep learning models:

1. **Object Detection (CenterNet)** - Localizes FRB candidates in time-DM space
2. **Binary Classification (ResNet)** - Verifies candidates as real FRBs or RFI/noise

## Directory Structure

```
src/training/
├── object_detection/           # CenterNet training scripts
│   ├── centernet_train.py      # Training script
│   ├── centernet_data.py       # Dataset and data augmentation
│   └── data_label_example.txt  # Example label file format
├── binary_classification/      # ResNet classifier training scripts
│   ├── binary_train.py         # Training script
│   └── binary_data.py          # Dataset and data augmentation
└── README.md                   # This file
```

---

## Object Detection Training (CenterNet)

### Data Preparation

1. Prepare your training data as `.npy` files with shape `(freq_slices, 1024, 8192)`
2. Create a `data_label.txt` CSV file with the following format:

```csv
save_name,freq_slice,time_center,dm_center,time_left,dm_left
00000.npy,0,7743.7,564.66,7613.21,627.03
00001.npy,1,-1.0,-1.0,-1.0,-1.0
```

**Columns:**
- `save_name`: NPY filename
- `freq_slice`: Frequency band index
- `time_center`, `dm_center`: Center coordinates of burst box
- `time_left`, `dm_left`: Left-bottom coordinates of burst box
- Use `-1.0` for all coordinates when no burst is present

3. Place data files in `./Data/` directory
4. Place `data_label.txt` in the same directory as `centernet_train.py`

### Training

```bash
cd src/training/object_detection

# Train with ResNet18 backbone (faster)
python centernet_train.py resnet18

# Train with ResNet50 backbone (more accurate)
python centernet_train.py resnet50
```

### Output

Training creates a `logs_{backbone}/` directory containing:
- `best_model.pth` - Best performing model checkpoint
- `EpochXX_TLossX.XXX_VLossX.XXX.pth` - Epoch-specific checkpoints
- `logs_{backbone}.json` - Training metrics per epoch

### Using Trained Models

After training, copy the best model to the models directory:

```bash
cp logs_resnet18/best_model.pth ../../models/cent_resnet18.pth
```

---

## Binary Classification Training (ResNet)

### Data Preparation

1. Organize data in the following structure:

```
Data/
├── True/      # Real FRB candidates (.npy files)
└── False/     # RFI/noise samples (.npy files)
```

Each `.npy` file should contain a 512x512 array representing a candidate patch.

2. Place data directory in `./Data/` relative to the training script

### Training

```bash
cd src/training/binary_classification

# Standard training with ResNet18 (fixed 512x512 input)
python binary_train.py resnet18 BinaryNet True

# Training with Spatial Pyramid Pooling (arbitrary input sizes)
python binary_train.py resnet18 SPPResNet True

# Training with random input sizes (data augmentation)
python binary_train.py resnet18 BinaryNet False
```

**Arguments:**
- `arg1`: Backbone architecture (`resnet18` or `resnet50`)
- `arg2`: Model type (`BinaryNet` or `SPPResNet`)
- `arg3`: Fixed size (`True`) or random size (`False`)

### Output

Training creates a `logs_{model}_{backbone}_fix/` or `logs_{model}_{backbone}_ran/` directory containing:
- `best_model.pth` - Best performing model checkpoint
- `EpochXXX_TlossX.XXX_TaccX.XXX_VlossX.XXX_VaccX.XXX.pth` - Epoch checkpoints
- `logs.npy` - Training history array

### Using Trained Models

After training, copy the best model to the models directory:

```bash
cp logs_res_resnet18_fix/best_model.pth ../../models/class_resnet18.pth
```

---

## Training Configuration

### Object Detection (CenterNet)

Default parameters in `centernet_train.py`:
- Input size: 512x512
- Model scale: 4
- Batch size: 4
- Epochs: 100
- Patience: 100 (early stopping)
- Optimizer: Adam
- Scheduler: Cosine LR with warmup

### Binary Classification (ResNet)

Default parameters in `binary_train.py`:
- Input size: 512x512
- Batch size: 16
- Epochs: 50
- Classes: 2 (BURST / NO-BURST)
- Optimizer: AdamW
- Scheduler: Cosine LR with warmup
- Loss: CrossEntropyLoss

---

## Data Augmentation

### Object Detection
- Random cropping
- Random downsampling
- Multi-image combination (1-5 images)
- Gaussian heatmap generation

### Binary Classification
- Random rotation (±25°)
- Random vertical flip (50%)
- Random horizontal flip (50%)
- Synthetic RFI injection:
  - Zero padding
  - Linear streaks
  - Parabolic dispersed signals
  - Periodic horizontal bands
  - Narrowband diagonal lines
  - Gradient bands
  - Scattered noise
  - Gaussian noise

---

## Requirements

The training scripts require the following dependencies (already in `requirements.txt`):

```
torch>=2.0.0
torchvision
numpy
scipy
scikit-learn
scikit-image
matplotlib
seaborn
opencv-python-headless
pandas
tqdm
timm
```

---

## Notes

- Training scripts are from the original DRAFTS repository
- Scripts use absolute imports and expect specific directory structures
- For best results, use GPU-enabled systems (CUDA)
- Training time varies: ~hours for CenterNet, ~30min-1h for binary classifier
- Model performance depends heavily on training data quality and quantity

---

## References

- Original DRAFTS repository: https://github.com/SukiYume/DRAFTS
- Paper: Zhang et al. (2024), arXiv:2410.03200

