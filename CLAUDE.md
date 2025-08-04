# Weather Image Classification Project

## Project Overview

A PyTorch-based weather image classification system optimized for RTX 3090 GPU training. Classifies weather conditions into 11 categories using state-of-the-art deep learning models including ResNet and EfficientNet architectures.

## Dataset

- **Size**: 6,862 images across 11 weather classes
- **Classes**: dew, fogsmog, frost, glaze, hail, lightning, rain, rainbow, rime, sandstorm, snow
- **Split**: 80% training (5,489 images) / 20% validation (1,373 images)
- **Format**: JPG images organized in class-based directories

## Model Architecture

### Supported Models

**ResNet Family:**
- ResNet18: 11.2M parameters
- ResNet34: 21.3M parameters  
- ResNet50: 23.5M parameters
- ResNet101: 42.5M parameters

**EfficientNet Family (Full B0-B7 Support):**
- EfficientNet B0: 4.0M parameters
- EfficientNet B1: 6.5M parameters
- EfficientNet B2: 7.7M parameters
- EfficientNet B3: 10.7M parameters
- EfficientNet B4: 17.6M parameters
- EfficientNet B5: 28.4M parameters
- EfficientNet B6: 40.8M parameters
- EfficientNet B7: 63.8M parameters

### Architecture Details

- **Base**: Pre-trained ImageNet weights
- **Head**: Custom classification layer with dropout
- **Input**: 224x224 RGB images
- **Output**: 11-class weather classification

## Core Files

### Essential Scripts

- **`train_rtx3090.py`**: RTX 3090 optimized training launcher
- **`train_pytorch.py`**: Main training script with full configuration options
- **`evaluate_pytorch.py`**: Model evaluation and analysis script
- **`model.py`**: Model definitions and architecture implementations
- **`dataset.py`**: Dataset loading and preprocessing utilities

### Training Infrastructure

- **Mixed Precision Training**: AMP support for faster training
- **Model Compilation**: PyTorch 2.0 torch.compile optimization
- **Memory Optimization**: channels_last memory format
- **Advanced Scheduling**: Cosine annealing, step, and plateau schedulers
- **Regularization**: Label smoothing, dropout, weight decay

## Training Configuration

### RTX 3090 Optimized Settings

```python
# Current RTX 3090 configuration
model_type = "efficientnet_b4"
batch_size = 64
epochs = 10
learning_rate = 1e-3
optimizer = "adamw"
scheduler = "cosine"
use_amp = True
channels_last = True
compile_model = True
```

### Performance Optimizations

- **Automatic Mixed Precision (AMP)**: ~2x speedup with minimal accuracy loss
- **Channels-Last Memory Format**: Better GPU memory utilization
- **Model Compilation**: PyTorch 2.0 graph optimization
- **Efficient Data Loading**: Multi-worker data loading with pin_memory

## Usage Examples

### Quick Training (RTX 3090)
```bash
python train_rtx3090.py
```

### Custom Training
```bash
python train_pytorch.py \
    --model-type efficientnet_b2 \
    --batch-size 32 \
    --epochs 50 \
    --lr 1e-3 \
    --use-amp \
    --channels-last
```

### Model Evaluation
```bash
python evaluate_pytorch.py rtx3090_training/best_model.pth \
    --data-dir dataset \
    --split val
```

## Recent Changes

### EfficientNet Integration (Latest)
- ✅ Added full EfficientNet B0-B7 support
- ✅ Updated training script choices for all variants
- ✅ Tested all model variants (B0: 4.0M to B7: 63.8M parameters)
- ✅ Maintained compatibility with existing training pipeline

### Repository Cleanup
- ✅ Removed Ray Data experiment files (moved to pure PyTorch)
- ✅ Cleaned up debug and testing scripts
- ✅ Streamlined to essential training pipeline
- ✅ Preserved RTX 3090 optimized workflows

### Training Pipeline Features
- ✅ Stratified train/validation splitting
- ✅ Comprehensive logging and checkpointing
- ✅ Training curves and confusion matrix visualization
- ✅ Early stopping and best model tracking
- ✅ Classification reports with per-class metrics

## Technical Implementation

### Data Pipeline
- **Preprocessing**: Resize, normalize with ImageNet statistics
- **Augmentation**: Random horizontal flip, rotation, color jitter
- **Loading**: Efficient DataLoader with multi-worker support
- **Error Handling**: Robust image loading with fallback mechanisms

### Training Infrastructure
- **Checkpointing**: Regular saves with configurable retention
- **Monitoring**: Real-time loss/accuracy logging
- **Visualization**: Automatic generation of training curves and confusion matrices
- **Evaluation**: Comprehensive validation metrics and analysis

### GPU Optimizations
- **Memory Efficiency**: Optimized for RTX 3090 memory constraints
- **Speed**: Mixed precision and compilation optimizations
- **Scalability**: Configurable batch sizes and worker counts
- **Stability**: Gradient scaling and numerical stability features

## Project Status

**Current State**: Production-ready PyTorch training pipeline with full EfficientNet support
**Performance**: Optimized for single RTX 3090 GPU training
**Maintainability**: Clean, documented codebase with minimal dependencies
**Extensibility**: Easy to add new model architectures and optimizations