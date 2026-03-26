# VGG-16 on CIFAR-10 Example

A simple implementation of VGG-16 convolutional neural network for CIFAR-10 image classification, following KISS principles.

## Overview

This example demonstrates how to use ML Odyssey's shared library to build, train, and run inference with the VGG-16
architecture on the CIFAR-10 dataset.

**Architecture**: VGG-16 (Simonyan & Zisserman, 2014) - Very Deep Networks

**Dataset**: CIFAR-10 (10 classes of RGB images: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

**Status**: ✅ **Complete Implementation** - Full manual backpropagation through all 16 layers implemented

## Quick Start

### 1. Download Dataset

```bash
python examples/vgg16_cifar10/download_cifar10.py
```text

This downloads the CIFAR-10 dataset (50,000 training samples, 10,000 test samples, 10 classes) to `datasets/cifar10/`.

### 2. Train Model

```bash
mojo run examples/vgg16_cifar10/train.mojo --epochs 200 --batch-size 128 --lr 0.01
```text

### 3. Run Inference

```bash
# Evaluate on test set
mojo run examples/vgg16_cifar10/inference.mojo --weights-dir vgg16_weights
```text

## Key Innovation: Depth

VGG-16 pioneered the use of **very deep networks** with a simple, repetitive architecture:

- **16 layers**: 13 convolutional + 3 fully connected
- **Uniform 3×3 filters**: All conv layers use small 3×3 kernels
- **Depth over width**: Stacks multiple conv layers to increase depth rather than using large kernels
- **Simple design**: Repetitive blocks make the architecture easy to implement and understand

### Why 3×3 Convolutions?

Two 3×3 conv layers have the same receptive field as one 5×5 layer, but:

- **Fewer parameters**: 2×(3²) = 18 vs 5² = 25 (28% fewer)
- **More non-linearities**: 2 ReLU activations instead of 1
- **Better feature learning**: More expressive due to additional non-linearity

## Model Architecture

### VGG-16 (Adapted for 32×32 Input)

The classic VGG-16 architecture adapted for CIFAR-10's smaller image size (32×32 vs 224×224).

```text
Input (32×32×3)
    ↓
Block 1 (64 channels):
    Conv2D (64, 3×3, pad=1) → ReLU
    Conv2D (64, 3×3, pad=1) → ReLU
    MaxPool (2×2, stride=2) → (16×16×64)
    ↓
Block 2 (128 channels):
    Conv2D (128, 3×3, pad=1) → ReLU
    Conv2D (128, 3×3, pad=1) → ReLU
    MaxPool (2×2, stride=2) → (8×8×128)
    ↓
Block 3 (256 channels):
    Conv2D (256, 3×3, pad=1) → ReLU
    Conv2D (256, 3×3, pad=1) → ReLU
    Conv2D (256, 3×3, pad=1) → ReLU
    MaxPool (2×2, stride=2) → (4×4×256)
    ↓
Block 4 (512 channels):
    Conv2D (512, 3×3, pad=1) → ReLU
    Conv2D (512, 3×3, pad=1) → ReLU
    Conv2D (512, 3×3, pad=1) → ReLU
    MaxPool (2×2, stride=2) → (2×2×512)
    ↓
Block 5 (512 channels):
    Conv2D (512, 3×3, pad=1) → ReLU
    Conv2D (512, 3×3, pad=1) → ReLU
    Conv2D (512, 3×3, pad=1) → ReLU
    MaxPool (2×2, stride=2) → (1×1×512)
    ↓
Flatten (512)
    ↓
Linear (512 → 512) → ReLU → Dropout(0.5)
    ↓
Linear (512 → 512) → ReLU → Dropout(0.5)
    ↓
Linear (512 → 10)
    ↓
Output (10 classes)
```text

### Parameters

- **Input Shape**: (batch, 3, 32, 32)
- **Output Shape**: (batch, 10)
- **Total Parameters**: ~15M (scaled down from original 138M for ImageNet)
  - Block 1: 2 conv layers = 38,464 params
  - Block 2: 2 conv layers = 221,696 params
  - Block 3: 3 conv layers = 1,475,584 params
  - Block 4: 3 conv layers = 7,079,424 params
  - Block 5: 3 conv layers = 7,079,424 params
  - FC1: 512×512+512 = 262,656 params
  - FC2: 512×512+512 = 262,656 params
  - FC3: 512×10+10 = 5,130 params

**Note**: Scaled down from original VGG-16 (138M parameters on ImageNet) by reducing FC layer sizes from 4096 to 512.

## Dataset Information

### CIFAR-10 Dataset

The CIFAR-10 dataset consists of 60,000 32×32 color images in 10 classes, with 6,000 images per class.

- **Source**: Canadian Institute for Advanced Research (CIFAR)
- **Format**: 32×32 RGB images
- **File Format**: Binary batches (5 training batches + 1 test batch)

### Classes

| Index | Class      | Description               |
|-------|------------|---------------------------|
| 0     | airplane   | Various types of aircraft |
| 1     | automobile | Cars and trucks           |
| 2     | bird       | Various bird species      |
| 3     | cat        | Domestic cats             |
| 4     | deer       | Deer in various poses     |
| 5     | dog        | Domestic dogs             |
| 6     | frog       | Frogs and toads           |
| 7     | horse      | Horses                    |
| 8     | ship       | Boats and ships           |
| 9     | truck      | Large trucks              |

## File Structure

```text
examples/vgg16_cifar10/
├── README.md              # This file
├── model.mojo             # VGG-16 model with save/load
├── train.mojo             # Training with manual backward passes
├── inference.mojo         # Inference with weight loading
├── data_loader.mojo       # CIFAR-10 binary format loading
├── weights.mojo           # Hex-based weight serialization
├── download_cifar10.py    # Python script to download dataset
├── run_example.sh         # Complete workflow script
└── GAP_ANALYSIS.md        # Implementation gap analysis
```text

## Usage Details

### Training Options

```bash
mojo run examples/vgg16_cifar10/train.mojo \
    --epochs 200 \
    --batch-size 128 \
    --lr 0.01 \
    --momentum 0.9 \
    --data-dir datasets/cifar10 \
    --weights-dir vgg16_weights
```text

**Arguments**:

- `--epochs`: Number of training epochs (default: 200)
- `--batch-size`: Mini-batch size (default: 128)
- `--lr`: Initial learning rate for SGD (default: 0.01)
- `--momentum`: Momentum factor for SGD (default: 0.9)
- `--data-dir`: Path to CIFAR-10 dataset directory (default: `datasets/cifar10`)
- `--weights-dir`: Directory to save model weights (default: `vgg16_weights`)

### Inference Options

```bash
# Test set evaluation
mojo run examples/vgg16_cifar10/inference.mojo \
    --weights-dir vgg16_weights \
    --data-dir datasets/cifar10
```text

**Arguments**:

- `--weights-dir`: Directory containing saved model weights (default: `vgg16_weights`)
- `--data-dir`: Path to CIFAR-10 dataset for test set evaluation (default: `datasets/cifar10`)

## Implementation Status

### ✅ Completed

- [x] Dataset download script (Python)
- [x] VGG-16 model architecture with functional ops
- [x] Forward pass through all 16 layers
- [x] **Complete manual backpropagation** through all 16 layers
  - ✅ Gradients for 13 conv layers + 5 pool layers + 3 FC layers + 2 dropout layers
  - ✅ Full backward pass from loss to input
- [x] CIFAR-10 data loading with RGB support
- [x] Hex-based weight serialization (32 parameter files)
- [x] **Full parameter updates** with SGD+Momentum optimizer
  - ✅ 32 velocity tensors initialized (one per parameter)
  - ✅ Momentum updates applied to all parameters
- [x] **Batch slicing** for mini-batch training
  - ✅ Efficient batch extraction using shared library utilities
  - ✅ Proper handling of partial batches at epoch end
  - ✅ Memory-efficient mini-batch processing
- [x] Learning rate decay scheduling (step decay every 60 epochs by 0.2x)
- [x] Inference script with weight loading
- [x] Comprehensive documentation

### 🔄 Optional Enhancements

- [ ] **Data augmentation** (RandomCrop and RandomHorizontalFlip available in `shared.data`)
- [ ] **Batch normalization** (optional variant of VGG-16)
- [ ] SIMD vectorization for convolutions
- [ ] Multi-threading for data loading
- [ ] Mixed precision training (float16 for speed)

## Expected Performance

Based on the reference implementation and similar experiments:

- **Training Time**: ~30-40 hours on CPU for 200 epochs (batch_size=128)
- **Expected Accuracy**: 91-93% on CIFAR-10 after 200 epochs
- **Peak Accuracy**: 93-94% with data augmentation
- **Memory Usage**: ~60MB for model weights

### Comparison with Other Architectures

| Model | Parameters | CIFAR-10 Accuracy | Training Time (200 epochs) |
|-------|------------|-------------------|----------------------------|
| LeNet-5 | 61K | 70-75% | 2-3 hours |
| AlexNet | 2.3M | 80-85% | 8-12 hours |
| VGG-16 | 15M | 91-93% | 30-40 hours |

## Advanced Features

### Learning Rate Decay

The training script includes automatic learning rate decay using a step schedule:

- **Schedule**: Decay by 5x every 60 epochs
- **Formula**: `lr = initial_lr * (0.2 ** (epoch // 60))`
- **Example**:
  - Epochs 0-59: lr = 0.01
  - Epochs 60-119: lr = 0.002
  - Epochs 120-179: lr = 0.0004
  - Epochs 180+: lr = 0.00008

This matches the typical learning rate schedule used for VGG training.

### Weight Initialization

VGG-16 uses **He initialization** (He et al., 2015) for all convolutional layers:

- Designed for ReLU activations
- Helps prevent vanishing/exploding gradients in deep networks
- Formula: `weights ~ N(0, sqrt(2 / fan_in))`

### Dropout Regularization

Dropout (p=0.5) is applied to FC layers to prevent overfitting:

- Only active during training
- Disabled during inference
- Standard technique from original VGG paper

## Design Principles (KISS)

This example follows **Keep It Simple, Stupid** principles:

1. **Minimal Dependencies**: Uses only ML Odyssey shared library
2. **Functional Design**: Model uses functional ops (conv2d, linear, relu, dropout) from shared/core
3. **Clear Structure**: Separate files for model, training, and inference
4. **Simple Interfaces**: Command-line arguments for configuration
5. **No Over-Engineering**: Direct implementation without unnecessary abstractions
6. **Pattern Reuse**: Follows the same structure as AlexNet example

## References

### Papers

1. **VGG Networks (Original)**:
   Simonyan, K., & Zisserman, A. (2014).
   Very deep convolutional networks for large-scale image recognition.
   *arXiv preprint arXiv:1409.1556*.
   [arXiv Paper](https://arxiv.org/abs/1409.1556)

2. **He Initialization**:
   He, K., Zhang, X., Ren, S., & Sun, J. (2015).
   Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification.
   *ICCV 2015*.
   [Paper](https://arxiv.org/abs/1502.01852)

3. **CIFAR-10 Dataset**:
   Krizhevsky, A., & Hinton, G. (2009).
   Learning multiple layers of features from tiny images.
   *Technical report, University of Toronto*.
   [Tech Report](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

### Datasets

- **CIFAR-10 Official Page**: <https://www.cs.toronto.edu/~kriz/cifar.html>
- **Download**: <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>

### Reference Implementations

- **VGG PyTorch**: <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>
  - Official PyTorch implementation
  - Demonstrates architecture details

### Related Resources

- **Papers with Code - VGG**: <https://paperswithcode.com/method/vgg>
- **VGG Explained**: <https://neurohive.io/en/popular-networks/vgg16/>

## Contributing

This example is part of ML Odyssey. Contributions welcome!

1. Complete manual backpropagation through all 16 layers
2. Implement proper batch slicing for mini-batch training
3. Add batch normalization variant (VGG-16-BN)
4. Integrate data augmentation into training loop
5. Add SIMD optimization for 3×3 convolutions
6. Implement cosine annealing learning rate schedule

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

This example is part of ML Odyssey and follows the project's license.

See [LICENSE](../../LICENSE) for details.

## Acknowledgments

- **Dataset**: CIFAR (Canadian Institute for Advanced Research) for creating and releasing the CIFAR-10 dataset
- **Architecture**: Karen Simonyan and Andrew Zisserman for the VGG architecture
- **Inspiration**: AlexNet CIFAR-10 example for establishing the implementation patterns
- **ML Odyssey**: The shared library providing functional neural network operations
