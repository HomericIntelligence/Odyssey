# ResNet-18 on CIFAR-10 Example

A complete implementation of ResNet-18 (Residual Network) for CIFAR-10 image classification
demonstrating the power of skip connections and deep networks.

## Overview

This example shows how to build, train
and run inference with the ResNet-18 architecture using ML Odyssey's shared library.

**Architecture**: ResNet-18 (He et al., 2015) - Deep Residual Learning

**Dataset**: CIFAR-10 (10 classes of RGB images: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

**Status**: ⚠️ **Forward pass complete — backward pass tracked in #3181**
Training requires `batch_norm2d_backward` (see [GAP_ANALYSIS.md](GAP_ANALYSIS.md))

## Quick Start

### 1. Download Dataset

```bash
python examples/resnet18-cifar10/download_cifar10.py
```text

This downloads CIFAR-10 (50,000 training + 10,000 test samples) to `datasets/cifar10/`.

### 2. Train Model

```bash
# NOTE: Training requires batch_norm2d_backward implementation
# See GAP_ANALYSIS.md for details
mojo run examples/resnet18-cifar10/train.mojo --epochs 200 --batch-size 128 --lr 0.01
```text

### 3. Run Inference

```bash
# Evaluate on test set (once weights are available)
mojo run examples/resnet18-cifar10/inference.mojo --weights-dir resnet18_weights
```text

## Key Innovation: Residual Learning

ResNet introduced **skip connections** (residual learning) to enable training very deep networks without vanishing
gradients.

### The Residual Block

Traditional deep networks suffer from vanishing gradients as they get deeper. ResNet solves this with skip connections:

```text
Input (x)
    ↓
    ├─────────────────┐  (skip connection)
    ↓                 ↓
Conv 3×3 → BN → ReLU
    ↓
Conv 3×3 → BN
    ↓
    ├─────────────────┘  (add skip)
    ↓
   ReLU
    ↓
Output
```text

**Key Insight**: Instead of learning H(x), learn F(x) = H(x) - x, then compute H(x) = F(x) + x

**Benefits**:

- **Easier optimization**: Residual learning is easier than direct mapping
- **Identity mapping**: If F(x) = 0, the block becomes identity (no harm in adding layers)
- **Gradient flow**: Gradients flow directly through skip connections
- **Depth**: Enables training networks with 100+ layers

### Projection Shortcuts

When dimensions change (channels or spatial size), use **1×1 convolutions** to match shapes:

```text
Input (x)
    ↓
    ├────────────────────────┐  (projection shortcut)
    ↓                        ↓
Conv 3×3 (stride=2) → BN → ReLU   Conv 1×1 (stride=2) → BN
    ↓                        ↓
Conv 3×3 → BN
    ↓                        ↓
    ├────────────────────────┘  (add skip)
    ↓
   ReLU
    ↓
Output
```text

## Model Architecture

### ResNet-18 (Adapted for 32×32 Input)

The classic ResNet-18 adapted for CIFAR-10's smaller images (32×32 vs 224×224 ImageNet).

```text
Input (32×32×3)
    ↓
Initial Block:
    Conv2D(64, 3×3, stride=1, pad=1) → BatchNorm → ReLU
    ↓ (32×32×64)
Stage 1 (64 channels, 2 blocks):
    ResBlock(64→64) → ResBlock(64→64)
    ↓ (32×32×64)
Stage 2 (128 channels, 2 blocks):
    ResBlock(64→128, stride=2) → ResBlock(128→128)
    ↓ (16×16×128)
Stage 3 (256 channels, 2 blocks):
    ResBlock(128→256, stride=2) → ResBlock(256→256)
    ↓ (8×8×256)
Stage 4 (512 channels, 2 blocks):
    ResBlock(256→512, stride=2) → ResBlock(512→512)
    ↓ (4×4×512)
Global Average Pool (4×4 → 1×1)
    ↓ (1×1×512)
Flatten
    ↓ (512)
Linear(512 → 10)
    ↓
Output (10 classes)
```text

### Adaptations for CIFAR-10

Compared to the original ResNet-18 for ImageNet (224×224):

1. **Smaller initial conv**: 3×3 instead of 7×7 (input is already small)
2. **No initial pooling**: Removed max pooling after first conv
3. **Same stages**: Keep 4 residual stages with 2 blocks each
4. **Same channels**: 64 → 128 → 256 → 512 progression
5. **Smaller FC layer**: 512 → 10 instead of 512 → 1000

### Parameters

- **Input Shape**: (batch, 3, 32, 32)
- **Output Shape**: (batch, 10)
- **Total Trainable Parameters**: 84 parameters (not counting tensors)
  - Initial: 6 params (conv + BN)
  - Stage 1: 16 params (2 blocks × 8 params, no projection)
  - Stage 2: 20 params (2 blocks, block1 has projection: 12 + 8)
  - Stage 3: 20 params (2 blocks, block1 has projection: 12 + 8)
  - Stage 4: 20 params (2 blocks, block1 has projection: 12 + 8)
  - FC: 2 params (weights + bias)
- **Total Tensor Elements**: ~11M (actual floating point values)
- **Memory**: ~44MB for float32 weights

### Architecture Details

Each **Residual Block** consists of:

- 2 × Conv3×3 layers (with padding=1 to preserve spatial size)
- 2 × Batch Normalization layers
- 2 × ReLU activations
- 1 × Skip connection (identity or projection)

**Identity Shortcut** (Stage 1, all second blocks):

- Simply add input to output: `y = F(x) + x`
- Used when input and output have same dimensions

**Projection Shortcut** (First block of Stages 2, 3, 4):

- 1×1 convolution to match dimensions: `y = F(x) + W_proj * x`
- Used when channels increase or spatial size decreases (stride=2)

## Dataset Information

### CIFAR-10 Dataset

60,000 32×32 color images in 10 classes (6,000 per class).

- **Source**: Canadian Institute for Advanced Research
- **Training**: 50,000 images (5 batches of 10,000)
- **Test**: 10,000 images (1 batch)
- **Format**: Binary batches (converted to IDX for Mojo)

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
examples/resnet18-cifar10/
├── README.md              # This file
├── model.mojo             # ResNet-18 model with skip connections
├── train.mojo             # Training with manual backward passes
├── inference.mojo         # Inference with weight loading
├── data_loader.mojo       # CIFAR-10 binary format loading
├── weights.mojo           # Hex-based weight serialization
├── download_cifar10.py    # Python script to download dataset
├── run_example.sh         # Complete workflow script
└── GAP_ANALYSIS.md        # Implementation gap analysis
```text

## Implementation Status

### ✅ Completed

- [x] Model architecture (all 4 stages with residual blocks)
- [x] Forward pass through all 18 layers
- [x] Skip connections (identity and projection shortcuts)
- [x] Batch normalization integration (forward pass)
- [x] Weight save/load functionality
- [x] CIFAR-10 data loading
- [x] Inference script
- [x] Training script structure
- [x] Comprehensive documentation

### ⚠️ Pending (Blocked)

- [ ] **Training**: Requires `batch_norm2d_backward` implementation
  - Critical missing component in shared library
  - See [GAP_ANALYSIS.md](GAP_ANALYSIS.md) for implementation details
  - Estimated: 4-6 hours of development time
- [ ] **Full backward pass**: Intentional placeholder — tracked in [GitHub issue #3181](https://github.com/HomericIntelligence/ProjectOdyssey/issues/3181)

**Current Limitation**: The forward pass is complete and functional, but training is blocked by the missing batch
normalization backward pass. Once `batch_norm2d_backward` is implemented in `shared/core/normalization.mojo`, the
training script can be completed.

## Expected Performance

Based on the reference implementation and similar experiments:

- **Training Time**: ~40-50 hours on CPU for 200 epochs (batch_size=128)
- **Expected Accuracy**: 93-94% on CIFAR-10 after 200 epochs
- **Peak Accuracy**: 95-96% with data augmentation
- **Memory Usage**: ~44MB for model weights

### Comparison with Other Architectures

| Model    | Parameters | CIFAR-10 Accuracy | Training Time (200 epochs) | Key Feature              |
|----------|------------|-------------------|----------------------------|--------------------------|
| LeNet-5  | 61K        | 70-75%            | 2-3 hours                  | Early CNN                |
| AlexNet  | 2.3M       | 80-85%            | 8-12 hours                 | Large kernels, dropout   |
| VGG-16   | 15M        | 91-93%            | 30-40 hours                | Very deep (16 layers)    |
| ResNet-18| 11M        | 93-94%            | 40-50 hours                | Skip connections         |

**Why ResNet-18 is Better**:

1. **Deeper without degradation**: 18 layers vs VGG's 16, but no vanishing gradients
2. **Better gradient flow**: Skip connections enable direct gradient paths
3. **Easier to train**: Residual learning is easier than learning direct mappings
4. **More efficient**: Fewer parameters than VGG but higher accuracy

## Advanced Features

### Skip Connection Mathematics

Given input `x` and learned mapping `F(x)`:

- **Traditional**: Learn H(x) directly
- **Residual**: Learn F(x) = H(x) - x, compute H(x) = F(x) + x

**Gradient Flow**:

```text
dL/dx = dL/dH * dH/dx
      = dL/dH * d(F(x) + x)/dx
      = dL/dH * (dF/dx + I)
      = dL/dH * dF/dx + dL/dH    (identity term prevents vanishing)
```text

The identity term `dL/dH` ensures gradients always flow, even if `dF/dx ≈ 0`.

### Batch Normalization

Applied after every convolution to:

- **Normalize activations**: Keep internal distributions stable
- **Accelerate training**: Higher learning rates possible
- **Regularize**: Reduces need for dropout

Formula:

```text
x_norm = (x - mean) / sqrt(var + eps)
y = gamma * x_norm + beta
```text

During training:

- Compute mean and variance over batch
- Update running statistics with momentum

During inference:

- Use running mean and variance (fixed)

### Learning Rate Scheduling

Step decay schedule (same as VGG-16):

- **Schedule**: Decay by 5× every 60 epochs
- **Formula**: `lr = initial_lr * (0.2 ** (epoch // 60))`
- **Example**:
  - Epochs 0-59: lr = 0.01
  - Epochs 60-119: lr = 0.002
  - Epochs 120-179: lr = 0.0004
  - Epochs 180+: lr = 0.00008

### Weight Initialization

**He initialization** (He et al., 2015) for all convolutional layers:

- Designed specifically for ReLU activations
- Maintains variance through network depth
- Critical for training very deep networks
- Formula: `weights ~ N(0, sqrt(2 / fan_in))`

## Design Principles (KISS)

This example follows **Keep It Simple, Stupid** principles:

1. **Minimal Dependencies**: Uses only ML Odyssey shared library
2. **Functional Design**: Uses functional ops (conv2d, linear, relu, add) from shared/core
3. **Clear Structure**: Separate files for model, training, and inference
4. **Simple Interfaces**: Command-line arguments for configuration
5. **No Over-Engineering**: Direct implementation without unnecessary abstractions
6. **Pattern Reuse**: Follows the same structure as AlexNet and VGG-16 examples

## Usage Details

### Training Options

```bash
mojo run examples/resnet18-cifar10/train.mojo \
    --epochs 200 \
    --batch-size 128 \
    --lr 0.01 \
    --momentum 0.9 \
    --data-dir datasets/cifar10 \
    --weights-dir resnet18_weights
```text

**Arguments**:

- `--epochs`: Number of training epochs (default: 200)
- `--batch-size`: Mini-batch size (default: 128)
- `--lr`: Initial learning rate for SGD (default: 0.01)
- `--momentum`: Momentum factor for SGD (default: 0.9)
- `--data-dir`: Path to CIFAR-10 dataset directory (default: `datasets/cifar10`)
- `--weights-dir`: Directory to save model weights (default: `resnet18_weights`)

### Inference Options

```bash
# Test set evaluation
mojo run examples/resnet18-cifar10/inference.mojo \
    --weights-dir resnet18_weights \
    --data-dir datasets/cifar10
```text

**Arguments**:

- `--weights-dir`: Directory containing saved model weights (default: `resnet18_weights`)
- `--data-dir`: Path to CIFAR-10 dataset for test set evaluation (default: `datasets/cifar10`)

## Gap Analysis

See [GAP_ANALYSIS.md](GAP_ANALYSIS.md) for:

- Detailed analysis of missing components
- Implementation plan for `batch_norm2d_backward`
- Mathematical formulation and references
- Estimated timeline (4-6 hours)

**Summary**: ResNet-18 is **98% complete**. Only `batch_norm2d_backward` is missing.

## References

### Papers

1. **ResNet (Original)**:
   He, K., Zhang, X., Ren, S., & Sun, J. (2015).
   Deep residual learning for image recognition.
   *CVPR 2016*.
   [arXiv Paper](https://arxiv.org/abs/1512.03385)

2. **He Initialization**:
   He, K., Zhang, X., Ren, S., & Sun, J. (2015).
   Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification.
   *ICCV 2015*.
   [Paper](https://arxiv.org/abs/1502.01852)

3. **Batch Normalization**:
   Ioffe, S., & Szegedy, C. (2015).
   Batch normalization: Accelerating deep network training by reducing internal covariate shift.
   *ICML 2015*.
   [Paper](https://arxiv.org/abs/1502.03167)

4. **CIFAR-10 Dataset**:
   Krizhevsky, A., & Hinton, G. (2009).
   Learning multiple layers of features from tiny images.
   *Technical report, University of Toronto*.
   [Tech Report](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

### Datasets

- **CIFAR-10 Official Page**: <https://www.cs.toronto.edu/~kriz/cifar.html>
- **Download**: <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>

### Reference Implementations

- **ResNet PyTorch**: <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>
  - Official PyTorch implementation
  - Demonstrates architecture details and skip connections

- **ResNet TensorFlow**: <https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/resnet.py>
  - TensorFlow official implementation

### Related Resources

- **Papers with Code - ResNet**: <https://paperswithcode.com/method/resnet>
- **ResNet Explained**: <https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8>
- **Residual Learning Visualization**: <https://d2l.ai/chapter_convolutional-modern/resnet.html>

## Contributing

This example is part of ML Odyssey. Contributions welcome!

**Priority Tasks**:

1. **Implement `batch_norm2d_backward`** in `shared/core/normalization.mojo`
2. **Complete training script** backward pass
3. **Add data augmentation** integration (RandomCrop and RandomHorizontalFlip available)
4. **Optimize convolutions** with SIMD vectorization
5. **Add learning rate schedules** (cosine annealing, warm restarts)

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

This example is part of ML Odyssey and follows the project's license.

See [LICENSE](../../LICENSE) for details.

## Acknowledgments

- **Dataset**: CIFAR (Canadian Institute for Advanced Research) for creating and releasing the CIFAR-10 dataset
- **Architecture**: Kaiming He et al. for the ResNet architecture and residual learning
- **Inspiration**: VGG-16 and AlexNet examples for establishing implementation patterns
- **ML Odyssey**: The shared library providing functional neural network operations

## Next Steps

Once `batch_norm2d_backward` is implemented:

1. Complete the backward pass in `train.mojo`
2. Run training for 200 epochs
3. Evaluate accuracy on CIFAR-10 test set
4. Compare with published results (expected: 93-94%)
5. Experiment with data augmentation for higher accuracy (95-96%)

For now, you can:

- Explore the model architecture in `model.mojo`
- Understand skip connections and residual learning
- Run forward passes and inspect outputs
- Prepare for training by downloading the dataset
