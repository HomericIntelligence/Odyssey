# MobileNetV1 on CIFAR-10 Example

A complete implementation of MobileNetV1 for CIFAR-10 image classification
demonstrating efficient neural network design for mobile and embedded devices.

## Overview

This example shows how to build, train
and run inference with the MobileNetV1 architecture using ML Odyssey's shared library.

**Architecture**: MobileNetV1 (Howard et al., 2017) - Efficient Convolutional Neural Networks

**Dataset**: CIFAR-10 (10 classes of RGB images: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

**Status**: 🚧 **In Development** - Implementation in progress

## Quick Start

### 1. Download Dataset

```bash
python examples/mobilenetv1-cifar10/download_cifar10.py
```text

This downloads CIFAR-10 (50,000 training + 10,000 test samples) to `datasets/cifar10/`.

### 2. Train Model

```bash
mojo run examples/mobilenetv1-cifar10/train.mojo --epochs 200 --batch-size 128 --lr 0.01
```text

### 3. Run Inference

```bash
mojo run examples/mobilenetv1-cifar10/inference.mojo --weights-dir mobilenetv1_weights
```text

## Key Innovation: Depthwise Separable Convolutions

MobileNetV1 introduced **depthwise separable convolutions**
a revolutionary factorization that reduces computation by 8-9× compared to standard convolutions.

### Standard Convolution

Traditional convolution applies filters across all input channels:

```text
Input: (H × W × C_in)
Filters: (K × K × C_in × C_out)
Output: (H × W × C_out)

Operations: H × W × C_in × C_out × K × K
```text

**Example**: 32×32 input with 64 channels, 3×3 filters, 128 output channels:

- Operations: 32 × 32 × 64 × 128 × 3 × 3 = **75,497,472 operations**

### Depthwise Separable Convolution

Splits convolution into two stages:

```text
Stage 1: Depthwise Convolution (spatial filtering)
    Input: (H × W × C_in)
    Filters: (K × K × 1) per channel
    Output: (H × W × C_in)
    Operations: H × W × C_in × K × K

Stage 2: Pointwise Convolution (channel mixing)
    Input: (H × W × C_in)
    Filters: (1 × 1 × C_in × C_out)
    Output: (H × W × C_out)
    Operations: H × W × C_in × C_out

Total: H × W × C_in × (K² + C_out)
```text

**Same Example**:

- Depthwise: 32 × 32 × 64 × 3 × 3 = 589,824 operations
- Pointwise: 32 × 32 × 64 × 128 = 8,388,608 operations
- **Total: 8,978,432 operations** (8.4× reduction!)

### Mathematical Comparison

**Reduction Factor**:

```text
Standard:   H × W × C_in × C_out × K²
Depthwise:  H × W × C_in × (K² + C_out)

Reduction = (K² + C_out) / (C_out × K²) ≈ 1/C_out + 1/K²
```text

For typical values (C_out=128, K=3):

- Reduction ≈ 1/128 + 1/9 ≈ 0.119 ≈ **8.4× fewer operations**

## Model Architecture

### MobileNetV1 (Adapted for 32×32 Input)

The classic MobileNetV1 adapted for CIFAR-10's smaller images (32×32 vs 224×224 ImageNet).

```text
Input (32×32×3)
    ↓
Initial Block:
    Conv2D(32, 3×3, stride=2, pad=1) → BN → ReLU
    ↓ (16×16×32)
Depthwise Separable 1:
    DepthwiseConv(32, 3×3, stride=1) → BN → ReLU → PointwiseConv(64, 1×1) → BN → ReLU
    ↓ (16×16×64)
Depthwise Separable 2 (stride=2):
    DepthwiseConv(64, 3×3, stride=2) → BN → ReLU → PointwiseConv(128, 1×1) → BN → ReLU
    ↓ (8×8×128)
Depthwise Separable 3:
    DepthwiseConv(128, 3×3, stride=1) → BN → ReLU → PointwiseConv(128, 1×1) → BN → ReLU
    ↓ (8×8×128)
Depthwise Separable 4 (stride=2):
    DepthwiseConv(128, 3×3, stride=2) → BN → ReLU → PointwiseConv(256, 1×1) → BN → ReLU
    ↓ (4×4×256)
Depthwise Separable 5:
    DepthwiseConv(256, 3×3, stride=1) → BN → ReLU → PointwiseConv(256, 1×1) → BN → ReLU
    ↓ (4×4×256)
Depthwise Separable 6 (stride=2):
    DepthwiseConv(256, 3×3, stride=2) → BN → ReLU → PointwiseConv(512, 1×1) → BN → ReLU
    ↓ (2×2×512)
Depthwise Separable 7-11 (5 blocks):
    DepthwiseConv(512, 3×3, stride=1) → BN → ReLU → PointwiseConv(512, 1×1) → BN → ReLU
    ↓ (2×2×512)
Depthwise Separable 12 (stride=2):
    DepthwiseConv(512, 3×3, stride=2) → BN → ReLU → PointwiseConv(1024, 1×1) → BN → ReLU
    ↓ (1×1×1024)
Depthwise Separable 13:
    DepthwiseConv(1024, 3×3, stride=1) → BN → ReLU → PointwiseConv(1024, 1×1) → BN → ReLU
    ↓ (1×1×1024)
Global Average Pool (1×1 → 1×1)
    ↓ (1024)
Linear(1024 → 10)
    ↓
Output (10 classes)
```text

### Adaptations for CIFAR-10

Compared to the original MobileNetV1 for ImageNet (224×224):

1. **Smaller initial stride**: stride=2 instead of stride=2 (input is already small)
2. **Fewer downsampling steps**: Adjusted stride-2 layers for 32×32 input
3. **Same number of layers**: Keep 13 depthwise separable blocks
4. **Same channel progression**: 32 → 64 → 128 → 256 → 512 → 1024
5. **Smaller FC layer**: 1024 → 10 instead of 1024 → 1000

### Parameters

- **Input Shape**: (batch, 3, 32, 32)
- **Output Shape**: (batch, 10)
- **Total Trainable Parameters**: ~4.2M (smallest yet!)
  - Initial conv: ~900
  - Depthwise separable blocks: ~4.1M
  - FC layer: ~10K
- **Memory**: ~17MB for float32 weights
- **Operations per inference**: ~60M (vs VGG-16's 15B!)

### Architecture Details

Each **Depthwise Separable Block** consists of:

1. **Depthwise Convolution**: Applies one filter per input channel
   - Filters: (3×3×1) per channel
   - Stride: 1 or 2 (for downsampling)
   - Padding: 1 (to preserve spatial size when stride=1)
   - Batch Normalization + ReLU

2. **Pointwise Convolution**: 1×1 convolution for channel mixing
   - Filters: (1×1×C_in×C_out)
   - Stride: 1
   - No padding
   - Batch Normalization + ReLU

**Key Benefits**:

- **Efficiency**: 8-9× fewer operations than standard convolution
- **Fewer parameters**: Smaller model size
- **Same accuracy**: Competitive performance with standard CNNs
- **Mobile-friendly**: Designed for resource-constrained devices

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
examples/mobilenetv1-cifar10/
├── README.md              # This file
├── model.mojo             # MobileNetV1 model with depthwise separable convolutions
├── train.mojo             # Training with manual backward passes
├── inference.mojo         # Inference with weight loading
├── data_loader.mojo       # CIFAR-10 binary format loading (symlink to resnet18)
├── weights.mojo           # Hex-based weight serialization
└── download_cifar10.py    # Python script to download dataset (symlink to resnet18)
```text

## Implementation Status

### ✅ Planned

- [ ] Model architecture (13 depthwise separable blocks)
- [ ] Forward pass through all 28 layers
- [ ] Depthwise separable convolution implementation
- [ ] Batch normalization integration
- [ ] Weight save/load functionality
- [ ] CIFAR-10 data loading (reuse from ResNet-18)
- [ ] Inference script
- [ ] Training script structure
- [ ] Comprehensive documentation

### ⚠️ Pending (Intentional Placeholder)

- [ ] **Full backward pass**: ~2000 lines — tracked in [GitHub issue #3187](https://github.com/HomericIntelligence/ProjectOdyssey/issues/3187)
  - The training script demonstrates the structure but uses a placeholder backward pass
  - For actual training, consider using automatic differentiation

### 🔮 Future Enhancements

- [ ] Width multiplier (0.25, 0.5, 0.75, 1.0)
- [ ] Resolution multiplier
- [ ] Data augmentation integration
- [ ] SIMD optimization for depthwise convolutions
- [ ] Quantization-aware training

## Expected Performance

Based on reference implementations and similar experiments:

- **Training Time**: ~25-35 hours on CPU for 200 epochs (batch_size=128)
- **Expected Accuracy**: 90-92% on CIFAR-10 after 200 epochs
- **Peak Accuracy**: 92-94% with data augmentation
- **Memory Usage**: ~17MB for model weights

### Comparison with Other Architectures

| Model      | Parameters | Operations | CIFAR-10 Accuracy | Training Time | Key Feature                    |
|------------|------------|------------|-------------------|---------------|--------------------------------|
| LeNet-5    | 61K        | 0.4M       | 70-75%            | 2-3 hours     | Early CNN                      |
| AlexNet    | 2.3M       | 0.7B       | 80-85%            | 8-12 hours    | Large kernels, dropout         |
| VGG-16     | 15M        | 15.5B      | 91-93%            | 30-40 hours   | Very deep (16 layers)          |
| ResNet-18  | 11M        | 1.8B       | 93-94%            | 40-50 hours   | Skip connections               |
| GoogLeNet  | 6.8M       | 1.5B       | 92-94%            | 35-45 hours   | Inception modules              |
| MobileNetV1| 4.2M       | 60M        | 90-92%            | 25-35 hours   | Depthwise separable convs      |

**Why MobileNetV1 is Efficient**:

1. **Fewest operations**: 60M vs GoogLeNet's 1.5B (25× reduction!)
2. **Smallest model**: 4.2M parameters vs 15M for VGG-16
3. **Fastest training**: ~30 hours vs 40-50 hours for ResNet-18
4. **Mobile-friendly**: Designed for resource-constrained devices
5. **Reasonable accuracy**: 90-92% is competitive for the size

## Advanced Features

### Depthwise Convolution Mathematics

For depthwise convolution with input `x` (shape: [B, C, H, W]):

1. **Apply one filter per channel** (no cross-channel mixing):

   ```text
   For each channel c in C:
       output[:, c, :, :] = conv2d(input[:, c:c+1, :, :], kernel_c)
   ```text

2. **Parameters**: K × K × C (vs standard conv: K × K × C × C_out)

3. **Operations**: H × W × C × K² (vs standard: H × W × C × C_out × K²)

### Pointwise Convolution (1×1 Conv)

1. **Channel mixing without spatial filtering**:

   ```text
   output = conv2d(input, weights_1x1, kernel_size=1)
   ```text

2. **Parameters**: 1 × 1 × C_in × C_out

3. **Operations**: H × W × C_in × C_out

### Backward Pass

**Depthwise Separable Block Backward**:

1. **Pointwise backward** (standard 1×1 conv backward):
   - `grad_input, grad_weights, grad_bias = conv2d_backward(...)`

2. **Depthwise backward** (channel-wise convolution backward):
   - Apply conv2d_backward independently per channel
   - Accumulate weight gradients per channel

### Batch Normalization

Applied after both depthwise and pointwise convolutions:

```text
x_norm = (x - mean) / sqrt(var + eps)
y = gamma * x_norm + beta
```text

### Learning Rate Scheduling

Step decay schedule (same as other models):

- **Schedule**: Decay by 5× every 60 epochs
- **Formula**: `lr = initial_lr * (0.2 ** (epoch // 60))`
- **Example**:
  - Epochs 0-59: lr = 0.01
  - Epochs 60-119: lr = 0.002
  - Epochs 120-179: lr = 0.0004
  - Epochs 180+: lr = 0.00008

### Weight Initialization

**He initialization** for depthwise convolutions:

- Formula: `weights ~ N(0, sqrt(2 / fan_in))`
- fan_in = K × K × 1 (per-channel kernel)

**Xavier initialization** for pointwise convolutions:

- Formula: `weights ~ N(0, sqrt(2 / (fan_in + fan_out)))`
- Better for 1×1 convolutions

## Design Principles (KISS)

This example follows **Keep It Simple, Stupid** principles:

1. **Minimal Dependencies**: Uses only ML Odyssey shared library
2. **Functional Design**: Uses functional ops from shared/core
3. **Clear Structure**: Separate files for model, training, and inference
4. **Simple Interfaces**: Command-line arguments for configuration
5. **No Over-Engineering**: Direct implementation without unnecessary abstractions
6. **Pattern Reuse**: Follows the same structure as previous examples

## Usage Details

### Training Options

```bash
mojo run examples/mobilenetv1-cifar10/train.mojo \
    --epochs 200 \
    --batch-size 128 \
    --lr 0.01 \
    --momentum 0.9 \
    --data-dir datasets/cifar10 \
    --weights-dir mobilenetv1_weights
```text

**Arguments**:

- `--epochs`: Number of training epochs (default: 200)
- `--batch-size`: Mini-batch size (default: 128)
- `--lr`: Initial learning rate for SGD (default: 0.01)
- `--momentum`: Momentum factor for SGD (default: 0.9)
- `--data-dir`: Path to CIFAR-10 dataset directory (default: `datasets/cifar10`)
- `--weights-dir`: Directory to save model weights (default: `mobilenetv1_weights`)

### Inference Options

```bash
mojo run examples/mobilenetv1-cifar10/inference.mojo \
    --weights-dir mobilenetv1_weights \
    --data-dir datasets/cifar10
```text

**Arguments**:

- `--weights-dir`: Directory containing saved model weights (default: `mobilenetv1_weights`)
- `--data-dir`: Path to CIFAR-10 dataset for test set evaluation (default: `datasets/cifar10`)

## References

### Papers

1. **MobileNets (Original)**:
   Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., ... & Adam, H. (2017).
   MobileNets: Efficient convolutional neural networks for mobile vision applications.
   *arXiv preprint*.
   [arXiv Paper](https://arxiv.org/abs/1704.04861)

2. **Depthwise Separable Convolutions**:
   Chollet, F. (2017).
   Xception: Deep learning with depthwise separable convolutions.
   *CVPR 2017*.
   [Paper](https://arxiv.org/abs/1610.02357)

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

- **MobileNets PyTorch**: <https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py>
  - Official PyTorch implementation
  - Demonstrates depthwise separable convolution architecture

- **MobileNets TensorFlow**: <https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet>
  - TensorFlow implementation with width/resolution multipliers

### Related Resources

- **Papers with Code - MobileNets**: <https://paperswithcode.com/method/mobilenetv1>
- **MobileNetV1 Explained**: <https://towardsdatascience.com/the-tiny-giant-mobilenetv1/>
- **Depthwise Separable Convolutions**:
<https://eli.thegreenplace.net/2018/depthwise-separable-convolutions-for-machine-learning/>

## Contributing

This example is part of ML Odyssey. Contributions welcome!

**Priority Tasks**:

1. **Implement depthwise convolution** operation in shared library
2. **Complete model architecture** (13 depthwise separable blocks)
3. **Implement training script** with backward pass
4. **Add width multiplier** support (0.25, 0.5, 0.75, 1.0)
5. **Optimize depthwise convolutions** with SIMD vectorization

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

This example is part of ML Odyssey and follows the project's license.

See [LICENSE](../../LICENSE) for details.

## Acknowledgments

- **Dataset**: CIFAR (Canadian Institute for Advanced Research) for creating and releasing the CIFAR-10 dataset
- **Architecture**: Andrew G. Howard et al. for the MobileNets architecture
- **Inspiration**: Previous examples (ResNet-18, GoogLeNet, VGG-16) for establishing patterns
- **ML Odyssey**: The shared library providing functional neural network operations

## Next Steps

1. Implement depthwise convolution operation in shared library
2. Build the complete MobileNetV1 model with 13 depthwise separable blocks
3. Implement training script with backward pass through depthwise separable blocks
4. Test on CIFAR-10 dataset
5. Compare efficiency with other architectures (operations, memory, speed)
6. Experiment with width multipliers for different accuracy/efficiency tradeoffs
