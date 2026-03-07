# LeNet-5 on EMNIST Example

A simple implementation of LeNet-5 convolutional neural network for EMNIST character recognition, following KISS principles.

## Overview

This example demonstrates how to use ML Odyssey's shared library to build, train, and run inference with the classic
LeNet-5 architecture on the EMNIST dataset.

**Architecture**: LeNet-5 (LeCun et al., 1998)

**Dataset**: EMNIST Balanced (47 classes: digits 0-9, uppercase A-Z, and select lowercase letters)

**Status**: ✅ **Fully Functional** - Complete implementation working on Mojo 0.26.1 with training and inference
achieving 81% test accuracy on EMNIST Balanced (47 classes).

## Quick Start

### 1. Download Dataset

```bash
python scripts/download_emnist.py --split balanced
```text

This downloads the EMNIST Balanced split (131,600 training samples, 18,800 test samples, 47 classes) to
`datasets/emnist/`.

### 2. Train Model

```bash
pixi run mojo run -I . examples/lenet-emnist/train.mojo --epochs 10 --batch-size 32 --lr 0.01
```text

**Note**: Use `pixi run mojo` (not just `mojo`) since Mojo is installed via pixi. The `-I .` flag includes the
current directory in the module search path.

### 3. Run Inference

```bash
# Evaluate on test set
pixi run mojo run -I . examples/lenet-emnist/inference.mojo \
    --weights-dir lenet5_weights \
    --data-dir datasets/emnist
```text

## Dataset Information

### EMNIST Dataset

The EMNIST dataset is an extension of MNIST that includes handwritten letters in addition to digits.

- **Source**: NIST Special Database 19
- **Format**: 28×28 grayscale images
- **File Format**: IDX (same as MNIST) and MATLAB

### Available Splits

| Split        | Samples | Classes | Description                         |
|--------------|---------|---------|-------------------------------------|
| **Balanced** | 131,600 | 47      | Recommended - balanced distribution |
| ByClass      | 814,255 | 62      | Unbalanced, all characters          |
| ByMerge      | 814,255 | 47      | Unbalanced, merged similar chars    |
| Digits       | 280,000 | 10      | Digits only (0-9)                   |
| Letters      | 145,600 | 26      | Uppercase letters only (A-Z)        |
| MNIST        | 70,000  | 10      | Original MNIST for comparison       |

**Default**: This example uses the **Balanced** split (47 classes).

### EMNIST Balanced Classes

- **Digits (10)**: 0-9
- **Uppercase Letters (26)**: A-Z
- **Lowercase Letters (11)**: a, b, d, e, f, g, h, n, q, r, t

Note: Only select lowercase letters are included to reduce confusion with similar-looking uppercase letters.

## Model Architecture

### LeNet-5

The classic convolutional neural network architecture introduced by LeCun et al. (1998).

```text
Input (28×28×1)
    ↓
Conv2D (6 filters, 5×5) + ReLU
    ↓
MaxPool (2×2, stride=2)
    ↓
Conv2D (16 filters, 5×5) + ReLU
    ↓
MaxPool (2×2, stride=2)
    ↓
Flatten (16×4×4 = 256)
    ↓
Linear (256 → 120) + ReLU
    ↓
Linear (120 → 84) + ReLU
    ↓
Linear (84 → 47)
    ↓
Output (47 classes)
```text

### Parameters

- **Input Shape**: (batch, 1, 28, 28)
- **Output Shape**: (batch, 47)
- **Total Parameters**: ~61,706
  - Conv1: 6×(1×5×5+1) = 156
  - Conv2: 16×(6×5×5+1) = 2,416
  - FC1: 256×120+120 = 30,840
  - FC2: 120×84+84 = 10,164
  - FC3: 84×47+47 = 3,995

## File Structure

```text
examples/lenet-emnist/
├── README.md           # This file
├── model.mojo          # LeNet-5 model with save/load
├── train.mojo          # Training with manual backward passes
├── inference.mojo      # Inference with weight loading
├── run_infer.mojo      # Inference entry point for running predictions on test data
├── weights.mojo        # Hex-based weight serialization
└── run_example.sh      # Complete workflow script
```text

## Usage Details

### Training Options

```bash
pixi run mojo run -I . examples/lenet-emnist/train.mojo \
    --epochs 10 \
    --batch-size 32 \
    --lr 0.01 \
    --data-dir datasets/emnist \
    --weights-dir lenet5_weights
```text

**Arguments**:

- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Mini-batch size (default: 32)
- `--lr`: Learning rate for SGD (default: 0.01)
- `--data-dir`: Path to EMNIST dataset directory (default: `datasets/emnist`)
- `--weights-dir`: Directory to save model weights (default: `lenet5_weights`)

### Inference Options

```bash
# Test set evaluation
pixi run mojo run -I . examples/lenet-emnist/inference.mojo \
    --weights-dir lenet5_weights \
    --data-dir datasets/emnist
```text

**Important**: The `-I .` flag is **required** to include the current directory in Mojo's module search path. Without it, Mojo cannot find the `shared/` library modules.

**Arguments**:

- `--weights-dir`: Directory containing saved model weights (default: `lenet5_weights`)
- `--data-dir`: Path to EMNIST dataset for test set evaluation (default: `datasets/emnist`)

## Implementation Status

### ✅ Completed

- [x] Dataset download script with IDX format support (Python)
- [x] LeNet-5 model architecture with functional ops
- [x] IDX file loading via shared.data module (replace local data_loader.mojo)
- [x] Manual backward pass implementation (no autograd)
- [x] Hex-based weight serialization/deserialization
- [x] Training loop with SGD optimizer
- [x] Inference script with weight loading
- [x] Tensor slicing for mini-batch processing
- [x] Full train → save → load → inference workflow
- [x] Comprehensive documentation
- [x] Migrate to shared data loaders (normalize_images, one_hot_encode)

### 🔄 Future Optimizations

- [ ] SIMD vectorization for operations
- [ ] Multi-threading for data loading
- [ ] Memory-mapped file I/O for large datasets
- [ ] Learning rate scheduling
- [ ] Data augmentation

### Mojo 0.26.1 Migration

All code has been successfully migrated to Mojo 0.26.1 and is fully functional:

- ✅ All 61 files updated for Mojo 0.26.1 compatibility
- ✅ Fixed parameter conventions (`inout` → `mut`/`out`)
- ✅ Updated collections API (`DynamicVector` → `List`)
- ✅ Fixed memory management (`UnsafePointer`, ownership)
- ✅ Fixed List[Int] constructor bugs (transpose, broadcasting, shape ops)
- ✅ Fixed broadcasting arithmetic operations
- ✅ Training completes successfully (81% test accuracy)
- ✅ Inference works correctly on full test set (18,800 samples)

## Expected Performance

Based on the reference implementation and similar experiments:

- **Training Time**: ~2-3 hours on CPU for 20 epochs (batch_size=256)
- **Expected Accuracy**: 93-95% on EMNIST Balanced after 10 epochs
- **Peak Accuracy**: 98%+ on EMNIST Digits (easier task)

## Design Principles (KISS)

This example follows **Keep It Simple, Stupid** principles:

1. **Minimal Dependencies**: Uses only ML Odyssey shared library
2. **Functional Design**: Model uses functional ops (conv2d, linear, relu) from shared/core
3. **Clear Structure**: Separate files for model, training, and inference
4. **Simple Interfaces**: Command-line arguments for configuration
5. **No Over-Engineering**: Direct implementation without unnecessary abstractions

## References

### Papers

1. **LeNet-5 (Original)**:
   LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).
   Gradient-based learning applied to document recognition.
   *Proceedings of the IEEE*, 86(11), 2278-2324.
   [DOI: 10.1109/5.726791](https://doi.org/10.1109/5.726791)

2. **EMNIST Dataset**:
   Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017).
   EMNIST: an extension of MNIST to handwritten letters.
   *arXiv preprint arXiv:1702.05373v1*.
   [arXiv:1702.05373](https://arxiv.org/abs/1702.05373)

### Datasets

- **EMNIST Official Page**: <https://www.nist.gov/itl/products-and-services/emnist-dataset>
- **Download**: <https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip>

### Reference Implementations

- **LeNet from Scratch (NumPy)**: <https://github.com/mattwang44/LeNet-from-Scratch>
  - Pure NumPy implementation without deep learning frameworks
  - Demonstrates manual backpropagation
  - Achieves 98.6% accuracy on MNIST

### Related Resources

- **Original MNIST**: <http://yann.lecun.com/exdb/mnist/>
- **LeNet-5 Architecture**: <http://yann.lecun.com/exdb/lenet/>

## Converting Custom Images

The `--image` flag in `run_infer.mojo` expects IDX format. Use the provided conversion script to
convert any PNG or JPEG image:

```bash
python scripts/convert_image_to_idx.py my_digit.png my_digit.idx
pixi run mojo run -I . examples/lenet-emnist/run_infer.mojo \
    --checkpoint lenet5_weights \
    --image my_digit.idx
```

The script converts to 28×28 grayscale and applies the EMNIST transpose+flip transform required
by LeNet-5 weights trained on EMNIST. Use `--no-emnist-transform` for models trained without it.

```bash
# Show all options
python scripts/convert_image_to_idx.py --help
```

## Image Loading Limitations

PNG/JPEG image loading is not natively supported in Mojo v0.26.1 (no stdlib image IO).
The `scripts/convert_image_to_idx.py` script provides a Python interop wrapper that handles this
via Pillow. See [#3198](https://github.com/homericintelligence/projectodyssey/issues/3198)
and [#3087](https://github.com/homericintelligence/projectodyssey/issues/3087).

## Contributing

This example is part of ML Odyssey. Contributions welcome!

1. Complete pending TODOs when Mojo stdlib stabilizes
2. Optimize performance (SIMD, parallelization)
3. Add data augmentation
4. Implement learning rate scheduling
5. Add visualization tools

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

This example is part of ML Odyssey and follows the project's license.

See [LICENSE](../../LICENSE) for details.

## Acknowledgments

- **Dataset**: NIST for creating and releasing the EMNIST dataset
- **Architecture**: Yann LeCun et al. for the LeNet-5 architecture
- **Reference Implementation**: Matt Wang for the clear NumPy reference implementation
- **ML Odyssey**: The shared library providing functional neural network operations
