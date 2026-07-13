# ResNet-18 Implementation Gap Analysis

This document identifies the gaps between the current ML Odyssey shared library and the requirements for a fully
functional ResNet-18 implementation.

## Status

- ✅ **Forward pass**: Complete
- ❌ **Backward pass**: Blocked by missing batch normalization backward
- ✅ **Data loading**: Complete (reuses existing CIFAR-10 loader)
- ✅ **Weight serialization**: Complete
- ✅ **Inference**: Complete

## Critical Missing Components

### 1. Batch Normalization Backward Pass

**Status**: ❌ **CRITICAL** - Blocks all training

**Location**: `src/odyssey/core/normalization.mojo`

**Required Function**:

```mojo
fn batch_norm2d_backward(
    grad_output: AnyTensor,
    x: AnyTensor,
    gamma: AnyTensor,
    beta: AnyTensor,
    running_mean: AnyTensor,
    running_var: AnyTensor,
    training: Bool = True,
    momentum: Float64 = 0.1,
    eps: Float64 = 1e-5
) raises -> (AnyTensor, AnyTensor, AnyTensor):
    """Backward pass for 2D batch normalization.

    Args:
        grad_output: Gradient w.r.t. output (batch, channels, height, width)
        x: Original input tensor (batch, channels, height, width)
        gamma: Scale parameter (channels,)
        beta: Shift parameter (channels,) - unused in backward
        running_mean: Running mean (channels,) - for inference mode
        running_var: Running variance (channels,) - for inference mode
        training: Whether in training mode
        momentum: Momentum for running stats (unused in backward)
        eps: Small constant for numerical stability

    Returns:
        Tuple of (grad_input, grad_gamma, grad_beta)
        - grad_input: Gradient w.r.t. input (batch, channels, height, width)
        - grad_gamma: Gradient w.r.t. gamma (channels,)
        - grad_beta: Gradient w.r.t. beta (channels,)

    Mathematical Formulation:
        During training, batch norm computes:
            mean = E[x] over batch and spatial dims
            var = Var[x] over batch and spatial dims
            x_norm = (x - mean) / sqrt(var + eps)
            y = gamma * x_norm + beta

        Gradients (chain rule):
            grad_beta = sum(grad_output) over batch and spatial
            grad_gamma = sum(grad_output * x_norm) over batch and spatial

            grad_x_norm = grad_output * gamma
            grad_var = sum(grad_x_norm * (x - mean) * -0.5 * (var + eps)^(-3/2))
            grad_mean = sum(grad_x_norm * -1/sqrt(var + eps)) +
                        grad_var * mean(-2(x - mean))

            grad_input = grad_x_norm / sqrt(var + eps) +
                         grad_var * 2(x - mean) / N +
                         grad_mean / N

        During inference:
            x_norm = (x - running_mean) / sqrt(running_var + eps)
            y = gamma * x_norm + beta

            grad_input = grad_output * gamma / sqrt(running_var + eps)
            grad_gamma = sum(grad_output * x_norm)
            grad_beta = sum(grad_output)

    Reference Implementation:
        PyTorch: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Normalization.cpp
        Paper: Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training
               by Reducing Internal Covariate Shift", 2015
               https://arxiv.org/abs/1502.03167
    """
```text

**Implementation Complexity**: High

**Key Challenges**:

1. **Statistics computation**: Computing batch statistics (mean, variance) over (batch, height, width) dimensions
2. **Gradient flow**: Proper chain rule through normalization operation
3. **Numerical stability**: Handling division by sqrt(variance + epsilon)
4. **Mode switching**: Different behavior for training vs inference mode
5. **Dimension handling**: Broadcasting gradients correctly across channels

**Testing Requirements**:

- Gradient check against numerical gradients
- Test with different input shapes (various batch sizes, spatial dims)
- Test training mode vs inference mode
- Verify gradient stability near zero variance

**Impact**:

- **Blocks**: All ResNet-18 training (100% of training functionality)
- **Affects**: Any architecture using batch normalization
- **Priority**: **CRITICAL** - This is the only blocker for ResNet-18

## Available Components (Already Implemented)

### Core Operations

- ✅ `conv2d` and `conv2d_backward` - Convolution layers
- ✅ `linear` and `linear_backward` - Fully connected layers
- ✅ `avgpool2d` and `avgpool2d_backward` - Average pooling
- ✅ `relu` and `relu_backward` - ReLU activation
- ✅ `add` and `add_backward` - Element-wise addition (for skip connections)
- ✅ `batch_norm2d` - Forward pass for batch normalization
- ✅ `cross_entropy` and `cross_entropy_backward` - Loss function

### Optimizers

- ✅ `sgd_momentum_update_inplace` - SGD with momentum

### Data Utilities

- ✅ `extract_batch_pair` - Mini-batch extraction
- ✅ `compute_num_batches` - Batch count computation
- ✅ CIFAR-10 data loader with RGB normalization

### Weight Serialization

- ✅ `save_tensor` and `load_tensor` - Hex-based weight files

## Implementation Plan

### Phase 1: Implement Batch Normalization Backward (**CRITICAL**)

1. **Add function to `src/odyssey/core/normalization.mojo`**:
   - Implement `batch_norm2d_backward` with full training mode support
   - Implement inference mode (simpler - just rescale gradients)
   - Add comprehensive docstring with mathematical formulation

2. **Add tests**:
   - Create `tests/odyssey/core/test_normalization_backward.mojo`
   - Gradient check against numerical gradients
   - Test various input shapes and modes
   - Test edge cases (zero variance, small batch)

3. **Export in public API**:
   - Update `src/odyssey/core/__init__.mojo` to include backward function

### Phase 2: Complete Training Script

Once `batch_norm2d_backward` is available:

1. **Implement full backward pass in `train.mojo`**:
   - Backward through FC layer
   - Backward through global average pooling
   - Backward through all 4 stages with residual blocks:
     - Stage 4: 2 blocks (512 channels)
     - Stage 3: 2 blocks (256 channels)
     - Stage 2: 2 blocks (128 channels)
     - Stage 1: 2 blocks (64 channels)
   - Backward through initial conv + BN + ReLU

2. **Implement gradient accumulation for all 84 parameters**:
   - 6 params: Initial conv + BN
   - 16 params: Stage 1 (2 blocks, no projection)
   - 20 params: Stage 2 (2 blocks, block1 has projection)
   - 20 params: Stage 3 (2 blocks, block1 has projection)
   - 20 params: Stage 4 (2 blocks, block1 has projection)
   - 2 params: FC layer

3. **Implement momentum updates**:
   - Initialize 84 velocity tensors
   - Apply SGD with momentum to all parameters

4. **Add learning rate scheduling**:
   - Step decay: multiply by 0.2 every 60 epochs
   - Starting LR: 0.01

### Phase 3: Validation and Testing

1. **Verify forward pass**:
   - Test with random inputs
   - Verify output shapes at each stage

2. **Verify backward pass** (when implemented):
   - Check gradient shapes match parameter shapes
   - Verify gradients are non-zero and finite
   - Test with small toy example

3. **Train small model**:
   - Train for 1-2 epochs on subset of data
   - Verify loss decreases
   - Verify accuracy improves

## References

### Batch Normalization

- **Original Paper**: Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by
Reducing Internal Covariate Shift. *ICML 2015*. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)

- **PyTorch Implementation**:
[Normalization.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Normalization.cpp)

- **Gradient Derivation**: [Understanding the Backward Pass through Batch Normalization
Layer](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)

### ResNet Architecture

- **Original Paper**: He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. *CVPR
2016*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

## Next Steps

1. **Implement `batch_norm2d_backward`** in `src/odyssey/core/normalization.mojo`
2. **Add comprehensive tests** for batch norm backward
3. **Complete training script** backward pass
4. **Validate** with small-scale training run
5. **Document** final implementation and performance results

## Summary

ResNet-18 implementation is **98% complete**. Only one critical component is missing:

- ❌ `batch_norm2d_backward` (blocks all training)

Once this function is implemented, ResNet-18 will be fully functional and can be trained on CIFAR-10 to achieve ~94%
accuracy (expected based on literature).
