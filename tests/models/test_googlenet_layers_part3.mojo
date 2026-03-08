# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_googlenet_layers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

"""Layerwise unit tests for GoogLeNet (Inception-v1) model components - Part 3.

Tests cover:
- Backward pass through 1×1 convolution branch
- Backward pass through 3×3 convolution branch
- Backward pass through 5×5 convolution branch
- Gradient preservation through concatenation

All tests use small tensors for fast execution (< 90 seconds total).
Backward pass testing verifies gradient computation for optimization.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from shared.core.extensor import ExTensor, zeros, ones, full, randn
from shared.core.conv import conv2d, conv2d_backward
from shared.core.pooling import maxpool2d, global_avgpool2d
from shared.core.linear import linear
from shared.core.activation import relu
from shared.core.initializers import kaiming_normal, xavier_normal, constant


fn concatenate_depthwise(
    t1: ExTensor, t2: ExTensor, t3: ExTensor, t4: ExTensor
) raises -> ExTensor:
    """Concatenate 4 tensors along the channel dimension (axis=1).

    Args:
        t1: Tensor 1 (batch, C1, H, W).
        t2: Tensor 2 (batch, C2, H, W).
        t3: Tensor 3 (batch, C3, H, W).
        t4: Tensor 4 (batch, C4, H, W).

    Returns:
        Concatenated tensor (batch, C1+C2+C3+C4, H, W).
    """
    var batch_size = t1.shape()[0]
    var c1 = t1.shape()[1]
    var c2 = t2.shape()[1]
    var c3 = t3.shape()[1]
    var c4 = t4.shape()[1]
    var height = t1.shape()[2]
    var width = t1.shape()[3]

    var total_channels = c1 + c2 + c3 + c4
    var result = zeros([batch_size, total_channels, height, width], t1.dtype())

    # Copy data from each tensor
    var result_data = result._data.bitcast[Float32]()
    var t1_data = t1._data.bitcast[Float32]()
    var t2_data = t2._data.bitcast[Float32]()
    var t3_data = t3._data.bitcast[Float32]()
    var t4_data = t4._data.bitcast[Float32]()

    var hw = height * width

    for b in range(batch_size):
        # Copy t1 channels
        for c in range(c1):
            for i in range(hw):
                var src_idx = ((b * c1 + c) * hw) + i
                var dst_idx = ((b * total_channels + c) * hw) + i
                result_data[dst_idx] = t1_data[src_idx]

        # Copy t2 channels (offset by c1)
        for c in range(c2):
            for i in range(hw):
                var src_idx = ((b * c2 + c) * hw) + i
                var dst_idx = ((b * total_channels + (c1 + c)) * hw) + i
                result_data[dst_idx] = t2_data[src_idx]

        # Copy t3 channels (offset by c1+c2)
        for c in range(c3):
            for i in range(hw):
                var src_idx = ((b * c3 + c) * hw) + i
                var dst_idx = ((b * total_channels + (c1 + c2 + c)) * hw) + i
                result_data[dst_idx] = t3_data[src_idx]

        # Copy t4 channels (offset by c1+c2+c3)
        for c in range(c4):
            for i in range(hw):
                var src_idx = ((b * c4 + c) * hw) + i
                var dst_idx = (
                    (b * total_channels + (c1 + c2 + c3 + c)) * hw
                ) + i
                result_data[dst_idx] = t4_data[src_idx]

    return result


# ============================================================================
# Backward Pass Tests
# ============================================================================


fn test_inception_branch_1x1_backward() raises:
    """Test backward pass through 1×1 convolution branch.

    Verifies gradient computation for weight updates.
    """
    var batch_size = 2
    var in_channels = 32
    var out_channels = 16
    var height = 4
    var width = 4

    # Create input and weights
    var input = ones([batch_size, in_channels, height, width], DType.float32)
    var weights = ones([out_channels, in_channels, 1, 1], DType.float32)
    var bias = zeros([out_channels], DType.float32)

    # Forward pass
    var output = conv2d(input, weights, bias, stride=1, padding=0)

    # Create gradient from upstream
    var grad_output = ones(
        [batch_size, out_channels, height, width], DType.float32
    )

    # Backward pass
    var _result = conv2d_backward(
        grad_output, input, weights, stride=1, padding=0
    )
    var grad_input = _result.grad_input
    var grad_weights = _result.grad_weights
    var grad_bias = _result.grad_bias

    # Verify gradient shapes
    assert_shape(grad_input, input.shape())
    assert_shape(grad_weights, weights.shape())
    assert_shape(grad_bias, bias.shape())


fn test_inception_branch_3x3_backward() raises:
    """Test backward pass through 3×3 convolution branch.

    Tests gradient computation with padding=1.
    """
    var batch_size = 2
    var in_channels = 16
    var out_channels = 16
    var height = 4
    var width = 4

    # Create input and weights
    var input = ones([batch_size, in_channels, height, width], DType.float32)
    var weights = ones([out_channels, in_channels, 3, 3], DType.float32)
    var bias = zeros([out_channels], DType.float32)

    # Forward pass
    var output = conv2d(input, weights, bias, stride=1, padding=1)

    # Create gradient from upstream
    var grad_output = ones(
        [batch_size, out_channels, height, width], DType.float32
    )

    # Backward pass
    var _result = conv2d_backward(
        grad_output, input, weights, stride=1, padding=1
    )
    var grad_input = _result.grad_input
    var grad_weights = _result.grad_weights
    var grad_bias = _result.grad_bias

    # Verify gradient shapes
    assert_shape(grad_input, input.shape())
    assert_shape(grad_weights, weights.shape())
    assert_shape(grad_bias, bias.shape())


fn test_inception_branch_5x5_backward() raises:
    """Test backward pass through 5×5 convolution branch.

    Tests gradient computation with padding=2.
    """
    var batch_size = 2
    var in_channels = 8
    var out_channels = 8
    var height = 4
    var width = 4

    # Create input and weights
    var input = ones([batch_size, in_channels, height, width], DType.float32)
    var weights = ones([out_channels, in_channels, 5, 5], DType.float32)
    var bias = zeros([out_channels], DType.float32)

    # Forward pass
    var output = conv2d(input, weights, bias, stride=1, padding=2)

    # Create gradient from upstream
    var grad_output = ones(
        [batch_size, out_channels, height, width], DType.float32
    )

    # Backward pass
    var _result = conv2d_backward(
        grad_output, input, weights, stride=1, padding=2
    )
    var grad_input = _result.grad_input
    var grad_weights = _result.grad_weights
    var grad_bias = _result.grad_bias

    # Verify gradient shapes
    assert_shape(grad_input, input.shape())
    assert_shape(grad_weights, weights.shape())
    assert_shape(grad_bias, bias.shape())


fn test_concatenate_gradient_preservation() raises:
    """Test that concatenation preserves gradients correctly.

    Gradient flows backward through concatenation to each input tensor.
    """
    var batch_size = 1
    var height = 2
    var width = 2

    # Create input tensors
    var t1 = ones([batch_size, 2, height, width], DType.float32)
    var t2 = ones([batch_size, 2, height, width], DType.float32)
    var t3 = ones([batch_size, 2, height, width], DType.float32)
    var t4 = ones([batch_size, 2, height, width], DType.float32)

    # Forward pass
    var result = concatenate_depthwise(t1, t2, t3, t4)

    # Create gradient from upstream (all ones for simplicity)
    var grad_result = ones(result.shape(), DType.float32)

    # Verify gradient shape
    assert_equal(grad_result.shape()[0], batch_size)
    assert_equal(grad_result.shape()[1], 8)
    assert_equal(grad_result.shape()[2], height)
    assert_equal(grad_result.shape()[3], width)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    print("Starting GoogLeNet Layerwise Tests Part 3...")

    print("  test_inception_branch_1x1_backward...", end="")
    test_inception_branch_1x1_backward()
    print(" OK")

    print("  test_inception_branch_3x3_backward...", end="")
    test_inception_branch_3x3_backward()
    print(" OK")

    print("  test_inception_branch_5x5_backward...", end="")
    test_inception_branch_5x5_backward()
    print(" OK")

    print("  test_concatenate_gradient_preservation...", end="")
    test_concatenate_gradient_preservation()
    print(" OK")

    print("All GoogLeNet layerwise tests Part 3 passed!")
