"""Unit tests for 2D convolution layer - Part 3: Backward Pass and Integration Tests.

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_conv.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- conv2d_backward: Backward pass computing gradients w.r.t. input, kernel, and bias
- conv2d_no_bias_backward: Backward pass without bias
- Gradient shape verification
- Gradient value correctness for multi-channel configurations with padding=0 and padding=1
- Border pixel gradient reduction verification for padded convolutions
- Forward-backward integration checks
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from tests.shared.conftest import TestFixtures
from shared.core.extensor import AnyTensor, zeros, ones, full
from shared.core.conv import (
    conv2d,
    conv2d_no_bias,
    conv2d_backward,
    conv2d_no_bias_backward,
)


# ============================================================================
# Conv2D Backward Pass Tests
# ============================================================================


fn test_conv2d_backward_shapes() raises:
    """Test that conv2d_backward returns correct gradient shapes."""
    var batch = 1
    var in_channels = 1
    var in_height = 4
    var in_width = 4
    var out_channels = 1
    var kH = 3
    var kW = 3
    var stride = 1
    var padding = 0

    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var x = ones(input_shape, DType.float32)

    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = conv2d(x, kernel, bias, stride, padding)

    # Gradient w.r.t. output (ones_like)
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var result = conv2d_backward(grad_output, x, kernel, stride, padding)
    var grad_input = result.grad_input
    var grad_kernel = result.grad_weights
    var grad_bias = result.grad_bias

    # grad_input should match input shape
    assert_equal(grad_input.shape()[0], batch)
    assert_equal(grad_input.shape()[1], in_channels)
    assert_equal(grad_input.shape()[2], in_height)
    assert_equal(grad_input.shape()[3], in_width)

    # grad_kernel should match kernel shape
    assert_equal(grad_kernel.shape()[0], out_channels)
    assert_equal(grad_kernel.shape()[1], in_channels)
    assert_equal(grad_kernel.shape()[2], kH)
    assert_equal(grad_kernel.shape()[3], kW)

    # grad_bias should match bias shape
    assert_equal(grad_bias.shape()[0], out_channels)


fn test_conv2d_backward_bias_gradient() raises:
    """Test that conv2d_backward computes correct gradient w.r.t. bias.

    grad_bias[oc] = sum of grad_output over all (batch, height, width) positions
    for that output channel.
    """
    var batch = 2
    var in_channels = 1
    var in_height = 4
    var in_width = 4
    var out_channels = 2
    var kH = 3
    var kW = 3
    var stride = 1
    var padding = 0

    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var x = ones(input_shape, DType.float32)

    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = conv2d(x, kernel, bias, stride, padding)

    # Output shape: (2, 2, 2, 2) - batch=2, out_channels=2, out_h=2, out_w=2
    # grad_output = ones
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var result = conv2d_backward(grad_output, x, kernel, stride, padding)
    var grad_bias = result.grad_bias

    # grad_bias[oc] = sum of grad_output for that output channel
    # With ones gradient and output shape (2, 2, 2, 2):
    # Each channel has 2*2*2 = 8 output elements
    # grad_bias should be 8.0 for each channel
    var out_h = output.shape()[2]
    var out_w = output.shape()[3]
    var expected_grad_bias = Float32(batch * out_h * out_w)

    for oc in range(out_channels):
        assert_almost_equal(
            grad_bias._data.bitcast[Float32]()[oc],
            expected_grad_bias,
            tolerance=1e-4,
        )


fn test_conv2d_no_bias_backward_shapes() raises:
    """Test that conv2d_no_bias_backward returns correct gradient shapes."""
    var batch = 1
    var in_channels = 2
    var in_height = 5
    var in_width = 5
    var out_channels = 3
    var kH = 3
    var kW = 3
    var stride = 1
    var padding = 0

    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var x = ones(input_shape, DType.float32)

    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    var output = conv2d_no_bias(x, kernel, stride, padding)
    var grad_output = ones(output.shape(), DType.float32)

    var result = conv2d_no_bias_backward(
        grad_output, x, kernel, stride, padding
    )
    var grad_input = result.grad_input
    var grad_kernel = result.grad_weights

    # grad_input should match input shape
    assert_equal(grad_input.shape()[0], batch)
    assert_equal(grad_input.shape()[1], in_channels)
    assert_equal(grad_input.shape()[2], in_height)
    assert_equal(grad_input.shape()[3], in_width)

    # grad_kernel should match kernel shape
    assert_equal(grad_kernel.shape()[0], out_channels)
    assert_equal(grad_kernel.shape()[1], in_channels)
    assert_equal(grad_kernel.shape()[2], kH)
    assert_equal(grad_kernel.shape()[3], kW)


fn test_conv2d_backward_multichannel_shapes() raises:
    """Test conv2d_backward returns correct gradient shapes for multi-channel config.

    Tests in_channels=3, out_channels=8 — typical first conv layer configuration.
    Verifies grad_weights shape is (out_channels, in_channels, kH, kW) and
    grad_input shape is (batch, in_channels, H, W).
    """
    var batch = 1
    var in_channels = 3
    var out_channels = 8
    var in_height = 6
    var in_width = 6
    var kH = 3
    var kW = 3
    var stride = 1
    var padding = 1

    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var x = ones(input_shape, DType.float32)

    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass: output shape (1, 8, 6, 6) with padding=1
    var output = conv2d(x, kernel, bias, stride, padding)
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var result = conv2d_backward(grad_output, x, kernel, stride, padding)
    var grad_input = result.grad_input
    var grad_kernel = result.grad_weights
    var grad_bias = result.grad_bias

    # grad_input should match input shape: (batch, in_channels, H, W)
    assert_equal(grad_input.shape()[0], batch)
    assert_equal(grad_input.shape()[1], in_channels)
    assert_equal(grad_input.shape()[2], in_height)
    assert_equal(grad_input.shape()[3], in_width)

    # grad_weights should match kernel shape: (out_channels, in_channels, kH, kW)
    assert_equal(grad_kernel.shape()[0], out_channels)
    assert_equal(grad_kernel.shape()[1], in_channels)
    assert_equal(grad_kernel.shape()[2], kH)
    assert_equal(grad_kernel.shape()[3], kW)

    # grad_bias should match bias shape: (out_channels,)
    assert_equal(grad_bias.shape()[0], out_channels)


fn test_conv2d_backward_multichannel_values() raises:
    """Test conv2d_backward computes correct gradients for multi-channel config.

    Uses analytically tractable configuration (all ones, single spatial output)
    to verify gradient accumulation across input channels and output channels.

    Config: batch=1, in_channels=3, out_channels=8
    Input: (1, 3, 3, 3) all ones, kernel: (8, 3, 3, 3) all ones
    stride=1, padding=0 -> output shape: (1, 8, 1, 1)
    grad_output = ones((1, 8, 1, 1))

    Analytical expected values:
    - grad_weights[oc, ic, kh, kw] = sum over (batch, oh, ow) of
        grad_output[b, oc, oh, ow] * x[b, ic, oh+kh, ow+kw]
        = 1.0 * 1.0 = 1.0 for every weight position
    - grad_input[b, ic, ih, iw] = sum over oc of
        grad_output[b, oc, oh, ow] * kernel[oc, ic, kh, kw]
        = 8 output channels * 1.0 * 1.0 = 8.0 for every input position
    - grad_bias[oc] = sum over (batch, oh, ow) of grad_output[b, oc, oh, ow]
        = 1.0 (single spatial position, single batch item)
    """
    var batch = 1
    var in_channels = 3
    var out_channels = 8
    var kH = 3
    var kW = 3
    var stride = 1
    var padding = 0

    # Input: (1, 3, 3, 3) all ones
    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_channels)
    input_shape.append(3)
    input_shape.append(3)
    var x = ones(input_shape, DType.float32)

    # Kernel: (8, 3, 3, 3) all ones
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass: output shape (1, 8, 1, 1) - 3x3 input, 3x3 kernel, no padding
    var output = conv2d(x, kernel, bias, stride, padding)
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var result = conv2d_backward(grad_output, x, kernel, stride, padding)
    var grad_input = result.grad_input
    var grad_kernel = result.grad_weights
    var grad_bias = result.grad_bias

    # Verify grad_weights values: each weight gradient = 1.0
    # (single spatial output position, all ones input and grad_output)
    var n_weights = out_channels * in_channels * kH * kW
    var grad_weights_data = grad_kernel._data.bitcast[Float32]()
    for i in range(n_weights):
        assert_almost_equal(
            grad_weights_data[i],
            Float32(1.0),
            tolerance=1e-4,
        )

    # Verify grad_input values: each input gradient = 8.0
    # (8 output channels each contributing kernel=1.0 * grad_output=1.0)
    var n_inputs = batch * in_channels * 3 * 3
    var grad_input_data = grad_input._data.bitcast[Float32]()
    for i in range(n_inputs):
        assert_almost_equal(
            grad_input_data[i],
            Float32(out_channels),
            tolerance=1e-4,
        )

    # Verify grad_bias values: each bias gradient = 1.0
    # (1 batch * 1 spatial position * grad_output=1.0)
    var grad_bias_data = grad_bias._data.bitcast[Float32]()
    for oc in range(out_channels):
        assert_almost_equal(
            grad_bias_data[oc],
            Float32(1.0),
            tolerance=1e-4,
        )


fn test_conv2d_backward_multichannel_padding1_values() raises:
    """Test conv2d_backward computes correct gradient values with padding=1.

    Verifies that border pixels receive fewer gradient contributions than interior
    pixels — a correctness property unique to padded convolutions.

    Config: batch=1, in_channels=3, out_channels=8, spatial=5x5, kH=kW=3,
    stride=1, padding=1 -> output shape (1, 8, 5, 5) (same-padding).

    With all-ones input, all-ones kernel, all-ones grad_output:
    grad_input[0, ic, ih, iw] = out_channels * overlap_count(ih, iw)

    Where overlap_count is the number of output positions (oh, ow) whose
    receptive field covers (ih, iw). With padding=1 and 5x5 spatial:
    - Input row/col 0 or 4: covered by 2 output positions in that dimension
    - Input row/col 1, 2, 3: covered by 3 output positions in that dimension

    Expected grad_input values (same for all in_channels):
    - Corner (0,0): 2*2 = 4 covering outputs -> 4 * 8 = 32.0
    - Edge non-corner (e.g., 0,1): 2*3 = 6 -> 6 * 8 = 48.0
    - Interior (e.g., 1,1): 3*3 = 9 -> 9 * 8 = 72.0
    """
    var batch = 1
    var in_channels = 3
    var out_channels = 8
    var in_height = 5
    var in_width = 5
    var kH = 3
    var kW = 3
    var stride = 1
    var padding = 1

    # Input: (1, 3, 5, 5) all ones
    var input_shape = List[Int]()
    input_shape.append(batch)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var x = ones(input_shape, DType.float32)

    # Kernel: (8, 3, 3, 3) all ones
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass: output shape (1, 8, 5, 5) — same-padding preserves spatial dims
    var output = conv2d(x, kernel, bias, stride, padding)
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var result = conv2d_backward(grad_output, x, kernel, stride, padding)
    var grad_input = result.grad_input

    # Verify grad_input shape
    assert_equal(grad_input.shape()[0], batch)
    assert_equal(grad_input.shape()[1], in_channels)
    assert_equal(grad_input.shape()[2], in_height)
    assert_equal(grad_input.shape()[3], in_width)

    # Helper to index flat grad_input: shape (1, in_channels, 5, 5)
    # flat index = ic * 5*5 + ih * 5 + iw
    var grad_input_data = grad_input._data.bitcast[Float32]()

    # Corner pixel (0, 0): overlap = 2 * 2 = 4 -> expected 4 * 8 = 32.0
    var expected_corner = Float32(32.0)
    # Edge non-corner pixel (0, 1): overlap = 2 * 3 = 6 -> expected 6 * 8 = 48.0
    var expected_edge = Float32(48.0)
    # Interior pixel (1, 1): overlap = 3 * 3 = 9 -> expected 9 * 8 = 72.0
    var expected_interior = Float32(72.0)

    # Check all in_channels have the same pattern (symmetry: all-ones kernel)
    for ic in range(in_channels):
        var base = ic * in_height * in_width

        # Corner (ih=0, iw=0)
        assert_almost_equal(
            grad_input_data[base + 0 * in_width + 0],
            expected_corner,
            tolerance=1e-3,
        )

        # Edge non-corner (ih=0, iw=1)
        assert_almost_equal(
            grad_input_data[base + 0 * in_width + 1],
            expected_edge,
            tolerance=1e-3,
        )

        # Interior (ih=1, iw=1)
        assert_almost_equal(
            grad_input_data[base + 1 * in_width + 1],
            expected_interior,
            tolerance=1e-3,
        )

    # Verify border < interior: corner < edge < interior
    var corner_val = grad_input_data[0]
    var edge_val = grad_input_data[1]
    var interior_val = grad_input_data[1 * in_width + 1]
    assert_true(corner_val < edge_val)
    assert_true(edge_val < interior_val)


# ============================================================================
# Integration Tests
# ============================================================================


fn test_conv2d_forward_backward_consistency() raises:
    """Test that forward pass works correctly with various configurations.

    Verifies forward pass produces correct output shapes for batch processing.
    """
    var batch_size = 2
    var in_channels = 3
    var out_channels = 8
    var in_height = 16
    var in_width = 16
    var kH = 3
    var kW = 3
    var stride = 1
    var padding = 1

    # Create forward inputs
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var input = ones(input_shape, DType.float32)

    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kH)
    kernel_shape.append(kW)
    var kernel = ones(kernel_shape, DType.float32)

    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = conv2d(input, kernel, bias, stride, padding)

    # Verify output shape
    var out_height = (in_height + 2 * padding - kH) // stride + 1
    var out_width = (in_width + 2 * padding - kW) // stride + 1

    var output_shape = output.shape()
    assert_equal(output_shape[0], batch_size)
    assert_equal(output_shape[1], out_channels)
    assert_equal(output_shape[2], out_height)
    assert_equal(output_shape[3], out_width)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run Conv2D Part 3 tests: backward pass and integration tests."""
    print("Running Conv2D Part 3 tests (backward pass and integration)...")

    test_conv2d_backward_shapes()
    print("✓ test_conv2d_backward_shapes")

    test_conv2d_backward_bias_gradient()
    print("✓ test_conv2d_backward_bias_gradient")

    test_conv2d_no_bias_backward_shapes()
    print("✓ test_conv2d_no_bias_backward_shapes")

    test_conv2d_backward_multichannel_shapes()
    print("✓ test_conv2d_backward_multichannel_shapes")

    test_conv2d_backward_multichannel_values()
    print("✓ test_conv2d_backward_multichannel_values")

    test_conv2d_backward_multichannel_padding1_values()
    print("✓ test_conv2d_backward_multichannel_padding1_values")

    test_conv2d_forward_backward_consistency()
    print("✓ test_conv2d_forward_backward_consistency")

    print("\nAll Conv2D Part 3 tests passed!")
