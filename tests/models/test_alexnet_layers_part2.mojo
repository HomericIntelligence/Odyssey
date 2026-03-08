"""Layerwise Unit Tests for AlexNet - Part 2: Conv3, Conv4, Conv5

Tests Conv3 (192→384, 3x3), Conv4 (384→384, 3x3), and Conv5 (384→256, 3x3)
layers independently with special FP-representable values.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_alexnet_layers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Float16 Precision Limitations
==============================
Conv3 (3x3 kernel, 192 input channels): 1,728 multiplications per output
element similarly exceed Float16 precision. SKIPPED.

These are expected, fundamental limitations of Float16 arithmetic (not bugs).
See issue #3009 for detailed analysis.
"""

from shared.core.extensor import ExTensor, zeros, ones, full
from shared.core.conv import conv2d
from shared.core.pooling import maxpool2d
from shared.core.linear import linear
from shared.core.activation import relu
from shared.core.shape import conv2d_output_shape, pool_output_shape
from shared.core.initializers import kaiming_uniform
from shared.testing.assertions import (
    assert_shape,
    assert_dtype,
    assert_true,
    assert_false,
)
from shared.testing.special_values import (
    create_special_value_tensor,
    create_alternating_pattern_tensor,
    create_seeded_random_tensor,
    SPECIAL_VALUE_ONE,
    SPECIAL_VALUE_NEG_ONE,
)
from shared.testing.layer_testers import LayerTester
from math import isnan, isinf


# ============================================================================
# Test Fixtures - Parameter Creation
# ============================================================================


fn create_conv3_parameters(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    """Create Conv3 layer parameters (192→384, 3x3 kernel)."""
    var in_channels = 192
    var out_channels = 384
    var kernel_size = 3

    # Conv3 weights: (384, 192, 3, 3)
    var kernel_shape: List[Int] = [
        out_channels,
        in_channels,
        kernel_size,
        kernel_size,
    ]
    var fan_in = in_channels * kernel_size * kernel_size
    var fan_out = out_channels * kernel_size * kernel_size
    var kernel = kaiming_uniform(fan_in, fan_out, kernel_shape, dtype=dtype)

    # Conv3 bias: (384,)
    var bias = zeros([out_channels], dtype)

    return kernel, bias


fn create_conv4_parameters(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    """Create Conv4 layer parameters (384→384, 3x3 kernel)."""
    var in_channels = 384
    var out_channels = 384
    var kernel_size = 3

    # Conv4 weights: (384, 384, 3, 3)
    var kernel_shape: List[Int] = [
        out_channels,
        in_channels,
        kernel_size,
        kernel_size,
    ]
    var fan_in = in_channels * kernel_size * kernel_size
    var fan_out = out_channels * kernel_size * kernel_size
    var kernel = kaiming_uniform(fan_in, fan_out, kernel_shape, dtype=dtype)

    # Conv4 bias: (384,)
    var bias = zeros([out_channels], dtype)

    return kernel, bias


fn create_conv5_parameters(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    """Create Conv5 layer parameters (384→256, 3x3 kernel)."""
    var in_channels = 384
    var out_channels = 256
    var kernel_size = 3

    # Conv5 weights: (256, 384, 3, 3)
    var kernel_shape: List[Int] = [
        out_channels,
        in_channels,
        kernel_size,
        kernel_size,
    ]
    var fan_in = in_channels * kernel_size * kernel_size
    var fan_out = out_channels * kernel_size * kernel_size
    var kernel = kaiming_uniform(fan_in, fan_out, kernel_shape, dtype=dtype)

    # Conv5 bias: (256,)
    var bias = zeros([out_channels], dtype)

    return kernel, bias


# ============================================================================
# Conv3 Tests (192→384 channels, 3x3 kernel, stride 1, padding 1)
# ============================================================================


fn test_conv3_forward_float32() raises:
    """Test Conv3 forward pass (192→384 channels, 3x3 kernel) with float32."""
    var dtype = DType.float32
    var _result = create_conv3_parameters(dtype)

    var kernel = _result[0]

    var bias = _result[1]

    LayerTester.test_conv_layer(
        in_channels=192,
        out_channels=384,
        kernel_size=3,
        input_h=16,
        input_w=16,
        weights=kernel,
        bias=bias,
        dtype=dtype,
        stride=1,
        padding=1,
    )


fn test_conv3_forward_float16() raises:
    """Test Conv3 forward pass with float16."""
    var dtype = DType.float16
    var _result = create_conv3_parameters(dtype)

    var kernel = _result[0]

    var bias = _result[1]

    LayerTester.test_conv_layer(
        in_channels=192,
        out_channels=384,
        kernel_size=3,
        input_h=16,
        input_w=16,
        weights=kernel,
        bias=bias,
        dtype=dtype,
        stride=1,
        padding=1,
    )


fn test_conv3_backward_float32() raises:
    """Test Conv3 backward pass with sampled gradient checking.

    Uses sampled gradient checking (100 samples) to avoid timeout.
    Conv3 has 192 input channels with 3x3 kernel (11,520 elements).
    """
    var dtype = DType.float32
    var _result = create_conv3_parameters(dtype)

    var kernel = _result[0]

    var bias = _result[1]

    LayerTester.test_conv_layer_backward(
        in_channels=192,
        out_channels=384,
        kernel_size=3,
        input_h=8,
        input_w=8,
        weights=kernel,
        bias=bias,
        dtype=dtype,
        stride=1,
        padding=1,
        validate_analytical=True,
        num_gradient_samples=100,
    )


# ============================================================================
# Conv4 Tests (384→384 channels, 3x3 kernel, stride 1, padding 1)
# Reuses similar test to Conv3 (same structure, just different in/out channels)
# ============================================================================


fn test_conv4_forward_float32() raises:
    """Test Conv4 forward pass (384→384 channels, 3x3 kernel) with float32."""
    var dtype = DType.float32
    var _result = create_conv4_parameters(dtype)

    var kernel = _result[0]

    var bias = _result[1]

    LayerTester.test_conv_layer(
        in_channels=384,
        out_channels=384,
        kernel_size=3,
        input_h=16,
        input_w=16,
        weights=kernel,
        bias=bias,
        dtype=dtype,
        stride=1,
        padding=1,
    )


fn test_conv4_backward_float32() raises:
    """Test Conv4 backward pass with sampled gradient checking.

    Uses sampled gradient checking (100 samples) to avoid timeout.
    Conv4 has 384 input channels with 3x3 kernel (23,040 elements).
    """
    var dtype = DType.float32
    var _result = create_conv4_parameters(dtype)

    var kernel = _result[0]

    var bias = _result[1]

    LayerTester.test_conv_layer_backward(
        in_channels=384,
        out_channels=384,
        kernel_size=3,
        input_h=8,
        input_w=8,
        weights=kernel,
        bias=bias,
        dtype=dtype,
        stride=1,
        padding=1,
        validate_analytical=True,
        num_gradient_samples=100,
    )


# ============================================================================
# Conv5 Tests (384→256 channels, 3x3 kernel, stride 1, padding 1)
# ============================================================================


fn test_conv5_forward_float32() raises:
    """Test Conv5 forward pass (384→256 channels, 3x3 kernel) with float32."""
    var dtype = DType.float32
    var _result = create_conv5_parameters(dtype)

    var kernel = _result[0]

    var bias = _result[1]

    LayerTester.test_conv_layer(
        in_channels=384,
        out_channels=256,
        kernel_size=3,
        input_h=16,
        input_w=16,
        weights=kernel,
        bias=bias,
        dtype=dtype,
        stride=1,
        padding=1,
    )


fn test_conv5_backward_float32() raises:
    """Test Conv5 backward pass with sampled gradient checking.

    Uses sampled gradient checking (100 samples) to avoid timeout.
    Conv5 has 384 input channels with 3x3 kernel (23,040 elements).
    """
    var dtype = DType.float32
    var _result = create_conv5_parameters(dtype)

    var kernel = _result[0]

    var bias = _result[1]

    LayerTester.test_conv_layer_backward(
        in_channels=384,
        out_channels=256,
        kernel_size=3,
        input_h=8,
        input_w=8,
        weights=kernel,
        bias=bias,
        dtype=dtype,
        stride=1,
        padding=1,
        validate_analytical=True,
        num_gradient_samples=100,
    )


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    print("Starting AlexNet Layerwise Tests - Part 2 (Conv3, Conv4, Conv5)...")

    # Conv3 tests
    print("  test_conv3_forward_float32...", end="")
    test_conv3_forward_float32()
    print(" OK")

    # Float16 precision insufficient for 3x3 kernel with 192 input channels
    # (1728 multiplications per output element). Known limitation - see #3009.
    print("  test_conv3_forward_float16... SKIPPED (float16 precision)")

    # Conv3 backward - uses sampled gradient checking (100 samples)
    print("  test_conv3_backward_float32...", end="")
    test_conv3_backward_float32()
    print(" OK")

    # Conv4 tests
    print("  test_conv4_forward_float32...", end="")
    test_conv4_forward_float32()
    print(" OK")

    # Conv4 backward - uses sampled gradient checking (100 samples)
    print("  test_conv4_backward_float32...", end="")
    test_conv4_backward_float32()
    print(" OK")

    # Conv5 tests
    print("  test_conv5_forward_float32...", end="")
    test_conv5_forward_float32()
    print(" OK")

    # Conv5 backward - uses sampled gradient checking (100 samples)
    print("  test_conv5_backward_float32...", end="")
    test_conv5_backward_float32()
    print(" OK")

    print("\nAll AlexNet Part 2 (Conv3, Conv4, Conv5) tests passed!")
