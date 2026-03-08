"""Layerwise Unit Tests for AlexNet - Part 4: MaxPool3, FC1, FC2

Tests MaxPool3 (256 channels), FC1 (9216→4096), and FC2 (4096→4096) layers
independently with special FP-representable values.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_alexnet_layers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
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


fn create_fc1_parameters(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    """Create FC1 layer parameters (9216→4096)."""
    var in_features = 9216  # 256 * 6 * 6
    var out_features = 4096

    # FC1 weights: (4096, 9216)
    var weights_shape: List[Int] = [out_features, in_features]
    var weights = kaiming_uniform(
        in_features, out_features, weights_shape, dtype=dtype
    )

    # FC1 bias: (4096,)
    var bias = zeros([out_features], dtype)

    return weights, bias


fn create_fc2_parameters(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    """Create FC2 layer parameters (4096→4096)."""
    var in_features = 4096
    var out_features = 4096

    # FC2 weights: (4096, 4096)
    var weights_shape: List[Int] = [out_features, in_features]
    var weights = kaiming_uniform(
        in_features, out_features, weights_shape, dtype=dtype
    )

    # FC2 bias: (4096,)
    var bias = zeros([out_features], dtype)

    return weights, bias


# ============================================================================
# MaxPool3 Tests (3x3, stride 2)
# ============================================================================


fn test_maxpool3_forward_float32() raises:
    """Test MaxPool3 (3x3, stride 2) forward pass with float32."""
    var dtype = DType.float32

    LayerTester.test_pooling_layer(
        channels=256,
        input_h=13,
        input_w=13,
        pool_size=3,
        stride=2,
        dtype=dtype,
        pool_type="max",
        padding=0,
    )


fn test_maxpool3_forward_float16() raises:
    """Test MaxPool3 forward pass with float16."""
    var dtype = DType.float16

    LayerTester.test_pooling_layer(
        channels=256,
        input_h=13,
        input_w=13,
        pool_size=3,
        stride=2,
        dtype=dtype,
        pool_type="max",
        padding=0,
    )


# ============================================================================
# FC1 (Linear) Tests (9216→4096)
# ============================================================================


fn test_fc1_forward_float32() raises:
    """Test FC1 (9216→4096) forward pass with float32."""
    var dtype = DType.float32
    var _result = create_fc1_parameters(dtype)

    var weights = _result[0]

    var bias = _result[1]

    LayerTester.test_linear_layer(
        in_features=9216,
        out_features=4096,
        weights=weights,
        bias=bias,
        dtype=dtype,
    )


fn test_fc1_forward_float16() raises:
    """Test FC1 forward pass with float16."""
    var dtype = DType.float16
    var _result = create_fc1_parameters(dtype)

    var weights = _result[0]

    var bias = _result[1]

    LayerTester.test_linear_layer(
        in_features=9216,
        out_features=4096,
        weights=weights,
        bias=bias,
        dtype=dtype,
    )


fn test_fc1_backward_float32() raises:
    """Test FC1 backward pass with sampled gradient checking.

    Uses sampled gradient checking (30 samples) to avoid timeout.
    FC1 has 9,216 inputs, making exhaustive checking too slow.
    30 samples provides 95% statistical confidence while completing in ~54s.
    """
    var dtype = DType.float32
    var _result = create_fc1_parameters(dtype)

    var weights = _result[0]

    var bias = _result[1]

    LayerTester.test_linear_layer_backward(
        in_features=9216,
        out_features=4096,
        weights=weights,
        bias=bias,
        dtype=dtype,
        validate_analytical=True,
        num_gradient_samples=30,
    )


# ============================================================================
# FC2 (Linear) Tests (4096→4096)
# ============================================================================


fn test_fc2_forward_float32() raises:
    """Test FC2 (4096→4096) forward pass with float32."""
    var dtype = DType.float32
    var _result = create_fc2_parameters(dtype)

    var weights = _result[0]

    var bias = _result[1]

    LayerTester.test_linear_layer(
        in_features=4096,
        out_features=4096,
        weights=weights,
        bias=bias,
        dtype=dtype,
    )


fn test_fc2_forward_float16() raises:
    """Test FC2 forward pass with float16."""
    var dtype = DType.float16
    var _result = create_fc2_parameters(dtype)

    var weights = _result[0]

    var bias = _result[1]

    LayerTester.test_linear_layer(
        in_features=4096,
        out_features=4096,
        weights=weights,
        bias=bias,
        dtype=dtype,
    )


fn test_fc2_backward_float32() raises:
    """Test FC2 backward pass with sampled gradient checking.

    Uses sampled gradient checking (30 samples) to avoid timeout.
    FC2 has 4,096 inputs, making exhaustive checking too slow.
    30 samples provides 95% statistical confidence while completing in ~30s.
    """
    var dtype = DType.float32
    var _result = create_fc2_parameters(dtype)

    var weights = _result[0]

    var bias = _result[1]

    LayerTester.test_linear_layer_backward(
        in_features=4096,
        out_features=4096,
        weights=weights,
        bias=bias,
        dtype=dtype,
        validate_analytical=True,
        num_gradient_samples=30,
    )


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    print("Starting AlexNet Layerwise Tests - Part 4 (MaxPool3, FC1, FC2)...")

    # MaxPool3 tests
    print("  test_maxpool3_forward_float32...", end="")
    test_maxpool3_forward_float32()
    print(" OK")

    print("  test_maxpool3_forward_float16...", end="")
    test_maxpool3_forward_float16()
    print(" OK")

    # FC1 tests
    print("  test_fc1_forward_float32...", end="")
    test_fc1_forward_float32()
    print(" OK")

    print("  test_fc1_forward_float16...", end="")
    test_fc1_forward_float16()
    print(" OK")

    # FC1 backward - uses sampled gradient checking (30 samples)
    print("  test_fc1_backward_float32...", end="")
    test_fc1_backward_float32()
    print(" OK")

    # FC2 tests
    print("  test_fc2_forward_float32...", end="")
    test_fc2_forward_float32()
    print(" OK")

    print("  test_fc2_forward_float16...", end="")
    test_fc2_forward_float16()
    print(" OK")

    # FC2 backward - uses sampled gradient checking (30 samples)
    print("  test_fc2_backward_float32...", end="")
    test_fc2_backward_float32()
    print(" OK")

    print("\nAll AlexNet Part 4 (MaxPool3, FC1, FC2) tests passed!")
