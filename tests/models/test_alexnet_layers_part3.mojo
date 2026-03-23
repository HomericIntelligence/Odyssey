"""Layerwise Unit Tests for AlexNet - Part 3: ReLU and MaxPool1/MaxPool2

Tests ReLU activation and MaxPool1 (64 channels) and MaxPool2 (192 channels)
layers independently with special FP-representable values.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_alexnet_layers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from shared.tensor.any_tensor import AnyTensor, zeros, ones, full
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
# ReLU Tests (single test covers all, reused from LeNet-5 pattern)
# ============================================================================


fn test_relu_forward_float32() raises:
    """Test ReLU activation forward pass with float32."""
    var dtype = DType.float32
    var shape: List[Int] = [2, 256, 8, 8]

    LayerTester.test_activation_layer(shape, dtype, activation="relu")


fn test_relu_forward_float16() raises:
    """Test ReLU activation forward pass with float16."""
    var dtype = DType.float16
    var shape: List[Int] = [2, 256, 8, 8]

    LayerTester.test_activation_layer(shape, dtype, activation="relu")


fn test_relu_backward_float32() raises:
    """Test ReLU backward pass with gradient checking."""
    var dtype = DType.float32
    var shape: List[Int] = [2, 256, 4, 4]

    LayerTester.test_activation_layer_backward(shape, dtype, activation="relu")


# ============================================================================
# MaxPool Tests (3x3, stride 2)
# ============================================================================


fn test_maxpool1_forward_float32() raises:
    """Test MaxPool1 (3x3, stride 2) forward pass with float32."""
    var dtype = DType.float32

    LayerTester.test_pooling_layer(
        channels=64,
        input_h=24,
        input_w=24,
        pool_size=3,
        stride=2,
        dtype=dtype,
        pool_type="max",
        padding=0,
    )


fn test_maxpool1_forward_float16() raises:
    """Test MaxPool1 forward pass with float16."""
    var dtype = DType.float16

    LayerTester.test_pooling_layer(
        channels=64,
        input_h=24,
        input_w=24,
        pool_size=3,
        stride=2,
        dtype=dtype,
        pool_type="max",
        padding=0,
    )


fn test_maxpool2_forward_float32() raises:
    """Test MaxPool2 (3x3, stride 2) forward pass with float32."""
    var dtype = DType.float32

    LayerTester.test_pooling_layer(
        channels=192,
        input_h=16,
        input_w=16,
        pool_size=3,
        stride=2,
        dtype=dtype,
        pool_type="max",
        padding=0,
    )


fn test_maxpool2_forward_float16() raises:
    """Test MaxPool2 forward pass with float16."""
    var dtype = DType.float16

    LayerTester.test_pooling_layer(
        channels=192,
        input_h=16,
        input_w=16,
        pool_size=3,
        stride=2,
        dtype=dtype,
        pool_type="max",
        padding=0,
    )


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    print(
        "Starting AlexNet Layerwise Tests - Part 3 (ReLU, MaxPool1,"
        " MaxPool2)..."
    )

    # ReLU tests
    print("  test_relu_forward_float32...", end="")
    test_relu_forward_float32()
    print(" OK")

    print("  test_relu_forward_float16...", end="")
    test_relu_forward_float16()
    print(" OK")

    print("  test_relu_backward_float32...", end="")
    test_relu_backward_float32()
    print(" OK")

    # MaxPool1 tests
    print("  test_maxpool1_forward_float32...", end="")
    test_maxpool1_forward_float32()
    print(" OK")

    print("  test_maxpool1_forward_float16...", end="")
    test_maxpool1_forward_float16()
    print(" OK")

    # MaxPool2 tests
    print("  test_maxpool2_forward_float32...", end="")
    test_maxpool2_forward_float32()
    print(" OK")

    print("  test_maxpool2_forward_float16...", end="")
    test_maxpool2_forward_float16()
    print(" OK")

    print("\nAll AlexNet Part 3 (ReLU, MaxPool1, MaxPool2) tests passed!")
