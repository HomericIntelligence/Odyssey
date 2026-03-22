# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_heap_corruption_combined.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Heap corruption split part 1: Conv1, Conv2, and ReLU (forward) tests.

Split from test_heap_corruption_combined.mojo (ADR-009).
Contains 8 fn test_ functions (limit: 10).
"""

from shared.core.any_tensor import AnyTensor
from shared.core.conv import conv2d
from shared.core.activation import relu
from shared.testing.layer_params import ConvFixture
from shared.testing.assertions import (
    assert_shape,
    assert_dtype,
)
from shared.testing.layer_testers import LayerTester


# ============================================================================
# Fixtures
# ============================================================================


fn create_conv1_parameters(dtype: DType) raises -> Tuple[AnyTensor, AnyTensor]:
    """Create Conv1 layer parameters (1→6, 5x5 kernel)."""
    var fixture = ConvFixture(
        in_channels=1, out_channels=6, kernel_size=5, dtype=dtype
    )
    return fixture.kernel, fixture.bias


fn create_conv2_parameters(dtype: DType) raises -> Tuple[AnyTensor, AnyTensor]:
    """Create Conv2 layer parameters (6→16, 5x5 kernel)."""
    var fixture = ConvFixture(
        in_channels=6, out_channels=16, kernel_size=5, dtype=dtype
    )
    return fixture.kernel, fixture.bias


# ============================================================================
# Conv1 Tests
# ============================================================================


fn test_conv1_forward_float32() raises:
    """Test Conv1 forward pass float32."""
    var dtype = DType.float32
    var _result = create_conv1_parameters(dtype)
    var kernel = _result[0]
    var bias = _result[1]
    LayerTester.test_conv_layer(
        in_channels=1,
        out_channels=6,
        kernel_size=5,
        input_h=28,
        input_w=28,
        weights=kernel,
        bias=bias,
        dtype=dtype,
    )


fn test_conv1_forward_float16() raises:
    """Test Conv1 forward pass float16."""
    var dtype = DType.float16
    var _result = create_conv1_parameters(dtype)
    var kernel = _result[0]
    var bias = _result[1]
    LayerTester.test_conv_layer(
        in_channels=1,
        out_channels=6,
        kernel_size=5,
        input_h=28,
        input_w=28,
        weights=kernel,
        bias=bias,
        dtype=dtype,
    )


fn test_conv1_backward_float32() raises:
    """Test Conv1 backward pass."""
    var dtype = DType.float32
    var _result = create_conv1_parameters(dtype)
    var kernel = _result[0]
    var bias = _result[1]
    LayerTester.test_conv_layer_backward(
        in_channels=1,
        out_channels=6,
        kernel_size=5,
        input_h=8,
        input_w=8,
        weights=kernel,
        bias=bias,
        dtype=dtype,
    )


# ============================================================================
# Conv2 Tests
# ============================================================================


fn test_conv2_forward_float32() raises:
    """Test Conv2 forward pass float32."""
    var dtype = DType.float32
    var _result = create_conv2_parameters(dtype)
    var kernel = _result[0]
    var bias = _result[1]
    LayerTester.test_conv_layer(
        in_channels=6,
        out_channels=16,
        kernel_size=5,
        input_h=14,
        input_w=14,
        weights=kernel,
        bias=bias,
        dtype=dtype,
    )


fn test_conv2_forward_float16() raises:
    """Test Conv2 forward pass float16."""
    var dtype = DType.float16
    var _result = create_conv2_parameters(dtype)
    var kernel = _result[0]
    var bias = _result[1]
    LayerTester.test_conv_layer(
        in_channels=6,
        out_channels=16,
        kernel_size=5,
        input_h=14,
        input_w=14,
        weights=kernel,
        bias=bias,
        dtype=dtype,
    )


fn test_conv2_backward_float32() raises:
    """Test Conv2 backward pass."""
    var dtype = DType.float32
    var _result = create_conv2_parameters(dtype)
    var kernel = _result[0]
    var bias = _result[1]
    LayerTester.test_conv_layer_backward(
        in_channels=6,
        out_channels=16,
        kernel_size=5,
        input_h=8,
        input_w=8,
        weights=kernel,
        bias=bias,
        dtype=dtype,
    )


# ============================================================================
# ReLU Tests (forward only)
# ============================================================================


fn test_relu_forward_float32() raises:
    """Test ReLU forward pass float32."""
    var shape: List[Int] = [1, 6, 24, 24]
    LayerTester.test_activation_layer(shape, DType.float32, activation="relu")


fn test_relu_forward_float16() raises:
    """Test ReLU forward pass float16."""
    var shape: List[Int] = [1, 6, 24, 24]
    LayerTester.test_activation_layer(shape, DType.float16, activation="relu")


fn main() raises:
    """Run part 1 of heap corruption split tests (8 tests)."""
    print("Heap Corruption Split - Part 1 (Conv1, Conv2, ReLU forward)")
    print("=" * 60)

    print("[1/8] test_conv1_forward_float32...", end="")
    test_conv1_forward_float32()
    print(" OK")

    print("[2/8] test_conv1_forward_float16...", end="")
    test_conv1_forward_float16()
    print(" OK")

    print("[3/8] test_conv1_backward_float32...", end="")
    test_conv1_backward_float32()
    print(" OK")

    print("[4/8] test_conv2_forward_float32...", end="")
    test_conv2_forward_float32()
    print(" OK")

    print("[5/8] test_conv2_forward_float16...", end="")
    test_conv2_forward_float16()
    print(" OK")

    print("[6/8] test_conv2_backward_float32...", end="")
    test_conv2_backward_float32()
    print(" OK")

    print("[7/8] test_relu_forward_float32...", end="")
    test_relu_forward_float32()
    print(" OK")

    print("[8/8] test_relu_forward_float16...", end="")
    test_relu_forward_float16()
    print(" OK")

    print("")
    print("=" * 60)
    print("✅ ALL 8 TESTS PASSED (Part 1)")
