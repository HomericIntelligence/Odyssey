# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_heap_corruption_combined.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Heap corruption split part 2: ReLU (backward), MaxPool, and FC1 tests.

Split from test_heap_corruption_combined.mojo (ADR-009).
Contains 8 fn test_ functions (limit: 10).
"""

from shared.core.extensor import AnyTensor
from shared.core.pooling import maxpool2d
from shared.core.linear import linear
from shared.core.activation import relu
from shared.testing.layer_params import LinearFixture
from shared.testing.assertions import (
    assert_shape,
    assert_dtype,
)
from shared.testing.layer_testers import LayerTester


# ============================================================================
# Fixtures
# ============================================================================


fn create_fc1_parameters(dtype: DType) raises -> Tuple[AnyTensor, AnyTensor]:
    """Create FC1 layer parameters (400→120)."""
    var fixture = LinearFixture(in_features=400, out_features=120, dtype=dtype)
    return fixture.weights, fixture.bias


# ============================================================================
# ReLU Tests (backward)
# ============================================================================


fn test_relu_backward_float32() raises:
    """Test ReLU backward pass."""
    var shape: List[Int] = [1, 6, 8, 8]
    LayerTester.test_activation_layer_backward(
        shape, DType.float32, activation="relu"
    )


# ============================================================================
# MaxPool Tests
# ============================================================================


fn test_maxpool1_forward_float32() raises:
    """Test MaxPool1 forward pass float32."""
    LayerTester.test_pooling_layer(
        channels=6,
        input_h=24,
        input_w=24,
        pool_size=2,
        stride=2,
        dtype=DType.float32,
        pool_type="max",
        padding=0,
    )


fn test_maxpool1_forward_float16() raises:
    """Test MaxPool1 forward pass float16."""
    LayerTester.test_pooling_layer(
        channels=6,
        input_h=24,
        input_w=24,
        pool_size=2,
        stride=2,
        dtype=DType.float16,
        pool_type="max",
        padding=0,
    )


fn test_maxpool2_forward_float32() raises:
    """Test MaxPool2 forward pass float32."""
    LayerTester.test_pooling_layer(
        channels=16,
        input_h=10,
        input_w=10,
        pool_size=2,
        stride=2,
        dtype=DType.float32,
        pool_type="max",
        padding=0,
    )


fn test_maxpool2_forward_float16() raises:
    """Test MaxPool2 forward pass float16."""
    LayerTester.test_pooling_layer(
        channels=16,
        input_h=10,
        input_w=10,
        pool_size=2,
        stride=2,
        dtype=DType.float16,
        pool_type="max",
        padding=0,
    )


# ============================================================================
# FC1 Tests
# ============================================================================


fn test_fc1_forward_float32() raises:
    """Test FC1 forward pass float32."""
    var dtype = DType.float32
    var _result = create_fc1_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer(
        in_features=400,
        out_features=120,
        weights=weights,
        bias=bias,
        dtype=dtype,
    )


fn test_fc1_forward_float16() raises:
    """Test FC1 forward pass float16."""
    var dtype = DType.float16
    var _result = create_fc1_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer(
        in_features=400,
        out_features=120,
        weights=weights,
        bias=bias,
        dtype=dtype,
    )


fn test_fc1_backward_float32() raises:
    """Test FC1 backward pass."""
    var dtype = DType.float32
    var _result = create_fc1_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer_backward(
        in_features=400,
        out_features=120,
        weights=weights,
        bias=bias,
        dtype=dtype,
    )


fn main() raises:
    """Run part 2 of heap corruption split tests (8 tests)."""
    print("Heap Corruption Split - Part 2 (ReLU backward, MaxPool, FC1)")
    print("=" * 60)

    print("[1/8] test_relu_backward_float32...", end="")
    test_relu_backward_float32()
    print(" OK")

    print("[2/8] test_maxpool1_forward_float32...", end="")
    test_maxpool1_forward_float32()
    print(" OK")

    print("[3/8] test_maxpool1_forward_float16...", end="")
    test_maxpool1_forward_float16()
    print(" OK")

    print("[4/8] test_maxpool2_forward_float32...", end="")
    test_maxpool2_forward_float32()
    print(" OK")

    print("[5/8] test_maxpool2_forward_float16...", end="")
    test_maxpool2_forward_float16()
    print(" OK")

    print("[6/8] test_fc1_forward_float32...", end="")
    test_fc1_forward_float32()
    print(" OK")

    print("[7/8] test_fc1_forward_float16...", end="")
    test_fc1_forward_float16()
    print(" OK")

    print("[8/8] test_fc1_backward_float32...", end="")
    test_fc1_backward_float32()
    print(" OK")

    print("")
    print("=" * 60)
    print("✅ ALL 8 TESTS PASSED (Part 2)")
