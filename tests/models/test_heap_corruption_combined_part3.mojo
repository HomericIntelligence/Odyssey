# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_heap_corruption_combined.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Heap corruption split part 3: FC2, FC3, and Flatten tests.

Split from test_heap_corruption_combined.mojo (ADR-009).
Contains 8 fn test_ functions (limit: 10).
"""

from shared.core.extensor import ExTensor
from shared.core.linear import linear
from shared.testing.layer_params import LinearFixture
from shared.testing.assertions import (
    assert_shape,
    assert_dtype,
    assert_true,
    assert_false,
)
from shared.testing.special_values import (
    create_special_value_tensor,
    SPECIAL_VALUE_ONE,
)
from shared.testing.layer_testers import LayerTester
from math import isnan, isinf


# ============================================================================
# Fixtures
# ============================================================================


fn create_fc2_parameters(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    """Create FC2 layer parameters (120→84)."""
    var fixture = LinearFixture(in_features=120, out_features=84, dtype=dtype)
    return fixture.weights, fixture.bias


fn create_fc3_parameters(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    """Create FC3 layer parameters (84→10)."""
    var fixture = LinearFixture(in_features=84, out_features=10, dtype=dtype)
    return fixture.weights, fixture.bias


# ============================================================================
# FC2 Tests
# ============================================================================


fn test_fc2_forward_float32() raises:
    """Test FC2 forward pass float32."""
    var dtype = DType.float32
    var _result = create_fc2_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer(
        in_features=120,
        out_features=84,
        weights=weights,
        bias=bias,
        dtype=dtype,
    )


fn test_fc2_forward_float16() raises:
    """Test FC2 forward pass float16."""
    var dtype = DType.float16
    var _result = create_fc2_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer(
        in_features=120,
        out_features=84,
        weights=weights,
        bias=bias,
        dtype=dtype,
    )


fn test_fc2_backward_float32() raises:
    """Test FC2 backward pass."""
    var dtype = DType.float32
    var _result = create_fc2_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer_backward(
        in_features=120,
        out_features=84,
        weights=weights,
        bias=bias,
        dtype=dtype,
    )


# ============================================================================
# FC3 Tests
# ============================================================================


fn test_fc3_forward_float32() raises:
    """Test FC3 forward pass float32."""
    var dtype = DType.float32
    var _result = create_fc3_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer(
        in_features=84, out_features=10, weights=weights, bias=bias, dtype=dtype
    )


fn test_fc3_forward_float16() raises:
    """Test FC3 forward pass float16."""
    var dtype = DType.float16
    var _result = create_fc3_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer(
        in_features=84, out_features=10, weights=weights, bias=bias, dtype=dtype
    )


fn test_fc3_backward_float32() raises:
    """Test FC3 backward pass."""
    var dtype = DType.float32
    var _result = create_fc3_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer_backward(
        in_features=84, out_features=10, weights=weights, bias=bias, dtype=dtype
    )


# ============================================================================
# Flatten/Reshape Tests
# ============================================================================


fn test_flatten_operation_float32() raises:
    """Test reshape/flatten operation (16, 5, 5) -> (400,)."""
    var dtype = DType.float32
    var input = create_special_value_tensor(
        [1, 16, 5, 5], dtype, SPECIAL_VALUE_ONE
    )
    var flattened = input.reshape([1, 400])
    assert_shape(flattened, [1, 400], "Flatten shape mismatch")
    assert_dtype(flattened, dtype, "Flatten dtype mismatch")
    for i in range(flattened.numel()):
        var val = flattened._get_float64(i)
        assert_false(isnan(val), "Flatten produced NaN")
        assert_false(isinf(val), "Flatten produced Inf")


fn test_flatten_operation_float16() raises:
    """Test flatten with float16."""
    var dtype = DType.float16
    var input = create_special_value_tensor(
        [1, 16, 5, 5], dtype, SPECIAL_VALUE_ONE
    )
    var flattened = input.reshape([1, 400])
    assert_shape(flattened, [1, 400], "Flatten shape mismatch (float16)")
    assert_dtype(flattened, dtype, "Flatten dtype mismatch (float16)")


fn main() raises:
    """Run part 3 of heap corruption split tests (8 tests)."""
    print("Heap Corruption Split - Part 3 (FC2, FC3, Flatten)")
    print("=" * 60)

    print("[1/8] test_fc2_forward_float32...", end="")
    test_fc2_forward_float32()
    print(" OK")

    print("[2/8] test_fc2_forward_float16...", end="")
    test_fc2_forward_float16()
    print(" OK")

    print("[3/8] test_fc2_backward_float32...", end="")
    test_fc2_backward_float32()
    print(" OK")

    print("[4/8] test_fc3_forward_float32...", end="")
    test_fc3_forward_float32()
    print(" OK")

    print("[5/8] test_fc3_forward_float16...", end="")
    test_fc3_forward_float16()
    print(" OK")

    print("[6/8] test_fc3_backward_float32...", end="")
    test_fc3_backward_float32()
    print(" OK")

    print("[7/8] test_flatten_operation_float32...", end="")
    test_flatten_operation_float32()
    print(" OK")

    print("[8/8] test_flatten_operation_float16...", end="")
    test_flatten_operation_float16()
    print(" OK")

    print("")
    print("=" * 60)
    print("✅ ALL 8 TESTS PASSED (Part 3)")
