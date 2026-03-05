"""LeNet-5 Fully Connected Layer Tests

Tests FC1, FC2, FC3 layers independently with special FP-representable values.

Workaround for Issue #2942: This file contains <10 tests to avoid heap corruption
bug that occurs after running 15+ cumulative tests.

Tests:
- FC1 (400→120): forward float32, forward float16, backward float32
- FC2 (120→84): forward float32, forward float16, backward float32
- FC3 (84→10): forward float32, forward float16, backward float32

Float16 Precision Limitations
==============================
FC3 (84→10) accumulates 84 multiplications per output element. Float16's
~3.3 decimal digit precision (~11-bit mantissa) can cause rounding errors at
this accumulation depth, particularly with inputs near the boundary of
representable values. The test runs but results may have reduced accuracy
compared to Float32. This is an expected limitation of Float16 arithmetic
(not a bug). See issue #3009 for detailed analysis.
"""

from shared.core.extensor import ExTensor
from shared.testing.layer_params import LinearFixture
from shared.testing.layer_testers import LayerTester


# ============================================================================
# Test Fixtures
# ============================================================================


fn create_fc1_parameters(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    """Create FC1 layer parameters (400→120)."""
    var fixture = LinearFixture(in_features=400, out_features=120, dtype=dtype)
    return fixture.weights, fixture.bias


fn create_fc2_parameters(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    """Create FC2 layer parameters (120→84)."""
    var fixture = LinearFixture(in_features=120, out_features=84, dtype=dtype)
    return fixture.weights, fixture.bias


fn create_fc3_parameters(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    """Create FC3 layer parameters (84→10)."""
    var fixture = LinearFixture(in_features=84, out_features=10, dtype=dtype)
    return fixture.weights, fixture.bias


# ============================================================================
# FC1 (Linear) Tests
# ============================================================================


fn test_fc1_forward_float32() raises:
    """Test FC1 (400→120) forward pass with float32."""
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
    """Test FC1 forward pass with float16."""
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
    """Test FC1 backward pass with gradient checking."""
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


# ============================================================================
# FC2 (Linear) Tests
# ============================================================================


fn test_fc2_forward_float32() raises:
    """Test FC2 (120→84) forward pass with float32."""
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
    """Test FC2 forward pass with float16."""
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
    """Test FC2 backward pass with gradient checking."""
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
# FC3 (Linear) Tests
# ============================================================================


fn test_fc3_forward_float32() raises:
    """Test FC3 (84→10) forward pass with float32."""
    var dtype = DType.float32
    var _result = create_fc3_parameters(dtype)

    var weights = _result[0]

    var bias = _result[1]

    LayerTester.test_linear_layer(
        in_features=84,
        out_features=10,
        weights=weights,
        bias=bias,
        dtype=dtype,
    )


fn test_fc3_forward_float16() raises:
    """Test FC3 (84→10) forward pass with float16.

    NOTE: This test may be sensitive to float16 precision limitations.
    FC3 performs 84 multiplications per output, which can cause accumulation
    errors in float16 (limited to ~3.3 decimal digits precision).
    Known limitation - see issue #3009 for discussion.
    """
    var dtype = DType.float16
    var _result = create_fc3_parameters(dtype)

    var weights = _result[0]

    var bias = _result[1]

    LayerTester.test_linear_layer(
        in_features=84,
        out_features=10,
        weights=weights,
        bias=bias,
        dtype=dtype,
    )


fn test_fc3_backward_float32() raises:
    """Test FC3 backward pass with gradient checking."""
    var dtype = DType.float32
    var _result = create_fc3_parameters(dtype)

    var weights = _result[0]

    var bias = _result[1]

    LayerTester.test_linear_layer_backward(
        in_features=84,
        out_features=10,
        weights=weights,
        bias=bias,
        dtype=dtype,
    )


fn main() raises:
    """Run all fully connected layer tests."""
    print("LeNet-5 Fully Connected Layer Tests")
    print("=" * 50)

    # FC1 tests
    print("  test_fc1_forward_float32...", end="")
    test_fc1_forward_float32()
    print(" OK")

    print("  test_fc1_forward_float16...", end="")
    test_fc1_forward_float16()
    print(" OK")

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

    print("  test_fc2_backward_float32...", end="")
    test_fc2_backward_float32()
    print(" OK")

    # FC3 tests
    print("  test_fc3_forward_float32...", end="")
    test_fc3_forward_float32()
    print(" OK")

    print("  test_fc3_forward_float16...", end="")
    test_fc3_forward_float16()
    print(" OK")

    print("  test_fc3_backward_float32...", end="")
    test_fc3_backward_float32()
    print(" OK")

    print("\n✅ All fully connected layer tests passed (9/9)")
