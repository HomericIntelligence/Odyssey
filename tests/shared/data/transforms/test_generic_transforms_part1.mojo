# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_generic_transforms.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for generic data transformation utilities - Part 1.

Tests identity transforms and lambda transforms.
"""

from shared.core.extensor import ExTensor
from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_false,
    assert_almost_equal,
    assert_greater,
    assert_less,
    TestFixtures,
)
from shared.data.transforms import Transform
from shared.data.generic_transforms import (
    IdentityTransform,
    LambdaTransform,
    ConditionalTransform,
    ClampTransform,
    DebugTransform,
    BatchTransform,
    ToFloat32,
    ToInt32,
    SequentialTransform,
    AnyTransform,
)

# Type comptime for test convenience
comptime Tensor = ExTensor


# ============================================================================
# Identity Transform Tests
# ============================================================================


fn test_identity_basic() raises:
    """Test identity transform returns input unchanged."""
    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(3.0)
    var data = ExTensor(values^)
    var identity = IdentityTransform()

    var result = identity(data)

    assert_equal(result.num_elements(), 3)
    assert_almost_equal(result[0], 1.0)
    assert_almost_equal(result[1], 2.0)
    assert_almost_equal(result[2], 3.0)


fn test_identity_preserves_values() raises:
    """Test identity preserves all values exactly."""
    var values = List[Float32]()
    values.append(0.0)
    values.append(-1.5)
    values.append(42.0)
    values.append(100.5)
    var data = ExTensor(values^)
    var identity = IdentityTransform()

    var result = identity(data)

    for i in range(data.num_elements()):
        assert_almost_equal(result[i], data[i])


fn test_identity_empty_tensor() raises:
    """Test identity handles empty tensor."""
    var values = List[Float32]()
    var data = ExTensor(values^)
    var identity = IdentityTransform()

    var result = identity(data)

    assert_equal(result.num_elements(), 0)


# ============================================================================
# Lambda Transform Tests
# ============================================================================


fn test_lambda_double_values() raises:
    """Test lambda transform doubles all values."""
    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(3.0)
    var data = ExTensor(values^)

    fn double_fn(value: Float32) -> Float32:
        return value * 2.0

    var transform = LambdaTransform(double_fn)
    var result = transform(data)

    assert_almost_equal(result[0], 2.0)
    assert_almost_equal(result[1], 4.0)
    assert_almost_equal(result[2], 6.0)


fn test_lambda_add_constant() raises:
    """Test lambda transform adds constant."""
    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(3.0)
    var data = ExTensor(values^)

    fn add_ten(value: Float32) -> Float32:
        return value + 10.0

    var transform = LambdaTransform(add_ten)
    var result = transform(data)

    assert_almost_equal(result[0], 11.0)
    assert_almost_equal(result[1], 12.0)
    assert_almost_equal(result[2], 13.0)


fn test_lambda_square_values() raises:
    """Test lambda transform squares values."""
    var values = List[Float32]()
    values.append(2.0)
    values.append(3.0)
    values.append(4.0)
    var data = ExTensor(values^)

    fn square(value: Float32) -> Float32:
        return value * value

    var transform = LambdaTransform(square)
    var result = transform(data)

    assert_almost_equal(result[0], 4.0)
    assert_almost_equal(result[1], 9.0)
    assert_almost_equal(result[2], 16.0)


fn test_lambda_negative_values() raises:
    """Test lambda transform with negative values."""
    var values = List[Float32]()
    values.append(-1.0)
    values.append(-2.0)
    values.append(-3.0)
    var data = ExTensor(values^)

    fn abs_value(value: Float32) -> Float32:
        return abs(value)

    var transform = LambdaTransform(abs_value)
    var result = transform(data)

    assert_almost_equal(result[0], 1.0)
    assert_almost_equal(result[1], 2.0)
    assert_almost_equal(result[2], 3.0)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run identity and lambda transform tests."""
    print("Running generic transform tests - Part 1...")
    print()

    # Identity tests
    print("Testing IdentityTransform...")
    test_identity_basic()
    test_identity_preserves_values()
    test_identity_empty_tensor()
    print("  ✓ 3 identity tests passed")

    # Lambda tests
    print("Testing LambdaTransform...")
    test_lambda_double_values()
    test_lambda_add_constant()
    test_lambda_square_values()
    test_lambda_negative_values()
    print("  ✓ 4 lambda tests passed")

    print()
    print("✓ All 7 Part 1 generic transform tests passed!")
