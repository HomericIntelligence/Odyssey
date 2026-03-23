# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_fixtures.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for shared.testing.fixtures module - Part 2: Factory functions and tensor shape/dtype assertions.

Tests create_linear_model, create_test_input, create_test_targets factory functions,
and assert_tensor_shape and assert_tensor_dtype helpers.
"""

from testing import assert_true, assert_equal
from shared.testing.fixtures import (
    create_linear_model,
    create_test_input,
    create_test_targets,
    assert_tensor_shape,
    assert_tensor_dtype,
)
from shared.tensor.any_tensor import ones


fn test_create_linear_model() raises:
    """Test create_linear_model factory function."""
    var model = create_linear_model()
    assert_equal(model.in_features, 784)
    assert_equal(model.out_features, 10)

    var custom_model = create_linear_model(2048, 1024)
    assert_equal(custom_model.in_features, 2048)
    assert_equal(custom_model.out_features, 1024)


fn test_create_test_input() raises:
    """Test create_test_input utility function."""
    var input = create_test_input(32, 784)
    assert_equal(input._shape[0], 32)
    assert_equal(input._shape[1], 784)

    # Check all values are 1.0
    for i in range(input.numel()):
        var val = input._get_float64(i)
        assert_equal(val, 1.0)


fn test_create_test_input_custom_dtype() raises:
    """Test create_test_input with custom dtype."""
    var input = create_test_input(16, 512, DType.float64)
    assert_equal(input._shape[0], 16)
    assert_equal(input._shape[1], 512)
    assert_true(input._dtype == DType.float64)


fn test_create_test_targets() raises:
    """Test create_test_targets utility function."""
    var targets = create_test_targets(32, 10)
    assert_equal(targets._shape[0], 32)

    # Check all values are 0
    for i in range(targets.numel()):
        var val = targets._get_float64(i)
        assert_equal(val, 0.0)


fn test_assert_tensor_shape_valid() raises:
    """Test assert_tensor_shape with matching shapes."""
    var tensor = ones([32, 10], DType.float32)
    var expected: List[Int] = [32, 10]
    assert_true(assert_tensor_shape(tensor, expected))


fn test_assert_tensor_shape_invalid_dimensions() raises:
    """Test assert_tensor_shape with wrong number of dimensions."""
    var tensor = ones([32, 10], DType.float32)
    var expected: List[Int] = [32, 10, 5]
    assert_true(not assert_tensor_shape(tensor, expected))


fn test_assert_tensor_shape_invalid_size() raises:
    """Test assert_tensor_shape with wrong dimension sizes."""
    var tensor = ones([32, 10], DType.float32)
    var expected: List[Int] = [64, 10]
    assert_true(not assert_tensor_shape(tensor, expected))


fn test_assert_tensor_dtype_valid() raises:
    """Test assert_tensor_dtype with matching dtype."""
    var tensor = ones([32, 10], DType.float32)
    assert_true(assert_tensor_dtype(tensor, DType.float32))


fn main() raises:
    """Run all fixture tests - Part 2."""
    test_create_linear_model()

    test_create_test_input()
    test_create_test_input_custom_dtype()
    test_create_test_targets()

    test_assert_tensor_shape_valid()
    test_assert_tensor_shape_invalid_dimensions()
    test_assert_tensor_shape_invalid_size()

    test_assert_tensor_dtype_valid()

    print("All tests passed!")
