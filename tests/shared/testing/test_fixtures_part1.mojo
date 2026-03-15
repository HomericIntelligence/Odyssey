# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_fixtures.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for shared.testing.fixtures module - Part 1: Model initialization and forward passes.

Tests SimpleCNN and LinearModel structs plus the create_test_cnn factory function.
"""

from testing import assert_true, assert_equal
from shared.testing.models import SimpleCNN, LinearModel
from shared.testing.fixtures import (
    create_test_cnn,
)
from shared.core.extensor import ones


fn test_simple_cnn_initialization() raises:
    """Test SimpleCNN struct initialization."""
    var model = SimpleCNN(1, 8, 10)
    assert_equal(model.in_channels, 1)
    assert_equal(model.out_channels, 8)
    assert_equal(model.num_classes, 10)


fn test_simple_cnn_default_initialization() raises:
    """Test SimpleCNN with default parameters."""
    var model = SimpleCNN()
    assert_equal(model.in_channels, 1)
    assert_equal(model.out_channels, 8)
    assert_equal(model.num_classes, 10)


fn test_simple_cnn_get_output_shape() raises:
    """Test SimpleCNN output shape computation."""
    var model = SimpleCNN(1, 8, 10)
    var shape = model.get_output_shape(32)
    assert_equal(len(shape), 2)
    assert_equal(shape[0], 32)
    assert_equal(shape[1], 10)


fn test_simple_cnn_forward() raises:
    """Test SimpleCNN forward pass produces correct output shape."""
    var model = SimpleCNN(1, 8, 10)
    var batch_size = 32
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(1)
    input_shape.append(28)
    input_shape.append(28)

    var input = ones(input_shape, DType.float32)
    var output = model.forward(input)

    # Check output shape
    assert_equal(output._shape[0], batch_size)
    assert_equal(output._shape[1], 10)

    # Check all values are 0.1
    for i in range(output.numel()):
        var val = output._get_float64(i)
        assert_true(val > 0.099 and val < 0.101, "Output should be 0.1")


fn test_linear_model_initialization() raises:
    """Test LinearModel struct initialization."""
    var model = LinearModel(784, 10)
    assert_equal(model.in_features, 784)
    assert_equal(model.out_features, 10)


fn test_linear_model_get_output_shape() raises:
    """Test LinearModel output shape computation."""
    var model = LinearModel(784, 10)
    var shape = model.get_output_shape(32)
    assert_equal(len(shape), 2)
    assert_equal(shape[0], 32)
    assert_equal(shape[1], 10)


fn test_linear_model_forward() raises:
    """Test LinearModel forward pass produces correct output shape."""
    var model = LinearModel(784, 10)
    var batch_size = 32
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(784)

    var input = ones(input_shape, DType.float32)
    var output = model.forward(input)

    # Check output shape
    assert_equal(output._shape[0], batch_size)
    assert_equal(output._shape[1], 10)

    # Check all values are zeros
    for i in range(output.numel()):
        var val = output._get_float64(i)
        assert_equal(val, 0.0)


fn test_create_test_cnn() raises:
    """Test create_test_cnn factory function."""
    var model = create_test_cnn()
    assert_equal(model.in_channels, 1)
    assert_equal(model.out_channels, 8)
    assert_equal(model.num_classes, 10)

    var custom_model = create_test_cnn(3, 32, 1000)
    assert_equal(custom_model.in_channels, 3)
    assert_equal(custom_model.out_channels, 32)
    assert_equal(custom_model.num_classes, 1000)


fn main() raises:
    """Run all fixture tests - Part 1."""
    test_simple_cnn_initialization()
    test_simple_cnn_default_initialization()
    test_simple_cnn_get_output_shape()
    test_simple_cnn_forward()

    test_linear_model_initialization()
    test_linear_model_get_output_shape()
    test_linear_model_forward()

    test_create_test_cnn()

    print("All tests passed!")
