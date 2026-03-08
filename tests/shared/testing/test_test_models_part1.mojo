"""Tests for SimpleCNN and LinearModel test model definitions.

Split from test_test_models.mojo per ADR-009 to avoid Mojo heap corruption.

Coverage:
    - SimpleCNN initialization and forward pass
    - LinearModel initialization and forward pass
"""

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_test_models.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

from shared.testing import (
    SimpleCNN,
    LinearModel,
    assert_true,
    assert_equal,
)
from shared.core import (
    ExTensor,
    zeros,
    ones,
    full,
    zeros_like,
)


# ============================================================================
# SimpleCNN Tests
# ============================================================================


fn test_simple_cnn_initialization() raises:
    """Test SimpleCNN initialization with default and custom parameters."""
    # Default initialization
    var cnn_default = SimpleCNN()
    assert_equal(cnn_default.in_channels, 1)
    assert_equal(cnn_default.out_channels, 8)
    assert_equal(cnn_default.num_classes, 10)

    # Custom initialization
    var cnn_custom = SimpleCNN(3, 16, 100)
    assert_equal(cnn_custom.in_channels, 3)
    assert_equal(cnn_custom.out_channels, 16)
    assert_equal(cnn_custom.num_classes, 100)


fn test_simple_cnn_output_shape() raises:
    """Test SimpleCNN.get_output_shape() method."""
    var cnn = SimpleCNN(1, 8, 10)

    var shape_32 = cnn.get_output_shape(32)
    assert_equal(len(shape_32), 2)
    assert_equal(shape_32[0], 32)
    assert_equal(shape_32[1], 10)

    var shape_64 = cnn.get_output_shape(64)
    assert_equal(shape_64[0], 64)
    assert_equal(shape_64[1], 10)


fn test_simple_cnn_forward_pass() raises:
    """Test SimpleCNN forward pass shape and dtype."""
    var cnn = SimpleCNN(1, 8, 10)
    var input_shape = [32, 1, 28, 28]
    var input = ones(input_shape, DType.float32)

    var output = cnn.forward(input)

    # Check output shape
    assert_equal(len(output._shape), 2)
    assert_equal(output._shape[0], 32)
    assert_equal(output._shape[1], 10)

    # Check dtype preserved
    assert_true(
        output._dtype == DType.float32, "Output dtype should be float32"
    )

    # Check non-zero output
    var has_nonzero = False
    for i in range(output.numel()):
        if output._get_float64(i) != 0.0:
            has_nonzero = True
            break
    assert_true(has_nonzero, "CNN should produce non-zero output")


fn test_simple_cnn_batch_sizes() raises:
    """Test SimpleCNN with different batch sizes."""
    var cnn = SimpleCNN(1, 8, 10)

    for batch_size in range(1, 65, 16):
        var shape = List[Int]()
        shape.append(batch_size)
        shape.append(1)
        shape.append(28)
        shape.append(28)
        var input = zeros(shape, DType.float32)
        var output = cnn.forward(input)

        assert_equal(output._shape[0], batch_size)
        assert_equal(output._shape[1], 10)


# ============================================================================
# LinearModel Tests
# ============================================================================


fn test_linear_model_initialization() raises:
    """Test LinearModel initialization."""
    var linear = LinearModel(784, 10)
    assert_equal(linear.in_features, 784)
    assert_equal(linear.out_features, 10)

    var custom_linear = LinearModel(2048, 1024)
    assert_equal(custom_linear.in_features, 2048)
    assert_equal(custom_linear.out_features, 1024)


fn test_linear_model_output_shape() raises:
    """Test LinearModel.get_output_shape() method."""
    var linear = LinearModel(784, 10)

    var shape_32 = linear.get_output_shape(32)
    assert_equal(len(shape_32), 2)
    assert_equal(shape_32[0], 32)
    assert_equal(shape_32[1], 10)

    var shape_128 = linear.get_output_shape(128)
    assert_equal(shape_128[0], 128)
    assert_equal(shape_128[1], 10)


fn test_linear_model_forward_pass() raises:
    """Test LinearModel forward pass."""
    var linear = LinearModel(784, 10)
    var input_shape = [32, 784]
    var input = ones(input_shape, DType.float32)

    var output = linear.forward(input)

    # Check output shape
    assert_equal(len(output._shape), 2)
    assert_equal(output._shape[0], 32)
    assert_equal(output._shape[1], 10)

    # Check dtype preserved
    assert_true(
        output._dtype == DType.float32, "Output dtype should be float32"
    )

    # LinearModel forward produces zeros
    for i in range(output.numel()):
        assert_equal(output._get_float64(i), 0.0)


fn test_linear_model_batch_processing() raises:
    """Test LinearModel with different batch sizes."""
    var linear = LinearModel(100, 50)

    for batch_size in range(1, 65, 16):
        var shape = List[Int]()
        shape.append(batch_size)
        shape.append(100)
        var input = zeros(shape, DType.float32)
        var output = linear.forward(input)

        assert_equal(output._shape[0], batch_size)
        assert_equal(output._shape[1], 50)


fn main() raises:
    """Run all tests."""
    print("Testing SimpleCNN...")
    test_simple_cnn_initialization()
    test_simple_cnn_output_shape()
    test_simple_cnn_forward_pass()
    test_simple_cnn_batch_sizes()

    print("Testing LinearModel...")
    test_linear_model_initialization()
    test_linear_model_output_shape()
    test_linear_model_forward_pass()
    test_linear_model_batch_processing()

    print("All tests passed!")
