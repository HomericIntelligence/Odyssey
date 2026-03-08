"""Tests for SimpleLinearModel (no-bias, params), Parameter, factory functions, and integration.

Split from test_test_models.mojo per ADR-009 to avoid Mojo heap corruption.

Coverage:
    - SimpleLinearModel without bias
    - SimpleLinearModel num_parameters
    - Parameter struct initialization and shape
    - Factory functions (create_test_cnn, create_linear_model)
    - Integration tests (multiple models, MLP configs)
"""

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_test_models.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

from shared.testing import (
    SimpleMLP,
    SimpleLinearModel,
    Parameter,
    create_test_cnn,
    create_linear_model,
    assert_true,
    assert_equal,
    assert_close_float,
)
from shared.core import (
    ExTensor,
    zeros,
    ones,
)


# ============================================================================
# SimpleLinearModel Tests (continued)
# ============================================================================


fn test_simple_linear_model_no_bias() raises:
    """Test SimpleLinearModel without bias."""
    var model = SimpleLinearModel(10, 5, use_bias=False, init_value=0.1)

    var input = List[Float32]()
    for _ in range(10):
        input.append(1.0)

    var output = model.forward(input)

    assert_equal(len(output), 5)

    # Each output = sum(10 weights * 1.0) = 10 * 0.1 = 1.0
    var expected = Float64(1.0)
    for i in range(5):
        assert_close_float(Float64(output[i]), expected)


fn test_simple_linear_model_num_parameters() raises:
    """Test SimpleLinearModel parameter counting."""
    var model_with_bias = SimpleLinearModel(10, 5, use_bias=True)
    assert_equal(model_with_bias.num_parameters(), 55)  # 50 + 5

    var model_no_bias = SimpleLinearModel(10, 5, use_bias=False)
    assert_equal(model_no_bias.num_parameters(), 50)  # 50 only


# ============================================================================
# Parameter Tests
# ============================================================================


fn test_parameter_initialization() raises:
    """Test Parameter initialization."""
    var shape = [10, 5]

    var data = ones(shape, DType.float32)
    var param = Parameter(data)

    # Check shape preserved
    assert_equal(len(param.data._shape), 2)
    assert_equal(param.data._shape[0], 10)
    assert_equal(param.data._shape[1], 5)

    # Check gradient initialized to zeros
    assert_equal(param.grad.numel(), 50)
    for i in range(param.grad.numel()):
        assert_equal(param.grad._get_float64(i), 0.0)


fn test_parameter_shape() raises:
    """Test Parameter.shape() method."""
    var shape = [20, 15]

    var data = zeros(shape, DType.float32)
    var param = Parameter(data)

    var param_shape = param.shape()
    assert_equal(len(param_shape), 2)
    assert_equal(param_shape[0], 20)
    assert_equal(param_shape[1], 15)


# ============================================================================
# Factory Function Tests
# ============================================================================


fn test_create_test_cnn() raises:
    """Test create_test_cnn factory function."""
    # Default parameters
    var cnn1 = create_test_cnn()
    assert_equal(cnn1.in_channels, 1)
    assert_equal(cnn1.out_channels, 8)
    assert_equal(cnn1.num_classes, 10)

    # Custom parameters
    var cnn2 = create_test_cnn(3, 16, 100)
    assert_equal(cnn2.in_channels, 3)
    assert_equal(cnn2.out_channels, 16)
    assert_equal(cnn2.num_classes, 100)


fn test_create_linear_model() raises:
    """Test create_linear_model factory function."""
    # Default parameters
    var linear1 = create_linear_model()
    assert_equal(linear1.in_features, 784)
    assert_equal(linear1.out_features, 10)

    # Custom parameters
    var linear2 = create_linear_model(2048, 1024)
    assert_equal(linear2.in_features, 2048)
    assert_equal(linear2.out_features, 1024)


# ============================================================================
# Integration Tests
# ============================================================================


fn test_multiple_models_forward() raises:
    """Test multiple models in sequence."""
    # Create models
    var cnn = create_test_cnn(1, 8, 10)
    var linear = create_linear_model(784, 10)

    # Run CNN forward
    var shape = [32, 1, 28, 28]

    var cnn_input = ones(shape, DType.float32)
    var cnn_output = cnn.forward(cnn_input)
    assert_equal(cnn_output._shape[0], 32)
    assert_equal(cnn_output._shape[1], 10)

    # Run Linear forward
    # Reuse shape variable
    shape = [32, 784]

    var linear_input = zeros(shape, DType.float32)
    var linear_output = linear.forward(linear_input)
    assert_equal(linear_output._shape[0], 32)
    assert_equal(linear_output._shape[1], 10)


fn test_mlp_with_different_configs() raises:
    """Test MLP with various configurations."""
    # Small MLP
    var mlp_small = SimpleMLP(5, 10, 2, num_hidden_layers=1)
    assert_equal(mlp_small.num_parameters(), 5 * 10 + 10 + 10 * 2 + 2)

    # Large MLP
    var mlp_large = SimpleMLP(100, 200, 50, num_hidden_layers=2)
    expected_params = 100 * 200 + 200 + 200 * 200 + 200 + 200 * 50 + 50
    assert_equal(mlp_large.num_parameters(), expected_params)

    # Single hidden unit
    var mlp_minimal = SimpleMLP(1, 1, 1, num_hidden_layers=1)
    assert_equal(mlp_minimal.num_parameters(), 1 * 1 + 1 + 1 * 1 + 1)


fn main() raises:
    """Run all tests."""
    print("Testing SimpleLinearModel (no-bias, params)...")
    test_simple_linear_model_no_bias()
    test_simple_linear_model_num_parameters()

    print("Testing Parameter...")
    test_parameter_initialization()
    test_parameter_shape()

    print("Testing factory functions...")
    test_create_test_cnn()
    test_create_linear_model()

    print("Testing integration scenarios...")
    test_multiple_models_forward()
    test_mlp_with_different_configs()

    print("All tests passed!")
