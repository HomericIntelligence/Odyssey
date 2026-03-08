"""Tests for MockLayer and SimpleLinearModel (initialization and custom init).

Split from test_test_models.mojo per ADR-009 to avoid Mojo heap corruption.

Coverage:
    - MockLayer initialization and forward pass (truncate, pad, scale)
    - MockLayer num_parameters
    - SimpleLinearModel initialization
    - SimpleLinearModel custom init value
    - SimpleLinearModel forward pass
"""

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_test_models.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

from shared.testing import (
    MockLayer,
    SimpleLinearModel,
    assert_equal,
    assert_close_float,
)


# ============================================================================
# MockLayer Tests
# ============================================================================


fn test_mock_layer_initialization() raises:
    """Test MockLayer initialization."""
    var layer = MockLayer(10, 5)
    assert_equal(layer.input_dim, 10)
    assert_equal(layer.output_dim, 5)
    assert_equal(layer.scale, 1.0)

    var scaled_layer = MockLayer(20, 10, scale=2.0)
    assert_equal(scaled_layer.scale, 2.0)


fn test_mock_layer_forward_truncate() raises:
    """Test MockLayer forward pass with truncation."""
    var layer = MockLayer(10, 5, scale=1.0)

    var input = List[Float32]()
    for i in range(10):
        input.append(Float32(i))

    var output = layer.forward(input)

    assert_equal(len(output), 5)
    for i in range(5):
        assert_equal(output[i], Float32(i))


fn test_mock_layer_forward_pad() raises:
    """Test MockLayer forward pass with padding."""
    var layer = MockLayer(5, 10, scale=1.0)

    var input = List[Float32]()
    for i in range(5):
        input.append(Float32(i))

    var output = layer.forward(input)

    assert_equal(len(output), 10)
    # First 5 elements should be input
    for i in range(5):
        assert_equal(output[i], Float32(i))
    # Last 5 should be zeros
    for i in range(5, 10):
        assert_equal(output[i], 0.0)


fn test_mock_layer_forward_scale() raises:
    """Test MockLayer forward pass with scaling."""
    var layer = MockLayer(5, 5, scale=2.0)

    var input = List[Float32]()
    for _ in range(5):
        input.append(1.0)

    var output = layer.forward(input)

    assert_equal(len(output), 5)
    for i in range(5):
        assert_equal(output[i], 2.0)


fn test_mock_layer_num_parameters() raises:
    """Test MockLayer.num_parameters() method."""
    var layer = MockLayer(10, 5)
    assert_equal(layer.num_parameters(), 50)

    var layer2 = MockLayer(20, 15)
    assert_equal(layer2.num_parameters(), 300)


# ============================================================================
# SimpleLinearModel Tests
# ============================================================================


fn test_simple_linear_model_initialization() raises:
    """Test SimpleLinearModel initialization."""
    var model = SimpleLinearModel(10, 5)
    assert_equal(model.input_dim, 10)
    assert_equal(model.output_dim, 5)
    assert_equal(model.use_bias, True)
    assert_equal(len(model.weights), 50)
    assert_equal(len(model.bias), 5)

    var model_no_bias = SimpleLinearModel(10, 5, use_bias=False)
    assert_equal(model_no_bias.use_bias, False)
    assert_equal(len(model_no_bias.bias), 0)


fn test_simple_linear_model_custom_init_value() raises:
    """Test SimpleLinearModel with custom init value."""
    var model = SimpleLinearModel(10, 5, use_bias=True, init_value=0.5)

    for i in range(len(model.weights)):
        assert_equal(model.weights[i], 0.5)

    for i in range(len(model.bias)):
        assert_equal(model.bias[i], 0.5)


fn test_simple_linear_model_forward() raises:
    """Test SimpleLinearModel forward pass."""
    var model = SimpleLinearModel(10, 5, init_value=0.1)

    var input = List[Float32]()
    for _ in range(10):
        input.append(1.0)

    var output = model.forward(input)

    assert_equal(len(output), 5)

    # Each output = sum(10 weights * 1.0) + bias
    # = 10 * 0.1 + 0.1 = 1.1
    var expected = Float64(10 * 0.1 + 0.1)
    for i in range(5):
        assert_close_float(Float64(output[i]), expected)


fn main() raises:
    """Run all tests."""
    print("Testing MockLayer...")
    test_mock_layer_initialization()
    test_mock_layer_forward_truncate()
    test_mock_layer_forward_pad()
    test_mock_layer_forward_scale()
    test_mock_layer_num_parameters()

    print("Testing SimpleLinearModel (initialization and forward)...")
    test_simple_linear_model_initialization()
    test_simple_linear_model_custom_init_value()
    test_simple_linear_model_forward()

    print("All tests passed!")
