# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_layers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for neural network layers - Part 3: Property-Based and PyTorch Validation.

Tests cover:
- Property-based tests (batch independence, determinism)
- Numerical accuracy tests validated against PyTorch reference values

Split from test_layers.mojo per ADR-009 (≤10 fn test_ per file).
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from tests.shared.conftest import TestFixtures
from shared.tensor.any_tensor import AnyTensor, zeros, ones
from shared.core.linear import linear, linear_no_bias
from shared.core.activation import relu, sigmoid, tanh, softmax


# ============================================================================
# Property-Based Tests
# ============================================================================


fn test_layer_property_batch_independence() raises:
    """Property: Layer output for batch should equal individual outputs.

    Functional API:
        linear() should process batch elements independently.
        Processing a batch should give same results as processing individually.
    """
    var in_features = 4
    var out_features = 3
    var batch_size = 2

    # Create weights and bias
    var weight_shape = List[Int]()
    weight_shape.append(out_features)
    weight_shape.append(in_features)
    var weights = ones(weight_shape, DType.float32)
    for i in range(out_features * in_features):
        weights._data.bitcast[Float32]()[i] = 0.2

    var bias_shape = List[Int]()
    bias_shape.append(out_features)
    var bias = zeros(bias_shape, DType.float32)

    # Create batch input: (2, 4)
    var batch_input_shape = List[Int]()
    batch_input_shape.append(batch_size)
    batch_input_shape.append(in_features)
    var batch_input = ones(batch_input_shape, DType.float32)
    # Set different values for each batch element
    for i in range(in_features):
        batch_input._data.bitcast[Float32]()[i] = 1.0  # First batch element
        batch_input._data.bitcast[Float32]()[
            in_features + i
        ] = 2.0  # Second batch element

    # Process as batch
    var batch_output = linear(batch_input, weights, bias)

    # Process first element individually: (1, 4)
    var single_input_shape = List[Int]()
    single_input_shape.append(1)
    single_input_shape.append(in_features)
    var single_input_1 = ones(single_input_shape, DType.float32)
    for i in range(in_features):
        single_input_1._data.bitcast[Float32]()[i] = 1.0

    var single_output_1 = linear(single_input_1, weights, bias)

    # First batch element output should match individual processing
    for i in range(out_features):
        assert_almost_equal(
            batch_output._data.bitcast[Float32]()[i],
            single_output_1._data.bitcast[Float32]()[i],
            tolerance=1e-5,
        )


fn test_layer_property_deterministic() raises:
    """Property: Layer forward pass is deterministic.

    Functional API:
        Same input should always produce same output.
        Pure functional operations are inherently deterministic.
    """
    var in_features = 10
    var out_features = 5

    # Create weights and bias
    var weight_shape = List[Int]()
    weight_shape.append(out_features)
    weight_shape.append(in_features)
    var weights = ones(weight_shape, DType.float32)
    for i in range(out_features * in_features):
        weights._data.bitcast[Float32]()[i] = Float32(i) * 0.01

    var bias_shape = List[Int]()
    bias_shape.append(out_features)
    var bias = ones(bias_shape, DType.float32)
    for i in range(out_features):
        bias._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    # Create input
    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(in_features)
    var input = ones(input_shape, DType.float32)
    for i in range(2 * in_features):
        input._data.bitcast[Float32]()[i] = Float32(i % in_features)

    # Two forward passes with same input
    var output1 = linear(input, weights, bias)
    var output2 = linear(input, weights, bias)

    # Outputs should be identical
    var total_elements = 2 * out_features
    for i in range(total_elements):
        assert_almost_equal(
            output1._data.bitcast[Float32]()[i],
            output2._data.bitcast[Float32]()[i],
            tolerance=1e-9,  # Should be exactly equal
        )


# ============================================================================
# Numerical Accuracy Tests (PyTorch Validation)
# ============================================================================


fn test_linear_matches_pytorch() raises:
    """Test Linear matches PyTorch implementation numerically.

    This test validates numerical correctness against PyTorch reference values.

    PyTorch reference code:
        ```python
        import torch
        import torch.nn.functional as F

        # Input: shape (2, 4)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                          [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32)

        # Weights: shape (3, 4) - transposed in PyTorch linear
        weights = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                [0.5, 0.6, 0.7, 0.8],
                                [0.9, 1.0, 1.1, 1.2]], dtype=torch.float32)

        # Bias: shape (3,)
        bias = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        # Linear: y = x @ W.T + b
        output = F.linear(x, weights, bias)
        print(output)

        # Expected output (manual calculation):
        # Row 0: [1,2,3,4] @ W.T + bias
        #      = [1,2,3,4] @ [[0.1,0.5,0.9],[0.2,0.6,1.0],[0.3,0.7,1.1],[0.4,0.8,1.2]] + [1,2,3]
        #      = [3.0, 7.0, 11.0] + [1.0, 2.0, 3.0] = [4.0, 9.0, 14.0]
        # Row 1: [5,6,7,8] @ W.T + bias
        #      = [5,6,7,8] @ [[0.1,0.5,0.9],[0.2,0.6,1.0],[0.3,0.7,1.1],[0.4,0.8,1.2]] + [1,2,3]
        #      = [7.0, 17.4, 27.8] + [1.0, 2.0, 3.0] = [8.0, 19.4, 30.8]
        # tensor([[ 4.0000,  9.0000, 14.0000],
        #         [ 8.0000, 19.4000, 30.8000]])
        ```
    """
    # Create input: (2, 4)
    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(4)
    var input = zeros(input_shape, DType.float32)
    input._data.bitcast[Float32]()[0] = 1.0
    input._data.bitcast[Float32]()[1] = 2.0
    input._data.bitcast[Float32]()[2] = 3.0
    input._data.bitcast[Float32]()[3] = 4.0
    input._data.bitcast[Float32]()[4] = 5.0
    input._data.bitcast[Float32]()[5] = 6.0
    input._data.bitcast[Float32]()[6] = 7.0
    input._data.bitcast[Float32]()[7] = 8.0

    # Create weights: (3, 4)
    var weight_shape = List[Int]()
    weight_shape.append(3)
    weight_shape.append(4)
    var weights = zeros(weight_shape, DType.float32)
    weights._data.bitcast[Float32]()[0] = 0.1
    weights._data.bitcast[Float32]()[1] = 0.2
    weights._data.bitcast[Float32]()[2] = 0.3
    weights._data.bitcast[Float32]()[3] = 0.4
    weights._data.bitcast[Float32]()[4] = 0.5
    weights._data.bitcast[Float32]()[5] = 0.6
    weights._data.bitcast[Float32]()[6] = 0.7
    weights._data.bitcast[Float32]()[7] = 0.8
    weights._data.bitcast[Float32]()[8] = 0.9
    weights._data.bitcast[Float32]()[9] = 1.0
    weights._data.bitcast[Float32]()[10] = 1.1
    weights._data.bitcast[Float32]()[11] = 1.2

    # Create bias: (3,)
    var bias_shape = List[Int]()
    bias_shape.append(3)
    var bias = zeros(bias_shape, DType.float32)
    bias._data.bitcast[Float32]()[0] = 1.0
    bias._data.bitcast[Float32]()[1] = 2.0
    bias._data.bitcast[Float32]()[2] = 3.0

    # Forward pass
    var output = linear(input, weights, bias)

    # Validate against corrected reference values
    # Expected output: [[4.0, 9.0, 14.0], [8.0, 19.4, 30.8]]
    assert_almost_equal(output._data.bitcast[Float32]()[0], 4.0, tolerance=1e-5)
    assert_almost_equal(output._data.bitcast[Float32]()[1], 9.0, tolerance=1e-5)
    assert_almost_equal(
        output._data.bitcast[Float32]()[2], 14.0, tolerance=1e-5
    )
    assert_almost_equal(output._data.bitcast[Float32]()[3], 8.0, tolerance=1e-5)
    assert_almost_equal(
        output._data.bitcast[Float32]()[4], 19.4, tolerance=1e-5
    )
    assert_almost_equal(
        output._data.bitcast[Float32]()[5], 30.8, tolerance=1e-5
    )


fn test_relu_matches_pytorch() raises:
    """Test ReLU matches PyTorch implementation numerically.

    PyTorch reference code:
        ```python
        import torch
        import torch.nn.functional as F

        x = torch.tensor([-3.0, -1.5, -0.1, 0.0, 0.1, 1.5, 3.0], dtype=torch.float32)
        output = F.relu(x)
        print(output)

        # Expected output: tensor([0.0, 0.0, 0.0, 0.0, 0.1, 1.5, 3.0])
        ```
    """
    var shape = List[Int]()
    shape.append(7)
    var input = zeros(shape, DType.float32)
    input._data.bitcast[Float32]()[0] = -3.0
    input._data.bitcast[Float32]()[1] = -1.5
    input._data.bitcast[Float32]()[2] = -0.1
    input._data.bitcast[Float32]()[3] = 0.0
    input._data.bitcast[Float32]()[4] = 0.1
    input._data.bitcast[Float32]()[5] = 1.5
    input._data.bitcast[Float32]()[6] = 3.0

    var output = relu(input)

    # Validate against PyTorch reference values
    assert_almost_equal(output._data.bitcast[Float32]()[0], 0.0, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[1], 0.0, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[2], 0.0, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[3], 0.0, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[4], 0.1, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[5], 1.5, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[6], 3.0, tolerance=1e-6)


fn test_sigmoid_matches_pytorch() raises:
    """Test Sigmoid matches PyTorch implementation numerically.

    PyTorch reference code:
        ```python
        import torch
        import torch.nn.functional as F

        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
        output = torch.sigmoid(x)
        print(output)

        # Expected output (approximate):
        # tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])
        ```
    """
    var shape = List[Int]()
    shape.append(5)
    var input = zeros(shape, DType.float32)
    input._data.bitcast[Float32]()[0] = -2.0
    input._data.bitcast[Float32]()[1] = -1.0
    input._data.bitcast[Float32]()[2] = 0.0
    input._data.bitcast[Float32]()[3] = 1.0
    input._data.bitcast[Float32]()[4] = 2.0

    var output = sigmoid(input)

    # Validate against PyTorch reference values (6 decimal places)
    assert_almost_equal(
        output._data.bitcast[Float32]()[0], 0.1192, tolerance=1e-4
    )
    assert_almost_equal(
        output._data.bitcast[Float32]()[1], 0.2689, tolerance=1e-4
    )
    assert_almost_equal(
        output._data.bitcast[Float32]()[2], 0.5000, tolerance=1e-4
    )
    assert_almost_equal(
        output._data.bitcast[Float32]()[3], 0.7311, tolerance=1e-4
    )
    assert_almost_equal(
        output._data.bitcast[Float32]()[4], 0.8808, tolerance=1e-4
    )


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run property-based and PyTorch validation layer tests."""
    print("Running property-based tests...")
    test_layer_property_batch_independence()
    test_layer_property_deterministic()

    print("Running PyTorch validation tests...")
    test_linear_matches_pytorch()
    test_relu_matches_pytorch()
    test_sigmoid_matches_pytorch()

    print("\nAll part3 layer tests passed! ✓")
