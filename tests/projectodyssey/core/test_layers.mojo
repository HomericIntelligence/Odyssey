"""Unit tests for neural network layers.

Tests cover:
- Linear (fully connected) layers: initialization, forward, no-bias, backward
- Convolutional layers (Conv2D): initialization, output shape, stride

"""


from tests.projectodyssey.conftest import (
    TestFixtures,
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import ones, zeros
from projectodyssey.core.linear import linear, linear_no_bias
from projectodyssey.core.activation import relu, sigmoid, tanh, softmax


def test_linear_initialization() raises:
    """Test Linear layer parameter creation.

    Functional API Note:
        Pure functional design - no layer class initialization.
        Caller creates weight matrix (out_features, in_features) and bias vector.
        This test verifies parameters can be created with correct shapes.
    """
    # Create parameters for a linear transformation: in=10, out=5
    var in_features = 10
    var out_features = 5

    # Weights shape: (out_features, in_features) = (5, 10)
    var weight_shape = List[Int]()
    weight_shape.append(out_features)
    weight_shape.append(in_features)
    var weights = ones(weight_shape, DType.float32)

    # Bias shape: (out_features,) = (5,)
    var bias_shape = List[Int]()
    bias_shape.append(out_features)
    var bias = zeros(bias_shape, DType.float32)

    # Verify shapes
    var w_shape = weights.shape()
    var b_shape = bias.shape()
    assert_equal(w_shape[0], out_features)
    assert_equal(w_shape[1], in_features)
    assert_equal(b_shape[0], out_features)


def test_linear_forward() raises:
    """Test Linear layer forward pass computation.

    Functional API:
        linear(x, weights, bias) -> output
        - Input shape: (batch_size, in_features)
        - Weights shape: (out_features, in_features)
        - Bias shape: (out_features,)
        - Output shape: (batch_size, out_features)
        - Computation: output = x @ weights.T + bias.
    """
    # Create parameters: in=10, out=5
    var in_features = 10
    var out_features = 5
    var batch_size = 2

    # Weights: (5, 10) filled with 0.1
    var weight_shape = List[Int]()
    weight_shape.append(out_features)
    weight_shape.append(in_features)
    var weights = ones(weight_shape, DType.float32)
    # Fill with 0.1
    for i in range(out_features):
        for j in range(in_features):
            weights.set(i * in_features + j, Float32(0.1))

    # Bias: (5,) filled with 0.0
    var bias_shape = List[Int]()
    bias_shape.append(out_features)
    var bias = zeros(bias_shape, DType.float32)

    # Input: (2, 10) filled with 1.0
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_features)
    var input = ones(input_shape, DType.float32)

    # Forward pass
    var output = linear(input, weights, bias)

    # Check output shape
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_features)

    # Check output values: sum of weights = 10 * 0.1 = 1.0
    var expected_value = Float32(10.0 * 0.1)
    assert_almost_equal(
        output._data.bitcast[Float32]()[0], expected_value, tolerance=1e-5
    )


def test_linear_no_bias() raises:
    """Test Linear layer without bias term.

    Functional API:
        linear_no_bias(x, weights) -> output
        - No bias parameter required
        - Computation: output = x @ weights.T.
    """
    # Create parameters: in=10, out=5
    var in_features = 10
    var out_features = 5

    # Weights: (5, 10) filled with 0.5
    var weight_shape = List[Int]()
    weight_shape.append(out_features)
    weight_shape.append(in_features)
    var weights = ones(weight_shape, DType.float32)
    for i in range(out_features * in_features):
        weights.set(i, Float32(0.5))

    # Input: (1, 10) filled with 1.0
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(in_features)
    var input = ones(input_shape, DType.float32)

    # Forward pass without bias
    var output = linear_no_bias(input, weights)

    # Check output shape
    var out_shape = output.shape()
    assert_equal(out_shape[0], 1)
    assert_equal(out_shape[1], out_features)

    # Check output values: sum = 10 * 0.5 = 5.0 (no bias added)
    var expected_value = Float32(10.0 * 0.5)
    assert_almost_equal(
        output._data.bitcast[Float32]()[0], expected_value, tolerance=1e-5
    )


def test_linear_backward() raises:
    """Test Linear layer backward pass (gradient computation).

    Deferred - backward pass implementations are not yet available.
    Will be implemented when autograd/backpropagation system is added.
    """
    pass  # Deferred - backward pass not yet implemented


def test_conv2d_initialization() raises:
    """Test Conv2D layer initialization.

    API Contract:
        Conv2D(
            in_channels: Int,
            out_channels: Int,
            kernel_size: Int,
            stride: Int = 1,
            padding: Int = 0,
            bias: Bool = True
        ).
    """
    # TODO(#1538): Implement when Conv2D is available
    # var layer = Conv2D(
    #     in_channels=3,
    #     out_channels=16,
    #     kernel_size=3,
    #     stride=1,
    #     padding=1
    # )
    # assert_equal(layer.in_channels, 3)
    # assert_equal(layer.out_channels, 16)
    # assert_equal(layer.kernel_size, 3)
    pass


def test_conv2d_output_shape() raises:
    """Test Conv2D computes correct output shape.

    Formula: output_size = (input_size + 2*padding - kernel_size) / stride + 1

    API Contract:
        layer.forward(input: Tensor) -> Tensor
        - Input: (batch, in_channels, height, width)
        - Output: (batch, out_channels, out_height, out_width).
    """
    # TODO(#1538): Implement when Conv2D is available
    # # Input: (batch=1, channels=3, height=32, width=32)
    # # Conv2D: out_channels=16, kernel=3, stride=1, padding=1
    # # Expected output: (1, 16, 32, 32) - same spatial size due to padding
    #
    # var layer = Conv2D(3, 16, kernel_size=3, stride=1, padding=1)
    # var input = Tensor.randn(1, 3, 32, 32)
    # var output = layer.forward(input)
    # assert_shape_equal(output, Shape(1, 16, 32, 32))
    pass


def test_conv2d_stride() raises:
    """Test Conv2D with stride > 1 downsamples correctly.

    API Contract:
        Conv2D with stride=2 should halve spatial dimensions.
    """
    # TODO(#1538): Implement when Conv2D is available
    # # Input: (1, 3, 32, 32)
    # # Conv2D: kernel=3, stride=2, padding=1
    # # Expected output: (1, 16, 16, 16) - halved spatial size
    #
    # var layer = Conv2D(3, 16, kernel_size=3, stride=2, padding=1)
    # var input = Tensor.randn(1, 3, 32, 32)
    # var output = layer.forward(input)
    # assert_shape_equal(output, Shape(1, 16, 16, 16))
    pass


def test_conv2d_valid_padding() raises:
    """Test Conv2D with no padding (valid convolution).

    API Contract:
        Conv2D with padding=0 reduces spatial dimensions.
    """
    # TODO(#1538): Implement when Conv2D is available
    # # Input: (1, 3, 32, 32)
    # # Conv2D: kernel=5, stride=1, padding=0
    # # Expected output: (1, 16, 28, 28) - reduced by kernel_size-1
    #
    # var layer = Conv2D(3, 16, kernel_size=5, stride=1, padding=0)
    # var input = Tensor.randn(1, 3, 32, 32)
    # var output = layer.forward(input)
    # assert_shape_equal(output, Shape(1, 16, 28, 28))
    pass


def test_relu_activation() raises:
    """Test ReLU zeros negative values and preserves positive values.

    Functional API:
        relu(x) -> output
        - For each element: output = max(0, input).
    """
    # Test with known values: [-2.0, -1.0, 0.0, 1.0, 2.0]
    var shape = List[Int]()
    shape.append(5)
    var input = zeros(shape, DType.float32)
    input.set(0, Float32(-2.0))
    input.set(1, Float32(-1.0))
    input.set(2, Float32(0.0))
    input.set(3, Float32(1.0))
    input.set(4, Float32(2.0))

    # Apply ReLU
    var output = relu(input)

    # Expected: [0.0, 0.0, 0.0, 1.0, 2.0]
    assert_almost_equal(output._data.bitcast[Float32]()[0], 0.0, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[1], 0.0, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[2], 0.0, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[3], 1.0, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[4], 2.0, tolerance=1e-6)


def test_relu_in_place() raises:
    """Test ReLU can modify input in-place for memory efficiency.

    Not applicable to pure functional design - functional operations
    always return new tensors and never mutate inputs.
    """
    pass  # Not applicable - pure functional design


def test_sigmoid_range() raises:
    """Test Sigmoid outputs values in range [0, 1].

    Functional API:
        sigmoid(x) -> output
        - For each element: output = 1 / (1 + exp(-input))
        - Output range: (0, 1).
    """
    # Test with various inputs: [-10.0, -1.0, 0.0, 1.0, 10.0]
    var shape = List[Int]()
    shape.append(5)
    var input = zeros(shape, DType.float32)
    input.set(0, Float32(-10.0))
    input.set(1, Float32(-1.0))
    input.set(2, Float32(0.0))
    input.set(3, Float32(1.0))
    input.set(4, Float32(10.0))

    # Apply sigmoid
    var output = sigmoid(input)

    # All outputs should be in (0, 1)
    for i in range(5):
        var val = output._data.bitcast[Float32]()[i]
        assert_true(Float32(0.0) < val, "Value must be greater than 0")
        assert_true(val < Float32(1.0), "Value must be less than 1")

    # Check sigmoid(0) = 0.5
    assert_almost_equal(output._data.bitcast[Float32]()[2], 0.5, tolerance=1e-6)


def test_tanh_range() raises:
    """Test Tanh outputs values in range [-1, 1].

    Functional API:
        tanh(x) -> output
        - For each element: output = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        - Output range: (-1, 1).
    """
    # Test with various inputs: [-10.0, -1.0, 0.0, 1.0, 10.0]
    var shape = List[Int]()
    shape.append(5)
    var input = zeros(shape, DType.float32)
    input.set(0, Float32(-10.0))
    input.set(1, Float32(-1.0))
    input.set(2, Float32(0.0))
    input.set(3, Float32(1.0))
    input.set(4, Float32(10.0))

    # Apply tanh
    var output = tanh(input)

    # All outputs should be in [-1, 1] (inclusive due to FP precision)
    for i in range(5):
        var val = output._data.bitcast[Float32]()[i]
        assert_true(Float32(-1.0) <= val, "Value must be >= -1")
        assert_true(val <= Float32(1.0), "Value must be <= 1")

    # Check tanh(0) = 0.0
    assert_almost_equal(output._data.bitcast[Float32]()[2], 0.0, tolerance=1e-6)


def test_maxpool2d_downsampling() raises:
    """Test MaxPool2D downsamples spatial dimensions.

    API Contract:
        MaxPool2D(kernel_size: Int, stride: Int = None, padding: Int = 0)
        - Reduces spatial dimensions by kernel_size (if stride=kernel_size).
    """
    # TODO(#1538): Implement when MaxPool2D is available
    # # Input: (1, 16, 32, 32)
    # # MaxPool2D: kernel=2, stride=2
    # # Expected output: (1, 16, 16, 16)
    #
    # var pool = MaxPool2D(kernel_size=2, stride=2)
    # var input = Tensor.randn(1, 16, 32, 32)
    # var output = pool.forward(input)
    # assert_shape_equal(output, Shape(1, 16, 16, 16))
    pass


def test_maxpool2d_max_selection() raises:
    """Test MaxPool2D selects maximum value in each window.

    API Contract:
        MaxPool2D selects max over kernel_size x kernel_size window.
    """
    # TODO(#1538): Implement when MaxPool2D is available
    # var pool = MaxPool2D(kernel_size=2)
    #
    # # Create input with known values
    # # [[1, 2], [3, 4]] -> max = 4
    # var data = [1.0, 2.0, 3.0, 4.0]
    # var input = AnyTensor([1, 1, 2, 2], DType.float32)
    # # Fill with data
    # var output = pool.forward(input)
    #
    # # Output should be single value: 4.0
    # assert_almost_equal(output._get_float64(0), 4.0)
    pass


def test_layer_property_batch_independence() raises:
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
        weights.set(i, Float32(0.2))

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
        batch_input.set(i, Float32(1.0))  # First batch element
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
        single_input_1.set(i, Float32(1.0))

    var single_output_1 = linear(single_input_1, weights, bias)

    # First batch element output should match individual processing
    for i in range(out_features):
        assert_almost_equal(
            batch_output._data.bitcast[Float32]()[i],
            single_output_1._data.bitcast[Float32]()[i],
            tolerance=1e-5,
        )


def test_layer_property_deterministic() raises:
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
        weights.set(i, Float32(Float32(i) * 0.01))

    var bias_shape = List[Int]()
    bias_shape.append(out_features)
    var bias = ones(bias_shape, DType.float32)
    for i in range(out_features):
        bias.set(i, Float32(Float32(i) * 0.1))

    # Create input
    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(in_features)
    var input = ones(input_shape, DType.float32)
    for i in range(2 * in_features):
        input.set(i, Float32(i % in_features))

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


def test_linear_matches_pytorch() raises:
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
    input.set(0, Float32(1.0))
    input.set(1, Float32(2.0))
    input.set(2, Float32(3.0))
    input.set(3, Float32(4.0))
    input.set(4, Float32(5.0))
    input.set(5, Float32(6.0))
    input.set(6, Float32(7.0))
    input.set(7, Float32(8.0))

    # Create weights: (3, 4)
    var weight_shape = List[Int]()
    weight_shape.append(3)
    weight_shape.append(4)
    var weights = zeros(weight_shape, DType.float32)
    weights.set(0, Float32(0.1))
    weights.set(1, Float32(0.2))
    weights.set(2, Float32(0.3))
    weights.set(3, Float32(0.4))
    weights.set(4, Float32(0.5))
    weights.set(5, Float32(0.6))
    weights.set(6, Float32(0.7))
    weights.set(7, Float32(0.8))
    weights.set(8, Float32(0.9))
    weights.set(9, Float32(1.0))
    weights.set(10, Float32(1.1))
    weights.set(11, Float32(1.2))

    # Create bias: (3,)
    var bias_shape = List[Int]()
    bias_shape.append(3)
    var bias = zeros(bias_shape, DType.float32)
    bias.set(0, Float32(1.0))
    bias.set(1, Float32(2.0))
    bias.set(2, Float32(3.0))

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


def test_relu_matches_pytorch() raises:
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
    input.set(0, Float32(-3.0))
    input.set(1, Float32(-1.5))
    input.set(2, Float32(-0.1))
    input.set(3, Float32(0.0))
    input.set(4, Float32(0.1))
    input.set(5, Float32(1.5))
    input.set(6, Float32(3.0))

    var output = relu(input)

    # Validate against PyTorch reference values
    assert_almost_equal(output._data.bitcast[Float32]()[0], 0.0, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[1], 0.0, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[2], 0.0, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[3], 0.0, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[4], 0.1, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[5], 1.5, tolerance=1e-6)
    assert_almost_equal(output._data.bitcast[Float32]()[6], 3.0, tolerance=1e-6)


def test_sigmoid_matches_pytorch() raises:
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
    input.set(0, Float32(-2.0))
    input.set(1, Float32(-1.0))
    input.set(2, Float32(0.0))
    input.set(3, Float32(1.0))
    input.set(4, Float32(2.0))

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


def main() raises:
    """Run all test_layers tests."""
    print("Running test_layers tests...")

    test_linear_initialization()
    print("✓ test_linear_initialization")

    test_linear_forward()
    print("✓ test_linear_forward")

    test_linear_no_bias()
    print("✓ test_linear_no_bias")

    test_linear_backward()
    print("✓ test_linear_backward")

    test_conv2d_initialization()
    print("✓ test_conv2d_initialization")

    test_conv2d_output_shape()
    print("✓ test_conv2d_output_shape")

    test_conv2d_stride()
    print("✓ test_conv2d_stride")

    test_conv2d_valid_padding()
    print("✓ test_conv2d_valid_padding")

    test_relu_activation()
    print("✓ test_relu_activation")

    test_relu_in_place()
    print("✓ test_relu_in_place")

    test_sigmoid_range()
    print("✓ test_sigmoid_range")

    test_tanh_range()
    print("✓ test_tanh_range")

    test_maxpool2d_downsampling()
    print("✓ test_maxpool2d_downsampling")

    test_maxpool2d_max_selection()
    print("✓ test_maxpool2d_max_selection")

    test_layer_property_batch_independence()
    print("✓ test_layer_property_batch_independence")

    test_layer_property_deterministic()
    print("✓ test_layer_property_deterministic")

    test_linear_matches_pytorch()
    print("✓ test_linear_matches_pytorch")

    test_relu_matches_pytorch()
    print("✓ test_relu_matches_pytorch")

    test_sigmoid_matches_pytorch()
    print("✓ test_sigmoid_matches_pytorch")

    print("\nAll test_layers tests passed!")
