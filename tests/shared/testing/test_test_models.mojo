"""Tests for SimpleCNN and LinearModel test model definitions.

Split from test_test_models.mojo per ADR-009 to avoid Mojo heap corruption.

Coverage:
    - SimpleCNN initialization and forward pass
    - LinearModel initialization and forward pass
"""


from shared.testing import (
    SimpleCNN,
    LinearModel,
    assert_true,
    assert_equal,
)
from shared.tensor.any_tensor import (
    AnyTensor,
    zeros,
    ones,
    full,
    zeros_like,
)
from shared.testing import (
    SimpleMLP,
    assert_true,
    assert_equal,
)
from shared.tensor.any_tensor import (
    AnyTensor,
    zeros,
    ones,
)
from shared.testing import (
    SimpleMLP,
    assert_equal,
)
from shared.testing import (
    MockLayer,
    SimpleLinearModel,
    assert_equal,
    assert_close_float,
)
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


def test_simple_cnn_initialization() raises:
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


def test_simple_cnn_output_shape() raises:
    """Test SimpleCNN.get_output_shape() method."""
    var cnn = SimpleCNN(1, 8, 10)

    var shape_32 = cnn.get_output_shape(32)
    assert_equal(len(shape_32), 2)
    assert_equal(shape_32[0], 32)
    assert_equal(shape_32[1], 10)

    var shape_64 = cnn.get_output_shape(64)
    assert_equal(shape_64[0], 64)
    assert_equal(shape_64[1], 10)


def test_simple_cnn_forward_pass() raises:
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


def test_simple_cnn_batch_sizes() raises:
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


def test_linear_model_initialization() raises:
    """Test LinearModel initialization."""
    var linear = LinearModel(784, 10)
    assert_equal(linear.in_features, 784)
    assert_equal(linear.out_features, 10)

    var custom_linear = LinearModel(2048, 1024)
    assert_equal(custom_linear.in_features, 2048)
    assert_equal(custom_linear.out_features, 1024)


def test_linear_model_output_shape() raises:
    """Test LinearModel.get_output_shape() method."""
    var linear = LinearModel(784, 10)

    var shape_32 = linear.get_output_shape(32)
    assert_equal(len(shape_32), 2)
    assert_equal(shape_32[0], 32)
    assert_equal(shape_32[1], 10)

    var shape_128 = linear.get_output_shape(128)
    assert_equal(shape_128[0], 128)
    assert_equal(shape_128[1], 10)


def test_linear_model_forward_pass() raises:
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


def test_linear_model_batch_processing() raises:
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


def test_simple_mlp_initialization_1_hidden() raises:
    """Test SimpleMLP with 1 hidden layer."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=1)
    assert_equal(mlp.input_dim, 10)
    assert_equal(mlp.hidden_dim, 20)
    assert_equal(mlp.output_dim, 5)
    assert_equal(mlp.num_hidden_layers, 1)

    # Check weight dimensions
    assert_equal(len(mlp.layer1_weights), 10 * 20)
    assert_equal(len(mlp.layer1_bias), 20)
    assert_equal(len(mlp.layer2_weights), 20 * 5)
    assert_equal(len(mlp.layer2_bias), 5)
    assert_equal(len(mlp.layer3_weights), 0)
    assert_equal(len(mlp.layer3_bias), 0)


def test_simple_mlp_initialization_2_hidden() raises:
    """Test SimpleMLP with 2 hidden layers."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=2)
    assert_equal(mlp.input_dim, 10)
    assert_equal(mlp.hidden_dim, 20)
    assert_equal(mlp.output_dim, 5)
    assert_equal(mlp.num_hidden_layers, 2)

    # Check weight dimensions
    assert_equal(len(mlp.layer1_weights), 10 * 20)
    assert_equal(len(mlp.layer1_bias), 20)
    assert_equal(len(mlp.layer2_weights), 20 * 20)
    assert_equal(len(mlp.layer2_bias), 20)
    assert_equal(len(mlp.layer3_weights), 20 * 5)
    assert_equal(len(mlp.layer3_bias), 5)


def test_simple_mlp_forward_1_hidden() raises:
    """Test SimpleMLP forward pass with 1 hidden layer."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=1)

    # Test with List[Float32] input
    var input = List[Float32]()
    for _ in range(10):
        input.append(1.0)

    var output = mlp.forward(input)
    assert_equal(len(output), 5)


def test_simple_mlp_forward_2_hidden() raises:
    """Test SimpleMLP forward pass with 2 hidden layers."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=2)

    var input = List[Float32]()
    for _ in range(10):
        input.append(1.0)

    var output = mlp.forward(input)
    assert_equal(len(output), 5)


def test_simple_mlp_forward_anytensor_1_hidden() raises:
    """Test SimpleMLP AnyTensor forward pass with 1 hidden layer."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=1)
    var input_shape = [10]
    var input = zeros(input_shape, DType.float32)

    var output = mlp.forward(input)

    # Check output shape
    assert_equal(len(output._shape), 1)
    assert_equal(output._shape[0], 5)

    # Check dtype preserved
    assert_true(
        output._dtype == DType.float32, "Output dtype should be float32"
    )


def test_simple_mlp_forward_anytensor_2_hidden() raises:
    """Test SimpleMLP AnyTensor forward pass with 2 hidden layers."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=2)
    var input_shape = [10]
    var input = ones(input_shape, DType.float32)

    var output = mlp.forward(input)

    # Check output shape
    assert_equal(len(output._shape), 1)
    assert_equal(output._shape[0], 5)


def test_simple_mlp_num_parameters_1_hidden() raises:
    """Test SimpleMLP parameter count with 1 hidden layer."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=1)
    # Layer1: 10*20 weights + 20 bias = 220
    # Layer2: 20*5 weights + 5 bias = 105
    # Total: 325
    assert_equal(mlp.num_parameters(), 325)


def test_simple_mlp_num_parameters_2_hidden() raises:
    """Test SimpleMLP parameter count with 2 hidden layers."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=2)
    # Layer1: 10*20 weights + 20 bias = 220
    # Layer2: 20*20 weights + 20 bias = 420
    # Layer3: 20*5 weights + 5 bias = 105
    # Total: 745
    assert_equal(mlp.num_parameters(), 745)


def test_simple_mlp_get_weights() raises:
    """Test SimpleMLP.get_weights() method."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=1)
    var weights = mlp.get_weights()

    # Should contain all weights and biases
    var expected_size = 10 * 20 + 20 + 20 * 5 + 5
    assert_equal(weights.numel(), expected_size)


def test_simple_mlp_parameters() raises:
    """Test SimpleMLP.parameters() method."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=1)
    var params = mlp.parameters()

    # Should have 4 parameter tensors (w1, b1, w2, b2)
    assert_equal(len(params), 4)

    # Check shapes
    assert_equal(params[0].numel(), 10 * 20)  # w1
    assert_equal(params[1].numel(), 20)  # b1
    assert_equal(params[2].numel(), 20 * 5)  # w2
    assert_equal(params[3].numel(), 5)  # b2


def test_simple_mlp_state_dict_1_hidden() raises:
    """Test SimpleMLP.state_dict() with 1 hidden layer."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=1)
    var state = mlp.state_dict()

    # Should have 4 entries: layer1_weights, layer1_bias, layer2_weights, layer2_bias
    assert_equal(len(state), 4)


def test_simple_mlp_state_dict_2_hidden() raises:
    """Test SimpleMLP.state_dict() with 2 hidden layers."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=2)
    var state = mlp.state_dict()

    # Should have 6 entries
    assert_equal(len(state), 6)


def test_simple_mlp_zero_grad() raises:
    """Test SimpleMLP.zero_grad() method."""
    var mlp = SimpleMLP(10, 20, 5)
    # This should not raise
    mlp.zero_grad()


def test_mock_layer_initialization() raises:
    """Test MockLayer initialization."""
    var layer = MockLayer(10, 5)
    assert_equal(layer.input_dim, 10)
    assert_equal(layer.output_dim, 5)
    assert_equal(layer.scale, 1.0)

    var scaled_layer = MockLayer(20, 10, scale=2.0)
    assert_equal(scaled_layer.scale, 2.0)


def test_mock_layer_forward_truncate() raises:
    """Test MockLayer forward pass with truncation."""
    var layer = MockLayer(10, 5, scale=1.0)

    var input = List[Float32]()
    for i in range(10):
        input.append(Float32(i))

    var output = layer.forward(input)

    assert_equal(len(output), 5)
    for i in range(5):
        assert_equal(output[i], Float32(i))


def test_mock_layer_forward_pad() raises:
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


def test_mock_layer_forward_scale() raises:
    """Test MockLayer forward pass with scaling."""
    var layer = MockLayer(5, 5, scale=2.0)

    var input = List[Float32]()
    for _ in range(5):
        input.append(1.0)

    var output = layer.forward(input)

    assert_equal(len(output), 5)
    for i in range(5):
        assert_equal(output[i], 2.0)


def test_mock_layer_num_parameters() raises:
    """Test MockLayer.num_parameters() method."""
    var layer = MockLayer(10, 5)
    assert_equal(layer.num_parameters(), 50)

    var layer2 = MockLayer(20, 15)
    assert_equal(layer2.num_parameters(), 300)


def test_simple_linear_model_initialization() raises:
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


def test_simple_linear_model_custom_init_value() raises:
    """Test SimpleLinearModel with custom init value."""
    var model = SimpleLinearModel(10, 5, use_bias=True, init_value=0.5)

    for i in range(len(model.weights)):
        assert_equal(model.weights[i], 0.5)

    for i in range(len(model.bias)):
        assert_equal(model.bias[i], 0.5)


def test_simple_linear_model_forward() raises:
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


def test_simple_linear_model_no_bias() raises:
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


def test_simple_linear_model_num_parameters() raises:
    """Test SimpleLinearModel parameter counting."""
    var model_with_bias = SimpleLinearModel(10, 5, use_bias=True)
    assert_equal(model_with_bias.num_parameters(), 55)  # 50 + 5

    var model_no_bias = SimpleLinearModel(10, 5, use_bias=False)
    assert_equal(model_no_bias.num_parameters(), 50)  # 50 only


def test_parameter_initialization() raises:
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


def test_parameter_shape() raises:
    """Test Parameter.shape() method."""
    var shape = [20, 15]

    var data = zeros(shape, DType.float32)
    var param = Parameter(data)

    var param_shape = param.shape()
    assert_equal(len(param_shape), 2)
    assert_equal(param_shape[0], 20)
    assert_equal(param_shape[1], 15)


def test_create_test_cnn() raises:
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


def test_create_linear_model() raises:
    """Test create_linear_model factory function."""
    # Default parameters
    var linear1 = create_linear_model()
    assert_equal(linear1.in_features, 784)
    assert_equal(linear1.out_features, 10)

    # Custom parameters
    var linear2 = create_linear_model(2048, 1024)
    assert_equal(linear2.in_features, 2048)
    assert_equal(linear2.out_features, 1024)


def test_multiple_models_forward() raises:
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


def test_mlp_with_different_configs() raises:
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


def main() raises:
    """Run all test_test_models tests."""
    print("Running test_test_models tests...")

    test_simple_cnn_initialization()
    print("✓ test_simple_cnn_initialization")

    test_simple_cnn_output_shape()
    print("✓ test_simple_cnn_output_shape")

    test_simple_cnn_forward_pass()
    print("✓ test_simple_cnn_forward_pass")

    test_simple_cnn_batch_sizes()
    print("✓ test_simple_cnn_batch_sizes")

    test_linear_model_initialization()
    print("✓ test_linear_model_initialization")

    test_linear_model_output_shape()
    print("✓ test_linear_model_output_shape")

    test_linear_model_forward_pass()
    print("✓ test_linear_model_forward_pass")

    test_linear_model_batch_processing()
    print("✓ test_linear_model_batch_processing")

    test_simple_mlp_initialization_1_hidden()
    print("✓ test_simple_mlp_initialization_1_hidden")

    test_simple_mlp_initialization_2_hidden()
    print("✓ test_simple_mlp_initialization_2_hidden")

    test_simple_mlp_forward_1_hidden()
    print("✓ test_simple_mlp_forward_1_hidden")

    test_simple_mlp_forward_2_hidden()
    print("✓ test_simple_mlp_forward_2_hidden")

    test_simple_mlp_forward_anytensor_1_hidden()
    print("✓ test_simple_mlp_forward_anytensor_1_hidden")

    test_simple_mlp_forward_anytensor_2_hidden()
    print("✓ test_simple_mlp_forward_anytensor_2_hidden")

    test_simple_mlp_num_parameters_1_hidden()
    print("✓ test_simple_mlp_num_parameters_1_hidden")

    test_simple_mlp_num_parameters_2_hidden()
    print("✓ test_simple_mlp_num_parameters_2_hidden")

    test_simple_mlp_get_weights()
    print("✓ test_simple_mlp_get_weights")

    test_simple_mlp_parameters()
    print("✓ test_simple_mlp_parameters")

    test_simple_mlp_state_dict_1_hidden()
    print("✓ test_simple_mlp_state_dict_1_hidden")

    test_simple_mlp_state_dict_2_hidden()
    print("✓ test_simple_mlp_state_dict_2_hidden")

    test_simple_mlp_zero_grad()
    print("✓ test_simple_mlp_zero_grad")

    test_mock_layer_initialization()
    print("✓ test_mock_layer_initialization")

    test_mock_layer_forward_truncate()
    print("✓ test_mock_layer_forward_truncate")

    test_mock_layer_forward_pad()
    print("✓ test_mock_layer_forward_pad")

    test_mock_layer_forward_scale()
    print("✓ test_mock_layer_forward_scale")

    test_mock_layer_num_parameters()
    print("✓ test_mock_layer_num_parameters")

    test_simple_linear_model_initialization()
    print("✓ test_simple_linear_model_initialization")

    test_simple_linear_model_custom_init_value()
    print("✓ test_simple_linear_model_custom_init_value")

    test_simple_linear_model_forward()
    print("✓ test_simple_linear_model_forward")

    test_simple_linear_model_no_bias()
    print("✓ test_simple_linear_model_no_bias")

    test_simple_linear_model_num_parameters()
    print("✓ test_simple_linear_model_num_parameters")

    test_parameter_initialization()
    print("✓ test_parameter_initialization")

    test_parameter_shape()
    print("✓ test_parameter_shape")

    test_create_test_cnn()
    print("✓ test_create_test_cnn")

    test_create_linear_model()
    print("✓ test_create_linear_model")

    test_multiple_models_forward()
    print("✓ test_multiple_models_forward")

    test_mlp_with_different_configs()
    print("✓ test_mlp_with_different_configs")

    print("\nAll test_test_models tests passed!")
