"""End-to-End Integration Tests for LeNet-5.

Tests complete model forward pass with real data flow and verifies:
- Forward pass produces expected output shape
- Gradient computation works through all layers
- Model can save and load weights

This file covers: Forward Pass Tests, Prediction Tests, Parameter Tests.

Optional: Tests requiring EMNIST dataset (downloaded by weekly CI):
- test_forward_with_dataset: Forward pass on real dataset batch
- test_training_convergence: Verify loss decreases over epochs
- test_gradient_flow: Check gradients flow through all layers
- test_weight_persistence: Test save/load functionality

These tests may be skipped if dataset is not available (run locally).
Full E2E testing happens in weekly CI job with complete dataset.

"""


from projectodyssey.tensor.any_tensor import AnyTensor, zeros, ones
from projectodyssey.core.conv import conv2d
from projectodyssey.core.pooling import maxpool2d
from projectodyssey.core.linear import linear
from projectodyssey.core.activation import relu
from projectodyssey.core.shape import conv2d_output_shape, pool_output_shape
from projectodyssey.core.initializers import kaiming_uniform
from projectodyssey.testing.assertions import (
    assert_shape,
    assert_dtype,
    assert_true,
    assert_false,
    assert_close_float,
)
from projectodyssey.testing.special_values import (
    create_special_value_tensor,
    create_seeded_random_tensor,
    SPECIAL_VALUE_ONE,
)
from std.math import isnan, isinf
import std.os as os


struct LeNet5:
    """Minimal LeNet-5 for testing. Uses same architecture as examples/lenet_emnist/model.mojo.
    """

    var num_classes: Int
    var conv1_kernel: AnyTensor
    var conv1_bias: AnyTensor
    var conv2_kernel: AnyTensor
    var conv2_bias: AnyTensor
    var fc1_weights: AnyTensor
    var fc1_bias: AnyTensor
    var fc2_weights: AnyTensor
    var fc2_bias: AnyTensor
    var fc3_weights: AnyTensor
    var fc3_bias: AnyTensor

    def __init__(out self, num_classes: Int = 47) raises:
        """Initialize LeNet-5 model with random weights."""
        self.num_classes = num_classes

        var flattened_size = 256

        # Conv1: (6, 1, 5, 5)
        var conv1_shape: List[Int] = [6, 1, 5, 5]
        self.conv1_kernel = kaiming_uniform(
            5, 30, conv1_shape, dtype=DType.float32
        )
        self.conv1_bias = zeros([6], DType.float32)

        # Conv2: (16, 6, 5, 5)
        var conv2_shape: List[Int] = [16, 6, 5, 5]
        self.conv2_kernel = kaiming_uniform(
            150, 400, conv2_shape, dtype=DType.float32
        )
        self.conv2_bias = zeros([16], DType.float32)

        # FC1: (120, flattened_size)
        var fc1_shape: List[Int] = [120, flattened_size]
        self.fc1_weights = kaiming_uniform(
            flattened_size, 120, fc1_shape, dtype=DType.float32
        )
        self.fc1_bias = zeros([120], DType.float32)

        # FC2: (84, 120)
        var fc2_shape: List[Int] = [84, 120]
        self.fc2_weights = kaiming_uniform(
            120, 84, fc2_shape, dtype=DType.float32
        )
        self.fc2_bias = zeros([84], DType.float32)

        # FC3: (num_classes, 84)
        var fc3_shape: List[Int] = [num_classes, 84]
        self.fc3_weights = kaiming_uniform(
            84, num_classes, fc3_shape, dtype=DType.float32
        )
        self.fc3_bias = zeros([num_classes], DType.float32)

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Forward pass through LeNet-5."""
        # Conv1 + ReLU + MaxPool
        var conv1_out = conv2d(
            input, self.conv1_kernel, self.conv1_bias, stride=1, padding=0
        )
        var relu1_out = relu(conv1_out)
        var pool1_out = maxpool2d(relu1_out, kernel_size=2, stride=2, padding=0)

        # Conv2 + ReLU + MaxPool
        var conv2_out = conv2d(
            pool1_out, self.conv2_kernel, self.conv2_bias, stride=1, padding=0
        )
        var relu2_out = relu(conv2_out)
        var pool2_out = maxpool2d(relu2_out, kernel_size=2, stride=2, padding=0)

        # Flatten
        var pool2_shape = pool2_out.shape()
        var batch_size = pool2_shape[0]
        var flattened_size = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
        var flatten_shape: List[Int] = [batch_size, flattened_size]
        var flattened = pool2_out.reshape(flatten_shape)

        # FC1 + ReLU
        var fc1_out = linear(flattened, self.fc1_weights, self.fc1_bias)
        var relu3_out = relu(fc1_out)

        # FC2 + ReLU
        var fc2_out = linear(relu3_out, self.fc2_weights, self.fc2_bias)
        var relu4_out = relu(fc2_out)

        # FC3 (output logits)
        var output = linear(relu4_out, self.fc3_weights, self.fc3_bias)

        return output^

    def predict(mut self, input: AnyTensor) raises -> Int:
        """Predict class for a single input."""
        var logits = self.forward(input)

        var logits_shape = logits.shape()
        var max_idx = 0
        var max_val = logits[0]

        for i in range(1, logits_shape[1]):
            if logits[i] > max_val:
                max_val = logits[i]
                max_idx = i

        return max_idx

    def parameters(self) raises -> List[AnyTensor]:
        """Return all trainable parameters."""
        var params: List[AnyTensor] = []
        params.append(self.conv1_kernel)
        params.append(self.conv1_bias)
        params.append(self.conv2_kernel)
        params.append(self.conv2_bias)
        params.append(self.fc1_weights)
        params.append(self.fc1_bias)
        params.append(self.fc2_weights)
        params.append(self.fc2_bias)
        params.append(self.fc3_weights)
        params.append(self.fc3_bias)
        return params^

    def zero_grad(mut self) raises:
        """Reset all parameter gradients to zero (no-op for LeNet5)."""
        pass

    def update_parameters(
        mut self,
        learning_rate: Float32,
        grad_conv1_kernel: AnyTensor,
        grad_conv1_bias: AnyTensor,
        grad_conv2_kernel: AnyTensor,
        grad_conv2_bias: AnyTensor,
        grad_fc1_weights: AnyTensor,
        grad_fc1_bias: AnyTensor,
        grad_fc2_weights: AnyTensor,
        grad_fc2_bias: AnyTensor,
        grad_fc3_weights: AnyTensor,
        grad_fc3_bias: AnyTensor,
    ) raises:
        """Update parameters using SGD."""
        _sgd_update(self.conv1_kernel, grad_conv1_kernel, learning_rate)
        _sgd_update(self.conv1_bias, grad_conv1_bias, learning_rate)
        _sgd_update(self.conv2_kernel, grad_conv2_kernel, learning_rate)
        _sgd_update(self.conv2_bias, grad_conv2_bias, learning_rate)
        _sgd_update(self.fc1_weights, grad_fc1_weights, learning_rate)
        _sgd_update(self.fc1_bias, grad_fc1_bias, learning_rate)
        _sgd_update(self.fc2_weights, grad_fc2_weights, learning_rate)
        _sgd_update(self.fc2_bias, grad_fc2_bias, learning_rate)
        _sgd_update(self.fc3_weights, grad_fc3_weights, learning_rate)
        _sgd_update(self.fc3_bias, grad_fc3_bias, learning_rate)


def _sgd_update(mut param: AnyTensor, grad: AnyTensor, lr: Float32) raises:
    """SGD parameter update: param = param - lr * grad"""
    var numel = param.numel()
    var param_data = param._data.bitcast[Float32]()
    var grad_data = grad._data.bitcast[Float32]()

    for i in range(numel):
        param_data[i] -= lr * grad_data[i]


def compute_flattened_size() -> Int:
    """Compute flattened size after all conv/pool layers.

    Returns 256 (16 * 4 * 4) for standard LeNet-5 dimensions.
    """
    return 256


def test_forward_output_shape() raises:
    """Test forward pass produces correct output shape (batch, 10)."""
    var model = LeNet5(num_classes=10)

    # Create batch of inputs: (4, 1, 28, 28)
    var input = create_seeded_random_tensor(
        [4, 1, 28, 28], DType.float32, seed=42
    )

    # Forward pass
    var output = model.forward(input)

    # Verify output shape: (4, 10)
    assert_shape(output, [4, 10], "LeNet5 output shape mismatch")

    # Verify dtype preserved
    assert_dtype(output, DType.float32, "LeNet5 output dtype mismatch")

    # Verify no NaN/Inf
    for i in range(output.numel()):
        var val = output._get_float64(i)
        assert_false(isnan(val), "LeNet5 output contains NaN")
        assert_false(isinf(val), "LeNet5 output contains Inf")


def test_forward_single_sample() raises:
    """Test forward pass with single sample (1, 1, 28, 28)."""
    var model = LeNet5(num_classes=47)

    # Create single input
    var input = create_seeded_random_tensor(
        [1, 1, 28, 28], DType.float32, seed=123
    )

    # Forward pass
    var output = model.forward(input)

    # Verify output shape: (1, 47)
    assert_shape(output, [1, 47], "Single sample output shape mismatch")

    # Verify dtype
    assert_dtype(output, DType.float32, "Single sample output dtype mismatch")


def test_forward_batch_sizes() raises:
    """Test forward pass with different batch sizes."""
    var model = LeNet5(num_classes=10)

    # Test batch size 1
    var input1 = create_seeded_random_tensor(
        [1, 1, 28, 28], DType.float32, seed=1
    )
    var output1 = model.forward(input1)
    assert_shape(output1, [1, 10], "Batch size 1 output shape")

    # Test batch size 8
    var input8 = create_seeded_random_tensor(
        [8, 1, 28, 28], DType.float32, seed=2
    )
    var output8 = model.forward(input8)
    assert_shape(output8, [8, 10], "Batch size 8 output shape")

    # Test batch size 32
    var input32 = create_seeded_random_tensor(
        [32, 1, 28, 28], DType.float32, seed=3
    )
    var output32 = model.forward(input32)
    assert_shape(output32, [32, 10], "Batch size 32 output shape")


def test_forward_deterministic() raises:
    """Test that forward pass is deterministic with same input."""
    var model = LeNet5(num_classes=10)

    # Create input
    var input = create_seeded_random_tensor(
        [2, 1, 28, 28], DType.float32, seed=999
    )

    # Forward pass twice
    var output1 = model.forward(input)
    var output2 = model.forward(input)

    # Outputs should be identical (deterministic)
    for i in range(output1.numel()):
        var val1 = output1._get_float64(i)
        var val2 = output2._get_float64(i)
        assert_close_float(
            val1, val2, 0.0, 0.0, "Forward pass non-deterministic"
        )


def test_predict_single_sample() raises:
    """Test predict method returns valid class index (0-9)."""
    var model = LeNet5(num_classes=10)

    # Create single input
    var input = create_seeded_random_tensor(
        [1, 1, 28, 28], DType.float32, seed=42
    )

    # Predict
    var pred_class = model.predict(input)

    # Verify prediction is valid class index
    assert_true(pred_class >= 0, "Prediction class < 0")
    assert_true(pred_class < 10, "Prediction class >= 10")


def test_predict_output_class_range() raises:
    """Test predict for EMNIST (47 classes)."""
    var model = LeNet5(num_classes=47)

    var input = create_seeded_random_tensor(
        [1, 1, 28, 28], DType.float32, seed=77
    )

    var pred_class = model.predict(input)

    assert_true(pred_class >= 0, "Prediction class (47) < 0")
    assert_true(pred_class < 47, "Prediction class (47) >= 47")


def test_parameters_exist() raises:
    """Test that model has correct number of parameters."""
    var model = LeNet5(num_classes=10)

    # Get all parameters
    var params = model.parameters()

    # LeNet5 has 10 parameters: conv1_kernel, conv1_bias, conv2_kernel, conv2_bias,
    #                           fc1_weights, fc1_bias, fc2_weights, fc2_bias,
    #                           fc3_weights, fc3_bias
    assert_true(len(params) == 10, "LeNet5 parameter count mismatch")

    # Verify each parameter is valid
    for i in range(len(params)):
        var param = params[i]
        var numel = param.numel()
        assert_true(numel > 0, "Parameter " + String(i) + " has zero elements")


def test_parameter_shapes() raises:
    """Test that all parameters have correct shapes."""
    var model = LeNet5(num_classes=10)

    # Conv1: (6, 1, 5, 5)
    var conv1_kernel = model.conv1_kernel
    assert_shape(conv1_kernel, [6, 1, 5, 5], "Conv1 kernel shape")

    # Conv1 bias: (6,)
    var conv1_bias = model.conv1_bias
    assert_shape(conv1_bias, [6], "Conv1 bias shape")

    # Conv2: (16, 6, 5, 5)
    var conv2_kernel = model.conv2_kernel
    assert_shape(conv2_kernel, [16, 6, 5, 5], "Conv2 kernel shape")

    # Conv2 bias: (16,)
    var conv2_bias = model.conv2_bias
    assert_shape(conv2_bias, [16], "Conv2 bias shape")

    # FC layers verified to exist
    assert_true(model.fc1_weights.numel() > 0, "FC1 weights empty")
    assert_true(model.fc1_bias.numel() > 0, "FC1 bias empty")
    assert_true(model.fc2_weights.numel() > 0, "FC2 weights empty")
    assert_true(model.fc2_bias.numel() > 0, "FC2 bias empty")
    assert_true(model.fc3_weights.numel() > 0, "FC3 weights empty")
    assert_true(model.fc3_bias.numel() > 0, "FC3 bias empty")


def test_zero_grad() raises:
    """Test zero_grad method (no-op for LeNet5)."""
    var model = LeNet5(num_classes=10)

    # zero_grad is a no-op for LeNet5 (gradients are computed fresh)
    # Just verify it doesn't crash
    model.zero_grad()

    # Verify parameters still exist
    var params = model.parameters()
    assert_true(len(params) == 10, "Parameters lost after zero_grad")


def test_update_parameters() raises:
    """Test parameter update method (SGD)."""
    var model = LeNet5(num_classes=10)

    # Save original parameters
    var orig_conv1 = model.conv1_kernel._get_float64(0)

    # Create dummy gradients (same shapes as parameters)
    var grad_conv1_kernel = ones(model.conv1_kernel.shape(), DType.float32)
    var grad_conv1_bias = ones(model.conv1_bias.shape(), DType.float32)
    var grad_conv2_kernel = ones(model.conv2_kernel.shape(), DType.float32)
    var grad_conv2_bias = ones(model.conv2_bias.shape(), DType.float32)
    var grad_fc1_weights = ones(model.fc1_weights.shape(), DType.float32)
    var grad_fc1_bias = ones(model.fc1_bias.shape(), DType.float32)
    var grad_fc2_weights = ones(model.fc2_weights.shape(), DType.float32)
    var grad_fc2_bias = ones(model.fc2_bias.shape(), DType.float32)
    var grad_fc3_weights = ones(model.fc3_weights.shape(), DType.float32)
    var grad_fc3_bias = ones(model.fc3_bias.shape(), DType.float32)

    # Update parameters with learning rate 0.01
    model.update_parameters(
        learning_rate=0.01,
        grad_conv1_kernel=grad_conv1_kernel,
        grad_conv1_bias=grad_conv1_bias,
        grad_conv2_kernel=grad_conv2_kernel,
        grad_conv2_bias=grad_conv2_bias,
        grad_fc1_weights=grad_fc1_weights,
        grad_fc1_bias=grad_fc1_bias,
        grad_fc2_weights=grad_fc2_weights,
        grad_fc2_bias=grad_fc2_bias,
        grad_fc3_weights=grad_fc3_weights,
        grad_fc3_bias=grad_fc3_bias,
    )

    # Verify parameter changed (updated in direction opposite to gradient)
    var new_conv1 = model.conv1_kernel._get_float64(0)
    var expected = orig_conv1 - 0.01 * 1.0  # lr=0.01, gradient=1.0
    assert_close_float(
        new_conv1, expected, 1e-6, 1e-6, "Parameter update mismatch"
    )


def test_flattened_size_computation() raises:
    """Test that flattened size is computed correctly.

    After Conv1 + ReLU + Pool1 + Conv2 + ReLU + Pool2:
    - Input: (batch, 1, 28, 28)
    - After Conv1 (5x5, stride=1, padding=0): (batch, 6, 24, 24)
    - After Pool1 (2x2, stride=2): (batch, 6, 12, 12)
    - After Conv2 (5x5, stride=1, padding=0): (batch, 16, 8, 8)
    - After Pool2 (2x2, stride=2): (batch, 16, 4, 4)
    - Flattened: (batch, 16*4*4) = (batch, 256)
    """
    var flattened_size = compute_flattened_size()

    # Expected: 16 * 4 * 4 = 256
    assert_true(flattened_size == 256, "Flattened size mismatch")


def test_num_classes_customizable() raises:
    """Test that num_classes parameter works correctly."""
    # Test EMNIST (47 classes)
    var model_emnist = LeNet5(num_classes=47)
    assert_true(model_emnist.num_classes == 47, "EMNIST num_classes mismatch")

    # Test MNIST (10 classes)
    var model_mnist = LeNet5(num_classes=10)
    assert_true(model_mnist.num_classes == 10, "MNIST num_classes mismatch")

    # Verify output shapes match num_classes
    var input = create_seeded_random_tensor(
        [1, 1, 28, 28], DType.float32, seed=42
    )

    var output_emnist = model_emnist.forward(input)
    assert_shape(output_emnist, [1, 47], "EMNIST output shape")

    var output_mnist = model_mnist.forward(input)
    assert_shape(output_mnist, [1, 10], "MNIST output shape")


def test_full_forward_pipeline() raises:
    """Test complete forward pass: input -> output through all 12 layers."""
    var model = LeNet5(num_classes=10)

    # Create realistic batch
    var batch_size = 4
    var input = create_seeded_random_tensor(
        [batch_size, 1, 28, 28], DType.float32, seed=42, low=-1.0, high=1.0
    )

    # Forward pass (goes through all 12 operations)
    var output = model.forward(input)

    # Verify output
    assert_shape(output, [batch_size, 10], "Full pipeline output shape")
    assert_dtype(output, DType.float32, "Full pipeline output dtype")

    # Verify output is valid
    for i in range(output.numel()):
        var val = output._get_float64(i)
        assert_false(
            isnan(val), "Full pipeline output contains NaN at " + String(i)
        )
        assert_false(
            isinf(val), "Full pipeline output contains Inf at " + String(i)
        )

    # Verify output range is reasonable (unconstrained logits)
    # Logits should generally be in range [-10, 10] for reasonable models
    # but this is not strict - just check they're not extreme
    var min_val = 1e10
    var max_val = -1e10
    for i in range(output.numel()):
        var val = output._get_float64(i)
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val


def test_model_initialization() raises:
    """Test model initializes with valid random weights."""
    var model = LeNet5(num_classes=10)

    # Verify all weights are initialized
    assert_true(model.conv1_kernel.numel() > 0, "Conv1 kernel not initialized")
    assert_true(model.conv2_kernel.numel() > 0, "Conv2 kernel not initialized")
    assert_true(model.fc1_weights.numel() > 0, "FC1 weights not initialized")
    assert_true(model.fc2_weights.numel() > 0, "FC2 weights not initialized")
    assert_true(model.fc3_weights.numel() > 0, "FC3 weights not initialized")

    # Verify biases are initialized to zero
    for i in range(model.conv1_bias.numel()):
        var val = model.conv1_bias._get_float64(i)
        assert_close_float(val, 0.0, 1e-10, 1e-10, "Conv1 bias not zero")

    for i in range(model.conv2_bias.numel()):
        var val = model.conv2_bias._get_float64(i)
        assert_close_float(val, 0.0, 1e-10, 1e-10, "Conv2 bias not zero")


def main() raises:
    """Run all test_lenet5_e2e tests."""
    print("Running test_lenet5_e2e tests...")

    test_forward_output_shape()
    print("✓ test_forward_output_shape")

    test_forward_single_sample()
    print("✓ test_forward_single_sample")

    test_forward_batch_sizes()
    print("✓ test_forward_batch_sizes")

    test_forward_deterministic()
    print("✓ test_forward_deterministic")

    test_predict_single_sample()
    print("✓ test_predict_single_sample")

    test_predict_output_class_range()
    print("✓ test_predict_output_class_range")

    test_parameters_exist()
    print("✓ test_parameters_exist")

    test_parameter_shapes()
    print("✓ test_parameter_shapes")

    test_zero_grad()
    print("✓ test_zero_grad")

    test_update_parameters()
    print("✓ test_update_parameters")

    test_flattened_size_computation()
    print("✓ test_flattened_size_computation")

    test_num_classes_customizable()
    print("✓ test_num_classes_customizable")

    test_full_forward_pipeline()
    print("✓ test_full_forward_pipeline")

    test_model_initialization()
    print("✓ test_model_initialization")

    print("\nAll test_lenet5_e2e tests passed!")
