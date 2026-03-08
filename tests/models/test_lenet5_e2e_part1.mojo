"""End-to-End Integration Tests for LeNet-5 (Part 1)

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

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_lenet5_e2e.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from shared.core.extensor import ExTensor, zeros, ones
from shared.core.conv import conv2d
from shared.core.pooling import maxpool2d
from shared.core.linear import linear
from shared.core.activation import relu
from shared.core.shape import conv2d_output_shape, pool_output_shape
from shared.core.initializers import kaiming_uniform
from shared.testing.assertions import (
    assert_shape,
    assert_dtype,
    assert_true,
    assert_false,
    assert_close_float,
)
from shared.testing.special_values import (
    create_special_value_tensor,
    create_seeded_random_tensor,
    SPECIAL_VALUE_ONE,
)
from math import isnan, isinf
import os


# ============================================================================
# LeNet-5 Model Definition (for testing)
# ============================================================================


struct LeNet5:
    """Minimal LeNet-5 for testing. Uses same architecture as examples/lenet-emnist/model.mojo
    """

    var num_classes: Int
    var conv1_kernel: ExTensor
    var conv1_bias: ExTensor
    var conv2_kernel: ExTensor
    var conv2_bias: ExTensor
    var fc1_weights: ExTensor
    var fc1_bias: ExTensor
    var fc2_weights: ExTensor
    var fc2_bias: ExTensor
    var fc3_weights: ExTensor
    var fc3_bias: ExTensor

    fn __init__(out self, num_classes: Int = 47) raises:
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

    fn forward(mut self, input: ExTensor) raises -> ExTensor:
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

    fn predict(mut self, input: ExTensor) raises -> Int:
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

    fn parameters(self) raises -> List[ExTensor]:
        """Return all trainable parameters."""
        var params: List[ExTensor] = []
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

    fn zero_grad(mut self) raises:
        """Reset all parameter gradients to zero (no-op for LeNet5)."""
        pass


# ============================================================================
# Forward Pass Tests
# ============================================================================


fn test_forward_output_shape() raises:
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


fn test_forward_single_sample() raises:
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


fn test_forward_batch_sizes() raises:
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


fn test_forward_deterministic() raises:
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
        assert_close_float(val1, val2, 0.0, "Forward pass non-deterministic")


# ============================================================================
# Prediction Tests
# ============================================================================


fn test_predict_single_sample() raises:
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


fn test_predict_output_class_range() raises:
    """Test predict for EMNIST (47 classes)."""
    var model = LeNet5(num_classes=47)

    var input = create_seeded_random_tensor(
        [1, 1, 28, 28], DType.float32, seed=77
    )

    var pred_class = model.predict(input)

    assert_true(pred_class >= 0, "Prediction class (47) < 0")
    assert_true(pred_class < 47, "Prediction class (47) >= 47")


# ============================================================================
# Parameter Tests
# ============================================================================


fn test_parameters_exist() raises:
    """Test that model has correct number of parameters."""
    var model = LeNet5(num_classes=10)

    # Get all parameters
    var params = model.parameters()

    # LeNet5 has 10 parameters: conv1_kernel, conv1_bias, conv2_kernel, conv2_bias,
    #                           fc1_weights, fc1_bias, fc2_weights, fc2_bias,
    #                           fc3_weights, fc3_bias
    assert_true(params.size() == 10, "LeNet5 parameter count mismatch")

    # Verify each parameter is valid
    for i in range(params.size()):
        var param = params[i]
        var numel = param.numel()
        assert_true(numel > 0, "Parameter " + String(i) + " has zero elements")


fn test_parameter_shapes() raises:
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


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    print("Starting LeNet-5 E2E Tests (Part 1)...")

    print("  test_forward_output_shape...", end="")
    test_forward_output_shape()
    print(" OK")

    print("  test_forward_single_sample...", end="")
    test_forward_single_sample()
    print(" OK")

    print("  test_forward_batch_sizes...", end="")
    test_forward_batch_sizes()
    print(" OK")

    print("  test_forward_deterministic...", end="")
    test_forward_deterministic()
    print(" OK")

    print("  test_predict_single_sample...", end="")
    test_predict_single_sample()
    print(" OK")

    print("  test_predict_output_class_range...", end="")
    test_predict_output_class_range()
    print(" OK")

    print("  test_parameters_exist...", end="")
    test_parameters_exist()
    print(" OK")

    print("  test_parameter_shapes...", end="")
    test_parameter_shapes()
    print(" OK")

    print("\nAll LeNet-5 E2E Tests (Part 1) passed!")
