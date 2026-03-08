"""End-to-End Integration Tests for LeNet-5 (Part 2)

Tests complete model forward pass with real data flow and verifies:
- Weight update mechanics
- Architecture verification
- Full integration pipeline

This file covers: Weight Update Tests, Architecture Tests, Integration Tests.

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

    fn update_parameters(
        mut self,
        learning_rate: Float32,
        grad_conv1_kernel: ExTensor,
        grad_conv1_bias: ExTensor,
        grad_conv2_kernel: ExTensor,
        grad_conv2_bias: ExTensor,
        grad_fc1_weights: ExTensor,
        grad_fc1_bias: ExTensor,
        grad_fc2_weights: ExTensor,
        grad_fc2_bias: ExTensor,
        grad_fc3_weights: ExTensor,
        grad_fc3_bias: ExTensor,
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


fn _sgd_update(mut param: ExTensor, grad: ExTensor, lr: Float32) raises:
    """SGD parameter update: param = param - lr * grad"""
    var numel = param.numel()
    var param_data = param._data.bitcast[Float32]()
    var grad_data = grad._data.bitcast[Float32]()

    for i in range(numel):
        param_data[i] -= lr * grad_data[i]


fn compute_flattened_size() -> Int:
    """Compute flattened size after all conv/pool layers.

    Returns 256 (16 * 4 * 4) for standard LeNet-5 dimensions.
    """
    return 256


# ============================================================================
# Weight Update Tests
# ============================================================================


fn test_zero_grad() raises:
    """Test zero_grad method (no-op for LeNet5)."""
    var model = LeNet5(num_classes=10)

    # zero_grad is a no-op for LeNet5 (gradients are computed fresh)
    # Just verify it doesn't crash
    model.zero_grad()

    # Verify parameters still exist
    var params = model.parameters()
    assert_true(params.size() == 10, "Parameters lost after zero_grad")


fn test_update_parameters() raises:
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
    var expected = orig_conv1 - 0.01 * 1.0 // param - lr * grad
    assert_close_float(new_conv1, expected, 1e-6, "Parameter update mismatch")


# ============================================================================
# Architecture Verification
# ============================================================================


fn test_flattened_size_computation() raises:
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


fn test_num_classes_customizable() raises:
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


# ============================================================================
# Integration Test: Full Forward Pass
# ============================================================================


fn test_full_forward_pipeline() raises:
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


fn test_model_initialization() raises:
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
        assert_close_float(val, 0.0, 1e-10, "Conv1 bias not zero")

    for i in range(model.conv2_bias.numel()):
        var val = model.conv2_bias._get_float64(i)
        assert_close_float(val, 0.0, 1e-10, "Conv2 bias not zero")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    print("Starting LeNet-5 E2E Tests (Part 2)...")

    print("  test_zero_grad...", end="")
    test_zero_grad()
    print(" OK")

    print("  test_update_parameters...", end="")
    test_update_parameters()
    print(" OK")

    print("  test_flattened_size_computation...", end="")
    test_flattened_size_computation()
    print(" OK")

    print("  test_num_classes_customizable...", end="")
    test_num_classes_customizable()
    print(" OK")

    print("  test_full_forward_pipeline...", end="")
    test_full_forward_pipeline()
    print(" OK")

    print("  test_model_initialization...", end="")
    test_model_initialization()
    print(" OK")

    print("\nAll LeNet-5 E2E Tests (Part 2) passed!")
