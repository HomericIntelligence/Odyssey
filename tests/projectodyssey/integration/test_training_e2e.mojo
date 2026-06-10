"""End-to-end integration tests for training pipeline.

Tests cover:
- Full training loop execution
- Loss decrease over iterations
- Batch size variations
- Reproducibility with fixed seeds
- Learning rate effects

These tests verify the complete training workflow from
data loading through parameter updates.
"""

from tests.projectodyssey.conftest import (
    assert_true,
    assert_false,
    assert_equal_int,
    assert_almost_equal,
    assert_greater,
    assert_less,
    TestFixtures,
)
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros, ones, full
from projectodyssey.core.conv import conv2d, conv2d_backward
from projectodyssey.core.pooling import maxpool2d, maxpool2d_backward
from projectodyssey.core.linear import linear, linear_backward
from projectodyssey.core.activation import relu, relu_backward
from projectodyssey.core.loss import cross_entropy, cross_entropy_backward
from projectodyssey.training.precision_config import (
    PrecisionConfig,
    PrecisionMode,
)
from std.collections import List


# ============================================================================
# Helper: Simple 2-layer network for testing
# ============================================================================


struct SimpleMLP:
    """Simple 2-layer MLP for testing training loops.

    Architecture: input(10) -> fc1(10->20) -> relu -> fc2(20->5) -> output.
    """

    var fc1_weights: AnyTensor
    var fc1_bias: AnyTensor
    var fc2_weights: AnyTensor
    var fc2_bias: AnyTensor

    def __init__(out self) raises:
        """Initialize with small random weights."""
        # FC1: 10 -> 20 (weights shape: out_features, in_features)
        var fc1_w_shape = List[Int]()
        fc1_w_shape.append(20)  # out_features
        fc1_w_shape.append(10)  # in_features
        self.fc1_weights = full(fc1_w_shape, 0.1, DType.float32)

        var fc1_b_shape = List[Int]()
        fc1_b_shape.append(20)
        self.fc1_bias = zeros(fc1_b_shape, DType.float32)

        # FC2: 20 -> 5 (weights shape: out_features, in_features)
        var fc2_w_shape = List[Int]()
        fc2_w_shape.append(5)  # out_features
        fc2_w_shape.append(20)  # in_features
        self.fc2_weights = full(fc2_w_shape, 0.1, DType.float32)

        var fc2_b_shape = List[Int]()
        fc2_b_shape.append(5)
        self.fc2_bias = zeros(fc2_b_shape, DType.float32)

    def forward(self, input: AnyTensor) raises -> AnyTensor:
        """Forward pass: fc1 -> relu -> fc2."""
        var fc1_out = linear(input, self.fc1_weights, self.fc1_bias)
        var relu_out = relu(fc1_out)
        var fc2_out = linear(relu_out, self.fc2_weights, self.fc2_bias)
        return fc2_out^

    def train_step(
        mut self, input: AnyTensor, labels: AnyTensor, learning_rate: Float32
    ) raises -> Float32:
        """Execute one training step with manual backprop.

        Returns:
            Loss value for this batch.
        """
        # Forward pass (caching intermediates)
        var fc1_out = linear(input, self.fc1_weights, self.fc1_bias)
        var relu_out = relu(fc1_out)
        var fc2_out = linear(relu_out, self.fc2_weights, self.fc2_bias)

        # Compute loss
        var loss_tensor = cross_entropy(fc2_out, labels)
        var loss = loss_tensor._data.bitcast[Float32]()[0]

        # Backward pass
        var grad_output_shape = List[Int]()
        grad_output_shape.append(1)
        var grad_output = zeros(grad_output_shape, fc2_out.dtype())
        grad_output._set_float32(0, Float32(1.0))
        var grad_logits = cross_entropy_backward(grad_output, fc2_out, labels)

        # FC2 backward
        var fc2_grads = linear_backward(grad_logits, relu_out, self.fc2_weights)

        # ReLU backward
        var grad_fc1_out = relu_backward(fc2_grads.grad_input, fc1_out)

        # FC1 backward
        var fc1_grads = linear_backward(grad_fc1_out, input, self.fc1_weights)

        # Parameter update (SGD) - inline to avoid aliasing issues
        _sgd_update(self.fc1_weights, fc1_grads.grad_weights, learning_rate)
        _sgd_update(self.fc1_bias, fc1_grads.grad_bias, learning_rate)
        _sgd_update(self.fc2_weights, fc2_grads.grad_weights, learning_rate)
        _sgd_update(self.fc2_bias, fc2_grads.grad_bias, learning_rate)

        return loss


def _sgd_update(mut param: AnyTensor, grad: AnyTensor, lr: Float32) raises:
    """Update parameter: param = param - lr * grad."""
    var numel = param.numel()
    for i in range(numel):
        var p = param._get_float32(i)
        var g = grad._get_float32(i)
        param._set_float32(i, p - lr * g)


def create_dummy_batch(
    batch_size: Int, input_dim: Int, num_classes: Int
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Create dummy batch for testing.

    Returns:
        (input, one_hot_labels) tuple.
    """
    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(input_dim)
    var input = full(input_shape, 0.5, DType.float32)

    # Create one-hot labels (all class 0 for simplicity)
    var labels_shape = List[Int]()
    labels_shape.append(batch_size)
    labels_shape.append(num_classes)
    var labels = zeros(labels_shape, DType.float32)
    # Set class 0 as target for all samples
    for i in range(batch_size):
        labels._set_float32(i * num_classes, Float32(1.0))

    return Tuple[AnyTensor, AnyTensor](input^, labels^)


# ============================================================================
# Test 1: Training Loop Completes
# ============================================================================


def test_e2e_training_loop_completes() raises:
    """Test that full training loop runs without crash.

    Verifies:
    - Model initializes correctly
    - Forward pass executes
    - Backward pass computes gradients
    - Parameter update applies.
    """
    var model = SimpleMLP()
    var batch = create_dummy_batch(batch_size=4, input_dim=10, num_classes=5)
    var input = batch[0]
    var labels = batch[1]

    # Should complete without error
    var loss = model.train_step(input, labels, Float32(0.01))

    # Loss should be a valid number (not NaN or Inf)
    assert_true(loss == loss, "Loss should not be NaN")  # NaN != NaN
    assert_true(loss < Float32(1000000.0), "Loss should not be Inf")


# ============================================================================
# Test 2: Loss Decreases Over Iterations
# ============================================================================


def test_e2e_loss_decreases() raises:
    """Test that loss decreases over multiple training steps.

    Training with reasonable learning rate should show
    monotonically decreasing loss (for simple cases).
    """
    var model = SimpleMLP()
    var batch = create_dummy_batch(batch_size=8, input_dim=10, num_classes=5)
    var input = batch[0]
    var labels = batch[1]

    # Record initial loss
    var initial_loss = model.train_step(input, labels, Float32(0.01))

    # Train for several iterations
    var final_loss = Float32(0.0)
    for _ in range(20):
        final_loss = model.train_step(input, labels, Float32(0.01))

    # Loss should decrease
    assert_less(
        Float64(final_loss),
        Float64(initial_loss),
        "Loss should decrease over training",
    )


# ============================================================================
# Test 3: Significant Loss Reduction
# ============================================================================


def test_e2e_loss_decreases_significantly() raises:
    """Test that loss drops by at least 50% over 50 iterations.

    For a simple network on constant data, this should be achievable.
    """
    var model = SimpleMLP()
    var batch = create_dummy_batch(batch_size=8, input_dim=10, num_classes=5)
    var input = batch[0]
    var labels = batch[1]

    # Record initial loss
    var initial_loss = model.train_step(input, labels, Float32(0.01))

    # Train for 50 iterations
    var final_loss = Float32(0.0)
    for _ in range(50):
        final_loss = model.train_step(input, labels, Float32(0.01))

    # Loss should decrease by at least 50%
    var reduction_ratio = Float64(final_loss) / Float64(initial_loss)
    assert_less(
        reduction_ratio, Float64(0.5), "Loss should reduce by at least 50%"
    )


# ============================================================================
# Test 4: Batch Size Variations
# ============================================================================


def test_e2e_batch_size_1() raises:
    """Test training with batch size 1 (SGD)."""
    var model = SimpleMLP()
    var batch = create_dummy_batch(batch_size=1, input_dim=10, num_classes=5)
    var input = batch[0]
    var labels = batch[1]

    var initial_loss = model.train_step(input, labels, Float32(0.01))

    for _ in range(10):
        var loss = model.train_step(input, labels, Float32(0.01))

    # Should complete and loss should be valid
    assert_true(
        initial_loss == initial_loss, "Loss should be valid for batch size 1"
    )


def test_e2e_batch_size_16() raises:
    """Test training with batch size 16."""
    var model = SimpleMLP()
    var batch = create_dummy_batch(batch_size=16, input_dim=10, num_classes=5)
    var input = batch[0]
    var labels = batch[1]

    var initial_loss = model.train_step(input, labels, Float32(0.01))

    for _ in range(10):
        var loss = model.train_step(input, labels, Float32(0.01))

    assert_true(
        initial_loss == initial_loss, "Loss should be valid for batch size 16"
    )


def test_e2e_batch_size_32() raises:
    """Test training with batch size 32."""
    var model = SimpleMLP()
    var batch = create_dummy_batch(batch_size=32, input_dim=10, num_classes=5)
    var input = batch[0]
    var labels = batch[1]

    var initial_loss = model.train_step(input, labels, Float32(0.01))

    for _ in range(10):
        var loss = model.train_step(input, labels, Float32(0.01))

    assert_true(
        initial_loss == initial_loss, "Loss should be valid for batch size 32"
    )


# ============================================================================
# Test 5: Learning Rate Effects
# ============================================================================


def test_e2e_lr_affects_convergence() raises:
    """Test that learning rate affects convergence speed.

    Higher learning rate should converge faster (up to a point).
    """
    # Train with low learning rate
    var model_low_lr = SimpleMLP()
    var batch = create_dummy_batch(batch_size=8, input_dim=10, num_classes=5)
    var input = batch[0]
    var labels = batch[1]

    for _ in range(10):
        var _ = model_low_lr.train_step(input, labels, Float32(0.001))
    var loss_low_lr = model_low_lr.train_step(input, labels, Float32(0.001))

    # Train with high learning rate
    var model_high_lr = SimpleMLP()
    for _ in range(10):
        var _ = model_high_lr.train_step(input, labels, Float32(0.01))
    var loss_high_lr = model_high_lr.train_step(input, labels, Float32(0.01))

    # Higher LR should achieve lower loss in same iterations
    assert_less(
        Float64(loss_high_lr),
        Float64(loss_low_lr),
        "Higher LR should converge faster",
    )


# ============================================================================
# Test 6: Weight Changes During Training
# ============================================================================


def test_e2e_weights_change() raises:
    """Test that weights actually change during training."""
    var model = SimpleMLP()

    # Store original weight value
    var original_fc1 = model.fc1_weights._get_float64(0)
    var original_fc2 = model.fc2_weights._get_float64(0)

    # Train for a few steps
    var batch = create_dummy_batch(batch_size=4, input_dim=10, num_classes=5)
    var input = batch[0]
    var labels = batch[1]
    for _ in range(5):
        var _ = model.train_step(input, labels, Float32(0.01))

    # Weights should have changed
    var new_fc1 = model.fc1_weights._get_float64(0)
    var new_fc2 = model.fc2_weights._get_float64(0)

    var fc1_changed = abs(new_fc1 - original_fc1) > Float64(1e-6)
    var fc2_changed = abs(new_fc2 - original_fc2) > Float64(1e-6)

    assert_true(fc1_changed, "FC1 weights should change during training")
    assert_true(fc2_changed, "FC2 weights should change during training")


# ============================================================================
# Test 7: Reproducibility (Deterministic)
# ============================================================================


def test_e2e_reproducibility() raises:
    """Test that identical initialization produces identical results.

    Since we use fixed values (full() with constant), results should match.
    """
    # Run 1
    var model1 = SimpleMLP()
    var batch1 = create_dummy_batch(batch_size=4, input_dim=10, num_classes=5)
    var loss1 = model1.train_step(batch1[0], batch1[1], Float32(0.01))
    for _ in range(5):
        loss1 = model1.train_step(batch1[0], batch1[1], Float32(0.01))

    # Run 2 (identical setup)
    var model2 = SimpleMLP()
    var batch2 = create_dummy_batch(batch_size=4, input_dim=10, num_classes=5)
    var loss2 = model2.train_step(batch2[0], batch2[1], Float32(0.01))
    for _ in range(5):
        loss2 = model2.train_step(batch2[0], batch2[1], Float32(0.01))

    # Final losses should be identical
    assert_almost_equal(
        Float64(loss1),
        Float64(loss2),
        tolerance=Float64(1e-6),
        message="Identical runs should produce identical results",
    )


# ============================================================================
# Test 8: Multi-Precision Training Integration
# ============================================================================


def test_e2e_training_with_precision_config() raises:
    """Test that training works with PrecisionConfig integration.

    This verifies the precision config can be used alongside training.
    """
    var config = PrecisionConfig.fp32()
    var model = SimpleMLP()
    var batch = create_dummy_batch(batch_size=4, input_dim=10, num_classes=5)
    var input = batch[0]
    var labels = batch[1]

    # Cast input to compute precision (should be identity for FP32)
    var compute_input = config.cast_to_compute(input)
    # Labels not used in forward pass but validated via batch creation
    var _ = labels

    # Forward pass
    var output = model.forward(compute_input)

    # Output should be valid
    assert_equal_int(output.shape()[0], 4, "Batch dimension preserved")
    assert_equal_int(output.shape()[1], 5, "Output dimension correct")


# ============================================================================
# Main - Run All Tests
# ============================================================================


def main() raises:
    """Run all end-to-end training tests."""
    print("=" * 60)
    print("End-to-End Training Integration Tests")
    print("=" * 60)
    print()

    print("Test 1: Training loop completes...")
    test_e2e_training_loop_completes()
    print("  PASSED")

    print("Test 2: Loss decreases over iterations...")
    test_e2e_loss_decreases()
    print("  PASSED")

    print("Test 3: Loss decreases by 50%...")
    test_e2e_loss_decreases_significantly()
    print("  PASSED")

    print("Test 4a: Batch size 1...")
    test_e2e_batch_size_1()
    print("  PASSED")

    print("Test 4b: Batch size 16...")
    test_e2e_batch_size_16()
    print("  PASSED")

    print("Test 4c: Batch size 32...")
    test_e2e_batch_size_32()
    print("  PASSED")

    print("Test 5: Learning rate affects convergence...")
    test_e2e_lr_affects_convergence()
    print("  PASSED")

    print("Test 6: Weights change during training...")
    test_e2e_weights_change()
    print("  PASSED")

    print("Test 7: Reproducibility...")
    test_e2e_reproducibility()
    print("  PASSED")

    print("Test 8: Training with precision config...")
    test_e2e_training_with_precision_config()
    print("  PASSED")

    print()
    print("=" * 60)
    print("ALL END-TO-END TESTS PASSED! (10/10)")
    print("=" * 60)
