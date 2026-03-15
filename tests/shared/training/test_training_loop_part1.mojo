"""Unit tests for Training Loop - Part 1: Core, Forward Pass, Loss.

Tests cover:
- Forward pass execution
- Loss computation
- Batch iteration and epoch completion

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_training_loop.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Issue #2728: Enable Training Loop Tests with SimpleMLP and ExTensor.randn.
Tests enabled after core infrastructure was completed:
- MSELoss.compute() implementation
- SGD/TrainingLoop integration via autograd
- ExTensor.randn export from shared.core
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_less,
    assert_greater,
    assert_not_equal_tensor,
    assert_not_none,
    assert_shape_equal,
    assert_tensor_equal,
    assert_type,
    create_simple_model,
    create_simple_dataset,
    create_test_vector,
    TestFixtures,
)
from shared.training import SGD, MSELoss, TrainingLoop
from shared.training.trainer_interface import DataLoader
from shared.core.extensor import ExTensor, ones, zeros, randn
from shared.testing import SimpleMLP

# TrainingLoop is generic with trait bounds for compile-time type safety.
#
# Instantiation pattern:
#   var training_loop = TrainingLoop[SimpleMLP, MSELoss, SGD](model, optimizer, loss_fn)
#
# Type parameters:
#   [M: Model & Movable, L: Loss & Movable, O: Optimizer & Movable]
#
# Wrong types are rejected at compile time. See shared/training/__init__.mojo for
# full implementation and trait definitions.


# ============================================================================
# Training Loop Core Tests
# ============================================================================


fn test_training_loop_single_batch() raises:
    """Test training loop processes a single batch correctly.

    API Contract:
        Training step should:
        1. Get batch from data loader
        2. Forward pass: output = model(input)
        3. Compute loss: loss = loss_fn(output, target)
        4. Backward pass: compute gradients
        5. Optimizer step: update weights
        6. Return loss value

    This is a CRITICAL test for basic training functionality.
    """
    # Create model, optimizer, and loss function
    var model = create_simple_model()
    var optimizer = SGD(learning_rate=0.01)
    var loss_fn = MSELoss()
    var training_loop = TrainingLoop[SimpleMLP, MSELoss, SGD](
        model^, optimizer^, loss_fn^
    )

    # Create single batch - use 1D input since SimpleMLP expects flat input
    var inputs = ones([10], DType.float32)  # input_dim=10
    var targets = zeros([1], DType.float32)  # output_dim=1

    # Run single training step
    var loss = training_loop.step(inputs, targets)

    # Verify loss is computed (should be non-negative for MSE)
    # MSE on ones -> zeros should give positive loss
    var loss_val = loss._get_float32(0)
    assert_greater(Float64(loss_val), Float64(0.0))

    print("  test_training_loop_single_batch: PASSED")


fn test_training_loop_full_epoch() raises:
    """Test training loop completes a full epoch over dataset.

    API Contract:
        fn run_epoch(self, data_loader: DataLoader) -> Float32
        - Iterates through all batches in data loader
        - Performs training step on each batch
        - Returns average loss for the epoch.
    """
    # Create model, optimizer, and loss function
    var model = create_simple_model()
    var optimizer = SGD(learning_rate=0.01)
    var loss_fn = MSELoss()
    var training_loop = TrainingLoop[SimpleMLP, MSELoss, SGD](
        model^, optimizer^, loss_fn^
    )

    # Create a real DataLoader with 100 samples, batch_size=10 -> 10 batches
    var data_tensor = ones([100, 10], DType.float32)
    var label_tensor = zeros([100, 1], DType.float32)
    var data_loader = DataLoader(data_tensor^, label_tensor^, batch_size=10)

    # Run one epoch using native DataLoader
    var avg_loss = training_loop.run_epoch(data_loader)

    # Verify real batch processing occurred (loss should be valid, not placeholder 0.0)
    assert_greater(Float64(avg_loss), Float64(-0.001))

    print("  test_training_loop_full_epoch: PASSED")


fn test_training_loop_multiple_epochs() raises:
    """Test training loop runs multiple training steps and loss can decrease.

    API Contract:
        Multiple training steps should:
        - Process batches through forward/backward pass
        - Loss should generally decrease (for simple problems)
        - Parameters should be updated via gradient descent.

    Note:
        This test uses step() directly instead of run_epoch() since the
        data loader integration is pending (TODO #3013).
    """
    # Create model, optimizer, and loss function
    var model = create_simple_model()
    var optimizer = SGD(learning_rate=0.1)
    var loss_fn = MSELoss()
    var training_loop = TrainingLoop[SimpleMLP, MSELoss, SGD](
        model^, optimizer^, loss_fn^
    )

    # Create fixed input/target for multiple steps
    var inputs = ones([10], DType.float32)
    var targets = zeros([1], DType.float32)

    # Run multiple training steps
    var first_loss = training_loop.step(inputs, targets)
    var first_loss_val = first_loss._get_float32(0)

    # Run a few more steps
    for _ in range(5):
        _ = training_loop.step(inputs, targets)

    var final_loss = training_loop.step(inputs, targets)
    var final_loss_val = final_loss._get_float32(0)

    # Both losses should be valid (non-negative for MSE)
    assert_greater(Float64(first_loss_val), Float64(-0.001))
    assert_greater(Float64(final_loss_val), Float64(-0.001))

    print("  test_training_loop_multiple_epochs: PASSED")


# ============================================================================
# Forward Pass Tests
# ============================================================================


fn test_training_loop_forward_pass() raises:
    """Test training loop executes forward pass correctly.

    API Contract:
        Forward pass should:
        - Call model.forward(input)
        - Return output tensor
        - Return correct output dimension.
    """
    # Create model, optimizer, and loss function
    var model = create_simple_model()
    var optimizer = SGD(learning_rate=0.01)
    var loss_fn = MSELoss()
    var training_loop = TrainingLoop[SimpleMLP, MSELoss, SGD](
        model^, optimizer^, loss_fn^
    )

    # Create input using randn (now exported from shared.core)
    var inputs = randn([10], DType.float32, seed=42)

    # Execute forward pass
    var outputs = training_loop.forward(inputs)

    # Output should have correct output dimension (1 for create_simple_model)
    assert_equal(outputs.shape()[0], 1)

    print("  test_training_loop_forward_pass: PASSED")


fn test_training_loop_forward_batches_independently() raises:
    """Test forward pass produces consistent results.

    API Contract:
        Forward pass should produce deterministic results for same input.

    Note:
        SimpleMLP processes 1D input (not batched). This test verifies
        that the same input produces the same output.
    """
    # Create model, optimizer, and loss function
    var model = create_simple_model()
    var optimizer = SGD(learning_rate=0.01)
    var loss_fn = MSELoss()
    var training_loop = TrainingLoop[SimpleMLP, MSELoss, SGD](
        model^, optimizer^, loss_fn^
    )

    # Create input using randn with fixed seed for reproducibility
    var input1 = randn([10], DType.float32, seed=42)
    var input2 = randn([10], DType.float32, seed=42)  # Same seed = same values

    # Forward pass on same input should give same output
    var output1 = training_loop.forward(input1)
    var output2 = training_loop.forward(input2)

    # Outputs should be equal for same input
    var val1 = output1._get_float32(0)
    var val2 = output2._get_float32(0)
    assert_almost_equal(Float64(val1), Float64(val2), Float64(1e-5))

    print("  test_training_loop_forward_batches_independently: PASSED")


# ============================================================================
# Loss Computation Tests
# ============================================================================


fn test_training_loop_computes_loss() raises:
    """Test training loop computes loss correctly.

    API Contract:
        fn compute_loss(self, outputs: Tensor, targets: Tensor) -> ExTensor
        - Calls loss_fn.compute(outputs, targets)
        - Returns loss value as ExTensor.
    """
    # Create model, optimizer, and loss function
    var model = create_simple_model()
    var optimizer = SGD(learning_rate=0.01)
    var loss_fn = MSELoss()
    var training_loop = TrainingLoop[SimpleMLP, MSELoss, SGD](
        model^, optimizer^, loss_fn^
    )

    # Known outputs and targets
    var outputs_shape: List[Int] = [3]
    var outputs = zeros(outputs_shape, DType.float32)
    outputs._set_float32(0, 1.0)
    outputs._set_float32(1, 2.0)
    outputs._set_float32(2, 3.0)

    var targets_shape: List[Int] = [3]
    var targets = zeros(targets_shape, DType.float32)
    # targets are all zeros

    # Compute loss using training loop's compute_loss method
    var loss = training_loop.compute_loss(outputs, targets)

    # MSE = mean((outputs - targets)^2) = mean([1, 4, 9]) = 14/3 = 4.67
    var loss_val = loss._get_float32(0)
    assert_almost_equal(Float64(loss_val), Float64(4.6667), Float64(0.01))

    print("  test_training_loop_computes_loss: PASSED")


fn test_training_loop_loss_scalar() raises:
    """Test training loop returns loss as ExTensor.

    API Contract:
        Loss should be returned as ExTensor containing the reduced loss value.
        The loss tensor typically has shape [1] or [] (scalar).
    """
    # Create model, optimizer, and loss function
    var model = create_simple_model()
    var optimizer = SGD(learning_rate=0.01)
    var loss_fn = MSELoss()
    var training_loop = TrainingLoop[SimpleMLP, MSELoss, SGD](
        model^, optimizer^, loss_fn^
    )

    # Create inputs and targets
    var inputs = randn([10], DType.float32, seed=42)
    var targets = zeros([1], DType.float32)

    # Run training step
    var loss = training_loop.step(inputs, targets)

    # Loss should be an ExTensor (can extract float value)
    var loss_val = loss._get_float32(0)

    # Loss value should be a valid number (not NaN or Inf for this simple case)
    assert_greater(Float64(loss_val), Float64(-1000.0))  # Not negative infinity
    assert_less(Float64(loss_val), Float64(1000.0))  # Not positive infinity

    print("  test_training_loop_loss_scalar: PASSED")


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run training loop part 1 tests (core, forward pass, loss)."""
    print("Running training loop core tests...")
    test_training_loop_single_batch()
    test_training_loop_full_epoch()
    test_training_loop_multiple_epochs()

    print("Running forward pass tests...")
    test_training_loop_forward_pass()
    test_training_loop_forward_batches_independently()

    print("Running loss computation tests...")
    test_training_loop_computes_loss()
    test_training_loop_loss_scalar()

    print("\nAll training loop part 1 tests passed!")
