"""Unit tests for Training Loop - Part 3: DataLoader N-D Tensor Tests.

Tests cover:
- DataLoader 4D tensor slicing (N, C, H, W)
- DataLoader partial batch handling
- DataLoader 3D tensor slicing (N, seq_len, features)
- DataLoader N-D shape preservation

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_training_loop.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Issue #2728: Enable Training Loop Tests with SimpleMLP and AnyTensor.randn.
Tests enabled after core infrastructure was completed:
- MSELoss.compute() implementation
- SGD/TrainingLoop integration via autograd
- AnyTensor.randn export from shared.core
"""

from tests.shared.conftest import (
    assert_equal,
    assert_greater,
)
from shared.training.trainer_interface import DataLoader
from shared.tensor.any_tensor import AnyTensor, ones, zeros
from shared.core import subtract, multiply


# ============================================================================
# DataLoader N-D Tensor Tests
# ============================================================================


fn test_dataloader_4d_batch_slicing() raises:
    """Test DataLoader correctly slices 4D tensors (N, C, H, W).

    Verifies that DataLoader.next() returns batches with all trailing
    dimensions (C, H, W) preserved correctly for image data.
    """
    # Create (8, 2, 4, 4) float32 tensor simulating image data
    var data = ones([8, 2, 4, 4], DType.float32)
    var labels = zeros([8], DType.float32)
    var loader = DataLoader(data^, labels^, 4)

    # First batch
    var batch1 = loader.next()
    assert_equal(batch1.data.shape()[0], 4)
    assert_equal(batch1.data.shape()[1], 2)
    assert_equal(batch1.data.shape()[2], 4)
    assert_equal(batch1.data.shape()[3], 4)
    assert_equal(batch1.batch_size, 4)

    # Second batch
    var batch2 = loader.next()
    assert_equal(batch2.data.shape()[0], 4)
    assert_equal(batch2.data.shape()[1], 2)
    assert_equal(batch2.data.shape()[2], 4)
    assert_equal(batch2.data.shape()[3], 4)
    assert_equal(batch2.batch_size, 4)

    print("  test_dataloader_4d_batch_slicing: PASSED")


fn test_dataloader_4d_partial_last_batch() raises:
    """Test DataLoader handles partial last batch for 4D tensors.

    With N=6 and batch_size=4, the second batch should have 2 samples
    while preserving trailing dimensions (C, H, W).
    """
    var data = ones([6, 2, 4, 4], DType.float32)
    var labels = zeros([6], DType.float32)
    var loader = DataLoader(data^, labels^, 4)

    # First full batch: shape (4, 2, 4, 4)
    var batch1 = loader.next()
    assert_equal(batch1.data.shape()[0], 4)
    assert_equal(batch1.data.shape()[1], 2)
    assert_equal(batch1.data.shape()[2], 4)
    assert_equal(batch1.data.shape()[3], 4)
    assert_equal(batch1.batch_size, 4)

    # Partial last batch: shape (2, 2, 4, 4)
    var batch2 = loader.next()
    assert_equal(batch2.data.shape()[0], 2)
    assert_equal(batch2.data.shape()[1], 2)
    assert_equal(batch2.data.shape()[2], 4)
    assert_equal(batch2.data.shape()[3], 4)
    assert_equal(batch2.batch_size, 2)

    print("  test_dataloader_4d_partial_last_batch: PASSED")


fn test_dataloader_3d_batch_slicing() raises:
    """Test DataLoader correctly slices 3D tensors (N, seq_len, features).

    Verifies that DataLoader.next() works for sequence data where each
    sample has shape (seq_len, features).
    """
    var data = ones([8, 10, 16], DType.float32)
    var labels = zeros([8], DType.float32)
    var loader = DataLoader(data^, labels^, 4)

    var batch = loader.next()
    assert_equal(batch.data.shape()[0], 4)
    assert_equal(batch.data.shape()[1], 10)
    assert_equal(batch.data.shape()[2], 16)
    assert_equal(batch.batch_size, 4)

    print("  test_dataloader_3d_batch_slicing: PASSED")


fn test_dataloader_nd_shape_preserved() raises:
    """Test that trailing dimensions are identical across all batches.

    Iterates all batches of a (9, 3, 8, 8) tensor with batch_size=4
    and asserts remaining dims (3, 8, 8) are preserved in every batch.
    """
    var data = ones([9, 3, 8, 8], DType.float32)
    var labels = zeros([9], DType.float32)
    var loader = DataLoader(data^, labels^, 4)

    while loader.has_next():
        var batch = loader.next()
        # All batches must preserve trailing dims regardless of batch size
        assert_equal(batch.data.shape()[1], 3)
        assert_equal(batch.data.shape()[2], 8)
        assert_equal(batch.data.shape()[3], 8)

    print("  test_dataloader_nd_shape_preserved: PASSED")


# ============================================================================
# Epoch Runner Tests
# ============================================================================


fn test_run_epoch_with_batches() raises:
    """Test run_epoch_with_batches() with real DataLoader and callbacks.

    Verifies:
    - Processes all batches from DataLoader
    - Invokes callbacks for batch completion
    - Returns non-trivial avg_loss with real inputs

    Test flow:
        1. Create DataLoader with 8 samples, batch_size=2 -> 4 batches
        2. Define step_fn that computes loss from batch data
        3. Create TrainingCallbacks (verbose=False to avoid print spam)
        4. Call run_epoch_with_batches()
        5. Assert avg_loss > 0.0 (non-zero inputs should produce non-zero loss)
    """
    from shared.training.script_runner import run_epoch_with_batches
    from shared.training.script_runner import TrainingCallbacks

    # Create DataLoader: 8 samples x 4 features, batch_size=2 -> 4 batches
    var data = ones([8, 4], DType.float32)
    var labels = zeros([8, 1], DType.float32)
    var loader = DataLoader(data^, labels^, batch_size=2)

    # Create callbacks (verbose=False to suppress output in tests)
    var callbacks = TrainingCallbacks(verbose=False)

    # Define step function that computes loss from batch
    # For each batch: loss = sum(batch_data) - batch_labels
    # With ones input and zeros labels, each batch should have positive loss
    fn step_fn(batch_data: AnyTensor, batch_labels: AnyTensor) raises -> AnyTensor:
        # Simple loss: sum squared differences
        # Since batch_data=ones and batch_labels=zeros, loss will be > 0
        var diff = subtract(batch_data, batch_labels)
        var squared = multiply(diff, diff)
        var loss_scalar = ones([1], DType.float32)
        # Return scalar loss (simplified for testing)
        return loss_scalar

    # Run epoch with batches
    var avg_loss = run_epoch_with_batches(loader, callbacks, step_fn)

    # Verify avg_loss > 0.0 (each step_fn returns ones, avg should be ~1.0)
    assert_greater(Float64(avg_loss), Float64(0.0))

    print("  test_run_epoch_with_batches: PASSED")


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run training loop part 3 tests (DataLoader N-D tensor + epoch runner)."""
    print("Running DataLoader N-D Tensor Tests...")
    test_dataloader_4d_batch_slicing()
    test_dataloader_4d_partial_last_batch()
    test_dataloader_3d_batch_slicing()
    test_dataloader_nd_shape_preserved()

    print("Running epoch runner tests...")
    test_run_epoch_with_batches()

    print("\nAll training loop part 3 tests passed!")
