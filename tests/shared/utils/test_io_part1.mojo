# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_io.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for file I/O utilities module - Part 1: Checkpoint Save/Load.

This module tests model checkpoint functionality including:
- Model checkpoint save/load
- Checkpoint serialization (model state, optimizer state, metadata)
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_not_equal,
    TestFixtures,
)


# ============================================================================
# Test Checkpoint Save/Load
# ============================================================================


fn test_save_checkpoint():
    """Test saving model checkpoint to file."""
    # TODO(#44): Implement when save_checkpoint exists
    # Create model with known parameters
    # Save checkpoint to temp file
    # Verify file exists and has content
    # Clean up temp file
    pass


fn test_load_checkpoint():
    """Test loading model checkpoint from file."""
    # TODO(#44): Implement when load_checkpoint exists
    # Create checkpoint file with known parameters
    # Load checkpoint
    # Verify parameters match saved values
    # Clean up temp file
    pass


fn test_checkpoint_roundtrip():
    """Test saving and loading checkpoint preserves values."""
    # TODO(#44): Implement when checkpoint save/load exist
    # Create model with random parameters
    # Save checkpoint
    # Load checkpoint into new model
    # Verify all parameters match exactly
    pass


fn test_checkpoint_serialization_with_model_state() raises:
    """Test checkpoint serialization includes model_state dict (Issue #2585)."""
    from shared.utils.file_io import Checkpoint, _serialize_checkpoint

    var checkpoint = Checkpoint()
    checkpoint.set_epoch(10)
    checkpoint.set_loss(0.25)
    checkpoint.set_accuracy(0.95)

    # Add model state entries
    checkpoint.model_state["layer1.weight"] = "tensor_data_1"
    checkpoint.model_state["layer1.bias"] = "tensor_data_2"
    checkpoint.model_state["layer2.weight"] = "tensor_data_3"

    # Serialize
    var serialized = _serialize_checkpoint(checkpoint)

    # Verify model state lines are present
    assert_true(serialized.__contains__("MODEL:layer1.weight=tensor_data_1"))
    assert_true(serialized.__contains__("MODEL:layer1.bias=tensor_data_2"))
    assert_true(serialized.__contains__("MODEL:layer2.weight=tensor_data_3"))

    print("PASS: test_checkpoint_serialization_with_model_state")


fn test_checkpoint_serialization_with_optimizer_state() raises:
    """Test checkpoint serialization includes optimizer_state dict (Issue #2585).
    """
    from shared.utils.file_io import Checkpoint, _serialize_checkpoint

    var checkpoint = Checkpoint()
    checkpoint.set_epoch(5)
    checkpoint.set_loss(0.5)
    checkpoint.set_accuracy(0.85)

    # Add optimizer state entries
    checkpoint.optimizer_state["momentum.layer1"] = "0.9"
    checkpoint.optimizer_state["momentum.layer2"] = "0.95"
    checkpoint.optimizer_state["lr"] = "0.001"

    # Serialize
    var serialized = _serialize_checkpoint(checkpoint)

    # Verify optimizer state lines are present
    assert_true(serialized.__contains__("OPTIMIZER:momentum.layer1=0.9"))
    assert_true(serialized.__contains__("OPTIMIZER:momentum.layer2=0.95"))
    assert_true(serialized.__contains__("OPTIMIZER:lr=0.001"))

    print("PASS: test_checkpoint_serialization_with_optimizer_state")


fn test_checkpoint_serialization_with_metadata() raises:
    """Test checkpoint serialization includes metadata dict (Issue #2585)."""
    from shared.utils.file_io import Checkpoint, _serialize_checkpoint

    var checkpoint = Checkpoint()
    checkpoint.set_epoch(15)
    checkpoint.set_loss(0.1)
    checkpoint.set_accuracy(0.98)

    # Add metadata entries using set_metadata
    checkpoint.set_metadata("timestamp", "2025-12-10T10:30:00")
    checkpoint.set_metadata("hostname", "train-node-01")
    checkpoint.set_metadata("git_commit", "abc123def")

    # Serialize
    var serialized = _serialize_checkpoint(checkpoint)

    # Verify metadata lines are present
    assert_true(serialized.__contains__("META:timestamp=2025-12-10T10:30:00"))
    assert_true(serialized.__contains__("META:hostname=train-node-01"))
    assert_true(serialized.__contains__("META:git_commit=abc123def"))

    print("PASS: test_checkpoint_serialization_with_metadata")


fn test_save_checkpoint_with_metadata():
    """Test saving checkpoint with training metadata."""
    # TODO(#44): Implement when checkpoint format supports metadata
    # Save checkpoint with:
    # - Model parameters
    # - Optimizer state
    # - Epoch number
    # - Loss value
    # - Timestamp
    # Load and verify all metadata is preserved
    pass


fn test_save_checkpoint_atomic():
    """Test checkpoint save is atomic (no partial writes)."""
    # TODO(#44): Implement when atomic save exists
    # Start saving large checkpoint
    # Interrupt save (simulate crash)
    # Verify: either complete file exists OR no file exists
    # No partial/corrupted file should exist
    pass


fn main() raises:
    """Run all tests."""
    test_save_checkpoint()
    test_load_checkpoint()
    test_checkpoint_roundtrip()
    test_checkpoint_serialization_with_model_state()
    test_checkpoint_serialization_with_optimizer_state()
    test_checkpoint_serialization_with_metadata()
    test_save_checkpoint_with_metadata()
    test_save_checkpoint_atomic()
