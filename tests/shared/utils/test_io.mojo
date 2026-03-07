"""Tests for file I/O utilities module.

This module tests file I/O functionality including:
- Model checkpoint save/load
- Tensor serialization/deserialization
- Safe file operations (atomic writes, backups)
- Binary and text file handling
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


# ============================================================================
# Test Tensor Serialization
# ============================================================================


fn test_serialize_tensor():
    """Test serializing tensor to bytes."""
    # TODO(#44): Implement when Tensor.serialize exists
    # Create tensor with known values
    # Serialize to bytes
    # Verify bytes contain tensor data and metadata (shape, dtype)
    pass


fn test_deserialize_tensor():
    """Test deserializing tensor from bytes."""
    # TODO(#44): Implement when Tensor.deserialize exists
    # Create serialized tensor bytes
    # Deserialize to tensor
    # Verify shape, dtype, and values match original
    pass


fn test_tensor_roundtrip():
    """Test serializing and deserializing tensor preserves values."""
    # TODO(#44): Implement when Tensor serialization exists
    # Create random tensor
    # Serialize to bytes
    # Deserialize back to tensor
    # Verify all values match exactly
    pass


fn test_serialize_large_tensor():
    """Test serializing large tensor (> 1GB)."""
    # TODO(#44): Implement when Tensor serialization exists
    # Create large tensor (e.g., 256M Float32 = 1GB)
    # Serialize to file
    # Verify file size is correct
    # Deserialize and verify values (spot check)
    pass


fn test_serialize_tensor_formats():
    """Test serializing tensors with different dtypes."""
    # TODO(#44): Implement when Tensor serialization exists
    # Test serialization for:
    # - Float32, Float64
    # - Int8, Int16, Int32, Int64
    # - Bool
    # Verify dtype is preserved in serialization
    pass


# ============================================================================
# Test Safe File Operations
# ============================================================================


fn test_atomic_write():
    """Test atomic file write (write to temp, then rename)."""
    # TODO(#44): Implement when atomic_write exists
    # Write data to file atomically
    # Verify temp file is created first
    # Verify temp file is renamed to target
    # No partial writes visible at target path
    pass


fn test_write_with_backup():
    """Test writing file creates backup of existing file."""
    # TODO(#44): Implement when write_with_backup exists
    # Create file with content "old"
    # Write new content "new" with backup
    # Verify: original file has "new"
    # Verify: backup file has "old"
    # Clean up temp files
    pass


fn test_safe_remove() raises:
    """Test remove_safely() actually deletes the file."""
    from shared.utils.file_io import remove_safely, safe_write_file, file_exists

    var test_path = "/tmp/test_remove_safely_3283.txt"

    # Create file
    var written = safe_write_file(test_path, "test content")
    assert_true(written)
    assert_true(file_exists(test_path))

    # Remove it
    var removed = remove_safely(test_path)
    assert_true(removed)
    assert_false(file_exists(test_path))

    # Removing nonexistent file returns False
    var removed_again = remove_safely(test_path)
    assert_false(removed_again)

    print("PASS: test_safe_remove")


fn test_create_directory_safe():
    """Test creating directory safely (no error if exists)."""
    # TODO(#44): Implement when mkdir_safe exists
    # Create directory
    # Create same directory again
    # Verify no error on second create
    # Clean up directory
    pass


fn test_file_exists_check():
    """Test checking if file exists."""
    # TODO(#44): Implement when file_exists helper exists
    # Create temp file
    # Verify file_exists returns True
    # Remove file
    # Verify file_exists returns False
    pass


# ============================================================================
# Test Binary File Operations
# ============================================================================


fn test_write_binary_file():
    """Test writing binary data to file."""
    # TODO(#44): Implement when write_binary exists
    # Create byte array with known values
    # Write to temp file
    # Read file and verify bytes match
    # Clean up temp file
    pass


fn test_read_binary_file():
    """Test reading binary data from file."""
    # TODO(#44): Implement when read_binary exists
    # Create temp file with binary data
    # Read file
    # Verify bytes match original data
    # Clean up temp file
    pass


fn test_binary_file_roundtrip():
    """Test writing and reading binary file preserves data."""
    # TODO(#44): Implement when binary I/O exists
    # Create random binary data
    # Write to file
    # Read from file
    # Verify all bytes match exactly
    pass


fn test_read_binary_in_chunks():
    """Test reading large binary file in chunks."""
    # TODO(#44): Implement when chunked read exists
    # Create large binary file (e.g., 100MB)
    # Read in 10MB chunks
    # Verify all chunks read correctly
    # Total data matches file size
    pass


# ============================================================================
# Test Text File Operations
# ============================================================================


fn test_write_text_file():
    """Test writing text to file."""
    # TODO(#44): Implement when write_text exists
    # Create text string
    # Write to temp file
    # Read file and verify text matches
    # Clean up temp file
    pass


fn test_read_text_file():
    """Test reading text from file."""
    # TODO(#44): Implement when read_text exists
    # Create temp file with text
    # Read file
    # Verify text matches original
    # Clean up temp file
    pass


fn test_read_text_lines():
    """Test reading text file line by line."""
    # TODO(#44): Implement when read_lines exists
    # Create file with multiple lines
    # Read lines as list
    # Verify line count and content
    pass


fn test_append_to_text_file():
    """Test appending text to existing file."""
    # TODO(#44): Implement when append_text exists
    # Create file with "line 1\n"
    # Append "line 2\n"
    # Read file
    # Verify: "line 1\nline 2\n"
    pass


# ============================================================================
# Test Path Operations
# ============================================================================


fn test_resolve_path():
    """Test resolving relative paths to absolute paths."""
    # TODO(#44): Implement when resolve_path exists
    # Resolve "./data/file.csv"
    # Verify returns absolute path
    # Resolve "~/data/file.csv"
    # Verify ~ is expanded to home directory
    pass


fn test_join_paths():
    """Test joining path components."""
    # TODO(#44): Implement when join_path exists
    # Join ["data", "train", "images.csv"]
    # Verify: "data/train/images.csv" (Unix) or "data\\train\\images.csv" (Windows)
    pass


fn test_split_path():
    """Test splitting path into directory and filename."""
    # TODO(#44): Implement when split_path exists
    # Split "data/train/images.csv"
    # Verify: directory="data/train", filename="images.csv"
    pass


fn test_get_file_extension():
    """Test extracting file extension."""
    # TODO(#44): Implement when get_extension exists
    # Extension of "model.mojo" -> ".mojo"
    # Extension of "data.tar.gz" -> ".gz" or ".tar.gz"?
    # Extension of "no_extension" -> ""
    pass


fn test_list_directory():
    """Test listing files in directory."""
    # TODO(#44): Implement when list_dir exists
    # Create temp directory with files: a.txt, b.txt, c.csv
    # List directory
    # Verify returns ["a.txt", "b.txt", "c.csv"]
    # Clean up temp directory
    pass


# ============================================================================
# Test Error Handling
# ============================================================================


fn test_load_nonexistent_file():
    """Test loading nonexistent file raises error."""
    # TODO(#44): Implement when load_checkpoint exists
    # Try to load "nonexistent.checkpoint"
    # Verify FileNotFoundError is raised
    pass


fn test_save_to_readonly_directory():
    """Test saving to read-only directory raises error."""
    # TODO(#44): Implement when save_checkpoint exists
    # Try to save to "/read_only_dir/model.checkpoint"
    # Verify PermissionError is raised
    pass


fn test_load_corrupted_checkpoint():
    """Test loading corrupted checkpoint raises error."""
    # TODO(#44): Implement when load_checkpoint exists
    # Create checkpoint file with invalid/corrupted data
    # Try to load
    # Verify error is raised (ParseError, ValidationError, etc.)
    pass


fn test_disk_full_error():
    """Test handling disk full error during save."""
    # TODO(#44): Implement when save handles disk errors
    # Simulate disk full condition
    # Try to save checkpoint
    # Verify error is raised
    # Verify no partial file is left
    pass


# ============================================================================
# Test Compression
# ============================================================================


fn test_save_compressed_checkpoint():
    """Test saving compressed checkpoint."""
    # TODO(#44): Implement when compression support exists
    # Create model checkpoint
    # Save with compression
    # Verify: compressed file is smaller than uncompressed
    # Load compressed checkpoint
    # Verify: parameters match original
    pass


fn test_compression_formats():
    """Test different compression formats (gzip, zstd, etc.)."""
    # TODO(#44): Implement when multiple compression formats exist
    # Save checkpoint with gzip compression
    # Save checkpoint with zstd compression
    # Verify both can be loaded correctly
    # Compare compression ratios and speed
    pass


# ============================================================================
# Integration Tests
# ============================================================================


fn test_checkpoint_integration_training():
    """Test checkpoint save/load integrates with training loop."""
    # TODO(#44): Implement when full training workflow exists
    # Train model for 5 epochs
    # Save checkpoint
    # Load checkpoint into new model
    # Continue training for 5 more epochs
    # Verify training continues correctly from checkpoint
    pass


fn test_resume_training_from_checkpoint():
    """Test resuming training from saved checkpoint."""
    # TODO(#44): Implement when checkpoint includes optimizer state
    # Train for 5 epochs
    # Save checkpoint with optimizer state
    # Load checkpoint
    # Continue training
    # Verify: optimizer state is restored
    # Verify: training continues smoothly
    pass


fn main() raises:
    """Run all tests."""
    test_save_checkpoint()
    test_load_checkpoint()
    test_checkpoint_roundtrip()
    test_save_checkpoint_with_metadata()
    test_save_checkpoint_atomic()
    test_checkpoint_serialization_with_model_state()
    test_checkpoint_serialization_with_optimizer_state()
    test_checkpoint_serialization_with_metadata()
    test_serialize_tensor()
    test_deserialize_tensor()
    test_tensor_roundtrip()
    test_serialize_large_tensor()
    test_serialize_tensor_formats()
    test_atomic_write()
    test_write_with_backup()
    test_safe_remove()
    test_create_directory_safe()
    test_file_exists_check()
    test_write_binary_file()
    test_read_binary_file()
    test_binary_file_roundtrip()
    test_read_binary_in_chunks()
    test_write_text_file()
    test_read_text_file()
    test_read_text_lines()
    test_append_to_text_file()
    test_resolve_path()
    test_join_paths()
    test_split_path()
    test_get_file_extension()
    test_list_directory()
    test_load_nonexistent_file()
    test_save_to_readonly_directory()
    test_load_corrupted_checkpoint()
    test_disk_full_error()
    test_save_compressed_checkpoint()
    test_compression_formats()
    test_checkpoint_integration_training()
    test_resume_training_from_checkpoint()
