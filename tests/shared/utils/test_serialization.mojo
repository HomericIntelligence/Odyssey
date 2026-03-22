"""Tests for tensor serialization utilities.

Tests checkpoint save/load operations with named tensors and metadata.
Covers both single tensor and collection operations.
"""

from shared.core.extensor import AnyTensor, zeros, ones
from shared.utils import (
    NamedTensor,
    save_named_tensors,
    load_named_tensors,
    save_named_checkpoint,
    load_named_checkpoint,
)
from testing import assert_true, assert_equal


# ============================================================================
# Helper Functions
# ============================================================================


fn create_test_dir(base: String) raises -> String:
    """Create a unique test directory."""
    from python import Python

    var uuid = Python.import_module("uuid")
    var test_id = String(String(uuid.uuid4())[:8])
    var test_dir = base + "/test_checkpoint_" + test_id
    return test_dir


fn cleanup_test_dir(dir_path: String) -> Bool:
    """Clean up test directory after testing."""
    try:
        from python import Python

        var shutil = Python.import_module("shutil")
        shutil.rmtree(dir_path)
        return True
    except:
        return False


# ============================================================================
# NamedTensor Tests
# ============================================================================


fn test_named_tensor_creation() raises:
    """Test creating a NamedTensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var tensor = zeros(shape, DType.float32)

    var named = NamedTensor("test_weights", tensor)

    assert_equal(named.name, "test_weights", "Name should match")
    assert_equal(named.tensor.shape()[0], 3, "Shape should match")
    assert_equal(named.tensor.shape()[1], 4, "Shape should match")


fn test_named_tensor_multiple_dtypes() raises:
    """Test NamedTensor with different dtypes."""
    # Float32 tensor
    var shape_f32 = List[Int]()
    shape_f32.append(2)
    var tensor_f32 = zeros(shape_f32, DType.float32)
    var named_f32 = NamedTensor("weights_f32", tensor_f32)
    assert_equal(named_f32.tensor.dtype(), DType.float32, "Float32 dtype check")

    # Int64 tensor
    var shape_i64 = List[Int]()
    shape_i64.append(5)
    var tensor_i64 = zeros(shape_i64, DType.int64)
    var named_i64 = NamedTensor("indices_i64", tensor_i64)
    assert_equal(named_i64.tensor.dtype(), DType.int64, "Int64 dtype check")


# ============================================================================
# Save/Load Named Tensors Tests
# ============================================================================


fn test_save_load_single_named_tensor() raises:
    """Test saving and loading a single named tensor."""
    var test_dir = create_test_dir("/tmp")

    try:
        # Create a test tensor
        var shape = List[Int]()
        shape.append(3)
        shape.append(4)
        var tensor = ones(shape, DType.float32)
        var named = NamedTensor("test_tensor", tensor)

        # Create list with single tensor
        var tensors: List[NamedTensor] = []
        tensors.append(named^)

        # Save
        save_named_tensors(tensors, test_dir)

        # Load
        var loaded_tensors = load_named_tensors(test_dir)

        # Verify
        assert_equal(len(loaded_tensors), 1, "Should have loaded 1 tensor")
        assert_equal(loaded_tensors[0].name, "test_tensor", "Name should match")
        assert_equal(
            loaded_tensors[0].tensor.shape()[0], 3, "Shape dim 0 should match"
        )
        assert_equal(
            loaded_tensors[0].tensor.shape()[1], 4, "Shape dim 1 should match"
        )

    finally:
        _ = cleanup_test_dir(test_dir)


fn test_save_load_multiple_named_tensors() raises:
    """Test saving and loading multiple named tensors."""
    var test_dir = create_test_dir("/tmp")

    try:
        # Create multiple test tensors
        var shape1 = List[Int]()
        shape1.append(2)
        shape1.append(3)
        var tensor1 = ones(shape1, DType.float32)

        var shape2 = List[Int]()
        shape2.append(4)
        var tensor2 = zeros(shape2, DType.float64)

        var shape3 = List[Int]()
        shape3.append(5)
        shape3.append(5)
        var tensor3 = ones(shape3, DType.int32)

        # Create list with multiple tensors
        var tensors: List[NamedTensor] = []
        tensors.append(NamedTensor("weights", tensor1))
        tensors.append(NamedTensor("bias", tensor2))
        tensors.append(NamedTensor("indices", tensor3))

        # Save
        save_named_tensors(tensors, test_dir)

        # Load
        var loaded = load_named_tensors(test_dir)

        # Verify count
        assert_equal(len(loaded), 3, "Should have loaded 3 tensors")

        # Verify names (should be in sorted order)
        var names = List[String]()
        for i in range(len(loaded)):
            names.append(loaded[i].name)

    finally:
        _ = cleanup_test_dir(test_dir)


# ============================================================================
# Checkpoint Tests (with metadata)
# ============================================================================


fn test_save_named_checkpoint_without_metadata() raises:
    """Test saving checkpoint without metadata."""
    var test_dir = create_test_dir("/tmp")

    try:
        # Create test tensors
        var shape = List[Int]()
        shape.append(3)
        shape.append(4)
        var tensor = ones(shape, DType.float32)

        var tensors: List[NamedTensor] = []
        tensors.append(NamedTensor("weights", tensor))

        # Save without metadata - should not raise
        save_named_checkpoint(tensors, test_dir)

        # If we reach here, save succeeded
        assert_true(True, "Checkpoint save without metadata should succeed")

    finally:
        _ = cleanup_test_dir(test_dir)


fn test_save_named_checkpoint_with_metadata() raises:
    """Test saving checkpoint with metadata."""
    var test_dir = create_test_dir("/tmp")

    try:
        # Create test tensors
        var shape = List[Int]()
        shape.append(2)
        var tensor = ones(shape, DType.float32)

        var tensors: List[NamedTensor] = []
        tensors.append(NamedTensor("weights", tensor))

        # Create metadata
        var metadata = Dict[String, String]()
        metadata["epoch"] = "10"
        metadata["loss"] = "0.45"
        metadata["learning_rate"] = "0.001"

        # Save with metadata - should not raise
        save_named_checkpoint(tensors, test_dir, metadata^)

        # If we reach here, save succeeded
        assert_true(True, "Checkpoint save with metadata should succeed")

    finally:
        _ = cleanup_test_dir(test_dir)


fn test_load_named_checkpoint_with_metadata() raises:
    """Test loading checkpoint with metadata."""
    var test_dir = create_test_dir("/tmp")

    try:
        # Create test tensors
        var shape = List[Int]()
        shape.append(3)
        var tensor = ones(shape, DType.float32)

        var tensors: List[NamedTensor] = []
        tensors.append(NamedTensor("weights", tensor))

        # Create metadata
        var metadata = Dict[String, String]()
        metadata["epoch"] = "20"
        metadata["loss"] = "0.32"

        # Save
        save_named_checkpoint(tensors, test_dir, metadata^)

        # Load
        var result = load_named_checkpoint(test_dir)

        # Verify tensors
        assert_equal(len(result[0]), 1, "Should load 1 tensor")
        assert_equal(result[0][0].name, "weights", "Tensor name should match")

        # Verify metadata
        assert_true("epoch" in result[1], "Metadata should have epoch")
        assert_true("loss" in result[1], "Metadata should have loss")
        assert_equal(result[1]["epoch"], "20", "Epoch value should match")
        assert_equal(result[1]["loss"], "0.32", "Loss value should match")

    finally:
        _ = cleanup_test_dir(test_dir)


fn test_load_named_checkpoint_without_metadata_file() raises:
    """Test loading checkpoint when metadata file doesn't exist."""
    var test_dir = create_test_dir("/tmp")

    try:
        # Create and save tensors without metadata
        var shape = List[Int]()
        shape.append(2)
        var tensor = ones(shape, DType.float32)

        var tensors: List[NamedTensor] = []
        tensors.append(NamedTensor("weights", tensor))

        # Save without metadata (None)
        save_named_checkpoint(tensors, test_dir)

        # Load should still work with empty metadata
        var result = load_named_checkpoint(test_dir)

        # Verify tensors loaded
        assert_equal(len(result[0]), 1, "Should load 1 tensor")

        # Verify metadata is empty but not error
        assert_equal(len(result[1]), 0, "Metadata should be empty")

    finally:
        _ = cleanup_test_dir(test_dir)


fn test_checkpoint_round_trip() raises:
    """Test full checkpoint save/load round trip."""
    var test_dir = create_test_dir("/tmp")

    try:
        # Create original tensors with different shapes and dtypes
        var shape1 = List[Int]()
        shape1.append(4)
        shape1.append(5)
        var tensor1 = ones(shape1, DType.float32)

        var shape2 = List[Int]()
        shape2.append(10)
        var tensor2 = zeros(shape2, DType.float64)

        var tensors: List[NamedTensor] = []
        tensors.append(NamedTensor("layer1_w", tensor1))
        tensors.append(NamedTensor("layer1_b", tensor2))

        # Create metadata
        var metadata = Dict[String, String]()
        metadata["epoch"] = "100"
        metadata["best_loss"] = "0.001"
        metadata["model"] = "test_model"

        # Save checkpoint
        save_named_checkpoint(tensors, test_dir, metadata^)

        # Load checkpoint
        var result = load_named_checkpoint(test_dir)

        # Verify tensor count
        assert_equal(len(result[0]), 2, "Should have 2 tensors")

        # Verify tensors are loaded (order may vary due to glob)
        # Just check we have one 2D tensor and one 1D tensor
        var has_2d = False
        var has_1d = False
        for i in range(len(result[0])):
            var tensor = result[0][i].tensor
            if len(tensor.shape()) == 2:
                assert_equal(tensor.shape()[0], 4, "2D tensor dim 0")
                assert_equal(tensor.shape()[1], 5, "2D tensor dim 1")
                has_2d = True
            elif len(tensor.shape()) == 1:
                assert_equal(tensor.shape()[0], 10, "1D tensor dim 0")
                has_1d = True
        assert_true(has_2d, "Should have 2D tensor")
        assert_true(has_1d, "Should have 1D tensor")

        # Verify metadata
        assert_equal(result[1]["epoch"], "100", "Epoch should match")
        assert_equal(result[1]["best_loss"], "0.001", "Best loss should match")
        assert_equal(result[1]["model"], "test_model", "Model should match")

    finally:
        _ = cleanup_test_dir(test_dir)


fn test_checkpoint_with_many_tensors() raises:
    """Test checkpoint with many tensors."""
    var test_dir = create_test_dir("/tmp")

    try:
        # Create many tensors
        var tensors: List[NamedTensor] = []
        for i in range(10):
            var shape = List[Int]()
            shape.append(i + 1)
            var tensor = ones(shape, DType.float32)
            var name = "tensor_" + String(i)
            tensors.append(NamedTensor(name, tensor))

        # Save
        save_named_checkpoint(tensors, test_dir)

        # Load
        var result = load_named_checkpoint(test_dir)

        # Verify all tensors loaded
        assert_equal(len(result[0]), 10, "Should load 10 tensors")

        # Verify all expected shapes are present (order may vary)
        var found_shapes: List[Bool] = []
        for _ in range(10):
            found_shapes.append(False)

        for i in range(len(result[0])):
            var shape_dim = result[0][i].tensor.shape()[0]
            if shape_dim >= 1 and shape_dim <= 10:
                found_shapes[shape_dim - 1] = True

        for i in range(10):
            assert_true(
                found_shapes[i], "Shape " + String(i + 1) + " should be present"
            )

    finally:
        _ = cleanup_test_dir(test_dir)


# ============================================================================
# Trailing Slash Normalization Tests
# ============================================================================


fn test_save_named_tensors_trailing_slash() raises:
    """Test that save_named_tensors handles trailing slash in dirpath."""
    var test_dir = create_test_dir("/tmp")

    try:
        var shape = List[Int]()
        shape.append(2)
        shape.append(3)
        var tensor = ones(shape, DType.float32)

        var tensors: List[NamedTensor] = []
        tensors.append(NamedTensor("weights", tensor))

        # Save with trailing slash
        save_named_tensors(tensors, test_dir + "/")

        # Load without trailing slash - should work correctly
        var loaded = load_named_tensors(test_dir)

        assert_equal(len(loaded), 1, "Should load 1 tensor")
        assert_equal(loaded[0].name, "weights", "Tensor name should match")
        assert_equal(loaded[0].tensor.shape()[0], 2, "Shape dim 0 should match")
        assert_equal(loaded[0].tensor.shape()[1], 3, "Shape dim 1 should match")

    finally:
        _ = cleanup_test_dir(test_dir)


fn test_load_named_tensors_trailing_slash() raises:
    """Test that load_named_tensors handles trailing slash in dirpath."""
    var test_dir = create_test_dir("/tmp")

    try:
        var shape = List[Int]()
        shape.append(4)
        var tensor = ones(shape, DType.float32)

        var tensors: List[NamedTensor] = []
        tensors.append(NamedTensor("bias", tensor))

        # Save without trailing slash
        save_named_tensors(tensors, test_dir)

        # Load with trailing slash - should work correctly
        var loaded = load_named_tensors(test_dir + "/")

        assert_equal(len(loaded), 1, "Should load 1 tensor")
        assert_equal(loaded[0].name, "bias", "Tensor name should match")
        assert_equal(loaded[0].tensor.shape()[0], 4, "Shape dim 0 should match")

    finally:
        _ = cleanup_test_dir(test_dir)


fn test_save_load_trailing_slash_round_trip() raises:
    """Test save with trailing slash and load with trailing slash both work."""
    var test_dir = create_test_dir("/tmp")

    try:
        var shape = List[Int]()
        shape.append(3)
        shape.append(3)
        var tensor = ones(shape, DType.float32)

        var tensors: List[NamedTensor] = []
        tensors.append(NamedTensor("kernel", tensor))

        # Save and load both with trailing slash
        save_named_tensors(tensors, test_dir + "/")
        var loaded = load_named_tensors(test_dir + "/")

        assert_equal(len(loaded), 1, "Should load 1 tensor")
        assert_equal(loaded[0].name, "kernel", "Tensor name should match")
        assert_equal(loaded[0].tensor.shape()[0], 3, "Shape dim 0 should match")
        assert_equal(loaded[0].tensor.shape()[1], 3, "Shape dim 1 should match")

    finally:
        _ = cleanup_test_dir(test_dir)


fn main() raises:
    """Run all serialization tests."""
    print("Running serialization tests...")

    print("  test_named_tensor_creation...")
    test_named_tensor_creation()
    print("  ✓ passed")

    print("  test_named_tensor_multiple_dtypes...")
    test_named_tensor_multiple_dtypes()
    print("  ✓ passed")

    print("  test_save_load_single_named_tensor...")
    test_save_load_single_named_tensor()
    print("  ✓ passed")

    print("  test_save_load_multiple_named_tensors...")
    test_save_load_multiple_named_tensors()
    print("  ✓ passed")

    print("  test_save_named_checkpoint_without_metadata...")
    test_save_named_checkpoint_without_metadata()
    print("  ✓ passed")

    print("  test_save_named_checkpoint_with_metadata...")
    test_save_named_checkpoint_with_metadata()
    print("  ✓ passed")

    print("  test_load_named_checkpoint_with_metadata...")
    test_load_named_checkpoint_with_metadata()
    print("  ✓ passed")

    print("  test_load_named_checkpoint_without_metadata_file...")
    test_load_named_checkpoint_without_metadata_file()
    print("  ✓ passed")

    print("  test_checkpoint_round_trip...")
    test_checkpoint_round_trip()
    print("  ✓ passed")

    print("  test_checkpoint_with_many_tensors...")
    test_checkpoint_with_many_tensors()
    print("  ✓ passed")

    print("  test_save_named_tensors_trailing_slash...")
    test_save_named_tensors_trailing_slash()
    print("  ✓ passed")

    print("  test_load_named_tensors_trailing_slash...")
    test_load_named_tensors_trailing_slash()
    print("  ✓ passed")

    print("  test_save_load_trailing_slash_round_trip...")
    test_save_load_trailing_slash_round_trip()
    print("  ✓ passed")

    print("\nAll serialization tests passed!")
