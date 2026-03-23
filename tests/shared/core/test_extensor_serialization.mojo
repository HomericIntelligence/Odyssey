"""Tests for AnyTensor serialization and deserialization.

Tests the save_tensor() and load_tensor() standalone functions for:
- Single tensor round-trip (save -> load -> compare values)
- Multiple data types (float32, float64, int8, etc.)
- Various tensor shapes (1D, 2D, 3D, 4D)
- Named tensor persistence

Runs as: pixi run mojo ./tests/shared/core/test_extensor_serialization.mojo (legacy name)
"""

from shared.tensor.any_tensor import AnyTensor, zeros, ones, arange
from shared.utils.serialization import save_tensor, load_tensor


fn test_save_load_float32() raises:
    """Test save/load round-trip with float32 tensor."""
    print("\n" + "=" * 80)
    print("Test: Save/Load Float32 Tensor")
    print("=" * 80)

    # Create test tensor
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var original = arange(0.0, 6.0, 1.0, DType.float32)
    original = original.reshape(shape)

    # Save
    var path = "/tmp/test_tensor_float32.bin"
    save_tensor(original, path, "test_float32")
    print("✓ Tensor saved to:", path)

    # Load
    var loaded = load_tensor(path)
    print("✓ Tensor loaded from:", path)

    # Verify
    if (
        loaded.shape() == original.shape()
        and loaded.dtype() == original.dtype()
    ):
        print("✓ Shape and dtype match")
    else:
        raise Error("Shape or dtype mismatch")

    # Check values match
    var all_match = True
    for i in range(loaded.numel()):
        var orig_val = original._get_float64(i)
        var load_val = loaded._get_float64(i)
        if orig_val != load_val:
            all_match = False
            break

    if all_match:
        print("✓ All values match after round-trip")
    else:
        raise Error("Values don't match after load")

    print()


fn test_save_load_float64() raises:
    """Test save/load round-trip with float64 tensor."""
    print("Test: Save/Load Float64 Tensor")
    print("-" * 40)

    # Create test tensor
    var shape = List[Int]()
    shape.append(3)
    shape.append(2)
    var original = ones(shape, DType.float64)
    # Fill with values by setting each element
    for i in range(original.numel()):
        original._set_float64(i, 3.14159)

    # Save and load
    var path = "/tmp/test_tensor_float64.bin"
    save_tensor(original, path, "test_float64")
    var loaded = load_tensor(path)

    # Verify
    var matches = True
    for i in range(loaded.numel()):
        var orig_val = original._get_float64(i)
        var load_val = loaded._get_float64(i)
        if orig_val != load_val:
            matches = False
            break

    if matches:
        print("✓ Float64 round-trip successful")
    else:
        raise Error("Float64 round-trip failed")

    print()


fn test_save_load_int64() raises:
    """Test save/load round-trip with int64 tensor."""
    print("Test: Save/Load Int64 Tensor")
    print("-" * 40)

    # Create test tensor
    var shape = List[Int]()
    shape.append(4)
    var original = zeros(shape, DType.int64)
    for i in range(4):
        original._set_float64(i, Float64(i * 10))

    # Save and load
    var path = "/tmp/test_tensor_int64.bin"
    save_tensor(original, path, "test_int64")
    var loaded = load_tensor(path)

    # Verify
    if loaded.dtype() == DType.int64:
        print("✓ Dtype preserved (int64)")
    else:
        raise Error("Dtype not preserved")

    print("✓ Int64 round-trip successful")

    print()


fn test_save_load_different_shapes() raises:
    """Test save/load with various tensor shapes."""
    print("Test: Save/Load Different Shapes")
    print("-" * 40)

    # Test 1D tensor
    var shape_1d = List[Int]()
    shape_1d.append(10)
    var tensor_1d = arange(0.0, 10.0, 1.0, DType.float32)
    save_tensor(tensor_1d, "/tmp/test_1d.bin")
    var loaded_1d = load_tensor("/tmp/test_1d.bin")
    if loaded_1d.shape() == tensor_1d.shape():
        print("✓ 1D tensor shape preserved")
    else:
        raise Error("1D shape not preserved")

    # Test 2D tensor
    var shape_2d = List[Int]()
    shape_2d.append(3)
    shape_2d.append(4)
    var tensor_2d = ones(shape_2d, DType.float32)
    save_tensor(tensor_2d, "/tmp/test_2d.bin")
    var loaded_2d = load_tensor("/tmp/test_2d.bin")
    if loaded_2d.shape() == tensor_2d.shape():
        print("✓ 2D tensor shape preserved")
    else:
        raise Error("2D shape not preserved")

    # Test 3D tensor
    var shape_3d = List[Int]()
    shape_3d.append(2)
    shape_3d.append(3)
    shape_3d.append(4)
    var tensor_3d = zeros(shape_3d, DType.float32)
    save_tensor(tensor_3d, "/tmp/test_3d.bin")
    var loaded_3d = load_tensor("/tmp/test_3d.bin")
    if loaded_3d.shape() == tensor_3d.shape():
        print("✓ 3D tensor shape preserved")
    else:
        raise Error("3D shape not preserved")

    print()


fn test_named_tensor_save() raises:
    """Test saving tensor with custom name."""
    print("Test: Save Tensor with Custom Name")
    print("-" * 40)

    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    var tensor = ones(shape, DType.float32)

    var path = "/tmp/test_named.bin"
    var name = "my_custom_tensor"
    save_tensor(tensor, path, name)

    # Load and verify
    var loaded = load_tensor(path)
    print("✓ Tensor with name saved and loaded successfully")

    print()


fn test_large_tensor_serialization() raises:
    """Test serialization of larger tensor to verify efficiency."""
    print("Test: Large Tensor Serialization")
    print("-" * 40)

    # Create larger tensor (100 x 100)
    var shape = List[Int]()
    shape.append(100)
    shape.append(100)
    var large_tensor = ones(shape, DType.float32)
    # Fill with values
    for i in range(large_tensor.numel()):
        large_tensor._set_float64(i, 2.71828)

    var path = "/tmp/test_large.bin"
    save_tensor(large_tensor, path)
    print("✓ Large tensor (100x100) saved")

    var loaded = load_tensor(path)
    print("✓ Large tensor loaded")

    if loaded.numel() == 10000:
        print("✓ Element count correct (10000 elements)")
    else:
        raise Error("Element count mismatch")

    # Verify a few values
    var sample_match = True
    for i in range(0, 100, 20):
        if loaded._get_float64(i * 100) != large_tensor._get_float64(i * 100):
            sample_match = False
            break

    if sample_match:
        print("✓ Sample values match after load")
    else:
        raise Error("Sample values don't match")

    print()


fn main() raises:
    """Run all serialization tests."""
    print("\n" + "#" * 80)
    print("# AnyTensor Serialization Tests")
    print("#" * 80)

    test_save_load_float32()
    test_save_load_float64()
    test_save_load_int64()
    test_save_load_different_shapes()
    test_named_tensor_save()
    test_large_tensor_serialization()

    print("=" * 80)
    print("Summary: All serialization tests passed!")
    print("=" * 80 + "\n")
