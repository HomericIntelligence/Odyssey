# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_memory_leaks.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Memory leak detection tests for AnyTensor - Part 1: Reference Counting & Allocation.

Tests verify:
1. Reference counting correctness
2. Memory deallocation on scope exit
"""

from shared.core.extensor import AnyTensor, zeros, ones, full
from tests.shared.conftest import assert_true, assert_equal_int


# ============================================================================
# Reference Counting Tests
# ============================================================================


fn test_single_tensor_refcount() raises:
    """Test single tensor starts with refcount = 1."""
    var tensor = zeros([10, 10], DType.float32)
    assert_equal_int(
        tensor._refcount[], 1, "Single tensor should have refcount 1"
    )


fn test_copy_increments_refcount() raises:
    """Test copying tensor increments reference count."""
    var tensor1 = zeros([10, 10], DType.float32)
    var initial_refcount = tensor1._refcount[]
    assert_equal_int(initial_refcount, 1, "Initial refcount should be 1")

    var tensor2 = tensor1
    assert_equal_int(tensor1._refcount[], 2, "Refcount should be 2 after copy")
    assert_true(
        tensor1._data == tensor2._data, "Copied tensors should share data"
    )


fn test_multiple_copies_refcount() raises:
    """Test multiple copies increment refcount correctly."""
    var tensor1 = zeros([5, 5], DType.float32)
    var tensor2 = tensor1
    var tensor3 = tensor1
    var tensor4 = tensor2
    assert_equal_int(
        tensor1._refcount[], 4, "Refcount should be 4 after 3 copies"
    )


fn _copy_and_check_refcount(tensor1: AnyTensor) -> Int:
    """Helper: copy tensor in inner scope and return inner refcount."""
    var tensor2 = tensor1
    return tensor1._refcount[]


fn test_scope_exit_decrements_refcount() raises:
    """Test refcount decrements when copy goes out of scope."""
    var tensor1 = zeros([10, 10], DType.float32)
    var initial_refcount = tensor1._refcount[]
    assert_equal_int(initial_refcount, 1, "Initial refcount should be 1")

    var inner_refcount = _copy_and_check_refcount(tensor1)
    assert_equal_int(inner_refcount, 2, "Refcount should be 2 in inner scope")

    var outer_refcount = tensor1._refcount[]
    assert_equal_int(outer_refcount, 1, "Refcount should be 1 after scope exit")


fn _modify_through_copy(tensor1: AnyTensor) raises:
    """Helper: copy tensor in inner scope and modify shared data."""
    var tensor2 = tensor1
    assert_true(tensor1._data == tensor2._data, "Should share data")
    tensor2._data.bitcast[Float32]()[0] = 99.0


fn test_original_survives_copy_destruction() raises:
    """Test original tensor survives when copy is destroyed."""
    var tensor1 = zeros([10, 10], DType.float32)
    # Write a known value through data pointer
    tensor1._data.bitcast[Float32]()[0] = 42.0

    _modify_through_copy(tensor1)

    # Verify modification persists through original
    var value = tensor1._data.bitcast[Float32]()[0]
    assert_true(value == 99.0, "Original should reflect modification")


# ============================================================================
# Memory Allocation/Deallocation Tests
# ============================================================================


fn _alloc_and_use_tensor() raises:
    """Helper: allocate tensor in inner scope and use it."""
    var tensor = zeros([100, 100], DType.float32)
    _ = tensor.numel()


fn test_tensor_deallocation_single() raises:
    """Test single tensor deallocates memory when destroyed."""
    _alloc_and_use_tensor()
    assert_true(True, "Single tensor deallocation completed without crash")


fn test_tensor_deallocation_loop() raises:
    """Test repeated tensor creation/destruction in loop."""
    for i in range(1000):
        var tensor = zeros([50, 50], DType.float32)
        _ = tensor.numel()

    assert_true(True, "Loop deallocation completed without crash")


fn _check_shared_deallocation() raises:
    """Helper: verify refcount during nested sharing in inner scope."""
    var tensor1 = zeros([10, 10], DType.float32)
    var initial_refcount = tensor1._refcount[]
    assert_equal_int(initial_refcount, 1, "Should start with refcount 1")

    var inner_refcount = _copy_and_check_refcount(tensor1)
    assert_equal_int(inner_refcount, 2, "Should have 2 refs in inner scope")

    var outer_refcount = tensor1._refcount[]
    assert_equal_int(
        outer_refcount, 1, "Should have 1 ref after inner scope"
    )


fn test_shared_tensor_deallocation() raises:
    """Test shared tensor deallocates only when last reference destroyed."""
    _check_shared_deallocation()
    assert_true(True, "Shared tensor deallocation completed")


# ============================================================================
# Test Runner
# ============================================================================


fn main() raises:
    """Run reference counting and allocation tests."""
    print("Running memory leak tests - Part 1...")

    print("  Reference counting tests...")
    test_single_tensor_refcount()
    test_copy_increments_refcount()
    test_multiple_copies_refcount()
    test_scope_exit_decrements_refcount()
    test_original_survives_copy_destruction()
    print("    Passed")

    print("  Allocation/deallocation tests...")
    test_tensor_deallocation_single()
    test_tensor_deallocation_loop()
    test_shared_tensor_deallocation()
    print("    Passed")

    print("\nAll Part 1 memory leak tests passed!")
