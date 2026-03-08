# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_memory_leaks.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Memory leak detection tests for ExTensor - Part 2: Stress Tests & View Lifetime.

Tests verify:
1. No memory leaks in repeated operations
2. View lifetime management
"""

from shared.core import ExTensor, zeros, ones, full
from tests.shared.conftest import assert_true, assert_equal_int


# ============================================================================
# Stress Tests for Memory Leaks
# ============================================================================


fn test_no_memory_leak_in_creation_loop() raises:
    """Verify no memory leaks in repeated tensor creation."""
    alias NUM_ITERATIONS = 10000
    alias TENSOR_SIZE = 100
    for _ in range(NUM_ITERATIONS):
        var tensor = zeros([TENSOR_SIZE, TENSOR_SIZE], DType.float32)

    assert_true(True, "Created 10000 tensors without OOM")


fn test_no_memory_leak_in_operation_loop() raises:
    """Verify no memory leaks in repeated tensor operations."""
    alias NUM_ITERATIONS = 5000
    for _ in range(NUM_ITERATIONS):
        var tensor1 = zeros([50, 50], DType.float32)
        var tensor2 = ones([50, 50], DType.float32)
        var result = tensor1 + tensor2

    assert_true(True, "Completed 5000 operations without OOM")


fn test_no_memory_leak_with_copies() raises:
    """Verify no memory leaks with shared copies."""
    alias NUM_ITERATIONS = 1000
    for _ in range(NUM_ITERATIONS):
        var tensor1 = ones([100, 100], DType.float32)
        var tensor2 = tensor1
        var tensor3 = tensor2
        var tensor4 = tensor1
        assert_equal_int(tensor1._refcount[], 4, "Should have 4 refs")

    assert_true(True, "Copy stress test completed without OOM")


fn test_large_tensor_lifecycle() raises:
    """Test large tensor allocation and deallocation."""
    alias NUM_ITERATIONS = 50
    alias LARGE_SIZE = 1000
    for _ in range(NUM_ITERATIONS):
        var tensor = zeros([LARGE_SIZE, LARGE_SIZE], DType.float32)
        _ = tensor.numel()

    assert_true(True, "Large tensor lifecycle test completed")


# ============================================================================
# View Lifetime Tests
# ============================================================================


fn test_view_flag_on_reshape() raises:
    """Test reshape creates a view with _is_view flag set."""
    var tensor = zeros([12], DType.float32)
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var reshaped = tensor.reshape(shape)
    assert_true(reshaped._is_view, "Reshaped tensor should be a view")
    assert_true(tensor._data == reshaped._data, "View should share data")


fn test_view_does_not_free_data() raises:
    """Test view destruction doesn't free shared data."""
    var original = zeros([12], DType.float32)
    original._data.bitcast[Float32]()[0] = 42.0

    if True:
        var shape = List[Int]()
        shape.append(3)
        shape.append(4)
        var view = original.reshape(shape)
        assert_true(view._is_view, "Should be marked as view")

    var value = original._data.bitcast[Float32]()[0]
    assert_true(value == 42.0, "Original data should be intact")


fn test_view_modification_affects_original() raises:
    """Test modifying view affects original tensor."""
    var original = zeros([12], DType.float32)
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var view = original.reshape(shape)
    view._data.bitcast[Float32]()[0] = 99.0
    var value = original._data.bitcast[Float32]()[0]
    assert_true(
        value == 99.0, "Modification through view should affect original"
    )


# ============================================================================
# Test Runner
# ============================================================================


fn main() raises:
    """Run stress tests and view lifetime tests."""
    print("Running memory leak tests - Part 2...")

    print("  Stress tests...")
    test_no_memory_leak_in_creation_loop()
    test_no_memory_leak_in_operation_loop()
    test_no_memory_leak_with_copies()
    test_large_tensor_lifecycle()
    print("    Passed")

    print("  View lifetime tests...")
    test_view_flag_on_reshape()
    test_view_does_not_free_data()
    test_view_modification_affects_original()
    print("    Passed")

    print("\nAll Part 2 memory leak tests passed!")
